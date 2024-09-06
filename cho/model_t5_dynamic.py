import os
import torch
import shutil
import logging
import warnings
import numpy as np
import pandas as pd

from collections import Counter
import re

from tqdm import tqdm
from rouge import Rouge
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset, DataLoader

from transformers import (T5ForConditionalGeneration, T5TokenizerFast, 
                          get_cosine_schedule_with_warmup)
from typing import List, Dict, Any
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
import torch.distributed as dist
import torch.cuda.amp as amp

# 경고 메시지 무시 및 로깅 설정
warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

file_handler = logging.FileHandler('training.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Wandb 초기화 : 자신에 맞게 수정
wandb.init(
    entity="dl12",
    project="lm",
    name="lcw99/t5-large-dynamic-summary",
)

# CUDA 설정 : 성능 최적화
torch.backends.cudnn.benchmark = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 시퀀스를 작은 청크로 나누는 함수
def chunk_sequence(sequence, chunk_size):
    return [sequence[i:i + chunk_size] for i in range(0, len(sequence), chunk_size)]

class Preprocess:
    def __init__(self, tokenizer: T5TokenizerFast):
        self.tokenizer = tokenizer
        self.bos_token = tokenizer.bos_token or ''
        self.eos_token = tokenizer.eos_token or ''
        self.chunk_size = 512

    @staticmethod
    def make_set_as_df(file_path: str, is_train: bool = True) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        
        replacements = {
            'ㅋㅋ': '웃기다', 'ㅇ로': '으로', '제ㅏ': '제가', 'ㅍ알': ' 알', 'ㄷ거': '거',
            '##': '#', '회사 #에서': '회사에서',
            '#작은': '#Person2#: 작은', '#여기서': '#Person1#: 여기서', '#나': '#Person2#: 나',
            '#페리에와': '#Person1#: 페리에와', '#샐러드용': '#Person1#: 샐러드용',
            '#어디': '#Person1#: 어디', '#잠깐만요': '#Person1#: 잠깐만요',
            '#하지만': '#Person1#: 하지만', '#사람1만기': '#Person1#: 만기',
            '#PhoneNumber이고': '#PhoneNumber#이고', '#Person1:': '#Person1#:',
            '#Person2:': '#Person2#:', '#Person#': '#Person2#:', '사람1#:': '#Person1#:',
            '#고객님:': '#Person2#: 고객님', '선생님: ': '', '로저스 씨: ': '',
            '남자: 아악.': '', '남자: 고마워.': ''
        }
        
        df['dialogue'] = df['dialogue'].replace(replacements, regex=True)

        if 'summary' in df.columns:
            summary_replacements = {
                '사람1#': '#Person1#', '사람2#': '#Person2#', '#사람1#': '#Person1#'
            }
            df['summary'] = df['summary'].replace(summary_replacements, regex=True)

        if is_train:
            return df[['fname', 'dialogue', 'summary', 'topic']]
        else:
            return df[['fname', 'dialogue']]

    # 모델 입력 생성 : 대화를 청크로 나눔, 각 청크에 대해 프롬프트 생성
    def make_input(self, dataset: pd.DataFrame, is_test: bool = False):
        encoder_input, decoder_input, decoder_output = [], [], []

        for _, row in dataset.iterrows():
            dialogue = str(row['dialogue'])
            summary = str(row['summary']) if not is_test else ""
            
            for chunk in [dialogue[i:i+self.chunk_size] for i in range(0, len(dialogue), self.chunk_size)]:
                encoder_input.append(create_prompt(chunk))
                if not is_test:
                    decoder_input.append(self.bos_token + summary)
                    decoder_output.append(summary + self.eos_token)
                else:
                    decoder_input.append(self.bos_token)
                    decoder_output.append("")

        return (encoder_input, decoder_input, decoder_output) if not is_test else (encoder_input, decoder_input)

# TS 모델을 상속받아 긴 시퀀스를 처리하도록 확장
class LongformerEncoderDecoderForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

# 배치 내의 시퀀스 길이를 동적으로 패딩
class DynamicPaddingCollator:
    def __init__(self, pad_token_id: int, label_pad_token_id: int):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if all(len(x["input_ids"]) == len(features[0]["input_ids"]) for x in features):
            return self.stack_tensors(features)
        else:
            return self.pad_tensors(features)
    
    # 모든 텐서를 스택
    def stack_tensors(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return {k: torch.stack([torch.tensor(f[k]) if isinstance(f[k], list) else f[k] for f in features]) for k in features[0].keys()}
    
    # 가장 긴 시퀀스에 맞춰 나머지 시퀀스를 패딩
    def pad_tensors(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(x["input_ids"]) for x in features)
        padded_features = {}
        for key in features[0].keys():
            padding_value = self.label_pad_token_id if key == "labels" else self.pad_token_id
            padded_features[key] = torch.nn.utils.rnn.pad_sequence(
                [f[key] for f in features], batch_first=True, padding_value=padding_value
            )
        return padded_features

def create_optimized_dataloaders(config, tokenizer, preprocess):
    train_df = preprocess.make_set_as_df(os.path.join(config['general']['data_path'], 'train.csv'), is_train=True)
    val_df = preprocess.make_set_as_df(os.path.join(config['general']['data_path'], 'dev.csv'), is_train=True)

    train_encoder_input, train_decoder_input, train_decoder_output = preprocess.make_input(train_df, is_test=False)
    val_encoder_input, val_decoder_input, val_decoder_output = preprocess.make_input(val_df, is_test=False)

    train_dataset = HFDataset.from_dict({
        "encoder_input": train_encoder_input,
        "decoder_input": train_decoder_input,
        "decoder_output": train_decoder_output
    })

    val_dataset = HFDataset.from_dict({
        "encoder_input": val_encoder_input,
        "decoder_input": val_decoder_input,
        "decoder_output": val_decoder_output
    })

    # 입력과 레이블을 토큰화
    def preprocess_function(examples):
        model_inputs = tokenizer(examples["encoder_input"], max_length=config['tokenizer']['encoder_max_len'], truncation=True)
        labels = tokenizer(text_target=examples["decoder_output"], max_length=config['tokenizer']['decoder_max_len'], truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=4, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(preprocess_function, batched=True, num_proc=4, remove_columns=val_dataset.column_names)

    # 분산 학습을 위한 샘플러 설정
    train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.is_initialized() else None

    collator = DynamicPaddingCollator(pad_token_id=tokenizer.pad_token_id, label_pad_token_id=tokenizer.pad_token_id)

    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['per_device_train_batch_size'], 
                                  sampler=train_sampler, collate_fn=collator, pin_memory=True, num_workers=4)
    eval_dataloader = DataLoader(val_dataset, batch_size=config['training']['per_device_eval_batch_size'], 
                                 sampler=val_sampler, collate_fn=collator, pin_memory=True, num_workers=4)
    
    return train_dataloader, eval_dataloader

def save_checkpoint(model, optimizer, epoch, step, loss, config, is_best=False):
    checkpoint_dir = os.path.join(config['general']['output_dir'], "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_step_{step}.pt")
    
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    torch.save({
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss
    }, checkpoint_path)
    
    if is_best:
        best_model_path = os.path.join(config['general']['output_dir'], "best_model.pt")
        shutil.copyfile(checkpoint_path, best_model_path)
    
    logger.info(f"Checkpoint saved at {checkpoint_path}")
    return checkpoint_path

def load_tokenizer(config):
    tokenizer_path = os.path.join(config['general']['output_dir'], "final_tokenizer")
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path)
    logger.info(f"Tokenizer loaded from {tokenizer_path}")
    return tokenizer

def save_tokenizer(tokenizer, output_dir):
    tokenizer_save_path = os.path.join(output_dir, "final_tokenizer")
    tokenizer.save_pretrained(tokenizer_save_path)
    logger.info(f"Final tokenizer saved at {tokenizer_save_path}")

def load_tokenizer_and_model(config, device, for_inference=False):
    logger.info(f"{'Loading tokenizer & model for inference' if for_inference else 'Loading tokenizer & model for training'}")
    model_name = config['general']['model_name']
    
    if for_inference:
        tokenizer = load_tokenizer(config)
        model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained(
            os.path.join(config['general']['output_dir'], "final_model")
        )
    else:
        tokenizer = T5TokenizerFast.from_pretrained(model_name)
        special_tokens = config['tokenizer'].get('special_tokens', [])
        if special_tokens:
            special_tokens_dict = {'additional_special_tokens': special_tokens}
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            logger.info(f"Added {num_added_toks} special tokens: {special_tokens}")

        model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained(model_name)
        
        if num_added_toks > 0:
            model.resize_token_embeddings(len(tokenizer))
            logger.info(f"Model embeddings resized to accommodate {num_added_toks} new tokens")
    
    tokenizer.do_not_trim_special_tokens = True
    
    model.to(device)
    return model, tokenizer

# 저장공간을 위해 오래된 체크포인트 자동 삭제(n개만 유지)
def cleanup_old_checkpoints(config: Dict[str, Any], keep_last_n: int = 5):
    checkpoint_dir = os.path.join(config['general']['output_dir'], "checkpoints")
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pt")], reverse=True)
    for checkpoint in checkpoint_files[keep_last_n:]:
        os.remove(os.path.join(checkpoint_dir, checkpoint))
        logger.info(f"Removed old checkpoint: {checkpoint}")

class EarlyStoppingCallback:
    def __init__(self, patience: int, threshold: float):
        self.patience = patience
        self.threshold = threshold
        self.best_score = np.inf
        self.counter = 0
        self.should_stop = False

    def __call__(self, eval_loss: float, epoch: int):
        if eval_loss < self.best_score * (1 - self.threshold):
            self.best_score = eval_loss
            self.counter = 0
        elif eval_loss > (self.best_score - self.threshold):
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# 모델학습의 전체 과정 관리 함수
def train_and_save(config):
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl')
        local_rank = dist.get_rank()
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model, tokenizer = load_tokenizer_and_model(config, device)
    if torch.cuda.device_count() > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    preprocess = Preprocess(tokenizer)
    train_dataloader, eval_dataloader = create_optimized_dataloaders(config, tokenizer, preprocess)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    num_training_steps = len(train_dataloader) * config['training']['num_train_epochs']
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=config['training']['warmup_steps'], num_training_steps=num_training_steps)
    scaler = amp.GradScaler()
    early_stopping_callback = EarlyStoppingCallback(patience=config['training']['early_stopping_patience'], threshold=config['training']['early_stopping_threshold'])

    best_eval_loss = float('inf')
    best_model_path = None

    for epoch in range(config['training']['num_train_epochs']):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}", dynamic_ncols=True, ascii=True)):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with amp.autocast():
                outputs = model(**batch)
                loss = outputs.loss
            
            total_loss += loss.item()
            scaler.scale(loss).backward()

            if (step + 1) % config['training']['gradient_accumulation_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()

            if (not dist.is_initialized() or dist.get_rank() == 0) and step % config['training']['logging_steps'] == 0:
                logger.info(f"Epoch {epoch+1}, Step {step}: Loss {loss.item():.4f}")
                wandb.log({"train_loss": loss.item(), "epoch": epoch + 1, "step": step})

            if (not dist.is_initialized() or dist.get_rank() == 0) and step % config['training']['save_steps'] == 0:
                save_checkpoint(model, optimizer, epoch, step, loss.item(), config)
                cleanup_old_checkpoints(config)

        avg_train_loss = total_loss / len(train_dataloader)
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info(f"Epoch {epoch+1} - Average train loss: {avg_train_loss:.4f}")
            wandb.log({"avg_train_loss": avg_train_loss, "epoch": epoch + 1})
        
        eval_loss, rouge_scores = evaluate(model, eval_dataloader, tokenizer, device, config)
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info(f"Epoch {epoch+1} - Eval Loss: {eval_loss:.4f}")
            logger.info(f"Epoch {epoch+1} - Rouge Scores:")
            for k, v in rouge_scores.items():
                logger.info(f"  {k.upper()}: {v['f']:.4f}")
        
            wandb.log({
                "eval_loss": eval_loss,
                "rouge-1_f": rouge_scores['rouge-1']['f'],
                "rouge-2_f": rouge_scores['rouge-2']['f'],
                "rouge-l_f": rouge_scores['rouge-l']['f'],
                "epoch": epoch + 1
            })

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_model_path = save_checkpoint(model, optimizer, epoch, -1, eval_loss, config, is_best=True)
            logger.info(f"New best model saved with eval loss: {eval_loss:.4f}")
        else:
            logger.info(f"No improvement in eval loss. Current best: {best_eval_loss:.4f}")

        if early_stopping_callback(eval_loss, epoch):
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            if best_model_path:
                logger.info(f"Loading best model from {best_model_path}")
                checkpoint = torch.load(best_model_path)
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    model.module.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint['model_state_dict'])
            break
    
    # 최종 모델 저장 (항상 best model 저장)
    final_model_path = os.path.join(config['general']['output_dir'], "final_model")
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.save_pretrained(final_model_path)
    else:
        model.save_pretrained(final_model_path)
    logger.info(f"Final model (best performing) saved at {final_model_path}")
    
    # 학습이 끝난 후 토크나이저 저장
    save_tokenizer(tokenizer, config['general']['output_dir'])

# 성능 평가 함수
def evaluate(model: torch.nn.Module, dataloader: DataLoader, tokenizer: T5TokenizerFast, device: torch.device, config: Dict[str, Any]):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", dynamic_ncols=True, ascii=True):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config['training']['generation_max_length'],
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                min_length=10,
                use_cache=True
            )
            
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            all_preds.extend([pred.strip() for pred in decoded_preds])
            all_labels.extend([label.strip() for label in decoded_labels])
            
            del input_ids, attention_mask, labels, outputs, generated_ids
            torch.cuda.empty_cache()
    
     # 평균 손실 계산
    avg_loss = total_loss / len(dataloader)

     # ROUGE 점수 계산
    rouge = Rouge()
    if all_preds and all_labels:
        try:
            scores = rouge.get_scores(all_preds, all_labels, avg=True)
        except ValueError as e:
            logger.error(f"Error in ROUGE calculation: {str(e)}")
            logger.info(f"Sample predictions: {all_preds[:5]}")
            logger.info(f"Sample labels: {all_labels[:5]}")
            scores = {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}
    else:
        logger.warning("No valid predictions or labels for ROUGE calculation.")
        scores = {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}
    
    # 결과 로깅
    logger.info(f"Evaluation Loss: {avg_loss:.4f}")
    logger.info("ROUGE Scores:")
    for k, v in scores.items():
        logger.info(f"  {k}: {v['f']:.4f}")
    
    return avg_loss, scores

# from konlpy.tag import Okt
# okt = Okt()

# def extract_keywords(text, n=5):
#     # 형태소 분석 및 품사 태깅
#     morphs = okt.pos(text)
    
#     # 명사, 동사, 형용사만 선택
#     words = [word for word, pos in morphs if pos in ['Noun', 'Verb', 'Adjective']]
    
#     # 불용어 제거 
#     stopwords = set(['있다', '하다', '되다', '이다', '돋다', '않다', '같다', '보다', '이렇다', '그렇다'])
#     words = [word for word in words if word not in stopwords and len(word) > 1]
    
#     # 가장 빈번한 단어 n개 추출
#     return [word for word, _ in Counter(words).most_common(n)]

# 대화에서 주요 키워드 추출(향후 업데이트 할 부분 : 리더보드 점수 향상?)
def extract_keywords(text, n=5):
    # 특수 문자 제거 및 단어 분리
    words = re.findall(r'\w+', text.lower())
    
    # 불용어 제거 (더 포괄적인 리스트 사용 가능)
    stopwords = set(['은', '는', '이', '가', '을', '를', '에', '의', '와', '과', '들', '하다'])
    words = [word for word in words if word not in stopwords]
    
    # 가장 빈번한 단어 n개 추출
    return [word for word, _ in Counter(words).most_common(n)]

# 대화로부터 주제를 추출하는 함수
def generate_topic_from_dialogue(dialogue, n=3, max_length=30):
    keywords = extract_keywords(dialogue, n)
    topic = " ".join(keywords)
    
    # topic의 길이를 제한합니다
    if len(topic) > max_length:
        topic = topic[:max_length].rsplit(' ', 1)[0]  # 단어를 잘라서 자연스럽게 유지
    return topic if topic else "일반 주제"

def extract_subjects(dialogue):
    # 특수 토큰을 가진 모든 화자 탐지
    subjects = re.findall(r'#Person\d*#', dialogue)
    
    # 일반 텍스트로 표현된 모든 화자 탐지 (예: "John: Hello")
    subjects.extend(re.findall(r'(\w+):', dialogue))
    
    # 중복된 화자를 제거하고 순서를 유지하며 반환
    return list(dict.fromkeys(subjects))  # 중복 제거 후 화자 목록 반환

# 특수 토큰 제거
def remove_special_tokens(text, remove_tokens):
    for token in remove_tokens:
        text = text.replace(token, '')
    return text

def post_process_summary(summary):
    summary = re.sub(r'<pad>', '', summary)
    summary = re.sub(r'\s+', ' ', summary)
    
    special_tokens = ['#Person1#', '#Person#', '#Person4#', '#Person2#', '#Person3#', '#Person5#', '#Person6#', '#Person7#']
    for token in special_tokens:
        summary = summary.replace(token, f'{token}')
    
    summary = summary.strip()
    summary = re.sub(r'</s>$', '', summary)
    return summary

# 모델 입력용 프롬프트 생성 (sLLM 사용? 추가 작업 필요)
def create_prompt(dialogue, is_test=False, topic=None, max_topic_length=50):
    subjects = extract_subjects(dialogue)  # 모든 화자를 추출

    if is_test:
        # 테스트 데이터에서는 대화에서 주제를 추출
        topic = generate_topic_from_dialogue(dialogue, max_length=max_topic_length)
    else:
        # 학습 데이터에서는 주어진 topic을 사용
        topic = topic if topic else "주제 없음"
        if len(topic) > max_topic_length:
            topic = topic[:max_topic_length].rsplit(' ', 1)[0]  # 단어를 잘라서 자연스럽게 유지

    keywords = extract_keywords(dialogue)
    subject_list = ', '.join(subjects)  # 화자 목록을 문자열로 변환

    # Basic
    prompt = f"summarize: {dialogue}"
    # 각 화자가 대화에서 한 역할을 설명하는 프롬프트 생성
    #prompt = f"대화 내용 요약:\n- 대화자들: {subject_list}\n- 주제: {topic}\n- 주요 키워드: {', '.join(keywords)}\n\n다음 대화에서 각 화자가 한 역할을 반영하여 요약문을 생성하세요:\n{dialogue}"
    
    return prompt

# 테스트 데이터 적용 함수
def inference(config, model, tokenizer, preprocessor):
    test_data = preprocessor.make_set_as_df(os.path.join(config['general']['data_path'], 'test.csv'), is_train=False)
    fnames = test_data['fname'].tolist()
    test_data.set_index('fname', inplace=True)
    
    encoder_input_test = test_data['dialogue'].apply(lambda x: create_prompt(x, is_test=True)).tolist()
    
    inputs = tokenizer(encoder_input_test, return_tensors="pt", padding=True,
                       add_special_tokens=True, truncation=True,
                       max_length=config['tokenizer']['encoder_max_len'])
    
    dataset = list(zip(inputs['input_ids'], inputs['attention_mask'], fnames))
    
    def collate_fn(batch):
        input_ids, attention_masks, fnames = zip(*batch)
        return torch.stack(input_ids), torch.stack(attention_masks), fnames

    dataloader = DataLoader(dataset, batch_size=config['inference']['batch_size'], 
                            num_workers=4, pin_memory=True, collate_fn=collate_fn)

    model.eval()
    summary = []
    text_ids = []
    
    with torch.no_grad():
        for input_ids, attention_mask, ids in tqdm(dataloader, desc="Inference", dynamic_ncols=True, ascii=True):
            text_ids.extend(ids)
            
            input_ids = input_ids.to(model.device, non_blocking=True)
            attention_mask = attention_mask.to(model.device, non_blocking=True)
            
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                early_stopping=config['inference']['early_stopping'],
                max_length=config['inference']['generate_max_length'],
                #min_length=config['inference']['min_length'],
                num_beams=config['inference']['num_beams'],
                # length_penalty=1.0,
                # repetition_penalty=1.1,
                do_sample=False,
                #temperature=0.3,
                #top_k=50,
                #top_p=0.9,
                output_scores=True,
                return_dict_in_generate=True,
            )
            
            decoded_summaries = tokenizer.batch_decode(generated_ids.sequences, skip_special_tokens=False)
            processed_summaries = [post_process_summary(summary) for summary in decoded_summaries]
            summary.extend(processed_summaries)
    
    output = pd.DataFrame({"fname": text_ids, "summary": summary})

    output_file = os.path.join(config['inference']['result_path'], "inference_output.csv")
    os.makedirs(config['inference']['result_path'], exist_ok=True)
    output.to_csv(output_file, index=False)
    logger.info(f"Inference results saved to: {output_file}")

    return output

def main():
    config = {
        "general": {
            "data_path": "../data/",
            "model_name": "lcw99/t5-large-korean-text-summary",
            "output_dir": "./results"
        },
        "tokenizer": {
            "encoder_max_len": 1024,
            "decoder_max_len": 512,
            "special_tokens": ['#Person1#', '#Person#', '#Person4#', '#CarNumber#', '#Person2#', '#SSN#', '#Person6#',
                               '#DateOfBirth#', '#Email#', '#PhoneNumber#', '#Address#', '#Person3#', '#CardNumber#',
                               '#PassportNumber#', '#Person5#', '#Person7#'],
            'preserve_special_tokens': True,
        },
        "training": {
            "num_train_epochs": 7,
            "learning_rate": 1e-5,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "logging_steps": 1000,
            "eval_steps": 1000,
            "save_steps": 1000,
            "save_total_limit": 6,
            "fp16": True,
            "gradient_accumulation_steps": 6,
            "generation_max_length": 256,
            "early_stopping_patience": 5,
            "early_stopping_threshold": 0.001,
            "max_grad_norm": 1.0,
        },
        "inference": {
            "result_path": "./prediction/",
            "no_repeat_ngram_size": 2,
            "early_stopping": True,
            "generate_max_length": 512, #256
            #"min_length": 50,
            "num_beams": 5, #4
            "batch_size": 4,
            "remove_tokens": ['<usr>', '</s>', '<s>', '<pad>'],
            "ckt_path": "./results/final_t5_model.pt"
        },
    }

    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(dist.get_rank())
    
    logger.info("Starting training...")
    max_retries = 3
    retry_count = 0

    # CUDA Out of Memory Exception 처리 (계속 에러가 나는 경우 kill -9 로 백그라운 파이썬 강제 처리 해야함)
    while retry_count < max_retries:
        try:
            train_and_save(config)
            break
        except RuntimeError as e:
            if "out of memory" in str(e):
                retry_count += 1
                logger.warning(f"CUDA out of memory error occurred. Attempt {retry_count} of {max_retries}.")
                torch.cuda.empty_cache()
                if retry_count == max_retries:
                    logger.error("Max retries reached. Exiting.")
                    raise
            else:
                logger.error("Unexpected error occurred.", exc_info=True)
                raise

    logger.info("Starting inference...")
    try:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model, tokenizer = load_tokenizer_and_model(config, device, for_inference=True)
        preprocessor = Preprocess(tokenizer)
        output = inference(config, model, tokenizer, preprocessor)
        logger.info(f"Inference completed. Output saved to: {os.path.join(config['inference']['result_path'], 'inference_output.csv')}")
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {str(e)}")
    except Exception as e:
        logger.error(f"Error occurred during inference: {str(e)}", exc_info=True)

if __name__ == "__main__":
    torch.set_num_threads(16)
    main()