## paanmego@gmail.com Maded by 2024.09
import os
import torch
import re
import logging
import warnings

import pandas as pd

import re
from tqdm import tqdm
from rouge import Rouge
from transformers import PreTrainedTokenizerFast
from transformers import GenerationConfig
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, BartTokenizerFast
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import EarlyStoppingCallback
from typing import List, Dict, Any
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import torch.distributed as dist

# 경고 메시지 무시 및 로깅 설정
warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

file_handler = logging.FileHandler('training.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# CUDA 설정 : 성능 최적화
torch.backends.cudnn.benchmark = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Preprocess:
    def __init__(self, tokenizer: BartTokenizerFast):
        self.tokenizer = tokenizer
        self.bos_token = tokenizer.bos_token or ''
        self.eos_token = tokenizer.eos_token or '</s>'  # BART는 </s>를 종결 토큰으로 사용

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
            return df[['fname', 'dialogue', 'summary']]
        else:
            return df[['fname', 'dialogue']]

    def make_input(self, dataset: pd.DataFrame, is_test: bool = False):
        if is_test:
            # BART 모델은 prefix가 필요하지 않음
            encoder_input = dataset['dialogue'].apply(lambda x: str(x))  
            decoder_input = [self.bos_token] * len(dataset['dialogue'])
            return encoder_input.tolist(), list(decoder_input)
        else:
            encoder_input = dataset['dialogue'].apply(lambda x: str(x))  # 요약할 대상을 그대로 입력
            decoder_input = dataset['summary'].apply(lambda x: self.bos_token + str(x))  # BOS 토큰 추가
            decoder_output = dataset['summary'].apply(lambda x: str(x) + self.eos_token)  # EOS 토큰 추가
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()

class DatasetForTrain(Dataset):
    def __init__(self, encoder_input, decoder_input, labels):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item['labels'] = self.labels['input_ids'][idx]
        return item

    def __len__(self):
        return len(self.encoder_input['input_ids'])

class DatasetForVal(Dataset):
    def __init__(self, encoder_input, decoder_input, labels):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item['labels'] = self.labels['input_ids'][idx]
        return item

    def __len__(self):
        return len(self.encoder_input['input_ids'])

class DatasetForInference(Dataset):
    def __init__(self, encoder_input, test_id):
        self.encoder_input = encoder_input
        self.test_id = test_id

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item['ID'] = self.test_id[idx]
        return item

    def __len__(self):
        return len(self.encoder_input['input_ids'])

def load_tokenizer_and_model(config, device, is_train=True):
    print('Load tokenizer & model')
    model_name = config['general']['model_name']
    
    # PreTrainedTokenizerFast로 일관성 맞춤
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    
    # 특수 토큰을 추가
    special_tokens_dict = {'additional_special_tokens': config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)

    # BART 기반의 모델 로드
    if is_train:
        model = BartForConditionalGeneration.from_pretrained(config['general']['model_name'], ignore_mismatched_sizes=True)
    else:
        ckt_path = config['inference']['ckt_path']
        model = BartForConditionalGeneration.from_pretrained(ckt_path)

    # 특수 토큰에 맞게 모델의 임베딩 크기를 조정
    model.resize_token_embeddings(len(tokenizer))

    # generation_config.json을 사용하지 않고 직접 설정
    generation_config = GenerationConfig(
        forced_eos_token_id=tokenizer.eos_token_id,
        max_length=config['inference']['generate_max_length'],  
        num_beams=config['inference']['num_beams'],
        no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
        early_stopping=config['inference']['early_stopping']
        # decoder_start_token_id=tokenizer.bos_token_id,  
        # bos_token_id=tokenizer.bos_token_id 
    )
     
    # 모델의 GenerationConfig 적용
    model.generation_config = generation_config

    model.to(device)
    print('Load tokenizer & model complete')
    return model, tokenizer

def prepare_train_dataset(config, preprocessor, data_path, tokenizer):
    train_file_path = os.path.join(data_path, 'train.csv')
    val_file_path = os.path.join(data_path, 'dev.csv')

    train_data = preprocessor.make_set_as_df(train_file_path)
    val_data = preprocessor.make_set_as_df(val_file_path)

    encoder_input_train, decoder_input_train, decoder_output_train = preprocessor.make_input(train_data)
    encoder_input_val, decoder_input_val, decoder_output_val = preprocessor.make_input(val_data)

    tokenized_encoder_inputs = tokenizer(encoder_input_train, return_tensors="pt", padding=True,
                                         add_special_tokens=True, truncation=True,
                                         max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False)
    tokenized_decoder_inputs = tokenizer(decoder_input_train, return_tensors="pt", padding=True,
                                         add_special_tokens=True, truncation=True,
                                         max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)
    tokenized_decoder_ouputs = tokenizer(text_target=decoder_output_train, return_tensors="pt", padding=True,
                                         add_special_tokens=True, truncation=True,
                                         max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)

    train_inputs_dataset = DatasetForTrain(tokenized_encoder_inputs, tokenized_decoder_inputs, tokenized_decoder_ouputs)

    val_tokenized_encoder_inputs = tokenizer(encoder_input_val, return_tensors="pt", padding=True,
                                             add_special_tokens=True, truncation=True,
                                             max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False)
    val_tokenized_decoder_inputs = tokenizer(decoder_input_val, return_tensors="pt", padding=True,
                                             add_special_tokens=True, truncation=True,
                                             max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)
    val_tokenized_decoder_ouputs = tokenizer(text_target=decoder_output_val, return_tensors="pt", padding=True,
                                             add_special_tokens=True, truncation=True,
                                             max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)

    val_inputs_dataset = DatasetForVal(val_tokenized_encoder_inputs, val_tokenized_decoder_inputs, val_tokenized_decoder_ouputs)

    print('Make dataset complete')
    return train_inputs_dataset, val_inputs_dataset

def compute_metrics(config, tokenizer, pred):
    rouge = Rouge()
    predictions = pred.predictions
    labels = pred.label_ids

    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    decoded_preds = tokenizer.batch_decode(predictions, clean_up_tokenization_spaces=True)
    labels = tokenizer.batch_decode(labels, clean_up_tokenization_spaces=True)

    # 정확한 평가를 위해 미리 정의된 불필요한 생성토큰들을 제거합니다.
    replaced_predictions = decoded_preds.copy()
    replaced_labels = labels.copy()
    remove_tokens = config['inference']['remove_tokens']
    for token in remove_tokens:
        replaced_predictions = [sentence.replace(token," ") for sentence in replaced_predictions]
        replaced_labels = [sentence.replace(token," ") for sentence in replaced_labels]

    # 최종적인 ROUGE 점수를 계산합니다.
    results = rouge.get_scores(replaced_predictions, replaced_labels, avg=True)

    # ROUGE 점수 중 F-1 score를 통해 평가합니다.
    result = {
        "rouge-1": results["rouge-1"]["f"],
        "rouge-2": results["rouge-2"]["f"],
        "rouge-l": results["rouge-l"]["f"]
    }
    return result

def load_trainer_for_train(config, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset):
    print('-'*10, 'Make training arguments', '-'*10,)

    training_args = Seq2SeqTrainingArguments(
        output_dir=config['general']['output_dir'],
        overwrite_output_dir=config['training']['overwrite_output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        learning_rate=config['training']['learning_rate'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        warmup_ratio=config['training']['warmup_ratio'],
        weight_decay=config['training']['weight_decay'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        optim=config['training']['optim'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        evaluation_strategy=config['training']['evaluation_strategy'], 
        save_strategy=config['training']['save_strategy'],
        save_total_limit=config['training']['save_total_limit'],
        fp16=config['training']['fp16'],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=config['training']['seed'],
        logging_dir=config['training']['logging_dir'],
        logging_strategy=config['training']['logging_strategy'],
        predict_with_generate=config['training']['predict_with_generate'],
        generation_max_length=config['training']['generation_max_length'],
        do_train=config['training']['do_train'],
        do_eval=config['training']['do_eval'],
        report_to=config['training']['report_to']
    )

    wandb.init(
        entity=config['wandb']['entity'],
        project=config['wandb']['project'],
        name=config['wandb']['name'],
        config=config,  
    )

    os.environ["WANDB_LOG_MODEL"] = "end"

    print('-'*10, 'Make training arguments complete', '-'*10,)
    print('-'*10, 'Make trainer', '-'*10,)

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=config['training']['early_stopping_patience'],
        early_stopping_threshold=config['training']['early_stopping_threshold']
    )

    trainer = Seq2SeqTrainer(
        model=generate_model,
        args=training_args,
        train_dataset=train_inputs_dataset,
        eval_dataset=val_inputs_dataset,
        compute_metrics=lambda pred: compute_metrics(config, tokenizer, pred),
        callbacks=[early_stopping_callback]
    )
    print('Make trainer complete')

    return trainer

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
    
    preprocessor = Preprocess(tokenizer)
    train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(config, preprocessor, config['general']['data_path'], tokenizer)

    trainer = load_trainer_for_train(config, model, tokenizer, train_inputs_dataset, val_inputs_dataset)
    trainer.train()

    wandb.finish()

    # 최종 모델 저장 (체크포인트 대신 전체 모델 저장)
    final_model_path = os.path.join(config['general']['output_dir'], "final_kobart_model")
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.save_pretrained(final_model_path)
    else:
        model.save_pretrained(final_model_path)
    logger.info(f"Final model saved at {final_model_path}")
    
    # 토크나이저 저장
    tokenizer_save_path = os.path.join(config['general']['output_dir'], "final_kobart_tokenizer")
    tokenizer.save_pretrained(tokenizer_save_path)
    logger.info(f"Final tokenizer saved at {tokenizer_save_path}")
    
    # 저장공간 확보 : 체크포인트 삭제
    for file in os.listdir(config['general']['output_dir']):
        if file.startswith('checkpoint-'):
            file_path = os.path.join(config['general']['output_dir'], file)
            if os.path.isdir(file_path):
                for subfile in os.listdir(file_path):
                    os.remove(os.path.join(file_path, subfile))
                os.rmdir(file_path)
            else:
                os.remove(file_path)

def prepare_test_dataset(config, preprocessor, tokenizer):
    test_file_path = os.path.join(config['general']['data_path'], 'test.csv')

    test_data = preprocessor.make_set_as_df(test_file_path, is_train=False)
    test_id = test_data['fname']

    encoder_input_test, decoder_input_test = preprocessor.make_input(test_data, is_test=True)
    print('Load data complete')

    test_tokenized_encoder_inputs = tokenizer(encoder_input_test, return_tensors="pt", padding=True,
                                              add_special_tokens=True, truncation=True,
                                              max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False)

    test_encoder_inputs_dataset = DatasetForInference(test_tokenized_encoder_inputs, test_id)
    print('Make dataset complete')

    return test_data, test_encoder_inputs_dataset

def load_tokenizer_and_model(config, device, is_train=True):
    print('Load tokenizer & model')
    model_name = config['general']['model_name']

    # PreTrainedTokenizerFast로 일관성 맞춤
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name, clean_up_tokenization_spaces=True)

    # 특수 토큰을 추가 (코드2에서 가져옴)
    special_tokens_dict = {'additional_special_tokens': config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)

    # BART 기반의 모델 로드
    if is_train:
        model = BartForConditionalGeneration.from_pretrained(config['general']['model_name'], ignore_mismatched_sizes=True)
    else:
        ckt_path = config['inference']['ckt_path']
        model = BartForConditionalGeneration.from_pretrained(ckt_path)

    # 특수 토큰에 맞게 모델의 임베딩 크기를 조정
    model.resize_token_embeddings(len(tokenizer))

    generation_config = GenerationConfig(
        forced_eos_token_id=tokenizer.eos_token_id,
        max_length=config['inference']['generate_max_length'],
        num_beams=config['inference']['num_beams'],
        no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
        early_stopping=config['inference']['early_stopping'],
        decoder_start_token_id=tokenizer.bos_token_id,  
        bos_token_id=tokenizer.bos_token_id 
    )
    
    # 모델의 GenerationConfig 적용
    model.generation_config = generation_config

    model.to(device)
    print('Load tokenizer & model complete')
    return model, tokenizer

def extract_first_sentence(summary):
    # 특수 토큰을 유지하면서 첫 번째 문장을 추출
    sentences = re.split(r'(?<=[.!?])\s+', summary)
    if sentences:
        return sentences[0].strip()
    return summary.strip()

def clean_summary(summary):
    # 중복되는 #Person 태그 제거
    summary = re.sub(r'(#Person\d*#)\s*\1+', r'\1', summary)
    
    # #Person 태그와 조사 사이의 불필요한 공백 제거
    summary = re.sub(r'(#Person\d*#)\s+([은는이가을를에의])', r'\1\2', summary)
    
    # 문장 시작 부분의 공백 제거
    summary = summary.strip()
    
    # 여러 개의 연속된 공백을 하나의 공백으로 변경
    summary = re.sub(r'\s+', ' ', summary)
    
    return summary

def inference(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('-'*10, f'device : {device}', '-'*10,)
    print(torch.__version__)

    generate_model, tokenizer = load_tokenizer_and_model(config, device, False)

    data_path = config['general']['data_path']
    preprocessor = Preprocess(tokenizer)

    test_data, test_encoder_inputs_dataset = prepare_test_dataset(config, preprocessor, tokenizer)
    dataloader = DataLoader(test_encoder_inputs_dataset, batch_size=config['inference']['batch_size'])

    summary = []
    text_ids = []
    with torch.no_grad():
        for item in tqdm(dataloader):
            text_ids.extend(item['ID'])
            generated_ids = generate_model.generate(
                input_ids=item['input_ids'].to(device),
                attention_mask=item['attention_mask'].to(device),
                no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                early_stopping=config['inference']['early_stopping'],
                max_length=config['inference']['generate_max_length'],
                num_beams=config['inference']['num_beams'],
            )
            
            for ids in generated_ids:
                result = tokenizer.decode(ids, skip_special_tokens=False)
                summary.append(extract_first_sentence(result))

    # 특수 토큰 목록
    special_tokens = ['<s>', '</s>', '<pad>']

    # 불필요한 특수 토큰 제거 및 요약문 정리
    cleaned_summary = [clean_summary(re.sub(f"({'|'.join(re.escape(token) for token in special_tokens)})", "", sent)) for sent in summary]
    
    output = pd.DataFrame({
        "fname": test_data['fname'],
        "summary": cleaned_summary,
    })

    result_path = config['inference']['result_path']
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    output.to_csv(os.path.join(result_path, "output.csv"), index=False, sep=',')

    return output

if __name__ == "__main__":
    config = {
        "general": {
            "data_path": "../data/",
            "model_name": "digit82/kobart-summarization",
            "output_dir": "./results"
        },
        "tokenizer": {
            "encoder_max_len": 1024,
            "decoder_max_len": 512,
            "special_tokens": ['#Person1#', '#Person#', '#Person4#', '#CarNumber#', '#Person2#', '#SSN#', '#Person6#',
                               '#DateOfBirth#', '#Email#', '#PhoneNumber#', '#Address#', '#Person3#', '#CardNumber#',
                               '#PassportNumber#', '#Person5#', '#Person7#']
        },
        "training": {
            "overwrite_output_dir": True,
            "num_train_epochs": 20,
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 8,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "lr_scheduler_type": 'cosine',
            "optim": 'adamw_torch',
            "gradient_accumulation_steps": 6,
            "evaluation_strategy": 'epoch',
            "save_strategy": 'epoch',  
            "save_total_limit": 1, # 저장공간을 위해
            "fp16": True,
            "load_best_model_at_end": True,
            "seed": 42,
            "logging_dir": "./logs",
            "logging_strategy": "epoch",
            "predict_with_generate": True,
            "generation_max_length": 256,
            "do_train": True,
            "do_eval": True,
            "early_stopping_patience": 3,
            "early_stopping_threshold": 0.001,
            "report_to": "wandb"
        },
        "wandb": {
            "entity": "dl12",
            "project": "lm",
            "name": "digit82/kobart-summarization"
        },
        "inference": {
            "ckt_path": "./results/final_kobart_model",
            "result_path": "./prediction/",
            "no_repeat_ngram_size": 2,
            "early_stopping": True,
            "generate_max_length": 512,
            "num_beams": 5,
            "batch_size": 4,
            "remove_tokens": ['<usr>', "</s>", "<s>", "<pad>"]
        }
    }

    train_and_save(config)
    output = inference(config)