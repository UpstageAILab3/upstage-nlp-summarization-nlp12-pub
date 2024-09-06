import os
import re
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import EarlyStoppingCallback
from rouge import Rouge
import wandb

class Preprocess:
    def __init__(self, tokenizer):
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token

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


        return df[['fname', 'dialogue', 'summary']] if is_train else df[['fname', 'dialogue']]

    # BART 모델의 입력, 출력 형태를 맞추기 위해 전처리를 진행합니다.
    def make_input(self, dataset, is_test=False):
        encoder_input = dataset['dialogue']
        if is_test:
            decoder_input = [self.bos_token] * len(encoder_input)
            return encoder_input.tolist(), decoder_input
        else:
            decoder_input = dataset['summary'].apply(lambda x: self.bos_token + str(x))
            decoder_output = dataset['summary'].apply(lambda x: str(x) + self.eos_token)
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()
        
class CustomDataset(Dataset):
    def __init__(self, encoder_input, decoder_input, labels=None):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item.update({f'decoder_{key}': val[idx].clone().detach() for key, val in self.decoder_input.items()})
        if self.labels is not None:
            item['labels'] = self.labels['input_ids'][idx]
        return item

    def __len__(self):
        return len(self.encoder_input['input_ids'])

def compute_metrics(config, tokenizer, pred):
    rouge = Rouge()
    predictions, labels = pred.predictions, pred.label_ids
    
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    decoded_preds = tokenizer.batch_decode(predictions, clean_up_tokenization_spaces=True)
    decoded_labels = tokenizer.batch_decode(labels, clean_up_tokenization_spaces=True)

    remove_tokens = config['inference']['remove_tokens']
    for token in remove_tokens:
        decoded_preds = [sentence.replace(token, " ") for sentence in decoded_preds]
        decoded_labels = [sentence.replace(token, " ") for sentence in decoded_labels]

    results = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
    return {key: value["f"] for key, value in results.items()}

def load_trainer(config, model, tokenizer, train_dataset, val_dataset):
    # EarlyStoppingCallback에 사용될 파라미터 분리
    early_stopping_patience = config['training'].pop('early_stopping_patience', 3)
    early_stopping_threshold = config['training'].pop('early_stopping_threshold', 0.001)

    # Seq2SeqTrainingArguments에 전달될 파라미터만 남김
    training_args = Seq2SeqTrainingArguments(**config['training'])

    wandb.init(**config['wandb'])

    os.environ["WANDB_LOG_MODEL"] = "true"
    os.environ["WANDB_WATCH"] = "false"

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold
    )

    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda pred: compute_metrics(config, tokenizer, pred),
        callbacks=[early_stopping_callback]
    )

def load_tokenizer_and_model(config, device, is_train=True):
    if is_train:
        tokenizer = AutoTokenizer.from_pretrained(config['general']['model_name'])
        model = BartForConditionalGeneration.from_pretrained(config['general']['model_name'])
        tokenizer.add_special_tokens({'additional_special_tokens': config['tokenizer']['special_tokens']})
    else:
        model_path = os.path.join(config['general']['output_dir'], 'final_kobart_model')
        tokenizer_path = os.path.join(config['general']['output_dir'], 'final_kobart_tokenizer')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        model = BartForConditionalGeneration.from_pretrained(model_path)

    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    return model, tokenizer

def prepare_dataset(config, preprocessor, tokenizer, file_path, is_train=True):
    data = preprocessor.make_set_as_df(file_path, is_train)
    
    if is_train:
        encoder_input, decoder_input, decoder_output = preprocessor.make_input(data)
    else:
        encoder_input, decoder_input = preprocessor.make_input(data, is_test=True)
        decoder_output = None

    encoder_inputs = tokenizer(encoder_input, return_tensors="pt", padding=True, truncation=True, 
                               max_length=config['tokenizer']['encoder_max_len'])
    decoder_inputs = tokenizer(decoder_input, return_tensors="pt", padding=True, truncation=True, 
                               max_length=config['tokenizer']['decoder_max_len'])
    
    if decoder_output:
        labels = tokenizer(decoder_output, return_tensors="pt", padding=True, truncation=True, 
                           max_length=config['tokenizer']['decoder_max_len'])
    else:
        labels = None

    return CustomDataset(encoder_inputs, decoder_inputs, labels), data['fname'] if not is_train else None

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_tokenizer_and_model(config, device)
    preprocessor = Preprocess(tokenizer)

    train_dataset, _ = prepare_dataset(config, preprocessor, tokenizer, os.path.join(config['general']['data_path'], 'train.csv'))
    val_dataset, _ = prepare_dataset(config, preprocessor, tokenizer, os.path.join(config['general']['data_path'], 'dev.csv'))

    trainer = load_trainer(config, model, tokenizer, train_dataset, val_dataset)
    trainer.train()

    model_save_path = os.path.join(config['general']['output_dir'], 'final_kobart_model')
    tokenizer_save_path = os.path.join(config['general']['output_dir'], 'final_kobart_tokenizer')
    
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)

    wandb.finish()

def clean_summary(text):
    text = re.sub(r'#Person\d*#', lambda m: f'PERSONTAGPLACEHOLDER{m.group()}PERSONTAGPLACEHOLDER', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'PERSONTAGPLACEHOLDER(#Person\d*#)PERSONTAGPLACEHOLDER', r'\1', text)
    text = re.sub(r'(#Person\d*#)(\s*\1)+', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(#Person\d*#)\s', r'\1', text)
    text = text.strip()
    
    return text

def inference(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_tokenizer_and_model(config, device, False)
    preprocessor = Preprocess(tokenizer)

    test_dataset, test_ids = prepare_dataset(config, preprocessor, tokenizer, 
                                             os.path.join(config['general']['data_path'], 'test.csv'), is_train=False)
    dataloader = DataLoader(test_dataset, batch_size=config['inference']['batch_size'])

    summaries = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            generated_ids = model.generate(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                **{k: v for k, v in config['inference'].items() if k not in ['result_path', 'batch_size', 'remove_tokens']}
            )
            summaries.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=False))

    summaries = [clean_summary(summary) for summary in summaries]

    output = pd.DataFrame({"fname": test_ids, "summary": summaries})
    result_path = config['inference']['result_path']
    os.makedirs(result_path, exist_ok=True)
    output.to_csv(os.path.join(result_path, "output.csv"), index=False)

    return output

if __name__ == "__main__":
    config = {
        "general": {
            "data_path": "../data/",
            "model_name": "digit82/kobart-summarization",
            "output_dir": "./results"
        },
        "tokenizer": {
            "encoder_max_len": 512,
            "decoder_max_len": 128,
            "special_tokens": ['#Person1#', '#Person#', '#Person4#', '#CarNumber#', '#Person2#', '#SSN#', '#Person6#',
                               '#DateOfBirth#', '#Email#', '#PhoneNumber#', '#Address#', '#Person3#', '#CardNumber#',
                               '#PassportNumber#', '#Person5#', '#Person7#']
        },
        "training": {
            "output_dir": "./results",
            "overwrite_output_dir": True,
            "num_train_epochs": 20,
            "learning_rate": 1e-5,
            "per_device_train_batch_size": 50,
            "per_device_eval_batch_size": 32,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "lr_scheduler_type": 'cosine',
            "optim": 'adamw_torch',
            "gradient_accumulation_steps": 1,
            "evaluation_strategy": 'epoch',
            "save_strategy": 'epoch',
            "save_total_limit": 1,
            "fp16": True,
            "load_best_model_at_end": True,
            "seed": 42,
            "logging_dir": "./logs",
            "logging_strategy": "epoch",
            "predict_with_generate": True,
            "generation_max_length": 100,
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
            "result_path": "./prediction/",
            "no_repeat_ngram_size": 2,
            "early_stopping": True,
            "max_length": 128,
            "num_beams": 5,
            "batch_size": 32,
            "remove_tokens": ['<usr>', '<s>', '</s>', '<pad>']
        }
    }

    train(config)
    output = inference(config)