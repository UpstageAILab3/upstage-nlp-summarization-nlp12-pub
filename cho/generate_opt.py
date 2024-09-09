import os
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import logging
import warnings

from rouge import Rouge
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset, DataLoader

from transformers import (T5ForConditionalGeneration, T5TokenizerFast, 
                          get_cosine_schedule_with_warmup)

# 경고 메시지 무시 및 로깅 설정
warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LongformerEncoderDecoderForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

def load_tokenizer(config):
    tokenizer_path = os.path.join(config['general']['output_dir'], "final_t5_tokenizer")
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path)
    logger.info(f"Tokenizer loaded from {tokenizer_path}")
    return tokenizer

def load_tokenizer_and_model(config, device, for_inference=False):
    logger.info(f"{'Loading tokenizer & model for inference' if for_inference else 'Loading tokenizer & model for training'}")
    model_name = config['general']['model_name']
    
    if for_inference:
        tokenizer = load_tokenizer(config)
        model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained(
            os.path.join(config['general']['output_dir'], "final_t5_model")
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

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge import Rouge

def load_data(filepath):
    data = pd.read_csv(filepath)
    data.set_index('fname', inplace=True)
    return data

def create_prompt(dialogue):
    return f"summarize: {dialogue}"

def prepare_dataset(data, tokenizer, config):
    encoder_input = data['dialogue'].apply(create_prompt).tolist()
    inputs = tokenizer(
        encoder_input,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['encoder_max_len']
    )
    return list(zip(inputs['input_ids'], inputs['attention_mask'], data.index.tolist()))

def create_dataloader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),
            torch.stack([item[1] for item in batch]),
            [item[2] for item in batch]
        )
    )

def validate_model(model, dataloader, tokenizer, device, config, data):
    model.eval()
    rouge = Rouge()
    summaries, references = [], []

    with torch.no_grad():
        # Updated tqdm configuration for single line display
        with tqdm(total=len(dataloader), desc="Validation", bar_format="{desc}: {n_fmt}/{total_fmt}", leave=True, dynamic_ncols=True, ascii=True) as pbar:
            for input_ids, attention_mask, fnames in dataloader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                # Exclude batch_size from the generation config when calling generate
                generate_config = {key: value for key, value in config['inference'].items() if key != 'batch_size'}

                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generate_config  # Updated to use the filtered config
                )
                decoded_summaries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                summaries.extend(decoded_summaries)
                references.extend(data.loc[fnames, 'summary'])

                # Update progress bar after each batch
                pbar.update(1)

    scores = rouge.get_scores(summaries, references, avg=True)
    
    average_score = (scores['rouge-1']['f'] + scores['rouge-2']['f'] + scores['rouge-l']['f']) / 3
    
    print(f"ROUGE-1: {scores['rouge-1']['f']:.4f}")
    print(f"ROUGE-2: {scores['rouge-2']['f']:.4f}")
    print(f"ROUGE-L: {scores['rouge-l']['f']:.4f}")
    print(f"Average ROUGE: {average_score:.4f}")
    
    return average_score

def optimize_parameters(config, model, tokenizer, device, data):
    best_score, best_config = 0, {}
    parameters = {
        'no_repeat_ngram_size': [2, 4],
        'num_beams': [4, 6],
        'max_length': [128, 256, 512],
        'do_sample': [True, False],  # do_sample 파라미터 추가
        'top_k': [50, 100],
        'top_p': [0.9, 0.95],
        'temperature': [0.1, 0.7],
        'repetition_penalty': [1.0, 1.2],
        'length_penalty': [0.8, 1.2],
        'diversity_penalty': [0.0, 0.5],
        'num_return_sequences': [1, 2],
        'min_length': [20, 50]
    }

    dataset = prepare_dataset(data, tokenizer, config)

    for ngram in parameters['no_repeat_ngram_size']:
        for beam in parameters['num_beams']:
            for length in parameters['max_length']:
                for do_sample in parameters['do_sample']:
                    # do_sample이 True일 때만 샘플링 관련 파라미터를 사용
                    if do_sample:
                        top_k_values = parameters['top_k']
                        top_p_values = parameters['top_p']
                        temp_values = parameters['temperature']
                    else:
                        top_k_values = [None]
                        top_p_values = [None]
                        temp_values = [None]
                    
                    for top_k in top_k_values:
                        for top_p in top_p_values:
                            for temp in temp_values:
                                for rep_penalty in parameters['repetition_penalty']:
                                    for len_penalty in parameters['length_penalty']:
                                        for div_penalty in parameters['diversity_penalty']:
                                            for num_return in parameters['num_return_sequences']:
                                                for min_len in parameters['min_length']:
                                                    inference_config = {
                                                        'no_repeat_ngram_size': ngram,
                                                        'num_beams': beam,
                                                        'max_length': length,
                                                        'do_sample': do_sample,
                                                        'repetition_penalty': rep_penalty,
                                                        'length_penalty': len_penalty,
                                                        'diversity_penalty': div_penalty,
                                                        'num_return_sequences': num_return,
                                                        'min_length': min_len
                                                    }
                                                    
                                                    # do_sample이 True일 때만 샘플링 관련 파라미터 추가
                                                    if do_sample:
                                                        inference_config.update({
                                                            'top_k': top_k,
                                                            'top_p': top_p,
                                                            'temperature': temp
                                                        })
                                                    
                                                    config['inference'].update(inference_config)
                                                    dataloader = create_dataloader(dataset, config['inference']['batch_size'])
                                                    score = validate_model(model, dataloader, tokenizer, device, config, data)

                                                    if score > best_score:
                                                        print(f"New best config: ngram:{ngram}, beam:{beam}, length:{length}, "
                                                              f"do_sample:{do_sample}, top_k:{top_k}, top_p:{top_p}, temp:{temp}, "
                                                              f"rep_penalty:{rep_penalty}, len_penalty:{len_penalty}, "
                                                              f"div_penalty:{div_penalty}, num_return:{num_return}, min_len:{min_len}")
                                                        best_score = score
                                                        best_config = config['inference'].copy()

    print("Best Configuration:", best_config)
    return best_config

def main():
    config = {
        "general": {
            "data_path": "../data/",
            "model_name": "lcw99/t5-large-korean-text-summary",
            "output_dir": "./results"
        },
        "tokenizer": {
            "encoder_max_len": 1024,
            "decoder_max_len": 512
        },
        'inference': {
            'batch_size': 4,
            'early_stopping': False
        }
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_tokenizer_and_model(config, device, for_inference=True)
 
    dev_data_path = os.path.join(config['general']['data_path'], 'dev.csv')
    data = load_data(dev_data_path)
    best_config = optimize_parameters(config, model, tokenizer, device, data)
    print(best_config)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()