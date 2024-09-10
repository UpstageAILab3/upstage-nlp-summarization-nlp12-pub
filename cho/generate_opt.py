## paanmego@gmail.com Maded by 2024.09

import os
import gc
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import warnings
from tqdm import tqdm

from rouge import Rouge
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import wandb
# Set logging and warnings
warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define model class
class LongformerEncoderDecoderForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

# Function to load tokenizer and model
def load_tokenizer_and_model(config, device, for_inference=False):
    logger.info(f"{'Loading tokenizer & model for inference' if for_inference else 'Loading tokenizer & model for training'}")
    model_name = config['general']['model_name']
    if for_inference:
        tokenizer = T5TokenizerFast.from_pretrained(os.path.join(config['general']['output_dir'], "final_t5_tokenizer"))
        model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained(os.path.join(config['general']['output_dir'], "final_t5_model"))
    else:
        tokenizer = T5TokenizerFast.from_pretrained(model_name)
        model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    return model, tokenizer

# Function to prepare dataset
def prepare_dataset(data, tokenizer, config):
    encoder_input = data['dialogue'].apply(lambda x: f"summarize: {x}").tolist()
    inputs = tokenizer(encoder_input, return_tensors="pt", padding=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'])
    return list(zip(inputs['input_ids'], inputs['attention_mask'], data.index.tolist()))

# Function to create DataLoader
def create_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True, collate_fn=lambda batch: (
        torch.stack([item[0] for item in batch]),
        torch.stack([item[1] for item in batch]),
        [item[2] for item in batch]
    ))

def optimize_parameters(config, model, tokenizer, device, data):
    best_score, best_config = 0, {}
    parameters = {
        'no_repeat_ngram_size': [2, 4],
        'num_beams': [4, 6],
        'max_length': [128, 256, 512],
        'do_sample': [True, False],
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
    dataloader = create_dataloader(dataset, config['inference']['batch_size'])
    
    total_iterations = sum(len(values) for values in parameters.values())
    
    with tqdm(total=total_iterations, desc="Optimizing parameters") as pbar:
        for param_name, param_values in parameters.items():
            for value in param_values:
                config['inference'][param_name] = value
                
                score = validate_model(model, dataloader, tokenizer, device, config, data, config['inference'])
                
                if score > best_score:
                    best_score = score
                    best_config = config['inference'].copy()
                    pbar.set_postfix(best_score=f"{best_score:.4f}", param=f"{param_name}={value}")
                
                wandb.log({
                    param_name: value,
                    'score': score,
                    **{k: v for k, v in config['inference'].items() if k != param_name}
                })
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
                
                pbar.update(1)
    
    print("Best Configuration:", best_config)
    return best_config

def validate_model(model, dataloader, tokenizer, device, config, data, parameters):
    model.eval()
    rouge = Rouge()
    summaries, references = [], []
    
    with torch.no_grad():
        for input_ids, attention_mask, fnames in tqdm(dataloader, desc="Validating", leave=False):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            
            generated_ids = model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                **{k: v for k, v in parameters.items() if k != 'batch_size'}
            )
            
            decoded_summaries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            summaries.extend(decoded_summaries)
            references.extend(data.loc[fnames, 'summary'])
            
            del input_ids, attention_mask, generated_ids
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    scores = rouge.get_scores(summaries, references, avg=True)
    average_score = (scores['rouge-1']['f'] + scores['rouge-2']['f'] + scores['rouge-l']['f']) / 3
    
    return average_score

# Main function
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
            'batch_size': 1,
            'early_stopping': False
        }
    }

    wandb.init(
        entity="dl12",
        project="lm",
        name="lcw99/t5-generate-parameter",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_tokenizer_and_model(config, device, for_inference=True)
    data = pd.read_csv(os.path.join(config['general']['data_path'], 'dev.csv'))
    data.set_index('fname', inplace=True)
    best_config = optimize_parameters(config, model, tokenizer, device, data)
    print(best_config)
    wandb.finish()

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
