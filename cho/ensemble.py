import os
import re
import numpy as np
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from torch.utils.data import DataLoader, TensorDataset
from rouge import Rouge
from scipy.optimize import minimize
from tqdm import tqdm

def load_models_and_tokenizers(t5_path, kobart_path, t5_tokenizer_path, kobart_tokenizer_path, device):
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_path).to(device)
    t5_tokenizer = T5TokenizerFast.from_pretrained(t5_tokenizer_path)

    kobart_model = BartForConditionalGeneration.from_pretrained(kobart_path).to(device)
    kobart_tokenizer = PreTrainedTokenizerFast.from_pretrained(kobart_tokenizer_path)

    return t5_model, t5_tokenizer, kobart_model, kobart_tokenizer

def prepare_data(file_path, t5_tokenizer, kobart_tokenizer, max_length, is_test=False):
    df = pd.read_csv(file_path)
    
    t5_inputs = ["summarize: " + text for text in df['dialogue']]
    kobart_inputs = df['dialogue'].tolist()

    t5_encoded = t5_tokenizer(t5_inputs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    kobart_encoded = kobart_tokenizer(kobart_inputs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    if not is_test:
        if 'summary' in df.columns:
            summaries = df['summary'].tolist()
        else:
            raise KeyError("'summary' column is missing in the dataset.")
    else:
        summaries = None

    return (
        TensorDataset(t5_encoded['input_ids'], t5_encoded['attention_mask']),
        TensorDataset(kobart_encoded['input_ids'], kobart_encoded['attention_mask']),
        summaries
    )

def generate_summaries(model, tokenizer, dataloader, config, device):
    model.eval()
    summaries = []
    
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(dataloader, desc="Generating summaries", unit="batch"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config['inference']['generate_max_length'],
                num_beams=config['inference']['num_beams'],
                no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                early_stopping=config['inference']['early_stopping'],
                use_cache=False
            )
            
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            cleaned_summaries = [clean_summary(summary) for summary in decoded]
            summaries.extend(cleaned_summaries)

            # 메모리 정리
            del input_ids, attention_mask, outputs
            torch.cuda.empty_cache()
    
    return summaries

def ensemble_summaries(t5_summaries, kobart_summaries, weights):
    ensembled_summaries = []
    for t5_sum, kobart_sum in zip(t5_summaries, kobart_summaries):
        t5_tokens = t5_sum.split()
        kobart_tokens = kobart_sum.split()
        
        ensembled_tokens = []
        for i in range(max(len(t5_tokens), len(kobart_tokens))):
            if i < len(t5_tokens) and i < len(kobart_tokens):
                if np.random.rand() < weights[0]:
                    ensembled_tokens.append(t5_tokens[i])
                else:
                    ensembled_tokens.append(kobart_tokens[i])
            elif i < len(t5_tokens):
                ensembled_tokens.append(t5_tokens[i])
            else:
                ensembled_tokens.append(kobart_tokens[i])
        
        ensembled_summaries.append(' '.join(ensembled_tokens))
    
    return ensembled_summaries

def calculate_rouge(predictions, references):
    rouge = Rouge()
    scores = rouge.get_scores(predictions, references, avg=True)
    return scores

def rouge_objective(weights, t5_summaries, kobart_summaries, reference_summaries):
    weights = np.array(weights)
    weights /= np.sum(weights)  # 정규화
    ensembled_summaries = ensemble_summaries(t5_summaries, kobart_summaries, weights)
    scores = calculate_rouge(ensembled_summaries, reference_summaries)
    return -(scores['rouge-1']['f'] + scores['rouge-2']['f'] + scores['rouge-l']['f']) / 3

def optimize_weights(t5_summaries, kobart_summaries, reference_summaries):
    initial_weights = [0.5, 0.5]
    bounds = [(0, 1), (0, 1)]
    constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    result = minimize(
        rouge_objective, 
        initial_weights, 
        args=(t5_summaries, kobart_summaries, reference_summaries),
        method='SLSQP',
        bounds=bounds,
        constraints=constraint,
        options={'ftol': 1e-6, 'maxiter': 1000}
    )
    
    optimized_weights = result.x / np.sum(result.x)  # 정규화
    return optimized_weights

def clean_summary(text):
    text = re.sub(r'#Person\d*#', lambda m: f'PERSONTAGPLACEHOLDER{m.group()}PERSONTAGPLACEHOLDER', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'PERSONTAGPLACEHOLDER(#Person\d*#)PERSONTAGPLACEHOLDER', r'\1', text)
    text = re.sub(r'(#Person\d*#)(\s*\1)+', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(#Person\d*#)\s', r'\1', text)
    text = text.strip()
    
    return text

def inference_with_ensemble(config, t5_model, t5_tokenizer, kobart_model, kobart_tokenizer, 
                            test_t5_dataloader, test_kobart_dataloader, 
                            val_t5_dataloader, val_kobart_dataloader, val_references):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Generating summaries for validation data...")
    val_t5_summaries = generate_summaries(t5_model, t5_tokenizer, val_t5_dataloader, config, device)
    val_kobart_summaries = generate_summaries(kobart_model, kobart_tokenizer, val_kobart_dataloader, config, device)
    
    print("Optimizing weights...")
    best_weights = optimize_weights(val_t5_summaries, val_kobart_summaries, val_references)
    print(f"Optimized weights: T5 = {best_weights[0]:.4f}, KoBART = {best_weights[1]:.4f}")
    
    # 최적화된 가중치를 사용한 ROUGE 점수 계산
    ensembled_val_summaries = ensemble_summaries(val_t5_summaries, val_kobart_summaries, best_weights)
    val_scores = calculate_rouge(ensembled_val_summaries, val_references)
    print("Validation ROUGE scores with optimized weights:")
    print(f"ROUGE-1: {val_scores['rouge-1']['f']:.4f}")
    print(f"ROUGE-2: {val_scores['rouge-2']['f']:.4f}")
    print(f"ROUGE-L: {val_scores['rouge-l']['f']:.4f}")
    
    print("Generating summaries for test data...")
    test_t5_summaries = generate_summaries(t5_model, t5_tokenizer, test_t5_dataloader, config, device)
    test_kobart_summaries = generate_summaries(kobart_model, kobart_tokenizer, test_kobart_dataloader, config, device)
    
    print("Ensembling summaries...")
    final_summaries = ensemble_summaries(test_t5_summaries, test_kobart_summaries, best_weights)
      
    return final_summaries

if __name__ == "__main__":
    config = {
        "general": {
            "data_path": "../data/",
            "t5_model_path": "./results/final_t5_model",
            "kobart_model_path": "./results/final_kobart_model",
            "t5_tokenizer_path": "./results/final_t5_tokenizer",
            "kobart_tokenizer_path": "./results/final_kobart_tokenizer",
            "output_dir": "./results"
        },
        "tokenizer": {
            "max_length": 512
        },
        "inference": {
            "generate_max_length": 256,
            "num_beams": 5,
            "no_repeat_ngram_size": 2,
            "early_stopping": True,
            "batch_size": 1
        }
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # GPU 메모리 할당 최적화
    torch.cuda.empty_cache()
    
    print("Loading models and tokenizers...")
    t5_model, t5_tokenizer, kobart_model, kobart_tokenizer = load_models_and_tokenizers(
        config['general']['t5_model_path'], 
        config['general']['kobart_model_path'], 
        config['general']['t5_tokenizer_path'],
        config['general']['kobart_tokenizer_path'],
        device
    )
    
    print("Preparing validation data...")
    val_file_path = os.path.join(config['general']['data_path'], 'dev.csv')
    val_t5_dataset, val_kobart_dataset, val_references = prepare_data(
        val_file_path, t5_tokenizer, kobart_tokenizer, config['tokenizer']['max_length'], is_test=False
    )
    
    val_t5_dataloader = DataLoader(val_t5_dataset, batch_size=config['inference']['batch_size'])
    val_kobart_dataloader = DataLoader(val_kobart_dataset, batch_size=config['inference']['batch_size'])
    
    print("Preparing test data...")
    test_file_path = os.path.join(config['general']['data_path'], 'test.csv')
    test_t5_dataset, test_kobart_dataset, _ = prepare_data(
        test_file_path, t5_tokenizer, kobart_tokenizer, config['tokenizer']['max_length'], is_test=True
    )
    
    test_t5_dataloader = DataLoader(test_t5_dataset, batch_size=config['inference']['batch_size'])
    test_kobart_dataloader = DataLoader(test_kobart_dataset, batch_size=config['inference']['batch_size'])

    print("Starting ensemble inference...")
    final_summaries = inference_with_ensemble(
        config, t5_model, t5_tokenizer, kobart_model, kobart_tokenizer, 
        test_t5_dataloader, test_kobart_dataloader, 
        val_t5_dataloader, val_kobart_dataloader, val_references
    )
    
    print("Saving results...")
    test_df = pd.read_csv(os.path.join(config['general']['data_path'], 'test.csv'))
    output = pd.DataFrame({
        "fname": test_df['fname'],
        "summary": final_summaries,
    })
    output.to_csv(os.path.join(config['general']['output_dir'], "ensemble_output.csv"), index=False)
    print("Ensemble inference completed and results saved.")


