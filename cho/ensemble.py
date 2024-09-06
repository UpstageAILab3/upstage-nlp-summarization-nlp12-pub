import os
import numpy as np
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from torch.utils.data import DataLoader
from rouge import Rouge
from scipy.optimize import minimize

def load_models_and_tokenizers(t5_path, kobart_path, device):
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_path).to(device)
    t5_tokenizer = T5TokenizerFast.from_pretrained(os.path.join(t5_path, "final_t5_tokenizer"))

    kobart_model = BartForConditionalGeneration.from_pretrained(kobart_path).to(device)
    kobart_tokenizer = PreTrainedTokenizerFast.from_pretrained(os.path.join(kobart_path, "final_kobart_tokenizer"))

    return t5_model, t5_tokenizer, kobart_model, kobart_tokenizer

def prepare_data(file_path, t5_tokenizer, kobart_tokenizer, max_length):
    df = pd.read_csv(file_path)
    
    t5_inputs = ["summarize: " + text for text in df['dialogue']]
    kobart_inputs = df['dialogue'].tolist()
    
    t5_encoded = t5_tokenizer(t5_inputs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    kobart_encoded = kobart_tokenizer(kobart_inputs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    
    return t5_encoded, kobart_encoded, df['summary'].tolist()

def generate_summaries(model, tokenizer, dataloader, config, device):
    model.eval()
    summaries = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config['inference']['generate_max_length'],
                num_beams=config['inference']['num_beams'],
                no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                early_stopping=config['inference']['early_stopping']
            )
            
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            summaries.extend(decoded)
    
    return summaries

def ensemble_summaries(t5_summaries, kobart_summaries, weights):
    ensembled_summaries = []
    for t5_sum, kobart_sum in zip(t5_summaries, kobart_summaries):
        t5_tokens = t5_sum.split()
        kobart_tokens = kobart_sum.split()
        
        ensembled_tokens = []
        for t5_token, kobart_token in zip(t5_tokens, kobart_tokens):
            if np.random.rand() < weights[0]:
                ensembled_tokens.append(t5_token)
            else:
                ensembled_tokens.append(kobart_token)
        
        # 남은 토큰들 추가
        if len(t5_tokens) > len(kobart_tokens):
            ensembled_tokens.extend(t5_tokens[len(kobart_tokens):])
        elif len(kobart_tokens) > len(t5_tokens):
            ensembled_tokens.extend(kobart_tokens[len(t5_tokens):])
        
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
        constraints=constraint
    )
    
    optimized_weights = result.x / np.sum(result.x)  # 정규화
    return optimized_weights

def inference_with_ensemble(config, t5_model, t5_tokenizer, kobart_model, kobart_tokenizer, 
                            test_t5_dataloader, test_kobart_dataloader, 
                            val_t5_dataloader, val_kobart_dataloader, val_references):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 검증 데이터로 각 모델의 요약 생성
    val_t5_summaries = generate_summaries(t5_model, t5_tokenizer, val_t5_dataloader, config, device)
    val_kobart_summaries = generate_summaries(kobart_model, kobart_tokenizer, val_kobart_dataloader, config, device)
    
    # 최적의 가중치 찾기
    best_weights = optimize_weights(val_t5_summaries, val_kobart_summaries, val_references)
    print(f"Optimized weights: T5 = {best_weights[0]:.2f}, KoBART = {best_weights[1]:.2f}")
    
    # 테스트 데이터에 대한 요약 생성
    test_t5_summaries = generate_summaries(t5_model, t5_tokenizer, test_t5_dataloader, config, device)
    test_kobart_summaries = generate_summaries(kobart_model, kobart_tokenizer, test_kobart_dataloader, config, device)
    
    # 최적화된 가중치로 앙상블
    final_summaries = ensemble_summaries(test_t5_summaries, test_kobart_summaries, best_weights)
    
    return final_summaries

if __name__ == "__main__":
    config = {
        "general": {
            "data_path": "../data/",
            "t5_model_path": "./results/final_t5_model",
            "kobart_model_path": "./results/final_kobart_model",
            "output_dir": "./results"
        },
        "tokenizer": {
            "max_length": 1024
        },
        "inference": {
            "generate_max_length": 512,
            "num_beams": 6,
            "no_repeat_ngram_size": 2,
            "early_stopping": True,
            "batch_size": 16
        }
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델과 토크나이저 로드
    t5_model, t5_tokenizer, kobart_model, kobart_tokenizer = load_models_and_tokenizers(
        config['general']['t5_model_path'], 
        config['general']['kobart_model_path'], 
        device
    )
    
    # 검증 데이터 준비
    val_file_path = os.path.join(config['general']['data_path'], 'dev.csv')
    val_t5_inputs, val_kobart_inputs, val_references = prepare_data(
        val_file_path, t5_tokenizer, kobart_tokenizer, config['tokenizer']['max_length']
    )
    
    val_t5_dataloader = DataLoader(val_t5_inputs, batch_size=config['inference']['batch_size'])
    val_kobart_dataloader = DataLoader(val_kobart_inputs, batch_size=config['inference']['batch_size'])
    
    # 테스트 데이터 준비
    test_file_path = os.path.join(config['general']['data_path'], 'test.csv')
    test_t5_inputs, test_kobart_inputs, _ = prepare_data(
        test_file_path, t5_tokenizer, kobart_tokenizer, config['tokenizer']['max_length']
    )
    
    test_t5_dataloader = DataLoader(test_t5_inputs, batch_size=config['inference']['batch_size'])
    test_kobart_dataloader = DataLoader(test_kobart_inputs, batch_size=config['inference']['batch_size'])
    
    # 앙상블 추론 실행
    final_summaries = inference_with_ensemble(
        config, t5_model, t5_tokenizer, kobart_model, kobart_tokenizer, 
        test_t5_dataloader, test_kobart_dataloader, 
        val_t5_dataloader, val_kobart_dataloader, val_references
    )
    
    # 결과 저장
    test_df = pd.read_csv(test_file_path)
    output = pd.DataFrame({
        "fname": test_df['fname'],
        "summary": final_summaries,
    })
    output.to_csv(os.path.join(config['general']['output_dir'], "ensemble_output.csv"), index=False)
    print("Ensemble inference completed and results saved.")