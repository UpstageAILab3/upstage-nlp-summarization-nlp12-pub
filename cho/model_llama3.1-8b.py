import os
import transformers
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    BitsAndBytesConfig, EarlyStoppingCallback
)
import pandas as pd
from tqdm import tqdm
import wandb
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Wandb 초기화
wandb.init(entity="dl12", project="lm", name="Llama3.1_8b")

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_LENGTH = 512

def load_model_and_tokenizer():
    # 모델 설정
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=False  # Disable KV cache to avoid incompatibility with gradient checkpointing
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    return model, tokenizer

def prepare_model_for_training(model):
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    return model

def preprocess_data(file_path, tokenizer, is_train=True):
    df = pd.read_csv(file_path)
    
    def tokenize_function(examples):
        dialogues = examples['dialogue']
        prompts = [f"Summarize the following dialogue:\n{d}\nSummary:" for d in dialogues]
        
        if is_train:
            summaries = examples['summary']
            inputs = tokenizer(prompts, text_target=summaries, truncation=True, max_length=MAX_LENGTH, padding="max_length")
        else:
            inputs = tokenizer(prompts, truncation=True, max_length=MAX_LENGTH, padding="max_length")
        
        return inputs
    
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    return tokenized_dataset

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge 점수 계산
    rouge = transformers.EvalPrediction(decoded_preds, decoded_labels)
    return rouge.compute()

def train_model(model, train_dataset, eval_dataset):
    training_args = TrainingArguments(
        output_dir="./llama_results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=50,
        save_steps=50,
        warmup_steps=100,
        metric_for_best_model="rougeL",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    return trainer

def generate_summaries(model, tokenizer, test_file):
    df = pd.read_csv(test_file)
    summaries = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Summarizing"):
        dialogue = row['dialogue']
        prompt = f"Summarize the following dialogue into one complete sentence:\n\n{dialogue}\n\nSummary:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=100)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        summaries.append({"fname": row['fname'], "summary": summary})

    return pd.DataFrame(summaries)

if __name__ == "__main__":
    # Set environment variable to suppress kernel warning
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    model, tokenizer = load_model_and_tokenizer()
    model = prepare_model_for_training(model)

    try:
        train_dataset = preprocess_data("../data/train.csv", tokenizer, is_train=True)
        eval_dataset = preprocess_data("../data/dev.csv", tokenizer, is_train=True)

        training_args = TrainingArguments(
            output_dir="./results",
            run_name="llama3.1_8b_run",  # Set a unique run name
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=1e-4,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=50,
            save_steps=50,
            warmup_steps=100,
            metric_for_best_model="rougeL",
            load_best_model_at_end=True,
            gradient_checkpointing=True,  # Enable gradient checkpointing
            gradient_checkpointing_kwargs={"use_reentrant": False},  # Explicitly set use_reentrant to False
            optim="paged_adamw_32bit",
            ddp_find_unused_parameters=False,
            max_grad_norm=0.3,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        trainer.train()

        # Save the fine-tuned model
        trainer.save_model("./lora_finetuned_model")

        # Generate summaries for test set
        test_summaries = generate_summaries(model, tokenizer, "../data/test.csv")
        test_summaries.to_csv("output.csv", index=False, encoding='utf-8')
        print("Summaries have been saved to output.csv")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise