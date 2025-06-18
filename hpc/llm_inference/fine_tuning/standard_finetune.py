#!/usr/bin/env python3
"""
standard_finetune.py - Standard fine-tuning example for LLMs on NYU HPC
"""

import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch

def setup_model_and_tokenizer(model_name="microsoft/phi-4"):
    """Setup model and tokenizer for standard fine-tuning"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with BF16 to save memory
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use BF16 for RTX 8000
        device_map="auto",
        trust_remote_code=True
    )
    
    return model, tokenizer

def prepare_dataset(data_path, tokenizer, max_length=1024):
    """Prepare dataset for training"""
    
    # Load dataset
    dataset = load_dataset('json', data_files=data_path, split='train')
    
    def tokenize_function(examples):
        # Format: <|user|>\n{prompt}<|end|>\n<|assistant|>\n{response}<|end|>
        formatted_texts = []
        for instruction, response in zip(examples['instruction'], examples['response']):
            formatted_text = f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n{response}<|end|>"
            formatted_texts.append(formatted_text)
        
        # Tokenize
        tokenized = tokenizer(
            formatted_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # Process dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def main():
    """Main fine-tuning function"""
    
    # Setup
    os.environ['HF_HOME'] = f"/vast/{os.environ.get('USER', 'default')}/hf_cache"
    os.environ['TRANSFORMERS_CACHE'] = os.environ['HF_HOME']
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Prepare dataset
    data_path = os.path.expandvars("$SCRATCH/phi4_finetune_data/train.jsonl")
    train_dataset = prepare_dataset(data_path, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="$SCRATCH/phi4_standard_output",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,      # Reduced for memory constraints
        gradient_accumulation_steps=8,       # Effective batch size = 1 * 8 = 8
        learning_rate=2e-5,                 # Lower learning rate for stability
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=True,                          # Use BF16 for RTX 8000
        gradient_checkpointing=True,        # Save memory
        dataloader_pin_memory=False,
        report_to="none",                   # Disable wandb/tensorboard
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("Starting standard fine-tuning...")
    trainer.train()
    
    # Save model
    trainer.save_model("$SCRATCH/phi4_standard_final")
    tokenizer.save_pretrained("$SCRATCH/phi4_standard_final")
    print("Standard fine-tuning completed!")

if __name__ == "__main__":
    main()