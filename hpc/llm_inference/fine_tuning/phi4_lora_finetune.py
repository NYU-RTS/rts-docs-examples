#!/usr/bin/env python3
"""
phi4_lora_finetune.py - LoRA fine-tuning for Phi-4 on NYU HPC
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import load_dataset
import json
from datetime import datetime

def setup_environment():
    """Configure environment for fine-tuning"""
    os.environ['HF_HOME'] = f"/vast/{os.environ.get('USER', 'default')}/hf_cache"
    os.environ['TRANSFORMERS_CACHE'] = os.environ['HF_HOME']
    
    # Disable wandb for this demo (enable for real experiments)
    os.environ['WANDB_DISABLED'] = 'true'
    
    print(f"[{datetime.now()}] Environment configured for fine-tuning")

def load_and_prepare_model():
    """Load Phi-4 and prepare for LoRA fine-tuning"""
    model_id = "microsoft/phi-4"
    
    print(f"[{datetime.now()}] Loading Phi-4 for fine-tuning...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=os.environ['HF_HOME']
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in 4-bit for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,  # Use BF16 for RTX 8000 (optimized)
        device_map="auto",
        trust_remote_code=True,
        cache_dir=os.environ['HF_HOME']
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,                    # LoRA rank
        lora_alpha=32,          # LoRA alpha
        lora_dropout=0.1,       # LoRA dropout
        target_modules=[        # Target modules for Phi-4
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    print(f"[{datetime.now()}] Model prepared with LoRA")
    model.print_trainable_parameters()
    
    return model, tokenizer

def prepare_dataset(tokenizer, data_path):
    """Load and prepare the dataset for training"""
    
    def formatting_prompts_func(examples):
        """Format examples into instruction-following format"""
        texts = []
        for instruction, response in zip(examples['instruction'], examples['response']):
            # Phi-4 instruction format
            text = f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n{response}<|end|>"
            texts.append(text)
        return texts
    
    # Load dataset
    dataset = load_dataset('json', data_files=data_path, split='train')
    
    # Format and tokenize
    def tokenize_function(examples):
        formatted_texts = formatting_prompts_func(examples)
        return tokenizer(
            formatted_texts,
            truncation=True,
            padding=True,
            max_length=1024,
            return_tensors="pt"
        )
    
    # Process dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    print(f"[{datetime.now()}] Dataset prepared: {len(tokenized_dataset)} samples")
    
    return tokenized_dataset

def create_trainer(model, tokenizer, train_dataset, output_dir):
    """Create the Trainer with optimized settings for NYU HPC"""
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch size = 2 * 4 = 8
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        bf16=True,                      # Use BF16 for RTX 8000 (optimized)
        gradient_checkpointing=True,    # Save memory
        report_to="none",               # Disable reporting for demo
        dataloader_pin_memory=False,    # Important for NYU HPC
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    return trainer

def main():
    """Main fine-tuning pipeline"""
    print("="*80)
    print("Phi-4 LoRA Fine-Tuning Demonstration")
    print("NYU Greene HPC")
    print("="*80)
    
    # Setup
    setup_environment()
    
    # Paths
    data_path = os.path.expandvars("$SCRATCH/phi4_finetune_data/train.jsonl")
    output_dir = os.path.expandvars("$SCRATCH/phi4_lora_output")
    
    # Load model and tokenizer
    model, tokenizer = load_and_prepare_model()
    
    # Prepare dataset
    train_dataset = prepare_dataset(tokenizer, data_path)
    
    # Create trainer
    trainer = create_trainer(model, tokenizer, train_dataset, output_dir)
    
    # Start training
    print(f"\n[{datetime.now()}] Starting fine-tuning...")
    trainer.train()
    
    # Save the fine-tuned model
    print(f"\n[{datetime.now()}] Saving fine-tuned model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n[{datetime.now()}] Fine-tuning completed!")
    print(f"Model saved to: {output_dir}")

if __name__ == "__main__":
    main()