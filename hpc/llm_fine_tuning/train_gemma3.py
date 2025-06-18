#!/usr/bin/env python3
"""
train_gemma3.py - Fine-tune Gemma-3-4B with LoRA on OpenAssistant Guanaco dataset
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig  
)
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer

# Model and dataset configuration
model_name = "google/gemma-3-4b-pt"  # Base pretrained model
dataset_name = "timdettmers/openassistant-guanaco"
output_dir = "./gemma3_output"

# Load tokenizer
print(f"Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model with quantization for memory efficiency
print(f"Loading model {model_name}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Prepare model for training
model.config.use_cache = False
model.config.pretraining_tp = 1

# LoRA configuration
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA to the model
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Load dataset
print(f"Loading dataset {dataset_name}...")
dataset = load_dataset(dataset_name, split="train")

# For testing, use a small subset (remove or adjust for full training)
# dataset = dataset.select(range(1000))  # Uncomment to use only first 1000 examples

# Preprocessing function
def preprocess_function(examples):
    # Format conversations for instruction tuning
    texts = []
    for conversation in examples["conversations"]:
        formatted_text = ""
        for turn in conversation:
            if turn["from"] == "human":
                formatted_text += f"### Human: {turn['value']}\n\n"
            else:
                formatted_text += f"### Assistant: {turn['value']}\n\n"
        texts.append(formatted_text.strip())
    return {"text": texts}

# Apply preprocessing
print("Preprocessing dataset...")
dataset = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    optim="paged_adamw_32bit",
    warmup_ratio=0.03,
    report_to="none",  # Disable wandb/tensorboard
    load_best_model_at_end=False,
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    packing=False,
)

# Start training
print("Starting training...")
trainer.train()

# Save the model
print(f"Saving model to {output_dir}...")
trainer.save_model()

print("Training complete!")