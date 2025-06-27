#!/usr/bin/env python3
"""
test_finetuned_model.py - Test the fine-tuned Phi-4 model
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_finetuned_model():
    """Load the fine-tuned model with LoRA adapters"""
    base_model_id = "microsoft/phi-4"
    adapter_path = os.path.expandvars("$SCRATCH/phi4_lora_output")
    
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,  # Use BF16 for RTX 8000
        device_map="auto",
        trust_remote_code=True
    )
    
    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    return model, tokenizer

def test_model(model, tokenizer):
    """Test the fine-tuned model with new prompts"""
    
    test_prompts = [
        "Explain the concept of gradient accumulation in distributed training.",
        "What are the best practices for model parallelism?",
        "How does mixed precision training improve performance?"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 60)
        
        # Format as instruction
        formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(formatted_prompt):].strip()
        
        print(f"Response: {response}")

if __name__ == "__main__":
    model, tokenizer = load_finetuned_model()
    test_model(model, tokenizer)