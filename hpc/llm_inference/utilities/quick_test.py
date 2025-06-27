#!/usr/bin/env python3
# quick_test.py - Quick model testing

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def quick_test(model_name="microsoft/phi-4"):
    """Quick test of model loading and inference"""
    
    print(f"Testing {model_name}...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Test prompt
    prompt = "Explain machine learning in one sentence:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()
    
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"✅ Model test successful!")

if __name__ == "__main__":
    quick_test()