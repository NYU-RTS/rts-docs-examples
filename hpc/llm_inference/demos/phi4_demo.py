#!/usr/bin/env python3
"""
phi4_demo.py - Demonstration of Microsoft Phi-4 instruction-tuned model
Author: Your Name
Date: 2025-06-07
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import json

def setup_environment():
    """Configure environment for optimal performance"""
    # Set cache directory to VAST for faster model loading
    os.environ['HF_HOME'] = f"/vast/{os.environ.get('USER', 'default')}/hf_cache"
    os.environ['TRANSFORMERS_CACHE'] = os.environ['HF_HOME']
    
    # Ensure cache directory exists
    os.makedirs(os.environ['HF_HOME'], exist_ok=True)
    
    print(f"[{datetime.now()}] Environment configured")
    print(f"Cache directory: {os.environ['HF_HOME']}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

def load_phi4_model():
    """Load the Phi-4 instruction-tuned model with optimizations"""
    model_id = "microsoft/phi-4"
    
    print(f"\n[{datetime.now()}] Loading Phi-4 model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=os.environ['HF_HOME']
    )
    
    # Configure tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optimizations for inference
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # Use BF16 for RTX 8000 (optimized)
        device_map="auto",          # Automatic device placement
        trust_remote_code=True,
        cache_dir=os.environ['HF_HOME'],
        low_cpu_mem_usage=True     # Reduce CPU memory usage during loading
    )
    
    # Enable eval mode for inference
    model.eval()
    
    print(f"[{datetime.now()}] Model loaded successfully!")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    
    return tokenizer, model

def generate_response(tokenizer, model, prompt, **kwargs):
    """Generate a response for a given prompt"""
    # Default generation parameters
    gen_params = {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    # Update with any provided parameters
    gen_params.update(kwargs)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_params)
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the response
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    return response

def run_demonstrations(tokenizer, model):
    """Run various demonstration prompts"""
    
    demonstrations = [
        {
            "category": "Technical Explanation",
            "prompt": "Explain the benefits of using LoRA for fine-tuning large language models.",
            "max_new_tokens": 300
        },
        {
            "category": "Code Generation",
            "prompt": "Write a Python function that implements binary search on a sorted list. Include docstring and type hints.",
            "max_new_tokens": 400
        },
        {
            "category": "Scientific Writing",
            "prompt": "Explain how attention mechanisms work in transformer models. Use an analogy to make it accessible to beginners.",
            "max_new_tokens": 350
        },
        {
            "category": "Data Analysis",
            "prompt": "What are the key considerations when choosing between supervised and unsupervised learning for a new project?",
            "max_new_tokens": 300
        },
        {
            "category": "Research Planning",
            "prompt": "Outline a research plan for evaluating the efficiency of different LLM inference optimization techniques on HPC systems.",
            "max_new_tokens": 400
        }
    ]
    
    results = []
    
    for i, demo in enumerate(demonstrations, 1):
        print(f"\n{'='*80}")
        print(f"Demonstration {i}/{len(demonstrations)}: {demo['category']}")
        print(f"{'='*80}")
        print(f"Prompt: {demo['prompt']}")
        print(f"{'-'*80}")
        
        start_time = datetime.now()
        
        response = generate_response(
            tokenizer, 
            model, 
            demo['prompt'],
            max_new_tokens=demo['max_new_tokens']
        )
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        print(f"Response:\n{response}")
        print(f"\nGeneration time: {generation_time:.2f} seconds")
        
        # Store results
        results.append({
            "category": demo['category'],
            "prompt": demo['prompt'],
            "response": response,
            "generation_time": generation_time,
            "timestamp": datetime.now().isoformat()
        })
    
    return results

def save_results(results, output_dir="$SCRATCH/phi4_demo_results"):
    """Save demonstration results to JSON"""
    output_dir = os.path.expandvars(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"phi4_demo_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

def main():
    """Main execution function"""
    print("="*80)
    print("Phi-4 Instruction-Tuned Model Demonstration")
    print("NYU Greene HPC")
    print("="*80)
    
    # Setup environment
    setup_environment()
    
    # Load model
    tokenizer, model = load_phi4_model()
    
    # Run demonstrations
    results = run_demonstrations(tokenizer, model)
    
    # Save results
    save_results(results)
    
    print("\nDemonstration completed successfully!")

if __name__ == "__main__":
    main()