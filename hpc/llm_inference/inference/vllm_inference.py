#!/usr/bin/env python3
# vllm_inference.py

from vllm import LLM, SamplingParams
import os

def run_vllm_inference():
    """Example inference with vLLM for optimized performance"""
    
    # Load model with vLLM optimizations
    llm = LLM(
        model="microsoft/phi-4",
        tensor_parallel_size=1,  # Single GPU
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        trust_remote_code=True,
        dtype="bfloat16"  # Use BF16 for RTX 8000
    )
    
    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512
    )
    
    # Example prompts
    prompts = [
        "Explain quantum computing in simple terms:",
        "Write a Python function to calculate fibonacci numbers:",
        "What are the key principles of machine learning?"
    ]
    
    # Generate responses (batched for efficiency)
    outputs = llm.generate(prompts, sampling_params)
    
    # Print results
    for i, output in enumerate(outputs):
        print(f"\n=== Prompt {i+1} ===")
        print(f"Prompt: {output.prompt}")
        print(f"Response: {output.outputs[0].text}")

if __name__ == "__main__":
    run_vllm_inference()