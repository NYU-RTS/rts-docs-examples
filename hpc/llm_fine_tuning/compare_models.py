#!/usr/bin/env python3
"""
compare_models.py - Compare outputs from base, fine-tuned, and instruction-tuned Gemma-3 models
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import argparse

def load_base_model(model_name="google/gemma-3-4b-pt"):
    """Load the base pretrained model"""
    print(f"Loading base model: {model_name}")
    
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
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def load_finetuned_model(base_model_name="google/gemma-3-4b-pt", adapter_path="./gemma3_output"):
    """Load the LoRA fine-tuned model"""
    print(f"Loading fine-tuned model from: {adapter_path}")
    
    # First load the base model
    base_model, tokenizer = load_base_model(base_model_name)
    
    # Then load the LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()  # Merge LoRA weights for inference
    
    return model, tokenizer

def load_instruction_tuned_model(model_name="google/gemma-3-4b-it"):
    """Load the official instruction-tuned model"""
    print(f"Loading instruction-tuned model: {model_name}")
    
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
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=256):
    """Generate response from a model"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the input prompt from the response
    response = response[len(prompt):].strip()
    
    return response

def main():
    parser = argparse.ArgumentParser(description="Compare Gemma-3 model variants")
    parser.add_argument("--prompt", type=str, 
                       default="Explain quantum computing in simple terms for a beginner",
                       help="Prompt to test the models")
    parser.add_argument("--adapter-path", type=str, default="./gemma3_output",
                       help="Path to the LoRA adapter checkpoint")
    parser.add_argument("--compare-all", action="store_true",
                       help="Compare all three models (base, fine-tuned, instruction-tuned)")
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"Prompt: {args.prompt}")
    print("="*80)
    
    # Test base pretrained model
    if args.compare_all:
        print("\n### Base Pretrained Model (google/gemma-3-4b-pt):")
        print("-"*80)
        try:
            model, tokenizer = load_base_model()
            response = generate_response(model, tokenizer, args.prompt)
            print(response)
            del model  # Free memory
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error loading base model: {e}")
    
    # Test fine-tuned model
    print("\n### LoRA Fine-tuned Model (Gemma-3-4B-PT + LoRA on Guanaco):")
    print("-"*80)
    try:
        model, tokenizer = load_finetuned_model(adapter_path=args.adapter_path)
        # Format prompt for instruction-following
        formatted_prompt = f"### Human: {args.prompt}\n\n### Assistant:"
        response = generate_response(model, tokenizer, formatted_prompt)
        print(response)
        del model  # Free memory
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        print("Make sure you've completed training and the adapter is saved at:", args.adapter_path)
    
    # Test instruction-tuned model
    if args.compare_all:
        print("\n### Official Instruction-tuned Model (google/gemma-3-4b-it):")
        print("-"*80)
        try:
            model, tokenizer = load_instruction_tuned_model()
            # Format prompt for instruction model
            formatted_prompt = f"<start_of_turn>user\n{args.prompt}<end_of_turn>\n<start_of_turn>model\n"
            response = generate_response(model, tokenizer, formatted_prompt)
            print(response)
            del model  # Free memory
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error loading instruction-tuned model: {e}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()