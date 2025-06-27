#!/usr/bin/env python3
"""
prepare_dataset.py - Prepare a sample dataset for Phi-4 fine-tuning
"""

import json
import os

def create_sample_dataset():
    """Create a sample instruction-following dataset"""
    
    # Example: Fine-tuning for technical documentation
    samples = [
        {
            "instruction": "Explain the concept of distributed training in deep learning.",
            "response": "Distributed training in deep learning refers to the practice of training neural networks across multiple computing devices (GPUs or TPUs) to accelerate the training process and handle larger models. There are two main approaches: data parallelism, where the model is replicated across devices with different data batches, and model parallelism, where the model itself is split across devices. This approach enables training larger models and reduces training time significantly."
        },
        {
            "instruction": "What are the main differences between data parallelism and model parallelism?",
            "response": "Data parallelism and model parallelism are two fundamental approaches to distributed training:\n\n1. Data Parallelism: The model is replicated across devices, each processing different batches of data. Gradients are synchronized across devices after each batch. This approach is simpler to implement and works well when the model fits on a single device.\n\n2. Model Parallelism: The model itself is split across devices, with different layers or components on different devices. This is necessary when the model is too large to fit on a single device. It requires more complex coordination and communication between devices."
        },
        {
            "instruction": "Write a Python function to initialize a distributed training environment using PyTorch.",
            "response": "Here's a Python function to initialize distributed training in PyTorch:\n\n```python\nimport torch\nimport torch.distributed as dist\nimport os\n\ndef init_distributed():\n    # Get rank and world size from environment\n    rank = int(os.environ['RANK'])\n    world_size = int(os.environ['WORLD_SIZE'])\n    \n    # Initialize process group\n    dist.init_process_group(\n        backend='nccl',\n        init_method='env://',\n        world_size=world_size,\n        rank=rank\n    )\n    \n    # Set device\n    torch.cuda.set_device(rank)\n    \n    return rank, world_size\n```"
        },
        {
            "instruction": "Explain the benefits of using mixed precision training.",
            "response": "Mixed precision training uses both 32-bit (FP32) and 16-bit (FP16 or BF16) floating-point representations during training. Key benefits include: 1) Reduced memory usage (nearly 50% reduction), allowing larger models or batch sizes; 2) Faster training due to optimized tensor operations on modern GPUs; 3) Maintained model accuracy through automatic loss scaling; 4) Better numerical stability with BF16 compared to FP16. This technique is particularly valuable for large language model training on GPUs like RTX 8000."
        },
        {
            "instruction": "What is gradient accumulation and when should it be used?",
            "response": "Gradient accumulation is a technique where gradients from multiple forward passes are accumulated before performing a backward pass and parameter update. It's used when: 1) GPU memory is limited and you can't fit desired batch sizes; 2) You want to simulate larger batch sizes without additional hardware; 3) Training with very large models that require small per-device batch sizes. The effective batch size becomes: per_device_batch_size × gradient_accumulation_steps × num_gpus. This helps maintain training stability and convergence properties of larger batch training."
        }
    ]
    
    # Extend with more examples for better fine-tuning results
    extended_samples = []
    topics = [
        "optimization techniques", "neural network architectures", "attention mechanisms",
        "transformer models", "reinforcement learning", "computer vision",
        "natural language processing", "model evaluation", "data preprocessing",
        "regularization methods", "loss functions", "activation functions",
        "batch normalization", "dropout techniques", "learning rate scheduling"
    ]
    
    for i, topic in enumerate(topics):
        extended_samples.append({
            "instruction": f"Explain {topic} in machine learning and provide practical examples.",
            "response": f"{topic.title()} in machine learning involves sophisticated techniques for improving model performance and training efficiency. This includes understanding the theoretical foundations, practical implementation considerations, and real-world applications. Key aspects include optimization strategies, computational efficiency, and integration with modern deep learning frameworks like PyTorch and TensorFlow."
        })
    
    samples.extend(extended_samples)
    
    # Save to JSONL format
    output_dir = os.path.expandvars("$SCRATCH/phi4_finetune_data")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/train.jsonl", 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Created {len(samples)} training samples in {output_dir}/train.jsonl")

if __name__ == "__main__":
    create_sample_dataset()