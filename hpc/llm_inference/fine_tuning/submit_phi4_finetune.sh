#!/bin/bash
#SBATCH --job-name=phi4_lora_finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200GB
#SBATCH --gres=gpu:rtx8000:1           # Request RTX 8000 for 48GB memory
#SBATCH --time=06:00:00                 # 6 hours for fine-tuning
#SBATCH --output=phi4_finetune_%j.out
#SBATCH --error=phi4_finetune_%j.err

echo "Fine-tuning job started on $(date)"
echo "Running on node: $(hostname)"

# Navigate to scratch
cd $SCRATCH

# Create working directory
mkdir -p phi4_finetune
cd phi4_finetune

# Copy scripts
cp $HOME/llm_demos/phi4_instruction/prepare_dataset.py .
cp $HOME/llm_demos/phi4_instruction/phi4_lora_finetune.py .

# Load modules
module purge
module load singularity/3.11.3

# Environment setup
export HF_HOME="/vast/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"

# First, prepare the dataset
echo "Preparing dataset..."
singularity exec --nv \
    --overlay $HOME/containers/ml_overlay.img:rw \
    $HOME/containers/pytorch_2.1.0-cuda12.1-cudnn8-runtime.sif \
    python prepare_dataset.py

# Run fine-tuning
echo "Starting LoRA fine-tuning..."
singularity exec --nv \
    --overlay $HOME/containers/ml_overlay.img:rw \
    $HOME/containers/pytorch_2.1.0-cuda12.1-cudnn8-runtime.sif \
    python phi4_lora_finetune.py

echo "Fine-tuning completed on $(date)"