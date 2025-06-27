#!/bin/bash
#SBATCH --job-name=phi4_standard_finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300GB                     # Higher memory for standard fine-tuning
#SBATCH --gres=gpu:rtx8000:1           # Need RTX 8000 for 48GB memory
#SBATCH --time=24:00:00                 # Longer time for standard fine-tuning
#SBATCH --partition=gpu48

cd $SCRATCH

# Load modules
module purge
module load singularity/3.11.3

# Set cache directory
export HF_HOME="/vast/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"

# Copy training script
cp $HOME/llm_demos/phi4_instruction/standard_finetune.py .

# Run standard fine-tuning
singularity exec --nv \
    --overlay $HOME/containers/ml_overlay.img:rw \
    $HOME/containers/pytorch_2.1.0-cuda12.1-cudnn8-runtime.sif \
    python standard_finetune.py