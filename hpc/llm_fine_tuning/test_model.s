#!/bin/bash
#SBATCH --job-name=gemma3-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --account=pr_100_tandon_priority
#SBATCH --output=/scratch/%u/fine-tune/gemma3_test_%j.out
#SBATCH --error=/scratch/%u/fine-tune/gemma3_test_%j.err

# Simple batch script to test the fine-tuned model

# Set up environment variables
export HF_HOME=/scratch/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export CUDA_VISIBLE_DEVICES=0

# Change to working directory
cd /scratch/$USER/fine-tune

# Run model comparison in Singularity container
singularity exec --nv \
  --overlay /scratch/$USER/fine-tune/overlay-25GB-conda.ext3:rw \
  /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash -c "
    # Activate conda environment
    source /ext3/miniconda3/bin/activate
    
    # Change to working directory
    cd /scratch/$USER/fine-tune
    
    # Run the comparison script
    # Test only the fine-tuned model (faster)
    python compare_models.py --prompt 'Explain quantum computing in simple terms for a beginner'
    
    # Uncomment below to compare all three models
    # python compare_models.py --compare-all --prompt 'Explain quantum computing in simple terms for a beginner'
"