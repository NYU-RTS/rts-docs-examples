#!/bin/bash
#SBATCH --job-name=gemma3-finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=pr_100_tandon_priority
#SBATCH --output=/scratch/%u/fine-tune/gemma3_train_%j.out
#SBATCH --error=/scratch/%u/fine-tune/gemma3_train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=%u@nyu.edu

# Note: %u in SBATCH directives will be replaced with your username automatically

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Set up environment variables
export HF_HOME=/scratch/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export CUDA_VISIBLE_DEVICES=0

# Change to working directory
cd /scratch/$USER/fine-tune

# Verify required files exist
if [ ! -f "overlay-25GB-conda.ext3" ]; then
    echo "Error: overlay-25GB-conda.ext3 not found!"
    echo "Please run setup_environment.sh first"
    exit 1
fi

if [ ! -f "train_gemma3.py" ]; then
    echo "Error: train_gemma3.py not found!"
    exit 1
fi

# Run training in Singularity container
echo "Starting Singularity container and training..."
singularity exec --nv \
  --overlay /scratch/$USER/fine-tune/overlay-25GB-conda.ext3:rw \
  /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash -c "
    # Activate conda environment
    source /ext3/miniconda3/bin/activate
    
    # Verify Python and package availability
    echo 'Python version:' \$(python --version)
    echo 'PyTorch version:' \$(python -c 'import torch; print(torch.__version__)')
    echo 'CUDA available:' \$(python -c 'import torch; print(torch.cuda.is_available())')
    echo 'GPU count:' \$(python -c 'import torch; print(torch.cuda.device_count())')
    
    # Change to working directory
    cd /scratch/$USER/fine-tune
    
    # Run the training script
    python train_gemma3.py
"

echo "Job completed at: $(date)"