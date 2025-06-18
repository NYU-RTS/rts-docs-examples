#!/bin/bash
#SBATCH --job-name=phi4_instruction_demo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1                    # Request any available GPU
#SBATCH --time=02:00:00                 # 2 hours should be sufficient
#SBATCH --output=phi4_demo_%j.out
#SBATCH --error=phi4_demo_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL      # Email notifications
#SBATCH --mail-user=your_netid@nyu.edu  # Replace with your email

# Print job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"

# Navigate to scratch directory
cd $SCRATCH

# Create demo directory
mkdir -p phi4_instruction_demo
cd phi4_instruction_demo

# Copy demo script from home
cp $HOME/llm_demos/phi4_instruction/phi4_demo.py .

# Load required modules
module purge
module load singularity/3.11.3

# Set up environment variables
export HF_HOME="/vast/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export CUDA_VISIBLE_DEVICES=0

# Print GPU information
nvidia-smi

# Run the demonstration using unified Singularity container
echo "Starting Phi-4 demonstration..."

singularity exec --nv \
    --overlay $HOME/containers/ml_overlay.img:ro \
    $HOME/containers/pytorch_2.1.0-cuda12.1-cudnn8-runtime.sif \
    python phi4_demo.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "Demonstration completed successfully!"
else
    echo "Demonstration failed with exit code $?"
fi

echo "Job completed on $(date)"