#!/bin/bash
# dev_environment.sh - Quick setup for development

# Request interactive session
echo "Requesting interactive GPU session..."
srun --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 \
     --mem=100GB --gres=gpu:1 --time=04:00:00 \
     --partition=interact \
     bash -c "
         cd \$SCRATCH
         
         # Load modules
         module load singularity/3.11.3
         
         # Set environment
         export HF_HOME='/vast/\$USER/hf_cache'
         export TRANSFORMERS_CACHE='\$HF_HOME'
         
         # Start development container
         singularity shell --nv \
             --overlay \$HOME/containers/ml_overlay.img:rw \
             \$HOME/containers/pytorch_2.1.0-cuda12.1-cudnn8-runtime.sif
     "