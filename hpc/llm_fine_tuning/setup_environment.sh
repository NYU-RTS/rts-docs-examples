#!/bin/bash
# setup_environment.sh - Environment setup for LLM fine-tuning on NYU HPC
# This script sets up the Singularity overlay and conda environment for fine-tuning Gemma-3

# Exit on any error
set -e

# Check if NetID is provided
if [ -z "$1" ]; then
    echo "Usage: ./setup_environment.sh <NetID>"
    echo "Example: ./setup_environment.sh abc123"
    exit 1
fi

NETID=$1
WORK_DIR="/scratch/${NETID}/fine-tune"

echo "Setting up environment for NetID: ${NETID}"
echo "Working directory: ${WORK_DIR}"

# Create working directory
echo "Creating working directory..."
mkdir -p ${WORK_DIR}
cd ${WORK_DIR}

# Create Singularity overlay (25GB)
echo "Creating Singularity overlay..."
if [ ! -f "overlay-25GB-conda.ext3" ]; then
    singularity overlay create --size 25000 overlay-25GB-conda.ext3
    echo "Overlay created successfully"
else
    echo "Overlay already exists, skipping creation"
fi

# Download Miniconda installer if not present
echo "Downloading Miniconda installer..."
if [ ! -f "Miniconda3-latest-Linux-x86_64.sh" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
else
    echo "Miniconda installer already downloaded"
fi

# Enter Singularity container to install conda and packages
echo "Entering Singularity container to set up conda environment..."
singularity exec --overlay ${WORK_DIR}/overlay-25GB-conda.ext3:rw \
  /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash << 'EOF'

# Install Miniconda in the overlay
if [ ! -d "/ext3/miniconda3" ]; then
    echo "Installing Miniconda..."
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3
else
    echo "Miniconda already installed"
fi

# Activate conda
source /ext3/miniconda3/bin/activate

# Install required packages
echo "Installing required Python packages..."
pip install torch transformers datasets accelerate peft trl bitsandbytes

echo "Python environment setup complete!"
EOF

# Create HuggingFace cache directory
echo "Creating HuggingFace cache directory..."
mkdir -p /scratch/${NETID}/.cache/huggingface

echo ""
echo "Environment setup complete!"
echo ""
echo "To use this environment, run:"
echo "singularity shell --nv \\"
echo "  --overlay ${WORK_DIR}/overlay-25GB-conda.ext3:rw \\"
echo "  /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif"
echo ""
echo "Then activate conda with:"
echo "source /ext3/miniconda3/bin/activate"
echo ""
echo "Remember to set HF_HOME in your scripts:"
echo "export HF_HOME=/scratch/${NETID}/.cache/huggingface"