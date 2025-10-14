#!/bin/bash
# Setup script for GRPO training with Reasoning Gym + Verifiers
# This script installs all dependencies needed for RL training

set -e  # Exit on error

echo "========================================"
echo "GRPO + Reasoning Gym Setup Script"
echo "========================================"

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Install uv (dependency manager)
echo "Installing uv package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "uv already installed"
fi

# Create project directory
echo "Creating project directory..."
mkdir -p grpo_training
cd grpo_training

# Initialize uv project
echo "Initializing uv project..."
uv init --python 3.12

# Activate virtual environment
echo "Creating virtual environment..."
uv venv --python 3.12
source .venv/bin/activate

# Install reasoning-gym
echo "Installing reasoning-gym..."
uv add reasoning-gym

# Install verifiers with training extras
echo "Installing verifiers with training support..."
uv add 'verifiers[train]'

# Install flash-attention (required for GPU training)
echo "Installing flash-attention..."
uv pip install flash-attn --no-build-isolation

# Install optional but useful packages
echo "Installing additional utilities..."
uv add datasets transformers accelerate wandb

# Set environment variables
echo "Setting up environment variables..."
cat > .env << 'EOF'
# Set a dummy OpenAI API key for vLLM (required even though we're not using OpenAI)
export OPENAI_API_KEY="dummy-key-for-vllm"

# Increase socket limit for high concurrency
ulimit -n 4096

# NCCL settings (uncomment if you have GPU communication issues)
# export NCCL_P2P_DISABLE=1
# export NCCL_CUMEM_ENABLE=1

# HuggingFace token (add your token here if needed)
# export HF_TOKEN="your_token_here"

# Weights & Biases (optional, set to None to disable)
# export WANDB_API_KEY="your_wandb_key"
EOF

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "To activate the environment, run:"
echo "  cd grpo_training"
echo "  source .venv/bin/activate"
echo "  source .env"
echo ""
echo "Next steps:"
echo "  1. Review and edit .env file for your credentials"
echo "  2. Run the training script: python train_minimal.py"
echo ""