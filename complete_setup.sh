#!/bin/bash
# Complete setup script for GRPO training with Reasoning Gym + Verifiers
# This installs EVERYTHING you need in one go

set -e  # Exit on error

echo "========================================"
echo "GRPO + Reasoning Gym Complete Setup"
echo "========================================"
echo ""

# Check Python version
echo "Checking Python version..."
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD=python3.12
elif command -v python3.11 &> /dev/null; then
    PYTHON_CMD=python3.11
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo "Error: Python 3.11 or 3.12 not found!"
    exit 1
fi

python_version=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Install uv (dependency manager)
echo ""
echo "Installing uv package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "uv already installed"
fi

# Create project directory
echo ""
echo "Creating project directory..."
mkdir -p grpo_training
cd grpo_training

# Initialize uv project
echo "Initializing uv project..."
if [ ! -f "pyproject.toml" ]; then
    uv init --python $PYTHON_CMD
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
uv venv --python $PYTHON_CMD

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install reasoning-gym
echo ""
echo "Installing reasoning-gym..."
uv add reasoning-gym

# Install verifiers with training extras from GitHub (latest version)
echo ""
echo "Installing verifiers with training support..."
uv add 'verifiers[train] @ git+https://github.com/PrimeIntellect-ai/verifiers.git'

# Install flash-attention (required for GPU training)
echo ""
echo "Installing flash-attention..."
uv pip install flash-attn --no-build-isolation

# Install additional utilities
echo ""
echo "Installing additional utilities..."
uv add datasets transformers accelerate

# Install the reasoning_gym_env environment module
echo ""
echo "Installing reasoning_gym_env environment module..."
if command -v vf-install &> /dev/null; then
    vf-install reasoning_gym_env --from-repo || echo "Warning: Could not install reasoning_gym_env module (continuing anyway)"
else
    echo "Warning: vf-install not available yet (continuing anyway)"
fi

# Set environment variables
echo ""
echo "Setting up environment variables..."
cat > .env << 'EOF'
# Environment variables for GRPO training

# Dummy OpenAI API key for vLLM (required even though we're not using OpenAI)
export OPENAI_API_KEY="dummy-key-for-vllm"

# Increase socket limit for high concurrency
ulimit -n 4096

# NCCL settings (uncomment if you have GPU communication issues)
# export NCCL_P2P_DISABLE=1
# export NCCL_CUMEM_ENABLE=1

# HuggingFace token (optional - add your token if needed)
# export HF_TOKEN="your_token_here"

# Weights & Biases (optional - uncomment and add key for W&B logging)
# export WANDB_API_KEY="your_wandb_key"
EOF

# Create activation helper script
cat > activate.sh << 'EOF'
#!/bin/bash
# Helper script to activate the environment
source .venv/bin/activate
source .env
echo "Environment activated!"
echo "You can now run: python train.py"
EOF
chmod +x activate.sh

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "Project created in: $(pwd)"
echo ""
echo "Next steps:"
echo "  1. Copy train.py into this directory"
echo "  2. Activate environment: source activate.sh"
echo "  3. Run training: python train.py"
echo ""
echo "Or run everything in one line:"
echo "  source activate.sh && python train.py"
echo ""