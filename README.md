# GRPO Training with Reasoning Gym + Verifiers

This guide provides the bare minimum setup for training models using GRPO (Group Relative Policy Optimization) with Reasoning Gym datasets and the Verifiers library.

## Prerequisites

- **Python 3.11 or 3.12** (required)
- **CUDA-capable GPU** (recommended for training, can use CPU for testing)
- **Linux or macOS** (Windows requires WSL)
- At least **16GB RAM** and **16GB VRAM** for training small models

## Quick Start (3 Steps)

### Step 1: Run the Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Install `uv` package manager
- Create a virtual environment
- Install `reasoning-gym` and `verifiers` with all dependencies
- Install `flash-attention` for efficient GPU training
- Create environment variable file

### Step 2: Activate Environment

```bash
cd grpo_training
source .venv/bin/activate
source .env
```

### Step 3: Run Training

Choose one of two options:

**Option A: Ultra-minimal (recommended for first try)**
```bash
python train_ultra_minimal.py
```

**Option B: Full control with custom rewards**
```bash
python train_minimal.py
```

## What Each Script Does

### `setup.sh`
- Complete installation script
- Sets up all dependencies
- Creates project structure
- Configures environment variables

### `train_ultra_minimal.py` (Easiest)
- Uses `ReasoningGymEnv` directly
- Automatically handles dataset creation and rewards
- ~30 lines of code
- Best for getting started quickly

### `train_minimal.py` (More Control)
- Shows how to create custom reward functions
- Demonstrates manual dataset preparation
- More flexible for customization
- ~150 lines with comments

## Available Reasoning Gym Tasks

You can train on 100+ tasks. Some examples:

**Easy Tasks:**
- `leg_counting` - Count animal legs
- `arithmetic_basic` - Simple arithmetic
- `missing_letter` - Find missing letters

**Medium Tasks:**
- `algebra_linear_1var` - Solve linear equations
- `fizzbuzz` - FizzBuzz variants
- `chess_notation` - Chess move notation

**Hard Tasks:**
- `countdown` - Number game with operations
- `rubiks_cube` - Rubik's cube solving
- `sudoku` - Sudoku puzzles

See full list: https://github.com/open-thought/reasoning-gym/blob/main/GALLERY.md

## GPU Requirements

| Model Size | Minimum VRAM | Recommended Setup |
|------------|--------------|-------------------|
| 1.5B | 16GB | 1x RTX 4090 or A6000 |
| 3B | 24GB | 1x RTX 6000 Ada |
| 7B | 40GB | 1x A100 or 2x RTX 4090 |

For multi-GPU training, see the advanced configuration section.

## Configuration Tips

### Reduce Memory Usage
```python
training_args = vf.GRPOConfig(
    per_device_train_batch_size=1,  # Smaller batch
    gradient_accumulation_steps=8,  # More accumulation
    vllm_gpu_memory_utilization=0.3,  # Less VRAM for vLLM
)
```

### Speed Up Training
```python
training_args = vf.GRPOConfig(
    per_device_train_batch_size=2,
    num_generations=2,  # Fewer generations per prompt
    max_new_tokens=256,  # Shorter responses
)
```

### Better Results
```python
training_args = vf.GRPOConfig(
    num_train_epochs=3,  # More epochs
    num_generations=8,  # More diverse samples
    temperature=0.8,  # More exploration
    learning_rate=5e-6,  # Slower learning
)
```

## Multi-GPU Training

For training with multiple GPUs:

**Step 1: Launch vLLM inference server (terminal 1)**
```bash
CUDA_VISIBLE_DEVICES=0,1,2 vf-vllm \
    --model 'Qwen/Qwen2.5-1.5B-Instruct' \
    --data-parallel-size 3 \
    --enforce-eager \
    --disable-log-requests
```

**Step 2: Launch training (terminal 2)**
```bash
CUDA_VISIBLE_DEVICES=3 accelerate launch \
    --num-processes 1 \
    --config-file configs/zero3.yaml \
    train_minimal.py
```

Create `configs/zero3.yaml`:
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  gradient_accumulation_steps: 4
  zero3_init_flag: true
  zero_stage: 3
```

## Troubleshooting

### "NCCL error" or GPU communication hangs
```bash
export NCCL_P2P_DISABLE=1
```

### "Too many open files"
```bash
ulimit -n 4096
```

### "CUDA out of memory"
- Reduce `per_device_train_batch_size`
- Reduce `vllm_gpu_memory_utilization`
- Use smaller model (1.5B instead of 3B)
- Reduce `num_generations`

### ImportError or module not found
```bash
source .venv/bin/activate  # Make sure venv is activated
uv sync --all-extras  # Reinstall dependencies
```

## Monitoring Training

### With Weights & Biases
1. Get API key from https://wandb.ai
2. Set in `.env`: `export WANDB_API_KEY="your_key"`
3. Change in training args: `report_to="wandb"`

### With TensorBoard
```bash
# In training args, keep report_to="tensorboard"
tensorboard --logdir ./output/runs
```

### Console Output
Watch for:
- **Reward increasing**: Model is learning
- **Loss decreasing**: Optimization working
- **Tokens/second**: Training speed

## Next Steps

1. **Try different tasks**: Change `TASK_NAME` to explore other tasks
2. **Adjust difficulty**: Pass task-specific configs like `max_animals=20`
3. **Compose datasets**: Mix multiple tasks for curriculum learning
4. **Scale up**: Use larger models and more GPUs
5. **Custom tasks**: Create your own Reasoning Gym generators

## Resources

- **Reasoning Gym**: https://github.com/open-thought/reasoning-gym
- **Verifiers**: https://github.com/PrimeIntellect-ai/verifiers
- **Verifiers Docs**: https://verifiers.readthedocs.io
- **GRPO Paper**: https://arxiv.org/abs/2402.03300
- **Reasoning Gym Paper**: https://arxiv.org/abs/2505.24760

## Common Issues

**Q: Can I use CPU instead of GPU?**
A: Yes, but training will be very slow. Set `use_vllm=False` in training args.

**Q: How long does training take?**
A: For this minimal example (~100 samples, 1 epoch), about 10-30 minutes on a single GPU.

**Q: Can I use my own dataset?**
A: Yes! Create a HuggingFace Dataset with `question` and `answer` columns, then use `SingleTurnEnv`.

**Q: Does this work with other models?**
A: Yes! Any HuggingFace model that supports causal LM. Popular choices: Qwen, Llama, Gemma, Phi.

## License

- Reasoning Gym: Apache 2.0
- Verifiers: Apache 2.0