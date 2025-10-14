# GRPO Training Quick Reference

## Installation (Choose One Method)

### Method 1: Using setup.sh (Recommended)
```bash
chmod +x setup.sh && ./setup.sh
cd grpo_training && source .venv/bin/activate && source .env
```

### Method 2: Using pip manually
```bash
python3 -m venv venv
source venv/bin/activate
pip install reasoning-gym 'verifiers[train]'
pip install flash-attn --no-build-isolation
export OPENAI_API_KEY="dummy-key"
```

## Run Training (One Command)

```bash
python train_ultra_minimal.py
```

## Change Task

Edit the script and change:
```python
TASK_NAME = "leg_counting"  # Change to any task below
```

### Popular Tasks
- `leg_counting` - Animal leg counting (easiest)
- `arithmetic_basic` - Basic math
- `algebra_linear_1var` - Linear equations
- `fizzbuzz` - FizzBuzz game
- `chess_notation` - Chess moves
- `countdown` - Number operations game
- `sudoku` - Sudoku puzzles

## Common Configurations

### Smallest/Fastest Setup (for testing)
```python
env = vf.ReasoningGymEnv(task_name="leg_counting", num_examples=50)
training_args = vf.GRPOConfig(
    num_train_epochs=1,
    per_device_train_batch_size=1,
    num_generations=2,
    max_new_tokens=128,
)
```

### Production Setup (better results)
```python
env = vf.ReasoningGymEnv(task_name="algebra_linear_1var", num_examples=1000)
training_args = vf.GRPOConfig(
    num_train_epochs=3,
    per_device_train_batch_size=2,
    num_generations=8,
    learning_rate=5e-6,
)
```

## Composite Dataset (Multiple Tasks)

```python
import reasoning_gym
from reasoning_gym.composite import DatasetSpec

specs = [
    DatasetSpec(name='leg_counting', weight=1, config={}),
    DatasetSpec(name='fizzbuzz', weight=1, config={}),
    DatasetSpec(name='arithmetic_basic', weight=2, config={}),
]

dataset = reasoning_gym.create_dataset('composite', size=500, datasets=specs)
```

## Model Options

Small models (fit on 16GB GPU):
- `Qwen/Qwen2.5-1.5B-Instruct` (fastest)
- `microsoft/phi-2` (2.7B)
- `google/gemma-2b-it`

Medium models (need 24GB+ GPU):
- `Qwen/Qwen2.5-3B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`

## Essential GRPOConfig Parameters

```python
vf.GRPOConfig(
    output_dir="./output",           # Where to save model
    num_train_epochs=1,               # How many times through data
    per_device_train_batch_size=1,   # Batch size per GPU
    gradient_accumulation_steps=4,   # Effective batch = batch_size * this
    learning_rate=1e-5,               # How fast to learn
    num_generations=4,                # Responses per prompt
    temperature=0.7,                  # Randomness (0=deterministic, 1=creative)
    max_new_tokens=512,               # Max response length
    use_vllm=True,                    # Use vLLM for fast inference
    beta=0.0,                         # KL penalty (usually 0)
    report_to="none",                 # Or "wandb"/"tensorboard"
)
```

## Check Progress

Training output shows:
```
Epoch 1/1 [=====>] 100/100 [00:30<00:00, 3.33it/s]
Reward: 0.75 | Loss: 0.45 | Tokens/sec: 1250
```

Good signs:
- ✅ Reward increasing over time
- ✅ Loss decreasing
- ✅ No errors or warnings

## Evaluate After Training

```python
import reasoning_gym

# Create test set
test_data = reasoning_gym.create_dataset('leg_counting', size=10, seed=999)

# Load your trained model
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./output")
tokenizer = AutoTokenizer.from_pretrained("./output")

# Test on a question
question = test_data[0]['question']
inputs = tokenizer(question, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0])

print(f"Question: {question}")
print(f"Model: {response}")
print(f"Answer: {test_data[0]['answer']}")
```

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| CUDA OOM | Reduce `per_device_train_batch_size` to 1 |
| Too slow | Reduce `num_examples` and `num_generations` |
| Poor results | Increase `num_train_epochs` and `learning_rate` |
| NCCL errors | `export NCCL_P2P_DISABLE=1` |
| Too many files open | `ulimit -n 4096` |

## Environment Variables

Add to your shell or `.env` file:
```bash
export OPENAI_API_KEY="dummy-key-for-vllm"
export HF_TOKEN="your_huggingface_token"  # Optional
export WANDB_API_KEY="your_wandb_key"     # Optional
export CUDA_VISIBLE_DEVICES="0"           # Which GPU to use
ulimit -n 4096
```

## File Structure After Setup

```
grpo_training/
├── .venv/                  # Virtual environment
├── .env                    # Environment variables
├── train_ultra_minimal.py  # Training script
├── train_minimal.py        # Alternative script
├── output/                 # Saved models (created during training)
└── pyproject.toml          # Dependencies (if using uv)
```

## Next Steps Checklist

- [ ] Run `setup.sh`
- [ ] Activate environment
- [ ] Run `train_ultra_minimal.py` with default settings
- [ ] Check that training completes without errors
- [ ] Try different task: change `TASK_NAME`
- [ ] Adjust `num_examples` and `num_train_epochs`
- [ ] Enable wandb logging for better monitoring
- [ ] Try a larger model if you have more GPU memory

## Resources

- Tasks gallery: https://github.com/open-thought/reasoning-gym/blob/main/GALLERY.md
- Verifiers docs: https://verifiers.readthedocs.io
- Get help: GPU-Mode Discord #reasoning-gym channel

---

**Pro tip**: Start with `leg_counting` task and 50 examples to verify everything works, then scale up!