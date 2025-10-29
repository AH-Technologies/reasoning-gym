# Reasoning Gym Training & Benchmarking

Train and benchmark language models on reasoning tasks using GRPO (Group Relative Policy Optimization) with verifiable rewards.

## What This Project Does

This framework lets you:
1. **Train** models using reinforcement learning (GRPO) on reasoning tasks with automatic verification
2. **Benchmark** multiple models across multiple tasks in parallel to measure their reasoning abilities
3. **Compare** your fine-tuned models against baseline models with clean visualizations

The key insight: reasoning tasks have **verifiable answers**, which means we can automatically check if the model got the right answer and use that signal to improve the model through reinforcement learning.

## Understanding Reasoning Gym

[Reasoning Gym](https://github.com/reasoning-gym/reasoning-gym) is a library that provides procedurally-generated reasoning tasks. Each task comes with built-in verification.

### How Reasoning Gym Works

```python
import reasoning_gym

# Create a dataset
dataset = reasoning_gym.create_dataset("leg_counting", size=100, seed=42)

# Each entry has:
for entry in dataset:
    entry['question']   # The question to ask the model
    entry['answer']     # The correct answer
    entry['metadata']   # Additional info about the problem

# The dataset can verify answers automatically
score = dataset.score_answer(
    answer="42",     # Model's answer
    entry=entry      # The original problem
)
# Returns 1.0 if correct, 0.0 if wrong (some tasks return partial credit)
```

Available tasks include:
- `leg_counting`: Count total legs given animals
- `gsm_symbolic`: Math word problems
- `letter_counting`: Count letters in words
- `basic_arithmetic`: Simple arithmetic
- `tower_of_hanoi`: Tower of Hanoi puzzles
- `mini_sudoku`: 4x4 Sudoku puzzles
- `simple_geometry`: Geometry problems
- `shortest_path`: Graph shortest path problems

### Universal Answer Format

This project uses a **unified answer format** that works across all task types:

```
Think step by step and solve the problem carefully. When you have your final answer, format it as: <answer>your answer here</answer>
```

Why XML tags?
- Works for all answer types (numbers, sequences, text)
- Doesn't conflict with task-specific instructions
- Easy to extract programmatically using `reasoning_gym.utils.extract_answer()`
- Follows reasoning-gym's official evaluation approach

This means training and benchmarking use the **exact same prompting and extraction logic**.

## How GRPO Training Works

GRPO (Group Relative Policy Optimization) is a reinforcement learning algorithm designed for training language models with verifiable rewards.

### The Training Loop

```
1. Generate multiple completions per question (e.g., 8 completions)
2. Extract answers from each completion using XML tags
3. Verify each answer using the task's built-in verifier
4. Compute rewards based on correctness
5. Use GRPO to update the model to prefer correct answers
6. Repeat
```

### Why Verifiers Matter

Without verifiers, we can't tell if a model's answer is correct, so we can't provide learning signal. With verifiers:
- Immediate feedback (no human labeling needed)
- Scalable training (generate unlimited practice problems)
- Task-specific correctness (the dataset knows what counts as correct)

### Reward Functions

Located in `src/rewards/`, rewards compute scores for model outputs.

**Correctness Reward** (`src/rewards/correctness.py`):
```python
def compute(self, prompt, completion, answer, info):
    # Extract model's answer from completion
    model_answer = extract_answer_from_response(completion)

    # Score using dataset verifier
    score = self.rg_dataset.score_answer(
        answer=model_answer,
        entry=info  # Original problem metadata
    )

    return score  # 1.0 = correct, 0.0 = wrong
```

You can add custom rewards and combine them:
```yaml
rewards:
  - name: "correctness"
    weight: 0.8
  - name: "brevity"  # Reward shorter answers
    weight: 0.2
```

## Project Structure

```
reasoning-gym/
├── configs/                      # Configuration files
│   ├── base.yaml                 # Base training settings
│   ├── models/                   # Model configs (architecture, LoRA settings)
│   │   └── qwen-7b.yaml
│   ├── tasks/                    # Task configs (which reasoning gym task to use)
│   │   └── leg_counting.yaml
│   └── experiments/              # Experiment configs (combines model + task + settings)
│       └── leg_counting_qwen7b.yaml
│
├── src/                          # Core library code
│   ├── data/                     # Dataset handling
│   │   ├── dataset_utils.py     # Universal formatting, extraction, scoring
│   │   └── dataset_loader.py    # Load datasets from configs
│   ├── rewards/                  # Reward functions for GRPO
│   │   ├── base_reward.py       # Base class for rewards
│   │   ├── correctness.py       # Correctness reward (uses verifier)
│   │   └── registry.py          # Register and load rewards
│   ├── models/                   # Model loading
│   │   └── model_loader.py      # Load HF models or checkpoints with LoRA
│   ├── training/                 # Training logic
│   │   └── trainer.py           # GRPO training loop
│   └── utils/                    # Utilities
│       ├── config.py            # Config loading with inheritance
│       └── memory_tracker.py    # Memory usage tracking
│
├── scripts/                      # Executable scripts
│   ├── train.py                 # Main training script
│   ├── benchmark_models_parallel.py  # Parallel benchmarking
│   ├── plot_benchmark.py        # Generate charts from results
│   └── generate_slurm.py        # Generate SLURM job scripts
│
├── experiments/                  # Training outputs (checkpoints, logs)
├── benchmark_results/            # Benchmark results (JSON + charts)
└── run_*.slurm                   # SLURM job scripts
```

## Training Models

### Quick Start

```bash
# Activate environment
source activate.sh

# Local training
python scripts/train.py configs/experiments/leg_counting_qwen7b.yaml

# Generate SLURM script for cluster
python scripts/generate_slurm.py \
    configs/experiments/leg_counting_qwen7b.yaml \
    your-slurm-account \
    --gpus 4 \
    --time 04:00:00

# Submit to SLURM
sbatch slurm_leg_counting_qwen7b.sh
```

### How Training Works

1. **Dataset Creation** (`src/data/dataset_loader.py`):
   - Generates reasoning problems using reasoning-gym
   - Formats questions with universal XML answer instructions
   - Stores correct answers and metadata

2. **Model Loading** (`src/models/model_loader.py`):
   - Loads base model from HuggingFace
   - Applies LoRA adapters for parameter-efficient fine-tuning
   - Configures for training (gradient checkpointing, etc.)

3. **GRPO Training** (`src/training/trainer.py`):
   - Generates multiple completions per question
   - Extracts answers using `extract_answer_from_response()`
   - Verifies answers using `dataset.score_answer()`
   - Computes rewards and updates model with GRPO

4. **Checkpointing**:
   - Saves LoRA adapters to `{output_dir}/checkpoint-{step}/`
   - Each checkpoint can be loaded as a full model later

### Creating Your Own Experiment

1. **Create a task config** (`configs/tasks/your_task.yaml`):
```yaml
data:
  task_name: "tower_of_hanoi"  # Any reasoning-gym task
  num_examples: 2048
  seed: 42
  add_instructions: true  # Add XML answer instructions

rewards:
  - name: "correctness"
    weight: 1.0
```

2. **Create an experiment config** (`configs/experiments/your_experiment.yaml`):
```yaml
_base_:
  - ../base.yaml
  - ../models/qwen-7b.yaml
  - ../tasks/your_task.yaml

training:
  output_dir: "./experiments/your_experiment"
  num_epochs: 3
  learning_rate: 2.0e-6

logging:
  run_name: "your_experiment"
  wandb_project: "reasoning-gym"
```

3. **Run training**:
```bash
python scripts/train.py configs/experiments/your_experiment.yaml
```

### Config Inheritance

Configs support inheritance via `_base_` to avoid duplication:

- `base.yaml`: Default training hyperparameters (batch size, epochs, etc.)
- `models/*.yaml`: Model-specific settings (which model, LoRA config)
- `tasks/*.yaml`: Task-specific settings (which task, dataset size, rewards)
- `experiments/*.yaml`: Combines the above with experiment-specific overrides

Any setting in child configs overrides parent settings.

## Benchmarking Models

### Parallel Benchmarking

Benchmark multiple models on multiple tasks simultaneously across multiple GPUs:

```bash
# Edit run_benchmark_parallel.slurm to set:
# - TASKS: which reasoning tasks to test
# - MODELS: which models to benchmark
# - NUM_EXAMPLES: how many test problems per task

sbatch run_benchmark_parallel.slurm
```

Example configuration:
```bash
# Test 2 models on 7 tasks = 14 combinations
TASKS=("gsm_symbolic" "letter_counting" "basic_arithmetic" "simple_geometry" "tower_of_hanoi" "mini_sudoku" "shortest_path")

MODELS=(
    "experiments/leg_counting_qwen7b/checkpoint-450"  # Your trained model
    "Qwen/Qwen2.5-7B-Instruct"                        # Baseline
)

# All 14 combinations run in parallel across 8 GPUs
NUM_GPUS=8
```

### How Benchmarking Works

1. **Dataset Creation** (`scripts/benchmark_models_parallel.py`):
   - Creates fixed test sets for each task using same seed
   - Formats questions with XML answer instructions (same as training)

2. **Parallel Execution**:
   - Creates all (task, model) combinations
   - Distributes combinations across GPUs using round-robin assignment
   - Each GPU evaluates one task-model pair at a time
   - All combinations run simultaneously

3. **Answer Extraction & Scoring**:
   - Generates model completion for each test question
   - Extracts answer using `extract_answer_from_response()`
   - Scores using `dataset.score_answer()` (same as training)
   - Computes accuracy: `correct / total`

4. **Results**:
   - Per-task JSON files with detailed results
   - Per-task comparison charts showing all models
   - Combined results across all tasks
   - Summary chart with per-task and average performance

Output files in `benchmark_results/`:
```
results_leg_counting_20251026_121530.json
comparison_leg_counting_20251026_121530.png
results_all_tasks_20251026_121530.json
summary_all_tasks_20251026_121530.png
```

### Visualizing Results

Generate custom charts from benchmark results:

```bash
# Basic usage
python scripts/plot_benchmark.py benchmark_results/results_*.json

# Custom title and renamed models
python scripts/plot_benchmark.py results.json \
    --title "Performance on Math Word Problems" \
    --rename "Qwen/Qwen2.5-7B-Instruct=Qwen 7B" \
    --rename "experiments/leg_counting_qwen7b/checkpoint-450=Fine-tuned Qwen 7B" \
    --output custom_chart.png

# Bulk renaming from file
python scripts/plot_benchmark.py results.json --rename-file mappings.json
```

Rename mapping file format (`mappings.json`):
```json
{
  "Qwen/Qwen2.5-7B-Instruct": "Qwen 7B",
  "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B",
  "experiments/leg_counting_qwen7b/checkpoint-450": "Fine-tuned Qwen 7B"
}
```

The script auto-detects single-task vs multi-task result files and generates appropriate charts.

## Unified Dataset Handling

Both training and benchmarking use identical dataset handling from `src/data/dataset_utils.py`:

### Question Formatting

```python
def format_question(question: str) -> str:
    """Add universal XML answer instruction to any reasoning-gym question."""
    instruction = """Think step by step and solve the problem carefully. When you have your final answer, format it as: <answer>your answer here</answer>

"""
    return instruction + question
```

### Answer Extraction

```python
from reasoning_gym.utils import extract_answer

def extract_answer_from_response(response: str) -> Optional[str]:
    """Extract answer from model response using reasoning-gym's parser."""
    return extract_answer(response, tag_name="answer", strip=True)
```

This function handles:
- Finding the last `<answer>...</answer>` tag in the response
- Extracting and trimming the content
- Returning None if no answer tag found

### Answer Verification

```python
def score_answer(
    dataset: reasoning_gym.dataset.ProceduralDataset,
    model_answer: Optional[str],
    entry: dict[str, Any]
) -> float:
    """Score using the dataset's built-in verifier."""
    try:
        return dataset.score_answer(answer=model_answer, entry=entry)
    except Exception:
        return 0.0
```

Each reasoning-gym dataset has its own `score_answer()` method that knows how to verify answers for that task:
- Number tasks: Numerical comparison with tolerance
- Sequence tasks: Exact sequence matching
- Text tasks: String comparison

This means you **never need to write task-specific verification code** - it's all handled by reasoning-gym's built-in verifiers.

## Key Design Decisions

### Why XML Tags?

Alternative approaches and their issues:
- **Task-specific formats** (e.g., "Final answer: X"): Breaks when tasks already have instructions
- **Regex patterns**: Different tasks have different formats, hard to maintain
- **End-of-sequence**: Model might explain after giving answer

**XML tags solve all of these:**
- Universal format across all tasks
- Built-in to reasoning-gym's evaluation framework
- Doesn't conflict with existing task instructions
- Easy to parse reliably

### Why Unified Dataset Code?

Previously, training and benchmarking had separate dataset handling code, leading to:
- Different prompting between training and evaluation
- Inconsistent answer extraction logic
- Harder to maintain and debug

Now both use `src/data/dataset_utils.py`:
- Same prompts in training and evaluation
- Same answer extraction logic
- Single source of truth for dataset handling
- Changes automatically apply to both training and benchmarking

### Why Verifiers Enable GRPO?

Traditional supervised fine-tuning requires human-labeled examples. GRPO uses reinforcement learning, which requires:

1. **Environment**: The reasoning task (questions from reasoning-gym)
2. **Actions**: Model completions (generated answers)
3. **Rewards**: Correctness scores (from verifiers)

Verifiers provide automatic, scalable rewards:
- No human labeling needed
- Can generate unlimited training problems
- Immediate feedback for learning
- Task-appropriate scoring logic

## Development

### Adding a New Reward Function

1. Create `src/rewards/my_reward.py`:
```python
from src.rewards.base_reward import BaseReward
from src.rewards.registry import register_reward

class MyReward(BaseReward):
    def compute(self, prompt, completion, answer, info):
        # Your reward logic here
        score = ...
        return score

register_reward('my_reward', MyReward)
```

2. Import it in `src/rewards/__init__.py`

3. Use in task config:
```yaml
rewards:
  - name: "correctness"
    weight: 0.8
  - name: "my_reward"
    weight: 0.2
```

### Adding a New Task

No code changes needed! Just create a task config:

```yaml
data:
  task_name: "any_reasoning_gym_task"  # Must exist in reasoning-gym
  num_examples: 2048
  seed: 42
  add_instructions: true

rewards:
  - name: "correctness"
    weight: 1.0
```

### Adding a New Model

No code changes needed! Just create a model config:

```yaml
model:
  name: "HuggingFace/model-name"

lora:
  r: 64
  lora_alpha: 16
  target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj
  lora_dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"
```

## Common Workflows

### Train and Compare

```bash
# 1. Train a model
python scripts/train.py configs/experiments/leg_counting_qwen7b.yaml

# 2. Benchmark trained vs baseline
# Edit run_benchmark_parallel.slurm:
MODELS=(
    "experiments/leg_counting_qwen7b/checkpoint-450"
    "Qwen/Qwen2.5-7B-Instruct"
)

sbatch run_benchmark_parallel.slurm

# 3. Generate clean comparison chart
python scripts/plot_benchmark.py \
    benchmark_results/results_leg_counting_*.json \
    --title "Impact of Fine-tuning on Leg Counting" \
    --rename "experiments/leg_counting_qwen7b/checkpoint-450=Fine-tuned" \
    --rename "Qwen/Qwen2.5-7B-Instruct=Baseline"
```

### Multi-Task Evaluation

```bash
# Test generalization across multiple reasoning tasks
# Edit run_benchmark_parallel.slurm:
TASKS=("gsm_symbolic" "letter_counting" "tower_of_hanoi" "mini_sudoku")

sbatch run_benchmark_parallel.slurm

# Results include per-task and average performance
```

### Hyperparameter Search

```bash
# Create experiments with different hyperparameters
# configs/experiments/leg_counting_lr1.yaml
training:
  learning_rate: 1.0e-6

# configs/experiments/leg_counting_lr2.yaml
training:
  learning_rate: 2.0e-6

# Train all variants
python scripts/train.py configs/experiments/leg_counting_lr1.yaml
python scripts/train.py configs/experiments/leg_counting_lr2.yaml

# Benchmark all checkpoints together
MODELS=(
    "experiments/leg_counting_lr1/checkpoint-450"
    "experiments/leg_counting_lr2/checkpoint-450"
)
```

## Troubleshooting

### Memory Issues

- Reduce `training.per_device_train_batch_size`
- Reduce `training.generation_config.num_return_sequences` (fewer completions per question)
- Enable gradient checkpointing (already on by default)
- Reduce LoRA rank (`lora.r`)

### Low Accuracy

- Check if model is extracting answers correctly (look at generations in training logs)
- Try training longer (more epochs)
- Increase dataset size (`data.num_examples`)
- Adjust learning rate
- Verify the task is appropriate for the model size

### Benchmark Errors

- Ensure all models are accessible (HF models or valid checkpoint paths)
- Check GPU memory (reduce `MAX_NEW_TOKENS` if needed)
- Verify task names are valid reasoning-gym tasks
- Check SLURM logs in `logs/` directory

## References

- [Reasoning Gym](https://github.com/reasoning-gym/reasoning-gym): Procedural reasoning task generation
- [GRPO Paper](https://arxiv.org/abs/2402.03300): Group Relative Policy Optimization
- [LoRA Paper](https://arxiv.org/abs/2106.09685): Low-Rank Adaptation for efficient fine-tuning
- [TRL Library](https://github.com/huggingface/trl): Transformer Reinforcement Learning (used for GRPO implementation)
