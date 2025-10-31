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
- `mini_sudoku`: 4x4 Sudoku puzzles
- `simple_geometry`: Geometry problems

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

You can add custom rewards and combine them (not done for our experiments):
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
│   └── plot_benchmark.py        # Generate charts from results
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

# Training on Idun cluster (GPU required)
# Edit run_training.slurm with your configuration, then:
sbatch run_training.slurm
```

Note: This project has only been tested on the Idun cluster with GPUs. Local training without GPUs has not been tested. You'll need to manually create your own SLURM job script (see `run_training.slurm` and `run_benchmark_parallel.slurm` for examples).

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

3. **Run training on Idun**:
```bash
# Edit run_training.slurm to use your experiment config
sbatch run_training.slurm
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

The parallel benchmark script generates graphs automatically, but you may want to reformat them for presentations or papers. The `plot_benchmark.py` script regenerates graphs from the JSON result files with custom formatting:

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

This is useful for making graphs more presentable by using cleaner model names, custom titles, and different styling.

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

### Why Unified Dataset Code?

Both training and benchmarking use `src/data/dataset_utils.py` for all dataset operations:
- Same prompts in training and evaluation
- Same answer extraction logic
- Single source of truth for dataset handling
- Changes automatically apply to both training and benchmarking

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
# 1. Train a model on Idun
# Edit run_training.slurm to use configs/experiments/leg_counting_qwen7b.yaml
sbatch run_training.slurm

# 2. Benchmark trained vs baseline
# Edit run_benchmark_parallel.slurm:
MODELS=(
    "experiments/leg_counting_qwen7b/checkpoint-450"
    "Qwen/Qwen2.5-7B-Instruct"
)

sbatch run_benchmark_parallel.slurm

# 3. Generate clean comparison chart (can run locally)
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
