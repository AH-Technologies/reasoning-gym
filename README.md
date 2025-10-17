# GRPO Training with Reasoning Gym

Modular framework for training language models using GRPO (Group Relative Policy Optimization) on Reasoning Gym tasks with verifiers.

## Project Structure

```
reasoning-gym/
├── configs/
│   ├── base.yaml
│   ├── models/
│   │   └── qwen-7b.yaml
│   ├── tasks/
│   │   └── leg_counting.yaml
│   └── experiments/
│       └── leg_counting_qwen7b.yaml
├── src/
│   ├── data/
│   ├── rewards/
│   ├── models/
│   ├── training/
│   └── utils/
├── scripts/
│   ├── train.py
│   └── generate_slurm.py
├── templates/
│   └── slurm_template.sh
└── experiments/
```

## Quick Start

### Local Training

```bash
source activate.sh
python scripts/train.py configs/experiments/leg_counting_qwen7b.yaml
```

### SLURM Training

Generate a SLURM job script:

```bash
python scripts/generate_slurm.py \
    configs/experiments/leg_counting_qwen7b.yaml \
    your-slurm-account \
    --gpus 4 \
    --cpus 16 \
    --memory 320G \
    --time 04:00:00
```

Submit the job:

```bash
sbatch slurm_leg_counting_qwen7b.sh
```

## Configuration

### Config Composition

Configs support inheritance via `_base_`. Experiment configs compose base + model + task settings:

```yaml
_base_:
  - ../base.yaml
  - ../models/qwen-7b.yaml
  - ../tasks/leg_counting.yaml

training:
  output_dir: "./experiments/my_experiment"
```

### Creating New Experiments

1. Create a task config in `configs/tasks/`:

```yaml
data:
  task_name: "your_task"
  num_examples: 2048

rewards:
  - name: "correctness"
    weight: 1.0
```

2. Create an experiment config in `configs/experiments/`:

```yaml
_base_:
  - ../base.yaml
  - ../models/qwen-7b.yaml
  - ../tasks/your_task.yaml

training:
  output_dir: "./experiments/your_experiment"
  learning_rate: 2.0e-6

logging:
  run_name: "your_experiment"
```

3. Run training:

```bash
python scripts/train.py configs/experiments/your_experiment.yaml
```

## Key Features

### Modular Reward Functions

Add custom rewards by extending `BaseReward` and registering them:

```python
from src.rewards.base_reward import BaseReward
from src.rewards.registry import register_reward

class MyReward(BaseReward):
    def compute(self, prompt, completion, answer, info):
        return score

register_reward('my_reward', MyReward)
```

Use in config:

```yaml
rewards:
  - name: "correctness"
    weight: 0.8
  - name: "my_reward"
    weight: 0.2
```

### Automatic Config Logging

Training configs are automatically saved to `{output_dir}/config.yaml` for reproducibility.

### SLURM Generator Options

```bash
python scripts/generate_slurm.py <config> <account> [options]

Options:
  --work-dir     Working directory (default: current directory)
  --partition    SLURM partition (default: GPUQ)
  --time         Time limit (default: 04:00:00)
  --gpus         Number of GPUs (default: 4)
  --gpu-type     GPU type (default: a100)
  --cpus         CPUs per task (default: 16)
  --memory       System memory (default: 320G)
  --output       Output file (default: slurm_<job_name>.sh)
```

## Development

### Adding New Reasoning Tasks

1. Ensure the task exists in Reasoning Gym
2. Create a task config in `configs/tasks/`
3. No code changes needed!

### Adding New Models

1. Create a model config in `configs/models/` with model name and hyperparameters
2. No code changes needed!

### Running Tests

```bash
pytest tests/
```
