"""Training entry point."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, save_config
from src.data.dataset_loader import load_dataset_from_config
from src.rewards.registry import create_rewards_from_config
from src.models.model_loader import load_model_and_tokenizer
from src.training.trainer import (
    create_environment,
    create_training_config,
    create_trainer,
    run_training
)


def main():
    """Main training pipeline."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/train.py <config_path>")
        print("\nExample:")
        print("  python scripts/train.py configs/experiments/leg_counting_qwen7b.yaml")
        sys.exit(1)

    config_path = sys.argv[1]

    print("="*60)
    print("GRPO Training with Reasoning Gym")
    print("="*60)

    config = load_config(config_path)
    print(f"\nConfig loaded from: {config_path}")

    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / 'config.yaml')
    print(f"Config saved to: {output_dir / 'config.yaml'}")

    dataset = load_dataset_from_config(config.to_dict())

    task_name = config['data']['task_name']
    rewards = create_rewards_from_config(config['rewards'], task_name)
    print(f"\n[Rewards] Loaded {len(rewards)} reward function(s)")

    env = create_environment(dataset, rewards)

    model, tokenizer = load_model_and_tokenizer(config.to_dict())

    training_config = create_training_config(config.to_dict())

    trainer = create_trainer(model, tokenizer, training_config, env)

    run_training(trainer)

    print(f"\n✓ Model saved to: {config['training']['output_dir']}")
    print("="*60)


if __name__ == "__main__":
    main()
