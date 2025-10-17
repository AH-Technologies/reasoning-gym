#!/usr/bin/env python3
"""
GRPO Training Script
This is the main entry point for training, typically called from slurm scripts.

Usage:
    python scripts/train.py configs/experiments/leg_counting_qwen7b.yaml
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import src modules
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


def print_training_banner(config):
    """Print detailed training information banner."""
    print("\n" + "="*60)
    print("GRPO Training with Reasoning Gym")
    print("="*60)
    print(f"Model: {config['model']['name']}")
    print(f"Task: {config['data']['task_name']}")
    print(f"Examples: {config['data']['num_examples']}")
    print(f"Output: {config['training']['output_dir']}")
    print("="*60)
    print("\nThis will take 10-30 minutes depending on your GPU.")
    print("Press Ctrl+C to stop training at any time.")
    print("="*60)


def main():
    """Main training pipeline.

    Orchestrates the complete training workflow:
    1. Load configuration
    2. Create dataset with formatting
    3. Load model and tokenizer
    4. Create reward functions
    5. Create training environment
    6. Configure and run training
    """
    if len(sys.argv) != 2:
        print("Usage: python scripts/train.py <config_path>")
        print("\nExample:")
        print("  python scripts/train.py configs/experiments/leg_counting_qwen7b.yaml")
        print("  python scripts/train.py configs/base.yaml")
        sys.exit(1)

    config_path = sys.argv[1]

    # Load configuration
    print(f"\n[Config] Loading from: {config_path}")
    config = load_config(config_path)

    # Create output directory and save config
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / 'config.yaml')
    print(f"[Config] Saved to: {output_dir / 'config.yaml'}")

    # Print training banner
    print_training_banner(config.to_dict())

    # Step 1: Create dataset (with prompt formatting from old code)
    print("\n[1/5] Creating dataset...")
    dataset = load_dataset_from_config(config.to_dict())

    # Step 2: Create reward functions (with debug logging from old code)
    print("\n[2/5] Setting up reward functions...")
    task_name = config['data']['task_name']
    rewards = create_rewards_from_config(config['rewards'], task_name)
    print(f"✓ Created {len(rewards)} reward function(s)")
    for reward in rewards:
        print(f"  - {reward.__class__.__name__} (weight={reward.weight})")

    # Step 3: Create environment
    print("\n[3/5] Creating training environment...")
    env = create_environment(dataset, rewards)

    # Step 4: Load model and tokenizer
    print("\n[4/5] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config.to_dict())

    # Step 5: Configure training
    print("\n[5/5] Configuring training...")
    training_config = create_training_config(config.to_dict())

    # Create trainer
    trainer = create_trainer(model, tokenizer, training_config, env)

    # Run training
    run_training(trainer)

    # Final success message
    print("\n" + "="*60)
    print("✓ Training complete!")
    print(f"✓ Model saved to: {config['training']['output_dir']}")
    print("="*60)


if __name__ == "__main__":
    main()
