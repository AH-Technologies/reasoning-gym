"""Dataset loading utilities for Reasoning Gym tasks.

This module provides a simple interface for loading datasets, using the unified
dataset utilities that follow reasoning-gym's standard approach.
"""

from typing import Dict, Any

from datasets import Dataset

from .dataset_utils import create_dataset


def create_reasoning_gym_dataset(
    task_name: str,
    num_examples: int,
    seed: int = 42,
    add_instructions: bool = True
) -> Dataset:
    """Create a Reasoning Gym dataset and convert to HuggingFace format.

    This function uses the unified dataset utilities that format questions
    with XML answer tags, following the official reasoning-gym evaluation approach.

    Args:
        task_name: Name of the Reasoning Gym task
        num_examples: Number of examples to generate
        seed: Random seed for reproducibility
        add_instructions: Whether to add XML answer formatting instructions to prompts

    Returns:
        HuggingFace Dataset with question, answer, and info fields
    """
    return create_dataset(
        task_name=task_name,
        num_examples=num_examples,
        seed=seed,
        format_questions=add_instructions
    )


def load_dataset_from_config(config: Dict[str, Any]) -> Dataset:
    """Load dataset based on configuration.

    Args:
        config: Configuration dictionary with data settings

    Returns:
        HuggingFace Dataset
    """
    data_config = config['data']

    task_name = data_config['task_name']
    num_examples = data_config['num_examples']
    seed = data_config.get('seed', 42)
    add_instructions = data_config.get('add_instructions', True)

    print(f"\n[Dataset] Creating Reasoning Gym dataset...")
    print(f"  Task: {task_name}")
    print(f"  Examples: {num_examples}")
    print(f"  Seed: {seed}")
    print(f"  Add instructions: {add_instructions}")

    dataset = create_reasoning_gym_dataset(
        task_name=task_name,
        num_examples=num_examples,
        seed=seed,
        add_instructions=add_instructions
    )

    print(f"✓ Created dataset with {len(dataset)} examples")
    print(f"  Sample Q: {dataset[0]['question'][:200]}...")
    print(f"  Sample A: {dataset[0]['answer']}")

    return dataset
