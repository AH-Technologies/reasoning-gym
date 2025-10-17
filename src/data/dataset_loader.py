"""Dataset loading utilities for Reasoning Gym tasks."""

import reasoning_gym
from datasets import Dataset
from typing import Dict, List, Any


def create_reasoning_gym_dataset(
    task_name: str,
    num_examples: int,
    seed: int = 42
) -> Dataset:
    """Create a Reasoning Gym dataset and convert to HuggingFace format.

    Args:
        task_name: Name of the Reasoning Gym task
        num_examples: Number of examples to generate
        seed: Random seed for reproducibility

    Returns:
        HuggingFace Dataset with question, answer, and info fields
    """
    rg_data = reasoning_gym.create_dataset(
        task_name,
        size=num_examples,
        seed=seed
    )

    hf_dataset = Dataset.from_dict({
        'question': [entry['question'] for entry in rg_data],
        'answer': [entry['answer'] for entry in rg_data],
        'info': [entry['metadata'] for entry in rg_data]
    })

    return hf_dataset


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

    print(f"\n[Dataset] Creating Reasoning Gym dataset...")
    print(f"  Task: {task_name}")
    print(f"  Examples: {num_examples}")
    print(f"  Seed: {seed}")

    dataset = create_reasoning_gym_dataset(
        task_name=task_name,
        num_examples=num_examples,
        seed=seed
    )

    print(f"✓ Created dataset with {len(dataset)} examples")
    print(f"  Sample Q: {dataset[0]['question']}")
    print(f"  Sample A: {dataset[0]['answer']}")

    return dataset
