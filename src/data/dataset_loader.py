"""Dataset loading utilities for Reasoning Gym tasks."""

import reasoning_gym
from datasets import Dataset
from typing import Dict, List, Any, Optional


def format_question_with_instructions(question: str, task_name: str = "leg_counting") -> str:
    """Add clear instructions for how to format the answer.

    Args:
        question: The question to format
        task_name: Name of the task (for task-specific instructions)

    Returns:
        Formatted question with instructions
    """
    instruction = """You are solving a math problem. Follow these steps:
1. Think through the problem step by step
2. Show your calculation
3. On the last line, write your final answer EXACTLY in this format: "Final Answer: [number]"

"""
    return instruction + question


def create_reasoning_gym_dataset(
    task_name: str,
    num_examples: int,
    seed: int = 42,
    add_instructions: bool = True
) -> Dataset:
    """Create a Reasoning Gym dataset and convert to HuggingFace format.

    Args:
        task_name: Name of the Reasoning Gym task
        num_examples: Number of examples to generate
        seed: Random seed for reproducibility
        add_instructions: Whether to add formatting instructions to prompts

    Returns:
        HuggingFace Dataset with question, answer, and info fields
    """
    rg_data = reasoning_gym.create_dataset(
        task_name,
        size=num_examples,
        seed=seed
    )

    # Format questions with instructions if requested
    questions = [entry['question'] for entry in rg_data]
    if add_instructions:
        questions = [format_question_with_instructions(q, task_name) for q in questions]

    hf_dataset = Dataset.from_dict({
        'question': questions,
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
