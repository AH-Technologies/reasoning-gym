"""Unified dataset utilities for reasoning-gym tasks.

This module provides consistent dataset handling for both training and benchmarking,
following the official reasoning-gym evaluation approach.
"""

import json
from fractions import Fraction
from typing import Any, Optional

import numpy as np
import reasoning_gym
from datasets import Dataset
from reasoning_gym.utils import extract_answer


def format_question(question: str) -> str:
    """Format question with universal XML-based answer instruction.

    This instruction format works for all reasoning-gym tasks (numbers, sequences, text)
    and follows the official reasoning-gym evaluation approach.

    Args:
        question: The question to format

    Returns:
        Formatted question with XML answer instructions
    """
    instruction = """Think step by step and solve the problem carefully. When you have your final answer, format it as: <answer>your answer here</answer>

"""
    return instruction + question


def extract_answer_from_response(response: str) -> Optional[str]:
    """Extract answer from model response using reasoning-gym's standard extraction.

    This function uses reasoning-gym's built-in XML tag extraction which handles
    all answer types (numbers, sequences, text) uniformly.

    Args:
        response: The model's response text

    Returns:
        Extracted answer string, or None if no answer found
    """
    return extract_answer(response, tag_name="answer", strip=True)


def score_answer(
    dataset: reasoning_gym.dataset.ProceduralDataset,
    model_answer: Optional[str],
    entry: dict[str, Any]
) -> float:
    """Score a model's answer using the dataset's built-in scoring.

    Args:
        dataset: The reasoning-gym dataset instance
        model_answer: The model's extracted answer
        entry: The dataset entry being evaluated

    Returns:
        Score between 0.0 and 1.0
    """
    try:
        return dataset.score_answer(answer=model_answer, entry=entry)
    except Exception:
        # If scoring fails, return 0.0
        return 0.0


def convert_to_serializable(obj: Any) -> Any:
    """Convert non-serializable objects to serializable types.

    This is needed because PyArrow (used by datasets library) cannot handle
    certain Python types like Fraction objects.

    Args:
        obj: Object to convert

    Returns:
        Serializable version of the object
    """
    if isinstance(obj, Fraction):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj


def create_dataset(
    task_name: str,
    num_examples: int,
    seed: int = 42,
    format_questions: bool = True
) -> Dataset:
    """Create a reasoning-gym dataset with consistent formatting.

    Args:
        task_name: Name of the reasoning-gym task
        num_examples: Number of examples to generate
        seed: Random seed for reproducibility
        format_questions: Whether to add XML answer formatting instructions

    Returns:
        HuggingFace Dataset with formatted questions
    """
    # Create reasoning-gym dataset
    rg_data = reasoning_gym.create_dataset(
        task_name,
        size=num_examples,
        seed=seed
    )

    # Format questions if requested
    questions = [entry['question'] for entry in rg_data]
    if format_questions:
        questions = [format_question(q) for q in questions]

    # Convert metadata to JSON strings to avoid PyArrow serialization issues
    metadata = [json.dumps(convert_to_serializable(entry['metadata'])) for entry in rg_data]

    # Create HuggingFace dataset
    dataset = Dataset.from_dict({
        'question': questions,
        'answer': [entry['answer'] for entry in rg_data],
        'info': metadata
    })

    return dataset
