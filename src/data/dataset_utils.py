"""Unified dataset utilities for reasoning-gym tasks.

This module provides consistent dataset handling for both training and benchmarking,
using the 'Final Answer:' format for answer extraction.
"""

import json
import re
from fractions import Fraction
from typing import Any, Optional

import numpy as np
import reasoning_gym
from datasets import Dataset


def format_question(question: str) -> str:
    """Format question with 'Final Answer:' instruction.

    This instruction format is used consistently in both training and benchmarking
    to ensure models learn the same format they'll be evaluated on.

    Args:
        question: The question to format

    Returns:
        Formatted question with answer instructions
    """
    instruction = """You are solving a math problem. Follow these steps:
1. Think through the problem step by step
2. Show your calculation
3. On the last line, write your final answer EXACTLY in this format: "Final Answer: [number]"

"""
    return instruction + question


def extract_answer_from_response(response: str) -> Optional[str]:
    """Extract answer from model response using 'Final Answer:' format.

    Uses multiple fallback strategies to extract answers:
    1. Look for exact format "Final Answer: [number]"
    2. If not found, look for any number pattern at end of output
    3. As last resort, return the full response

    This matches the extraction logic used during training to ensure consistency
    between training and evaluation.

    Args:
        response: The model's response text

    Returns:
        Extracted answer string
    """
    # Strategy 1: Look for exact format "Final Answer: [number]"
    final_answer_match = re.search(
        r'Final Answer:\s*(\d+)',
        response,
        re.IGNORECASE
    )
    if final_answer_match:
        return final_answer_match.group(1)

    # Strategy 2: Fallback - Look for any number pattern at end of output
    numbers = re.findall(r'\b(\d+)\b', response)
    if numbers:
        return numbers[-1]  # Take the last number found

    # Strategy 3: Last resort - use full output
    return response


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
