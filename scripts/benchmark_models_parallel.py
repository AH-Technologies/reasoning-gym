"""Benchmark multiple LLMs in parallel on reasoning-gym tasks.

This script evaluates multiple models in parallel using multiple GPUs.
Supports both HuggingFace model names, local model checkpoints, and LoRA adapters.

Examples:
    # HuggingFace models
    --models "Qwen/Qwen2.5-7B-Instruct" "meta-llama/Meta-Llama-3-8B-Instruct"

    # Local trained models (regular or LoRA - automatically detected)
    --models "./experiments/leg_counting_qwen7b/checkpoint-256"

    # Mix of both
    --models "Qwen/Qwen2.5-7B-Instruct" "./experiments/leg_counting_qwen7b/checkpoint-256"
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import reasoning_gym
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset_utils import (
    format_question,
    extract_answer_from_response,
    score_answer,
    create_dataset as create_unified_dataset
)
from src.models.model_loader import load_model_and_tokenizer_from_path


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(processName)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/benchmark_parallel.log')
    ]
)
logger = logging.getLogger(__name__)


def is_local_path(model_name_or_path: str) -> bool:
    """Check if the model identifier is a local path or HuggingFace model name."""
    return os.path.exists(model_name_or_path) or model_name_or_path.startswith(('./', '/', '~'))


def get_model_display_name(model_name_or_path: str) -> str:
    """Get a clean display name for the model.

    For HuggingFace models: return as-is (e.g., "Qwen/Qwen2.5-7B-Instruct")
    For local paths: return the checkpoint name (e.g., "checkpoint-256" or "qwen7b-trained")
    """
    if is_local_path(model_name_or_path):
        path = Path(model_name_or_path)
        # If it's a checkpoint-XXX, include parent folder name for context
        if path.name.startswith('checkpoint-'):
            parent = path.parent.name
            return f"{parent}/{path.name}"
        return path.name
    return model_name_or_path


def create_benchmark_dataset(task_name: str, num_examples: int, seed: int = 42) -> tuple[Dataset, reasoning_gym.dataset.ProceduralDataset]:
    """Create a fixed benchmark dataset from reasoning-gym.

    Returns both the HuggingFace Dataset and the original reasoning-gym dataset
    for scoring purposes.
    """
    logger.info(f"Creating benchmark dataset: {task_name} with {num_examples} examples (seed={seed})")

    # Create the unified dataset
    hf_dataset = create_unified_dataset(
        task_name=task_name,
        num_examples=num_examples,
        seed=seed,
        format_questions=True
    )

    # Also create the reasoning-gym dataset for scoring
    rg_dataset = reasoning_gym.create_dataset(
        task_name,
        size=num_examples,
        seed=seed
    )

    logger.info(f"Created dataset with {len(hf_dataset)} examples")
    return hf_dataset, rg_dataset


def load_model_and_tokenizer(model_name_or_path: str, device: str = "cuda") -> tuple:
    """Load model and tokenizer from HuggingFace or local checkpoint.

    Automatically detects and loads LoRA adapters if present.

    Args:
        model_name_or_path: HuggingFace model name or path to local checkpoint
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    is_local = is_local_path(model_name_or_path)
    model_type = "local checkpoint" if is_local else "HuggingFace model"

    logger.info(f"Loading {model_type}: {model_name_or_path} on {device}")

    # Use the LoRA-aware loader
    model, tokenizer = load_model_and_tokenizer_from_path(
        model_name_or_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    model.eval()
    logger.info(f"Model loaded successfully on {device}")

    return model, tokenizer


def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
) -> str:
    """Generate answer from model."""
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": question}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
    else:
        inputs = tokenizer(question, return_tensors="pt").input_ids

    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,  # Disable cache for compatibility
        )

    generated_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return generated_text


def evaluate_model_on_gpu(
    model_name_or_path: str,
    task_name: str,
    dataset_dict: Dict,
    rg_dataset_params: Dict,
    gpu_id: int,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """Evaluate a single model on a specific task on a specific GPU.

    This function is designed to run in a separate process.

    Args:
        model_name_or_path: HuggingFace model name or path to local checkpoint
        task_name: Name of the task being evaluated
        dataset_dict: Dataset as dictionary
        rg_dataset_params: Parameters to recreate reasoning-gym dataset for scoring
        gpu_id: GPU ID to use
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Dictionary with evaluation results including task_name
    """
    # Set GPU
    device = f"cuda:{gpu_id}"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Recreate datasets
    dataset = Dataset.from_dict(dataset_dict)
    # reasoning_gym.create_dataset expects task name as first positional arg
    rg_dataset = reasoning_gym.create_dataset(
        rg_dataset_params['task_name'],
        size=rg_dataset_params['size'],
        seed=rg_dataset_params['seed']
    )
    rg_entries = list(rg_dataset)

    # Get display name for logging
    display_name = get_model_display_name(model_name_or_path)

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {display_name} on task '{task_name}' (GPU {gpu_id})")
    logger.info(f"{'='*60}")

    start_time = time.time()

    # Load model
    try:
        model, tokenizer = load_model_and_tokenizer(model_name_or_path, "cuda:0")  # Within process, use cuda:0
    except Exception as e:
        logger.error(f"Failed to load model {display_name}: {e}")
        return {
            'task_name': task_name,
            'model_name': model_name_or_path,
            'display_name': display_name,
            'gpu_id': gpu_id,
            'error': str(e),
            'accuracy': 0.0,
            'num_correct': 0,
            'num_total': len(dataset),
        }

    # Evaluate on all examples
    results = []
    total_score = 0.0

    for idx, example in enumerate(dataset):
        try:
            generated = generate_answer(
                model, tokenizer, example['question'],
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )

            # Extract answer using unified extraction
            predicted_answer = extract_answer_from_response(generated)
            true_answer = str(example['answer'])

            # Score using dataset's scoring method
            rg_entry = rg_entries[idx]
            answer_score = score_answer(rg_dataset, predicted_answer, rg_entry)

            # Consider it correct if score is 1.0 (perfect match)
            is_correct = (answer_score == 1.0)
            if is_correct:
                total_score += 1.0
            else:
                total_score += answer_score

            results.append({
                'example_id': idx,
                'question': example['question'][:100] + '...',
                'true_answer': true_answer,
                'predicted_answer': predicted_answer,
                'generated_text': generated,
                'score': answer_score,
                'correct': is_correct
            })

            if (idx + 1) % 10 == 0:
                current_acc = total_score / (idx + 1) * 100
                logger.info(f"  [{display_name}] Progress: {idx + 1}/{len(dataset)} | Accuracy: {current_acc:.1f}%")

        except Exception as e:
            logger.error(f"Error on example {idx}: {e}")
            results.append({
                'example_id': idx,
                'error': str(e),
                'score': 0.0,
                'correct': False
            })

    # Calculate metrics
    accuracy = total_score / len(dataset) * 100
    num_correct = sum(1 for r in results if r.get('correct', False))
    elapsed_time = time.time() - start_time

    logger.info(f"\nResults for {display_name} on {task_name}:")
    logger.info(f"  Correct: {num_correct}/{len(dataset)}")
    logger.info(f"  Accuracy: {accuracy:.2f}%")
    logger.info(f"  Average Score: {total_score / len(dataset):.3f}")
    logger.info(f"  Time: {elapsed_time:.1f}s")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return {
        'task_name': task_name,
        'model_name': model_name_or_path,
        'display_name': display_name,
        'gpu_id': gpu_id,
        'accuracy': accuracy,
        'num_correct': num_correct,
        'num_total': len(dataset),
        'average_score': total_score / len(dataset),
        'time_seconds': elapsed_time,
        'examples': results[:5],
    }


def create_bar_chart(results: List[Dict[str, Any]], output_path: str, title: str = None):
    """Create a bar chart comparing model accuracies."""
    results = sorted(results, key=lambda x: x['accuracy'], reverse=True)

    # Use display_name if available, otherwise fall back to extracting from model_name
    model_names = [r.get('display_name', r['model_name'].split('/')[-1]) for r in results]
    accuracies = [r['accuracy'] for r in results]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(model_names)), accuracies, color='steelblue', alpha=0.8)

    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    if title is None:
        title = 'Model Performance Comparison on Reasoning Task'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Bar chart saved to: {output_path}")


def create_summary_chart(all_task_results: Dict[str, List[Dict[str, Any]]], output_path: str):
    """Create a grouped bar chart comparing models across multiple tasks."""
    # Extract unique models and tasks
    tasks = list(all_task_results.keys())
    all_models = set()
    for task_results in all_task_results.values():
        for result in task_results:
            all_models.add(result['model_name'])
    models = sorted(all_models)

    # Get display names
    model_display_names = [get_model_display_name(m) for m in models]

    # Build accuracy matrix: models x tasks
    accuracy_matrix = []
    for model in models:
        model_accuracies = []
        for task in tasks:
            task_results = all_task_results[task]
            # Find this model's result for this task
            acc = 0.0
            for result in task_results:
                if result['model_name'] == model:
                    acc = result['accuracy']
                    break
            model_accuracies.append(acc)
        accuracy_matrix.append(model_accuracies)

    # Calculate averages for each model
    model_averages = [np.mean(accs) for accs in accuracy_matrix]

    # Create grouped bar chart
    x = np.arange(len(models))
    num_groups = len(tasks) + 1  # tasks + average
    width = 0.8 / num_groups  # Width of bars

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = plt.cm.Set3(np.linspace(0, 1, len(tasks)))

    # Plot bars for each task
    for i, task in enumerate(tasks):
        offset = width * i - (width * num_groups / 2) + width / 2
        accuracies = [accuracy_matrix[j][i] for j in range(len(models))]
        bars = ax.bar(x + offset, accuracies, width, label=task, color=colors[i], alpha=0.8)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only show label if there's data
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.0f}', ha='center', va='bottom', fontsize=8)

    # Add average bars
    offset = width * len(tasks) - (width * num_groups / 2) + width / 2
    avg_bars = ax.bar(x + offset, model_averages, width, label='Average',
                      color='darkgray', alpha=0.9, edgecolor='black', linewidth=1.5)

    # Add value labels on average bars with bold formatting
    for bar, avg in zip(avg_bars, model_averages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{avg:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Across Multiple Tasks', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_display_names, rotation=45, ha='right')
    ax.set_ylim(0, 110)  # Extra space for labels
    ax.legend(title='Tasks', loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Summary chart saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark multiple LLMs in parallel on multiple tasks')
    parser.add_argument('--models', nargs='+', required=True,
                        help='List of models to benchmark (HuggingFace names or local checkpoint paths)')
    parser.add_argument('--tasks', nargs='+', default=['leg_counting'],
                        help='Reasoning-gym task names (can specify multiple tasks)')
    parser.add_argument('--num-examples', type=int, default=100,
                        help='Number of examples to test per task')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for dataset generation')
    parser.add_argument('--output-dir', type=str, default='./benchmark_results',
                        help='Directory to save results')
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='Number of GPUs to use (default: all available)')
    parser.add_argument('--max-new-tokens', type=int, default=512,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Sampling temperature')

    args = parser.parse_args()

    # Determine number of GPUs
    num_gpus = args.num_gpus
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    logger.info(f"Using {num_gpus} GPUs for parallel evaluation")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    logger.info(f"\n{'='*60}")
    logger.info("PARALLEL BENCHMARK CONFIGURATION")
    logger.info(f"{'='*60}")
    logger.info(f"Tasks: {', '.join(args.tasks)}")
    logger.info(f"Number of examples per task: {args.num_examples}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Number of GPUs: {num_gpus}")
    logger.info(f"Models to test: {len(args.models)}")
    for model in args.models:
        logger.info(f"  - {model}")
    logger.info(f"{'='*60}\n")

    # Create datasets for all tasks upfront
    logger.info("Creating datasets for all tasks...")
    task_datasets = {}
    task_rg_params = {}
    for task in args.tasks:
        logger.info(f"  Creating dataset for {task}...")
        hf_dataset, rg_dataset = create_benchmark_dataset(task, args.num_examples, args.seed)
        task_datasets[task] = {
            'question': hf_dataset['question'],
            'answer': hf_dataset['answer'],
            'info': hf_dataset['info']
        }
        # Store parameters to recreate RG dataset in subprocess
        task_rg_params[task] = {
            'task_name': task,
            'size': args.num_examples,
            'seed': args.seed
        }
    logger.info("All datasets created!\n")

    # Create all (task, model) combinations
    all_combinations = []
    for task in args.tasks:
        for model in args.models:
            all_combinations.append((task, model))

    # Assign each combination to a GPU (round-robin)
    task_model_gpu_assignments = [
        (task, model, i % num_gpus)
        for i, (task, model) in enumerate(all_combinations)
    ]

    logger.info(f"Starting fully parallel evaluation of {len(all_combinations)} task-model combinations...")
    logger.info(f"Task-Model-GPU assignments:")
    for task, model, gpu in task_model_gpu_assignments:
        display = get_model_display_name(model)
        logger.info(f"  {task} + {display} -> GPU {gpu}")
    logger.info("")

    # Run all combinations in parallel
    all_results = []
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for task, model_name, gpu_id in task_model_gpu_assignments:
            future = executor.submit(
                evaluate_model_on_gpu,
                model_name,
                task,
                task_datasets[task],
                task_rg_params[task],
                gpu_id,
                args.max_new_tokens,
                args.temperature
            )
            futures.append(future)

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
                task = result['task_name']
                display = result.get('display_name', result['model_name'])
                logger.info(f"✓ Completed: {task} + {display} - Accuracy: {result['accuracy']:.2f}%")
            except Exception as e:
                logger.error(f"Error in parallel execution: {e}")

    # Organize results by task
    all_task_results = {}
    for task in args.tasks:
        task_results = [r for r in all_results if r['task_name'] == task]
        all_task_results[task] = task_results

        # Save per-task results
        task_results_file = output_dir / f'results_{task}_{timestamp}.json'
        with open(task_results_file, 'w') as f:
            json.dump({
                'config': {
                    'task': task,
                    'num_examples': args.num_examples,
                    'seed': args.seed,
                    'num_gpus': num_gpus,
                    'models': args.models,
                    'timestamp': timestamp,
                },
                'results': task_results
            }, f, indent=2)
        logger.info(f"\nTask results saved to: {task_results_file}")

        # Create per-task bar chart
        task_chart_file = output_dir / f'comparison_{task}_{timestamp}.png'
        create_bar_chart(task_results, str(task_chart_file), title=f'Performance on {task}')

        # Print task summary
        logger.info(f"\n{'='*60}")
        logger.info(f"TASK SUMMARY: {task}")
        logger.info(f"{'='*60}")
        logger.info(f"{'Model':<40} {'GPU':<8} {'Accuracy':<15} {'Correct/Total':<15}")
        logger.info(f"{'-'*78}")
        for result in sorted(task_results, key=lambda x: x['accuracy'], reverse=True):
            display = result.get('display_name', result['model_name'].split('/')[-1])[:38]
            if 'error' in result:
                logger.info(f"{display:<40} {result.get('gpu_id', 'N/A'):<8} {'ERROR':<15} {'N/A':<15}")
            else:
                logger.info(f"{display:<40} {result['gpu_id']:<8} {result['accuracy']:>6.2f}%{' '*8} "
                           f"{result['num_correct']}/{result['num_total']}")
        logger.info(f"{'='*60}\n")

    # Save combined results across all tasks
    combined_results_file = output_dir / f'results_all_tasks_{timestamp}.json'
    with open(combined_results_file, 'w') as f:
        json.dump({
            'config': {
                'tasks': args.tasks,
                'num_examples': args.num_examples,
                'seed': args.seed,
                'num_gpus': num_gpus,
                'models': args.models,
                'timestamp': timestamp,
            },
            'results_by_task': all_task_results
        }, f, indent=2)
    logger.info(f"\nCombined results saved to: {combined_results_file}")

    # Create summary chart across all tasks (if multiple tasks)
    if len(args.tasks) > 1:
        summary_chart_file = output_dir / f'summary_all_tasks_{timestamp}.png'
        create_summary_chart(all_task_results, str(summary_chart_file))

    # Print final summary across all tasks
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SUMMARY - ALL TASKS")
    logger.info(f"{'='*60}")
    for task, task_results in all_task_results.items():
        logger.info(f"\n{task}:")
        logger.info(f"{'Model':<40} {'Accuracy':<15}")
        logger.info(f"{'-'*55}")
        for result in sorted(task_results, key=lambda x: x['accuracy'], reverse=True):
            display = result.get('display_name', result['model_name'].split('/')[-1])[:38]
            if 'error' in result:
                logger.info(f"{display:<40} {'ERROR':<15}")
            else:
                logger.info(f"{display:<40} {result['accuracy']:>6.2f}%")
    logger.info(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
