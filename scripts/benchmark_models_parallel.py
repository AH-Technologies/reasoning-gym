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


def format_question_simple(question: str) -> str:
    """Format question with simple instructions - no reasoning hints."""
    instruction = """Answer the following question directly. Provide your final answer as a number on the last line in this format: "Final Answer: [number]"

"""
    return instruction + question


def create_benchmark_dataset(task_name: str, num_examples: int, seed: int = 42) -> Dataset:
    """Create a fixed benchmark dataset from reasoning-gym."""
    logger.info(f"Creating benchmark dataset: {task_name} with {num_examples} examples (seed={seed})")

    rg_data = reasoning_gym.create_dataset(
        task_name,
        size=num_examples,
        seed=seed
    )

    questions = [format_question_simple(entry['question']) for entry in rg_data]

    dataset = Dataset.from_dict({
        'question': questions,
        'answer': [entry['answer'] for entry in rg_data],
        'info': [entry['metadata'] for entry in rg_data]
    })

    logger.info(f"Created dataset with {len(dataset)} examples")
    return dataset


def extract_answer(text: str) -> Optional[str]:
    """Extract answer from model output."""
    final_answer_match = re.search(r'Final Answer:\s*(\d+)', text, re.IGNORECASE)
    if final_answer_match:
        return final_answer_match.group(1)

    numbers = re.findall(r'\d+', text)
    if numbers:
        return numbers[-1]

    return None


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
    dataset_dict: Dict,
    gpu_id: int,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """Evaluate a single model on a specific GPU.

    This function is designed to run in a separate process.

    Args:
        model_name_or_path: HuggingFace model name or path to local checkpoint
        dataset_dict: Dataset as dictionary
        gpu_id: GPU ID to use
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Dictionary with evaluation results
    """
    # Set GPU
    device = f"cuda:{gpu_id}"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Recreate dataset from dict
    dataset = Dataset.from_dict(dataset_dict)

    # Get display name for logging
    display_name = get_model_display_name(model_name_or_path)

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {display_name} on GPU {gpu_id}")
    logger.info(f"{'='*60}")

    start_time = time.time()

    # Load model
    try:
        model, tokenizer = load_model_and_tokenizer(model_name_or_path, "cuda:0")  # Within process, use cuda:0
    except Exception as e:
        logger.error(f"Failed to load model {display_name}: {e}")
        return {
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
    correct = 0

    for idx, example in enumerate(dataset):
        try:
            generated = generate_answer(
                model, tokenizer, example['question'],
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )

            predicted_answer = extract_answer(generated)
            true_answer = str(example['answer'])

            is_correct = predicted_answer == true_answer
            if is_correct:
                correct += 1

            results.append({
                'example_id': idx,
                'question': example['question'][:100] + '...',
                'true_answer': true_answer,
                'predicted_answer': predicted_answer,
                'generated_text': generated,
                'correct': is_correct
            })

            if (idx + 1) % 10 == 0:
                current_acc = correct / (idx + 1) * 100
                logger.info(f"  [{display_name}] Progress: {idx + 1}/{len(dataset)} | Accuracy: {current_acc:.1f}%")

        except Exception as e:
            logger.error(f"Error on example {idx}: {e}")
            results.append({
                'example_id': idx,
                'error': str(e),
                'correct': False
            })

    # Calculate metrics
    accuracy = correct / len(dataset) * 100
    elapsed_time = time.time() - start_time

    logger.info(f"\nResults for {display_name}:")
    logger.info(f"  Correct: {correct}/{len(dataset)}")
    logger.info(f"  Accuracy: {accuracy:.2f}%")
    logger.info(f"  Time: {elapsed_time:.1f}s")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return {
        'model_name': model_name_or_path,
        'display_name': display_name,
        'gpu_id': gpu_id,
        'accuracy': accuracy,
        'num_correct': correct,
        'num_total': len(dataset),
        'time_seconds': elapsed_time,
        'examples': results[:5],
    }


def create_bar_chart(results: List[Dict[str, Any]], output_path: str):
    """Create a bar chart comparing model accuracies."""
    results = sorted(results, key=lambda x: x['accuracy'], reverse=True)

    # Use display_name if available, otherwise fall back to extracting from model_name
    model_names = [r.get('display_name', r['model_name'].split('/')[-1]) for r in results]
    accuracies = [r['accuracy'] for r in results]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(model_names)), accuracies, color='steelblue', alpha=0.8)

    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Model Performance Comparison on Reasoning Task', fontsize=14, fontweight='bold')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Bar chart saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark multiple LLMs in parallel')
    parser.add_argument('--models', nargs='+', required=True,
                        help='List of models to benchmark (HuggingFace names or local checkpoint paths)')
    parser.add_argument('--task', type=str, default='leg_counting',
                        help='Reasoning-gym task name')
    parser.add_argument('--num-examples', type=int, default=100,
                        help='Number of examples to test')
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
    logger.info(f"Task: {args.task}")
    logger.info(f"Number of examples: {args.num_examples}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Number of GPUs: {num_gpus}")
    logger.info(f"Models to test: {len(args.models)}")
    for model in args.models:
        logger.info(f"  - {model}")
    logger.info(f"{'='*60}\n")

    # Create benchmark dataset (same for all models)
    dataset = create_benchmark_dataset(args.task, args.num_examples, args.seed)

    # Convert to dict for multiprocessing
    dataset_dict = {
        'question': dataset['question'],
        'answer': dataset['answer'],
        'info': dataset['info']
    }

    # Evaluate models in parallel
    all_results = []

    # Assign models to GPUs (round-robin)
    model_gpu_pairs = [(model, i % num_gpus) for i, model in enumerate(args.models)]

    logger.info("Starting parallel evaluation...")
    logger.info(f"Model-GPU assignments:")
    for model, gpu in model_gpu_pairs:
        display = get_model_display_name(model)
        logger.info(f"  {display} -> GPU {gpu}")

    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for model_name, gpu_id in model_gpu_pairs:
            future = executor.submit(
                evaluate_model_on_gpu,
                model_name,
                dataset_dict,
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
                display = result.get('display_name', result['model_name'])
                logger.info(f"✓ Completed: {display} - Accuracy: {result['accuracy']:.2f}%")
            except Exception as e:
                logger.error(f"Error in parallel execution: {e}")

    # Save results
    results_file = output_dir / f'results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'task': args.task,
                'num_examples': args.num_examples,
                'seed': args.seed,
                'num_gpus': num_gpus,
                'models': args.models,
                'timestamp': timestamp,
            },
            'results': all_results
        }, f, indent=2)
    logger.info(f"\nResults saved to: {results_file}")

    # Create bar chart
    chart_file = output_dir / f'comparison_{timestamp}.png'
    create_bar_chart(all_results, str(chart_file))

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Model':<40} {'GPU':<8} {'Accuracy':<15} {'Correct/Total':<15}")
    logger.info(f"{'-'*78}")
    for result in sorted(all_results, key=lambda x: x['accuracy'], reverse=True):
        display = result.get('display_name', result['model_name'].split('/')[-1])[:38]
        if 'error' in result:
            logger.info(f"{display:<40} {result.get('gpu_id', 'N/A'):<8} {'ERROR':<15} {'N/A':<15}")
        else:
            logger.info(f"{display:<40} {result['gpu_id']:<8} {result['accuracy']:>6.2f}%{' '*8} "
                       f"{result['num_correct']}/{result['num_total']}")
    logger.info(f"{'='*60}\n")


if __name__ == '__main__':
    main()
