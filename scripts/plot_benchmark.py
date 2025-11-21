"""Generate customizable bar charts from benchmark JSON files.

This script allows you to create clean, publication-ready graphs from benchmark results
with customizable titles and model name mappings.

Examples:
    # Basic usage with default settings
    python plot_benchmark.py benchmark_results/results_20251022_173101.json

    # Custom title
    python plot_benchmark.py results.json --title "Performance on Leg Counting Task"

    # Rename models using a mapping file
    python plot_benchmark.py results.json --rename-file mappings.json

    # Inline model renaming
    python plot_benchmark.py results.json \
        --rename "Qwen/Qwen2.5-7B-Instruct=Qwen 7B" \
        --rename "meta-llama/Llama-3.1-8B-Instruct=Llama 3.1"

    # Combine options
    python plot_benchmark.py results.json \
        --title "My Custom Title" \
        --rename "microsoft/Phi-4-mini-reasoning=Phi-4" \
        --output custom_graph.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import numpy as np


def load_results(json_path: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def load_rename_mapping(rename_file: str) -> Dict[str, str]:
    """Load model name mappings from JSON file.

    Expected format:
    {
        "original_name": "display_name",
        "Qwen/Qwen2.5-7B-Instruct": "Qwen 7B",
        ...
    }
    """
    with open(rename_file, 'r') as f:
        mapping = json.load(f)
    return mapping


def parse_inline_renames(rename_args: List[str]) -> Dict[str, str]:
    """Parse inline rename arguments in format 'original=new'."""
    mapping = {}
    for arg in rename_args:
        if '=' not in arg:
            print(f"Warning: Ignoring invalid rename format '{arg}'. Expected 'original=new'")
            continue
        original, new = arg.split('=', 1)
        mapping[original.strip()] = new.strip()
    return mapping


def apply_name_mapping(
    results: List[Dict[str, Any]],
    name_mapping: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Apply name mapping to results, updating display names."""
    for result in results:
        # Check both model_name and display_name for matches
        model_name = result['model_name']
        display_name = result.get('display_name', model_name)

        if model_name in name_mapping:
            result['display_name'] = name_mapping[model_name]
        elif display_name in name_mapping:
            result['display_name'] = name_mapping[display_name]

    return results


def create_bar_chart(
    results: List[Dict[str, Any]],
    output_path: str,
    title: Optional[str] = None,
    xlabel: str = "Model",
    ylabel: str = "Accuracy (%)",
    figsize: tuple = (12, 6),
    color: str = 'steelblue',
    sort_by_accuracy: bool = True,
):
    """Create a customizable bar chart comparing model accuracies.

    Args:
        results: List of result dictionaries containing model_name, display_name, and accuracy
        output_path: Path to save the output image
        title: Chart title (default: "Model Performance Comparison")
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size as (width, height) tuple
        color: Bar color
        sort_by_accuracy: Whether to sort models by accuracy (descending)
    """
    if sort_by_accuracy:
        results = sorted(results, key=lambda x: x['accuracy'], reverse=True)

    # Extract model names and accuracies
    model_names = [r.get('display_name', r['model_name'].split('/')[-1]) for r in results]
    accuracies = [r['accuracy'] for r in results]

    # Create figure
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(model_names)), accuracies, color=color, alpha=0.8)

    # Labels and title
    plt.xlabel(xlabel, fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')

    if title is None:
        title = 'Model Performance Comparison'
    plt.title(title, fontsize=14, fontweight='bold')

    # X-axis ticks
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.ylim(0, 100)

    # Grid
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Add accuracy labels on top of bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 1,
            f'{acc:.1f}%',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {output_path}")
    plt.close()


def get_model_display_name(model_name: str, name_mapping: Dict[str, str] = None) -> str:
    """Get display name for a model, applying mapping if provided."""
    if name_mapping:
        if model_name in name_mapping:
            return name_mapping[model_name]
        # Check if it's a path-like name
        model_parts = model_name.split('/')
        if len(model_parts) > 1:
            short_name = '/'.join(model_parts[-2:]) if len(model_parts) > 2 else model_parts[-1]
            if short_name in name_mapping:
                return name_mapping[short_name]

    # Default: extract last part of path or return as-is
    return model_name.split('/')[-1] if '/' in model_name else model_name


def format_task_name(task_name: str) -> str:
    """Convert task name from underscore format to title case.

    Examples:
        'gsm_symbolic' -> 'Gsm Symbolic'
        'letter_counting' -> 'Letter Counting'
    """
    return task_name.replace('_', ' ').title()


def create_multi_task_summary_chart(
    all_task_results: Dict[str, List[Dict[str, Any]]],
    output_path: str,
    name_mapping: Dict[str, str] = None,
    title: Optional[str] = None,
    xlabel: str = "Language Models",
    ylabel: str = "Accuracy Score (%)",
    figsize: tuple = (14, 7),
):
    """Create a grouped bar chart comparing models across multiple tasks with averages.

    Args:
        all_task_results: Dictionary mapping task names to lists of results
        output_path: Path to save the output image
        name_mapping: Dictionary for renaming models
        title: Chart title (default: "Model Performance Across a Variety of Reasoning Tasks")
        xlabel: X-axis label (default: "Language Models")
        ylabel: Y-axis label (default: "Accuracy Score (%)")
        figsize: Figure size as (width, height) tuple
    """
    # Extract unique models and tasks
    tasks = list(all_task_results.keys())
    all_models = set()
    for task_results in all_task_results.values():
        for result in task_results:
            all_models.add(result['model_name'])
    models = sorted(all_models)

    # Get display names
    model_display_names = [get_model_display_name(m, name_mapping) for m in models]

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

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set3(np.linspace(0, 1, len(tasks)))

    # Plot bars for each task
    for i, task in enumerate(tasks):
        offset = width * i - (width * num_groups / 2) + width / 2
        accuracies = [accuracy_matrix[j][i] for j in range(len(models))]
        # Format task name for legend (e.g., 'gsm_symbolic' -> 'Gsm Symbolic')
        formatted_task_name = format_task_name(task)
        bars = ax.bar(x + offset, accuracies, width, label=formatted_task_name, color=colors[i], alpha=0.8)

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

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')

    if title is None:
        title = 'Model Performance Across a Variety of Reasoning Tasks'
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(model_display_names, rotation=0, ha='center')
    ax.set_ylim(0, 110)  # Extra space for labels
    ax.legend(title='Tasks', loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Multi-task summary chart saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate customizable bar charts from benchmark JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'json_file',
        type=str,
        help='Path to benchmark results JSON file'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path for the chart (default: same name as input with .png extension)'
    )

    parser.add_argument(
        '--title', '-t',
        type=str,
        default=None,
        help='Chart title (default: "Model Performance Comparison")'
    )

    parser.add_argument(
        '--xlabel',
        type=str,
        default='Language Models',
        help='X-axis label (default: "Language Models")'
    )

    parser.add_argument(
        '--ylabel',
        type=str,
        default='Accuracy Score (%)',
        help='Y-axis label (default: "Accuracy Score (%%)")'
    )

    parser.add_argument(
        '--rename',
        action='append',
        default=[],
        help='Rename a model inline: --rename "original=new". Can be used multiple times.'
    )

    parser.add_argument(
        '--rename-file',
        type=str,
        default=None,
        help='JSON file with model name mappings (keys: original names, values: new names)'
    )

    parser.add_argument(
        '--color',
        type=str,
        default='steelblue',
        help='Bar color (default: steelblue)'
    )

    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=[12, 6],
        metavar=('WIDTH', 'HEIGHT'),
        help='Figure size in inches (default: 12 6)'
    )

    parser.add_argument(
        '--no-sort',
        action='store_true',
        help='Do not sort models by accuracy (keeps original order)'
    )

    args = parser.parse_args()

    # Validate input file
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: File not found: {args.json_file}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = json_path.parent / f"{json_path.stem}_chart.png"

    # Load results
    print(f"Loading results from: {args.json_file}")
    data = load_results(args.json_file)

    # Build name mapping
    name_mapping = {}

    # Load from file if provided
    if args.rename_file:
        print(f"Loading name mappings from: {args.rename_file}")
        file_mapping = load_rename_mapping(args.rename_file)
        name_mapping.update(file_mapping)

    # Add inline renames (these override file mappings)
    if args.rename:
        inline_mapping = parse_inline_renames(args.rename)
        name_mapping.update(inline_mapping)

    # Check if this is a multi-task result file
    if 'results_by_task' in data:
        # Multi-task results
        print(f"Detected multi-task results")
        all_task_results = data['results_by_task']
        tasks = list(all_task_results.keys())
        print(f"Found {len(tasks)} tasks: {', '.join(tasks)}")

        # Apply name mapping to each task's results
        if name_mapping:
            print(f"Applying {len(name_mapping)} name mapping(s)")
            for task in tasks:
                all_task_results[task] = apply_name_mapping(all_task_results[task], name_mapping)

        # Create multi-task summary chart
        print(f"Creating multi-task summary chart...")
        create_multi_task_summary_chart(
            all_task_results=all_task_results,
            output_path=str(output_path),
            name_mapping=name_mapping,
            title=args.title,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            figsize=tuple(args.figsize),
        )
    else:
        # Single-task results
        results = data.get('results', [])

        if not results:
            print("Error: No results found in JSON file")
            sys.exit(1)

        print(f"Found {len(results)} models in results")

        # Apply name mapping
        if name_mapping:
            print(f"Applying {len(name_mapping)} name mapping(s)")
            results = apply_name_mapping(results, name_mapping)

        # Create single-task chart
        print(f"Creating bar chart...")
        create_bar_chart(
            results=results,
            output_path=str(output_path),
            title=args.title,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            figsize=tuple(args.figsize),
            color=args.color,
            sort_by_accuracy=not args.no_sort,
        )

    print(f"\nDone! Chart saved to: {output_path}")


if __name__ == '__main__':
    main()


# Example command to run the script with custom options (this is the one used for the report):

# python scripts/plot_benchmark.py results.json \
#       --title "Performance on Leg Counting Task" \
#       --color coral \
#       --figsize 14 8 \
#       --xlabel "Language Models" \
#       --ylabel "Accuracy Score (%)"