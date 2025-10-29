#!/bin/bash
#SBATCH --job-name=llm_benchmark_parallel
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:8
#SBATCH --output=logs/benchmark_parallel_%j.log
#SBATCH --error=logs/benchmark_parallel_%j.err


# Create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "=========================================="

# Load necessary modules
module purge
module load Python/3.11.5-GCCcore-13.2.0
echo "✓ Python module loaded"

# Activate virtual environment
source activate.sh
echo "✓ Virtual environment activated"

# Verify GPU availability
echo ""
echo "=========================================="
echo "GPU Information"
echo "=========================================="
nvidia-smi --list-gpus
echo ""
nvidia-smi
echo "=========================================="

# Configuration
# You can specify multiple tasks - each model will be tested on all tasks
# All task-model combinations will run in parallel across available GPUs
# TASKS=("leg_counting")  
TASKS=("gsm_symbolic" "letter_counting" "basic_arithmetic" "simple_geometry" "mini_sudoku")
NUM_EXAMPLES=100
SEED=42
OUTPUT_DIR="./benchmark_results"
MAX_NEW_TOKENS=512
TEMPERATURE=0.1
NUM_GPUS=$SLURM_GPUS_ON_NODE

# List of models to benchmark
# Supports both HuggingFace model names and local checkpoint paths
# Examples:
#   - HuggingFace models: "Qwen/Qwen2.5-7B-Instruct"
#   - Local checkpoints: "./experiments/leg_counting_qwen7b/checkpoint-256"
#   - Mix both to compare baseline vs trained models!

# MODELS=(
#     # "experiments/leg_counting_qwen7b/checkpoint-450"  # Uncomment to test your trained model!
#     "Qwen/Qwen2.5-7B-Instruct"
#     "meta-llama/Llama-3.1-8B-Instruct"
#     "microsoft/Phi-4-mini-reasoning"
#     "mistralai/Mistral-7B-Instruct-v0.3"
# )
# MODELS=(
#     "experiments/leg_counting_qwen7b/checkpoint-450"  # Uncomment to test your trained model!
#     "Qwen/Qwen2.5-0.5B-Instruct"
#     "Qwen/Qwen2.5-1.5B-Instruct"
#     "Qwen/Qwen2.5-3B-Instruct"
#     "Qwen/Qwen2.5-7B-Instruct"
#     "Qwen/Qwen2.5-14B-Instruct"
#     "Qwen/Qwen2.5-32B-Instruct"
# )
MODELS=(
    "experiments/leg_counting_qwen7b/checkpoint-450"
    "Qwen/Qwen2.5-7B-Instruct"
)

echo ""
echo "=========================================="
echo "Benchmark Configuration"
echo "=========================================="
echo "Tasks: ${TASKS[@]}"
echo "Number of examples per task: $NUM_EXAMPLES"
echo "Seed: $SEED"
echo "Output directory: $OUTPUT_DIR"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Temperature: $TEMPERATURE"
echo "Number of GPUs for parallel execution: $NUM_GPUS"
echo "Models to benchmark (${#MODELS[@]} total):"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo ""
echo "Total task-model combinations: $((${#TASKS[@]} * ${#MODELS[@]}))"
echo "These will run in parallel across $NUM_GPUS GPUs"
echo "=========================================="
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Run parallel benchmark
echo "Starting parallel benchmark at $(date)"
echo "All task-model combinations will run simultaneously across available GPUs"
echo ""

python scripts/benchmark_models_parallel.py \
    --models "${MODELS[@]}" \
    --tasks "${TASKS[@]}" \
    --num-examples $NUM_EXAMPLES \
    --seed $SEED \
    --output-dir "$OUTPUT_DIR" \
    --num-gpus $NUM_GPUS \
    --max-new-tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Parallel benchmark completed successfully!"
    echo ""
    echo "Results saved in: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    echo "  Per-task results:"
    ls -lh $OUTPUT_DIR/results_*_20*.json 2>/dev/null | grep -v "all_tasks" | tail -3
    echo "  Per-task charts:"
    ls -lh $OUTPUT_DIR/comparison_*_20*.png 2>/dev/null | grep -v "summary" | tail -3
    if [ ${#TASKS[@]} -gt 1 ]; then
        echo "  Combined results:"
        ls -lh $OUTPUT_DIR/results_all_tasks_*.json 2>/dev/null | tail -1
        echo "  Summary chart (all tasks with averages):"
        ls -lh $OUTPUT_DIR/summary_all_tasks_*.png 2>/dev/null | tail -1
    fi
    echo ""
    echo "To view results for a specific task:"
    echo "  cat $OUTPUT_DIR/results_<task>_*.json | jq '.results[] | {model: .display_name, accuracy: .accuracy}'"
else
    echo "✗ Benchmark failed with exit code $EXIT_CODE"
    echo "Check the log files for details"
fi

exit $EXIT_CODE
