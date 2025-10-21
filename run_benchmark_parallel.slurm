#!/bin/bash
#SBATCH --job-name=llm_benchmark_parallel
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
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
TASK="leg_counting"
NUM_EXAMPLES=100
SEED=42
OUTPUT_DIR="./benchmark_results"
MAX_NEW_TOKENS=512
TEMPERATURE=0.1
NUM_GPUS=$SLURM_GPUS_ON_NODE

# List of models to benchmark - CUSTOMIZE THIS LIST
# Supports both HuggingFace model names and local checkpoint paths
# Examples:
#   - HuggingFace models: "Qwen/Qwen2.5-7B-Instruct"
#   - Local checkpoints: "./experiments/leg_counting_qwen7b/checkpoint-256"
#   - Mix both to compare baseline vs trained models!
MODELS=(
    # "experiments/leg_counting_qwen7b/checkpoint-350"  # Uncomment to test your trained model!
    # "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    # "microsoft/Phi-4-mini-reasoning"
    # "mistralai/Mistral-7B-Instruct-v0.3"
)

echo ""
echo "=========================================="
echo "Benchmark Configuration"
echo "=========================================="
echo "Task: $TASK"
echo "Number of examples: $NUM_EXAMPLES"
echo "Seed: $SEED"
echo "Output directory: $OUTPUT_DIR"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Temperature: $TEMPERATURE"
echo "Number of GPUs for parallel execution: $NUM_GPUS"
echo "Models to benchmark (${#MODELS[@]} total):"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo "=========================================="
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Run parallel benchmark
echo "Starting parallel benchmark at $(date)"
echo "Models will be distributed across $NUM_GPUS GPUs"
echo ""

python scripts/benchmark_models_parallel.py \
    --models "${MODELS[@]}" \
    --task "$TASK" \
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
    ls -lh $OUTPUT_DIR/results_*.json 2>/dev/null | tail -1
    ls -lh $OUTPUT_DIR/comparison_*.png 2>/dev/null | tail -1
    echo ""
    echo "To view results:"
    echo "  cat $OUTPUT_DIR/results_*.json | jq '.results[] | {model: .model_name, accuracy: .accuracy}'"
else
    echo "✗ Benchmark failed with exit code $EXIT_CODE"
    echo "Check the log files for details"
fi

exit $EXIT_CODE
