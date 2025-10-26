#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi       # Replace with your account name
#SBATCH --time=02:00:00             # 4 hours for 7B model with 1024 examples
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8           # More CPUs for 4 GPUs
#SBATCH --mem=256G                   # More system memory for 4 GPUs
#SBATCH --gres=gpu:h100:4           # Request 4 GPUs (1 for vLLM, 3 for training)
#SBATCH --job-name="grpo_training"
#SBATCH --output=logs/training_%j.log    # Output log with job ID


# Print job info
echo "Starting job at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_JOB_NODELIST"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"

# Activate virtual environment
source .venv/bin/activate

# Start vLLM inference server (will use GPU 0 by default)
echo "Starting vLLM server on GPU 0..."
vf-vllm --model Qwen/Qwen2.5-7B-Instruct \
    --data-parallel-size 1 \
    --enforce-eager \
    --disable-log-requests \
    --gpu-memory-utilization 0.5 \
    --dtype bfloat16 \
    > logs/vllm_server.log 2>&1 &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Wait for vLLM server to be ready (check if port 8000 is listening)
echo "Waiting for vLLM server to be ready..."
for i in {1..300}; do
    # Check if process is still running
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server process died"
        echo "Check vllm_server.log for details"
        exit 1
    fi

    # Check if server is responding
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "vLLM server is ready! (after $i seconds)"
        break
    fi

    if [ $i -eq 300 ]; then
        echo "ERROR: vLLM server failed to start after 300 seconds"
        echo "Last 20 lines of vllm_server.log:"
        tail -n 20 vllm_server.log
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi

    # Print progress every 10 seconds
    if [ $((i % 10)) -eq 0 ]; then
        echo "  Still waiting... ($i/300 seconds)"
    fi

    sleep 1
done

# Run training with accelerate on GPUs 1,2,3 (3-way data parallel)
echo "Starting training with accelerate on GPUs 1,2,3..."
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --num-processes 3 --gpu-ids 1,2,3 --main_process_port 29500 scripts/train.py configs/experiments/leg_counting_qwen7b.yaml

# Cleanup: Stop vLLM server
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "Job finished at: $(date)"
