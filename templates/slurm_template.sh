#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --time={time}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}
#SBATCH --gres=gpu:{gpu_type}:{num_gpus}
#SBATCH --job-name="{job_name}"
#SBATCH --output={log_file}

echo "Starting job at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_JOB_NODELIST"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"

cd {work_dir}
source .venv/bin/activate

echo "Starting vLLM server on GPU 0..."
vf-vllm --model {model_name} \
    --data-parallel-size 1 \
    --enforce-eager \
    --disable-log-requests \
    --gpu-memory-utilization 0.5 \
    --dtype bfloat16 \
    > vllm_server.log 2>&1 &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

echo "Waiting for vLLM server to be ready..."
for i in {{1..300}}; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server process died"
        echo "Check vllm_server.log for details"
        exit 1
    fi

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

    if [ $((i % 10)) -eq 0 ]; then
        echo "  Still waiting... ($i/300 seconds)"
    fi

    sleep 1
done

echo "Starting training with accelerate on GPUs {training_gpus}..."
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
accelerate launch --num-processes {num_processes} --gpu-ids {training_gpus} --main_process_port 29500 scripts/train.py {config_path}

echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "Job finished at: $(date)"
