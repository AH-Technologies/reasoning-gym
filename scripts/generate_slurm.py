"""Generate SLURM job script from configuration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config


def generate_slurm_script(
    config_path: str,
    account: str,
    work_dir: str = None,
    partition: str = "GPUQ",
    time: str = "04:00:00",
    nodes: int = 1,
    num_gpus: int = 4,
    gpu_type: str = "a100",
    cpus_per_task: int = 16,
    memory: str = "320G",
    output_file: str = None
):
    """Generate SLURM script from config and parameters."""

    config = load_config(config_path)

    if work_dir is None:
        work_dir = str(Path.cwd())

    model_name = config['model']['name']
    job_name = Path(config['training']['output_dir']).name

    if output_file is None:
        output_file = f"slurm_{job_name}.sh"

    training_gpus = ",".join(str(i) for i in range(1, num_gpus))
    num_processes = num_gpus - 1

    log_file = f"{job_name}_%j.log"

    template_path = Path(__file__).parent.parent / "templates" / "slurm_template.sh"
    with open(template_path, 'r') as f:
        template = f.read()

    script = template.format(
        partition=partition,
        account=account,
        time=time,
        nodes=nodes,
        cpus=cpus_per_task,
        memory=memory,
        gpu_type=gpu_type,
        num_gpus=num_gpus,
        job_name=job_name,
        log_file=log_file,
        work_dir=work_dir,
        model_name=model_name,
        training_gpus=training_gpus,
        num_processes=num_processes,
        config_path=config_path
    )

    with open(output_file, 'w') as f:
        f.write(script)

    Path(output_file).chmod(0o755)

    print(f"✓ SLURM script generated: {output_file}")
    print(f"\nTo submit:")
    print(f"  sbatch {output_file}")


def main():
    """CLI for SLURM script generator."""
    if len(sys.argv) < 3:
        print("Usage: python scripts/generate_slurm.py <config_path> <account> [options]")
        print("\nPositional arguments:")
        print("  config_path    Path to config file")
        print("  account        SLURM account name")
        print("\nOptional arguments:")
        print("  --work-dir     Working directory (default: current directory)")
        print("  --partition    SLURM partition (default: GPUQ)")
        print("  --time         Time limit (default: 04:00:00)")
        print("  --gpus         Number of GPUs (default: 4)")
        print("  --gpu-type     GPU type (default: a100)")
        print("  --cpus         CPUs per task (default: 16)")
        print("  --memory       System memory (default: 320G)")
        print("  --output       Output file (default: slurm_<job_name>.sh)")
        print("\nExample:")
        print("  python scripts/generate_slurm.py configs/experiments/leg_counting_qwen7b.yaml share-ie-idi")
        print("  python scripts/generate_slurm.py configs/base.yaml my-account --gpus 2 --cpus 8 --memory 160G")
        sys.exit(1)

    config_path = sys.argv[1]
    account = sys.argv[2]

    kwargs = {
        'config_path': config_path,
        'account': account
    }

    i = 3
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--work-dir' and i + 1 < len(sys.argv):
            kwargs['work_dir'] = sys.argv[i + 1]
            i += 2
        elif arg == '--partition' and i + 1 < len(sys.argv):
            kwargs['partition'] = sys.argv[i + 1]
            i += 2
        elif arg == '--time' and i + 1 < len(sys.argv):
            kwargs['time'] = sys.argv[i + 1]
            i += 2
        elif arg == '--gpus' and i + 1 < len(sys.argv):
            kwargs['num_gpus'] = int(sys.argv[i + 1])
            i += 2
        elif arg == '--gpu-type' and i + 1 < len(sys.argv):
            kwargs['gpu_type'] = sys.argv[i + 1]
            i += 2
        elif arg == '--cpus' and i + 1 < len(sys.argv):
            kwargs['cpus_per_task'] = int(sys.argv[i + 1])
            i += 2
        elif arg == '--memory' and i + 1 < len(sys.argv):
            kwargs['memory'] = sys.argv[i + 1]
            i += 2
        elif arg == '--output' and i + 1 < len(sys.argv):
            kwargs['output_file'] = sys.argv[i + 1]
            i += 2
        else:
            print(f"Unknown argument: {arg}")
            sys.exit(1)

    generate_slurm_script(**kwargs)


if __name__ == "__main__":
    main()
