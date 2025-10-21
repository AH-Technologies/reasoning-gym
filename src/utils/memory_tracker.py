"""GPU memory tracking utilities."""
import torch
from typing import Optional


class MemoryTracker:
    """Track and log GPU memory usage throughout training."""

    def __init__(self, device: int = 0):
        self.device = device
        self.checkpoints = {}

    def log_memory(self, stage: str, verbose: bool = True) -> dict:
        """Log current GPU memory usage.

        Args:
            stage: Name of the stage (e.g., "after_model_load", "after_optimizer")
            verbose: Whether to print the information

        Returns:
            Dictionary with memory statistics in GB
        """
        if not torch.cuda.is_available():
            return {}

        torch.cuda.synchronize()

        allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**3  # GB
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3  # GB

        free = total - allocated

        stats = {
            'stage': stage,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'max_allocated_gb': max_allocated,
            'free_gb': free,
            'total_gb': total,
            'utilization_pct': (allocated / total) * 100
        }

        self.checkpoints[stage] = stats

        if verbose:
            print(f"\n{'='*60}")
            print(f"GPU Memory @ {stage}")
            print(f"{'='*60}")
            print(f"  Allocated:     {allocated:>6.2f} GB  ({stats['utilization_pct']:>5.1f}%)")
            print(f"  Reserved:      {reserved:>6.2f} GB")
            print(f"  Free:          {free:>6.2f} GB")
            print(f"  Total:         {total:>6.2f} GB")
            print(f"  Peak so far:   {max_allocated:>6.2f} GB")
            print(f"{'='*60}\n")

        return stats

    def get_delta(self, stage1: str, stage2: str) -> Optional[float]:
        """Get memory delta between two stages in GB."""
        if stage1 not in self.checkpoints or stage2 not in self.checkpoints:
            return None
        return self.checkpoints[stage2]['allocated_gb'] - self.checkpoints[stage1]['allocated_gb']

    def print_summary(self):
        """Print summary of all memory checkpoints."""
        if not self.checkpoints:
            print("No memory checkpoints recorded")
            return

        print(f"\n{'='*80}")
        print(f"GPU Memory Summary")
        print(f"{'='*80}")
        print(f"{'Stage':<30} {'Allocated (GB)':>15} {'Delta (GB)':>15} {'Utilization':>15}")
        print(f"{'-'*80}")

        stages = list(self.checkpoints.keys())
        for i, stage in enumerate(stages):
            stats = self.checkpoints[stage]
            allocated = stats['allocated_gb']
            util = stats['utilization_pct']

            # Calculate delta from previous stage
            if i > 0:
                prev_stage = stages[i-1]
                delta = self.get_delta(prev_stage, stage)
                delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
            else:
                delta_str = "-"

            print(f"{stage:<30} {allocated:>14.2f} {delta_str:>15} {util:>14.1f}%")

        print(f"{'='*80}\n")

    def reset_peak(self):
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)


def estimate_model_memory(num_parameters: int, dtype: str = "bf16") -> dict:
    """Estimate memory requirements for a model.

    Args:
        num_parameters: Total number of model parameters
        dtype: Data type (fp32, fp16, bf16)

    Returns:
        Dictionary with memory estimates in GB
    """
    bytes_per_param = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
        'int8': 1,
    }

    bytes_p = bytes_per_param.get(dtype, 2)

    # Model weights
    model_memory = (num_parameters * bytes_p) / 1024**3

    # Gradients (same size as model)
    gradient_memory = model_memory

    # Optimizer state (Adam: 2x model size for momentum + variance)
    optimizer_memory = 2 * model_memory

    # Activations (rough estimate: 10-20% of model size with gradient checkpointing)
    activation_memory_min = 0.1 * model_memory
    activation_memory_max = 0.2 * model_memory

    total_min = model_memory + gradient_memory + optimizer_memory + activation_memory_min
    total_max = model_memory + gradient_memory + optimizer_memory + activation_memory_max

    return {
        'model_gb': model_memory,
        'gradients_gb': gradient_memory,
        'optimizer_gb': optimizer_memory,
        'activations_min_gb': activation_memory_min,
        'activations_max_gb': activation_memory_max,
        'total_min_gb': total_min,
        'total_max_gb': total_max,
    }


def print_memory_estimate(num_parameters: int, dtype: str = "bf16"):
    """Print estimated memory requirements."""
    est = estimate_model_memory(num_parameters, dtype)

    print(f"\n{'='*60}")
    print(f"Estimated Memory Requirements ({dtype})")
    print(f"{'='*60}")
    print(f"  Parameters:        {num_parameters/1e9:.2f}B")
    print(f"  Model weights:     {est['model_gb']:>6.2f} GB")
    print(f"  Gradients:         {est['gradients_gb']:>6.2f} GB")
    print(f"  Optimizer state:   {est['optimizer_gb']:>6.2f} GB")
    print(f"  Activations:       {est['activations_min_gb']:>6.2f} - {est['activations_max_gb']:.2f} GB")
    print(f"  {'-'*58}")
    print(f"  Total estimate:    {est['total_min_gb']:>6.2f} - {est['total_max_gb']:.2f} GB")
    print(f"{'='*60}\n")
