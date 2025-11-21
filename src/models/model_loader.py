"""Model loading utilities with LoRA support."""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, Tuple, Optional


def is_lora_checkpoint(model_path: str) -> bool:
    """Check if a checkpoint contains LoRA adapters.

    Args:
        model_path: Path to model checkpoint

    Returns:
        True if checkpoint contains LoRA adapters
    """
    path = Path(model_path)

    # Check for PEFT adapter files
    adapter_config = path / "adapter_config.json"
    adapter_model = path / "adapter_model.safetensors"
    adapter_model_bin = path / "adapter_model.bin"

    return adapter_config.exists() and (adapter_model.exists() or adapter_model_bin.exists())


def get_base_model_name(lora_checkpoint_path: str) -> str:
    """Get the base model name from a LoRA checkpoint.

    Args:
        lora_checkpoint_path: Path to LoRA checkpoint

    Returns:
        Base model name (e.g., "Qwen/Qwen2.5-7B-Instruct")
    """
    try:
        from peft import PeftConfig
        peft_config = PeftConfig.from_pretrained(lora_checkpoint_path)
        return peft_config.base_model_name_or_path
    except Exception as e:
        raise ValueError(f"Could not load base model name from LoRA checkpoint: {e}")


def load_model_and_tokenizer_from_path(
    model_name_or_path: str,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = torch.bfloat16,
    trust_remote_code: bool = True,
    low_cpu_mem_usage: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer with automatic LoRA adapter support.

    This function automatically detects if the checkpoint is a LoRA checkpoint
    and loads the base model with adapters applied.

    Args:
        model_name_or_path: HuggingFace model name or path to checkpoint
        device_map: Device map for model loading
        torch_dtype: Data type for model
        trust_remote_code: Whether to trust remote code
        low_cpu_mem_usage: Whether to use low CPU memory

    Returns:
        Tuple of (model, tokenizer)

    Examples:
        # Load regular model
        model, tokenizer = load_model_and_tokenizer_from_path("Qwen/Qwen2.5-7B-Instruct")

        # Load LoRA model (automatically detected)
        model, tokenizer = load_model_and_tokenizer_from_path("./experiments/leg_counting_qwen7b")
    """
    # Check if this is a LoRA checkpoint
    if is_lora_checkpoint(model_name_or_path):
        print(f"  LoRA checkpoint detected: {model_name_or_path}")

        # Get base model name
        base_model_name = get_base_model_name(model_name_or_path)
        print(f"  Base model: {base_model_name}")
        print(f"  Loading base model + LoRA adapters...")

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

        # Load LoRA adapters
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, model_name_or_path)

        # Load tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=trust_remote_code)

        print(f"LoRA model loaded successfully")

    else:
        # Regular model loading
        print(f"  Loading model: {model_name_or_path}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

        print(f"Model loaded successfully")

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_model_and_tokenizer(config: Dict[str, Any]) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer based on configuration (for training)."""
    model_config = config['model']

    model_name = model_config['name']
    torch_dtype = getattr(torch, model_config.get('torch_dtype', 'bfloat16'))
    low_cpu_mem_usage = model_config.get('low_cpu_mem_usage', True)

    print(f"\n[Model] Loading model and tokenizer...")
    print(f"  Model: {model_name}")
    print(f"  dtype: {torch_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage
    )

    print(f"Model and tokenizer loaded")

    return model, tokenizer
