"""Model loading utilities."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, Tuple


def load_model_and_tokenizer(config: Dict[str, Any]) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer based on configuration."""
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

    print(f"✓ Model and tokenizer loaded")

    return model, tokenizer
