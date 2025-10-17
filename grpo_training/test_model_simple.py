#!/usr/bin/env python3
"""
Simple single-question tester (good for quick tests)
Usage: python test_model_simple.py "Your question here"
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_model(checkpoint_path, question):
    """Test model with a single question"""
    print(f"Loading model from: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    print(f"\nQuestion: {question}\n")

    # Prepare input
    messages = [{"role": "user", "content": question}]

    if hasattr(tokenizer, 'apply_chat_template'):
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)
    else:
        inputs = tokenizer(question, return_tensors="pt").input_ids.to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )

    # Decode
    if hasattr(tokenizer, 'apply_chat_template'):
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    else:
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Model Response:\n{response}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_model_simple.py \"Your question\" [checkpoint_path]")
        print("\nExample questions:")
        print('  python test_model_simple.py "How many legs do 3 dogs have?"')
        print('  python test_model_simple.py "What is the capital of France?"')
        sys.exit(1)

    question = sys.argv[1]
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else "./output"

    test_model(checkpoint, question)
