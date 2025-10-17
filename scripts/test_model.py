#!/usr/bin/env python3
"""
Test your trained model interactively
Usage: python test_model.py [checkpoint_path]
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(checkpoint_path):
    """Load model and tokenizer from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    print("This may take a minute...")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    print("✓ Model loaded successfully!\n")
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_length=512):
    """Generate a response from the model"""
    # Format as chat message
    messages = [{"role": "user", "content": prompt}]

    # Apply chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )

    # Decode
    if hasattr(tokenizer, 'apply_chat_template'):
        # Only decode the new tokens (skip the prompt)
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    else:
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        response = response[len(prompt):].strip()

    return response


def main():
    # Get checkpoint path
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = "./output"

    print("="*80)
    print("🤖 TRAINED MODEL CHATBOT TESTER")
    print("="*80)

    # Load model
    try:
        model, tokenizer = load_model(checkpoint_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nUsage: python test_model.py [checkpoint_path]")
        print("Default: ./output")
        sys.exit(1)

    print("Instructions:")
    print("  - Type your question and press Enter")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Try math problems to test leg counting training!")
    print("="*80)
    print()

    # Interactive loop
    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        # Generate response
        print("Model: ", end="", flush=True)
        try:
            response = generate_response(model, tokenizer, user_input)
            print(response)
        except Exception as e:
            print(f"❌ Error generating response: {e}")

        print()


if __name__ == "__main__":
    main()
