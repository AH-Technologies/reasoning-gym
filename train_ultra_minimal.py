"""
Ultra-Minimal GRPO Training with ReasoningGymEnv
Uses Verifiers' built-in ReasoningGymEnv for maximum simplicity.
"""

import verifiers as vf
from transformers import AutoTokenizer

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TASK_NAME = "leg_counting"
OUTPUT_DIR = "./output_minimal"

print(f"Training on Reasoning Gym task: {TASK_NAME}")

# Create environment directly from Reasoning Gym task
# This handles all the dataset creation and reward functions automatically
env = vf.ReasoningGymEnv(
    task_name=TASK_NAME,
    num_examples=100,  # Number of training examples
    seed=42
)

print(f"Environment created with {len(env.dataset)} examples")

# Configure training
training_args = vf.GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_generations=4,
    temperature=0.7,
    use_vllm=True,
    logging_steps=10,
    report_to="none",
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Create and run trainer
trainer = vf.GRPOTrainer(
    model=MODEL_NAME,
    args=training_args,
    tokenizer=tokenizer,
    environment=env,
)

print("\nStarting training...")
trainer.train()

print(f"\nTraining complete! Model saved to: {OUTPUT_DIR}")