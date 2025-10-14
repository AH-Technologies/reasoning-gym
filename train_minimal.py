"""
Minimal GRPO Training Script with Reasoning Gym + Verifiers
This is the bare minimum code to train a model using GRPO with Reasoning Gym tasks.
"""

import reasoning_gym
import verifiers as vf
from datasets import Dataset
from transformers import AutoTokenizer
import os

# ============================================
# 1. CONFIGURATION
# ============================================

# Model settings
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # Small model for quick testing
OUTPUT_DIR = "./output"

# Reasoning Gym task settings
TASK_NAME = "leg_counting"  # Simple counting task
DATASET_SIZE = 100  # Number of training examples
SEED = 42

# Training settings
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-5

# ============================================
# 2. CREATE REASONING GYM DATASET
# ============================================

print(f"Creating Reasoning Gym dataset: {TASK_NAME}")
rg_data = reasoning_gym.create_dataset(
    TASK_NAME,
    size=DATASET_SIZE,
    seed=SEED
)

# Convert to HuggingFace Dataset format
dataset_dict = {
    'question': [entry['question'] for entry in rg_data],
    'answer': [entry['answer'] for entry in rg_data],
    'metadata': [entry['metadata'] for entry in rg_data]
}

hf_dataset = Dataset.from_dict(dataset_dict)
print(f"Created dataset with {len(hf_dataset)} examples")

# Display sample
print("\nSample task:")
print(f"Question: {hf_dataset[0]['question']}")
print(f"Answer: {hf_dataset[0]['answer']}")

# ============================================
# 3. CREATE VERIFIERS ENVIRONMENT
# ============================================

print("\nSetting up Verifiers environment...")

# Define reward function using Reasoning Gym's scoring
def reasoning_gym_reward(prompt, completion, info):
    """
    Reward function that uses Reasoning Gym's built-in verifier
    """
    # Extract the answer from completion
    # Assuming completion contains the full response
    answer_text = completion[-1]['content'] if isinstance(completion, list) else completion
    
    # Get the scoring function for this task
    score_fn = reasoning_gym.get_score_answer_fn(info.get('source_dataset', TASK_NAME))
    
    # Create entry dict for scoring
    entry = {
        'answer': info.get('answer', ''),
        'metadata': info.get('metadata', {})
    }
    
    # Score the answer
    try:
        score = score_fn(answer=answer_text, entry=entry)
        return float(score)
    except Exception as e:
        print(f"Scoring error: {e}")
        return 0.0

# Create rubric with the reward function
rubric = vf.Rubric(
    funcs=[reasoning_gym_reward],
    weights=[1.0]
)

# Create the environment
env = vf.SingleTurnEnv(
    dataset=hf_dataset,
    rubric=rubric
)

print("Environment created successfully!")

# ============================================
# 4. CONFIGURE GRPO TRAINER
# ============================================

print("\nConfiguring GRPO trainer...")

# Training configuration
training_args = vf.GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    
    # GRPO specific settings
    num_generations=4,  # Number of responses per prompt
    temperature=0.7,
    max_new_tokens=512,
    
    # vLLM settings for inference
    use_vllm=True,
    vllm_device="cuda",
    vllm_gpu_memory_utilization=0.35,
    
    # Logging
    logging_steps=10,
    save_steps=100,
    report_to="none",  # Change to "wandb" if you want W&B logging
    
    # Optimization
    beta=0.0,  # KL divergence weight (0 = no KL penalty)
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
)

# ============================================
# 5. INITIALIZE AND TRAIN
# ============================================

print("\nInitializing trainer...")
print("Note: First run will download the model, which may take a few minutes.")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Create trainer
trainer = vf.GRPOTrainer(
    model=MODEL_NAME,
    args=training_args,
    tokenizer=tokenizer,
    environment=env,
)

print("\nStarting training...")
print("=" * 50)

# Train the model
trainer.train()

print("\n" + "=" * 50)
print("Training complete!")
print(f"Model saved to: {OUTPUT_DIR}")

# ============================================
# 6. TEST THE TRAINED MODEL (OPTIONAL)
# ============================================

print("\nTesting trained model on a sample task...")

# Generate a new task
test_data = reasoning_gym.create_dataset(TASK_NAME, size=1, seed=999)
test_question = test_data[0]['question']
test_answer = test_data[0]['answer']

print(f"\nTest Question: {test_question}")
print(f"Expected Answer: {test_answer}")
print("\nTo generate a response with the trained model:")
print("Load the model from the output directory and generate text.")