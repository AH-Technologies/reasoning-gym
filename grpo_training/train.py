"""
Simple GRPO Training Script
Run after complete_setup.sh

Make sure you're in the grpo_training directory and have activated the environment:
  cd grpo_training
  source activate.sh
  python train.py
"""

import sys
import os

# Check if we're in a virtual environment
if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print("ERROR: Virtual environment not activated!")
    print("Please run: source activate.sh")
    sys.exit(1)

import reasoning_gym
import verifiers as vf
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# ============================================
# CONFIGURATION - Edit these to change settings
# ============================================

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Small model for quick testing
TASK_NAME = "leg_counting"                  # Reasoning Gym task
NUM_EXAMPLES = 1024                          # Training dataset size
OUTPUT_DIR = "./output"                     # Where to save the model

# Training hyperparameters
NUM_EPOCHS = 1
BATCH_SIZE = 8  # 80GB A100s can handle larger batches for 7B model
GRAD_ACCUMULATION = 4
LEARNING_RATE = 1e-6  # Very conservative learning rate for stability
NUM_GENERATIONS = 8  # Must divide evenly into (BATCH_SIZE * GRAD_ACCUMULATION)
TEMPERATURE = 0.7
MAX_GRAD_NORM = 0.2  # Aggressive gradient clipping for stability
WARMUP_STEPS = 10  # Gradual warmup

print("\n" + "="*60)
print("GRPO Training with Reasoning Gym")
print("="*60)
print(f"Model: {MODEL_NAME}")
print(f"Task: {TASK_NAME}")
print(f"Examples: {NUM_EXAMPLES}")
print(f"Output: {OUTPUT_DIR}")
print("="*60)
print("\nThis will take 10-30 minutes depending on your GPU.")
print("Press Ctrl+C to stop training at any time.")
print("="*60)

# ============================================
# STEP 1: Create Dataset
# ============================================

print("\n[1/5] Creating Reasoning Gym dataset...")
rg_data = reasoning_gym.create_dataset(
    TASK_NAME,
    size=NUM_EXAMPLES,
    seed=42
)

# Add formatting instructions to prompts
def format_question_with_instructions(question):
    """Add clear instructions for how to format the answer"""
    instruction = """You are solving a math problem. Follow these steps:
1. Think through the problem step by step
2. Show your calculation
3. On the last line, write your final answer EXACTLY in this format: "Final Answer: [number]"

"""
    return instruction + question

# Convert to HuggingFace format with formatted questions
hf_dataset = Dataset.from_dict({
    'question': [format_question_with_instructions(entry['question']) for entry in rg_data],
    'answer': [entry['answer'] for entry in rg_data],
    'info': [entry['metadata'] for entry in rg_data]
})

print(f"✓ Created dataset with {len(hf_dataset)} examples")
print(f"  Sample Q: {hf_dataset[0]['question']}")
print(f"  Sample A: {hf_dataset[0]['answer']}")

# ============================================
# STEP 2: Create Reward Function
# ============================================

print("\n[2/5] Setting up reward function...")

def correctness_reward(prompt, completion, answer, info):
    """Check if model's answer is correct using Reasoning Gym verifier"""

    # Extract model's raw answer
    if isinstance(completion, list) and len(completion) > 0:
        model_output = completion[-1].get('content', '').strip()
    else:
        model_output = str(completion).strip()

    # Extract answer from the instructed format "Final Answer: [number]"
    # First try to find the exact format we asked for
    final_answer_match = re.search(r'Final Answer:\s*(\d+)', model_output, re.IGNORECASE)
    if final_answer_match:
        model_answer = final_answer_match.group(1)
    else:
        # Fallback: Look for any number pattern at end of output
        numbers = re.findall(r'\b(\d+)\b', model_output)
        if numbers:
            model_answer = numbers[-1]  # Take the last number found
        else:
            model_answer = model_output  # Last resort: use full output

    # Get scoring function
    score_fn = reasoning_gym.get_score_answer_fn(TASK_NAME)

    # Create entry for scoring
    entry = {
        'question': prompt[0]['content'] if isinstance(prompt, list) else str(prompt),
        'answer': answer,
        'metadata': info
    }

    # Score the answer
    try:
        score = score_fn(answer=model_answer, entry=entry)
        score_float = float(score)

        # Debug logging - print examples periodically
        if not hasattr(correctness_reward, 'call_count'):
            correctness_reward.call_count = 0

        correctness_reward.call_count += 1

        # Log first 10 examples, then every 50th
        should_log = (correctness_reward.call_count <= 10) or (correctness_reward.call_count % 50 == 0)

        if should_log:
            question = prompt[0]['content'] if isinstance(prompt, list) else str(prompt)
            # Check if model followed instructions
            used_correct_format = "Final Answer:" in model_output

            print(f"\n{'='*80}")
            print(f"[SAMPLE {correctness_reward.call_count}]")
            print(f"\n📝 QUESTION:\n{question[:300]}...")
            print(f"\n🤖 MODEL OUTPUT:\n{model_output}")
            print(f"\n🔍 EXTRACTED ANSWER: {model_answer}")
            print(f"   Format correct: {'✅ YES' if used_correct_format else '❌ NO (fallback used)'}")
            print(f"\n✅ CORRECT ANSWER: {answer}")
            print(f"\n⭐ SCORE: {score_float}")
            print(f"{'='*80}\n")

        return score_float
    except Exception as e:
        print(f"  Warning: Scoring error - {e}")
        return 0.0

print("✓ Reward function ready")

# ============================================
# STEP 3: Create Environment
# ============================================

print("\n[3/5] Creating training environment...")

rubric = vf.Rubric(
    funcs=[correctness_reward],
    weights=[1.0]
)

env = vf.SingleTurnEnv(
    dataset=hf_dataset,
    rubric=rubric
)

print("✓ Environment created")

# ============================================
# STEP 4: Configure Training
# ============================================

print("\n[4/5] Configuring trainer...")

training_args = vf.GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=WARMUP_STEPS,
    num_generations=NUM_GENERATIONS,
    temperature=TEMPERATURE,
    max_grad_norm=MAX_GRAD_NORM,  # Aggressive gradient clipping
    logging_steps=1,  # Log every step to monitor stability
    save_steps=50,
    report_to="wandb",  # Change to "wandb" for logging
    bf16=True,  # Use bfloat16 for better numerical stability
    gradient_checkpointing=True,  # Save memory by recomputing activations
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model - accelerate will handle device placement
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for better numerical stability
    low_cpu_mem_usage=True
)

trainer = vf.GRPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    env=env,
)

print("✓ Trainer configured")

# ============================================
# STEP 5: Train!
# ============================================

print("\n[5/5] Starting training...")
print("="*60)

try:
    trainer.train()
    
    print("\n" + "="*60)
    print("✓ Training complete!")
    print(f"✓ Model saved to: {OUTPUT_DIR}")
    print("="*60)
    
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user")
    
except Exception as e:
    print(f"\n\nError during training: {e}")
    import traceback
    traceback.print_exc()