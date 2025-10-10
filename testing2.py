import verifiers as vf
from verifiers.envs.reasoninggym_env import ReasoningGymEnv

# 1. Create environment
env = ReasoningGymEnv(
    gym=[
        "basic_arithmetic",
        "bitwise_arithmetic",
        "decimal_arithmetic",
    ],
    num_samples=100,
    num_eval_samples=50,
    max_concurrent=100,
)

# 2. Load model
model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")

# 3. Configure training  
args = vf.grpo_defaults(run_name="my-experiment")

# 4. Train
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=env,
    args=args,
)
trainer.train()