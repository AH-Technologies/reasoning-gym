"""Training orchestration module."""
import traceback
import verifiers as vf
from typing import Dict, Any
from datasets import Dataset


def create_rubric_from_rewards(rewards: list) -> vf.Rubric:
    """Create a verifiers Rubric from reward functions."""
    funcs = [reward for reward in rewards]
    weights = [reward.weight for reward in rewards]

    return vf.Rubric(funcs=funcs, weights=weights)


def create_environment(dataset: Dataset, rewards: list) -> vf.SingleTurnEnv:
    """Create training environment."""
    print(f"\n[Environment] Creating training environment...")

    rubric = create_rubric_from_rewards(rewards)
    env = vf.SingleTurnEnv(dataset=dataset, rubric=rubric)

    print(f"Environment created")
    return env


def create_training_config(config: Dict[str, Any]) -> vf.GRPOConfig:
    """Create GRPO training configuration."""
    train_config = config['training']

    print(f"\n[Training Config] Setting up training configuration...")

    grpo_config = vf.GRPOConfig(
        output_dir=train_config['output_dir'],
        num_train_epochs=train_config['num_train_epochs'],
        per_device_train_batch_size=train_config['per_device_train_batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        learning_rate=train_config['learning_rate'],
        lr_scheduler_type=train_config.get('lr_scheduler_type', 'constant_with_warmup'),
        warmup_steps=train_config.get('warmup_steps', 100),
        num_generations=train_config['num_generations'],
        temperature=train_config.get('temperature', 0.7),
        max_grad_norm=train_config.get('max_grad_norm', 0.2),
        logging_steps=train_config.get('logging_steps', 1),
        save_steps=train_config.get('save_steps', 50),
        report_to=config.get('logging', {}).get('report_to', 'wandb'),
        bf16=train_config.get('bf16', True),
        gradient_checkpointing=train_config.get('gradient_checkpointing', False),
        optim=train_config.get('optim', 'adamw_torch'),
        # Reference model / KL penalty settings
        beta=train_config.get('beta', 0.001),
        sync_ref_model=train_config.get('sync_ref_model', True),
        ref_model_sync_steps=train_config.get('ref_model_sync_steps', 100),
        ref_model_mixup_alpha=train_config.get('ref_model_mixup_alpha', 0.5),
    )

    print(f"  Training config created")
    print(f"  Output: {train_config['output_dir']}")
    print(f"  Epochs: {train_config['num_train_epochs']}")
    print(f"  Batch size: {train_config['per_device_train_batch_size']}")
    print(f"  Learning rate: {train_config['learning_rate']}")
    print(f"  Gradient checkpointing: {train_config.get('gradient_checkpointing', False)}")

    return grpo_config


def create_trainer(model, tokenizer, training_config: vf.GRPOConfig, env: vf.SingleTurnEnv, config: dict) -> vf.GRPOTrainer:
    """Create GRPO trainer."""
    print(f"\n[Trainer] Creating GRPO trainer...")

    # Check if LoRA is enabled
    peft_config = None
    if config.get('training', {}).get('use_lora', False):
        lora_r = config['training'].get('lora_r', 8)
        lora_alpha = config['training'].get('lora_alpha', 16)
        peft_config = vf.lora_defaults(r=lora_r, alpha=lora_alpha)
        print(f"  Using LoRA: r={lora_r}, alpha={lora_alpha}")

    trainer = vf.GRPOTrainer(
        model=model,
        args=training_config,
        processing_class=tokenizer,
        env=env,
        peft_config=peft_config,
    )

    print(f"Trainer created")
    return trainer


def run_training(trainer: vf.GRPOTrainer) -> None:
    """Run the training loop."""
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    try:
        trainer.train()

        print("\n" + "="*60)
        print("✓ Training complete!")
        print("="*60)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    except Exception as e:
        print(f"\n\nError during training: {e}")
        traceback.print_exc()
        raise
