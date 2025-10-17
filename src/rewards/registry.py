"""Reward function registry.

This registry pattern enables:
1. Config-driven reward selection - specify rewards by name in YAML configs
   instead of hardcoding reward classes in trainer code
2. Easy extensibility - add new reward types (length penalties, fluency rewards,
   format rewards, etc.) without modifying core training logic
3. Multi-reward composition - combine multiple rewards with different weights
   purely through configuration

While overkill for a single reward type, this becomes essential when scaling
to multiple tasks and reward functions. It separates reward logic from training
orchestration, making experiments reproducible and shareable via config files.
"""

from typing import Dict, Type, List, Any
from .base_reward import BaseReward
from .correctness import CorrectnessReward


REWARD_REGISTRY: Dict[str, Type[BaseReward]] = {
    'correctness': CorrectnessReward,
}


def get_reward_function(name: str, **kwargs) -> BaseReward:
    """Get a reward function by name."""
    if name not in REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward function: {name}. "
            f"Available: {list(REWARD_REGISTRY.keys())}"
        )

    return REWARD_REGISTRY[name](**kwargs)


def register_reward(name: str, reward_class: Type[BaseReward]) -> None:
    """Register a new reward function."""
    REWARD_REGISTRY[name] = reward_class


def create_rewards_from_config(reward_configs: List[Dict[str, Any]], task_name: str) -> List[BaseReward]:
    """Create reward functions from configuration."""
    rewards = []

    for reward_config in reward_configs:
        name = reward_config['name']
        weight = reward_config.get('weight', 1.0)

        kwargs = {'weight': weight}
        if name == 'correctness':
            kwargs['task_name'] = task_name
            # Add debug logging parameters if specified
            kwargs['debug_logging'] = reward_config.get('debug_logging', True)
            kwargs['log_first_n'] = reward_config.get('log_first_n', 10)
            kwargs['log_every_n'] = reward_config.get('log_every_n', 50)

        reward = get_reward_function(name, **kwargs)
        rewards.append(reward)

    return rewards
