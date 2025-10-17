"""Correctness-based reward functions."""

import reasoning_gym
from typing import Any, Dict
from .base_reward import BaseReward


class CorrectnessReward(BaseReward):
    """Reward based on answer correctness using Reasoning Gym verifier."""

    def __init__(self, task_name: str, weight: float = 1.0, **kwargs):
        super().__init__(weight=weight, **kwargs)
        self.task_name = task_name
        self.score_fn = reasoning_gym.get_score_answer_fn(task_name)

    def compute(
        self,
        prompt: Any,
        completion: Any,
        answer: str,
        info: Dict[str, Any]
    ) -> float:
        """Check if model's answer is correct."""
        if isinstance(completion, list) and len(completion) > 0:
            model_answer = completion[-1].get('content', '').strip()
        else:
            model_answer = str(completion).strip()

        entry = {
            'question': prompt[0]['content'] if isinstance(prompt, list) else str(prompt),
            'answer': answer,
            'metadata': info
        }

        try:
            score = self.score_fn(answer=model_answer, entry=entry)
            return float(score)
        except Exception as e:
            print(f"  Warning: Scoring error - {e}")
            return 0.0
