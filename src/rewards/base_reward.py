"""Base reward function interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseReward(ABC):
    """Abstract base class for reward functions."""

    def __init__(self, weight: float = 1.0, **kwargs):
        self.weight = weight
        self.kwargs = kwargs
        # Add __name__ attribute so verifiers library recognizes this as a function
        self.__name__ = self.__class__.__name__

    @abstractmethod
    def compute(
        self,
        prompt: Any,
        completion: Any,
        answer: str,
        info: Dict[str, Any]
    ) -> float:
        """Compute reward for a completion.

        Args:
            prompt: The input prompt
            completion: The model's completion
            answer: The ground truth answer
            info: Additional metadata

        Returns:
            Reward score (will be multiplied by self.weight)
        """
        pass

    def __call__(
        self,
        prompt: Any,
        completion: Any,
        answer: str,
        info: Dict[str, Any]
    ) -> float:
        """Compute weighted reward."""
        return self.weight * self.compute(prompt, completion, answer, info)
