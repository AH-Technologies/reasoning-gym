"""Correctness-based reward functions."""

import re
import reasoning_gym
from typing import Any, Dict
from .base_reward import BaseReward


class CorrectnessReward(BaseReward):
    """Reward based on answer correctness using Reasoning Gym verifier.

    Features:
    - Sophisticated answer extraction with regex patterns
    - Debug logging for first N examples and periodic samples
    - Tracks format compliance (whether model followed instructions)
    """

    def __init__(
        self,
        task_name: str,
        weight: float = 1.0,
        debug_logging: bool = True,
        log_first_n: int = 10,
        log_every_n: int = 50,
        **kwargs
    ):
        super().__init__(weight=weight, **kwargs)
        self.task_name = task_name
        self.score_fn = reasoning_gym.get_score_answer_fn(task_name)
        self.debug_logging = debug_logging
        self.log_first_n = log_first_n
        self.log_every_n = log_every_n
        self.call_count = 0

    def extract_answer(self, model_output: str) -> tuple[str, bool]:
        """Extract answer from model output with fallback strategies.
            Design decision we made here: even if the model deosnt answer in the correct format we still look for an answer
            and try to extract it. This is because we want to give partial credit to models that try to answer the question but fail to follow the format perfectly.

        Args:
            model_output: The raw model output text

        Returns:
            Tuple of (extracted_answer, used_correct_format)
        """
        # Strategy 1: Look for exact format "Final Answer: [number]"
        final_answer_match = re.search(
            r'Final Answer:\s*(\d+)',
            model_output,
            re.IGNORECASE
        )
        if final_answer_match:
            return final_answer_match.group(1), True

        # Strategy 2: Fallback - Look for any number pattern at end of output
        numbers = re.findall(r'\b(\d+)\b', model_output)
        if numbers:
            return numbers[-1], False  # Take the last number found

        # Strategy 3: Last resort - use full output
        return model_output, False

    def compute(
        self,
        prompt: Any,
        completion: Any,
        answer: str,
        info: Dict[str, Any]
    ) -> float:
        """Check if model's answer is correct with extraction."""
        # Extract model's raw output
        if isinstance(completion, list) and len(completion) > 0:
            model_output = completion[-1].get('content', '').strip()
        else:
            model_output = str(completion).strip()

        # Extract answer using extraction
        model_answer, used_correct_format = self.extract_answer(model_output)

        # Create entry for scoring
        entry = {
            'question': prompt[0]['content'] if isinstance(prompt, list) else str(prompt),
            'answer': answer,
            'metadata': info
        }

        # Score the answer
        try:
            score = self.score_fn(answer=model_answer, entry=entry)
            score_float = float(score)

            # Debug logging
            self.call_count += 1
            if self.debug_logging:
                should_log = (
                    (self.call_count <= self.log_first_n) or
                    (self.call_count % self.log_every_n == 0)
                )

                if should_log:
                    question = entry['question']
                    print(f"\n{'='*80}")
                    print(f"[SAMPLE {self.call_count}]")
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
