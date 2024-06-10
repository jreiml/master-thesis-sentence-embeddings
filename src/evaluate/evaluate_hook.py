from typing import Callable, List, Any

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator


class HookedSentenceEvaluator(SentenceEvaluator):
    def __init__(self, hooks: List[Callable[[SentenceTransformer, int, int, Any], None]],
                 sentence_evaluator: SentenceEvaluator = None):
        self.hooks = hooks
        self.sentence_evaluator = sentence_evaluator

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, **kwargs) -> float:
        result = 0.0 if self.sentence_evaluator is None else self.sentence_evaluator(model, output_path, epoch, steps)
        for hook in self.hooks:
            hook(model, epoch, steps, **kwargs)
        return result
