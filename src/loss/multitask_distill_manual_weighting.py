from typing import Iterable, Dict, List

from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch import nn

from loss.manual_weighting import ManualWeightedLoss
from loss.multitask_distill_loss import MultitaskDistillLoss


class ManualWeightedMultitaskDistillLoss(nn.Module):
    def __init__(
            self,
            model: SentenceTransformer,
            num_losses: int = None,
            distill_objective_weight: float = None,
            domain_objective_weights: List[float] = None,
            **kwargs
    ):
        super().__init__()

        # if no weight specified, use 1
        distill_objective_weight = 1 if distill_objective_weight is None else distill_objective_weight
        if domain_objective_weights is None and num_losses is None:
            raise ValueError("Cannot automatically deduce domain objective loss weights if num losses not specified!")
        if domain_objective_weights is None:
            domain_objective_weights = [1] * (num_losses - 1)
        weights = [distill_objective_weight, *domain_objective_weights]

        self.model = model
        self.loss = MultitaskDistillLoss(**kwargs, sentence_embedding_dim=model.get_sentence_embedding_dimension())
        self.manual_weighting = ManualWeightedLoss(weights)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        losses = self.loss(embeddings, labels)
        return self.manual_weighting(losses)
