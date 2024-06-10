from typing import Iterable, Dict, List
from typing import Optional

from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch import nn

from loss.cov_weighting import CoVWeightedLoss
from loss.multitask_distill_loss import MultitaskDistillLoss


class CoVWeightedMultitaskDistillLoss(nn.Module):
    def __init__(self,
                 model: SentenceTransformer,
                 num_losses: int,
                 mean_decay: bool = False,
                 mean_decay_param: Optional[float] = None,
                 **kwargs):
        super(CoVWeightedMultitaskDistillLoss, self).__init__()

        self.model = model
        self.loss = MultitaskDistillLoss(**kwargs, sentence_embedding_dim=model.get_sentence_embedding_dimension())
        self.cov_weighting = CoVWeightedLoss(num_losses, mean_decay, mean_decay_param)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        # Retrieve the unweighted losses.
        unweighted_losses = self.loss(embeddings, labels)
        return self.cov_weighting(unweighted_losses)
