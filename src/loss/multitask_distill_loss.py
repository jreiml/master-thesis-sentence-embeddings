from typing import Optional, Union, List

import torch
from torch import nn, Tensor
from torch.nn import MSELoss, CrossEntropyLoss

from util.args import DomainObjectiveType
from loss.mcr2 import MaximalCodingRateReduction, MaximalCodingRateReductionLoss


class MultitaskDistillLoss(nn.Module):
    def __init__(self,
                 sentence_embedding_dim: int = None,
                 dropout_percent: float = 0.1,
                 normalize: bool = True,
                 domain_objective_types: Union[DomainObjectiveType, List[DomainObjectiveType]] = None,
                 domain_label_count: int = None,
                 mcr2_gamma: Optional[float] = 1.0,
                 mcr2_eps: Optional[float] = 0.01,
                 **kwargs):
        super().__init__()
        self.normalize = normalize

        if isinstance(domain_objective_types, DomainObjectiveType):
            domain_objective_types = [domain_objective_types]

        self.domain_objective_types = domain_objective_types
        self.dropout = nn.Dropout(dropout_percent)
        self.extra_layers = nn.ModuleList()
        self.output_labels = []

        for domain_objective_type in domain_objective_types:
            if domain_objective_type == DomainObjectiveType.REGRESSION:
                self.output_labels.append(1)
                self.extra_layers.append(nn.Linear(sentence_embedding_dim, 1))
            elif domain_objective_type == DomainObjectiveType.CLASSIFICATION:
                if domain_label_count is None:
                    raise ValueError("Need to specify label count for classification tasks!")
                self.output_labels.append(domain_label_count)
                self.extra_layers.append(nn.Linear(sentence_embedding_dim, domain_label_count))
            elif domain_objective_type == DomainObjectiveType.MCR2 or \
                    domain_objective_type == DomainObjectiveType.MCR2_DISCRIM or \
                    domain_objective_type == DomainObjectiveType.MCR2_COMPRESS:
                # Used as loss function for MCR2
                self.output_labels.append(domain_label_count)
                self.extra_layers.append(MaximalCodingRateReduction(eps=mcr2_eps, gamma=mcr2_gamma))
            else:
                raise ValueError(f"Unsupported ObjectiveType {domain_objective_type} for Bi-Encoder")

    def compute_distill_loss(self, embeddings: List[Tensor], labels: Tensor) -> Tensor:
        output_predictions = torch.cosine_similarity(embeddings[0], embeddings[1])
        loss_fct = MSELoss()
        loss = loss_fct(output_predictions, labels.view(-1))
        return loss

    def _compute_domain_loss(self, embeddings: List[Tensor], domain_labels: Tensor,
                             domain_objective_index: int):
        domain_objective_type = self.domain_objective_types[domain_objective_index]
        embeddings = torch.cat((embeddings[0], embeddings[1]), dim=0)
        domain_labels = domain_labels.T.reshape(-1)

        if domain_objective_type == DomainObjectiveType.MCR2:
            label_count = self.output_labels[domain_objective_index]
            mcr2_loss: MaximalCodingRateReductionLoss = self.extra_layers[domain_objective_index](
                embeddings, domain_labels.long(), label_count)
            return mcr2_loss.loss
        if domain_objective_type == DomainObjectiveType.MCR2_DISCRIM:
            mcr2_loss: MaximalCodingRateReductionLoss = self.extra_layers[domain_objective_index].forward_discrimn(
                embeddings)
            return mcr2_loss.loss
        if domain_objective_type == DomainObjectiveType.MCR2_COMPRESS:
            label_count = self.output_labels[domain_objective_index]
            mcr2_loss: MaximalCodingRateReductionLoss = self.extra_layers[domain_objective_index].forward_compress(
                embeddings, domain_labels.long(), label_count)
            return mcr2_loss.loss

        embeddings = self.dropout(embeddings)
        domain_logits = self.extra_layers[domain_objective_index](embeddings)

        if domain_objective_type == DomainObjectiveType.REGRESSION:
            domain_loss_fct = MSELoss()
            return domain_loss_fct(domain_logits.squeeze(), domain_labels.float().squeeze())
        elif domain_objective_type == DomainObjectiveType.CLASSIFICATION:
            domain_loss_fct = CrossEntropyLoss()
            return domain_loss_fct(domain_logits.view(-1, self.output_labels[domain_objective_index]),
                                   domain_labels.long().view(-1))
        else:
            raise ValueError("Unhandled case in forward!")

    def compute_domain_losses(self, embeddings: List[Tensor], labels: Tensor):
        if self.normalize:
            embeddings = [torch.nn.functional.normalize(embedding, p=2, dim=1) for embedding in embeddings]

        domain_losses = []
        for i in range(len(self.domain_objective_types)):
            # Only one label supported for now
            # start_label_idx = 2 * i
            # end_label_idx = 2 * i + 1
            start_label_idx = 0
            end_label_idx = 1
            domain_labels = labels[:, [start_label_idx, end_label_idx]]
            domain_loss = self._compute_domain_loss(embeddings, domain_labels, i)
            domain_losses.append(domain_loss)
        return domain_losses

    def forward(self, embeddings: List[Tensor], labels: Tensor):
        # TODO: FELIP: optional add linear layer in front of embeddings here if you want to use smaller dims
        # alternatively, use smaller sentence embeddings models (see sbert.net)

        distill_labels = labels[:, 0]
        distill_loss = self.compute_distill_loss(embeddings, distill_labels)

        has_domain_labels = labels.shape[1] > 1
        domain_losses = []
        if has_domain_labels:
            domain_labels = labels[:, range(1, 3)]
            domain_losses = self.compute_domain_losses(embeddings, domain_labels)

        # Google Colab gives a syntax error without parentheses
        return (distill_loss, *domain_losses)
