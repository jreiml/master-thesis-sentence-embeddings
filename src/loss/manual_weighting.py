import logging
from typing import List

import torch
from torch import Tensor
from torch import nn

logger = logging.getLogger(__name__)


class ManualWeightedLoss(nn.Module):
    def __init__(self, weights: List[float]):
        super().__init__()

        if len(weights) == 0:
            raise ValueError("No weights specified for manually weighted loss!")

        # Normalize weights so they sum to one
        total_weight = sum(weights)
        self.weights = [weight/total_weight for weight in weights]

    def forward(self, unweighted_losses: List[Tensor]):
        if len(unweighted_losses) != len(self.weights):
            raise ValueError(
                f"Length mismatch for unweighted losses ({len(unweighted_losses)}) and weights {len(self.weights)}!"
            )

        device = unweighted_losses[0].device
        total_weighted_loss = torch.tensor(0., device=device)
        for unweighted_loss, weight in zip(unweighted_losses, self.weights):
            if torch.isnan(unweighted_loss):
                logger.error("Skipped batch with nan loss for loss! "
                              "Try increasing eps if you are using MCR².")
                continue

            total_weighted_loss += unweighted_loss * weight

        return total_weighted_loss

