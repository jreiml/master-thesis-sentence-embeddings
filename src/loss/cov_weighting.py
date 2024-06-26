import logging
from typing import List
from typing import Optional

import torch
from torch import Tensor
from torch import nn

logger = logging.getLogger(__name__)


class CoVWeightedLoss(nn.Module):
    def __init__(self,
                 num_losses: int,
                 mean_decay: bool = False,
                 mean_decay_param: Optional[float] = None):
        super(CoVWeightedLoss, self).__init__()

        self.num_losses = num_losses

        # How to compute the mean statistics: Full mean or decaying mean.
        self.mean_decay = mean_decay
        self.mean_decay_param = mean_decay_param

        self.current_iter = -1
        self.alphas = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor)

        # Initialize all running statistics at 0.
        self.running_mean_L = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor)
        self.running_mean_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor)
        self.running_S_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor)
        self.running_std_l = None

    def ensure_device_placement(self, device):
        if self.alphas.device != device:
            self.alphas = self.alphas.to(device)
        if self.running_mean_L.device != device:
            self.running_mean_L = self.running_mean_L.to(device)
        if self.running_mean_l.device != device:
            self.running_mean_l = self.running_mean_l.to(device)
        if self.running_S_l.device != device:
            self.running_S_l = self.running_S_l.to(device)

    def forward(self, unweighted_losses: List[Tensor]):
        device = unweighted_losses[0].device
        self.ensure_device_placement(device)

        # Put the losses in a list. Just for computing the weights.
        L = torch.tensor(unweighted_losses, requires_grad=False).to(device)
        if L.isnan().any().item():
            logger.error("Skipped batch with nan loss for loss! "
                         "Try increasing eps if you are using MCR².")
            return torch.tensor(float("nan")).to(device)

        # If we are doing validation, we would like to return an unweighted loss be able
        # to see if we do not overfit on the training set.
        if not self.training:
            return torch.sum(L)

        # Increase the current iteration parameter.
        self.current_iter += 1
        # If we are at the zero-th iteration, set L0 to L. Else use the running mean.
        L0 = L.clone() if self.current_iter == 0 else self.running_mean_L
        # Compute the loss ratios for the current iteration given the current loss L.
        l = L / L0

        # If we are in the first iteration set alphas to all 1/32
        if self.current_iter <= 1:
            self.alphas = torch.ones((self.num_losses,), requires_grad=False).type(
                torch.FloatTensor).to(device) / self.num_losses
        # Else, apply the loss weighting method.
        else:
            ls = self.running_std_l / self.running_mean_l
            self.alphas = ls / torch.sum(ls)

        # Apply Welford's algorithm to keep running means, variances of L,l. But only do this throughout
        # training the model.
        # 1. Compute the decay parameter the computing the mean.
        if self.current_iter == 0:
            mean_param = 0.0
        elif self.current_iter > 0 and self.mean_decay:
            mean_param = self.mean_decay_param
        else:
            mean_param = (1. - 1 / (self.current_iter + 1))

        # 2. Update the statistics for l
        x_l = l.clone().detach()
        new_mean_l = mean_param * self.running_mean_l + (1 - mean_param) * x_l
        self.running_S_l += (x_l - self.running_mean_l) * (x_l - new_mean_l)
        self.running_mean_l = new_mean_l

        # The variance is S / (t - 1), but we have current_iter = t - 1
        running_variance_l = self.running_S_l / (self.current_iter + 1)
        self.running_std_l = torch.sqrt(running_variance_l + 1e-8)

        # 3. Update the statistics for L
        x_L = L.clone().detach()
        self.running_mean_L = mean_param * self.running_mean_L + (1 - mean_param) * x_L

        # Get the weighted losses and perform a standard back-pass.
        weighted_losses = [self.alphas[i] * unweighted_losses[i] for i in range(len(unweighted_losses))]
        loss = sum(weighted_losses)
        return loss
