from typing import NamedTuple

import torch
from torch import FloatTensor, Tensor


class MaximalCodingRateReductionLoss(NamedTuple):
    loss: FloatTensor
    discrimn_loss: Tensor
    compress_loss: Tensor


class MaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, eps=0.01, gamma=1):
        super(MaximalCodingRateReduction, self).__init__()
        self.eps = eps
        self.gamma = gamma

    def compute_discrimn_loss(self, w):
        """Discriminative Loss."""
        p, m = w.shape
        identity = torch.eye(p, device=w.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(identity + scalar * w.matmul(w.T))
        return logdet / 2.

    def compute_compress_loss(self, w, pi):
        p, m = w.shape
        k, _, _ = pi.shape
        identity = torch.eye(p, device=w.device).expand((k, p, p))
        tri_pi = pi.sum(2) + 1e-8
        scale = (p / (tri_pi * self.eps)).view(k, 1, 1)

        w = w.view((1, p, m))
        log_det = torch.logdet(identity + scale * w.mul(pi).matmul(w.transpose(1, 2)))
        compress_loss = (tri_pi.squeeze() * log_det / (2 * m)).sum()
        return compress_loss

    def get_pi(self, labels, num_labels=None):
        if num_labels is None:
            num_labels = labels.max() + 1
        pi = torch.zeros((num_labels, 1, labels.shape[0]), device=labels.device)
        for indx, label in enumerate(labels):
            pi[label, 0, indx] = 1
        return pi

    def forward_compress(self, pooled_output, labels, num_labels=None):
        pi = self.get_pi(labels, num_labels)
        w = pooled_output.T
        compress_loss = self.compute_compress_loss(w, pi)
        discrimn_loss = torch.tensor(0., device=compress_loss.device)

        return MaximalCodingRateReductionLoss(
            loss=compress_loss,
            discrimn_loss=discrimn_loss,
            compress_loss=compress_loss
        )

    def forward_discrimn(self, pooled_output):
        w = pooled_output.T
        discrimn_loss = self.compute_discrimn_loss(w)
        compress_loss = torch.tensor(0., device=discrimn_loss.device)
        loss = -discrimn_loss

        return MaximalCodingRateReductionLoss(
            loss=loss,
            discrimn_loss=discrimn_loss,
            compress_loss=compress_loss
        )

    def forward(self, pooled_output, labels=None, num_labels=None):
        if labels is None or self.gamma == 0:
            return self.forward_discrimn(pooled_output)

        pi = self.get_pi(labels, num_labels)
        w = pooled_output.T
        discrimn_loss = self.compute_discrimn_loss(w)
        compress_loss = self.compute_compress_loss(w, pi)

        loss = -discrimn_loss + self.gamma * compress_loss
        return MaximalCodingRateReductionLoss(
            loss=loss,
            discrimn_loss=discrimn_loss,
            compress_loss=compress_loss
        )
