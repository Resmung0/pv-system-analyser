import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        ce_loss_exp = torch.exp(- ce_loss)
        focal_loss = self.alpha * (1 - ce_loss_exp) ** self.gamma * ce_loss
        return focal_loss.mean()
