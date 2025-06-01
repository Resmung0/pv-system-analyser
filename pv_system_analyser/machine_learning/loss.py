import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        ce_loss_exp = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - ce_loss_exp) ** self.gamma * ce_loss
        return focal_loss.mean()
