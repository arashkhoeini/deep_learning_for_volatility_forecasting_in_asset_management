import torch
import torch.nn as nn

class QLIKELoss(nn.Module):
    def __init__(self):
        super(QLIKELoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        """
        Computes the QLIKE loss.

        Args:
        y_pred (torch.Tensor): Predicted values (N,).
        y_true (torch.Tensor): Ground truth values (N,).

        Returns:
        torch.Tensor: Scalar loss value.
        """
        epsilon = 1e-8  # To avoid numerical instability with log
        qlike = torch.log(y_pred + epsilon) + (y_true / (y_pred + epsilon))
        return torch.mean(qlike)