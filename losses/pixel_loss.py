import torch
import torch.nn as nn

__all__ = [
    "PixelLoss",
    "CharbonnierLoss",
    "WeightedPixelLoss",
]


class PixelLoss(nn.Module):

    def __init__(self, criterion, loss_weight=1.0, reduction='mean'):
        super(PixelLoss, self).__init__()
        self.loss_weight = loss_weight
        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss(reduction=reduction)
        elif criterion == 'mse':
            self.criterion = torch.nn.MSELoss(reduction=reduction)
        elif criterion == 'charbonnier':
            self.criterion = CharbonnierLoss()
        else:
            raise NotImplementedError(
                f'{criterion} criterion has not been supported in this version.')

    def forward(self, x, gt, val=False):
        pixel_loss = self.criterion(x, gt)
        pixel_loss *= self.loss_weight
        if val:
            return pixel_loss
        else:
            return pixel_loss * self.loss_weight


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).
    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".
    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, eps=1e-12):
        super(CharbonnierLoss, self).__init__()

        self.loss_weight = loss_weight
        self.eps = eps

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        return self.loss_weight * torch.sqrt((pred - target) ** 2 + self.eps).mean()


class WeightedPixelLoss(nn.Module):

    def __init__(self, criterion, loss_weight=1.0):
        super(WeightedPixelLoss, self).__init__()
        self.loss_weight = loss_weight
        reduction = 'none'
        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss(reduction=reduction)
        elif criterion == 'mse':
            self.criterion = torch.nn.MSELoss(reduction=reduction)
        elif criterion == 'charbonnier':
            self.criterion = CharbonnierLoss()
        else:
            raise NotImplementedError(
                f'{criterion} criterion has not been supported in this version.')

    def forward(self, x, gt, weight=None, val=False):
        pixel_loss = self.criterion(x, gt)
        if weight is not None:
            pixel_loss = torch.mean(pixel_loss * weight)
        else:
            pixel_loss = torch.mean(pixel_loss)
        pixel_loss *= self.loss_weight
        if val:
            return pixel_loss
        else:
            return pixel_loss * self.loss_weight