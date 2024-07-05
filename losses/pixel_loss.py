import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, raft_small, RAFT
from torchvision.models.optical_flow._utils import make_coords_grid, upsample_flow

from einops import rearrange

__all__ = [
    "PixelLoss",
    "CharbonnierLoss",
    "WeightedPixelLoss",
    "ConsistencyLoss",
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

    def forward(self, x, gt, val=False, **kwargs):
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
            assert pixel_loss.dim() == weight.dim(), f'{pixel_loss.shape} dim not equal to {weight.shape} dim'
            pixel_loss = torch.mean(pixel_loss * weight)
        else:
            pixel_loss = torch.mean(pixel_loss)
        pixel_loss *= self.loss_weight
        if val:
            return pixel_loss
        else:
            return pixel_loss * self.loss_weight


class ConsistencyLoss(nn.Module):

    def __init__(self, type='large', loss_weight=1.0):
        super(ConsistencyLoss, self).__init__()
        self.loss_weight = loss_weight
        if type == 'large':
            self.optical_flow = raft_large(pretrained=True, progress=False)
        elif type == 'small':
            self.optical_flow = raft_small(pretrained=True, progress=False)
        else:
            raise NotImplementedError(type)

        self.optical_flow.__class__ = OpticalFlow

    @staticmethod
    def unbatch_timestep(frames):
        assert frames.ndim == 5, frames.shape
        assert frames.size(2) == 3, frames.shape
        frames = frames.unbind(1)
        frames_t0 = torch.cat(frames[:-1], dim=0)
        frames_t1 = torch.cat(frames[1:], dim=0)

        return frames_t0, frames_t1

    def extract_context(self, x):
        x_t0, x_t1 = self.unbatch_timestep(x)
        return self.optical_flow(x_t0, x_t1)

    def forward(self, x, gt, num_batch=1, val=False):

        x = rearrange(x, '(b t) c h w -> b t c h w', b=num_batch)
        gt = rearrange(gt, '(b t) c h w -> b t c h w', b=num_batch)
        x_context = self.extract_context(x)
        gt_context = self.extract_context(gt)
        loss = F.mse_loss(x_context, gt_context)

        if val:
            return loss
        else:
            return loss * self.loss_weight


class OpticalFlow(RAFT):

    def forward(self, image1, image2, num_flow_updates: int = 12):
        batch_size, _, h, w = image1.shape
        if (h, w) != image2.shape[-2:]:
            raise ValueError(f"input images should have the same shape, instead got ({h}, {w}) != {image2.shape[-2:]}")

        fmaps = self.feature_encoder(torch.cat([image1, image2], dim=0))
        fmap1, fmap2 = torch.chunk(fmaps, chunks=2, dim=0)

        self.corr_block.build_pyramid(fmap1, fmap2)

        context_out = self.context_encoder(image1)

        return context_out
