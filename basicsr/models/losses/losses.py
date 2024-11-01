# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss
from kornia.filters import sobel
from torchmetrics.image import TotalVariation

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)


@weighted_loss
def smooth_l1_loss(pred, target):
    return F.smooth_l1_loss(pred, target, reduction='none')


class SmoothL1Loss(nn.Module):
    """Smooth L1 loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(SmoothL1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * smooth_l1_loss(
            pred, target, weight, reduction=self.reduction)


class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()



# 首先，我们需要定义一个辅助函数来计算梯度图像
def calculate_gradients(image):
    # 使用kornia的Sobel算子计算梯度
    gradients = sobel(image)
    return gradients


class GradMSELoss(MSELoss):
    """Gradient MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for gradient MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(GradMSELoss, self).__init__(loss_weight, reduction)

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights.
        """
        # 计算预测图像和目标图像的梯度
        pred_grad = calculate_gradients(pred)
        target_grad = calculate_gradients(target)

        # 调用父类MSELoss的forward方法计算梯度图像的MSE
        return super(GradMSELoss, self).forward(pred_grad, target_grad, weight, **kwargs)


@weighted_loss
def calculate_tv_loss(pred, target):
	temp1 = torch.cat((pred[:, :, 1:, :], pred[:, :, -1, :].unsqueeze(2)), 2)
	temp2 = torch.cat((pred[:, :, :, 1:], pred[:, :, :, -1].unsqueeze(3)), 3)
	temp = (pred - temp1)**2 + (pred - temp2)**2
	return temp


class TVLoss(nn.Module):
    """Total Variation loss.

    Args:
        loss_weight (float): Loss weight for TV loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' |'mean' |'sum'. Default:'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(TVLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.tv = TotalVariation(reduction=reduction)
    
    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * self.tv(pred)


@weighted_loss
def glrt_loss(pred, target):
    loss = ((pred + target)/2.0).abs().log() - (pred.abs().log() + target.abs().log()) / 2 # glrt
    loss = loss.view(pred.shape[0], -1)
    return loss


class GLRTLoss(nn.Module):
    """
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(GLRTLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * glrt_loss(
            pred, target, weight, reduction=self.reduction)


@weighted_loss
def sarcnn_loss(pred, target):
    diff = target - pred  # ==log(R1/R2)/2 in https://www.math.u-bordeaux.fr/~cdeledal/files/articleTIP2009.pdf
    loss = F.softplus(2.0 * diff) / 2.0 + F.softplus(-2.0 * diff) / 2.0 - 0.693147180559945  # glrt
    # loss = loss.view(pred.shape[0], -1)

    return loss


class SARCNNLoss(nn.Module):
    """
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(SARCNNLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * sarcnn_loss(
            pred, target, weight, reduction=self.reduction)
