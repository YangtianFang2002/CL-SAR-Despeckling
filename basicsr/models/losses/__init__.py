# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from .losses import L1Loss, MSELoss, PSNRLoss, SmoothL1Loss, GradMSELoss, TVLoss, GLRTLoss
from .gan_loss import GANLoss
from .cr_loss import NT_Xent

__all__ = [
    'L1Loss', 'MSELoss', 'PSNRLoss', 'GANLoss', 'SmoothL1Loss', 'GradMSELoss', 'TVLoss', 'GLRTLoss', 
    'NT_Xent', 
]
