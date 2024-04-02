import torch
import numpy
from toolz.curried import curry

@curry
def mse_loss(y1, y2, scale=1.):
    """standard MSE definition"""
    return ((y1 - y2) ** 2).sum() / y1.data.nelement() * scale

@curry
def mse_loss_steps(y1, y2, steps, scale=1.):
    """standard MSE definition"""
    
    loss = 0.0
    for t in range(steps):
        loss += ((y1[:,t] - y2[:,t]) ** 2).sum() / y1[:,t].data.nelement() * scale
    return loss

@curry
def max_loss(y1, y2, scale=1.):
    """standard MSE definition"""
    return (y1 - y2).abs().max() * scale

@curry
def mean_diff(y1, y2, scale=1.):
    return (y1.sum(axis=(1,2,3)) - y2.sum(axis=(1,2,3))).abs().mean() * scale

@curry
def mae_loss(y1, y2, scale=1.):
    """standard MAE definition"""
    return (torch.abs(y1 - y2)).sum() / y1.data.nelement() * scale


@curry
def ssim_loss(y1, y2, scale=11):
    from chnet.ssim import SSIM
    ssim = SSIM(window_size=scale, size_average=True)
    return 1. - 0.5 * ssim(y1, y2)


@curry
def mssim_loss(y1, y2, scale=11):
    from pytorch_ssim import SSIM
    ssim = SSIM(window_size=scale, size_average=True)
    return 1. - 0.5 * ssim(y1, y2)