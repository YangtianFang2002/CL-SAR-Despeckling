# ------------------------------------------------------------------------
# modified from https://github.com/mayorx/matlab_ssim_pytorch_implementation/blob/main/calc_ssim.py
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import cv2
import numpy as np

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error

import torch

def calculate_psnr(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        img2 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1,2,0)
        
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    def _psnr(img1, img2):
        if test_y_channel:
            img1 = to_y_channel(img1)
            img2 = to_y_channel(img2)
        psnr = peak_signal_noise_ratio(img1, img2, data_range=1.0)  # fix to float 0-1
        return psnr
    
    if img1.ndim == 3 and img1.shape[2] == 6:
        l1, r1 = img1[:,:,:3], img1[:,:,3:]
        l2, r2 = img2[:,:,:3], img2[:,:,3:]
        return (_psnr(l1, l2) + _psnr(r1, r2))/2
    else:
        return _psnr(img1, img2)

def calculate_psnr_left(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    assert input_order == 'HWC'
    assert crop_border == 0

    img1 = img1[:,64:,:3]
    img2 = img2[:,64:,:3]
    return calculate_psnr(img1=img1, img2=img2, crop_border=0, input_order=input_order, test_y_channel=test_y_channel)

def _ssim(img1, img2, max_value):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * max_value)**2
    C2 = (0.03 * max_value)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def prepare_for_ssim(img, k):
    import torch
    with torch.no_grad():
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        conv = torch.nn.Conv2d(1, 1, k, stride=1, padding=k//2, padding_mode='reflect')
        conv.weight.requires_grad = False
        conv.weight[:, :, :, :] = 1. / (k * k)

        img = conv(img)

        img = img.squeeze(0).squeeze(0)
        img = img[0::k, 0::k]
    return img.detach().cpu().numpy()

def prepare_for_ssim_rgb(img, k):
    import torch
    with torch.no_grad():
        img = torch.from_numpy(img).float() #HxWx3

        conv = torch.nn.Conv2d(1, 1, k, stride=1, padding=k // 2, padding_mode='reflect')
        conv.weight.requires_grad = False
        conv.weight[:, :, :, :] = 1. / (k * k)

        new_img = []

        for i in range(3):
            new_img.append(conv(img[:, :, i].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)[0::k, 0::k])

    return torch.stack(new_img, dim=2).detach().cpu().numpy()

def _3d_gaussian_calculator(img, conv3d):
    out = conv3d(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    return out

def _generate_3d_gaussian_kernel():
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    kernel_3 = cv2.getGaussianKernel(11, 1.5)
    kernel = torch.tensor(np.stack([window * k for k in kernel_3], axis=0))
    conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode='replicate')
    conv3d.weight.requires_grad = False
    conv3d.weight[0, 0, :, :, :] = kernel
    return conv3d

def _ssim_3d(img1, img2, max_value):
    assert len(img1.shape) == 3 and len(img2.shape) == 3
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.

    Returns:
        float: ssim result.
    """
    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = _generate_3d_gaussian_kernel().cuda()

    img1 = torch.tensor(img1).float().cuda()
    img2 = torch.tensor(img2).float().cuda()


    mu1 = _3d_gaussian_calculator(img1, kernel)
    mu2 = _3d_gaussian_calculator(img2, kernel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _3d_gaussian_calculator(img1 ** 2, kernel) - mu1_sq
    sigma2_sq = _3d_gaussian_calculator(img2 ** 2, kernel) - mu2_sq
    sigma12 = _3d_gaussian_calculator(img1*img2, kernel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())

def _ssim_cly(img1, img2):
    assert len(img1.shape) == 2 and len(img2.shape) == 2
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    # print(kernel)
    window = np.outer(kernel, kernel.transpose())

    bt = cv2.BORDER_REPLICATE

    mu1 = cv2.filter2D(img1, -1, window, borderType=bt)
    mu2 = cv2.filter2D(img2, -1, window,borderType=bt)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window, borderType=bt) - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window, borderType=bt) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window, borderType=bt) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False,
                   ssim3d=True):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')

    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1,2,0)

    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    def _cal_ssim(img1, img2):
        if test_y_channel:
            img1 = to_y_channel(img1)
            img2 = to_y_channel(img2)
            return _ssim_cly(img1[..., 0], img2[..., 0])

        ssims = []
        
        for i in range(img1.shape[2]):
            ssims.append(structural_similarity(img1[..., i], img2[..., i], multichannel=False, data_range=1.0))  # fix to gray
        
        # # ssims_before = []

        # # skimage_before = skimage.metrics.structural_similarity(img1, img2, data_range=255., multichannel=True)
        # # print('.._skimage',
        # #       skimage.metrics.structural_similarity(img1, img2, data_range=255., multichannel=True))
        # max_value = 1 if img1.max() <= 1 else 255
        # with torch.no_grad():
        #     final_ssim = _ssim_3d(img1, img2, max_value) if ssim3d else _ssim(img1, img2, max_value)
        #     ssims.append(final_ssim)

        # # for i in range(img1.shape[2]):
        # #     ssims_before.append(_ssim(img1, img2))

        # # print('..ssim mean , new {:.4f}  and before {:.4f} .... skimage before {:.4f}'.format(np.array(ssims).mean(), np.array(ssims_before).mean(), skimage_before))
        #     # ssims.append(skimage.metrics.structural_similarity(img1[..., i], img2[..., i], multichannel=False))

        return np.array(ssims).mean()

    if img1.ndim == 3 and img1.shape[2] == 6:
        l1, r1 = img1[:,:,:3], img1[:,:,3:]
        l2, r2 = img2[:,:,:3], img2[:,:,3:]
        return (_cal_ssim(l1, l2) + _cal_ssim(r1, r2))/2
    else:
        return _cal_ssim(img1, img2)

def calculate_ssim_left(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False,
                   ssim3d=True):
    assert input_order == 'HWC'
    assert crop_border == 0

    img1 = img1[:,64:,:3]
    img2 = img2[:,64:,:3]
    return calculate_ssim(img1=img1, img2=img2, crop_border=0, input_order=input_order, test_y_channel=test_y_channel, ssim3d=ssim3d)

def calculate_skimage_ssim(img1, img2):
    return structural_similarity(img1, img2, multichannel=True)

def calculate_skimage_ssim_left(img1, img2):
    img1 = img1[:,64:,:3]
    img2 = img2[:,64:,:3]
    return calculate_skimage_ssim(img1=img1, img2=img2)

def speckle_suppression_index(denoised_img, original_img):
    return (np.std(denoised_img)/np.mean(denoised_img)) / (np.std(original_img)/np.mean(original_img))

def speckle_suppression_and_mean_preservation_index(denoised_img, original_img):
    return (1+np.abs(np.mean(original_img) - np.mean(denoised_img))) * (np.std(denoised_img) / np.std(original_img))

def equivalent_number_of_looks(denoised_img_intensity, _):
    return np.power(np.mean(denoised_img_intensity), 2) / np.var(denoised_img_intensity)

def calculate_snr(denoised_intensity, original_intensity):
    signal_mean = np.mean(denoised_intensity) 
    noise_mean = np.mean(denoised_intensity / original_intensity) 
    snr = 10 * np.log10(signal_mean / noise_mean) 
    return snr

def calculate_mor_vor_cvor(denoised_intensity, original_intensity):
    ratio_image = denoised_intensity / original_intensity
    MOR = np.mean(ratio_image)  # MOR: mean of ratio image
    VOR = np.var(ratio_image)  # VOR: var of ratio image 
    CVOR = np.sqrt(VOR)/MOR
    return MOR,VOR,CVOR

def calculate_mse(img1, img2):
    return mean_squared_error(img1, img2)
