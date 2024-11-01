# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import cv2
import random
from cv2 import rotate
import numpy as np
from scipy.ndimage import gaussian_filter


def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img


def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path):
    """Paired random crop.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    lq_patch_size = int(gt_patch_size // scale)

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs


def paired_random_crop_hw(img_gts, img_lqs, gt_patch_size_h, gt_patch_size_w, scale, gt_path):
    """Paired random crop.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    lq_patch_size_h = gt_patch_size_h // scale
    lq_patch_size_w = gt_patch_size_w // scale

    # if h_gt != h_lq * scale or w_gt != w_lq * scale:
    #     raise ValueError(
    #         f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
    #         f'multiplication of LQ ({h_lq}, {w_lq}).')
    # if h_lq < lq_patch_size or w_lq < lq_patch_size:
    #     raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
    #                      f'({lq_patch_size}, {lq_patch_size}). '
    #                      f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size_h)
    left = random.randint(0, w_lq - lq_patch_size_w)

    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size_h, left:left + lq_patch_size_w, ...]
        for v in img_lqs
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size_h, left_gt:left_gt + gt_patch_size_w, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs

def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False, vflip=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    if vflip or rotation:
        vflip = random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
            if img.shape[2] == 6:
                img = img[:,:,[3,4,5,0,1,2]].copy() # swap left/right
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img


# SAR Data Augmentation

def add_noise(image, L=None):
    if L is None:
        L = int(np.random.choice([1, 2, 3, 4]))
    assert isinstance(L, int)
    img_max = np.max(image)

    img_size_numpy=image.shape
    rows=img_size_numpy[0]
    columns=img_size_numpy[1]
    s = np.zeros((1,rows, columns))
    for k in range(0,L):
        gamma = np.abs( np.random.randn(1,rows,columns) + np.random.randn(1,rows,columns)*1j )**2/2
        s = s + gamma
    s_amplitude = np.sqrt(s/L).squeeze(0)
    if len(image.shape) >= 3:
        s_amplitude = s_amplitude[:, :, np.newaxis]

    noisy_image = np.multiply(image, s_amplitude)
    # mean = 0
    # img_p5 = np.percentile(image, 50)
    # std = noise_level * img_p5
    # noise = np.random.normal(mean, std, image.shape)
    # noisy_image = image + noise
    return np.clip(noisy_image, 0, img_max)

def adjust_contrast(lq, gt):
    mean = np.mean(gt)
    img_max = np.max(gt)
    factor = np.clip(random.random() * 2, 0.5, 1.5)
    lq, gt = map(lambda img: np.clip((img - mean) * factor + mean, 0, img_max), [lq, gt])
    return lq, gt

def adjust_brightness(lq, gt):
    img_p5 = np.percentile(gt, 50)
    img_max = np.max(gt)
    factor = np.clip(random.random() * 2 - 1, -0.5, 0.5)
    lq, gt = map(lambda img: np.clip(img + factor * img_p5, 0, img_max), [lq, gt])
    
    return lq, gt

def adjust_gamma(lq, gt):
    img_max = np.max(gt)
    gamma = max(0.9, random.random())
    lq, gt = map(lambda img: np.clip(np.power(img, gamma), 0, img_max), [lq, gt])
    return lq, gt


def generate_displacement_field(image_shape, alpha, sigma):
    random_field_x = np.random.uniform(-1, 1, image_shape) * alpha
    random_field_y = np.random.uniform(-1, 1, image_shape) * alpha

    # 高斯滤波平滑位移场
    smooth_field_x = gaussian_filter(random_field_x, [sigma, sigma])
    smooth_field_y = gaussian_filter(random_field_y, [sigma, sigma])

    return smooth_field_x, smooth_field_y


from scipy.ndimage import map_coordinates

def elastic_transform(image, displacement_field, interpolation_order=1):
    dx, dy = displacement_field

    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    transformed_image = map_coordinates(image, indices, order=interpolation_order, mode='reflect').reshape(image.shape)

    return transformed_image

def adjust_elasticity(lq, gt):
    alpha = 30 + int(random.random() * 30)
    # sigma = np.random.randn(1) * 4
    displacement_field = generate_displacement_field(gt.shape, alpha, 4)

    lq = elastic_transform(lq, displacement_field)
    gt = elastic_transform(gt, displacement_field)
    return lq, gt


def sar_augment(lq, gt, speckle=False, contrast=False, brightness=False, gamma=False, elastic=False):
    speckle = speckle and random.random() < 0.5
    contrast = contrast and random.random() < 0.5
    brightness = not contrast and brightness and random.random() < 0.5
    gamma = gamma and random.random() < 0.5
    elastic = elastic and random.random() < 0.5

    if speckle:
        lq = add_noise(lq)
    
    if contrast:
        lq, gt = adjust_contrast(lq, gt)
    
    if brightness:
        lq, gt = adjust_brightness(lq, gt)
    
    if gamma:
        lq, gt = adjust_gamma(lq, gt)
    
    if elastic:
        lq, gt = adjust_elasticity(lq, gt)

    return lq, gt

def view_sar(img, ratio=0.3):
    _sum = np.sum(img)
    _len = len(np.nonzero(img)[0])
    scale = ratio / (_sum / _len)
    img = img * scale
    img = np.where(img > 1, 1, img)
    img = (img * 255.)
    return img
