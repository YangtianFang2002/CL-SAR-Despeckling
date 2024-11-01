import argparse
import logging
from pathlib import Path
import torch
from os import path as osp
import glob

import numpy as np
import sys
import warnings
from tqdm import tqdm
sys.path.append('.')
from basicsr.data.data_util import intensity2normalizedAmp, max_denormalize, max_normalize, normalizedAmp2intensity
from basicsr.metrics.psnr_ssim import calculate_mor_vor_cvor, calculate_snr, equivalent_number_of_looks
from basicsr.utils.img_util import img2tensor, imwrite, tensor2img
from basicsr.models import create_model
from basicsr.utils import (get_env_info,
                           get_root_logger, get_time_str)
from basicsr.utils.options import _postprocess_yml_value, dict2str, parse
warnings.filterwarnings("ignore")

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    # folder containing the data for test
    parser.add_argument(
        '-i', '--input', type=str, default='val', help='Dataset Folder Path. Default: val')
    parser.add_argument(
            '--force_yml', nargs='+', default=None, help='Force to update yml files. Examples: train:ema_decay=0.999')
    # output folder
    parser.add_argument(
        '-o', '--output', type=str, default='', help='Output folder path. Default: test folder in visualization')
    parser.add_argument(
        '--tiff', action='store_true', help='input tiff files instead of npy. Default: False'
    )
    parser.add_argument(
        '--tiff_chw', action='store_true', help='is input tiff files following shape (C, H, W). Default: False, (H, W) or (H, W, C)'
    )
    parser.add_argument(
        '--jpg', action='store_true', help='output as jpg files instead of npy. Disable preview automatically. For OpenSARShip input, output gray img using VH(first channel)'
    )
    parser.add_argument(
        '--synthetic', action='store_true', help='convert intensity into VISIBLE GRAY, for evaluating SAR-USE'
    )
    parser.add_argument(
        '--lq_output', action='store_true', help='Output LQ as jpg at outputfolder-lq path. '
    )
    
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)
    
    # post process
    
    # force to update yml options
    if args.force_yml is not None:
        for entry in args.force_yml:
            # now do not support creating new keys
            keys, value = entry.split('=')
            keys, value = keys.strip(), value.strip()
            value = _postprocess_yml_value(value)
            eval_str = 'opt'
            for key in keys.split(':'):
                eval_str += f'["{key}"]'
            eval_str += '=value'
            # using exec function
            exec(eval_str)

    # disable tb_logger
    if 'use_tb_logger' in opt['logger']:
        opt['logger']['use_tb_logger'] = False

    opt['dist'] = False

    
    return opt, args


# parse options, set distributed setting, set ramdom seed
opt, args = parse_options(is_train=True)


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"predict_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    tb_logger = None  # disable tensorboard at inference phase
    return logger, tb_logger


def get_data(folder: str):
    """
    Get image data from test folder.
    """
    img_paths = list(glob.glob(osp.join(folder, '**', '*.npy' if not args.tiff else '*.tif*'), recursive=True))
    for img_path in tqdm(img_paths):
        
        try:
            if args.tiff:
                from osgeo import gdal
                tif = gdal.Open(img_path)
                ori = tif.ReadAsArray()
            else:
                ori = np.load(img_path)
            if len(ori.shape) > 2:  # 3
                if args.tiff_chw:
                    ori = ori.transpose(1, 2, 0)  # to HWC
                img_ori = ori[:, :, 0]
            else:
                img_ori = ori

            img = np.sqrt(img_ori)
            
            if args.synthetic:
                img, _, _ = max_normalize(img)  # amp to norm
                img = _view(img)  # norm to gray
            
            img = img[:, :, np.newaxis]
            img, lq_min, lq_max = max_normalize(img)
            
        except (IOError, OSError) as e:
            logger = get_root_logger()
            logger.warn(f'File client error: {e}, skip this file: {img_path}')
            continue
        img = img2tensor([img], bgr2rgb=True, float32=True)[0].unsqueeze(dim=0)
        
        data = {'lq': img, 'lq_ori': img_ori, 'lq_path': img_path, 'lq_min': lq_min, 'lq_max': lq_max, 
                'lq_ori_min':img_ori.min(), 'lq_ori_max': img_ori.max(), 'img_ori': ori}
        yield data

def _view(img):
    _sum = np.sum(img)
    _len = len(np.nonzero(img)[0])
    scale = 0.3 / (_sum / _len)
    img = img * scale
    img = np.where(img > 1, 1, img)
    img = (img * 255.)
    return img


def find_homogeneous_area(sar_image, window_size=(64, 64)):
    rows, cols = sar_image.shape
    p95 = np.percentile(sar_image, 95)
    target_img = np.where(sar_image > p95, p95, sar_image)
    min_std = float('inf')
    min_std_pos = (0, 0)

    for i in range(0, rows-window_size[0]+1, window_size[0]//2):
        for j in range(0, cols-window_size[1]+1, window_size[1]//2):
            window = target_img[i:i+window_size[0], j:j+window_size[1]]
            std_dev = np.std(window)

            if std_dev < min_std:
                min_std = std_dev
                min_std_pos = (i, j)

    return min_std, min_std_pos, window_size


def find_test_pair_homogeneous_area(sr_img_intensity, lq_img_intensity):
    min_std, min_std_pos, window_size = find_homogeneous_area(lq_img_intensity)
    sr_crop = sr_img_intensity[min_std_pos[0]:min_std_pos[0]+window_size[0], min_std_pos[1]:min_std_pos[1]+window_size[1]]
    lq_crop = lq_img_intensity[min_std_pos[0]:min_std_pos[0]+window_size[0], min_std_pos[1]:min_std_pos[1]+window_size[1]]
    return min_std, sr_crop, lq_crop


def sync_data_size(sr, lq):
    """_summary_

    Args:
        sr (_type_): HW
        lq (_type_): HW
    """
    pad_row = (lq.shape[0] - sr.shape[0])//2
    pad_col = (lq.shape[1] - sr.shape[1])//2

    if pad_row > 0:
        # sr = sr[pad_row: -pad_row, :]
        lq = lq[pad_row: -pad_row, :]
        
    if pad_col > 0:
        # sr = sr[:, pad_col: -pad_col]
        lq = lq[:, pad_col: -pad_col]
    
    return sr, lq        


def select_central(sr, lq, w=400):  # metric calculation: 128
    """_summary_

    Args:
        sr (_type_): HW
        lq (_type_): HW
    """
    central = (sr.shape[0]//2, sr.shape[1]//2)
    sr = sr[central[0]-w//2:central[0]+w//2, central[1]-w//2:central[1]+w//2]
    central = (lq.shape[0]//2, lq.shape[1]//2)
    lq = lq[central[0]-w//2:central[0]+w//2, central[1]-w//2:central[1]+w//2]
    return sr, lq


def main():
    # init torch flags
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # initialize loggers
    logger, _ = init_loggers(opt)

    # create model
    model = create_model(opt)  # weights are loaded

    logger.info(f'Start test')
    metric_results = {'SNR': [],}

    dataset = get_data(args.input)
    test_cnt = 0
    for val_data in dataset:
        img_name = osp.splitext(osp.basename(val_data['lq_path']))[0]
        
        lq_min, lq_max = val_data['lq_min'], val_data['lq_max']
        lq_ori_min, lq_ori_max = val_data['lq_ori_min'], val_data['lq_ori_max']
        
        with torch.cuda.amp.autocast(enabled=False):
            model.feed_data(val_data, is_val=False)
            
            model.test()
        
        visuals = model.get_current_visuals()
        
        # lq_img_unnormalized = tensor2img([visuals['lq']], rgb2bgr=True, out_type=np.float32)
        sr_img_unnormalized = tensor2img([torch.clamp(visuals['result'], -20., 20.)], rgb2bgr=True, out_type=np.float32)  # H,W

        lq_img_intensity = val_data['lq_ori']  # H,W

        # sr_img_unnormalized, lq_img_intensity = sync_data_size(sr_img_unnormalized, lq_img_intensity)
        # sr_img_unnormalized, lq_img_intensity = select_central(sr_img_unnormalized, lq_img_intensity)

        sr_img_intensity = normalizedAmp2intensity(sr_img_unnormalized, lq_min, lq_max)
        
        lq_img = intensity2normalizedAmp(lq_img_intensity)
        try:
            sr_img = intensity2normalizedAmp(sr_img_intensity)
        except:
            logger.error(f'skip {img_name}', exc_info=True)
            continue
        
        if args.synthetic:
            lq_img = max_denormalize(tensor2img(val_data['lq'], out_type=np.float32), lq_min, lq_max)
            lq_img_intensity = lq_img
            sr_img = max_denormalize(sr_img_unnormalized, lq_min, lq_max)
            sr_img_intensity = sr_img
            
        # calculate metrics
        
        snr = calculate_snr(sr_img_intensity, lq_img_intensity)
        
        metric_results['SNR'].append(snr)
        print(f"Tested {img_name}\tSNR: {snr:.4f}")
        
        # save npy
        if len(args.output):
            input_path = Path(args.input)
            img_path = Path(val_data['lq_path'])

            relative_path = str(img_path.relative_to(input_path))
            
            relative_folder = osp.dirname(relative_path)
            if len(relative_folder):
                os.makedirs(relative_folder, exist_ok=True)
                save_result_path = os.path.join(args.output, relative_path)
                save_img_path = os.path.join(args.output, f"{os.path.dirname(relative_path)}/{img_name}.jpg")
            else:
                save_result_path = os.path.join(args.output, relative_path)
                save_img_path = os.path.join(args.output, f"{img_name}.jpg")
        else:
            save_result_path = osp.join(
                opt['path']['visualization'], 'test',
                f'{img_name}.sr.npy')
            
            save_img_path = osp.join(
                opt['path']['visualization'], 'test',
                f'{img_name}.png')
            
        if len(val_data['img_ori'].shape) > len(sr_img_unnormalized.shape):
            sr_img_intensity = np.stack([sr_img_intensity, val_data['img_ori'][:, :, 1]], axis=2)
        
        lq_img_save = _view(lq_img)
        sr_img_save = _view(sr_img)
        if args.synthetic:
            lq_img_save = lq_img.astype(np.uint8)
            sr_img_save = sr_img.astype(np.uint8)
            
        concat = np.concatenate((lq_img_save, sr_img_save), axis=1)[:, :, np.newaxis]
        if args.jpg:
            imwrite(sr_img_save, save_img_path)
            if args.lq_output:
                lq_img_path = os.path.join(os.path.dirname(save_img_path) + '_lq', os.path.basename(save_img_path))
                imwrite(lq_img_save, lq_img_path)
                
        else:
            imwrite(concat, save_img_path)
            save_img_name, ext = osp.splitext(save_img_path)
            lq_img_path = f"{save_img_name}.lq{ext}"
            imwrite(lq_img_save[:, :, np.newaxis], lq_img_path)
            np.save(save_result_path, sr_img_intensity)
        
        test_cnt += 1
    metrics = {k: sum(v) / len(v) for k, v in metric_results.items()}
    metric_result = ""
    for k, v in metrics.items():
        metric_result += f'# {k}: {v:.4f} '
    logger.info(f'Tested {test_cnt} images from {osp.basename(args.input)}, Final Metric: {metric_result}')


if __name__ == '__main__':
    import os
    os.environ['GRPC_POLL_STRATEGY']='epoll1'
    main()
