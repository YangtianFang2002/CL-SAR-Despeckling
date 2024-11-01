# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib

import numpy as np
from basicsr.data.data_util import intensity2normalizedAmp, normalizedAmp2intensity, sar_val_normalize
import torch
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
import os
from tqdm import tqdm
from basicsr.data.data_util import max_denormalize, max_normalize
from basicsr.metrics.psnr_ssim import calculate_mor_vor_cvor, calculate_mse, calculate_snr, equivalent_number_of_looks, speckle_suppression_and_mean_preservation_index, speckle_suppression_index

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info

from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

class ImageRestorationModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationModel, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])

    def init_training_settings(self):
        self.net_g.train()
        
        if self.is_compile:
            logger = get_root_logger()
            logger.info("Compiling network...")
            self.net_g = torch.compile(self.net_g)
            
        
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('pixel_opt_1'):
            pixel_type = train_opt['pixel_opt_1'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix_1 = cri_pix_cls(**train_opt['pixel_opt_1']).to(
                self.device)
        else:
            self.cri_pix_1 = None
            
        if train_opt.get('pixel_opt_2'):
            pixel_type = train_opt['pixel_opt_2'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix_2 = cri_pix_cls(**train_opt['pixel_opt_2']).to(
                self.device)
        else:
            self.cri_pix_2 = None
        
        if train_opt.get('pixel_opt_3i'):
            pixel_type = train_opt['pixel_opt_3i'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix_3i = cri_pix_cls(**train_opt['pixel_opt_3i']).to(
                self.device)
        else:
            self.cri_pix_3i = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_pix_3i is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
        #         if k.startswith('module.offsets') or k.startswith('module.dcns'):
        #             optim_params_lowlr.append(v)
        #         else:
                optim_params.append(v)
            # else:
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        # ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                **train_opt['optim_g'])
            pass
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def grids(self):
        b, c, h, w = self.gt.size()
        self.original_size = (b, c, h, w)

        assert b == 1
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)


        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
        #adaptive step_i, step_j
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        import math
        step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i//scale*scale
        step_j = step_j//scale*scale

        parts = []
        idxes = []

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(self.lq[:, :, i // scale :(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.outs[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq

    def sync_data_size(self, data=None):
        pad_row = (self.gt.shape[2] - self.output.shape[2])//2
        pad_col = (self.gt.shape[3] - self.output.shape[3])//2

        if pad_row > 0:
            self.gt = self.gt[:, :, pad_row: -pad_row, :]
            self.lq = self.lq[:, :, pad_row: -pad_row, :]
            if data:
                data['lq'] = data['lq'][:, :, pad_row: -pad_row, :]
                data['gt'] = data['gt'][:, :, pad_row: -pad_row, :]
                # data['lq_ori'] = data['lq_ori'][:, pad_row: -pad_row, :]
                # data['gt_ori'] = data['gt_ori'][:, pad_row: -pad_row, :]
        if pad_col > 0:
            self.gt = self.gt[:, :, :, pad_col: -pad_col]
            self.lq = self.lq[:, :, :, pad_col: -pad_col]
            if data:
                data['lq'] = data['lq'][:, :, :, pad_col: -pad_col]
                data['gt'] = data['gt'][:, :, :, pad_col: -pad_col]
                # data['lq_ori'] = data['lq_ori'][:, :, pad_col: -pad_col]
                # data['gt_ori'] = data['gt_ori'][:, :, pad_col: -pad_col]

    def optimize_parameters(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            preds = self.net_g(self.lq)
            if not isinstance(preds, list):
                preds = [preds]

            self.output = preds[-1]
            
            # make lq, gt, pred the same resolution
            self.sync_data_size()

            l_total = 0
            loss_dict = OrderedDict()
            # pixel loss
            if self.cri_pix:
                l_pix = 0.
                for cri_name, cri_pix in [('l_pix', self.cri_pix), 
                                          ('l_pix_1', self.cri_pix_1), 
                                          ('l_pix_2', self.cri_pix_2)]:
                    if cri_pix:
                        l_pix = 0.
                        for pred in preds:
                            _l_pix = cri_pix(pred, self.gt)
                            l_pix += _l_pix

                        l_total += l_pix
                        loss_dict[cri_name] = l_pix
            
            if self.cri_pix_3i:
                l_pix = 0.
                cri_name = 'l_pix'
                
                l_pix = self.cri_pix_3i(self.lq, self.output, self.gt)

                l_total += l_pix
                loss_dict[cri_name] = l_pix

            # perceptual loss
            if self.cri_perceptual:
                l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            #
                if l_percep is not None:
                    l_total += l_percep
                    loss_dict['l_percep'] = l_percep
                if l_style is not None:
                    l_total += l_style
                    loss_dict['l_style'] = l_style

            # l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if self.scaler is not None:
            self.scaler.scale(l_total).backward()
            if use_grad_clip:
                self.scaler.unscale_(self.optimizer_g)
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
        else:
            l_total.backward()
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = len(self.lq)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                pred = self.net_g(self.lq[i:j])
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred.detach().cpu())
                i = j

            self.output = torch.cat(outs, dim=0)
        self.net_g.train()

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
            for sar_metric in ['MOR', 'CVOR', 'SNR', 'MSE', 'ssim_view', 'psnr_view']:
                self.metric_results[sar_metric] = 0

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data, is_val=True)
            
            self.test()
            self.sync_data_size(val_data)

            visuals = self.get_current_visuals()
            
            lq_min, lq_max = val_data['lq_min'].cpu().numpy(), val_data['lq_max'].cpu().numpy()
            sr_img_unnormalized = tensor2img([visuals['result']], rgb2bgr=rgb2bgr, out_type=np.float32)
            lq_img_intensity = tensor2img([self.lq], rgb2bgr=rgb2bgr, out_type=np.float32)
            gt_img_intensity = tensor2img([self.gt], rgb2bgr=rgb2bgr, out_type=np.float32)
            
            is_synthetic = False
            if 'is_synthetic' not in val_data and not dataloader.dataset.opt.get('is_view_to_gray', False):  # real SAR
                sr_img_intensity = normalizedAmp2intensity(sr_img_unnormalized, lq_min, lq_max)
                lq_img_intensity = normalizedAmp2intensity(lq_img_intensity, lq_min, lq_max)  # [B, C, H, W] -> **2 -> H,W
                gt_img_intensity = normalizedAmp2intensity(gt_img_intensity, lq_min, lq_max)  # intensity
                
                lq_img = intensity2normalizedAmp(lq_img_intensity)  # [0, 1]
                sr_img = intensity2normalizedAmp(sr_img_intensity)
                gt_img = intensity2normalizedAmp(gt_img_intensity)
            else:  # synthetic
                is_synthetic = True
                sr_img_intensity = max_denormalize(sr_img_unnormalized, lq_min, lq_max)
                lq_img_intensity = max_denormalize(lq_img_intensity, lq_min, lq_max)
                gt_img_intensity = max_denormalize(gt_img_intensity, lq_min, lq_max)
                
                lq_img, _, _ = max_normalize(lq_img_intensity)  # [0, 1]
                sr_img, _, _ = max_normalize(sr_img_intensity)
                gt_img, _, _ = max_normalize(gt_img_intensity)
                

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            # is_add_img = np.random.rand() < 0.5
            if save_img:
                save_img_path = osp.join(
                    self.opt['path']['visualization'], dataset_name,
                    f'{img_name}_sr.png')
                save_gt_img_path = osp.join(
                    self.opt['path']['visualization'], dataset_name,
                    f'{img_name}_gt.png')
                save_lq_img_path = osp.join(
                    self.opt['path']['visualization'], dataset_name,
                    f'{img_name}_lq.png')
                
                def _view(img):
                    if is_synthetic:
                        return (img * 255.).astype(np.uint8)
                    _sum = np.sum(img)
                    _len = len(np.nonzero(img)[0])
                    scale = 0.3 / (_sum / _len)
                    img = img * scale
                    img = np.where(img > 1, 1, img)
                    img = (img * 255.).astype(np.uint8)
                    return img
                
                lq_img_save = _view(lq_img)
                sr_img_save = _view(sr_img)
                gt_img_save = _view(gt_img)
                
                imwrite(sr_img_save, save_img_path)
                imwrite(gt_img_save, save_gt_img_path)
                imwrite(lq_img_save, save_lq_img_path)
                
                concat = np.concatenate((lq_img_save, sr_img_save, gt_img_save), axis=1)[:, :, np.newaxis]
                
                # add image to tb_logger
                if tb_logger is not None:
                    tb_logger.add_image(f'samples/samples_{idx}', concat, global_step=current_iter, dataformats="HWC")
                
                # calculate ssim and psnr for view
                psnr_view = psnr(sr_img_save, gt_img_save, data_range=255)
                ssim_view = ssim(sr_img_save, gt_img_save, data_range=255)
                self.metric_results['psnr_view'] += psnr_view
                self.metric_results['ssim_view'] += ssim_view
                    
            # save intensity numpy for compare
            save_npy_dir = osp.join(self.opt['path']['visualization'], dataset_name, 'eval_I')
            save_npy_path = osp.join(save_npy_dir, f'{img_name}_I.npy')
            os.makedirs(save_npy_dir, exist_ok=True)
            np.save(save_npy_path, sr_img_intensity)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                    
                    # for name, sar_metric in zip(['ssi','ssmi', 'enl'], 
                    #                             [speckle_suppression_index,
                    #                              speckle_suppression_and_mean_preservation_index, 
                    #                              equivalent_number_of_looks]
                    #                             ):
                    #     if name not in self.metric_results:
                    #         self.metric_results[name] = 0
                    #     self.metric_results[name] += sar_metric(sr_img, lq_img)  # note that this is comparaing between sr and lq
                    mor, _, cvor = calculate_mor_vor_cvor(sr_img_intensity, lq_img_intensity)
                    self.metric_results['MOR'] += mor
                    self.metric_results['CVOR'] += cvor
                    snr = calculate_snr(sr_img_intensity, lq_img_intensity)
                    self.metric_results['SNR'] += snr
                    # enl = equivalent_number_of_looks(sr_img_intensity, lq_img_intensity)
                    # self.metric_results['ENL'] += enl
                    mse = calculate_mse(sr_img, gt_img)
                    self.metric_results['MSE'] += mse
                else:
                    raise NotImplementedError
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric

    def dist_validation(self, *args, **kwargs):
        self.nondist_validation(*args, **kwargs)


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.12f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
