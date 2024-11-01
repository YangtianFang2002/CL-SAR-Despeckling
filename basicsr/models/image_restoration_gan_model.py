# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib

import torch
from collections import OrderedDict
from copy import deepcopy

from basicsr.models.archs import define_network
from basicsr.models.image_restoration_model import ImageRestorationModel

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

class ImageRestorationGanModel(ImageRestorationModel):
    """Base Deblur model for single image deblur."""

    def init_training_settings(self):
        
        # define network g
        self.net_g = define_network(deepcopy(self.opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        # define network net_d
        self.net_d = define_network(deepcopy(self.opt['network_d']))
        self.net_d = self.model_to_device(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            self.load_network(self.net_d, load_path,
                              self.opt['path'].get('strict_load_d', True), param_key=self.opt['path'].get('param_key', 'params'))
        
        self.net_d_iters = self.opt.get('net_d_iters', 1)
        self.net_d_init_iters = self.opt.get('net_d_init_iters', 0)
        
        self.net_g.train()
        self.net_d.train()
        
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

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
                    
        if train_opt.get('gan_opt'):
            gan_type = train_opt['gan_opt'].pop('type')
            cri_gan_cls = getattr(loss_module, gan_type)
            self.cri_gan = cri_gan_cls(**train_opt['gan_opt']).to(self.device)

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def optimize_parameters(self, current_iter, tb_logger):
        
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False
        
        self.optimizer_g.zero_grad()
        is_update_net_g = current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters
        with torch.cuda.amp.autocast(enabled=self.is_amp):
            preds = self.net_g(self.lq)
            if not isinstance(preds, list):
                preds = [preds]

            self.output = preds[-1]

            l_g_total = 0
            loss_dict = OrderedDict()
            if is_update_net_g:
                # pixel loss
                for cri_name, cri_pix in [('l_pix', self.cri_pix), 
                                          ('l_pix_1', self.cri_pix_1), 
                                          ('l_pix_2', self.cri_pix_2)]:
                    if cri_pix:
                        l_pix = 0.
                        for pred in preds:
                            _l_pix = cri_pix(pred, self.gt)
                            l_pix += _l_pix

                        l_g_total += l_pix
                        loss_dict[cri_name] = l_pix

                # perceptual loss
                if self.cri_perceptual:
                    l_percep, l_style = self.cri_perceptual(self.output, self.gt)
                    if l_percep is not None:
                        l_g_total += l_percep
                        loss_dict['l_percep'] = l_percep
                    if l_style is not None:
                        l_g_total += l_style
                        loss_dict['l_style'] = l_style
                        
                # gan loss
                fake_g_pred = self.net_d(self.output)
                l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_gan'] = l_g_gan

        if is_update_net_g:
            use_grad_clip = self.opt['train'].get('use_grad_clip', True)
            if self.scaler is not None:
                self.scaler.scale(l_g_total).backward()
                if use_grad_clip:
                    self.scaler.unscale_(self.optimizer_g)
                    torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
                self.scaler.step(self.optimizer_g)
                self.scaler.update()
            else:
                l_g_total.backward()
                if use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
                self.optimizer_g.step()
        
        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        l_d_total = 0
        with torch.cuda.amp.autocast(enabled=self.is_amp):
            # real
            real_d_pred = self.net_d(self.gt)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['l_out_d_real'] = torch.mean(real_d_pred.detach())
            l_d_total += l_d_real

            # fake
            fake_d_pred = self.net_d(self.output.detach())
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['l_out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_total += l_d_fake
            # self.optimizer_d.step()
        
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if self.scaler is not None:
            self.scaler.scale(l_d_total).backward()
            if use_grad_clip:
                self.scaler.unscale_(self.optimizer_d)
                torch.nn.utils.clip_grad_norm_(self.net_d.parameters(), 0.01)
            self.scaler.step(self.optimizer_d)
            self.scaler.update()
        else:
            l_d_total.backward()
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.net_d.parameters(), 0.01)
            self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
