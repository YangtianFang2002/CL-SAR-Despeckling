import importlib

import torch
from collections import OrderedDict

from basicsr.models.image_restoration_gan_model import ImageRestorationGanModel

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


class ImageRestorationClGanModel(ImageRestorationGanModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationClGanModel, self).__init__(opt)
        self.is_gan_disabled = opt.get('is_gan_disabled', False)
        self.is_cl_disabled = opt.get('is_cl_disabled', False)

        # self.net_cl_real_iters = 10000
        # self.net_cl_fake_iters = 20000
        self.net_cl_real_iters = opt['train']['net_cl_real_iters']
        self.net_cl_fake_iters = opt['train']['net_cl_fake_iters']
        self.log_dict = {}

    def init_training_settings(self):
        super().init_training_settings()
        # optimizers and schedulers already set up

        train_opt = self.opt['train']

        # define losses
        pixel_type = train_opt['cl_opt'].pop('type')
        cri_pix_cls = getattr(loss_module, pixel_type)
        self.cri_cl = cri_pix_cls(**train_opt['cl_opt']).to(self.device)

    def feed_data(self, data, is_val=False):
        if 'nrd_lq_i' in data:
            self.lq_i = torch.concat([data['nrd_lq_i'], data['sd_lq_i']], dim=0).to(self.device)
            self.gt_i = torch.concat([data['nrd_gt_i'], data['sd_gt_i']], dim=0).to(self.device)
            self.lq = self.lq_i
            self.gt = self.gt_i

            self.lq_j = torch.concat([data['nrd_lq_j'], data['sd_lq_j']], dim=0).to(self.device)
            self.gt_j = torch.concat([data['nrd_gt_j'], data['sd_gt_j']], dim=0).to(self.device)
        else:  # eval
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)

    def _optimize_cl_only(self, current_iter, tb_logger):
        if self.is_cl_disabled:
            return
        model_encoder = self.net_g.encoder
        loss_dict = OrderedDict()

        self.optimizer_g.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.is_amp):
            if current_iter < self.net_cl_real_iters:
                h_i, h_j, mlp_i, mlp_j = model_encoder(self.gt_i, self.gt_j)
            elif current_iter < self.net_cl_fake_iters:
                h_i, h_j, mlp_i, mlp_j = model_encoder(self.lq_i, self.lq_j)
            l_cr = self.cri_cl(mlp_i, mlp_j)
            loss_dict['l_cr'] = l_cr

        self._step_optimizer(l_cr, self.optimizer_g, self.net_g)

        self.log_dict.update(self.reduce_loss_dict(loss_dict))

    def _optimize_gan(self, loss_dict):
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

        self._step_optimizer(l_d_total, self.optimizer_d, self.net_d)
        return loss_dict

    def _step_optimizer(self, loss, optimizer, net):
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if use_grad_clip:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.01)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.01)
            optimizer.step()

    def optimize_parameters(self, current_iter, tb_logger):

        if current_iter < self.net_cl_fake_iters:
            self._optimize_cl_only(current_iter, tb_logger)
            return

        # optimize whole net_g (restore module)
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False
        
        self.optimizer_g.zero_grad()
        is_update_net_g = self.is_gan_disabled or (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters)
        with torch.cuda.amp.autocast(enabled=self.is_amp):
            pred, mlp_i, mlp_j = self.net_g(self.lq_i, self.lq_j)
            self.output = pred

            l_g_total = 0
            loss_dict = OrderedDict()
            if is_update_net_g:
                # pixel loss
                for cri_name, cri_pix in [('l_pix', self.cri_pix),
                                          ('l_pix_1', self.cri_pix_1),
                                          ('l_pix_2', self.cri_pix_2)]:
                    if cri_pix:
                        l_pix = cri_pix(pred, self.gt)

                        l_g_total += l_pix
                        loss_dict[cri_name] = l_pix

                # contrastive loss
                if not self.is_cl_disabled:
                    l_cr = self.cri_cl(mlp_i, mlp_j)
                    l_g_total += l_cr
                    loss_dict['l_cr'] = l_cr

                # gan loss
                if not self.is_gan_disabled:
                    fake_g_pred = self.net_d(self.output)
                    l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                    l_g_total += l_g_gan
                    loss_dict['l_gan'] = l_g_gan

        if is_update_net_g:
            self._step_optimizer(l_g_total, self.optimizer_g, self.net_g)
        
        # optimize net_d
        if not self.is_gan_disabled:
            loss_dict = self._optimize_gan(loss_dict)

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if current_iter < self.net_cl_fake_iters and current_iter != -1:
            return
        super().nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
