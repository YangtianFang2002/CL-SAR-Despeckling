'''
MLEM: Multi-looking Excitation Modulation Net
'''
import sys
from typing import List
sys.path.append('.')
sys.path.append('../')
sys.path.append('../../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


from basicsr.models.archs.arch_util import LayerNorm2d

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class ContextBlock(nn.Module):
    def __init__(self,inplanes,ratio,pooling_type='att',
                 fusion_types=('channel_add', ), kv_channel=256):
        super(ContextBlock, self).__init__()
        valid_fusion_types = ['channel_add', 'channel_mul']

        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        
        self.kv_channel = kv_channel
        self.kv_gn = nn.GroupNorm(16, self.kv_channel)
        self.kv_conv = nn.Conv2d(in_channels=self.kv_channel, out_channels=inplanes, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        

    def spatial_pool(self, x, k_v):
        # k_v [N, C]
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            k_v = self.kv_conv(self.kv_gn(k_v))
            k_v = F.interpolate(k_v, size=(height, width), mode='bilinear', align_corners=False)
            k_v = k_v.view(batch, channel, height * width)
            input_x = input_x.view(batch, channel, height * width) * k_v
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context

    def forward(self, x, k_v):
        # [N, C, 1, 1]
        context = self.spatial_pool(x, k_v)
        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


class FMB(nn.Module):
    def __init__(self, c, DW_Expand=2, kv_channel=256):
        super().__init__()

        # Encode
        self.gn = nn.GroupNorm(16, c)
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.gelu = nn.GELU()
        
        # ELA
        self.ela_gn = nn.GroupNorm(16, dw_channel)
        ela_kernel = 7
        pad = ela_kernel // 2
        self.ela_conv = nn.Conv1d(dw_channel, dw_channel, kernel_size=ela_kernel, padding=pad, groups=dw_channel, bias=False)
        self.ela_sigmoid = nn.Sigmoid()
        
        ## kv compress
        self.kv_gn = nn.GroupNorm(16, kv_channel)
        self.kv_conv = nn.Conv2d(in_channels=kv_channel, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        
        self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Glocal Context
        self.gc = ContextBlock(c, 1)
        # self.conv4 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # branch modulation
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        
    def efficient_localization_attention(self, x, k_v):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
        x_h = self.ela_sigmoid(self.ela_gn(self.ela_conv(x_h))).view(b, c, h, 1)
        x_w = self.ela_sigmoid(self.ela_gn(self.ela_conv(x_w))).view(b, c, 1, w)
        
        k_v = self.kv_conv(self.kv_gn(k_v))  # [B, kv_c, fs, fs] -> [B, c, fs, fs]
        k_v = F.interpolate(k_v, size=(h, w), mode='bilinear', align_corners=False)
        return x * x_h * x_w * k_v

    def forward(self, inp):
        x0, k_v0 = inp
        k_v = k_v0  # [B, c, fs, fs]

        x = self.gn(x0)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.efficient_localization_attention(x, k_v)
        x = self.conv3(x)
        
        x = x0 + x * self.beta

        y = self.gn(x)
        y = self.gc(y, k_v)

        return y + x * self.gamma, k_v0  # adapt to Sequential


class EAP(nn.Module):
    def __init__(self, channel: int, base_channel: int, depth: int, layer: int, pooling='max'):
        super().__init__()

        self.attn_blks = nn.ModuleList()
        for i in range(1, depth + 1):
            chan = base_channel * 2 ** (i - 1)  # input feature channel
            
            if i > layer:  # upsample, start from 1
                attn_blk = nn.Sequential()
                for _ in range(i - layer):
                    attn_blk.append(nn.Conv2d(chan, chan * 2, 1, bias=False)),
                    attn_blk.append(nn.PixelShuffle(2))
                    chan = chan // 2
                attn_blk.append(nn.Conv2d(chan, channel, kernel_size=3, padding=1, stride=1))
                attn_blk.append(NAFBlock(channel))
            else:
                attn_blk = nn.Sequential(nn.Conv2d(chan, channel, kernel_size=3, padding=1, stride=1))
                
            self.attn_blks.append(attn_blk)

        output_channel = base_channel * (2 ** (layer - 1))
        self.ending = nn.Conv2d(channel, output_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        self.layer = layer
        self.channel = channel
        self.pooling = pooling

    def forward(self, xs: List[torch.Tensor], anchor: torch.Tensor):
        target_size = (anchor.shape[-2], anchor.shape[-1])
        b = anchor.shape[0]
        ans = torch.ones((b, self.channel, *target_size), dtype=anchor.dtype, device=anchor.device)
        
        for i, x in enumerate(xs):
            x = self.attn_blks[i](x)
            if i < self.layer:
                if self.pooling == "max":
                    x = F.adaptive_max_pool2d(x, target_size)
                elif self.pooling == "avg":
                    x = F.adaptive_avg_pool2d(x, target_size)
                else:
                    raise NotImplementedError
            ans = ans * x
        ans = self.ending(ans)
        return ans


class MDN(nn.Module):

    def __init__(self, img_channel=3, width=32, sc_width=32, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], eap_pooling="max", is_eap=True, **kwargs):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.eaps = nn.ModuleList()
        self.ups = nn.ModuleList()                                                                                                                                                                                                                                                      
        self.downs = nn.ModuleList()
        self.is_eap = is_eap

        chan = width
        for i, num in enumerate(enc_blk_nums):
            self.encoders.append(
                nn.Sequential(
                    *[FMB(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[FMB(chan) for _ in range(middle_blk_num)]
            )
        
        for i, num in enumerate(dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.eaps.append(
                EAP(sc_width, width, depth=len(enc_blk_nums), layer=len(dec_blk_nums) - i, pooling=eap_pooling)
            )
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)
        
        # self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_uniform_(m.weight)
                init.constant_(m.weight, 0.)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, inp, k_v):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x, k_v = encoder([x, k_v])
            encs.append(x)
            x = down(x)

        x, _ = self.middle_blks([x, k_v])  # adapt to Sequential
        
        for decoder, eap, up, enc_skip in zip(self.decoders, self.eaps, self.ups, encs[::-1]):
            x = up(x)
            if self.is_eap:
                enc = eap(encs, enc_skip)
            else:
                enc = enc_skip
            x = x + enc
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class Encoder(nn.Module):
    def __init__(self, channel=256, feature_size=32, **kwargs):
        super(Encoder, self).__init__()
        
        self.c = channel
        self.fs = feature_size
        
        self.E = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            NAFBlock(128),
            nn.Conv2d(128, self.c, kernel_size=3, padding=1),
            NAFBlock(self.c, DW_Expand=1, FFN_Expand=1),
            nn.AdaptiveAvgPool2d(self.fs),
        )
        self.mlp_pooling = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.c, self.c),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.c, self.c),
        )

    def forward(self, x):
        # x: [B, 1, H, W]
        b, c, h, w = x.shape
        fea = self.E(x)  # [B, C, fs, fs]
        out = self.mlp_pooling(fea).view(b, -1)  # [B, C * fs * fs]
        out = self.mlp(out)

        return fea, out


class ResSimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResBlock used in DASR to obtain hi = f(x ̃i) = ResBlock(x ̃i)
    where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder):
        super(ResSimCLR, self).__init__()

        self.encoder = encoder

    def forward(self, x_i, x_j):
        h_i, mlp_i = self.encoder(x_i)
        h_j, mlp_j = self.encoder(x_j)

        return h_i, h_j, mlp_i, mlp_j


class MDN1(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MDN1, self).__init__()

        # Generator
        self.mdn = MDN(*args, **kwargs)

        # Encoder
        self.encoder = ResSimCLR(encoder=Encoder(**kwargs))

    def forward(self, lr_d_i, lr_d_j=None):
        if self.training:
            x_query = lr_d_i                          # b, c, h, w
            x_key = lr_d_j                           # b, c, h, w
            assert x_key is not None
            # degradation-aware represenetion learning
            h_i, h_j, mlp_i, mlp_j = self.encoder(x_query, x_key)
            # print(f"h_i: {h_i.shape}")
            # degradation-aware restore
            sr_i = self.mdn(lr_d_i, h_i)
            return sr_i, mlp_i, mlp_j
        else:
            # degradation-aware represenetion learning
            h_lr_i, _, _, _ = self.encoder(lr_d_i, lr_d_i)
            # print(f"h_i: {h_lr_i.shape}")
            # degradation-aware SR
            sr = self.mdn(lr_d_i, h_lr_i)
            return sr


if __name__ == '__main__':
    _img_channel = 1

    _width = 32
    _sc_width = 32
    _enc_blks = [1, 1, 2, 2]
    _middle_blk_num = 2
    _dec_blks = [2, 2, 1, 1]
    
    _net = MDN1(img_channel=_img_channel, width=_width, sc_width=_sc_width, middle_blk_num=_middle_blk_num,
                   enc_blk_nums=_enc_blks, dec_blk_nums=_dec_blks)
    _net.eval()
    check_inp = torch.randn(3, 1, 128, 128)
    
    _ = _net(check_inp)

    _inp_shape = (1, 256, 256)

    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(_net, _inp_shape, verbose=False, print_per_layer_stat=True)

    print(f"flops={flops}, params={params}")
    # flops=133.37 GMac, params=85.39 M for NAFU3NetGan
    # flops=109.58 GMac, params=62.82 M for MLEPNet1
    # flops=214.35 GMac, params=68.66 M for MLEMNet
    # flops=95.44 GMac, params=10.04 M for MLEMNet3
    # flops=86.03 GMac, params=11.49 M for MLEMNet3 wo eap
