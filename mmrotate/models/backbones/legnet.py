# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from typing import List
from torch import Tensor
import os
import copy
from mmcv.cnn import build_norm_layer
from math import log
import numpy
import matplotlib.pyplot as plt
from ..builder import ROTATED_BACKBONES
from .attention_modules import ConvMod
from .messdet_models_layer import ennConvModule
#from .re_ca import RE_CA
from .dyT import DyT

try:
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False


class DRFD(nn.Module):
    def __init__(self, dim, norm_layer, act_layer, is_strict=True):
        super().__init__()
        self.dim = dim
        self.outdim = dim * 2
        
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim)

        self.conv_c = ennConvModule(
            in_channels=dim * 2,
            out_channels=dim * 2,
            kernel_size=3,
            stride=2,  
            padding=1,
            groups=dim * 2,
            norm_layer=norm_layer,
            act_layer=act_layer
        )
        
        #self.conv_c = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=2, padding=1, groups=dim * 2)
        self.act_c = act_layer()
        #self.norm_c = build_norm_layer(norm_layer, dim * 2)[1]
        self.norm_c = DyT(num_features=dim * 2, alpha_init_value=0.5)
        self.max_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.norm_m = build_norm_layer(norm_layer, dim * 2)[1]
        self.norm_m = DyT(num_features=dim * 2, alpha_init_value=0.5)
        self.fusion = nn.Conv2d(dim * 4, self.outdim, kernel_size=1, stride=1)
        # gaussian
        self.gaussian = Gaussian(self.outdim, 5, 0.5, norm_layer, act_layer, feature_extra=False)
        #self.norm_g = build_norm_layer(norm_layer, self.outdim)[1]
        self.norm_g = DyT(num_features=self.outdim, alpha_init_value=0.5)

    def forward(self, x):  # x = [B, C, H, W]

        x = self.conv(x)  # x = [B, 2C, H, W]
        gaussian = self.gaussian(x)
        x = self.norm_g(x + gaussian)
        max = self.norm_m(self.max_m(x))  # m = [B, 2C, H/2, W/2]
        conv = self.norm_c(self.act_c(self.conv_c(x)))  # c = [B, 2C, H/2, W/2]
        x = torch.cat([conv, max], dim=1)  # x = [B, 2C+2C, H/2, W/2]  -->  [B, 4C, H/2, W/2]
        x = self.fusion(x)  # x = [B, 4C, H/2, W/2]     -->  [B, 2C, H/2, W/2]

        return x


def show_feature(out):
    out_cpu = out.cpu()
    feature_map = out_cpu.detach().numpy()
    # [N£¬ C, H, W] -> [H, W£¬ C]
    im = numpy.squeeze(feature_map)
    im = numpy.transpose(im, [1, 2, 0])
    for c in range(24):
        ax = plt.subplot(4, 6, c + 1)
        plt.axis('off') 
        # [H, W, C]
        plt.imshow(im[:, :, c], cmap=plt.get_cmap('Blues'))
    plt.show()


class Conv_Extra(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channel, 64, 1),
                                   #build_norm_layer(norm_layer, 64)[1],
                                   DyT(num_features=64, alpha_init_value=0.5),
                                   act_layer(),
                                   nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
                                   #build_norm_layer(norm_layer, 64)[1],
                                   DyT(num_features=64, alpha_init_value=0.5),
                                   act_layer(),
                                   nn.Conv2d(64, channel, 1),
                                   #build_norm_layer(norm_layer, channel)[1]
                                    DyT(num_features=channel, alpha_init_value=0.5))
    def forward(self, x):
        out = self.block(x)
        return out


class Scharr(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(Scharr, self).__init__()
        scharr_x = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        scharr_y = torch.tensor([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.conv_x = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_y = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_x.weight.data = scharr_x.repeat(channel, 1, 1, 1)
        self.conv_y.weight.data = scharr_y.repeat(channel, 1, 1, 1)
        #self.norm = build_norm_layer(norm_layer, channel)[1]
        self.norm = DyT(num_features=channel, alpha_init_value=0.5)
        self.act = act_layer()
        self.conv_extra = Conv_Extra(channel, norm_layer, act_layer)

    def forward(self, x):
        # show_feature(x)
        edges_x = self.conv_x(x)
        edges_y = self.conv_y(x)
        scharr_edge = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        scharr_edge = self.act(self.norm(scharr_edge))
        out = self.conv_extra(x + scharr_edge)

        return out


class Gaussian(nn.Module):
    def __init__(self, dim, size, sigma, norm_layer, act_layer, feature_extra=True):
        super().__init__()
        self.feature_extra = feature_extra
        gaussian = self.gaussian_kernel(size, sigma)
        gaussian = nn.Parameter(data=gaussian, requires_grad=False).clone()
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1)
        #self.norm = build_norm_layer(norm_layer, dim)[1]
        self.norm = DyT(num_features=dim, alpha_init_value=0.5)
        self.act = act_layer()
        if feature_extra == True:
            self.conv_extra = Conv_Extra(dim, norm_layer, act_layer)

    def forward(self, x):
        edges_o = self.gaussian(x)
        gaussian = self.act(self.norm(edges_o))
        if self.feature_extra == True:
            out = self.conv_extra(x + gaussian)
        else:
            out = gaussian
        return out
    
    def gaussian_kernel(self, size: int, sigma: float):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
             for y in range(-size // 2 + 1, size // 2 + 1)
             ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()


class LFEA(nn.Module):
    def __init__(self, channel, norm_layer, act_layer,
                 use_convmod=False, use_vss=False):
        super().__init__()
        self.use_vss = use_vss

        if use_vss:
            from .vss import VSSBlock
            self.conv2d = VSSBlock(channel)
        else:
            if use_convmod:
                self.conv2d = ConvMod(channel, norm_layer)
            else:
                self.conv2d = nn.Sequential(
                    nn.Conv2d(channel, channel, 3, stride=1, padding=1, bias=False),
                    DyT(num_features=channel, alpha_init_value=0.5),
                    act_layer(),
                )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        k = 3
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2)
        self.sigmoid = nn.Sigmoid()
        self.norm = DyT(num_features=channel, alpha_init_value=0.5)

    def forward(self, c, att):
        x = self.conv2d(c)

        att = c * att + c
        wei = self.avg_pool(att)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)) \
                .transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = self.norm(x + att * wei)
        return x



class LFE_Module(nn.Module):
    def __init__(self,
                 dim,
                 stage,
                 mlp_ratio,
                 drop_path,
                 act_layer,
                 norm_layer,
                 use_convmod=False,
                 mamba_stage=-1  
                 ):
        super().__init__()
        self.stage = stage
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            build_norm_layer(norm_layer, mlp_hidden_dim)[1],
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]
        self.mlp = nn.Sequential(*mlp_layer)

        use_vss = False
        if mamba_stage == -1:
            use_vss = False 
        elif mamba_stage == self.stage:
            use_vss = True  
        elif mamba_stage == 4: 
            use_vss = (self.stage == 2) or (self.stage == 3)

        self.LFEA = LFEA(
            channel=dim,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_convmod=use_convmod,
            use_vss=use_vss,
        )        

        if stage == 0:
            self.Scharr_edge = Scharr(dim, norm_layer, act_layer)
        else:
            self.gaussian = Gaussian(dim, 5, 1.0, norm_layer, act_layer)
        self.norm = DyT(num_features=dim, alpha_init_value=0.5)

    def forward(self, x):
        if self.stage == 0:
            att = self.Scharr_edge(x)
        else:
            att = self.gaussian(x)
        x_att = self.LFEA(x, att)
        x = x + self.norm(self.drop_path(self.mlp(x_att)))
        return x

class BasicStage(nn.Module):
    def __init__(self,
                 dim,
                 stage,
                 depth,
                 mlp_ratio,
                 drop_path,
                 norm_layer,
                 act_layer,
                 mamba_stage=-1,
                 use_convmod=False
                 ):
        super().__init__()

        blocks_list = [
            LFE_Module(
                dim=dim,
                stage=stage,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_convmod=use_convmod
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        return self.blocks(x)

class LoGFilter(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, sigma, norm_layer, act_layer):
        super().__init__()
        self.conv_init = nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3)
        ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax)
        kernel = (xx**2 + yy**2 - 2 * sigma**2) / (2 * math.pi * sigma**4) * torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel - kernel.mean()
        kernel = kernel / kernel.sum()
        log_kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.LoG = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2), groups=out_c, bias=False)
        self.LoG.weight.data = log_kernel.repeat(out_c, 1, 1, 1)
        self.act = act_layer()
        #self.norm1 = build_norm_layer(norm_layer, out_c)[1]
        self.norm1 = DyT(num_features=out_c, alpha_init_value=0.5)
        #self.norm2 = build_norm_layer(norm_layer, out_c)[1]
        self.norm2 = DyT(num_features=out_c, alpha_init_value=0.5)
    
    def forward(self, x):
        x = self.conv_init(x)
        LoG = self.LoG(x)
        LoG_sq = LoG ** 2
        LoG_edge = torch.pow(LoG_sq + 1e-8, 0.25)  
        LoG_edge = torch.clamp(LoG_edge, min=0.0, max=5.0)  
        LoG_edge = self.act(self.norm1(LoG_edge))
        return torch.clamp(x, min=-5.0, max=5.0)
    

class Stem(nn.Module):

    def __init__(self, in_chans, stem_dim, act_layer, norm_layer):
        super().__init__()
        out_c14 = int(stem_dim / 4)  # stem_dim / 2
        out_c12 = int(stem_dim / 2)  # stem_dim / 2
        # original size to 2x downsampling layer
        self.Conv_D = nn.Sequential(
            nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14),
            nn.Conv2d(out_c12, out_c12, kernel_size=3, stride=2, padding=1, groups=out_c12),
            #build_norm_layer(norm_layer, out_c12)[1]
            DyT(num_features=out_c12, alpha_init_value=0.5))
        self.LoG = LoGFilter(in_chans, out_c14, 7, 1.0, norm_layer, act_layer)
        # gaussian
        self.gaussian = Gaussian(out_c12, 9, 0.5, norm_layer, act_layer)
        #self.norm = build_norm_layer(norm_layer, out_c12)[1]
        self.norm = DyT(num_features=out_c12, alpha_init_value=0.5)
        #self.drfd = DRFD(out_c12, norm_layer, act_layer)
        self.drfd = DRFD(
            dim=out_c12, 
            norm_layer=norm_layer, 
            act_layer=act_layer,
            is_strict=True  
        )

    def forward(self, x):
        x = self.LoG(x)
        # original size to 2x downsampling layer
        x = self.Conv_D(x)
        x = self.norm(x + self.gaussian(x))
        x = self.drfd(x)

        return x  # x = [B, C, H/4, W/4]


@ROTATED_BACKBONES.register_module()
class LWEGNet(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 stem_dim=32,
                 depths=(1, 4, 4, 2),
                 norm_layer=dict(type='BN', requires_grad=True),
                 act_layer=nn.ReLU,
                 mlp_ratio=2.,
                 feature_dim=1280,
                 drop_path_rate=0.1,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                  mamba_stage=-1,
                 **kwargs):
        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.num_stages = len(depths)
        self.num_features = int(stem_dim * 2 ** (self.num_stages - 1))
        self.mamba_stage = mamba_stage

        if stem_dim == 96:
            act_layer = nn.ReLU

        self.Stem = Stem(in_chans=in_chans, stem_dim=stem_dim, act_layer=act_layer, norm_layer=norm_layer)

        dpr = [x.item()
               for x in torch.linspace(0, 0.05, sum(depths))] 

        # build layers
        stages_list = []
        for i_stage in range(self.num_stages):
            use_convmod = (i_stage == 1) or (i_stage == 2)
    
            stage = BasicStage(
                dim=int(stem_dim * 2 ** i_stage),
                stage=i_stage,
                depth=depths[i_stage],
                mlp_ratio=mlp_ratio,
                drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_convmod=use_convmod,
                mamba_stage=mamba_stage
            )
            stages_list.append(stage)

            if i_stage < self.num_stages - 1:
                stages_list.append(
                    DRFD(
                        dim=int(stem_dim * 2 ** i_stage), 
                        norm_layer=norm_layer, 
                        act_layer=act_layer,
                        is_strict=True  
                    )
                )
              #  stages_list.append(
              #     DRFD(dim=int(stem_dim * 2 ** i_stage), norm_layer=norm_layer, act_layer=act_layer)
              #  )
        self.stages = nn.Sequential(*stages_list)

        self.fork_feat = fork_feat
        self.forward = self.forward_det
        self.out_indices = [0, 2, 4, 6]
        for i_emb, i_layer in enumerate(self.out_indices):
            dim = int(stem_dim * 2 ** i_emb)
            layer = DyT(num_features=dim, alpha_init_value=0.5) 
            #layer = build_norm_layer(norm_layer, int(stem_dim * 2 ** i_emb))[1]
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        self.init_cfg = None 
        self.pretrained = None

        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    def init_weights(self):

        if self.init_cfg is not None:
            if self.init_cfg.get('type') == 'Pretrained' and 'checkpoint' in self.init_cfg:
                if has_mmdet:
                    logger = get_root_logger()
                    _load_checkpoint(
                        self,
                        self.init_cfg['checkpoint'],
                        prefix=self.init_cfg.get('prefix', ''),
                        logger=logger
                    )
                else:
                    raise ImportError("Install mmdetection to load pretrained weights (per <LEGNet.pdf>)")
            elif self.init_cfg.get('type') == 'Kaiming':
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_uniform_(
                            m.weight,
                            a=math.sqrt(5),
                            mode='fan_in',
                            nonlinearity='relu'
                        )
                    elif isinstance(m, ennConvModule):
                        nn.init.kaiming_uniform_(
                            m.conv.weight,
                            a=math.sqrt(5),
                            mode='fan_in',
                            nonlinearity='relu'
                        )
                    #elif isinstance(m, nn.BatchNorm2d):
                    elif isinstance(m, DyT):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                        nn.init.constant_(m.alpha, 0.5)
            else:
                raise ValueError(
                    f"LWEGNet only supports 'Pretrained' (<LEGNet.pdf>) or 'Kaiming' (MessDet) in init_cfg, "
                    f"but got type '{self.init_cfg.get('type')}'"
                )
        else:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, ennConvModule)):
                    nn.init.kaiming_uniform_(
                        m.conv.weight if isinstance(m, ennConvModule) else m.weight,
                        a=math.sqrt(5),
                        mode='fan_in',
                        nonlinearity='relu'
                    )
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    def forward_det(self, x):
        """Forward method for detection (compatible with both papers)"""
        outs = []
        x = self.Stem(x)  
        for i, block in enumerate(self.stages):
            x = block(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_norm = norm_layer(x)
                outs.append(x_norm)
        return tuple(outs)
        
'''
class MARNet(nn.Module): 
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 stem_dim=32,
                 depths=(1, 4, 4, 2),
                 norm_layer=dict(type='BN', requires_grad=True),
                 act_layer=nn.ReLU,
                 mlp_ratio=2.,
                 feature_dim=1280,
                 drop_path_rate=0.1,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.num_stages = len(depths)
        self.num_features = int(stem_dim * 2 ** (self.num_stages - 1))

        if stem_dim == 96:
            act_layer = nn.ReLU

        self.Stem = Stem(in_chans=in_chans, stem_dim=stem_dim, act_layer=act_layer, norm_layer=norm_layer)

        dpr = [x.item()
               for x in torch.linspace(0, 0.05, sum(depths))] 

        # build layers
        stages_list = []
        for i_stage in range(self.num_stages):
            use_convmod = (i_stage == 1) or (i_stage == 2)
    
            stage = BasicStage(
                dim=int(stem_dim * 2 ** i_stage),
                stage=i_stage,
                depth=depths[i_stage],
                mlp_ratio=mlp_ratio,
                drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_convmod=use_convmod
            )
            stages_list.append(stage)

            if i_stage < self.num_stages - 1:
                stages_list.append(
                    DRFD(
                        dim=int(stem_dim * 2 ** i_stage), 
                        norm_layer=norm_layer, 
                        act_layer=act_layer,
                        is_strict=True  
                    )
                )
              #  stages_list.append(
              #     DRFD(dim=int(stem_dim * 2 ** i_stage), norm_layer=norm_layer, act_layer=act_layer)
              #  )
        self.stages = nn.Sequential(*stages_list)

        self.fork_feat = fork_feat
        self.forward = self.forward_det
        self.out_indices = [0, 2, 4, 6]
        for i_emb, i_layer in enumerate(self.out_indices):
            dim = int(stem_dim * 2 ** i_emb)
            layer = DyT(num_features=dim, alpha_init_value=0.5) 
            #layer = build_norm_layer(norm_layer, int(stem_dim * 2 ** i_emb))[1]
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        self.init_cfg = None 
        self.pretrained = None

        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    def init_weights(self):

        if self.init_cfg is not None:
            if self.init_cfg.get('type') == 'Pretrained' and 'checkpoint' in self.init_cfg:
                if has_mmdet:
                    logger = get_root_logger()
                    _load_checkpoint(
                        self,
                        self.init_cfg['checkpoint'],
                        prefix=self.init_cfg.get('prefix', ''),
                        logger=logger
                    )
                else:
                    raise ImportError("Install mmdetection to load pretrained weights (per <LEGNet.pdf>)")
            elif self.init_cfg.get('type') == 'Kaiming':
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_uniform_(
                            m.weight,
                            a=math.sqrt(5),
                            mode='fan_in',
                            nonlinearity='relu'
                        )
                    elif isinstance(m, ennConvModule):
                        nn.init.kaiming_uniform_(
                            m.conv.weight,
                            a=math.sqrt(5),
                            mode='fan_in',
                            nonlinearity='relu'
                        )
                    #elif isinstance(m, nn.BatchNorm2d):
                    elif isinstance(m, DyT):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                        nn.init.constant_(m.alpha, 0.5)
            else:
                raise ValueError(
                    f"MARNet only supports 'Pretrained' (<LEGNet.pdf>) or 'Kaiming' (MessDet) in init_cfg, "
                    f"but got type '{self.init_cfg.get('type')}'"
                )
        else:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, ennConvModule)):
                    nn.init.kaiming_uniform_(
                        m.conv.weight if isinstance(m, ennConvModule) else m.weight,
                        a=math.sqrt(5),
                        mode='fan_in',
                        nonlinearity='relu'
                    )
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward_det(self, x):
        """Forward method for detection (compatible with both papers)"""
        outs = []
        x = self.Stem(x)  
        for i, block in enumerate(self.stages):
            x = block(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_norm = norm_layer(x)
                outs.append(x_norm)
        return tuple(outs)
'''
