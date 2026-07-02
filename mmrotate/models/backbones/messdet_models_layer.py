import math
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer, build_activation_layer  # 保持导入
from .dyT import DyT

class ennConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, norm_layer=None, act_layer=None):
        super(ennConvModule, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        
        # 修复 norm_layer（之前的逻辑正确，保留）
        if norm_layer is not None:
            #self.norm = build_norm_layer(norm_layer, out_channels)[1]
            self.norm = DyT(num_features=out_channels, alpha_init_value=0.5)
        else:
            self.norm = nn.Identity()
        
        # 修复 act_layer：区分类型是字典还是类型对象
        if act_layer is not None:
            # 情况1：若 act_layer 是配置字典（如 {"type": "ReLU"}），用 build_activation_layer 解析
            if isinstance(act_layer, dict):
                self.act = build_activation_layer(act_layer)
            # 情况2：若 act_layer 是类型（如 nn.ReLU），直接实例化
            else:
                self.act = act_layer()  # 注意加括号实例化（如 nn.ReLU()）
        else:
            self.act = nn.Identity()
        
        # 初始化逻辑
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x