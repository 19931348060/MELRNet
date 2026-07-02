import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer

class ConvMod(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        assert dim % 4 == 0
        self.dim = dim
        
        self.norm1 = build_norm_layer(norm_layer, dim)[1]
        self.a1 = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim // 4, 7, padding=3, groups=dim // 4)
        )
        self.v1 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.v11 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.v12 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.conv3_1 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        self.norm2 = build_norm_layer(norm_layer, dim // 2)[1]
        self.a2 = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim // 2, 9, padding=4, groups=dim // 2)
        )
        self.v2 = nn.Conv2d(dim // 2, dim // 2, 1)
        self.v21 = nn.Conv2d(dim // 2, dim // 2, 1)
        self.v22 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.proj2 = nn.Conv2d(dim // 2, dim // 4, 1)
        self.conv3_2 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        self.norm3 = build_norm_layer(norm_layer, dim * 3 // 4)[1]
        self.a3 = nn.Sequential(
            nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 11, padding=5, groups=dim * 3 // 4)
        )
        self.v3 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)
        self.v31 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)
        self.v32 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.proj3 = nn.Conv2d(dim * 3 // 4, dim // 4, 1)
        self.conv3_3 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        # 可学习缩放因子
        self.scale = nn.Parameter(torch.tensor(0.01))  # 初始缩放更小
        self.mid_scale = nn.Parameter(torch.tensor(0.1))  # 中间激活缩放
        
        # 初始化
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # 打印输入范围（调试用）
        # print(f"in: min={x.min().item():.3f}, max={x.max().item():.3f}, mean={x.mean().item():.3f}")
        
        x = self.norm1(x)
        x_split = torch.split(x, self.dim // 4, dim=1)

        a = self.a1(x_split[0])
        a = a.clamp(-10, 10) * self.mid_scale  # 限制范围并缩放
        mul = a * self.v1(x_split[0]).clamp(-10, 10)
        mul = self.v11(mul)
        x1 = self.conv3_1(self.v12(x_split[1]))
        x1 = x1 + a
        x1 = torch.cat((x1, mul), dim=1)

        x1 = self.norm2(x1)
        a = self.a2(x1).clamp(-10, 10) * self.mid_scale
        mul = a * self.v2(x1).clamp(-10, 10)
        mul = self.v21(mul)
        x2 = self.conv3_2(self.v22(x_split[2]))
        x2 = x2 + self.proj2(a)
        x2 = torch.cat((x2, mul), dim=1)

        x2 = self.norm3(x2)
        a = self.a3(x2).clamp(-10, 10) * self.mid_scale
        mul = a * self.v3(x2).clamp(-10, 10)
        mul = self.v31(mul)
        x3 = self.conv3_3(self.v32(x_split[3]))
        x3 = x3 + self.proj3(a)
        x = torch.cat((x3, mul), dim=1)

        # 输出再缩放
        return x * self.scale