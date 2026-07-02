import torch
import torch.nn as nn
import math

class AngleEmbedding(nn.Module):
    """RoMA中的角度嵌入模块，将角度θ编码为特征向量"""
    def __init__(self, dim, max_freq=10.0):
        super().__init__()
        self.dim = dim  # 嵌入维度（需与当前特征通道匹配）
        self.max_freq = max_freq
        # 生成频率参数（类似位置编码）
        self.freqs = nn.Parameter(torch.exp(torch.linspace(0, math.log(max_freq), dim//2) * -math.log(10)), requires_grad=False)

    def forward(self, theta):
        """
        Args:
            theta: 旋转角度（弧度），形状为 [B, 1, H, W] 或 [B, N]（N为目标数）
        Returns:
            angle_feat: 角度嵌入特征，形状为 [B, dim, H, W] 或 [B, N, dim]
        """
        # 角度映射到 [-π, π]
        theta = theta % (2 * math.pi)
        theta = theta - 2 * math.pi * (theta > math.pi)
        
        # 生成正弦/余弦嵌入
        freqs = self.freqs[None, :, None, None]  # [1, dim//2, 1, 1]
        theta = theta.unsqueeze(1)  # [B, 1, H, W]
        emb = theta * freqs  # [B, dim//2, H, W]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)  # [B, dim, H, W]
        return emb