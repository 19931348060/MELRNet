import torch
import torch.nn as nn
from mamba_ssm import Mamba

class VSSBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_in = nn.Conv2d(dim, dim, 1)
        self.mamba = Mamba(
            d_model=dim,
            d_state=16,
            expand=2,
        )
        self.proj_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj_in(x)

        # Rearrange [B,C,H,W] ¡ú [B,H*W,C]
        x = x.flatten(2).transpose(1, 2)

        # Mamba
        x = self.mamba(x)

        # back to [B,C,H,W]
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.proj_out(x)
        return x
