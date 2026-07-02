import torch
import torch.nn as nn
import math

# PyTorch 实现的核心工具函数（无需 Triton）
def cross_scan_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if in_channel_first:
        B, C, H, W = x.shape
        L = H * W
        if scans == 0:
            y = torch.stack([
                x.flatten(2, 3),
                x.transpose(2, 3).flatten(2, 3),
                torch.flip(x.flatten(2, 3), dims=[-1]),
                torch.flip(x.transpose(2, 3).flatten(2, 3), dims=[-1])
            ], dim=1)  # (B, 4, C, L)
        else:
            y = x.view(B, 1, C, L).repeat(1, 4, 1, 1)
    else:
        B, H, W, C = x.shape
        L = H * W
        if scans == 0:
            y = torch.stack([
                x.flatten(1, 2),
                x.transpose(1, 2).flatten(1, 2),
                torch.flip(x.flatten(1, 2), dims=[1]),
                torch.flip(x.transpose(1, 2).flatten(1, 2), dims=[1])
            ], dim=2)  # (B, L, 4, C)
        else:
            y = x.view(B, L, 1, C).repeat(1, 1, 4, 1)
    
    if in_channel_first != out_channel_first:
        y = y.permute(0, 3, 1, 2) if (in_channel_first and not out_channel_first) else y.permute(0, 2, 3, 1)
    return y

def cross_merge_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if in_channel_first:
        B, K, C, H, W = y.shape
        y = y.view(B, K, C, -1)
        if scans == 0:
            y = y[:, 0] + y[:, 1].view(B, C, W, H).transpose(2, 3).flatten(2, 3)
            y = y + y[:, 2].flip(dims=[-1]) + y[:, 3].view(B, C, W, H).transpose(2, 3).flatten(2, 3).flip(dims=[-1])
        else:
            y = y.sum(1)
        y = y.view(B, C, H, W)
    else:
        B, H, W, K, C = y.shape
        y = y.view(B, -1, K, C)
        if scans == 0:
            y = y[:, :, 0] + y[:, :, 1].view(B, W, H, C).transpose(1, 2).flatten(1, 2)
            y = y + y[:, :, 2].flip(dims=[1]) + y[:, :, 3].view(B, W, H, C).transpose(1, 2).flatten(1, 2).flip(dims=[1])
        else:
            y = y.sum(2)
        y = y.view(B, H, W, C)
    
    if in_channel_first != out_channel_first:
        y = y.permute(0, 2, 3, 1) if (in_channel_first and not out_channel_first) else y.permute(0, 3, 1, 2)
    return y

def selective_scan_torch(
    u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
    D: torch.Tensor = None, delta_bias: torch.Tensor = None, delta_softplus=True
):
    B, KC, L = u.shape
    N = A.shape[1]
    dtype = u.dtype

    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(-1)
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)
    
    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()
    
    K = B.shape[1]
    C_dim = KC // K
    B = B.view(B.shape[0], K, 1, N, L).repeat(1, 1, C_dim, 1, 1).view(B.shape[0], KC, N, L)
    C = C.view(C.shape[0], K, 1, N, L).repeat(1, 1, C_dim, 1, 1).view(C.shape[0], KC, N, L)
    
    x = torch.zeros(B.shape[0], KC, N, device=u.device, dtype=torch.float32)
    y_list = []
    for t in range(L):
        x = delta[:, :, t:t+1] * x + delta[:, :, t:t+1] * B[:, :, :, t] * u[:, :, t:t+1]
        y_t = torch.sum(x * C[:, :, :, t], dim=-1)
        y_list.append(y_t)
    y = torch.stack(y_list, dim=-1)
    
    if D is not None:
        y = y + u * D.unsqueeze(-1).float()
    
    return y.to(dtype)