import torch
import torch.nn as nn

class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)  # scalar，无通道维度
        self.weight = nn.Parameter(torch.ones(num_features))  # 形状 [num_features]（对应通道数 C）
        self.bias = nn.Parameter(torch.zeros(num_features))    # 形状 [num_features]（对应通道数 C）
    
    def forward(self, x):
        # x shape: [B, C, H, W]
        x = torch.tanh(self.alpha * x)  # alpha 是 scalar，自动广播到所有元素，shape 不变
        
        # 关键：将 weight 和 bias 扩展为 [1, C, 1, 1]，适配 x 的 shape，实现按通道广播
        weight = self.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [C] → [1, C, 1, 1]
        bias = self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)      # [C] → [1, C, 1, 1]
        
        return x * weight + bias  # 广播相乘，shape 仍为 [B, C, H, W]