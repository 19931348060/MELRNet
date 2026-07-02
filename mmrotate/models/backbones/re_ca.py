import torch
import torch.nn as nn
import math
from mmcv.cnn import build_norm_layer
try:
    import e2cnn.nn as enn
except ImportError:
    raise ImportError("e2cnn needs to be installed to support rotational equivariant features in MessDet (see paper appendix for dependencies).")

class RE_CA(nn.Module):
    """
    Rotation-Equivariant Channel Attention (RE-CA) Module
    Adapted from MessDet paper, compatible with LEGNet's feature grouping logic.
    """
    def __init__(self, channel, N=4, norm_layer=dict(type='BN'), act_layer=nn.ReLU):
        super(RE_CA, self).__init__()
        self.N = N  # Number of directional dimensions (rotation-equivariant groups)
        self.base_channels = channel // N  # Channels per directional group
        
        # Global average pooling (preserve batch and channel dims)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Fully connected layer for attention weight generation
        self.fc = nn.Linear(self.base_channels, self.base_channels)
        
        # Normalization layer (BatchNorm2d, built via MMCV)
        self.norm = build_norm_layer(norm_layer, self.base_channels)[1] 
        # Activation layer (consistent with LEGNet's act_layer)
        self.act = act_layer(inplace=True) 
    
    def forward(self, x):
        """
        Forward pass of RE-CA module.
        Args:
            x: Input tensor, either torch.Tensor ([B, C, H, W]) or enn.GeometricTensor (for rotation-equivariant features)
        Returns:
            feat_weighted: Attention-weighted feature tensor (same type as input)
        """
        # Handle rotation-equivariant input (e2cnn.GeometricTensor)
        if isinstance(x, enn.GeometricTensor):
            feat = x.tensor  # Extract raw tensor from GeometricTensor
            feat_type = x.type  # Preserve geometric type for output
        else:
            feat = x
            feat_type = None
        
        # Get input shape: Batch (B), Channels (C), Height (H), Width (W)
        B, C, H, W = feat.shape
        # Ensure channels are divisible by directional groups (N)
        assert C % self.N == 0, f"RE-CA requires input channels {C} to be divisible by directional dims {self.N}."

        # Step 1: Global average pooling (reduce spatial dims to 1x1)
        z = self.avg_pool(feat)  # Shape: [B, C, 1, 1]
        z = z.view(B, C)  # Flatten to 2D: [B, C]

        # Step 2: Group channels by direction (split C into N groups)
        z_grouped = z.view(B, self.N, self.base_channels)  # Shape: [B, N, base_channels]
        # Average across directional groups to get base weight
        z_base = z_grouped.mean(dim=1)  # Shape: [B, base_channels]

        # Step 3: Generate attention weights (safe dimension handling)
        s = self.fc(z_base)  # Shape: [B, base_channels] (2D)
        # 2D ¡ú 4D: Use reverse index (-1) to avoid hardcoding dims (compatible with any prior shape)
        s = s.unsqueeze(-1).unsqueeze(-1)  # Shape: [B, base_channels, 1, 1] (4D,ÊÊÅäBatchNorm2d)
        # Normalization + Activation
        s = self.norm(s)  # BatchNorm2d requires 4D input, now valid
        s = self.act(s)   # Activate to get non-linear weight
        
        # 4D ¡ú 2D: Auto-squeeze all dims with size 1 (safe for any intermediate shape)
        s = s.squeeze()  # Shape: [B, base_channels] (restore to 2D)

        # Step 4: Repeat weights to match original channel groups
        s_repeated = s.unsqueeze(1).repeat(1, self.N, 1)  # Shape: [B, N, base_channels]
        s_repeated = s_repeated.view(B, C)  # Reshape to [B, C] (match original channels)

        # Step 5: Apply attention weights to input features
        s_weight = s_repeated.view(B, C, 1, 1)  # Shape: [B, C, 1, 1] (broadcastable to [B,C,H,W])
        feat_weighted = feat * s_weight  # Element-wise multiplication (attention weighting)

        # Restore rotation-equivariant type if input was GeometricTensor
        if feat_type is not None:
            return enn.GeometricTensor(feat_weighted, feat_type)
        else:
            return feat_weighted