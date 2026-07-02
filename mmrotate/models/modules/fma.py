import torch
import torch.nn as nn

class FMA(nn.Module):
    def __init__(self, in_channels, mask_embedding_dim=8):
        super(FMA, self).__init__()
        self.in_channels = in_channels
        self.mask_embedding = nn.Sequential(
            nn.Conv2d(1, mask_embedding_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(mask_embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.attention_gate = nn.Sequential(
            nn.Conv2d(in_channels + mask_embedding_dim, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )
        self.feature_enhancement = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    @staticmethod
    def _create_foreground_mask(bboxes, img_shape, device, down_scale):
        B = len(bboxes)
        H, W = img_shape
        mask = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)
        for i in range(B):
            bbox = bboxes[i]
            valid_mask = (bbox[:, 2] > bbox[:, 0]) & (bbox[:, 3] > bbox[:, 1])
            bbox = bbox[valid_mask]
            if bbox.numel() == 0:
                continue
            bbox_scaled = bbox[:, :4] / down_scale
            x1 = bbox_scaled[:, 0].clamp(0, W - 1).long()
            y1 = bbox_scaled[:, 1].clamp(0, H - 1).long()
            x2 = bbox_scaled[:, 2].clamp(0, W - 1).long()
            y2 = bbox_scaled[:, 3].clamp(0, H - 1).long()
            for j in range(bbox.shape[0]):
                mask[i, 0, y1[j]:y2[j]+1, x1[j]:x2[j]+1] = 1.0
        return mask

    def forward(self, x, gt_bboxes, down_scale):
        B, C, H, W = x.shape
        device = x.device
        foreground_mask = self._create_foreground_mask(gt_bboxes, (H, W), device, down_scale)
        embedded_mask = self.mask_embedding(foreground_mask)
        combined = torch.cat([x, embedded_mask], dim=1)
        attention_weights = self.attention_gate(combined)
        attended_features = x * attention_weights
        enhanced_features = self.feature_enhancement(attended_features) + x
        return enhanced_features