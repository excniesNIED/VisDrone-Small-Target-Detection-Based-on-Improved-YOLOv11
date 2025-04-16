"""
Attention modules
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ('ChannelAttention', 'SpatialAttention', 'CBAM', 'EMA')

class EMA(nn.Module):
    """Efficient Multi-scale Attention module."""
    def __init__(self, c1, kernel_size=3, groups=8):
        """Initialize EMA module."""
        super().__init__()
        self.groups = groups
        self.avg_pool_x = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, None))
        self.conv = Conv(c1, c1, kernel_size)
        self.conv1x1 = Conv(c1 * 2, c1, 1)
        self.conv1x1_final = Conv(c1, c1, 1)
        
    def forward(self, x):
        """Forward pass of EMA."""
        B, C, H, W = x.shape
        G = self.groups
        
        # 分组处理
        x = x.view(B, G, C // G, H, W)
        
        # 1C-Module: 空间注意力
        x_pool_x = self.avg_pool_x(x.view(B * G, C // G, H, W))
        x_pool_y = self.avg_pool_y(x.view(B * G, C // G, H, W))
        x_pool = torch.cat([x_pool_x, x_pool_y], dim=2)
        x_pool = self.conv1x1(x_pool.view(B, C * 2, 1, W))
        spatial_att = torch.sigmoid(x_pool)
        x_spatial = x * spatial_att.view(B, 1, 1, 1, W)
        
        # 2C-Module: 通道注意力
        x_conv = self.conv(x.view(B * G, C // G, H, W)).view(B, G, C // G, H, W)
        x_pool_c = F.adaptive_avg_pool2d(x_spatial.view(B * G, C // G, H, W), (1, 1))
        x_pool_c = x_pool_c.view(B, G, C // G, 1, 1)
        channel_att = F.softmax(x_pool_c, dim=2)
        
        # 特征融合
        out = x_spatial * channel_att + x_conv * (1 - channel_att)
        out = self.conv1x1_final(out.view(B, C, H, W))
        
        return out 