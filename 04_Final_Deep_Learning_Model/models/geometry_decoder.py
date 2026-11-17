import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolution + GroupNorm + GELU block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class GeometryDecoder(nn.Module):
    """
    Predicts pixel-level road probability and intersection probability maps.
    Input: [B, in_channels (256), H_feat, W_feat] e.g., 32x32
    Output: [B, 2, H_img, W_img] e.g., 512x512
      - Channel 0: Road presence probability
      - Channel 1: Intersection presence probability
    """
    def __init__(self, in_channels: int = 256, feat_dim: int = 128):
        super().__init__()
        self.proj = ConvBlock(in_channels, feat_dim)

        # 4x upsampling stages (32 -> 64 -> 128 -> 256 -> 512)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(feat_dim, 64, kernel_size=2, stride=2),
            ConvBlock(64, 64),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            ConvBlock(32, 32),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            ConvBlock(16, 16),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            ConvBlock(16, 16),
        )

        # Output heads for road and intersection masks
        self.head_road = nn.Conv2d(16, 1, kernel_size=1)
        self.head_inter = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.proj(feat)
        x = self.up1(x)  # 2x
        x = self.up2(x)  # 4x
        x = self.up3(x)  # 8x
        x = self.up4(x)  # 16x -> full image resolution

        road_logit = self.head_road(x)
        inter_logit = self.head_inter(x)

        # [B, 2, H, W]
        logits = torch.cat([road_logit, inter_logit], dim=1)
        return logits

