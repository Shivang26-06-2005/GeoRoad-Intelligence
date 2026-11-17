import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam_encoder import SAMImageEncoder
from .geometry_decoder import GeometryDecoder
from .topology_decoder import TopologyDecoder


class RoadNetworkModel(nn.Module):
    """
    Unified Road Network Extraction Model.
    Integrates SAM ViT backbone, Geometry Decoder, and Topology Transformer.
    """
    def __init__(
        self,
        sam_checkpoint: str,
        freeze_encoder_blocks: int = 8,
        img_size: int = 512,
    ):
        super().__init__()
        # 1. SAM Backbone
        self.encoder = SAMImageEncoder(img_size=img_size)
        if sam_checkpoint:
            self.encoder.load_from_checkpoint(sam_checkpoint)
        if freeze_encoder_blocks > 0:
            self.encoder.freeze_early_blocks(freeze_encoder_blocks)

        # 2. Geometry Head
        self.geometry_decoder = GeometryDecoder(in_channels=256, feat_dim=128)

        # 3. Topology Head
        self.topology_decoder = TopologyDecoder(feat_dim=256, hidden_dim=128)

    def extract_features_at_coords(
        self,
        feat_map: torch.Tensor,
        coords: torch.Tensor,
        img_size: int = 512,
    ) -> torch.Tensor:
        """
        Bilinearly samples feature map embeddings at continuous (x, y) coordinates.
        feat_map: [B, C, H_f, W_f]
        coords: [B, N, 2] in pixel units [0, img_size]
        Returns: [B, N, C]
        """
        B, C, H_f, W_f = feat_map.shape
        # Normalize coords to [-1, 1] for grid_sample
        norm_x = (coords[..., 0] / (img_size - 1)) * 2.0 - 1.0
        norm_y = (coords[..., 1] / (img_size - 1)) * 2.0 - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(2)  # [B, N, 1, 2]

        sampled = F.grid_sample(feat_map, grid, align_corners=True, mode="bilinear")
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # [B, N, C]
        return sampled

    def forward_encoder(self, images: torch.Tensor) -> torch.Tensor:
        """Extracts SAM dense feature representation [B, 256, 32, 32]."""
        return self.encoder(images)

    def forward_geometry(self, feat_map: torch.Tensor) -> torch.Tensor:
        """Predicts road and intersection masks [B, 2, 512, 512]."""
        return self.geometry_decoder(feat_map)

    def forward_topology(
        self,
        feat_map: torch.Tensor,
        src_coords: torch.Tensor,
        tgt_coords: torch.Tensor,
        valid_mask: torch.Tensor = None,
        img_size: int = 512,
    ) -> torch.Tensor:
        """
        src_coords: [B, N_src, 2]
        tgt_coords: [B, N_src, N_nbr, 2]
        valid_mask: [B, N_src, N_nbr]
        Returns edge connection logits: [B, N_src, N_nbr]
        """
        B, N_src, N_nbr, _ = tgt_coords.shape

        # Sample source features [B, N_src, C]
        src_feats = self.extract_features_at_coords(feat_map, src_coords, img_size)

        # Sample target features [B, N_src * N_nbr, C] -> reshape
        tgt_flat = tgt_coords.view(B, N_src * N_nbr, 2)
        tgt_feats_flat = self.extract_features_at_coords(feat_map, tgt_flat, img_size)
        tgt_feats = tgt_feats_flat.view(B, N_src, N_nbr, -1)

        # Compute relative coordinate displacements
        rel_coords = tgt_coords - src_coords.unsqueeze(2)

        # Predict connectivity probabilities
        edge_logits = self.topology_decoder(src_feats, tgt_feats, rel_coords, valid_mask)
        return edge_logits

