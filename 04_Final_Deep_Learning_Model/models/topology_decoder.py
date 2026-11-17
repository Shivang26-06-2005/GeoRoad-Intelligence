import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologyDecoder(nn.Module):
    """
    Predicts edge existence between source vertices and candidate neighbor vertices.
    Inputs:
      - src_feats: [B, N_src, D_feat] visual features at source vertices
      - tgt_feats: [B, N_src, N_nbr, D_feat] visual features at target neighbor vertices
      - rel_coords: [B, N_src, N_nbr, 2] relative (dx, dy) offsets
      - mask: [B, N_src, N_nbr] boolean mask for valid neighbors
    Output:
      - edge_logits: [B, N_src, N_nbr] connection probability logits
    """
    def __init__(
        self,
        feat_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        # Project visual embeddings to hidden space
        self.proj_src = nn.Linear(feat_dim, hidden_dim)
        self.proj_tgt = nn.Linear(feat_dim, hidden_dim)

        # Coordinate offset spatial encoding (dx, dy, distance)
        self.coord_embed = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Multi-head Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Binary edge classification head
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        src_feats: torch.Tensor,
        tgt_feats: torch.Tensor,
        rel_coords: torch.Tensor,
        valid_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        src_feats: [B, N_src, C]
        tgt_feats: [B, N_src, N_nbr, C]
        rel_coords: [B, N_src, N_nbr, 2]
        """
        B, N_src, N_nbr, C = tgt_feats.shape

        # Compute Euclidean distance feature
        dist = torch.norm(rel_coords, dim=-1, keepdim=True)  # [B, N_src, N_nbr, 1]
        spatial_in = torch.cat([rel_coords, dist], dim=-1)   # [B, N_src, N_nbr, 3]

        # Embed components
        s_emb = self.proj_src(src_feats).unsqueeze(2)        # [B, N_src, 1, H]
        t_emb = self.proj_tgt(tgt_feats)                    # [B, N_src, N_nbr, H]
        c_emb = self.coord_embed(spatial_in)                # [B, N_src, N_nbr, H]

        # Fused relation tokens
        fused = t_emb + s_emb + c_emb                       # [B, N_src, N_nbr, H]

        # Reshape to run transformer across neighbors for each source point
        # [B * N_src, N_nbr, H]
        fused_flat = fused.view(B * N_src, N_nbr, -1)

        key_padding_mask = None
        if valid_mask is not None:
            # Transformer expects True for invalid/padded positions
            key_padding_mask = ~valid_mask.view(B * N_src, N_nbr)
            # Crucial PyTorch safeguard: if ALL tokens in a row are masked out, unmask them to prevent softmax(all -inf) = NaN
            all_masked = key_padding_mask.all(dim=-1, keepdim=True)
            if all_masked.any():
                key_padding_mask = key_padding_mask & (~all_masked)

        trans_out = self.transformer(fused_flat, src_key_padding_mask=key_padding_mask)

        # Predict edge logits
        logits_flat = self.edge_head(trans_out).squeeze(-1)  # [B * N_src, N_nbr]
        logits = logits_flat.view(B, N_src, N_nbr)
        return logits

