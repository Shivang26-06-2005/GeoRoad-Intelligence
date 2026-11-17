import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LayerNorm2d(nn.Module):
    """2D LayerNorm over channels."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, C, H, W -> B, H', W', C
        x = self.proj(x)
        return x.permute(0, 2, 3, 1)


class Attention(nn.Module):
    """Multi-head Attention with relative position bias support."""
    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos

        if self.use_rel_pos and input_size is not None:
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
        else:
            self.rel_pos_h = None
            self.rel_pos_w = None

    def get_rel_pos(self, q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
        max_rel_dist = 2 * max(q_size, k_size) - 1
        if rel_pos.shape[0] != max_rel_dist:
            rel_pos_resized = F.interpolate(
                rel_pos.reshape(1, 1, -1, rel_pos.shape[-1]).permute(0, 3, 1, 2),
                size=(1, max_rel_dist),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0).squeeze(1).permute(1, 0)
        else:
            rel_pos_resized = rel_pos

        q_coords = torch.arange(q_size, device=rel_pos.device)[:, None]
        k_coords = torch.arange(k_size, device=rel_pos.device)[None, :]
        relative_coords = (q_coords - k_coords) + (k_size - 1)
        return rel_pos_resized[relative_coords.long()]

    def add_decomposed_rel_pos(
        self,
        attn: torch.Tensor,
        q: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
    ) -> torch.Tensor:
        q_h, q_w = q_size
        k_h, k_w = k_size
        Rh = self.get_rel_pos(q_h, k_h, rel_pos_h)
        Rw = self.get_rel_pos(q_w, k_w, rel_pos_w)

        B, _, dim = q.shape
        r_q = q.reshape(B, q_h, q_w, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

        attn = (
            attn.view(B, q_h, q_w, k_h, k_w)
            + rel_h[:, :, :, :, None]
            + rel_w[:, :, :, None, :]
        ).view(B, q_h * q_w, k_h * k_w)
        return attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos and self.rel_pos_h is not None and self.rel_pos_w is not None:
            attn = self.add_decomposed_rel_pos(
                attn.reshape(B * self.num_heads, H * W, H * W),
                q.reshape(B * self.num_heads, H * W, -1),
                self.rel_pos_h,
                self.rel_pos_w,
                (H, W),
                (H, W),
            ).view(B, self.num_heads, H * W, H * W)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, H, W, -1)
        return self.proj(out)


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, act: nn.Module = nn.GELU):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class Block(nn.Module):
    """Transformer Block with optional window partitioning."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_rel_pos: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLPBlock(dim, int(dim * mlp_ratio))
        self.window_size = window_size

    def window_partition(self, x: torch.Tensor, window_size: int):
        B, H, W, C = x.shape
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows, (Hp, Wp)

    def window_unpartition(self, windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]):
        Hp, Wp = pad_hw
        H, W = hw
        B = windows.shape[0] // ((Hp // window_size) * (Wp // window_size))
        x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
        if Hp > H or Wp > W:
            x = x[:, :H, :W, :].contiguous()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = self.window_partition(x, self.window_size)
            x = self.attn(x)
            x = self.window_unpartition(x, self.window_size, pad_hw, (H, W))
        else:
            x = self.attn(x)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class SAMImageEncoder(nn.Module):
    """
    Self-contained SAM ViT-B image encoder.
    Directly loads official SAM checkpoint weights without needing external packages.
    """
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        use_rel_pos: bool = True,
        window_size: int = 14,
        global_attn_indexes: Tuple[int, ...] = (2, 5, 8, 11),
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # ViT-B standard SAM grid is 64x64 patches (1024x1024)
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, 64, embed_dim))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                use_rel_pos=use_rel_pos,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(64, 64),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def load_from_checkpoint(self, checkpoint_path: str):
        """Loads weights from the official SAM .pth checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith("image_encoder."):
                sub_key = k[len("image_encoder."):]
                encoder_state[sub_key] = v

        missing, unexpected = self.load_state_dict(encoder_state, strict=True)
        print(f"Loaded SAM Encoder weights: missing={len(missing)}, unexpected={len(unexpected)}")

    def freeze_early_blocks(self, num_blocks_to_freeze: int = 8):
        """Freezes first N ViT blocks to drastically reduce VRAM usage on GTX 1650."""
        self.patch_embed.eval()
        for p in self.patch_embed.parameters():
            p.requires_grad = False
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False

        for i in range(min(num_blocks_to_freeze, len(self.blocks))):
            self.blocks[i].eval()
            for p in self.blocks[i].parameters():
                p.requires_grad = False
        print(f"Frozen patch embedding and first {num_blocks_to_freeze} ViT blocks.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            H, W = x.shape[1], x.shape[2]
            pos_embed = F.interpolate(
                self.pos_embed.permute(0, 3, 1, 2),
                size=(H, W),
                mode="bicubic",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            x = x + pos_embed

        for blk in self.blocks:
            x = blk(x)

        # Neck expects [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        out = self.neck(x)
        return out
