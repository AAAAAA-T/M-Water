# arch.py
"""
Model architecture: paper-inspired MHNet.
"""
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = ConvBNAct(ch, ch, 3, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv2(self.conv1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias



class MSMamba(nn.Module):#MS-Mamba
    """
    Multi-Scale Mamba Scan Block
    (H, H/2, H/4 feature modeling)
    """

    def __init__(self, dim, d_state=16, d_conv=3, expand=2.0):
        super().__init__()

        self.dim = dim

        # shared norm
        self.norm = nn.LayerNorm(dim)

        # spatial bias (local inductive bias)
        self.spatial_bias = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        # multi-scale Mamba branches
        # self.mamba_full = Mamba(dim, d_state=d_state, d_conv=d_conv, expand=expand,bimamba_type=None)
        # self.mamba_half = Mamba(dim, d_state=d_state, d_conv=d_conv, expand=expand,bimamba_type=None)
        # self.mamba_quarter = Mamba(dim, d_state=d_state, d_conv=d_conv, expand=expand,bimamba_type=None)


        self.mamba_full = Mamba(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_half = Mamba(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_quarter = Mamba(dim, d_state=d_state, d_conv=d_conv, expand=expand)

        # scale fusion gate
        self.scale_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 3),  # 3 scales
            nn.Softmax(dim=-1)
        )

        # residual gating (IOU booster)
        self.res_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def _to_seq(self, x):
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2), (H, W)

    def _from_seq(self, x, shape):
        B, L, C = x.shape
        H, W = shape
        return x.transpose(1, 2).reshape(B, C, H, W)

    def forward(self, x):
        B, C, H, W = x.shape

        # --- local bias ---
        x = x + self.spatial_bias(x)

        # =========================
        # scale 1: full resolution
        # =========================
        x_full, shape_full = self._to_seq(x)
        x_full = self.norm(x_full)
        out_full = self.mamba_full(x_full)

        # =========================
        # scale 2: H/2
        # =========================
        x_half = F.adaptive_avg_pool2d(x, (H // 2, W // 2))
        x_half, shape_half = self._to_seq(x_half)
        x_half = self.norm(x_half)
        out_half = self.mamba_half(x_half)
        out_half = F.interpolate(
            self._from_seq(out_half, shape_half),
            size=(H, W),
            mode="bilinear",
            align_corners=False
        ).flatten(2).transpose(1, 2)

        # =========================
        # scale 3: H/4
        # =========================
        x_quarter = F.adaptive_avg_pool2d(x, (H // 4, W // 4))
        x_quarter, shape_quarter = self._to_seq(x_quarter)
        x_quarter = self.norm(x_quarter)
        out_quarter = self.mamba_quarter(x_quarter)
        out_quarter = F.interpolate(
            self._from_seq(out_quarter, shape_quarter),
            size=(H, W),
            mode="bilinear",
            align_corners=False
        ).flatten(2).transpose(1, 2)

        # =========================
        # adaptive fusion
        # =========================
        gate = self.scale_gate(x_full.mean(dim=1))  # (B, 3)

        out = (
            gate[:, 0:1].unsqueeze(-1) * out_full +
            gate[:, 1:2].unsqueeze(-1) * out_half +
            gate[:, 2:3].unsqueeze(-1) * out_quarter
        )

        # =========================
        # residual gating (IOU boost)
        # =========================
        res_g = self.res_gate(x_full)
        out = x_full + res_g * out

        # reshape back
        out = self._from_seq(out, (H, W))

        return out


class FeatureMask(nn.Module):# FeatureMask1
    """
    Structured feature masking for CNN feature maps.

    改进点：
    1) 按样本“实际保留比例”做重标定，而不是固定用 1/(1-mask_ratio)
    2) 对小特征图自动约束 block_size，避免 mask 过猛
    3) 支持训练过程中的 mask_ratio 调度
    4) 保持按空间块共享到所有通道，适合 CNN / segmentation feature regularization
    """
    def __init__(
        self,
        mask_ratio: float = 0.10,
        block_size: int = 8,
        min_keep_ratio: float = 0.05,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.base_mask_ratio = float(mask_ratio)
        self.mask_ratio = float(mask_ratio)   # 可外部动态调度
        self.block_size = int(block_size)
        self.min_keep_ratio = float(min_keep_ratio)
        self.eps = float(eps)

    def set_mask_ratio(self, mask_ratio: float):
        self.mask_ratio = float(mask_ratio)

    def reset_mask_ratio(self):
        self.mask_ratio = self.base_mask_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.mask_ratio <= 0.0:
            return x
        b, c, h, w = x.shape

        # ---- 小图保护：block 不超过当前空间尺寸 ----
        bs = max(1, min(self.block_size, h, w))

        gh = max(1, math.ceil(h / bs))
        gw = max(1, math.ceil(w / bs))

        # ---- 生成块级 mask：shape [B,1,gh,gw]，所有通道共享 ----
        keep = (torch.rand(b, 1, gh, gw, device=x.device) > self.mask_ratio).to(dtype=x.dtype)

        # ---- 上采样到原图尺寸 ----
        keep = F.interpolate(keep, size=(h, w), mode="nearest")

        # ---- 实际保留率重标定（逐样本）----
        actual_keep = keep.mean(dim=(1, 2, 3), keepdim=True).clamp_min(self.min_keep_ratio)
        keep = keep / actual_keep

        return x * keep


class MCFF(nn.Module):#AC-MCFF
    """
    Adaptive Cross-scale Fusion MCFF (AC-MCFF)

    Improvements over vanilla MCFF:
    1. pixel-wise scale weighting
    2. cross-scale interaction
    3. spatial + channel joint fusion
    4. residual calibration for stability
    """

    def __init__(self, in_channels: List[int], out_channels: int, reduction: int = 8):
        super().__init__()

        self.num_scales = len(in_channels)

        # --- 1. projection per scale ---
        self.proj = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1, bias=False)
            for c in in_channels
        ])

        # --- 2. cross-scale interaction ---
        self.cross_scale = nn.Sequential(
            nn.Conv2d(out_channels * self.num_scales, out_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False)
        )

        # --- 3. pixel-wise scale attention ---
        self.scale_attn = nn.Sequential(
            nn.Conv2d(out_channels, self.num_scales, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # --- 4. channel attention (lightweight SE) ---
        mid = max(out_channels // reduction, 4)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, mid, 1),
            nn.GELU(),
            nn.Conv2d(mid, out_channels, 1),
            nn.Sigmoid()
        )

        # --- 5. spatial refinement ---
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.GELU()
        )

    def forward(self, feats: List[torch.Tensor]):
        target_size = feats[-1].shape[-2:]

        # =========================
        # 1. projection + alignment
        # =========================
        aligned = []
        for feat, proj in zip(feats, self.proj):
            x = proj(feat)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            aligned.append(x)

        # stack: (B, S, C, H, W)
        stack = torch.stack(aligned, dim=1)

        # =========================
        # 2. cross-scale interaction
        # =========================
        fused = self.cross_scale(torch.cat(aligned, dim=1))

        # =========================
        # 3. pixel-wise scale weights
        # =========================
        attn = self.scale_attn(fused)  # (B, S, H, W)
        attn = attn.unsqueeze(2)       # (B, S, 1, H, W)

        # weighted sum across scales
        out = (stack * attn).sum(dim=1)

        # =========================
        # 4. spatial refinement
        # =========================
        out = out + self.spatial_refine(out)

        # =========================
        # 5. channel attention
        # =========================
        out = out * self.channel_attn(out)

        return out


class DecoderBlock(nn.Module):#DecoderBlock1
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, reduction: int = 16):
        super().__init__()
        mid = max(out_ch // reduction, 4)

        # 轻量 shortcut：保留 decoder 主干语义
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        # 原有主分支基本不动
        self.conv = nn.Sequential(
            ConvBNAct(in_ch + skip_ch, out_ch, 3, 1),
            ResidualConvBlock(out_ch),
        )

        # 轻量 SE 通道重标定
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, mid, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid, out_ch, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        identity = self.shortcut(x)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        x = x * self.attn(x)

        return x + identity

class FusionBlock(nn.Module):#FusionBlock1
    """
    轻量自适应融合：
    1) 全局单标量 -> 通道级先验权重
    2) 根据当前样本自适应调整每个通道的融合比例
    3) 初始行为接近原版，训练更稳
    """
    def __init__(self, in_channels: int, reduction: int = 8, init_alpha: float = 0.5):
        super().__init__()
        mid = max(in_channels // reduction, 4)

        # 通道级先验：比单一标量更细
        self.alpha = nn.Parameter(torch.full((1, in_channels, 1, 1), init_alpha))

        # 用 x_orig / x_mask / |x_orig - x_mask| 生成自适应通道门控
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 3, mid, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid, in_channels, kernel_size=1, bias=True),
        )

        # 初始时 delta≈0，使初始行为接近原版
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, x_orig: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(x_orig - x_mask)
        delta = self.gate(torch.cat([x_orig, x_mask, diff], dim=1))  # (B,C,1,1)

        # 通道级、样本自适应权重
        w = torch.sigmoid(self.alpha + delta)

        return w * x_orig + (1.0 - w) * x_mask


class MWater(nn.Module):#M³Water
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        mask_ratio: float = 0.30,
        mask_block_size: int = 8,
        heads: Tuple[int, int, int, int] = (1, 2, 4, 8),
        blocks: Tuple[int, int, int, int] = (1, 1, 1, 1),
    ):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.stem = nn.Sequential(
            ConvBNAct(in_channels, c1, 3, 1),
            ResidualConvBlock(c1),
        )
        self.stage1 = self._make_stage(c1, blocks[0], heads[0])
        self.down1 = ConvBNAct(c1, c2, 3, 2)
        self.stage2 = self._make_stage(c2, blocks[1], heads[1])
        self.down2 = ConvBNAct(c2, c3, 3, 2)
        self.stage3 = self._make_stage(c3, blocks[2], heads[2])
        self.down3 = ConvBNAct(c3, c4, 3, 2)
        self.stage4 = self._make_stage(c4, blocks[3], heads[3])

        self.mask1 = FeatureMask(mask_ratio=mask_ratio, block_size=mask_block_size)
        self.mask2 = FeatureMask(mask_ratio=mask_ratio, block_size=mask_block_size)
        self.mask3 = FeatureMask(mask_ratio=mask_ratio, block_size=mask_block_size)
        self.mask4 = FeatureMask(mask_ratio=mask_ratio, block_size=mask_block_size)

        self.fuse1 = FusionBlock(c1)
        self.fuse2 = FusionBlock(c2)
        self.fuse3 = FusionBlock(c3)
        self.fuse4 = FusionBlock(c4)

        self.mcff = MCFF([c1, c2, c3, c4], c4)
        self.dec3 = DecoderBlock(c4, c3, c3)
        self.dec2 = DecoderBlock(c3, c2, c2)
        self.dec1 = DecoderBlock(c2, c1, c1)
        self.out_head = nn.Sequential(
            ConvBNAct(c1, c1, 3, 1),
            nn.Conv2d(c1, num_classes, kernel_size=1),
        )

    @staticmethod
    def _make_stage(ch: int, n_blocks: int, n_heads: int) -> nn.Sequential:
        layers: List[nn.Module] = [ResidualConvBlock(ch)]
        for _ in range(n_blocks):
            layers.append(MSMamba(dim=ch))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        # ---- 编码 ----
        f1_raw = self.stage1(self.stem(x))
        f2_raw = self.stage2(self.down1(f1_raw))
        f3_raw = self.stage3(self.down2(f2_raw))
        f4_raw = self.stage4(self.down3(f3_raw))

        # ---- 生成掩码特征（训练时随机遮挡，推理时原样） ----
        f1_mask = self.mask1(f1_raw)
        f2_mask = self.mask2(f2_raw)
        f3_mask = self.mask3(f3_raw)
        f4_mask = self.mask4(f4_raw)

        # ---- 融合原始特征与掩码特征 ----
        f1 = self.fuse1(f1_raw, f1_mask)  # 用于 dec1 跳跃连接
        f2 = self.fuse2(f2_raw, f2_mask)  # 用于 dec2 跳跃连接
        f3 = self.fuse3(f3_raw, f3_mask)  # 用于 dec3 跳跃连接
        f4 = self.fuse4(f4_raw, f4_mask)  # 用于 MCFF

        # ---- 解码 ----
        x = self.mcff([f1, f2, f3, f4])
        x = self.dec3(x, f3)
        x = self.dec2(x, f2)
        x = self.dec1(x, f1)


        logits = self.out_head(x)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits

