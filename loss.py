# loss.py
"""
Loss functions for binary segmentation.
"""
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (probs * targets).sum(dim=dims)
        union = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = 1.0 - ((2.0 * inter + self.smooth) / (union + self.smooth)).mean()
        return self.bce_weight * bce + self.dice_weight * dice



# class BCEDiceLoss(nn.Module):
#     """
#     专为水体分割设计的增强损失：
#     - 边界加权 Dice (edge_weight)
#     - 连续性正则化 (continuity_weight)
#     - 动态类别平衡 (dynamic_balance)
#     组合公式：loss = bce_weight * bce + dice_weight * dice_edge + continuity_weight * continuity_loss
#     """
#     def __init__(
#         self,
#         bce_weight: float = 0.5,
#         dice_weight: float = 0.5,
#         smooth: float = 1.0,
#         edge_weight: float = 0.8,          # 边界权重系数 (0~1)
#         continuity_weight: float = 0.1,     # 连续性正则化系数
#         dynamic_balance: bool = True,       # 是否动态平衡 Dice 权重
#         min_dice_weight: float = 0.3,       # 动态平衡时 Dice 权重的最小值
#         max_dice_weight: float = 0.9,       # 动态平衡时 Dice 权重的最大值
#     ):
#         super().__init__()
#         self.bce_weight = bce_weight
#         self.dice_weight = dice_weight
#         self.smooth = smooth
#         self.edge_weight = edge_weight
#         self.continuity_weight = continuity_weight
#         self.dynamic_balance = dynamic_balance
#         self.min_dice_weight = min_dice_weight
#         self.max_dice_weight = max_dice_weight
#         self.bce = nn.BCEWithLogitsLoss(reduction='none')  # 需要逐像素损失用于加权
#
#     def _compute_edge_weight(self, targets: torch.Tensor) -> torch.Tensor:
#         """使用 Sobel 算子生成边界权重图，边缘区域权重大于内部区域。"""
#         # 计算梯度 (sobel)
#         sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=targets.device, dtype=targets.dtype).view(1,1,3,3)
#         sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=targets.device, dtype=targets.dtype).view(1,1,3,3)
#         # 对每个 batch 独立计算边界
#         grad_x = F.conv2d(targets, sobel_x, padding=1)
#         grad_y = F.conv2d(targets, sobel_y, padding=1)
#         edge = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
#         # 归一化到 [0,1] 并二值化 (阈值 0.1)
#         edge = (edge > 0.1).float()
#         # 边缘权重: 边缘区域 = edge_weight, 非边缘区域 = 1
#         weight = 1.0 + (self.edge_weight - 1.0) * edge
#         return weight.detach()  # 不参与梯度
#
#     def _continuity_loss(self, probs: torch.Tensor) -> torch.Tensor:
#         """计算预测概率图的连续性正则化（总变分 TV loss），鼓励相邻像素相似。"""
#         # 计算水平和垂直方向上的差异
#         diff_h = probs[:, :, 1:, :] - probs[:, :, :-1, :]
#         diff_w = probs[:, :, :, 1:] - probs[:, :, :, :-1]
#         loss = torch.abs(diff_h).mean() + torch.abs(diff_w).mean()
#         return loss
#
#     def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         # 1. 基础 BCE（逐像素）
#         bce_loss = self.bce(logits, targets)  # (B,1,H,W)
#
#         # 2. 边界加权 Dice
#         probs = torch.sigmoid(logits)
#         dims = (1, 2, 3)  # 空间维度
#
#         # 生成边界权重图
#         edge_weight_map = self._compute_edge_weight(targets)  # (B,1,H,W)
#
#         # 加权 Dice：分子和分母分别乘以边缘权重（仅在正样本区域加权）
#         # 注意：为了保持 Dice 公式，需要对预测和真值都应用边缘权重
#         inter = (probs * targets * edge_weight_map).sum(dim=dims)
#         union = (probs * edge_weight_map).sum(dim=dims) + (targets * edge_weight_map).sum(dim=dims)
#         dice = 1.0 - ((2.0 * inter + self.smooth) / (union + self.smooth)).mean()
#
#         # 3. 动态调整 Dice 权重（根据水体占比）
#         if self.dynamic_balance:
#             # 计算当前 batch 中水体像素比例
#             water_ratio = targets.mean().item()
#             # 当水体很少时（细长河流），提高 Dice 权重；水体很多时（大片水域），降低 Dice 权重？
#             # 这里取反逻辑：水体少时，让模型更关注前景，所以提高 dice_weight
#             # 映射范围: water_ratio 从 0 -> self.max_dice_weight, 1 -> self.min_dice_weight
#             adjusted_dice_weight = self.max_dice_weight - water_ratio * (self.max_dice_weight - self.min_dice_weight)
#             adjusted_dice_weight = max(self.min_dice_weight, min(self.max_dice_weight, adjusted_dice_weight))
#         else:
#             adjusted_dice_weight = self.dice_weight
#
#         # 4. 连续性正则化（对预测概率图）
#         continuity = self._continuity_loss(probs)
#
#         # 组合损失：BCE 对每个像素加权（边缘权重也可用于 BCE，可选）
#         # 为了简洁，BCE 不使用边缘权重，保持原始
#         final_bce = bce_loss.mean()
#         loss = self.bce_weight * final_bce + adjusted_dice_weight * dice + self.continuity_weight * continuity
#
#         return loss