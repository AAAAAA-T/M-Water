# metrics.py
"""
Evaluation metrics (IoU, F1, HD, etc.) and helper classes.
"""
import math
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

from typing import Optional, Union

SCIPY_AVAILABLE = True
from thop import profile

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


# ==================== 原始函数（全局累加，不推荐用于测试集评估） ====================
def confusion_from_logits(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> Tuple[float, float, float, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = (targets > 0.5).float()
    tp = (preds * targets).sum().item()
    fp = (preds * (1.0 - targets)).sum().item()
    fn = ((1.0 - preds) * targets).sum().item()
    tn = ((1.0 - preds) * (1.0 - targets)).sum().item()
    return tp, fp, fn, tn


def metrics_from_confusion(tp: float, fp: float, fn: float, tn: float) -> Dict[str, float]:
    eps = 1e-7
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    acc = (tp + tn) / (tp + fp + fn + tn + eps)
    return {
        "IoU": float(iou),
        "F1": float(dice),
        "Dice": float(dice),
        "Precision": float(precision),
        "Recall": float(recall),
        "Acc": float(acc),
    }


# ==================== 推荐：逐图像计算后取平均（宏观平均） ====================
def per_image_metrics(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> List[Dict[str, float]]:
    """
    对 batch 中的每张图像单独计算指标，返回一个列表，每个元素是一张图像的指标字典。
    这样你可以自行计算均值（宏观平均）或进行其他统计分析。
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = (targets > 0.5).float()
    batch_size = preds.size(0)
    results = []
    eps = 1e-7

    for i in range(batch_size):
        p = preds[i, 0].reshape(-1)   # 假设单通道，若多通道需调整
        t = targets[i, 0].reshape(-1)
        tp = (p * t).sum().item()
        fp = (p * (1 - t)).sum().item()
        fn = ((1 - p) * t).sum().item()
        tn = ((1 - p) * (1 - t)).sum().item()

        iou = tp / (tp + fp + fn + eps)
        f1 = (2 * tp) / (2 * tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        acc = (tp + tn) / (tp + fp + fn + tn + eps)

        results.append({
            "IoU": float(iou),
            "F1": float(f1),
            "Dice": float(f1),  # 别名
            "Precision": float(precision),
            "Recall": float(recall),
            "Acc": float(acc),
        })
    return results


def batch_metrics_macro_average(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> Dict[str, float]:
    """
    计算 batch 内所有图像的宏观平均（先逐图算指标，再平均）。
    这是遥感水体分割推荐的评估方式。
    """
    per_im = per_image_metrics(logits, targets, threshold)
    if not per_im:
        return {}
    avg_dict = {}
    for key in per_im[0].keys():
        values = [d[key] for d in per_im]
        avg_dict[key] = float(np.mean(values))
    return avg_dict


# ==================== Hausdorff 距离（已优化） ====================
# def hausdorff_distance_binary(
#     output, target, use_surface: bool = True, structuring_element: Optional[np.ndarray] = None
# ) -> float:
#     """
#     计算二值掩码的对称 Hausdorff 距离。
#     Args:
#         output: 预测（可为 logits, probs, 或 bool/numpy）
#         target: 真值（类似）
#         use_surface: 若 True，仅使用边界点（常见近似）；若 False，使用全部前景点（严格定义）。
#         structuring_element: 腐蚀的结构元素，默认 8-邻域（np.ones((3,3))）以保留细小结构。
#     """
#     # 转换为 numpy bool
#     if torch.is_tensor(output):
#         output = torch.sigmoid(output).detach().cpu().numpy()
#     if torch.is_tensor(target):
#         target = target.detach().cpu().numpy()
#
#     pred = (output > 0.5).astype(bool)
#     gt = (target > 0.5).astype(bool)
#
#     if not pred.any() and not gt.any():
#         return 0.0
#     if not pred.any() or not gt.any():
#         return np.nan
#
#     if use_surface:
#         if structuring_element is None:
#             structuring_element = np.ones((3, 3))  # 8-邻域，更适合保留细小结构
#         pred_surface = pred ^ binary_erosion(pred, structure=structuring_element)
#         gt_surface = gt ^ binary_erosion(gt, structure=structuring_element)
#         # 如果腐蚀后没有表面点，回退到全集（避免空集）
#         if not pred_surface.any():
#             pred_surface = pred
#         if not gt_surface.any():
#             gt_surface = gt
#     else:
#         pred_surface = pred
#         gt_surface = gt
#
#     dt_gt = distance_transform_edt(~gt_surface)
#     dt_pred = distance_transform_edt(~pred_surface)
#
#     hd1 = dt_gt[pred_surface].max()
#     hd2 = dt_pred[gt_surface].max()
#     return float(max(hd1, hd2))
def _to_bool_array(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """将 torch 张量或 numpy 数组安全地转换为 bool numpy 数组。"""
    if torch.is_tensor(x):
        x = x.detach().cpu()
        # 仅在浮点数时应用 sigmoid（兼容 logits），否则直接阈值
        if x.is_floating_point():
            x = torch.sigmoid(x)
        x = x.numpy()

    # 如果仍是浮点 numpy（非张量情况），应用阈值 0.5
    if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.floating):
        return x > 0.5

    # 其他情况（int, bool）直接转为 bool
    return np.asarray(x).astype(bool)


def hausdorff_distance_binary(
        output: Union[np.ndarray, torch.Tensor],
        target: Union[np.ndarray, torch.Tensor],
        use_surface: bool = True,
        structuring_element: Optional[np.ndarray] = None
) -> float:
    """
    计算二值掩码的对称 HD95（第95百分位 Hausdorff 距离）。

    Args:
        output: 预测值（logits、概率、或二值 np/torch 数组）
        target: 真值（类似）
        use_surface:
            - True（默认）：仅使用边界点（医学影像比赛主流标准，节省算力）
            - False：使用全部前景点（严格数学定义）
        structuring_element: 自定义腐蚀结构元素。若为 None，自动根据输入维度生成：
            2D -> np.ones((3, 3), dtype=bool)，3D -> np.ones((3, 3, 3), dtype=bool)

    Returns:
        HD95 值（float）。若两者均为空返回 0.0，若其一为空返回 np.nan。
    """
    # 1. 统一转为 bool numpy 数组
    pred = _to_bool_array(output)
    gt = _to_bool_array(target)

    # 2. 空集处理
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return np.nan

    # 3. 提取表面（若需要）
    if use_surface:
        # 核心修复：根据输入维度动态生成结构元素（不再硬编码 3x3）
        if structuring_element is None:
            ndim = pred.ndim
            structuring_element = np.ones((3,) * ndim, dtype=bool)

        # 表面 = 原集合 - 腐蚀后的内部集合（异或）
        pred_surface = pred ^ binary_erosion(pred, structure=structuring_element)
        gt_surface = gt ^ binary_erosion(gt, structure=structuring_element)

        # 回退：若腐蚀后无表面点（如细小结构），使用全集以避免空距离
        if not pred_surface.any():
            pred_surface = pred
        if not gt_surface.any():
            gt_surface = gt
    else:
        pred_surface = pred
        gt_surface = gt

    # 4. 距离变换（注意输入取反，计算背景到表面的距离）
    dt_gt = distance_transform_edt(np.logical_not(gt_surface))  # 真实表面距离场
    dt_pred = distance_transform_edt(np.logical_not(pred_surface))  # 预测表面距离场

    # 5. 计算双向第95百分位距离
    hd1 = np.percentile(dt_gt[pred_surface], 95)  # 预测点 → 真实表面
    hd2 = np.percentile(dt_pred[gt_surface], 95)  # 真实点 → 预测表面

    return float(max(hd1, hd2))

def batch_hd_from_logits(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, use_surface: bool = True
) -> float:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    hds = []
    for i in range(probs.shape[0]):
        pred = probs[i, 0] > threshold
        gt = targets_np[i, 0] > 0.5
        hds.append(hausdorff_distance_binary(pred, gt, use_surface=use_surface))
    # 过滤掉 nan（例如两个 mask 有一个为空的情况）
    valid_hds = [h for h in hds if not np.isnan(h)]
    return float(np.mean(valid_hds)) if valid_hds else float('nan')


def compute_flops(model, input_shape, device):
    """
    计算模型的 FLOPs（浮点运算次数）和参数量。
    input_shape: (C, H, W)
    返回 (flops, params)，单位均为 float；若计算失败返回 (None, None)
    """
    if profile is None:
        return None, None

    training = model.training
    model.eval()
    dummy_input = torch.randn(1, *input_shape).to(device)

    try:
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops = macs * 2          # MACs → FLOPs（乘加算两次）
        return flops, params
    except Exception as e:
        print(f"FLOPs calculation failed: {e}")
        return None, None
    finally:
        model.train(training)     # 恢复原有模式