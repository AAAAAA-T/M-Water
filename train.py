# train_test.py
"""
Main training, validation, and testing loops.
"""
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import math
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# TensorBoard (optional)
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
    print("TensorBoard not installed. Install with: pip install tensorboard")

# Avoid OpenMP over-subscription on CPU-only debugging
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
cv2.setNumThreads(0)

# Local modules
from model import MWater
from Dateset import (
    WaterSegDataset,
    SimpleSegTransform,
    read_split_file,
    ensure_dir,
)
from loss import BCEDiceLoss
from Metrics import (
    AverageMeter,
    confusion_from_logits,
    metrics_from_confusion,
    batch_hd_from_logits,
    SCIPY_AVAILABLE,
)


# ---------- auxiliary ----------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def save_args(args: argparse.Namespace, save_dir: Path) -> None:
    ensure_dir(save_dir)
    with (save_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)


def append_csv(path: Path, row: Dict[str, float], header: Optional[Iterable[str]] = None) -> None:
    ensure_dir(path.parent)
    write_header = not path.exists()
    keys = list(header) if header is not None else list(row.keys())
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in keys})


# ---------- building loaders ----------
def build_loaders(args: argparse.Namespace):
    root = Path(args.root) / args.dataset
    img_dir = root / "images"
    mask_dir = root / "masks" / "0"

    train_loader = val_loader = test_loader = None

    if (root / "train.txt").exists():
        train_ids = read_split_file(root / "train.txt")
        train_ds = WaterSegDataset(
            train_ids, img_dir, mask_dir, args.img_ext, args.mask_ext, args.input_channels,
            SimpleSegTransform(args.input_h, args.input_w, args.input_channels, mode="train", aug=not args.no_aug),
            mode="train",
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
            pin_memory=True, drop_last=True,
        )

    if (root / "val.txt").exists():
        val_ids = read_split_file(root / "val.txt")
        val_ds = WaterSegDataset(
            val_ids, img_dir, mask_dir, args.img_ext, args.mask_ext, args.input_channels,
            SimpleSegTransform(args.input_h, args.input_w, args.input_channels, mode="val", aug=False),
            mode="val",
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
            pin_memory=True, drop_last=False,
        )

    if (root / "test.txt").exists():
        test_ids = read_split_file(root / "test.txt")
        test_ds = WaterSegDataset(
            test_ids, img_dir, mask_dir, args.img_ext, args.mask_ext, args.input_channels,
            SimpleSegTransform(args.input_h, args.input_w, args.input_channels, mode="test", aug=False),
            mode="test",
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
            pin_memory=True, drop_last=False,
        )

    return train_loader, val_loader, test_loader


def create_model(args: argparse.Namespace) -> nn.Module:
    return MWater(
        in_channels=args.input_channels,
        num_classes=1,
        base_channels=args.base_channels,
        mask_ratio=args.mask_ratio,
        mask_block_size=args.mask_block_size,
        heads=(1, 2, 4, 8),
        blocks=(args.blocks1, args.blocks2, args.blocks3, args.blocks4),
    )


# ---------- training utilities ----------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    tp = fp = fn = tn = 0.0

    pbar = tqdm(loader, desc=f"Train {epoch}", ncols=120)
    for images, masks, _ in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp):
            logits = model(images)
            loss = criterion(logits, masks)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected at epoch {epoch}: {loss.item()}")

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), images.size(0))
        c = confusion_from_logits(logits.detach(), masks, threshold=0.5)
        tp += c[0]; fp += c[1]; fn += c[2]; tn += c[3]
        now_metrics = metrics_from_confusion(tp, fp, fn, tn)
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", IoU=f"{now_metrics['IoU']:.4f}")

    metrics = metrics_from_confusion(tp, fp, fn, tn)
    metrics["loss"] = loss_meter.avg
    return metrics


def save_predictions(
    logits: torch.Tensor,
    img_ids: List[str],
    save_dir: Path,
    threshold: float = 0.5,
):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    for i in range(probs.shape[0]):
        pred = (probs[i, 0] > threshold).astype(np.uint8) * 255
        save_path = save_dir / f"{img_ids[i]}.png"
        cv2.imwrite(str(save_path), pred)


def save_comparison(
    images: torch.Tensor,
    masks: torch.Tensor,
    logits: torch.Tensor,
    img_ids: List[str],
    save_dir: Path,
    threshold: float = 0.5,
):
    """
    Save side-by-side comparison images: original image | ground truth | prediction.
    Assumes images are in [0,1] range (or will be scaled appropriately).
    """
    probs = torch.sigmoid(logits).detach().cpu().numpy()  # (B, 1, H, W)
    images_np = images.cpu().numpy()                      # (B, C, H, W)
    masks_np = masks.cpu().numpy()                       # (B, 1, H, W)

    for i in range(images_np.shape[0]):
        # ---------- original image ----------
        img = images_np[i].transpose(1, 2, 0)  # (H, W, C)
        # Heuristic: if values in [0,1] scale to uint8; else try linear stretch
        if img.max() <= 1.0 and img.min() >= 0.0:
            img = (img * 255).astype(np.uint8)
        else:
            # Assume values are normalized (e.g., ImageNet), stretch to 0-255 for visualization
            img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        elif img.shape[-1] > 3:
            img = img[:, :, :3]  # take first three channels

        # ---------- ground truth ----------
        gt = (masks_np[i, 0] > 0.5).astype(np.uint8) * 255
        gt = np.stack([gt, gt, gt], axis=-1)  # to 3-channel

        # ---------- prediction ----------
        pred = (probs[i, 0] > threshold).astype(np.uint8) * 255
        pred = np.stack([pred, pred, pred], axis=-1)

        # ---------- concatenate horizontally ----------
        compare = np.concatenate([img, gt, pred], axis=1)
        save_path = save_dir / f"{img_ids[i]}.png"
        cv2.imwrite(str(save_path), compare)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    calc_hd: bool = False,
    amp: bool = False,
) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    tp = fp = fn = tn = 0.0
    hd_meter = AverageMeter()

    pbar = tqdm(loader, desc="Eval", ncols=120)
    for images, masks, _ in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with autocast(enabled=amp):
            logits = model(images)
            loss = criterion(logits, masks)
        loss_meter.update(loss.item(), images.size(0))
        c = confusion_from_logits(logits, masks, threshold=threshold)
        tp += c[0]; fp += c[1]; fn += c[2]; tn += c[3]
        if calc_hd:
            hd = batch_hd_from_logits(logits, masks, threshold=threshold)
            if math.isfinite(hd):
                hd_meter.update(hd, images.size(0))

    metrics = metrics_from_confusion(tp, fp, fn, tn)
    metrics["loss"] = loss_meter.avg
    if calc_hd:
        metrics["HD"] = hd_meter.avg if hd_meter.count > 0 else float("nan")
    return metrics


@torch.no_grad()
def find_best_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: np.ndarray,
    amp: bool = False,
) -> Tuple[float, float]:
    model.eval()
    stats = {float(th): [0.0, 0.0, 0.0, 0.0] for th in thresholds}

    pbar = tqdm(loader, desc="Find best threshold", ncols=120)
    for images, masks, _ in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with autocast(enabled=amp):
            logits = model(images)
        for th in thresholds:
            c = confusion_from_logits(logits, masks, threshold=float(th))
            s = stats[float(th)]
            s[0] += c[0]; s[1] += c[1]; s[2] += c[2]; s[3] += c[3]

    best_th = 0.5
    best_iou = -1.0
    for th, s in stats.items():
        iou = metrics_from_confusion(s[0], s[1], s[2], s[3])["IoU"]
        if iou > best_iou:
            best_iou = iou
            best_th = th
    return float(best_th), float(best_iou)


def load_checkpoint(model: nn.Module, resume: str, device: torch.device) -> Dict:
    if not resume:
        raise RuntimeError("--resume is required for val/test mode.")
    path = Path(resume)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find checkpoint: {path}")
    ckpt = torch.load(str(path), map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
        return ckpt
    model.load_state_dict(ckpt, strict=True)
    return {"model": ckpt}


def resolve_threshold(args: argparse.Namespace, ckpt: Dict) -> float:
    if args.threshold is not None and args.force_threshold:
        return float(args.threshold)
    if isinstance(ckpt, dict) and "threshold" in ckpt:
        return float(ckpt["threshold"])
    return float(args.threshold if args.threshold is not None else 0.5)


# ---------- training entry ----------
def run_train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)
    save_args(args, save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader, val_loader, _ = build_loaders(args)
    if train_loader is None:
        raise RuntimeError("train.txt is required for training.")
    if val_loader is None:
        raise RuntimeError("val.txt is required for training and threshold selection.")

    model = create_model(args).to(device)

    # print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total parameters: {total_params:,} ({total_params/1e6:.2f}M)  (trainable: {trainable_params:,} ({trainable_params/1e6:.2f}M))")

    criterion = BCEDiceLoss(args.bce_weight, args.dice_weight)


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # if args.optimizer == "adamw":
    #     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # else:  # SGD
    #     optimizer = torch.optim.SGD(model.parameters(), lr=args.lr * 10, momentum=0.9, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = GradScaler(enabled=args.amp)

    start_epoch = 1
    best_iou = -1.0
    best_th = 0.5

    if args.resume:
        ckpt = load_checkpoint(model, args.resume, device)
        if isinstance(ckpt, dict):
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_iou = float(ckpt.get("best_iou", -1.0))
            best_th = float(ckpt.get("threshold", 0.5))

    print(f"[Device] {device}")
    print(f"[Model] MHNetInspired | base_channels={args.base_channels} | mask_ratio={args.mask_ratio}")
    print(f"[AMP] {args.amp}")

    # TensorBoard writer
    writer = None
    if SummaryWriter is not None and not args.no_tensorboard:
        writer = SummaryWriter(log_dir=str(save_dir / "tensorboard"))
        writer.add_text("Hyperparameters", json.dumps(vars(args), indent=2), 0)

    log_header = [
        "epoch", "lr", "train_loss", "train_IoU", "train_F1", "val_loss",
        "val_IoU", "val_F1", "val_Dice", "val_Precision", "val_Recall", "val_Acc", "best_th"
    ]

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_log = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, args)

        if args.search_threshold:
            thresholds = np.arange(args.th_min, args.th_max + 1e-9, args.th_step)
            epoch_th, _ = find_best_threshold(model, val_loader, device, thresholds, amp=args.amp)
        else:
            epoch_th = args.threshold

        val_log = evaluate(model, val_loader, criterion, device, threshold=epoch_th, calc_hd=False, amp=args.amp)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_log["loss"],
            "train_IoU": train_log["IoU"],
            "train_F1": train_log["F1"],
            "val_loss": val_log["loss"],
            "val_IoU": val_log["IoU"],
            "val_F1": val_log["F1"],
            "val_Dice": val_log["Dice"],
            "val_Precision": val_log["Precision"],
            "val_Recall": val_log["Recall"],
            "val_Acc": val_log["Acc"],
            "best_th": epoch_th,
        }
        append_csv(save_dir / "log.csv", row, header=log_header)

        if writer is not None:
            writer.add_scalar("Loss/train", train_log["loss"], epoch)
            writer.add_scalar("Loss/val", val_log["loss"], epoch)
            writer.add_scalar("IoU/train", train_log["IoU"], epoch)
            writer.add_scalar("IoU/val", val_log["IoU"], epoch)
            writer.add_scalar("F1/train", train_log["F1"], epoch)
            writer.add_scalar("F1/val", val_log["F1"], epoch)
            writer.add_scalar("Dice/val", val_log["Dice"], epoch)
            writer.add_scalar("Precision/val", val_log["Precision"], epoch)
            writer.add_scalar("Recall/val", val_log["Recall"], epoch)
            writer.add_scalar("Acc/val", val_log["Acc"], epoch)
            writer.add_scalar("LR", lr_now, epoch)
            writer.add_scalar("Threshold/epoch", epoch_th, epoch)

        print(
            f"Epoch [{epoch}/{args.epochs}] | lr={lr_now:.6g} | "
            f"train_loss={train_log['loss']:.4f} train_IoU={train_log['IoU']:.4f} | "
            f"val_loss={val_log['loss']:.4f} val_IoU={val_log['IoU']:.4f} val_F1={val_log['F1']:.4f} | "
            f"th={epoch_th:.2f} | time={time.time() - t0:.1f}s"
        )

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_iou": best_iou,
            "threshold": best_th,
            "args": vars(args),
        }
        torch.save(ckpt, save_dir / "last.pth")

        if val_log["IoU"] > best_iou:
            best_iou = val_log["IoU"]
            best_th = epoch_th
            ckpt["best_iou"] = best_iou
            ckpt["threshold"] = best_th
            torch.save(ckpt, save_dir / "model.pth")
            print(f"=> Saved best model: epoch={epoch}, IoU={best_iou:.4f}, threshold={best_th:.2f}")

    if writer is not None:
        writer.close()
        print(f"TensorBoard logs saved to: {save_dir / 'tensorboard'}")


# ---------- evaluation or test entry (modified) ----------
def run_eval_or_test(args: argparse.Namespace, mode: str = "test") -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    _, val_loader, test_loader = build_loaders(args)
    loader = test_loader if mode == "test" else val_loader
    if loader is None:
        raise RuntimeError(f"{mode}.txt is required for {mode} mode.")

    model = create_model(args).to(device)

    # Directories for predictions and example comparisons
    pred_dir = Path(args.save_dir) / f"{mode}_predictions"
    examples_dir = pred_dir / "compare_examples"   # 存放随机挑选的示例对比图
    ensure_dir(pred_dir)  # 即使不保存预测图，目录也会创建，但不会有文件写入

    ckpt = load_checkpoint(model, args.resume, device)
    threshold = resolve_threshold(args, ckpt)
    criterion = BCEDiceLoss(args.bce_weight, args.dice_weight)

    print(f"[Device] {device}")
    print(f"[Model] loaded from: {args.resume}")
    print(f"[Threshold] {threshold:.2f}")
    print(f"[HD] calc_hd={args.calc_hd} | scipy_available={SCIPY_AVAILABLE}")
    print(f"[Save predictions] {args.save_pred}")      # 新增：显示是否保存预测掩码
    print(f"[Save comparison images] {args.save_compare}")

    model.eval()
    loss_meter = AverageMeter()
    tp = fp = fn = tn = 0.0
    hd_meter = AverageMeter()

    total_inference_time = 0.0
    total_samples = 0

    # ---------- 随机选择要保存示例的索引 ----------
    if args.save_compare:
        dataset_len = len(loader.dataset)
        num_examples = min(args.num_examples, dataset_len)
        selected_indices = set(random.sample(range(dataset_len), num_examples))
        ensure_dir(examples_dir)
        print(f"[Save examples] Will save {num_examples} comparison images to {examples_dir}")
    else:
        selected_indices = set()

    global_sample_idx = 0  # 全局样本计数器

    # ========== 关键修复：使用 torch.no_grad() 禁用梯度计算 ==========
    with torch.no_grad():
        pbar = tqdm(loader, desc=mode.upper(), ncols=120)
        for images, masks, img_ids in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # measure forward time
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()

            with autocast(enabled=args.amp):
                logits = model(images)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()

            batch_time = end_time - start_time
            total_inference_time += batch_time
            total_samples += images.size(0)

            loss = criterion(logits, masks)
            loss_meter.update(loss.item(), images.size(0))
            c = confusion_from_logits(logits, masks, threshold=threshold)
            tp += c[0]; fp += c[1]; fn += c[2]; tn += c[3]

            if args.calc_hd:
                hd = batch_hd_from_logits(logits, masks, threshold=threshold)
                if math.isfinite(hd):
                    hd_meter.update(hd, images.size(0))

            # ---------- 根据参数决定是否保存所有预测掩码 ----------
            if args.save_pred:
                save_predictions(logits, img_ids, pred_dir, threshold)

            # ---------- 仅当当前样本在选中列表中，才保存对比图 ----------
            if args.save_compare:
                # 对 batch 中的每个样本判断
                for i in range(len(img_ids)):
                    if global_sample_idx + i in selected_indices:
                        # 构造单样本张量并保存
                        single_images = images[i:i+1]
                        single_masks = masks[i:i+1]
                        single_logits = logits[i:i+1]
                        single_ids = [img_ids[i]]
                        save_comparison(single_images, single_masks, single_logits, single_ids, examples_dir, threshold)
                global_sample_idx += len(img_ids)

    avg_inference_time = total_inference_time / total_samples if total_samples > 0 else 0.0
    print(f"\n[{mode.upper()} Inference Time] Average per sample: {avg_inference_time:.6f} s  (total {total_inference_time:.2f}s for {total_samples} samples)")

    metrics = metrics_from_confusion(tp, fp, fn, tn)
    metrics["loss"] = loss_meter.avg
    if args.calc_hd:
        metrics["HD"] = hd_meter.avg if hd_meter.count > 0 else float("nan")

    print(f"\n[{mode.upper()} RESULTS]")
    for k in ["loss", "IoU", "F1", "Dice", "Precision", "Recall", "Acc", "HD"]:
        if k in metrics:
            print(f"{k}: {metrics[k]:.6f}")

    out_dir = Path(args.save_dir) if args.save_dir else Path(args.resume).parent
    ensure_dir(out_dir)
    out_json = out_dir / f"{mode}_metrics.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to: {out_json}")


# ---------- CLI ----------
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MHNet")

    parser.add_argument("--mode", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--root", type=str, default="inputs")
    parser.add_argument('--dataset', default='kaggle')
    parser.add_argument("--img_ext", type=str, default=".png")
    parser.add_argument("--mask_ext", type=str, default=".png")
    parser.add_argument("--save_dir", type=str, default="models/kaggle(29)")#新架构1，新mamba5,,新FeatureMask1,#FusionBlock1,##DecoderBlock1
    parser.add_argument("--resume", type=str, default="models/kaggle(29)/model.pth")
    # parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--input_channels", type=int, default=3)
    parser.add_argument("--input_h", type=int, default=256)
    parser.add_argument("--input_w", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--blocks1", type=int, default=1)
    parser.add_argument("--blocks2", type=int, default=1)
    parser.add_argument("--blocks3", type=int, default=1)
    parser.add_argument("--blocks4", type=int, default=1)
    parser.add_argument("--mask_ratio", type=float, default=0.30)
    parser.add_argument("--mask_block_size", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=5e-6)
    parser.add_argument("--bce_weight", type=float, default=0.5)
    parser.add_argument("--dice_weight", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--force_threshold", action="store_true", help="Use --threshold even if checkpoint stores a best threshold")
    parser.add_argument("--search_threshold", action="store_true", default=False)
    parser.add_argument("--no_search_threshold", dest="search_threshold", action="store_false",default=True)

    # parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])

    parser.add_argument("--th_min", type=float, default=0.10)
    parser.add_argument("--th_max", type=float, default=0.90)
    parser.add_argument("--th_step", type=float, default=0.05)

    parser.add_argument("--calc_hd", action="store_true", default=True)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # TensorBoard toggle
    parser.add_argument("--no_tensorboard", action="store_true", help="Disable TensorBoard logging")

    # ==================== 控制是否保存预测掩码 ====================
    parser.add_argument("--save_pred", action="store_true", default=False,
                        help="Save all predicted masks (default: True)")
    parser.add_argument("--no_save_pred", dest="save_pred", action="store_false",
                        help="Disable saving all predicted masks")

    # ==================== 控制是否保存对比示例图 ====================
    parser.add_argument("--save_compare", action="store_true", default=True,
                        help="Save a few random comparison images (number controlled by --num_examples)")
    parser.add_argument("--no_save_compare", dest="save_compare", action="store_false",
                        help="Disable saving any comparison images")
    parser.add_argument("--num_examples", type=int, default=4,
                        help="Number of random examples to save as comparison images")

    return parser.parse_args()


def main() -> None:
    args = get_args()
    if args.mode == "train":
        run_train(args)
    elif args.mode == "val":
        run_eval_or_test(args, mode="val")
    elif args.mode == "test":
        run_eval_or_test(args, mode="test")
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()