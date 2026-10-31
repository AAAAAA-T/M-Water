# dataset.py
"""
Dataset loading, transformations, and utility functions for data I/O.
"""
import math
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

cv2.setNumThreads(0)


# ---------- file utilities ----------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_split_file(txt_path: Path) -> List[str]:
    if not txt_path.exists():
        raise FileNotFoundError(f"Cannot find split file: {txt_path}")
    ids = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(Path(line).stem)
    if len(ids) == 0:
        raise RuntimeError(f"Split file is empty: {txt_path}")
    return ids


# ---------- image / mask reading ----------
def _safe_imread_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if img.ndim == 2:
        img = img[:, :, None]
    elif img.ndim == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    img = img.astype(np.float32)
    if img.max() > 255.0:
        img = img / 65535.0 * 255.0
    return img


def _make_extra_channels(img: np.ndarray, target_channels: int) -> np.ndarray:
    """Append simple deterministic channels if needed."""
    h, w, c = img.shape
    channels = [img[:, :, i:i + 1] for i in range(c)]

    rgb = img[:, :, :3] if c >= 3 else np.repeat(img[:, :, :1], 3, axis=2)
    r = rgb[:, :, 0:1]
    g = rgb[:, :, 1:2]
    b = rgb[:, :, 2:3]
    gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    ndwi_like = (g - b) / (g + b + 1e-6)
    ndwi_like = ((ndwi_like + 1.0) * 0.5 * 255.0).clip(0, 255)

    gx = cv2.Sobel(gray[:, :, 0], cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray[:, :, 0], cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    if edge.max() > 0:
        edge = edge / edge.max() * 255.0
    edge = edge[:, :, None]

    candidates = [ndwi_like, gray, edge]
    while len(channels) < target_channels:
        if candidates:
            channels.append(candidates.pop(0).astype(np.float32))
        else:
            channels.append(np.zeros((h, w, 1), dtype=np.float32))
    return np.concatenate(channels[:target_channels], axis=2)


def read_image(path: Path, input_channels: int) -> np.ndarray:
    img = _safe_imread_image(path)
    if img.shape[2] >= input_channels:
        img = img[:, :, :input_channels]
    else:
        img = _make_extra_channels(img, input_channels)
    return img.astype(np.float32)


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    mask = (mask > 127).astype(np.float32)
    return mask[:, :, None]


# ---------- transforms ----------
class SimpleSegTransform:
    def __init__(
        self,
        input_h: int,
        input_w: int,
        input_channels: int,
        mode: str = "train",
        aug: bool = True,
    ):
        self.input_h = input_h
        self.input_w = input_w
        self.input_channels = input_channels
        self.mode = mode
        self.aug = aug and mode == "train"

        base_mean = [0.485, 0.456, 0.406]
        base_std = [0.229, 0.224, 0.225]
        if input_channels > 3:
            base_mean += [0.5] * (input_channels - 3)
            base_std += [0.5] * (input_channels - 3)
        self.mean = np.array(base_mean[:input_channels], dtype=np.float32).reshape(1, 1, input_channels)
        self.std = np.array(base_std[:input_channels], dtype=np.float32).reshape(1, 1, input_channels)

    def _random_affine(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        angle = random.uniform(-10.0, 10.0)
        scale = random.uniform(0.92, 1.08)
        tx = random.uniform(-0.03, 0.03) * w
        ty = random.uniform(-0.03, 0.03) * h
        center = (w * 0.5, h * 0.5)
        mat = cv2.getRotationMatrix2D(center, angle, scale)
        mat[0, 2] += tx
        mat[1, 2] += ty
        image = cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if mask.ndim == 2:
            mask = mask[:, :, None]
        return image, mask

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.aug:
            if random.random() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1, :])
                mask = np.ascontiguousarray(mask[:, ::-1, :])
            if random.random() < 0.5:
                image = np.ascontiguousarray(image[::-1, :, :])
                mask = np.ascontiguousarray(mask[::-1, :, :])
            if random.random() < 0.5:
                k = random.randint(1, 3)
                image = np.ascontiguousarray(np.rot90(image, k, axes=(0, 1)))
                mask = np.ascontiguousarray(np.rot90(mask, k, axes=(0, 1)))
            if random.random() < 0.35:
                image, mask = self._random_affine(image, mask)
            if random.random() < 0.35:
                alpha = random.uniform(0.90, 1.10)
                beta = random.uniform(-12.0, 12.0)
                image = (image * alpha + beta).clip(0, 255)

        image = cv2.resize(image, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.input_w, self.input_h), interpolation=cv2.INTER_NEAREST)
        if mask.ndim == 2:
            mask = mask[:, :, None]

        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        mask = (mask > 0.5).astype(np.float32)

        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask.transpose(2, 0, 1)).float()
        return image_t, mask_t


# ---------- dataset class ----------
class WaterSegDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_ids: List[str],
        img_dir: Path,
        mask_dir: Path,
        img_ext: str,
        mask_ext: str,
        input_channels: int,
        transform: SimpleSegTransform,
        mode: str = "train",
    ):
        self.img_ids = img_ids
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.input_channels = input_channels
        self.transform = transform
        self.mode = mode
        print(f"[Dataset-{mode}] samples={len(img_ids)} | img_dir={self.img_dir} | mask_dir={self.mask_dir}")

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        img_path = self.img_dir / f"{img_id}{self.img_ext}"
        mask_path = self.mask_dir / f"{img_id}{self.mask_ext}"
        image = read_image(img_path, self.input_channels)
        mask = read_mask(mask_path)
        image_t, mask_t = self.transform(image, mask)
        return image_t, mask_t, img_id