from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    return (image - IMAGENET_MEAN) / IMAGENET_STD


def preprocess_det(image: np.ndarray, max_side_len: int = 960) -> tuple[np.ndarray, dict]:
    h, w = image.shape[:2]
    scale = min(max_side_len / max(h, w), 1.0)
    resized_h = max(32, int(round(h * scale / 32) * 32))
    resized_w = max(32, int(round(w * scale / 32) * 32))

    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    normed = normalize_image(resized)
    tensor = np.transpose(normed, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

    meta = {
        "orig_h": h,
        "orig_w": w,
        "resized_h": resized_h,
        "resized_w": resized_w,
        "scale_h": h / float(resized_h),
        "scale_w": w / float(resized_w),
    }
    return tensor, meta


def _resize_with_aspect(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = image.shape[:2]
    ratio = min(target_w / max(w, 1), target_h / max(h, 1))
    new_w = max(1, int(w * ratio))
    new_h = max(1, int(h * ratio))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    return canvas


def preprocess_rec(image: np.ndarray, target_h: int = 48, target_w: int = 320) -> np.ndarray:
    resized = _resize_with_aspect(image, target_h=target_h, target_w=target_w)
    normed = normalize_image(resized)
    return np.transpose(normed, (2, 0, 1))[np.newaxis, ...].astype(np.float32)


def preprocess_cls(image: np.ndarray, target_h: int = 48, target_w: int = 192) -> np.ndarray:
    resized = _resize_with_aspect(image, target_h=target_h, target_w=target_w)
    normed = normalize_image(resized)
    return np.transpose(normed, (2, 0, 1))[np.newaxis, ...].astype(np.float32)


def preprocess_layout(image: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> tuple[np.ndarray, dict]:
    h, w = image.shape[:2]
    target_w, target_h = target_size
    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    normed = normalize_image(resized)
    tensor = np.transpose(normed, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
    meta = {
        "orig_h": h,
        "orig_w": w,
        "scale_h": h / float(target_h),
        "scale_w": w / float(target_w),
    }
    return tensor, meta
