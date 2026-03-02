from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def postprocess_det(
    pred_map: np.ndarray,
    meta: dict,
    thresh: float = 0.3,
    min_area: float = 20.0,
) -> list[dict]:
    if pred_map.ndim == 4:
        pred_map = pred_map[0, 0]
    elif pred_map.ndim == 3:
        pred_map = pred_map[0]

    binary = (pred_map > thresh).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    results: list[dict] = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        area = cv2.contourArea(box.astype(np.float32))
        if area < min_area:
            continue

        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [box.astype(np.int32)], -1, 255, -1)
        score = float(pred_map[mask > 0].mean()) if np.any(mask > 0) else 0.0
        if score < thresh:
            continue

        box = order_points_clockwise(box)
        box[:, 0] = np.clip(box[:, 0] * meta["scale_w"], 0, meta["orig_w"] - 1)
        box[:, 1] = np.clip(box[:, 1] * meta["scale_h"], 0, meta["orig_h"] - 1)
        results.append({"box": box.tolist(), "score": score})

    return sorted(results, key=lambda item: (item["box"][0][1], item["box"][0][0]))


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    expv = np.exp(logits)
    return expv / np.sum(expv, axis=-1, keepdims=True)


def postprocess_cls(logits: np.ndarray) -> dict:
    if logits.ndim == 2:
        logits = logits[0]
    probs = softmax(logits)
    label = int(np.argmax(probs))
    labels = ["0", "180"]
    direction = labels[label] if label < len(labels) else str(label)
    return {"label": direction, "score": float(probs[label])}


def postprocess_rec(
    logits: np.ndarray,
    characters: list[str],
    blank_idx: int = 0,
) -> dict:
    if logits.ndim == 3:
        logits = logits[0]
    probs = softmax(logits)
    idxs = np.argmax(probs, axis=-1)

    output_chars: list[str] = []
    confs: list[float] = []
    prev_idx = None
    for t, idx in enumerate(idxs):
        idx = int(idx)
        if idx == blank_idx or idx == prev_idx:
            prev_idx = idx
            continue
        char_idx = idx - 1
        if 0 <= char_idx < len(characters):
            output_chars.append(characters[char_idx])
            confs.append(float(probs[t, idx]))
        prev_idx = idx

    text = "".join(output_chars)
    score = float(np.mean(confs)) if confs else 0.0
    return {"text": text, "score": score}


def postprocess_layout(raw_output: np.ndarray, meta: dict, score_thresh: float = 0.4) -> list[dict]:
    if raw_output.ndim == 3:
        raw_output = raw_output[0]

    results: list[dict] = []
    for row in raw_output:
        if len(row) < 6:
            continue
        x1, y1, x2, y2, score, class_id = row[:6]
        if score < score_thresh:
            continue
        bbox = [
            float(np.clip(x1 * meta["scale_w"], 0, meta["orig_w"] - 1)),
            float(np.clip(y1 * meta["scale_h"], 0, meta["orig_h"] - 1)),
            float(np.clip(x2 * meta["scale_w"], 0, meta["orig_w"] - 1)),
            float(np.clip(y2 * meta["scale_h"], 0, meta["orig_h"] - 1)),
        ]
        results.append({"bbox": bbox, "score": float(score), "class_id": int(class_id)})
    return sorted(results, key=lambda x: (x["bbox"][1], x["bbox"][0]))


def read_charset(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def sort_reading_order(items: Iterable[dict]) -> list[dict]:
    return sorted(items, key=lambda x: (x.get("box", [[0, 0]])[0][1], x.get("box", [[0, 0]])[0][0]))
