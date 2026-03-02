from __future__ import annotations

import gc
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

from front_end import OCRGradio, demo_ocr
from postprocess import (
    postprocess_cls,
    postprocess_det,
    postprocess_layout,
    postprocess_rec,
    read_charset,
    sort_reading_order,
)
from preprocess import preprocess_cls, preprocess_det, preprocess_layout, preprocess_rec

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
MODEL_ROOT = PROJECT_ROOT / "models"


class PaddleOCRPipeline:
    def __init__(self, model_dir: Path | None = None, device: str = "CPU") -> None:
        self.core = ov.Core()
        self.model_dir = model_dir or (MODEL_ROOT / "paddleocr-ov")
        self.device = device

        self.model_config = {
            "PP-OCRv4 (Chinese+English)": {
                "det": self.model_dir / "det" / "inference.xml",
                "rec": self.model_dir / "rec" / "inference.xml",
                "cls": self.model_dir / "cls" / "inference.xml",
                "dict": self.model_dir / "dict" / "ppocr_keys_v1.txt",
                "layout": self.model_dir / "layout" / "inference.xml",
                "table": self.model_dir / "table" / "inference.xml",
            },
            "PP-OCRv4 (English)": {
                "det": self.model_dir / "det" / "inference.xml",
                "rec": self.model_dir / "rec_en" / "inference.xml",
                "cls": self.model_dir / "cls" / "inference.xml",
                "dict": self.model_dir / "dict" / "en_dict.txt",
                "layout": self.model_dir / "layout" / "inference.xml",
                "table": self.model_dir / "table" / "inference.xml",
            },
        }

        self.det_model = None
        self.rec_model = None
        self.cls_model = None
        self.layout_model = None
        self.table_model = None
        self.char_dict: list[str] = []

        self.initialized = False

    def _compile_optional(self, path: Path):
        if not path.exists():
            return None
        return self.core.compile_model(str(path), self.device)

    def initialize(self, model_name: str, device: str = "CPU") -> None:
        if self.initialized and self.device == device:
            return

        if model_name not in self.model_config:
            raise ValueError(f"Unknown model config: {model_name}")

        self.release()
        cfg = self.model_config[model_name]
        self.device = device

        self.det_model = self.core.compile_model(str(cfg["det"]), device)
        self.rec_model = self.core.compile_model(str(cfg["rec"]), device)
        self.cls_model = self._compile_optional(cfg["cls"])
        self.layout_model = self._compile_optional(cfg["layout"])
        self.table_model = self._compile_optional(cfg["table"])

        dict_path = cfg["dict"]
        if not dict_path.exists():
            raise FileNotFoundError(f"Dictionary file missing: {dict_path}")
        self.char_dict = read_charset(str(dict_path))
        self.initialized = True

    def release(self) -> None:
        self.det_model = None
        self.rec_model = None
        self.cls_model = None
        self.layout_model = None
        self.table_model = None
        self.char_dict = []
        self.initialized = False
        gc.collect()

    @staticmethod
    def _infer(compiled_model, tensor: np.ndarray) -> np.ndarray:
        output = compiled_model([tensor])
        return next(iter(output.values()))

    @staticmethod
    def _crop_polygon(image: np.ndarray, box: list[list[float]]) -> np.ndarray:
        pts = np.array(box, dtype=np.float32)
        x_min, y_min = np.maximum(np.floor(pts.min(axis=0)).astype(int), 0)
        x_max, y_max = np.minimum(np.ceil(pts.max(axis=0)).astype(int), [image.shape[1] - 1, image.shape[0] - 1])
        if x_max <= x_min or y_max <= y_min:
            return np.zeros((48, 192, 3), dtype=np.uint8)
        return image[y_min:y_max, x_min:x_max]

    def ocr(self, image: np.ndarray, confidence_threshold: float = 0.5) -> list[dict]:
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized")

        det_tensor, det_meta = preprocess_det(image)
        det_out = self._infer(self.det_model, det_tensor)
        boxes = postprocess_det(det_out, det_meta)

        results: list[dict] = []
        for box_info in boxes:
            crop = self._crop_polygon(image, box_info["box"])
            if crop.size == 0:
                continue

            if self.cls_model is not None:
                cls_tensor = preprocess_cls(crop)
                cls_out = self._infer(self.cls_model, cls_tensor)
                cls_info = postprocess_cls(cls_out)
                if cls_info["label"] == "180":
                    crop = cv2.rotate(crop, cv2.ROTATE_180)

            rec_tensor = preprocess_rec(crop)
            rec_out = self._infer(self.rec_model, rec_tensor)
            rec_info = postprocess_rec(rec_out, self.char_dict)

            score = float((box_info["score"] + rec_info["score"]) / 2.0)
            if score < confidence_threshold or not rec_info["text"].strip():
                continue

            results.append({
                "type": "text",
                "box": box_info["box"],
                "text": rec_info["text"],
                "score": score,
            })

        return sort_reading_order(results)

    def structure(self, image: np.ndarray, confidence_threshold: float = 0.5) -> dict:
        if self.layout_model is None:
            ocr_results = self.ocr(image, confidence_threshold)
            return {"regions": ocr_results, "markdown": "\n\n".join(r["text"] for r in ocr_results)}

        layout_tensor, layout_meta = preprocess_layout(image)
        layout_out = self._infer(self.layout_model, layout_tensor)
        regions = postprocess_layout(layout_out, layout_meta)

        structured = []
        for region in regions:
            x1, y1, x2, y2 = map(int, region["bbox"])
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            if region["class_id"] in {0, 1, 2, 3}:  # text-like categories
                ocr_items = self.ocr(crop, confidence_threshold)
                text = "\n".join(item["text"] for item in ocr_items)
                structured.append({
                    "type": "text",
                    "bbox": region["bbox"],
                    "score": region["score"],
                    "text": text,
                    "children": ocr_items,
                })
            elif region["class_id"] == 4:
                structured.append({"type": "table", "bbox": region["bbox"], "score": region["score"], "text": ""})
            else:
                structured.append({"type": "figure", "bbox": region["bbox"], "score": region["score"], "text": ""})

        markdown_lines = []
        for item in structured:
            if item["type"] == "text" and item["text"]:
                markdown_lines.append(item["text"])
            elif item["type"] == "table":
                markdown_lines.append("[Table region detected]")
            elif item["type"] == "figure":
                markdown_lines.append("[Figure region detected]")

        return {"regions": structured, "markdown": "\n\n".join(markdown_lines)}

    def benchmark(self, repeats: int = 1) -> dict:
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized")

        repeats = max(1, int(repeats))
        dummy = np.ones((960, 960, 3), dtype=np.uint8) * 255

        det_times = []
        rec_times = []
        total_times = []
        region_counts = []

        for _ in range(repeats):
            start_total = time.time()
            det_t0 = time.time()
            det_tensor, det_meta = preprocess_det(dummy)
            det_out = self._infer(self.det_model, det_tensor)
            boxes = postprocess_det(det_out, det_meta)
            det_times.append(time.time() - det_t0)

            per_rec = []
            for box in boxes[:8]:
                crop = self._crop_polygon(dummy, box["box"])
                rec_t0 = time.time()
                rec_tensor = preprocess_rec(crop)
                _ = self._infer(self.rec_model, rec_tensor)
                per_rec.append(time.time() - rec_t0)

            rec_times.extend(per_rec or [0.0])
            region_counts.append(len(boxes))
            total_times.append(time.time() - start_total)

        avg_total = sum(total_times) / repeats
        return {
            "detection_latency_ms": (sum(det_times) / repeats) * 1000,
            "recognition_latency_ms": (sum(rec_times) / max(len(rec_times), 1)) * 1000,
            "end_to_end_latency_ms": avg_total * 1000,
            "throughput_images_per_s": repeats / sum(total_times),
            "regions_detected": int(sum(region_counts) / repeats),
        }


def main() -> None:
    pipeline = PaddleOCRPipeline()
    ocr_ui = OCRGradio(pipeline)
    demo = demo_ocr(ocr_ui)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
