from __future__ import annotations

import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

BASE_DIR = Path(__file__).resolve().parent


class OCRGradio:
    def __init__(self, ocr_pipeline) -> None:
        self.pipeline = ocr_pipeline

    def initialize(self, model_name: str, device: str):
        self.pipeline.initialize(model_name, device)

    def release(self):
        self.pipeline.release()


def draw_annotations(image: np.ndarray, results: list[dict], mode: str = "Quick OCR") -> np.ndarray:
    annotated = image.copy()
    color_map = {
        "text": (255, 0, 0),
        "table": (0, 200, 0),
        "formula": (0, 0, 255),
        "title": (255, 120, 0),
        "figure": (140, 140, 140),
    }

    for item in results:
        item_type = item.get("type", "text")
        color = color_map.get(item_type, (255, 0, 0))
        score = item.get("score", 0.0)
        label_text = item.get("text", item_type).strip()[:24]

        if "box" in item:
            pts = np.array(item["box"], dtype=np.int32)
            cv2.polylines(annotated, [pts], True, color, 2)
            pos = tuple(pts[0])
        else:
            x1, y1, x2, y2 = map(int, item.get("bbox", [0, 0, 0, 0]))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            pos = (x1, max(y1 - 5, 10))

        cv2.putText(
            annotated,
            f"{label_text} ({score:.2f})",
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return annotated


def demo_ocr(ocr_ov: OCRGradio):
    def process_image(image, mode, confidence_threshold):
        pipeline = ocr_ov.pipeline
        if not pipeline.initialized:
            return "", None, "Please load a model first.", {"error": "model not loaded"}
        if image is None:
            return "", None, "No image provided.", {"error": "no image"}

        if mode == "Quick OCR":
            results = pipeline.ocr(image, confidence_threshold)
            plain_text = "\n".join(r["text"] for r in results)
            markdown = "Use Document Parsing mode to generate markdown output."
            json_data = {"mode": mode, "results": results}
            annotated = draw_annotations(image, results, mode)
            return plain_text, annotated, markdown, json_data

        structure = pipeline.structure(image, confidence_threshold)
        regions = structure.get("regions", [])
        plain_text = "\n".join([r.get("text", "") for r in regions if r.get("text")])
        markdown = structure.get("markdown", "")
        json_data = {"mode": mode, "results": regions}
        annotated = draw_annotations(image, regions, mode)
        return plain_text, annotated, markdown, json_data

    def run_benchmark(repeats):
        if not ocr_ov.pipeline.initialized:
            return "Model not loaded"
        try:
            metrics = ocr_ov.pipeline.benchmark(int(repeats))
            return (
                f"Detection: {metrics['detection_latency_ms']:.2f} ms | "
                f"Recognition: {metrics['recognition_latency_ms']:.2f} ms/region | "
                f"Total: {metrics['end_to_end_latency_ms']:.2f} ms | "
                f"Throughput: {metrics['throughput_images_per_s']:.2f} img/s | "
                f"Regions: {metrics['regions_detected']}"
            )
        except Exception as exc:
            return f"Benchmark failed: {exc}"

    with gr.Blocks(title="Document OCR with PaddleOCR", analytics_enabled=False) as demo:
        with gr.Row():
            gr.Markdown("# Document OCR with PaddleOCR")
            release_button = gr.Button("🗑️ Release", variant="secondary")

        gr.Markdown("Upload an image and extract text with layout understanding.")

        with gr.Row():
            mode_selector = gr.Radio(["Quick OCR", "Document Parsing"], value="Quick OCR", label="Mode")
            language_dropdown = gr.Dropdown(
                choices=list(ocr_ov.pipeline.model_config.keys()),
                value=list(ocr_ov.pipeline.model_config.keys())[0],
                label="Language",
            )
            device_dropdown = gr.Dropdown(choices=["CPU"], value="CPU", label="Device")
            load_button = gr.Button("Load", variant="primary")

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                image_input = gr.Image(type="numpy", label="Input Image", height=480)
                process_button = gr.Button("Process", variant="primary")
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("Plain Text"):
                        plain_text = gr.Textbox(lines=20, label="Plain Text")
                    with gr.Tab("Annotated Image"):
                        annotated_image = gr.Image(type="numpy", label="Annotated")
                    with gr.Tab("Markdown"):
                        markdown_output = gr.Markdown("Markdown output appears in Document Parsing mode.")
                    with gr.Tab("Structured Data (JSON)"):
                        json_output = gr.JSON(label="Structured Data")

        with gr.Row():
            model_status_md = gr.Markdown("**Model: not loaded**")
            device_status_md = gr.Markdown("**Device: -**")

        with gr.Accordion("Settings", open=True):
            confidence_slider = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Confidence threshold")

        with gr.Accordion("Benchmark", open=True):
            repeats = gr.Slider(minimum=1, maximum=5, step=1, value=1, label="Benchmark repeats")
            bench_btn = gr.Button("Run Benchmark")
            bench_output = gr.Markdown("-")

        def on_load(model_name, device):
            ocr_ov.initialize(model_name, device)
            return f"**Model: `{model_name}`**", f"**Device: `{device}`**"

        def on_release():
            ocr_ov.release()
            return "**Model: not loaded**", "**Device: -**"

        load_button.click(on_load, [language_dropdown, device_dropdown], [model_status_md, device_status_md])
        release_button.click(on_release, outputs=[model_status_md, device_status_md])
        process_button.click(
            process_image,
            [image_input, mode_selector, confidence_slider],
            [plain_text, annotated_image, markdown_output, json_output],
        )
        bench_btn.click(run_benchmark, [repeats], [bench_output])

    return demo
