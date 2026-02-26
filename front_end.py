import time
import uuid
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image
from transformers.image_utils import load_image

# 以当前文件所在目录作为根目录，保证从任何工作目录启动都正常
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = ROOT_DIR / "examples"
stop_str = "<|im_end|>"


class OCRGradio:
    """
    - self.pipeline：后端 OCRDemo 实例
    - initialize(model_dir, device)
    - release()
    """

    def __init__(self, ocr_pipeline) -> None:
        self.pipeline = ocr_pipeline

    def initialize(self, model_dir, device):
        if self.pipeline.initialized is False:
            self.pipeline.initialize(model_dir, device)

    def release(self):
        if self.pipeline.initialized is True:
            self.pipeline.release()


def demo_ocr(ocr_ov: OCRGradio):
    """
    - 输入：OCRGradio 实例
    - 输出：gr.Blocks demo
    """
    # 上传目录，固定在 ocr/ 下
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    # 从 pipeline 里拿到可选模型列表
    model_choices = list(ocr_ov.pipeline.model_config.keys())
    device_choices = ["CPU"]  # 目前就一个设备选项，有需要再扩展

    def cleanup_old_files():
        current_time = time.time()
        for file_path in UPLOAD_FOLDER.glob("*"):
            if current_time - file_path.stat().st_mtime > 3600:  # 1 hour
                file_path.unlink()

    cleanup_old_files()

    # ====================== 核心推理逻辑（Plain Text OCR） ======================

    def process_image(image):
        """
        - 只支持 Plain Text OCR
        - 使用 ocr_ov.pipeline.model / processor 做推理
        """
        pipeline = ocr_ov.pipeline
        gr.Info(f"processing started...")
        if pipeline.initialized is False or pipeline.model is None or pipeline.processor is None:
            return "Error: model not loaded. Please select a model and click Load."

        if image is None:
            return "Error: No image provided"

        unique_id = str(uuid.uuid4())
        image_path = UPLOAD_FOLDER / f"{unique_id}.png"

        try:
            if isinstance(image, dict):
                composite_image = image.get("composite")
                if composite_image is None:
                    return "Error: No composite image found in ImageEditor output"
                img = composite_image
            else:
                img = image

            if isinstance(img, np.ndarray):
                cv2.imwrite(
                    str(image_path),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                )
            elif isinstance(img, Image.Image):
                img.save(str(image_path))
            else:
                return "Error: Unsupported image format"

            pil_img = load_image(str(image_path))

            inputs = pipeline.processor(pil_img, return_tensors="pt")
            generate_ids = pipeline.model.generate(
                **inputs,
                do_sample=False,
                tokenizer=pipeline.processor.tokenizer,
                stop_strings=stop_str,
                max_new_tokens=4096,
            )
            res = pipeline.processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            # 保留原来对 \title 的小修补
            res = res.replace("\\title", "\\title ")
            return res

        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            if image_path.exists():
                image_path.unlink()

    def run_benchmark(repeats):
        """
        benchmark：
        - 使用一张固定的白色图片
        - 连续跑 repeats 次 generate
        - 指标：
          - 总生成 token 数 / 总时间 = 平均 tokens/s
          - first token latency 统计（avg / min / max）
        """
        pipeline = ocr_ov.pipeline

        # 基本状态检查：模型没加载就直接返回，不抛异常
        if pipeline.initialized is False or pipeline.model is None or pipeline.processor is None:
            return "**tokens/s:** model not loaded (please load a model first)"

        import time as _time

        try:
            repeats = int(repeats)
        except Exception:
            repeats = 1

        if repeats < 1:
            repeats = 1

        # 构造 512x512 的 dummy 白图
        h, w = 512, 512
        dummy_np = np.ones((h, w, 3), dtype=np.uint8) * 255
        dummy_img = Image.fromarray(dummy_np)

        total_tokens = 0
        total_time = 0.0
        min_latency = float("inf")
        max_latency = 0.0

        first_latencies = []
        min_first = float("inf")
        max_first = 0.0

        try:
            for _ in range(repeats):
                # 预处理
                inputs = pipeline.processor(dummy_img, return_tensors="pt")
                input_len = inputs["input_ids"].shape[1]

                # -------- first token latency（单独跑一次 max_new_tokens=1）--------
                start_first = _time.time()
                _ = pipeline.model.generate(
                    **inputs,
                    do_sample=False,
                    tokenizer=pipeline.processor.tokenizer,
                    stop_strings=stop_str,
                    max_new_tokens=1,
                )
                end_first = _time.time()

                first_elapsed = max(end_first - start_first, 1e-6)
                first_latencies.append(first_elapsed)
                min_first = min(min_first, first_elapsed)
                max_first = max(max_first, first_elapsed)

                # -------- 吞吐测试：生成一段较长的文本 --------
                start = _time.time()
                generate_ids = pipeline.model.generate(
                    **inputs,
                    do_sample=False,
                    tokenizer=pipeline.processor.tokenizer,
                    stop_strings=stop_str,
                    max_new_tokens=64,
                )
                end = _time.time()

                elapsed = max(end - start, 1e-6)
                total_time += elapsed
                min_latency = min(min_latency, elapsed)
                max_latency = max(max_latency, elapsed)

                total_len = generate_ids.shape[1]
                gen_tokens = max(int(total_len - input_len), 1)
                total_tokens += gen_tokens

            # 统计结果
            avg_tps = total_tokens / total_time if total_time > 0 else 0.0
            avg_latency = total_time / repeats if repeats > 0 else 0.0

            if first_latencies:
                avg_first = sum(first_latencies) / len(first_latencies)
            else:
                avg_first = float("nan")
                min_first = float("nan")
                max_first = float("nan")

            return (
                f"**tokens/s:** {avg_tps:.2f}  \n"
                f"- runs: {repeats}  \n"
                f"- avg latency/run: {avg_latency:.3f} s  \n"
                f"- min latency: {min_latency:.3f} s, max latency: {max_latency:.3f} s  \n"
                f"- avg first token latency: {avg_first:.3f} s  \n"
                f"- min first token latency: {min_first:.3f} s, max first token latency: {max_first:.3f} s  \n"
                f"- total generated tokens: {total_tokens}"
            )

        except Exception as e:
            # 所有异常都被吃掉，返回字符串，不让前端炸掉
            return f"**tokens/s:** error during benchmark: {e}"

    # ====================== Gradio UI ======================

    with gr.Blocks(
        title="Image to Text",
        analytics_enabled=False,
        theme=gr.themes.Soft(primary_hue="blue"),
        css="""
        .button-center {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            min-height: 76px !important;
            padding-top: 20px !important;
            background: transparent !important;
        }
        .button-center button {
            margin: 0 !important;
        }
        """
    ) as demo:
        # 顶部标题 + Release 按钮
        with gr.Row():
            with gr.Column(scale=10):
                gr.HTML("""
                <div style="text-align: left; margin-bottom: 10px;">
                    <h1> Image to Text </h1>
                </div>
                """)
            with gr.Column(scale=1, min_width=80):
                release_button = gr.Button("🗑️ Release", variant="secondary", size="md", min_width=70)

        gr.HTML("""
        <div style="text-align: left; margin-bottom: 10px; font-size: 18px;">
            Plain Text OCR with GOT-OCR 2.0. Upload an image and extract text.
        </div>
        """)

        # 模型 / 设备选择 + Load
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=model_choices,
                label="Model",
                value=model_choices[0] if model_choices else None,
                interactive=True,
            )
            device_dropdown = gr.Dropdown(
                choices=device_choices,
                label="Device",
                value=device_choices[0],
                interactive=True,
            )
            with gr.Column(scale=1, elem_classes=["button-center"]):
                load_button = gr.Button("Load Model", variant="primary")

        # 主体区域：左边图片，右边文本输出
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                with gr.Group():
                    test_img_path = EXAMPLE_DIR / "ocr_test.png"
                    default_img = str(test_img_path) if test_img_path.exists() else None
                    image_input = gr.Image(
                        type="numpy",
                        label="Input Image",
                        height=500,
                        value=default_img
                    )
                    submit_button = gr.Button("Process", variant="primary")

            with gr.Column(scale=1):
                with gr.Group():
                    output_markdown = gr.Textbox(
                        label="Text output",
                        lines=24,
                    )

        # 模型状态显示
        with gr.Row():
            model_status_md = gr.Markdown("**Model: not loaded**")
            device_status_md = gr.Markdown("**Device: -**")

        # Benchmark
        with gr.Accordion("Benchmark using 512 * 512 dummy image", open=True):
            with gr.Column():
                bench_repeats = gr.Slider(
                    label="Benchmark repeats (for averaging)",
                    minimum=1,
                    maximum=3,
                    step=1,
                    value=1,
                    interactive=True,
                )
                bench_btn = gr.Button("📊 Run Benchmark", variant="primary")
                bench_result = gr.Markdown("**tokens/s:** -")

        # ----------------- 事件回调（闭包里用 ocr_ov / pipeline） -----------------

        def on_load(model_name, device):
            # 根据模型名字找到本地目录
            gr.Info(f"loading {model_name} on {device}",duration=4)
            model_dir = ocr_ov.pipeline.model_config.get(model_name)
            ocr_ov.initialize(model_dir, device)

            if ocr_ov.pipeline.initialized:
                model_txt = f"**Model: `{model_name}`**"
                device_txt = f"**Device: `{device}`**"
            else:
                model_txt = "**Model: not loaded**"
                device_txt = "**Device: -**"

            gr.Info(f"{model_name} on {device} loaded",duration=2)
            return model_txt, device_txt

        def on_release():
            ocr_ov.release()
            gr.Info("model released", duration=2)
            return "**Model: not loaded**", "**Device: -**"

        def on_ocr(image):
            return process_image(image)

        def on_bench(repeats):
            return run_benchmark(repeats)

        load_button.click(
            on_load,
            inputs=[model_dropdown, device_dropdown],
            outputs=[model_status_md, device_status_md],
        )

        release_button.click(
            on_release,
            inputs=[],
            outputs=[model_status_md, device_status_md],
        )

        submit_button.click(
            on_ocr,
            inputs=[image_input],
            outputs=[output_markdown],
        )

        bench_btn.click(
            on_bench,
            inputs=[bench_repeats],
            outputs=[bench_result],
        )

    return demo
