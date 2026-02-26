from pathlib import Path
import gc

import torch
from transformers import AutoProcessor
from optimum.intel.openvino import OVModelForVisualCausalLM

from front_end import OCRGradio, demo_ocr

try:
    from huggingface_hub import snapshot_download
except ImportError as e:
    raise ImportError(
        "huggingface_hub is required to download the model from Hugging Face.\n"
        "Please install it with: pip install huggingface_hub"
    ) from e

# === 全局配置 ===
ROOT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = ROOT_DIR.parent

MODEL_ROOT = PROJECT_ROOT / "models"

DEFAULT_LOCAL_MODEL_DIR = MODEL_ROOT / "GOT-OCR-2.0-OV-INT4"

HF_REPO_ID = "shawnxhong/GOT-OCR-2.0-OV-INT4"
HF_REVISION = "main"


class OCRDemo:
    def __init__(self, local_model_dir: Path | None = None) -> None:
        self.root_dir = ROOT_DIR
        self.local_model_dir = local_model_dir or DEFAULT_LOCAL_MODEL_DIR

        self.model_config = {
            "GOT-OCR-2.0 INT4": self.local_model_dir,
        }

        self.initialized: bool = False
        self.model = None
        self.processor = None
        self.device: str | None = None

    def _ensure_model_downloaded(self, local_dir: Path) -> Path:
        """
        确保 Hugging Face 上的模型已经下载到 local_dir 目录。
        如果目录不存在或为空，则调用 snapshot_download 拉取。
        """
        need_download = True
        if local_dir.exists() and any(local_dir.rglob("*")):
            need_download = False

        if need_download:
            print("[INFO] Local model not found, downloading from Hugging Face...")
            print(f"[INFO] repo_id = {HF_REPO_ID}, revision = {HF_REVISION}")
            local_dir.mkdir(parents=True, exist_ok=True)

            snapshot_download(
                repo_id=HF_REPO_ID,
                revision=HF_REVISION,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
            print(f"[INFO] Model downloaded to: {local_dir}")
        else:
            print(f"[INFO] Using existing local model at: {local_dir}")

        return local_dir

    def initialize(self, model_dir: str | Path | None, device: str = "CPU"):
        if self.initialized:
            print("[INFO] OCRDemo already initialized, skip re-init.")
            return

        if model_dir is None:
            model_path = self.local_model_dir
        else:
            model_path = Path(model_dir)

        model_path = self._ensure_model_downloaded(model_path)

        print("[INFO] Loading model...")
        print("[INFO] model_path =", model_path)
        print("[INFO] exists:", model_path.exists())

        processor = AutoProcessor.from_pretrained(model_path)
        model = OVModelForVisualCausalLM.from_pretrained(
            model_path,
            device=device,
        )

        model.eval()
        torch.set_grad_enabled(False)

        self.processor = processor
        self.model = model
        self.device = device
        self.initialized = True

        print("[INFO] processor:", type(processor))
        print("[INFO] tokenizer:", type(processor.tokenizer))
        print("[INFO] model:", type(model))
        print("[INFO] OCRDemo initialized on device:", device)

    def release(self):
        if not self.initialized:
            print("[INFO] OCRDemo not initialized, nothing to release.")
            return

        print("[INFO] Releasing OCR model and processor...")
        self.model = None
        self.processor = None
        self.device = None
        self.initialized = False

        gc.collect()
        print("[INFO] OCRDemo released.")


def main():
    """
    启动 Gradio Demo：
    - 构造后端 OCRDemo
    - 用 OCRGradio 包一层前端
    - 调用 demo_ocr(ocr_ov) 生成 Blocks
    """
    pipeline = OCRDemo()
    ocr_ui = OCRGradio(pipeline)
    demo = demo_ocr(ocr_ui)

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
