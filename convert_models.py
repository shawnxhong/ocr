from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

import openvino as ov

MODEL_URLS = {
    "det": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar",
    "rec": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar",
    "cls": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
    "layout": "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/PP-DocLayout_plus-L_infer.tar",
    "table": "https://paddleocr.bj.bcebos.com/ppstructure/models/table/SLANet_plus_infer.tar",
}

DICT_URLS = {
    "ppocr_keys_v1.txt": "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/ppocr_keys_v1.txt",
    "en_dict.txt": "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/en_dict.txt",
}


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    print(f"[download] {url}")
    urllib.request.urlretrieve(url, dst)


def find_paddle_files(extracted_dir: Path) -> tuple[Path, Path]:
    pdmodel = next(extracted_dir.rglob("*.pdmodel"))
    pdiparams = next(extracted_dir.rglob("*.pdiparams"))
    return pdmodel, pdiparams


def convert_single(name: str, archive_path: Path, out_root: Path) -> None:
    extract_dir = out_root / "_tmp" / name
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as tar:
        tar.extractall(path=extract_dir)

    pdmodel, pdiparams = find_paddle_files(extract_dir)
    print(f"[convert] {name}: {pdmodel.name}")
    model = ov.convert_model(str(pdmodel), example_input=None, input=None, extension=None)

    save_dir = out_root / name
    save_dir.mkdir(parents=True, exist_ok=True)
    ov.save_model(model, str(save_dir / "inference.xml"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert PaddleOCR models to OpenVINO IR")
    parser.add_argument("--output", type=Path, default=Path("../models/paddleocr-ov"))
    args = parser.parse_args()

    out_root = args.output.resolve()
    archive_dir = out_root / "_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    for name, url in MODEL_URLS.items():
        archive_path = archive_dir / f"{name}.tar"
        download(url, archive_path)
        try:
            convert_single(name, archive_path, out_root)
        except Exception as exc:
            print(f"[warn] failed to convert {name}: {exc}")

    dict_dir = out_root / "dict"
    for name, url in DICT_URLS.items():
        download(url, dict_dir / name)

    print(f"Done. Models are prepared under: {out_root}")


if __name__ == "__main__":
    main()
