from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

import openvino as ov

MODEL_URLS = {
    "det": [
        "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar",
    ],
    "rec": [
        "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar",
    ],
    "cls": [
        "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
    ],
    "layout": [
        "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/PP-DocLayout_plus-L_infer.tar",
        "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar",
    ],
    "table": [
        "https://paddleocr.bj.bcebos.com/ppstructure/models/table/SLANet_plus_infer.tar",
        "https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/ch_ppstructure_mobile_v2.0_SLANet_infer.tar",
    ],
}

DICT_URLS = {
    "ppocr_keys_v1.txt": "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/ppocr_keys_v1.txt",
    "en_dict.txt": "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/en_dict.txt",
}


def download(url: str, dst: Path, timeout: int = 60) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return

    tmp_dst = dst.with_suffix(dst.suffix + ".tmp")
    if tmp_dst.exists():
        tmp_dst.unlink()

    print(f"[download] {url}")
    with urllib.request.urlopen(url, timeout=timeout) as response, open(tmp_dst, "wb") as output:
        shutil.copyfileobj(response, output)

    tmp_dst.replace(dst)


def download_with_fallback(urls: list[str], dst: Path) -> str:
    errors = []
    for url in urls:
        try:
            download(url, dst)
            return url
        except urllib.error.HTTPError as exc:
            errors.append(f"{url} -> HTTP {exc.code}")
        except Exception as exc:
            errors.append(f"{url} -> {exc}")

    raise RuntimeError("all download URLs failed:\n  " + "\n  ".join(errors))


def safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    # Use the extraction filter to avoid the deprecation warning and improve safety on new Python versions.
    tar.extractall(path=path, filter="data")


def find_paddle_files(extracted_dir: Path) -> tuple[Path, Path]:
    models = [p for p in extracted_dir.rglob("*.pdmodel") if not p.name.startswith("._")]
    params = [p for p in extracted_dir.rglob("*.pdiparams") if not p.name.startswith("._")]
    if not models:
        raise FileNotFoundError(f"No .pdmodel found under {extracted_dir}")
    if not params:
        raise FileNotFoundError(f"No .pdiparams found under {extracted_dir}")

    # Prefer the largest model file to avoid macOS metadata artifacts.
    pdmodel = max(models, key=lambda p: p.stat().st_size)

    matching_param = pdmodel.with_suffix(".pdiparams")
    if matching_param.exists() and not matching_param.name.startswith("._"):
        return pdmodel, matching_param

    pdiparams = max(params, key=lambda p: p.stat().st_size)
    return pdmodel, pdiparams


def convert_single(name: str, archive_path: Path, out_root: Path) -> None:
    extract_dir = out_root / "_tmp" / name
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path) as tar:
        safe_extract(tar, extract_dir)

    pdmodel, _pdiparams = find_paddle_files(extract_dir)
    print(f"[convert] {name}: {pdmodel.name}")
    model = ov.convert_model(str(pdmodel))

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

    for name, urls in MODEL_URLS.items():
        archive_path = archive_dir / f"{name}.tar"
        try:
            used_url = download_with_fallback(urls, archive_path)
            print(f"[info] using source URL for {name}: {used_url}")
            convert_single(name, archive_path, out_root)
        except Exception as exc:
            print(f"[warn] failed to prepare {name}: {exc}")

    dict_dir = out_root / "dict"
    for name, url in DICT_URLS.items():
        try:
            download(url, dict_dir / name)
        except Exception as exc:
            print(f"[warn] failed to download dictionary {name}: {exc}")

    print(f"Done. Models are prepared under: {out_root}")


if __name__ == "__main__":
    main()
