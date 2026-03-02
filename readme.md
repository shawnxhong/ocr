# PaddleOCR + OpenVINO Demo

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Prepare PaddleOCR models

```bash
python convert_models.py --output ../models/paddleocr-ov
```

This downloads PaddleOCR detection / recognition / classifier / layout / table models,
converts them to OpenVINO IR, and downloads dictionary files.

## 3) Run app

```bash
python run_ocr.py
```

Then open `http://127.0.0.1:7860`.
