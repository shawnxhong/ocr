# PaddleOCR Migration Plan

## Overview

Replace GOT-OCR 2.0 (generative VLM via `optimum-intel`) with PaddleOCR (traditional detection+recognition pipeline via `openvino.Core`). This migration fundamentally changes the inference pattern from autoregressive token generation to multi-stage single-pass inference, yielding richer structured output (bounding boxes, confidence scores, layout analysis, table/formula recognition).

---

## Architecture Comparison

```
CURRENT (GOT-OCR 2.0):
  Image → [Generative VLM] → plain text string
  API: optimum-intel OVModelForVisualCausalLM → model.generate()

TARGET (PaddleOCR):
  Image → [Text Detector] → boxes → [Text Recognizer] → structured results
  Image → [Layout Detector] → regions → [Table/Formula/OCR] → Markdown + JSON
  API: openvino.Core → core.compile_model() → compiled_model([tensor])
```

---

## Phase 1: Model Preparation & Conversion Utility

### 1.1 Target Models

Use **PP-OCRv4** models for initial implementation (`.pdmodel` format — direct OpenVINO support, no conversion hassles). Upgrade path to PP-OCRv5 later.

| Model | Purpose | Source | Size |
|---|---|---|---|
| `ch_PP-OCRv4_det` | Text detection | PaddleOCR model zoo | ~4.7M |
| `ch_PP-OCRv4_rec` | Text recognition (CN+EN) | PaddleOCR model zoo | ~10M |
| `ch_ppocr_mobile_v2.0_cls` | Text direction classifier | PaddleOCR model zoo | ~1.4M |
| `PP-DocLayout_plus-L` | Layout detection (20 categories) | PaddleOCR model zoo | ~23M |
| `SLANet_plus` | Table structure recognition | PaddleOCR model zoo | ~7M |

For multilingual support, additional recognition models can be swapped in per language (e.g., `en_PP-OCRv4_rec`, `japan_PP-OCRv4_rec`).

### 1.2 Conversion Script: `convert_models.py`

Create a utility script that:

1. Downloads PaddleOCR inference models from the official model zoo (tar.gz archives)
2. Extracts `.pdmodel` + `.pdiparams` files
3. Converts each to OpenVINO IR (`.xml` + `.bin`) using `openvino.convert_model()`
4. Saves all converted models under `../models/paddleocr-ov/`

```
models/
└── paddleocr-ov/
    ├── det/
    │   ├── inference.xml
    │   └── inference.bin
    ├── rec/
    │   ├── inference.xml
    │   └── inference.bin
    ├── cls/
    │   ├── inference.xml
    │   └── inference.bin
    ├── layout/
    │   ├── inference.xml
    │   └── inference.bin
    └── table/
        ├── inference.xml
        └── inference.bin
```

**Why convert ahead of time instead of reading `.pdmodel` directly?**
- Faster model load times (OpenVINO IR is pre-optimized)
- No runtime dependency on PaddlePaddle framework
- Consistent format for all models

### 1.3 Character Dictionary Files

PaddleOCR's text recognition models need character dictionary files for CTC decoding. These must be downloaded alongside the models:

- `ppocr_keys_v1.txt` — Chinese+English character set (~6600 chars)
- `en_dict.txt` — English-only character set
- Additional language dicts as needed

Store under `../models/paddleocr-ov/dict/`.

---

## Phase 2: Backend Refactoring (`run_ocr.py`)

### 2.1 New Class: `PaddleOCRPipeline`

Replace `OCRDemo` with a new class that manages multiple OpenVINO compiled models:

```python
class PaddleOCRPipeline:
    def __init__(self, model_dir, device="CPU"):
        self.core = openvino.Core()
        self.device = device
        self.model_dir = model_dir

        # Models (loaded on initialize())
        self.det_model = None      # text detection
        self.rec_model = None      # text recognition
        self.cls_model = None      # text direction classification
        self.layout_model = None   # layout detection (for document parsing mode)
        self.table_model = None    # table structure recognition

        # Config
        self.char_dict = None      # character dictionary for CTC decoding
        self.initialized = False

    def initialize(self, device="CPU"):
        """Load and compile all sub-models."""

    def release(self):
        """Release all compiled models."""

    def ocr(self, image) -> list[dict]:
        """Quick OCR mode: detect + classify + recognize."""

    def structure(self, image) -> dict:
        """Document parsing mode: layout + OCR + table/formula."""
```

### 2.2 Preprocessing Functions

Each sub-model requires specific preprocessing. Implement these as standalone functions:

**Detection preprocessing (`preprocess_det`):**
- Resize: scale longest side to 960 (configurable), keep aspect ratio
- Normalize: `(img / 255.0 - mean) / std` with ImageNet mean/std
- Pad to multiple of 32
- Transpose to NCHW format
- Return preprocessed tensor + scale factors (for mapping boxes back)

**Recognition preprocessing (`preprocess_rec`):**
- Resize cropped text region to fixed height (48px for v4), variable width (max 320)
- Normalize same as detection
- Pad width to target width
- Transpose to NCHW

**Classifier preprocessing (`preprocess_cls`):**
- Resize to `3 x 48 x 192`
- Normalize
- Transpose to NCHW

### 2.3 Postprocessing Functions

**Detection postprocessing (`postprocess_det`):**
- Apply DB (Differentiable Binarization) post-processing:
  - Threshold the probability map (default 0.3)
  - Apply binary dilation
  - Find contours via OpenCV
  - Compute minimum bounding boxes (4-point polygons)
  - Filter by box area and confidence score
- Return list of polygon boxes + confidence scores

**Recognition postprocessing (`postprocess_rec`):**
- CTC greedy decoding:
  - Argmax along the character dimension
  - Remove blanks and consecutive duplicates
  - Map indices to characters using the character dictionary
- Return recognized text + confidence score

**Classifier postprocessing (`postprocess_cls`):**
- Softmax on output logits
- If class "180" has higher probability, rotate the text region 180 degrees

### 2.4 OCR Pipeline Flow (Quick OCR Mode)

```
Input image
    │
    ▼
[preprocess_det] → [det_model inference] → [postprocess_det]
    │
    ▼
List of text region polygons + scores
    │
    ▼
For each text region:
    ├── Crop from original image using polygon
    ├── [preprocess_cls] → [cls_model inference] → [postprocess_cls]
    ├── Rotate 180° if needed
    ├── [preprocess_rec] → [rec_model inference] → [postprocess_rec]
    └── Collect: { polygon, text, confidence }
    │
    ▼
Sort results by position (top-to-bottom, left-to-right)
    │
    ▼
Return structured results:
[
    { "box": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], "text": "Hello", "score": 0.98 },
    ...
]
```

### 2.5 Document Parsing Pipeline Flow (Structure Mode)

```
Input image
    │
    ▼
[preprocess_layout] → [layout_model inference] → [postprocess_layout]
    │
    ▼
List of layout regions: { bbox, category, score }
Categories: title, text, table, figure, formula, header, footer, etc.
    │
    ▼
For each region, by category:
    ├── "text" / "title" / "header" / "footer":
    │       └── Run Quick OCR pipeline on cropped region
    ├── "table":
    │       ├── [table_model inference] → HTML table structure
    │       └── Run OCR on each cell → fill in cell text
    ├── "formula":
    │       └── (Phase 2 stretch goal: formula recognition → LaTeX)
    └── "figure":
            └── Mark as image region (no OCR)
    │
    ▼
Apply reading order (top-to-bottom, left-to-right, respecting columns)
    │
    ▼
Generate outputs:
    ├── Markdown: assembled from recognized regions
    ├── JSON: full structured data with all boxes, types, texts, scores
    └── Plain text: concatenated text in reading order
```

### 2.6 Model Configuration

Keep the existing pattern of `model_config` dict but expand it:

```python
model_config = {
    "PP-OCRv4 (Chinese+English)": {
        "det": "paddleocr-ov/det",
        "rec": "paddleocr-ov/rec",
        "cls": "paddleocr-ov/cls",
        "dict": "paddleocr-ov/dict/ppocr_keys_v1.txt",
        "layout": "paddleocr-ov/layout",
        "table": "paddleocr-ov/table",
    },
}
```

---

## Phase 3: Frontend Refactoring (`front_end.py`)

### 3.1 Mode Selector

Replace the single-model dropdown with a **mode selector**:

| Mode | Backend pipeline | When to use |
|---|---|---|
| Quick OCR | det → cls → rec | Simple images, photos, single-column text |
| Document Parsing | layout → (OCR / table / formula per region) | Complex documents, multi-column, tables |

Implemented as a `gr.Radio` or `gr.Dropdown` at the top of the UI.

### 3.2 New Input Controls

Add below the mode selector:

- **Language dropdown**: `gr.Dropdown` — Chinese+English (default), English-only, Japanese, etc. This determines which recognition model/dict to use.
- **Confidence threshold slider**: `gr.Slider(0.1, 0.9, value=0.5)` — filter low-confidence detections before display.

### 3.3 Tabbed Output Panel

Replace the single `gr.Textbox` with `gr.Tabs` containing four views:

**Tab 1: Plain Text**
- `gr.Textbox` — all recognized text concatenated in reading order
- Same UX as current, users who just want the text get it immediately

**Tab 2: Annotated Image**
- `gr.Image` — original image with bounding boxes drawn on top
- Color-coded by type:
  - Blue: text regions (Quick OCR mode)
  - Green: table regions (Document Parsing mode)
  - Red: formula regions
  - Orange: title regions
  - Gray: figure regions
- Each box labeled with truncated text + confidence score
- Generated using OpenCV drawing functions on a copy of the input image

**Tab 3: Markdown** (Document Parsing mode only)
- `gr.Markdown` — rendered Markdown output
- Headings, paragraphs, tables (as Markdown tables), formulas (as LaTeX)
- Disabled/empty in Quick OCR mode with a note to use Document Parsing

**Tab 4: Structured Data (JSON)**
- `gr.JSON` or `gr.Code(language="json")` — full structured output
- Includes every detected region with: bounding box coordinates, text, confidence, region type
- Useful for developers integrating with downstream systems

### 3.4 Updated UI Layout

```
┌──────────────────────────────────────────────────────────────┐
│  Document OCR with PaddleOCR                   [Release]     │
│  Upload an image and extract text with layout understanding  │
├──────────────────────────────────────────────────────────────┤
│  Mode: (●) Quick OCR  ( ) Document Parsing                  │
│  Language: [Chinese+English ▾]   Device: [CPU ▾]   [Load]   │
├─────────────────────────┬────────────────────────────────────┤
│                         │ [Plain Text][Annotated][MD][JSON]  │
│                         │ ┌────────────────────────────────┐ │
│   Input Image           │ │                                │ │
│   (upload / drag-drop)  │ │   (active tab content)         │ │
│                         │ │                                │ │
│                         │ │                                │ │
│   [Process]             │ └────────────────────────────────┘ │
├─────────────────────────┴────────────────────────────────────┤
│  Model: PP-OCRv4   Device: CPU   Regions: 12   Conf: 0.95   │
├──────────────────────────────────────────────────────────────┤
│  ▸ Settings                                                  │
│    Confidence threshold: [====●=========] 0.5                │
│  ▸ Benchmark                                                 │
│    Benchmark repeats: [1]  [Run Benchmark]                   │
│    Detection: 45ms | Recognition: 12ms/region | Total: 189ms │
└──────────────────────────────────────────────────────────────┘
```

### 3.5 Annotated Image Generation

New helper function `draw_annotations(image, results, mode)`:

```python
def draw_annotations(image, results, mode="ocr"):
    """
    Draw bounding boxes and labels on image.

    For OCR mode: all boxes in blue with text+score labels
    For Structure mode: color-coded by region type
    """
    annotated = image.copy()
    for result in results:
        box = result["box"]           # 4-point polygon
        text = result["text"]         # recognized text
        score = result["score"]       # confidence
        rtype = result.get("type")    # region type (structure mode)

        color = TYPE_COLORS.get(rtype, (255, 0, 0))
        cv2.polylines(annotated, [np.array(box)], True, color, 2)
        label = f"{text[:20]}... {score:.2f}" if len(text) > 20 else f"{text} {score:.2f}"
        cv2.putText(annotated, label, tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return annotated
```

### 3.6 Process Function Refactoring

The current `process_image()` returns a single string. Replace with a function that returns multiple outputs for the tabbed display:

```python
def process_image(image, mode, language, confidence_threshold):
    """
    Returns: (plain_text, annotated_image, markdown, json_data)
    """
    if mode == "Quick OCR":
        results = pipeline.ocr(image)
    else:
        results = pipeline.structure(image)

    # Filter by confidence
    results = [r for r in results if r["score"] >= confidence_threshold]

    # Generate all output formats
    plain_text = "\n".join(r["text"] for r in results)
    annotated = draw_annotations(image, results, mode)
    markdown = generate_markdown(results) if mode == "Document Parsing" else plain_text
    json_data = results  # already structured

    return plain_text, annotated, markdown, json_data
```

### 3.7 Benchmark Updates

The current benchmark measures token generation speed (tokens/s, first token latency) — these metrics are specific to generative models and no longer apply.

New benchmark metrics for PaddleOCR:

| Metric | How measured |
|---|---|
| Detection latency | Time for det model inference on test image |
| Recognition latency | Average time per text region recognition |
| End-to-end latency | Total time from image input to full results |
| Throughput | Images processed per second |
| Regions detected | Number of text regions found |

---

## Phase 4: Dependencies & Configuration

### 4.1 Updated `requirements.txt`

```
# Core
openvino>=2024.0
opencv-python
numpy
pillow

# UI
gradio==5.49.1

# Model conversion (only needed for convert_models.py)
# paddle2onnx
# paddlepaddle
```

**Removed dependencies:**
- `torch`, `torchvision` — no longer needed (PaddleOCR models run via OpenVINO directly)
- `transformers` — no longer needed (no HuggingFace model loading)
- `optimum-intel` — no longer needed (was the GOT-OCR bridge to OpenVINO)
- `openvino-tokenizers` — no longer needed (no tokenization in traditional OCR)
- `nncf` — no longer needed (no runtime quantization)
- `verovio` — was unused

### 4.2 File Structure After Migration

```
ocr/
├── run_ocr.py              # PaddleOCRPipeline class + main()
├── front_end.py            # Gradio UI with tabs
├── preprocess.py           # Preprocessing functions for det/rec/cls/layout
├── postprocess.py          # Postprocessing: DB, CTC decode, layout NMS
├── convert_models.py       # Model download + conversion utility
├── requirements.txt        # Updated dependencies
├── readme.md               # Updated documentation
└── uploads/                # Temporary upload directory (auto-created)

models/                     # (parent directory)
└── paddleocr-ov/
    ├── det/
    ├── rec/
    ├── cls/
    ├── layout/
    ├── table/
    └── dict/
        └── ppocr_keys_v1.txt
```

---

## Phase 5: Implementation Order

### Step 1 — Model conversion utility
- [ ] Create `convert_models.py`
- [ ] Download PP-OCRv4 det/rec/cls models
- [ ] Convert to OpenVINO IR format
- [ ] Download character dictionaries
- [ ] Verify all models load with `openvino.Core`

### Step 2 — Core OCR pipeline (Quick OCR mode)
- [ ] Implement `preprocess.py` (det, rec, cls preprocessing)
- [ ] Implement `postprocess.py` (DB postprocessing, CTC decoding, cls softmax)
- [ ] Implement `PaddleOCRPipeline.ocr()` in `run_ocr.py`
- [ ] Test end-to-end: image → text regions with boxes + confidence

### Step 3 — Basic frontend with Plain Text + Annotated Image
- [ ] Refactor `front_end.py` with mode selector and tabbed output
- [ ] Implement Plain Text tab (concatenated OCR text)
- [ ] Implement Annotated Image tab (bounding boxes drawn on image)
- [ ] Implement JSON tab (structured results)
- [ ] Wire up process button to new pipeline

### Step 4 — Document parsing mode
- [ ] Download and convert layout detection model
- [ ] Download and convert table structure recognition model
- [ ] Implement `PaddleOCRPipeline.structure()` — layout detection + per-region OCR
- [ ] Implement reading order recovery
- [ ] Implement Markdown generation from structured results
- [ ] Enable Markdown tab in frontend

### Step 5 — Polish & benchmark
- [ ] Update benchmark to measure det/rec/e2e latency
- [ ] Add confidence threshold slider
- [ ] Add language dropdown (initially just CN+EN, EN-only)
- [ ] Update requirements.txt
- [ ] Update readme.md
- [ ] Clean up old GOT-OCR 2.0 code

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| PP-OCRv5 model format (`.json`) not supported by OpenVINO | Can't use latest models directly | Start with PP-OCRv4 (`.pdmodel`), add v5 support via `paddle2onnx` conversion later |
| Pre/post-processing mismatch causes poor accuracy | Wrong results despite correct models | Port preprocessing/postprocessing exactly from PaddleOCR source code; validate against PaddleOCR Python package output |
| Multiple model loading increases memory usage | Higher RAM consumption | Models are small (total ~47M vs GOT-OCR 2.0 at several GB); actually a net improvement |
| OpenVINO GPU inference bug with PaddleOCR | Garbled results on GPU (known issue #29367) | Target CPU initially; monitor OpenVINO fixes for GPU support |
| CTC decoding character dictionary mismatch | Garbled text output | Ensure dict file matches the exact model version; download dict from same model package |

---

## Expected Improvements After Migration

| Aspect | Before (GOT-OCR 2.0) | After (PaddleOCR) |
|---|---|---|
| Output format | Plain text string only | Structured: boxes + text + confidence + layout types |
| Layout understanding | None | 20 document element categories |
| Table extraction | None | HTML/Markdown table structure |
| Bounding boxes | None | Per-text-region polygon coordinates |
| Confidence scores | None | Per-region detection + recognition scores |
| Language support | Limited | 80+ languages (v4), 106 (v5) |
| Memory footprint | ~4GB+ (generative VLM) | ~50MB (all sub-models combined) |
| Dependencies | 10 packages (torch, transformers, etc.) | 4 packages (openvino, opencv, numpy, pillow) |
| Inference speed | Slow (autoregressive generation) | Fast (single-pass per sub-model) |
