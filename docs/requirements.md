# Requirements & Design — OCR Document Understanding

## 1. Project Overview

This project builds a prototype system that takes invoice or receipt images as input and produces structured JSON output containing key fields such as document type, vendor name, invoice/receipt number, date, and total amount.

The goal is not to build a production OCR product but to demonstrate a thoughtful approach: extracting text from messy real-world images, reasoning about document layout, and pulling out the information that matters using explainable rules.

## 2. Scope

### In scope
- Process single images and folders of images
- Extract 5 mandatory fields: `document_type`, `vendor_name`, `invoice_number` / `receipt_number`, `invoice_date`, `total_amount`
- Attempt good-to-have fields where feasible (currency, subtotal, tax amount)
- Handle common image quality issues: skew, uneven lighting, mild noise, low contrast
- Produce structured JSON output per image
- Visual debugging output for at least 5 images
- Ground-truth file covering 15+ images
- Field-wise accuracy evaluation with failure analysis
- **Bonus 2:** image quality scoring
- **Bonus 3:** comparison of two OCR engines (Tesseract vs PaddleOCR)

### Out of scope
- Training an OCR model from scratch
- Paid OCR or extraction APIs (Google Vision, AWS Textract, etc.)
- Real-time / production-grade performance
- Multilingual support (English-only)
- Handwritten document support
- **Bonus 1 (line items)** — attempted only if core finishes early

## 3. Pipeline Architecture

The system is a six-stage pipeline. Each stage is implemented as a separate module in `src/`, with a single responsibility.

1. **Image Input** — load an image file from disk into memory.
2. **Preprocessing** — clean up image issues before OCR: deskew, lighting correction, denoising.
3. **OCR** — run the OCR engine. Output: list of `(text, bounding_box, confidence)` items.
4. **Layout Analysis** — pure spatial reasoning over bounding boxes. Group text into lines, identify regions, find spatial neighbours.
5. **Field Extraction** — apply regex patterns and spatial rules to pull out the 5 mandatory fields.
6. **Output & Evaluation** — produce structured JSON; compare against ground truth to compute field-wise accuracy.

## 4. Tech Stack

### Language: Python 3.11

- **Why:** Python is the standard for CV/ML work. Every library we need is built for it first.
- **Why 3.11 specifically:** Mature, broad library support, stable pre-built wheels for all our dependencies. Avoids occasional 3.12+ compatibility issues (especially with PaddleOCR's deep-learning backend).
- **Why not C++ / JavaScript:** No comparable CV ecosystem; not worth the trade-off for a 3-day prototype.

### Environment: `venv`

- **Why:** Built into Python, lightweight, isolates this project's libraries from the system Python.
- **Why not conda:** Heavier install, overkill for a project of this size.

### Image Processing: OpenCV (`opencv-python`)

- **Why:** Industry-standard CV library. Handles every preprocessing step we need: rotation, contrast, denoising, adaptive thresholding, CLAHE.
- **Why not Pillow (PIL):** Fine for basic image I/O but lacks the algorithmic operations we need for deskew and shadow handling.
- **Why not scikit-image:** Smaller community, fewer tutorials. OpenCV's larger user base means faster debugging.

### OCR Engines: Tesseract + PaddleOCR (comparison)

We compare two engines because the assignment includes this as Bonus 3, and because the contrast is instructive.

**Tesseract** (`pytesseract`)
- Classical, mature, fast, lightweight.
- Strong on clean scanned documents.
- Weak on skewed, shadowy, or photo-captured images.
- Represents the pre-deep-learning OCR approach.

**PaddleOCR**
- Modern deep-learning-based OCR from Baidu.
- Strong on messy, real-world images and photos.
- Heavier; benefits from GPU (we have a GTX 1650 Ti).
- Represents the current state of open-source OCR.

- **Why not EasyOCR:** Also deep-learning-based; comparing it to PaddleOCR would be "two similar things." Tesseract vs PaddleOCR is a more instructive old-vs-new contrast.
- **Why not cloud APIs:** Forbidden by the assignment.

### Layout Analysis: custom Python (no library)

- **Why:** The assignment values *explainability*. Writing our own spatial logic over bounding boxes means we can justify every decision.
- **Why not LayoutLM / heavy models:** Black-box, heavyweight, hides reasoning, overkill for this prototype.

### Field Extraction: `re` (regex) + custom spatial rules

- **Why:** Explainable, deterministic, requires no training data, ships with Python.
- **Why not a trained extraction model:** Needs labelled training data, hides reasoning, time-prohibitive in 3 days. We may add a lightweight ML fallback for cases rules can't crack — clearly labelled as such.

### Evaluation: `pandas` (minimal)

- **Why:** Clean tabular output for the accuracy summary.
- **Why not plain Python:** Possible, but `pandas` makes the code shorter and the output more readable.

### Visual Debugging: OpenCV + `matplotlib`

- **Why:** Drawing bounding boxes and saving annotated images is trivial with these. The assignment explicitly requires visual debug output.

## 5. Repository Structure

- `README.md` — user-facing setup and run guide
- `report.md` — 2-3 page technical report (built incrementally)
- `requirements.txt` — exact library versions for reproducibility
- `.gitignore` — excludes venv, caches, OS junk
- `run_pipeline.py` — entry point: single image or folder
- `evaluate.py` — entry point: accuracy vs ground truth
- `ground_truth.json` — hand-typed correct answers (15+ images)
- `sample_images/` — input images (30-50)
- `outputs/` — JSON results and visual debug images
- `src/` — pipeline modules
  - `__init__.py` — marks `src` as a Python package
  - `preprocessing.py` — Stage 2 (image cleanup)
  - `ocr.py` — Stage 3 (both OCR engines)
  - `layout.py` — Stage 4 (spatial reasoning)
  - `extraction.py` — Stage 5 (field extraction)
  - `visualize.py` — visual debug helpers
- `docs/`
  - `requirements.md` — this document

## 6. Initial Day-by-Day Plan

This is the **plan as of project start**. Actual progress and any deviations are tracked in `report.md`.

- **Day 1 (Thursday → Friday morning):** Setup complete. Dataset assembled. First image flowing through one OCR engine.
- **Day 2 (Friday → Saturday):** Preprocessing module. Both OCR engines wired in. Layout module. Visual debugging.
- **Day 3 (Sunday):** Field extraction. JSON output. Batch processing.
- **Day 4 (Monday morning):** Ground truth + evaluation. Bonus 2 (quality scoring). Report and README polish. Final commit.

Slack time on Day 4 is intentional — something always takes longer than planned.

## 7. Assumptions

- Documents are in English.
- One document per image (no multi-page or multi-doc scans).
- Documents are roughly upright; we handle moderate skew (up to ~15°), not fully sideways or upside-down images.
- Currency symbols and date formats vary; we target common Indian and international formats but cannot cover every regional format perfectly.
- The system runs on the developer's laptop (Intel i7, 16 GB RAM, GTX 1650 Ti). It does not need to run in cloud or production environments.

## 8. Known Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| PaddleOCR install issues on Windows | Allocate setup time; fallback to EasyOCR if blocked beyond 1 hour |
| Field extraction accuracy varies by layout | Acceptable per assignment; focus on honest failure analysis |
| Limited time for line-item extraction | Marked optional from the start |
| Ground-truth labelling is tedious | Limit ground truth to 15-20 images; pick the most representative |
| OCR confidence on shadowy / low-light images | Address with CLAHE + adaptive thresholding in preprocessing |