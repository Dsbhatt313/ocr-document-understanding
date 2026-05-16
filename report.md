## Day 1 Notes

Set up Python 3.11 venv, installed CV stack (OpenCV, pytesseract, PaddleOCR, supporting libs), installed Tesseract 5.5 on Windows. Built dataset of 42 images (14 personal Indian invoices, 28 SROIE Southeast Asian receipts).

Ran first OCR test on sroie_05.jpg. Tesseract works but baseline accuracy is mixed: character confusions (g/a, T/J, #/H), wrong digits (38.37 → 36.37), missed characters (4X150 → 4X50), garbled handwriting. Bounding box output (`image_to_data`) gives per-word coordinates and confidence values — these become the input for layout reasoning and reliability flagging.

Key insights: (1) Tesseract's character classifier is the weak link; PaddleOCR likely to outperform. (2) `conf` values per word are usable as a reliability signal. (3) Cross-engine agreement gives us free sanity-checking.

# OCR Document Understanding — Project Report

## Day 1 — Setup & First OCR

- Repository initialized with full project structure
- Requirements locked (54 packages pinned)
- Design document written (`docs/requirements.md`)
- First OCR run on SROIE receipt (sroie_05.jpg) using Tesseract
- Tesseract output analyzed: bounding boxes, confidence scores, error patterns
- Key errors cataloged: #→H, $→§, digit confusions (0→O, 9→S), date mangling

## Day 2 — Preprocessing & OCR Module

### Preprocessing (`src/preprocessing.py`)

**Shipped functions:**
- `apply_clahe()` — local contrast enhancement via CLAHE
- `apply_denoise()` — non-local means denoising, preserves text edges
- `apply_adaptive_threshold()` — available but NOT used with Tesseract
- `preprocess()` — full pipeline (CLAHE + denoise)

**Result:** +55% improvement in confident word detection on clean scan (78 → 121 words with conf > 50).

### Deskew — Investigated, Not Shipped

- Attempted angle detection via `cv2.minAreaRect` on binary text pixel cloud
- Failed because text fills the frame on receipt images — the bounding rectangle of the text pixel cloud equals the image rectangle itself, so minAreaRect reports no tilt regardless of actual skew
- Tried edge cropping and connected-components filtering to isolate text blobs from border artifacts — still failed because the text cloud spans the full document even without borders
- Alternative (Hough line transform on text baselines) deferred due to time constraints
- SROIE receipts unaffected (already well-aligned scans)
- Personal phone photos with significant tilt remain a known limitation

### Key Finding: Tesseract vs Pre-binarized Images

- Plain grayscale input (28 words) is WORSE than raw BGR (78 words) for Tesseract — contradicts common assumption
- Adaptive thresholded input produced 0 detected words — Tesseract's internal binarization conflicts with pre-thresholded input
- Decision: feed CLAHE-enhanced grayscale to Tesseract, never pre-threshold

### OCR Module (`src/ocr.py`)

**Shipped functions:**
- `run_tesseract()` — returns list of {text, conf, bbox} dicts
- `run_paddleocr()` — returns same format, with 4-corner polygons converted to axis-aligned boxes
- Confidence normalized to 0.0–1.0 for both engines

### Engine Comparison (sroie_05.jpg)

| Aspect | Tesseract | PaddleOCR |
|---|---|---|
| Detection unit | Words (150) | Lines (54) |
| Special chars (#) | Often wrong (#→H) | Correct |
| Numbers | Frequent confusion (0→O, 9→S) | Generally accurate |
| Word spacing | Good separation | Sometimes merges words |
| Specific wins | "Beauty" correct | "Bhd" correct, "Reg" correct |
| Specific errors | "Phd" for "Bhd", "Reo" for "Reg" | "Heauty" for "Beauty", "oust" for "must" |
| Handwriting | Fails | Fails |
| Overall confidence | 0.62–0.96 for printed text | 0.93–0.96 for printed text |

**Conclusion:** neither engine is strictly better. PaddleOCR handles symbols and numbers more reliably. Tesseract handles word spacing better. Cross-referencing both engines' outputs could catch errors that either misses individually.

## Day 3 — (upcoming)
- Layout analysis: group OCR output into logical regions
- Field extraction: pull vendor, date, total, items
- Wire full pipeline in `run_pipeline.py`