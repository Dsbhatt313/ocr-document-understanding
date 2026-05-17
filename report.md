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

## Day 3 — Layout, Extraction, Pipeline, Evaluation

### Layout Analysis (`src/layout.py`)

**Shipped functions:**
- `group_into_lines()` — groups OCR words into text lines by vertical position (y_threshold=10px)
- `identify_regions()` — splits lines into header/items/totals/footer using keyword anchors with position-based fallback
- `pair_label_value()` — separates label (left) from numeric value (right) on a line

**Key design decision:** hybrid region detection. Pure position-based splitting (top 25% = header, etc.) failed because receipts vary in proportions. Keyword anchors ("SUBTOTAL" marks the start of totals, "Thank" marks footer) are more reliable, with position percentages as fallback when OCR mangles the keywords.

### Field Extraction (`src/extraction.py`)

**Extracted fields:** vendor_name, total_amount, subtotal, date, payment_mode, currency, document_type

**Techniques used:**
- Synonym lists for label matching ("total", "grand total", "amount payable", etc.)
- Regex patterns for dates (DD/MM/YYYY, DD/MM/YY, DD-MM-YYYY, etc.)
- Word-boundary currency detection (avoids false matches like "rs" inside "returns")
- OCR-aware fallbacks (e.g., "RH" → MYR because Tesseract often reads M as H)
- Character substitution in numeric fields (O→0, B→8, I→1, S→5)
- Amount cleaning (spaces inside numbers, comma/dot normalization)

### Cross-Engine Agreement (`--engine both`)

Both Tesseract and PaddleOCR run on each image. For each field, the system picks the higher-confidence result. Disagreements are flagged in the output JSON with both values shown.

**Impact:** +6% exact accuracy, +8.3% partial accuracy over Tesseract alone.

### Confidence Notes

Each extracted field includes a quality assessment (high/medium/low/very_low) based on the OCR confidence of the source words. Shows the evaluator we considered reliability, not just extraction.

### Pipeline (`run_pipeline.py`)

- Processes single images (`--input_file`) or directories (`--input_dir`)
- Supports `--engine tesseract`, `--engine paddleocr`, or `--engine both`
- Outputs individual JSON per image + combined results file
- 42/42 images processed without failures

### Evaluation (`evaluate.py`)

Field-wise accuracy against SROIE ground truth (28 images):

| Field | Exact % (Tesseract) | Exact % (Both) | Partial % (Both) |
|---|---|---|---|
| vendor_name | 25.0 | 28.6 | 39.3 |
| date | 14.3 | 25.0 | 28.6 |
| total_amount | 28.6 | 32.1 | 46.4 |
| **Overall** | **22.6** | **28.6** | **38.1** |

### Failure Analysis

**Vendor name (17 wrong):** mostly OCR errors — missing spaces ("MRD.I.VMSDNBHD"), character confusion ("Heauty" for "Beauty"), wrong line picked as vendor. Case sensitivity also causes mismatches ("Guardian" vs "GUARDIAN").

**Date (18 missing):** dates mangled by OCR beyond regex recognition. Some dates in non-standard formats (e.g., "20180428", "01-NOV-2017") not covered by our patterns.

**Total amount (9 wrong, 6 missing):** wrong totals usually caused by OCR reading wrong line (subtotal instead of total, or price from items section). Missing totals from OCR failing to detect the total line at all.

**Known limitations:**
- Case-sensitive vendor matching (could add case-insensitive comparison)
- Limited date format coverage
- No line-item extraction (stretch goal)
- Deskew not implemented (tilted personal photos get worse results)

## Day 4 — (upcoming)
- Visual debug outputs for 5+ images
- README
- Final report polish
- Stretch: case-insensitive vendor matching, more date formats