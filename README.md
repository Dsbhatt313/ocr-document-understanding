# OCR Document Understanding

A Python pipeline that extracts structured information from receipt and invoice images using OCR and spatial layout analysis. Built as a take-home assignment.

## What It Does

Takes a document image (receipt, invoice, tax invoice) as input and outputs structured JSON with extracted fields:

    {
      "vendor_name": "Guardian Health And Beauty Sdn Bhd",
      "total_amount": "38.37",
      "subtotal": "38.37",
      "date": "19/05/18",
      "payment_mode": "CASH",
      "currency": "MYR",
      "document_type": "receipt",
      "confidence_notes": {
        "vendor_name": "high",
        "total_amount": "high",
        "date": "medium"
      }
    }

**Extracted fields:**
- vendor_name — business or company name from the receipt header
- total_amount — final payable amount
- subtotal — pre-tax/pre-discount amount
- date — transaction date in whatever format appears on the receipt
- payment_mode — CASH, VISA, MASTERCARD, UPI, etc.
- currency — MYR, INR, USD, SGD (detected from text cues)
- document_type — receipt, invoice, or tax_invoice
- confidence_notes — per-field reliability assessment (high / medium / low / very_low)

## Pipeline Architecture

The system follows a 6-stage pipeline:

**Stage 1 — Image Input:** accepts JPG, PNG, BMP, TIFF files. Single image or batch directory processing.

**Stage 2 — Preprocessing:** CLAHE (Contrast Limited Adaptive Histogram Equalization) for local contrast enhancement, followed by non-local means denoising. This stage alone gives a +55% improvement in OCR word detection on clean scans, and larger gains on faded or poorly-lit images.

**Stage 3 — OCR:** two engines run in parallel. Tesseract 5.5 (classical, fast, word-level detection) and PaddleOCR (deep learning, line-level detection). Both produce output in a common format: list of {text, confidence, bounding_box} dicts. Confidence is normalized to 0.0-1.0 for both engines.

**Stage 4 — Layout Analysis:** groups OCR words into text lines by vertical position, then splits lines into four receipt regions (header, items, totals, footer) using keyword anchors with position-based fallback. Also pairs labels with their values on the same line (e.g., "TOTAL" on the left, "38.37" on the right).

**Stage 5 — Field Extraction:** pulls structured fields from the layout regions using synonym lists (e.g., "total" / "grand total" / "amount payable" all map to total_amount), regex patterns for dates, word-boundary currency detection, and OCR-aware fallbacks (e.g., Tesseract often reads "RM" as "RH", so both are treated as Malaysian Ringgit).

**Stage 6 — JSON Output:** structured results with metadata, confidence notes, and cross-engine agreement flags.

## Setup Instructions

### System Requirements

- **Python 3.11** (3.12+ may have compatibility issues with PaddlePaddle)
- **Tesseract OCR 5.5+** must be installed separately and available on PATH
- **16GB RAM** recommended (PaddleOCR loads ~50MB of model weights into memory)
- **Windows / Linux / macOS** — tested on Windows 10 with Intel i7 + GTX 1650 Ti

### Step 1: Install Tesseract OCR

Tesseract is a system-level dependency, not a Python package.

**Windows:** download the installer from https://github.com/UB-Mannheim/tesseract/wiki and run it. Default install path is C:\Program Files\Tesseract-OCR\. Make sure "Add to PATH" is checked during installation, or add it manually.

**Linux (Ubuntu/Debian):**

    sudo apt update
    sudo apt install tesseract-ocr

**macOS:**

    brew install tesseract

Verify installation:

    tesseract --version

You should see version 5.x.x.

### Step 2: Clone the Repository

    git clone https://github.com/Dsbhatt313/ocr-document-understanding.git
    cd ocr-document-understanding

### Step 3: Create Virtual Environment

    python -m venv .venv

Activate it:

    # Windows (PowerShell)
    .venv\Scripts\activate

    # Windows (CMD)
    .venv\Scripts\activate.bat

    # Linux / macOS
    source .venv/bin/activate

You should see (.venv) in your terminal prompt.

### Step 4: Install Python Dependencies

    pip install -r requirements.txt

This installs 54 packages including OpenCV, pytesseract, PaddleOCR, PaddlePaddle (CPU), numpy, pandas, matplotlib, and supporting libraries.

**Note on NumPy:** PaddleOCR requires numpy < 2.0. The requirements.txt pins numpy==1.26.4. Do not upgrade numpy or PaddleOCR will break.

**Note on PaddleOCR first run:** the first time you run PaddleOCR, it will download ~50MB of model weights from Baidu servers. This is a one-time download and may take a few minutes depending on your connection. The models are cached in ~/.paddleocr/ and reused on subsequent runs.

### Step 5: Verify Setup

    python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
    python -c "from paddleocr import PaddleOCR; print('PaddleOCR OK')"

Both commands should complete without errors.

## Usage

### Process a Single Image

    python run_pipeline.py --input_file sample_images/sroie_05.jpg

Output JSON is saved to outputs/sroie_05_tesseract.json

### Process All Images in a Directory

    python run_pipeline.py --input_dir sample_images --output_dir outputs

Processes every JPG/PNG/BMP/TIFF in the directory. Saves one JSON per image plus a combined all_results_tesseract.json.

### Choose the OCR Engine

    # Tesseract only (fastest, ~1 sec per image)
    python run_pipeline.py --input_dir sample_images --engine tesseract

    # PaddleOCR only (slower, ~3-5 sec per image on CPU)
    python run_pipeline.py --input_dir sample_images --engine paddleocr

    # Both engines with cross-validation (slowest, most accurate)
    python run_pipeline.py --input_dir sample_images --engine both

The "both" mode runs both engines on each image, then for each field picks the result from whichever engine has higher confidence. It also flags disagreements between engines in the output JSON.

### Run Evaluation

    python evaluate.py --results outputs/all_results_both.json

Compares pipeline output against ground truth (ground_truth.json) and prints field-wise accuracy with per-image details. Saves a detailed evaluation_report.json to the outputs directory.

### Generate Visual Debug Images

    python generate_debug.py

Generates 7 annotated images in outputs/ with:
- Color-coded bounding boxes (blue = header, green = items, orange = totals, gray = footer)
- Region labels on the left margin
- Extracted fields summary panel at the bottom

## Accuracy

Evaluated on 28 SROIE receipt images with verified ground truth.

### Final Results (cross-engine mode)

| Field | Correct | Partial | Wrong | Missing | Exact % | Partial % |
|---|---|---|---|---|---|---|
| vendor_name | 8 | 3 | 17 | 0 | 28.6 | 39.3 |
| date | 19 | 0 | 5 | 4 | 67.9 | 67.9 |
| total_amount | 9 | 4 | 9 | 6 | 32.1 | 46.4 |
| **Overall** | **36** | **7** | **31** | **10** | **42.9** | **51.2** |

### Accuracy Progression

| Improvement | Overall Exact % | Change |
|---|---|---|
| Tesseract only (baseline) | 22.6 | — |
| + Cross-engine agreement | 28.6 | +6.0 |
| + Date format coverage | 42.9 | +14.3 |

Each improvement was measured against ground truth before and after. No accuracy claims without evaluation data.

### Failure Analysis

**Vendor name (17 wrong):** primarily OCR character errors — missing spaces in PaddleOCR output ("MRD.I.VMSDNBHD"), character confusion in Tesseract ("Phd" for "Bhd", "Heauty" for "Beauty"), or wrong line selected as vendor name.

**Date (5 wrong, 4 missing):** wrong dates caused by OCR digit errors (e.g., "25/05/2016" extracted when actual is "25/05/2018" — the "18" was read as "16"). Missing dates from OCR mangling digits beyond regex recognition or non-standard formats (single-digit day/month, month-as-text like "01-NOV-2017").

**Total amount (9 wrong, 6 missing):** wrong totals usually from OCR reading the wrong line (subtotal instead of total, or a price from the items section). Missing totals from OCR failing to detect the total line entirely or the keyword "TOTAL" being too garbled to match.

## Project Structure

    ocr-document-understanding/
    ├── run_pipeline.py              # Main entry point — processes images end-to-end
    ├── evaluate.py                  # Field-wise accuracy evaluation against ground truth
    ├── generate_debug.py            # Generates visual debug images with bounding boxes
    ├── ground_truth.json            # Verified correct answers for 28 SROIE images
    ├── requirements.txt             # 54 pinned Python dependencies
    ├── report.md                    # Detailed project report (2-3 pages)
    │
    ├── sample_images/               # 42 test images
    │   ├── sroie_01.jpg ... sroie_28.jpg      # SROIE dataset receipts
    │   └── personal_01.jpg ... personal_14.jpg # Personal Indian invoices
    │
    ├── outputs/                     # Pipeline outputs (generated, not committed)
    │   ├── sroie_05_tesseract.json  # Per-image JSON results
    │   ├── all_results_both.json    # Combined results for all images
    │   ├── evaluation_report.json   # Detailed accuracy report
    │   └── debug_sroie_05.jpg       # Visual debug images
    │
    ├── src/                         # Pipeline modules
    │   ├── __init__.py              # Package marker (empty)
    │   ├── preprocessing.py         # Stage 2: CLAHE + denoising
    │   ├── ocr.py                   # Stage 3: Tesseract + PaddleOCR wrappers
    │   ├── layout.py                # Stage 4: line grouping + region detection
    │   ├── extraction.py            # Stage 5: field extraction + cross-engine
    │   └── visualize.py             # Visual debug image generation
    │
    ├── notebooks/                   # Jupyter exploration notebooks
    │   ├── 01_first_ocr.ipynb       # Day 1: first Tesseract run, error analysis
    │   ├── 02_preprocessing.ipynb   # Day 2: deskew attempts, CLAHE, thresholding
    │   ├── 03_ocr.ipynb             # Day 2: PaddleOCR setup, engine comparison
    │   └── 04_layout.ipynb          # Day 3: layout, extraction, evaluation
    │
    ├── sroie_txt/                   # SROIE ground truth label files
    │
    └── docs/
        └── requirements.md          # Technical design document

## Dataset

**28 SROIE receipts (sroie_01.jpg through sroie_28.jpg)**
- Southeast Asian retail receipts (Malaysian and Singaporean)
- Downloaded from Kaggle SROIE mirror
- Ground truth labels available: company, date, address, total
- Mix of clean scans and varied print quality

**14 personal images (personal_01.jpg through personal_14.jpg)**
- Real Indian invoices and receipts photographed with a phone
- Sources include: Samsung phone invoice, Vijay Sales (PS5 controller), Shoppers Stop, university fee receipts, Automark Motors service invoice
- All personal identifying information redacted
- Includes variety in lighting, tilt, image quality, and document layout
- Indian-specific characteristics: INR currency, GST/CGST/SGST, "Tax Invoice" headers, DD/MM/YYYY dates

## Key Design Decisions

**Two OCR engines for cross-validation, not just comparison.** Tesseract and PaddleOCR make different types of errors. Tesseract struggles with special characters (#, $) but handles word spacing well. PaddleOCR is better with symbols and numbers but sometimes merges words. Running both and cross-referencing gives us a free sanity check on each extraction.

**Rules-based extraction over ML.** The assignment values explainability. Every extraction rule can be justified: "we look for the keyword TOTAL in the totals region and take the number to its right." A trained ML model might score higher but can't explain why it picked a particular value.

**Preprocessing before OCR, not after.** CLAHE and denoising improve the image before OCR sees it, giving both engines better input. This is more effective than trying to correct OCR errors after the fact.

**Honest evaluation over inflated metrics.** The report documents what works, what doesn't, and why. Partial matches are reported separately from exact matches. Failures are analyzed by category.

## Known Limitations

- **Deskew not implemented** — investigated (minAreaRect approach) but failed on receipt images where text fills the frame. Hough line transform alternative deferred due to time constraints. Tilted phone photos get degraded OCR results.
- **Handwritten text** — neither Tesseract nor PaddleOCR handles handwriting. Both engines produce garbage on handwritten portions.
- **Some date formats missing** — single-digit day/month (e.g., "6/2/2017"), month-as-text (e.g., "01-NOV-2017"), and YYYYMMDD (e.g., "20180428") are not covered.
- **Vendor name accuracy** — limited by underlying OCR character errors and the simple heuristic of "first confident header line."
- **No line-item extraction** — individual product lines are detected in the items region but not parsed into structured line items (description, quantity, unit price, amount).

## Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.11.9 | Runtime |
| Tesseract OCR | 5.5.0 | Classical OCR engine |
| PaddleOCR | 2.8.1 | Deep learning OCR engine |
| PaddlePaddle | 2.6.2 | PaddleOCR backend (CPU) |
| OpenCV | 4.10.0 | Image processing (CLAHE, denoising, thresholding, drawing) |
| NumPy | 1.26.4 | Array operations on images |
| pytesseract | 0.3.13 | Python wrapper for Tesseract |
| matplotlib | 3.9.2 | Image display in notebooks |
| pandas | 2.2.3 | Evaluation tables |

## Author

Dhruvansh Bhatt
- Email: dsbhatt2581@gmail.com
- GitHub: https://github.com/Dsbhatt313
