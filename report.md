# OCR Document Understanding — Project Report

## 1. Approach

The system follows a 6-stage pipeline: image input, preprocessing, OCR text extraction, layout analysis, field extraction, and JSON output with evaluation. The core design philosophy was explainability over raw accuracy — every extraction rule can be justified, and every failure can be traced to a specific pipeline stage.

Two OCR engines (Tesseract and PaddleOCR) run on each image. Rather than picking one engine, we use cross-engine agreement as a quality signal: when both engines extract the same value for a field, confidence is high. When they disagree, the system picks the higher-confidence result and flags the disagreement in the output JSON.

The pipeline is rules-based (keyword matching, regex, spatial reasoning) rather than ML-based for field extraction. This was a deliberate choice — the assignment rewards explainability, and every rule in the system can be traced back to a specific observation about receipt layouts.

## 2. Dataset Preparation

The dataset contains 42 document images split into two groups:

**28 SROIE receipts (sroie_01.jpg through sroie_28.jpg):** Southeast Asian retail receipts (Malaysian and Singaporean) downloaded from the Kaggle SROIE dataset mirror. These are pre-scanned images with generally clean quality. Ground truth labels (company, date, address, total) were available as JSON-in-text files and converted into our ground_truth.json format.

**14 personal images (personal_01.jpg through personal_14.jpg):** real Indian invoices and receipts photographed with a phone camera. Sources include a Samsung phone invoice, Vijay Sales PS5 controller invoice, Shoppers Stop receipt, university fee receipts, and an Automark Motors service invoice. All personal identifying information (name, address, phone) was redacted using thick black markup before inclusion.

The dataset was designed to include variety across several dimensions: clean scans vs phone photos, different lighting conditions, different receipt layouts, multiple currencies (MYR, INR), different document types (receipt, invoice, tax invoice), and different text densities (short receipts vs multi-page invoices).

**Dataset split characteristics:**

| Characteristic | SROIE (28) | Personal (14) |
|---|---|---|
| Source | Scanned | Phone camera |
| Region | Malaysia/Singapore | India |
| Currency | MYR | INR |
| Quality | Generally clean | Variable (some tilted, shadows) |
| Ground truth | Available (from dataset) | Not available (manual verification only) |

## 3. Image Quality Handling

### What was implemented

**CLAHE (Contrast Limited Adaptive Histogram Equalization):** divides the image into an 8x8 grid of tiles and equalizes contrast independently in each tile. This handles uneven lighting — one corner of a receipt may be bright while another is in shadow. CLAHE boosted text from each region independently. Measured improvement: +55% more words detected with confidence > 50 on a clean scan (78 words to 121 words). The improvement on poorly-lit phone photos is larger.

**Non-local means denoising:** removes sensor noise and compression artifacts from phone camera images while preserving text edges. Unlike simple blurring (which smears text), non-local means finds similar-looking patches across the image and averages them, so sharp text edges survive while random noise is smoothed away. Measured improvement: +7 additional confident words on top of CLAHE on a clean scan.

**Preprocessing order matters:** geometric corrections (deskew) should come before pixel-value corrections (CLAHE, denoise). We applied CLAHE first, then denoising, because deskew was not implemented (see below).

### What was investigated but not shipped

**Deskew (rotation correction):** we attempted angle detection using cv2.minAreaRect on the binary text pixel cloud. The approach failed because on receipt images, text fills the entire frame — vendor name at the top, items in the middle, totals at the bottom, text spanning the full width. The bounding rectangle of the text pixel cloud equals the image rectangle itself, so minAreaRect always reports the image as axis-aligned regardless of actual tilt. We tried edge cropping and connected-components filtering to isolate text blobs from border artifacts, but the fundamental issue remained. The alternative approach (Hough line transform on text baselines) was deferred due to time constraints. SROIE receipts are unaffected (already well-aligned scans). Personal phone photos with significant tilt remain a known limitation.

**Adaptive thresholding:** converts the image to pure black-and-white using locally-adaptive thresholds. The thresholded image looked clean to human eyes, but Tesseract detected zero words from it. Tesseract performs its own internal binarization, and feeding it a pre-binarized image causes the two binarization steps to conflict. The function was kept available for potential PaddleOCR use but is not part of the default pipeline.

### Key finding

Plain grayscale input (28 words detected) performed worse than raw BGR color input (78 words) for Tesseract. This contradicts the common assumption that grayscale helps OCR. Tesseract's internal preprocessing handles color-to-grayscale conversion with its own optimizations. The lesson: either pass color images to Tesseract (let it do its own thing) or apply meaningful enhancement (CLAHE + denoise) after grayscale conversion. Never pass plain grayscale.

## 4. Layout Understanding

Layout analysis converts a flat list of OCR detections (words with bounding boxes) into structured document regions. Three operations are performed:

**Line grouping:** OCR engines return individual words (Tesseract: 150 words) or text lines (PaddleOCR: 54 lines). We group words into lines by vertical position — words whose vertical centers are within 10 pixels of each other are considered to be on the same line. Lines are sorted top-to-bottom, words within each line sorted left-to-right. This reconstructs reading order from scattered bounding boxes.

**Region detection:** lines are split into four receipt regions: header (vendor info), items (product lines), totals (subtotal, total, cash, change), and footer (thank-you message, store policies, date/time). We use a hybrid approach — keyword anchors with position-based fallback. The first line containing "SUBTOTAL" or "TOTAL" marks the boundary between items and totals. The first line containing "thank", "exchange", or "return" after totals marks the start of footer. The last line containing "tel", "reg", "company", or "gst" before items marks the end of header. If keywords are not found (OCR mangled them), we fall back to position-based percentages (top 20% = header, etc.).

**Label-value pairing:** receipt lines follow a label-on-left, value-on-right pattern ("TOTAL    38.37"). We walk from right to left through words on a line — numeric-looking words become the value, everything else becomes the label. This simple spatial rule correctly paired all key fields in our testing.

**Why pure position-based splitting failed:** our first attempt used fixed percentages (top 25% = header, 25-55% = items, etc.). This placed vendor registration numbers ("Tel", "Company Reg", "GST Reg") in the items region instead of header, and placed "SUBTOTAL" and "TOTAL" in items instead of totals. Receipts vary too much in proportions for fixed percentages to work.

## 5. Field Extraction

Each field is extracted using a combination of keyword matching, regex patterns, and spatial rules:

**vendor_name:** first line in the header region with average OCR confidence > 0.5 and length > 5 characters. This skips garbage lines (handwriting, scan artifacts) and picks the most prominent text near the top of the receipt.

**total_amount:** searches the totals region for lines matching any synonym from a list ("total", "grand total", "amount payable", "total (gst incl)", etc.). Uses label-value pairing to extract the numeric portion. Post-processing cleans up OCR artifacts: spaces inside numbers ("38 60" becomes "38.60"), character substitution in numeric contexts (O to 0, B to 8, I to 1, S to 5), and comma/dot normalization.

**subtotal:** same approach as total_amount with subtotal-specific synonyms ("subtotal", "sub total", "sub-total").

**date:** regex search across all receipt regions (dates can appear in header, totals, or footer). Patterns tried in priority order: DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, DD.MM.YYYY, DD/MM/YY, DD-MM-YY, DD.MM.YY. The DD-MM-YY pattern was added after evaluation showed most SROIE dates use this format — this single addition improved date accuracy from 25% to 67.9%.

**currency:** word-boundary matching against known indicators. "RM" or "MYR" in the text triggers Malaysian Ringgit. "INR" or standalone "Rs" triggers Indian Rupee. Includes an OCR-aware fallback: Tesseract often reads "RM" as "RH" (M and H look similar), so "RH" also triggers MYR detection. The word-boundary approach avoids false positives from substring matches (e.g., "rs" inside "returns" previously triggered false INR detection).

**payment_mode:** keyword search in the totals region for "cash", "visa", "mastercard", "debit card", "upi", etc.

**document_type:** keyword search across full text. "Tax invoice" triggers tax_invoice, "invoice" triggers invoice, default is receipt.

### Cross-engine agreement

When running with --engine both, both Tesseract and PaddleOCR process each image independently through the full pipeline. For each field, the system compares results and picks the higher-confidence value. Disagreements are flagged in the output JSON with both values shown, allowing downstream consumers to make their own judgment.

### Confidence notes

Each field in the output includes a quality assessment based on the OCR confidence of the source words: high (conf >= 0.8), medium (conf >= 0.6), low (conf >= 0.3), very_low (conf < 0.3).

## 6. Evaluation Results

### Field detection rates across the full dataset (42 images, cross-engine mode)

| Metric | SROIE (28 images) | Personal (14 images) | All (42 images) |
|---|---|---|---|
| vendor_name found | 28/28 (100%) | 13/14 (93%) | 41/42 (98%) |
| total_amount found | 22/28 (79%) | 5/14 (36%) | 27/42 (64%) |
| date found | 24/28 (86%) | 6/14 (43%) | 30/42 (71%) |

Personal images have lower detection rates because they include complex multi-section invoices (Automark Motors service invoice), varied layouts, and some images with significant tilt or poor lighting.

### Accuracy against ground truth (28 SROIE images with verified labels)

| Field | Correct | Partial | Wrong | Missing | Exact % | Partial % |
|---|---|---|---|---|---|---|
| vendor_name | 8 | 3 | 17 | 0 | 28.6 | 39.3 |
| date | 19 | 0 | 5 | 4 | 67.9 | 67.9 |
| total_amount | 9 | 4 | 9 | 6 | 32.1 | 46.4 |
| **Overall** | **36** | **7** | **31** | **10** | **42.9** | **51.2** |

### Accuracy progression through the project

| Improvement | Overall Exact % | Change |
|---|---|---|
| Tesseract only (baseline) | 22.6 | — |
| + Cross-engine agreement | 28.6 | +6.0 |
| + Date format coverage | 42.9 | +14.3 |

Each improvement was measured against ground truth. No accuracy claims without evaluation data.

## 7. Failure Cases

### Where the system worked well

- **sroie_05.jpg (Guardian Health And Beauty):** vendor, total (38.37), date (19/05/18), currency (MYR), payment mode (CASH) all correctly extracted.
- **sroie_26.jpg (Kedai Papan Yew Chuan):** vendor and date (11/05/2018) correct. Total extracted but slightly wrong (65.00 vs 68.90 — OCR read wrong digits).
- **sroie_19.jpg (Grandma Homes Restaurant):** vendor and date (13/06/2018) correct from cross-engine mode where Tesseract alone missed the date.
- **personal_01.jpg (Automark Motors):** vendor correctly identified, total (715.28) extracted, document correctly classified as tax_invoice.

### Where the system failed

**Vendor name errors (17/28 wrong):**
- OCR character errors: "Phd" instead of "Bhd", "Heauty" instead of "Beauty", "SANYO" instead of "SANYU"
- PaddleOCR merging words: "MRD.I.VMSDNBHD" instead of "MR. D.I.Y. (M) SDN BHD"
- Wrong line selected: "sscc= PLEASE VISIT US AGAIN" picked instead of the actual vendor name "CHECKERS HYPERMARKET" (the actual name wasn't in the header region due to receipt layout)
- Character substitution side effect: "99 SPEED MART" became "99.5PEE0MART" because S was substituted to 5 (this substitution should only apply to numeric fields, not vendor names — this is a bug)

**Date errors (5/28 wrong, 4/28 missing):**
- OCR digit errors: "25/05/2016" extracted when actual is "25/05/2018" (the "18" was misread as "16")
- Missing: formats not covered by our regex patterns — single-digit day/month ("6/2/2017"), YYYYMMDD ("20180428"), month-as-text ("01-NOV-2017")
- One image had no date visible to either OCR engine

**Total amount errors (9/28 wrong, 6/28 missing):**
- Wrong line read: pipeline extracted subtotal or an item price instead of the final total
- OCR digit errors: "27.27" extracted vs "27.25" actual (two-cent difference from OCR misreading)
- Garbled numbers: "1,201361.46" extracted for what should be "361.46" (OCR merged multiple numbers from adjacent lines)

## 8. Challenges Faced

**Deskew failure:** the most time-consuming challenge was attempting to build document deskew using minAreaRect. Three iterations of fixes (edge cropping, connected-components filtering, angle-convention handling) all failed because the fundamental assumption — that text occupies a smaller region than the full image — doesn't hold for receipts. This consumed significant Day 2 time before we abandoned it.

**Tesseract vs pre-binarized images:** discovering that adaptive thresholding produces 0 detected words in Tesseract was unexpected. The conflict between external binarization and Tesseract's internal binarization is documented in Tesseract's issue tracker but not prominently mentioned in tutorials.

**OCR engine differences:** Tesseract returns individual words, PaddleOCR returns full lines. Building a common format and ensuring layout analysis works with either granularity required careful handling of the y_threshold parameter and the label-value pairing logic.

**Date format diversity:** the SROIE dataset uses multiple date formats within the same dataset (DD/MM/YYYY, DD-MM-YY, DD/MM/YY). Initially only DD/MM/YYYY was covered, catching 4/28 dates. Adding DD-MM-YY alone recovered 15 additional dates.

**False positive patterns:** the YYYYMMDD regex pattern matched 8-digit product codes and registration numbers as dates. The \d{8} pattern was removed after evaluation showed more false positives than true matches.

## 9. What I Would Improve With More Time

**Hough-based deskew:** the minAreaRect approach failed, but Hough line transform (detecting straight lines formed by text baselines and measuring their angles) would likely work. This would help the 14 personal phone photos that currently suffer from tilt-related OCR degradation.

**Fuzzy vendor name matching:** currently vendor names must match exactly after lowercasing. Adding edit-distance or token-overlap matching (e.g., "Guardian Health And Beauty Sdn Phd" would partially match "GUARDIAN HEALTH AND BEAUTY SDN BHD" on a token basis despite "Phd" vs "BHD") would significantly improve the vendor accuracy metric.

**Line-item extraction:** the items region is detected but individual product lines are not parsed into structured data (description, quantity, unit price, amount). This would require column detection within the items region and alignment of text across columns.

**More date formats:** single-digit day/month ("6/2/2017"), month-as-text ("01-NOV-2017", "March 14, 2018"), and YYYYMMDD with validation (rejecting 8-digit numbers that aren't valid dates) would recover most of the remaining missing dates.

**Word-pair post-correction:** using co-occurrence statistics to fix OCR errors. "Company Rea" would be corrected to "Company Reg" because "Company Reg" is a high-frequency bigram on receipts while "Company Rea" is not.

**GPU-accelerated PaddleOCR:** currently running on CPU (~3-5 seconds per image). A GPU build would reduce this to under 1 second, making batch processing of large datasets practical.

## 10. External Tools and AI Assistance

**Libraries used:** OpenCV (image processing), pytesseract (Tesseract wrapper), PaddleOCR (deep learning OCR), NumPy (array operations), matplotlib (visualization), pandas (evaluation tables).

**AI assistance:** Claude (Anthropic) was used as a coding assistant throughout the project for code generation, debugging, architecture decisions, and documentation. All code was reviewed, tested, and understood before inclusion. The exploration notebooks document the iterative development process including failed attempts and debugging sessions.

**Dataset source:** SROIE receipt images from the Kaggle ICDAR 2019 SROIE dataset. Personal images were self-collected and redacted.