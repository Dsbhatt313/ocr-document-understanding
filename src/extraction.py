"""
Field extraction module — pulls structured fields from layout regions.

Uses keyword synonyms, regex patterns, and spatial rules to extract:
vendor_name, total_amount, subtotal, date, payment_mode, currency, document_type
"""

import re

def clean_amount(value):
    """
    Clean an extracted monetary amount.
    Fixes common OCR issues like spaces inside numbers.
    
    Examples:
        '38 60'     → '38.60'
        '9 759.70'  → '9759.70'
        '1,207 367' → '1,207.367'
        '. 12.00'   → '12.00'
        '2,00'      → '2.00'
    """
    if not value:
        return value
    
    # Remove leading dots and spaces
    value = value.lstrip(". ")
    
    # Remove trailing dash (negative sign on some receipts like '2.07-')
    value = value.rstrip("-")
    
    # If there's exactly one space and the part after it looks like decimals
    # (1-2 digits), replace space with dot: '38 60' → '38.60'
    parts = value.split()
    if len(parts) == 2:
        if len(parts[1]) <= 2 and parts[1].replace(".", "").isdigit():
            value = parts[0] + "." + parts[1].replace(".", "")
        else:
            # Space is a thousands separator: '9 759.70' → '9759.70'
            value = "".join(parts)
    elif len(parts) > 2:
        value = "".join(parts)
    
    # Replace comma used as decimal separator: '2,00' → '2.00'
    # But keep commas that are thousands separators: '1,207.00' stays
    if "," in value and "." not in value:
        # Comma is likely decimal separator
        last_comma = value.rfind(",")
        after_comma = value[last_comma + 1:]
        if len(after_comma) <= 2:
            value = value[:last_comma] + "." + after_comma
    
    return value


def fix_numeric_chars(value):
    """
    Fix common OCR character confusions in numeric fields.
    
    O → 0, o → 0, I → 1, l → 1, B → 8, S → 5, Z → 2, g → 9
    Only applied to fields expected to be numeric (totals, amounts).
    """
    if not value:
        return value
    
    replacements = {
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "B": "8",
        "S": "5",
        "Z": "2",
        "g": "9",
    }
    
    result = ""
    for char in value:
        if char in replacements:
            result += replacements[char]
        else:
            result += char
    
    return result


def get_confidence_note(conf):
    """Convert a confidence value (0.0-1.0) to a human-readable quality note."""
    if conf >= 0.80:
        return "high"
    elif conf >= 0.60:
        return "medium"
    elif conf >= 0.30:
        return "low"
    else:
        return "very_low"


def detect_currency(full_text):
    """Detect currency from receipt text using word-boundary matching."""
    words = full_text.lower().split()

    if "rm" in words or "ringgit" in words or "myr" in words:
        return "MYR"
    if "₹" in full_text or "inr" in words:
        return "INR"
    for i, w in enumerate(words):
        if w in ("rs", "rs.") and i + 1 < len(words):
            return "INR"
    if "usd" in words:
        return "USD"
    if "$" in full_text and "rm" not in words and "myr" not in words:
        return "USD"
    if "sgd" in words:
        return "SGD"
    if "s$" in full_text:
        return "SGD"
    if "rh" in words:
        return "MYR"

    return "UNKNOWN"


def pair_label_value(line):
    """
    Split a line into label (left) and value (right).
    The rightmost numeric-looking words become the value.
    """
    if not line:
        return ("", "")

    value_words = []
    label_words = []
    found_value = False

    for word in reversed(line):
        text = word["text"]
        cleaned = text.replace(",", "").replace(".", "").replace("-", "")
        cleaned = cleaned.replace("$", "").replace("§", "").replace("*", "")
        cleaned = cleaned.replace("=", "").replace("~", "").replace("|", "")

        if not found_value and (cleaned.isdigit() or text in [".", "-"]):
            value_words.insert(0, text)
            found_value = True
        elif found_value and (cleaned.isdigit() or text in [".", "-", "*", "="]):
            value_words.insert(0, text)
        else:
            label_words.insert(0, text)

    label = " ".join(label_words).strip()
    value = " ".join(value_words).strip()
    value = value.replace("§", "").replace("*", "").replace("=", "")
    value = value.replace("|", "").replace("~", "").strip()

    return label, value


def extract_fields(regions):
    """
    Extract structured fields from layout regions.

    Args:
        regions: dict from identify_regions (header/items/totals/footer)

    Returns:
        Dict with extracted fields.
    """
    fields = {}

    # --- VENDOR NAME ---
    for line in regions["header"]:
        avg_conf = sum(w["conf"] for w in line) / len(line)
        text = " ".join([w["text"] for w in line])
        if avg_conf > 0.5 and len(text) > 5:
            fields["vendor_name"] = text
            break
    

    # --- INVOICE / RECEIPT NUMBER ---
    invoice_keywords = {"invoice no", "invoice number", "inv no", "inv#",
                        "receipt no", "receipt number", "receipt#", "bill no",
                        "bill number", "tax invoice no", "taxinv", "ref no",
                        "reference no", "doc no", "document no"}
    
    for line in regions["header"] + regions["totals"] + regions.get("footer", []):
        line_text = " ".join([w["text"] for w in line]).lower()
        line_raw = " ".join([w["text"] for w in line])
        
        for keyword in invoice_keywords:
            if keyword in line_text:
                _, value = pair_label_value(line)
                # Value must look like an invoice number: contains digits and length >= 3
                if value and any(c.isdigit() for c in value) and len(value) >= 3:
                    # Reject if value is purely a money amount (already captured as total)
                    cleaned = value.replace(",", "").replace(".", "").replace("-", "")
                    reject_words = {"invoice", "receipt", "bill", "tax", "invoice-", "oice"}
                    if value.lower().strip() not in reject_words:
                        if not cleaned.replace(" ", "").isdigit() or len(cleaned) > 4:
                            fields["invoice_number"] = value
                            break
                
                # Fallback: extract alphanumeric content after the keyword
                idx = line_text.find(keyword)
                after = line_raw[idx + len(keyword):].strip()
                after = after.lstrip(":;.# ").strip()
                if after and len(after) >= 3 and any(c.isdigit() for c in after):
                    # Reject common false positives
                    if after.upper().strip().rstrip("-") not in ("INVOICE", "RECEIPT", "BILL", "TAX", "OICE", ""):
                        fields["invoice_number"] = after
                        break
        if "invoice_number" in fields:
            break
    
    # Fallback: search for common invoice number patterns
    if "invoice_number" not in fields:
        all_text = ""
        for region in regions.values():
            for line in region:
                all_text += " ".join([w["text"] for w in line]) + " "
        
        inv_patterns = [
            r'(?:TaxInv|Taxinv|TAXINV)\s*[:#]?\s*\d{3,}',
            r'(?:INV|RCP|RCPT|PIF|SURF)[-#]?\s*[A-Z0-9][-A-Z0-9]{3,}',
        ]
        for pattern in inv_patterns:
            match = re.search(pattern, all_text)
            if match:
                fields["invoice_number"] = match.group().strip()
                break


    # --- TOTAL AMOUNT ---
    total_synonyms = {"total", "grand total", "amount payable", "net amount",
                      "total amount", "amount due", "total (gst incl)",
                      "total(gst incl)", "totalgst incl"}

    for line in regions["totals"]:
        line_text = " ".join([w["text"] for w in line]).lower()
        line_clean = line_text.replace("<", "(").replace(")", ")").replace(">", ")")

        for synonym in total_synonyms:
            if synonym in line_clean:
                _, value = pair_label_value(line)
                if value:
                    fields["total_amount"] = fix_numeric_chars(clean_amount(value))
                    break
        if "total_amount" in fields:
            break

    # --- SUBTOTAL ---
    subtotal_synonyms = {"subtotal", "sub total", "sub-total"}

    for line in regions["totals"]:
        line_text = " ".join([w["text"] for w in line]).lower()
        for synonym in subtotal_synonyms:
            if synonym in line_text:
                _, value = pair_label_value(line)
                if value:
                    fields["subtotal"] = fix_numeric_chars(clean_amount(value))
                    break
        if "subtotal" in fields:
            break

    # --- DATE ---
    # Search ALL regions — date can appear anywhere on the receipt
    all_text = ""
    for region in regions.values():
        for line in region:
            all_text += " ".join([w["text"] for w in line]) + " "
    
    date_patterns = [
        r'\d{2}/\d{2}/\d{4}',           # DD/MM/YYYY
        r'\d{2}-\d{2}-\d{4}',           # DD-MM-YYYY
        r'\d{4}-\d{2}-\d{2}',           # YYYY-MM-DD
        r'\d{2}\.\d{2}\.\d{4}',         # DD.MM.YYYY
        r'\d{1,2}/\d{1,2}/\d{4}',       # D/M/YYYY (flexible digits)
        r'\d{1,2}-\d{1,2}-\d{4}',       # D-M-YYYY (flexible digits)
        r'\d{2}/\d{2}/\d{2}',           # DD/MM/YY
        r'\d{2}-\d{2}-\d{2}',           # DD-MM-YY
        r'\d{1,2}/\d{1,2}/\d{2}',       # D/M/YY (flexible digits)
        r'\d{1,2}-\d{1,2}-\d{2}',       # D-M-YY (flexible digits)
        r'\d{2}-[A-Z]{3}-\d{4}',        # DD-MMM-YYYY (01-NOV-2017)
        r'\d{2}\.\d{2}\.\d{2}',         # DD.MM.YY
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, all_text)
        if match:
            fields["date"] = match.group()
            break

    # --- PAYMENT MODE ---
    payment_synonyms = {"cash", "visa", "mastercard", "credit card",
                        "debit card", "nets", "amex", "upi", "card"}

    for line in regions["totals"]:
        line_text = " ".join([w["text"] for w in line]).lower()
        for synonym in payment_synonyms:
            if synonym in line_text:
                fields["payment_mode"] = synonym.upper()
                break
        if "payment_mode" in fields:
            break

    # --- CURRENCY ---
    full_text = ""
    for region in regions.values():
        for line in region:
            full_text += " ".join([w["text"] for w in line]) + " "

    fields["currency"] = detect_currency(full_text)

    # --- DOCUMENT TYPE ---
    full_lower = full_text.lower()
    if "tax invoice" in full_lower:
        fields["document_type"] = "tax_invoice"
    elif "invoice" in full_lower:
        fields["document_type"] = "invoice"
    else:
        fields["document_type"] = "receipt"

    # --- CONFIDENCE NOTES ---
    confidence_notes = {}
    
    # Vendor name confidence: average of the line we picked
    if "vendor_name" in fields:
        for line in regions["header"]:
            text = " ".join([w["text"] for w in line])
            if text == fields["vendor_name"]:
                avg_conf = sum(w["conf"] for w in line) / len(line)
                confidence_notes["vendor_name"] = get_confidence_note(avg_conf)
                break
    
    # Total/subtotal confidence: confidence of the numeric words in the matched line
    for field_name in ["total_amount", "subtotal"]:
        if field_name in fields:
            for line in regions["totals"]:
                _, value = pair_label_value(line)
                if value and clean_amount(value):
                    line_text = " ".join([w["text"] for w in line]).lower()
                    if field_name == "total_amount" and "total" in line_text:
                        numeric_confs = [w["conf"] for w in line if w["text"].replace(".", "").replace(",", "").isdigit()]
                        if numeric_confs:
                            confidence_notes[field_name] = get_confidence_note(sum(numeric_confs) / len(numeric_confs))
                        break
                    elif field_name == "subtotal" and "subtotal" in line_text:
                        numeric_confs = [w["conf"] for w in line if w["text"].replace(".", "").replace(",", "").isdigit()]
                        if numeric_confs:
                            confidence_notes[field_name] = get_confidence_note(sum(numeric_confs) / len(numeric_confs))
                        break
    
    # Date confidence: if found, mark as medium (regex matched but OCR may have errors)
    if "date" in fields:
        confidence_notes["date"] = "medium"
    
    fields["confidence_notes"] = confidence_notes

    # --- CLEAN INVOICE NUMBER ---
    if "invoice_number" in fields:
        inv = fields["invoice_number"]
        
        # Remove leading dashes and spaces
        inv = inv.lstrip("- ").strip()
        
        # Remove trailing date fragments: "CS00067741 Date14/03/2018" → "CS00067741"
        inv = re.split(r'\s*[Dd]ate\s*', inv)[0].strip()
        
        # Remove "INVOICENO" prefix: "INVOICENO18385" → "18385"
        inv_lower = inv.lower()
        for prefix in ["invoiceno", "invoice no", "receiptno", "receipt no"]:
            if inv_lower.startswith(prefix):
                inv = inv[len(prefix):].lstrip(":;.# ").strip()
                break
        
        # Final rejection: must contain a digit, must not be a common false positive
        inv_upper = inv.upper().strip().rstrip("-")
        reject = {"INVOICE", "RECEIPT", "BILL", "TAX", "TAX INVOICE",
                  "OICE", "REFUNDABLE", ""}
        if inv_upper in reject or not any(c.isdigit() for c in inv):
            del fields["invoice_number"]
        else:
            fields["invoice_number"] = inv

    return fields

def extract_fields_dual(regions_tess, regions_paddle):
    """
    Run extraction on both engines' layouts and pick the best result per field.
    
    For each field, if both engines found a value, pick the one with higher
    confidence. If only one found it, use that one.
    
    Args:
        regions_tess: regions from Tesseract OCR
        regions_paddle: regions from PaddleOCR
    
    Returns:
        Dict with best-of-both extracted fields.
    """
    result_tess = extract_fields(regions_tess)
    result_paddle = extract_fields(regions_paddle)
    
    # Remove confidence_notes temporarily for comparison
    conf_tess = result_tess.pop("confidence_notes", {})
    conf_paddle = result_paddle.pop("confidence_notes", {})
    
    merged = {}
    merged_conf = {}
    
    # Fields to compare
    compare_fields = ["vendor_name", "total_amount", "subtotal", "date",
                      "payment_mode", "currency", "document_type", "invoice_number"]
    
    for field in compare_fields:
        val_t = result_tess.get(field)
        val_p = result_paddle.get(field)
        
        if val_t and val_p:
            # Both found something — pick higher confidence
            conf_t = conf_tess.get(field, "low")
            conf_p = conf_paddle.get(field, "low")
            
            # Convert to numeric for comparison
            conf_rank = {"very_low": 0, "low": 1, "medium": 2, "high": 3}
            
            if conf_rank.get(conf_t, 1) >= conf_rank.get(conf_p, 1):
                merged[field] = val_t
                merged_conf[field] = conf_t
                merged_conf[field + "_source"] = "tesseract"
            else:
                merged[field] = val_p
                merged_conf[field] = conf_p
                merged_conf[field + "_source"] = "paddleocr"
            
            # Flag disagreement
            if val_t != val_p:
                merged_conf[field + "_agreement"] = "disagree"
                merged_conf[field + "_alt"] = val_p if merged[field] == val_t else val_t
            else:
                merged_conf[field + "_agreement"] = "agree"
                
        elif val_t:
            merged[field] = val_t
            merged_conf[field] = conf_tess.get(field, "low")
            merged_conf[field + "_source"] = "tesseract"
        elif val_p:
            merged[field] = val_p
            merged_conf[field] = conf_paddle.get(field, "low")
            merged_conf[field + "_source"] = "paddleocr"
    
    merged["confidence_notes"] = merged_conf
    
    return merged