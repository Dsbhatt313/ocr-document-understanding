"""
Layout analysis module — groups OCR output into lines and receipt regions.

Provides spatial structure for field extraction:
    - group_into_lines: merge individual words into text lines
    - identify_regions: split lines into header/items/totals/footer
    - pair_label_value: separate label from numeric value on a line
"""


def group_into_lines(ocr_results, y_threshold=10):
    """
    Group OCR results into lines based on vertical position.

    Args:
        ocr_results: list of dicts with 'text', 'conf', 'bbox' (x, y, w, h)
        y_threshold: max pixel distance between words on same line

    Returns:
        List of lines sorted top-to-bottom.
        Each line is a list of word-dicts sorted left-to-right.
    """
    if not ocr_results:
        return []

    items = []
    for r in ocr_results:
        x, y, w, h = r["bbox"]
        y_center = y + h / 2
        items.append({"result": r, "y_center": y_center})

    items.sort(key=lambda item: item["y_center"])

    lines = []
    current_line = [items[0]]

    for item in items[1:]:
        if abs(item["y_center"] - current_line[-1]["y_center"]) <= y_threshold:
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
    lines.append(current_line)

    sorted_lines = []
    for line in lines:
        line.sort(key=lambda item: item["result"]["bbox"][0])
        sorted_lines.append([item["result"] for item in line])

    return sorted_lines


def identify_regions(lines, image_height):
    """
    Split lines into receipt regions using keyword anchors + position fallback.

    Args:
        lines: output of group_into_lines
        image_height: height of the image in pixels

    Returns:
        Dict with keys 'header', 'items', 'totals', 'footer'.
        Each value is a list of lines.
    """
    header_keywords = {"tel", "phone", "fax", "reg", "gst", "company", "address"}
    totals_keywords = {"subtotal", "total", "cash", "change", "rounding", "visa",
                       "mastercard", "payment", "tendered", "balance"}
    footer_keywords = {"thank", "exchange", "return", "refund", "receipt",
                       "condition", "warranty", "goods", "medicine"}

    def line_text(line):
        return " ".join([w["text"] for w in line]).lower()

    def line_has_keyword(line, keywords):
        text = line_text(line)
        return any(kw in text for kw in keywords)

    # Find totals start (first SUBTOTAL or TOTAL line)
    totals_start = None
    for i, line in enumerate(lines):
        if line_has_keyword(line, {"subtotal", "total"}):
            totals_start = i
            break

    # Find footer start (first footer keyword after totals)
    footer_start = None
    search_from = totals_start + 1 if totals_start is not None else len(lines) // 2
    for i in range(search_from, len(lines)):
        if line_has_keyword(lines[i], footer_keywords):
            footer_start = i
            break

    # Find header end (last header keyword line before items)
    header_end = 0
    for i, line in enumerate(lines):
        if totals_start is not None and i >= totals_start:
            break
        if line_has_keyword(line, header_keywords):
            header_end = i + 1

    # Fallbacks
    if header_end == 0:
        for i, line in enumerate(lines):
            y = line[0]["bbox"][1]
            if y / image_height > 0.20:
                header_end = i
                break

    if totals_start is None:
        for i, line in enumerate(lines):
            y = line[0]["bbox"][1]
            if y / image_height > 0.60:
                totals_start = i
                break
        if totals_start is None:
            totals_start = len(lines)

    if footer_start is None:
        for i, line in enumerate(lines):
            y = line[0]["bbox"][1]
            if y / image_height > 0.75:
                footer_start = i
                break
        if footer_start is None:
            footer_start = len(lines)

    return {
        "header": lines[:header_end],
        "items":  lines[header_end:totals_start],
        "totals": lines[totals_start:footer_start],
        "footer": lines[footer_start:],
    }


def pair_label_value(line):
    """
    Split a line into label (left) and value (right).

    The rightmost numeric-looking words become the value.
    Everything else is the label.

    Args:
        line: list of word-dicts from one line

    Returns:
        (label_text, value_text) tuple.
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