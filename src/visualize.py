"""
Visual debug module — draws bounding boxes, region labels, and extraction
results on document images for visual inspection.

Usage:
    from src.visualize import draw_debug_image
    debug_img = draw_debug_image(image, ocr_results, regions, fields)
    cv2.imwrite("debug_output.jpg", debug_img)
"""

import cv2
import numpy as np


# Colors (BGR) for each region
REGION_COLORS = {
    "header": (255, 150, 0),    # blue
    "items":  (0, 200, 0),      # green
    "totals": (0, 100, 255),    # orange
    "footer": (180, 180, 180),  # gray
}


def draw_bounding_boxes(image, ocr_results, regions=None):
    """
    Draw OCR bounding boxes on the image, color-coded by region.

    Args:
        image: BGR numpy array.
        ocr_results: list of dicts with 'text', 'conf', 'bbox'.
        regions: dict from identify_regions (optional — if provided,
                 boxes are colored by region; otherwise all green).

    Returns:
        Annotated BGR numpy array (copy of input).
    """
    annotated = image.copy()

    if regions:
        # Build a lookup: word text+bbox → region name
        word_to_region = {}
        for region_name, lines in regions.items():
            for line in lines:
                for word in line:
                    key = (word["text"], word["bbox"])
                    word_to_region[key] = region_name

        for word in ocr_results:
            key = (word["text"], word["bbox"])
            region = word_to_region.get(key, "items")
            color = REGION_COLORS.get(region, (0, 255, 0))
            x, y, w, h = word["bbox"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
    else:
        for word in ocr_results:
            x, y, w, h = word["bbox"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return annotated


def draw_region_labels(image, regions):
    """
    Draw region name labels on the left side of the image.

    Args:
        image: BGR numpy array (already has bounding boxes drawn).
        regions: dict from identify_regions.

    Returns:
        Annotated BGR numpy array.
    """
    annotated = image.copy()

    for region_name, lines in regions.items():
        if not lines:
            continue

        # Find the vertical center of this region
        all_y = []
        for line in lines:
            for word in line:
                x, y, w, h = word["bbox"]
                all_y.append(y + h // 2)

        if all_y:
            mid_y = int(np.mean(all_y))
            color = REGION_COLORS.get(region_name, (0, 255, 0))
            label = region_name.upper()
            cv2.putText(annotated, label, (5, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return annotated


def draw_extracted_fields(image, fields):
    """
    Draw extracted field values in a box at the bottom of the image.

    Args:
        image: BGR numpy array.
        fields: dict of extracted fields.

    Returns:
        New image with a field summary panel appended at the bottom.
    """
    h, w = image.shape[:2]

    # Create a white panel below the image
    display_fields = ["vendor_name", "total_amount", "date",
                      "currency", "payment_mode", "document_type"]
    panel_height = 30 * (len(display_fields) + 1)
    panel = np.ones((panel_height, w, 3), dtype=np.uint8) * 255

    # Draw field values
    y_pos = 25
    cv2.putText(panel, "EXTRACTED FIELDS:", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    y_pos += 30

    for field in display_fields:
        value = fields.get(field, "—")
        text = f"{field}: {value}"
        cv2.putText(panel, text, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_pos += 30

    # Stack image and panel vertically
    combined = np.vstack([image, panel])
    return combined


def draw_debug_image(image, ocr_results, regions, fields):
    """
    Full visual debug output: bounding boxes + region labels + extracted fields.

    Args:
        image: BGR numpy array from cv2.imread.
        ocr_results: list of OCR result dicts.
        regions: dict from identify_regions.
        fields: dict of extracted fields.

    Returns:
        Annotated BGR numpy array with all debug info.
    """
    result = draw_bounding_boxes(image, ocr_results, regions)
    result = draw_region_labels(result, regions)
    result = draw_extracted_fields(result, fields)
    return result