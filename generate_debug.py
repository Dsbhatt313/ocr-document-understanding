"""Generate visual debug outputs for 5+ images."""

import cv2
import json
from pathlib import Path

from src.preprocessing import preprocess
from src.ocr import run_tesseract
from src.layout import group_into_lines, identify_regions
from src.extraction import extract_fields
from src.visualize import draw_debug_image


def generate(image_path, output_dir):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  SKIP: could not load {image_path.name}")
        return

    preprocessed = preprocess(image)
    ocr_results = run_tesseract(preprocessed)
    lines = group_into_lines(ocr_results)
    regions = identify_regions(lines, image.shape[0])
    fields = extract_fields(regions)

    debug_img = draw_debug_image(image, ocr_results, regions, fields)

    out_path = output_dir / f"debug_{image_path.stem}.jpg"
    cv2.imwrite(str(out_path), debug_img)
    print(f"  Saved: {out_path.name}")


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Pick a mix: 3 SROIE + 2 personal for variety
    images = [
        "sample_images/sroie_05.jpg",
        "sample_images/sroie_03.jpg",
        "sample_images/sroie_12.jpg",
        "sample_images/sroie_19.jpg",
        "sample_images/sroie_21.jpg",
        "sample_images/personal_01.jpg",
        "sample_images/personal_09.jpg",
    ]

    print(f"Generating {len(images)} debug images...\n")

    for img_path in images:
        path = Path(img_path)
        print(f"[{path.name}]")
        generate(path, output_dir)

    print(f"\nDone. Debug images saved to {output_dir}/")


if __name__ == "__main__":
    main()