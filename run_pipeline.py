"""
run_pipeline.py — Process document images and extract structured fields.

Usage:
    python run_pipeline.py --input_dir sample_images --output_dir outputs
    python run_pipeline.py --input_dir sample_images --output_dir outputs --engine paddleocr
    python run_pipeline.py --input_file sample_images/sroie_05.jpg
"""

import argparse
import json
import cv2
import sys
from pathlib import Path

from src.preprocessing import preprocess
from src.ocr import run_tesseract, run_paddleocr
from src.layout import group_into_lines, identify_regions
from src.extraction import extract_fields


def process_image(image_path, engine="tesseract"):
    """Run the full pipeline on a single image."""
    image = cv2.imread(str(image_path))
    if image is None:
        return {"error": f"Could not load {image_path}"}

    image_height = image.shape[0]

    # Stage 2: Preprocess
    preprocessed = preprocess(image)

    if engine == "both":
        # Run both engines
        tess_results = run_tesseract(preprocessed)
        paddle_results = run_paddleocr(image_path)

        tess_lines = group_into_lines(tess_results)
        paddle_lines = group_into_lines(paddle_results)

        regions_tess = identify_regions(tess_lines, image_height)
        regions_paddle = identify_regions(paddle_lines, image_height)

        from src.extraction import extract_fields_dual
        fields = extract_fields_dual(regions_tess, regions_paddle)
        fields["ocr_engine"] = "both"
    else:
        # Single engine
        if engine == "tesseract":
            ocr_results = run_tesseract(preprocessed)
        else:
            ocr_results = run_paddleocr(image_path)

        lines = group_into_lines(ocr_results)
        regions = identify_regions(lines, image_height)
        fields = extract_fields(regions)
        fields["ocr_engine"] = engine

    fields["source_file"] = Path(image_path).name

    return fields


def main():
    parser = argparse.ArgumentParser(description="OCR Document Understanding Pipeline")
    parser.add_argument("--input_dir", type=str, help="Directory of images to process")
    parser.add_argument("--input_file", type=str, help="Single image file to process")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for JSON results")
    parser.add_argument("--engine", type=str, default="tesseract",
                        choices=["tesseract", "paddleocr", "both"],
                        help="OCR engine to use (default: tesseract)")
    args = parser.parse_args()

    if not args.input_dir and not args.input_file:
        parser.error("Provide either --input_dir or --input_file")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Collect image files
    if args.input_file:
        image_files = [Path(args.input_file)]
    else:
        input_dir = Path(args.input_dir)
        image_files = sorted(
            [f for f in input_dir.iterdir()
             if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}]
        )

    if not image_files:
        print("No image files found.")
        sys.exit(1)

    print(f"Processing {len(image_files)} image(s) with {args.engine}...\n")

    all_results = []

    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] {image_path.name}...", end=" ")

        try:
            result = process_image(image_path, engine=args.engine)
            all_results.append(result)

            # Save individual JSON
            out_file = output_dir / f"{image_path.stem}_{args.engine}.json"
            with open(out_file, "w") as f:
                json.dump(result, f, indent=2)

            print("OK")

        except Exception as e:
            print(f"FAILED: {e}")
            all_results.append({
                "source_file": image_path.name,
                "error": str(e),
            })

    # Save combined results
    combined_file = output_dir / f"all_results_{args.engine}.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDone. {len(all_results)} results saved to {output_dir}/")
    print(f"Combined results: {combined_file}")


if __name__ == "__main__":
    main()