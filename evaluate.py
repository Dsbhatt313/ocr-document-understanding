"""
evaluate.py — Field-wise accuracy evaluation against ground truth.

Usage:
    python evaluate.py --results outputs/all_results_both.json
    python evaluate.py --results outputs/all_results_tesseract.json
"""

import argparse
import json
from pathlib import Path


def normalize(value):
    """Normalize a value for comparison — lowercase, strip whitespace and currency prefixes."""
    if not value:
        return ""
    value = str(value).strip().lower()
    # Remove common currency prefixes for fair comparison
    value = value.replace("rm", "").replace("rm ", "").strip()
    return value


def normalize_date(value):
    """Normalize date formats for comparison."""
    value = normalize(value)
    if not value:
        return ""
    # Convert 4-digit year to 2-digit for comparison
    # 08-01-2016 → 08-01-16, 14/03/2018 → 14/03/18
    import re
    value = re.sub(r'(\d{2}[/\-\.])(\d{2}[/\-\.])(\d{4})', 
                   lambda m: m.group(1) + m.group(2) + m.group(3)[-2:], value)
    # Normalize separators: all become /
    value = value.replace("-", "/").replace(".", "/")
    return value


def normalize_vendor(predicted, actual):
    """
    Token-overlap matching for vendor names.
    Splits both into words and checks how many words overlap.
    
    Returns:
        'correct' if >80% of actual tokens found in predicted
        'partial' if >50% of actual tokens found in predicted
        'wrong' otherwise
    """
    pred_tokens = set(normalize(predicted).split())
    actual_tokens = set(normalize(actual).split())
    
    if not actual_tokens or not pred_tokens:
        return None  # let normal comparison handle it
    
    # How many actual tokens appear in predicted?
    overlap = actual_tokens & pred_tokens
    overlap_ratio = len(overlap) / len(actual_tokens)
    
    if overlap_ratio >= 0.8:
        return "correct"
    elif overlap_ratio >= 0.5:
        return "partial"
    return None  # fall through to normal comparison


def compare_field(predicted, actual, field_name=""):
    """
    Compare predicted vs actual field value.
    
    Returns:
        'correct', 'partial', 'wrong', 'missing', or 'no_ground_truth'
    """
    pred = normalize(predicted)
    actual_norm = normalize(actual)
    
    if not actual_norm:
        return "no_ground_truth"
    
    if not pred:
        return "missing"
    
    # Date-specific normalization
    if field_name == "date":
        pred = normalize_date(predicted)
        actual_norm = normalize_date(actual)
    
    if pred == actual_norm:
        return "correct"
    
    if pred in actual_norm or actual_norm in pred:
        return "partial"
    
    # Vendor-specific: token overlap matching
    if field_name == "vendor_name":
        vendor_result = normalize_vendor(predicted, actual)
        if vendor_result:
            return vendor_result

    return "wrong"


def evaluate(results, ground_truth):
    """
    Evaluate pipeline results against ground truth.
    
    Args:
        results: list of dicts from pipeline output
        ground_truth: dict keyed by filename with correct values
    
    Returns:
        (per_image_results, summary) tuple
    """
    fields_to_eval = ["vendor_name", "date", "total_amount"]
    
    per_image = []
    field_counts = {f: {"correct": 0, "partial": 0, "wrong": 0, "missing": 0, "total": 0}
                    for f in fields_to_eval}
    
    for result in results:
        source = result.get("source_file", "")
        
        if source not in ground_truth:
            continue
        
        gt = ground_truth[source]
        image_eval = {"source_file": source}
        
        for field in fields_to_eval:
            pred_val = result.get(field, "")
            actual_val = gt.get(field, "")
            
            status = compare_field(pred_val, actual_val, field_name = field )
            image_eval[field] = {
                "predicted": pred_val if pred_val else "—",
                "actual": actual_val if actual_val else "—",
                "status": status,
            }
            
            if status != "no_ground_truth":
                field_counts[field][status] += 1
                field_counts[field]["total"] += 1
        
        per_image.append(image_eval)
    
    # Compute accuracy per field
    summary = {}
    for field in fields_to_eval:
        counts = field_counts[field]
        total = counts["total"]
        if total > 0:
            exact_acc = counts["correct"] / total * 100
            partial_acc = (counts["correct"] + counts["partial"]) / total * 100
        else:
            exact_acc = 0
            partial_acc = 0
        
        summary[field] = {
            "correct": counts["correct"],
            "partial": counts["partial"],
            "wrong": counts["wrong"],
            "missing": counts["missing"],
            "total": total,
            "exact_accuracy": round(exact_acc, 1),
            "partial_accuracy": round(partial_acc, 1),
        }
    
    return per_image, summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR pipeline accuracy")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to pipeline results JSON")
    parser.add_argument("--ground_truth", type=str, default="ground_truth.json",
                        help="Path to ground truth JSON")
    args = parser.parse_args()
    
    with open(args.results) as f:
        results = json.load(f)
    
    with open(args.ground_truth) as f:
        ground_truth = json.load(f)
    
    per_image, summary = evaluate(results, ground_truth)
    
    # Print per-image results
    print("=" * 90)
    print("FIELD-WISE EVALUATION")
    print("=" * 90)
    
    for img in per_image:
        print(f"\n--- {img['source_file']} ---")
        for field in ["vendor_name", "date", "total_amount"]:
            if field in img:
                info = img[field]
                status_marker = {"correct": "✓", "partial": "~", "wrong": "✗", "missing": "—"}
                marker = status_marker.get(info["status"], "?")
                print(f"  {marker} {field:<15} predicted='{info['predicted']}'")
                if info["status"] != "correct":
                    print(f"    {'':<17} actual='{info['actual']}'")
    
    # Print summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"\n{'Field':<20} {'Correct':<10} {'Partial':<10} {'Wrong':<10} {'Missing':<10} {'Exact %':<10} {'Partial %'}")
    print("-" * 80)
    
    for field, stats in summary.items():
        print(f"{field:<20} {stats['correct']:<10} {stats['partial']:<10} "
              f"{stats['wrong']:<10} {stats['missing']:<10} "
              f"{stats['exact_accuracy']:<10} {stats['partial_accuracy']}")
    
    # Overall
    total_correct = sum(s["correct"] for s in summary.values())
    total_partial = sum(s["partial"] for s in summary.values())
    total_wrong = sum(s["wrong"] for s in summary.values())
    total_missing = sum(s["missing"] for s in summary.values())
    total_all = sum(s["total"] for s in summary.values())
    
    if total_all > 0:
        overall_exact = round(total_correct / total_all * 100, 1)
        overall_partial = round((total_correct + total_partial) / total_all * 100, 1)
    else:
        overall_exact = 0
        overall_partial = 0
    
    print("-" * 80)
    print(f"{'OVERALL':<20} {total_correct:<10} {total_partial:<10} "
          f"{total_wrong:<10} {total_missing:<10} "
          f"{overall_exact:<10} {overall_partial}")
    
    # Save evaluation report
    eval_output = {
        "per_image": per_image,
        "summary": summary,
        "overall_exact_accuracy": overall_exact,
        "overall_partial_accuracy": overall_partial,
    }
    
    eval_path = Path(args.results).parent / "evaluation_report.json"
    with open(eval_path, "w") as f:
        json.dump(eval_output, f, indent=2)
    
    print(f"\nEvaluation saved to: {eval_path}")


if __name__ == "__main__":
    main()