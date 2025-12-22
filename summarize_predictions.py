#!/usr/bin/env python3
"""Script to summarize prediction accuracy from inference_results.txt"""

import re
from pathlib import Path


def parse_inference_results(file_path):
    """Parse the inference results file and extract prediction information.

    Returns:
        dict: Summary statistics including counts and accuracy metrics

    """
    total_predictions = 0
    correct_predictions = 0

    # Class-specific counters
    real_total = 0
    real_correct = 0
    fake_total = 0
    fake_correct = 0

    # Read and parse the file
    with open(file_path) as f:
        for line in f:
            # Skip header lines and empty lines
            if line.strip().startswith("[") and "|" in line:
                # Extract True and Pred values using regex
                # Format: [index] filename | True: XXXX | Pred: X (0=Real, 1=Fake) | ...
                true_match = re.search(r"True:\s*(\w+)", line)
                pred_match = re.search(r"Pred:\s*(\d+)", line)

                if true_match and pred_match:
                    true_label = true_match.group(1).upper()  # REAL or FAKE
                    pred_value = int(pred_match.group(1))  # 0 or 1

                    # Convert prediction to label
                    pred_label = "REAL" if pred_value == 0 else "FAKE"

                    total_predictions += 1

                    # Check if prediction is correct
                    is_correct = true_label == pred_label

                    if is_correct:
                        correct_predictions += 1

                    # Class-specific counting
                    if true_label == "REAL":
                        real_total += 1
                        if is_correct:
                            real_correct += 1
                    elif true_label == "FAKE":
                        fake_total += 1
                        if is_correct:
                            fake_correct += 1

    # Calculate accuracy metrics
    overall_accuracy = (
        (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    )
    real_accuracy = (real_correct / real_total * 100) if real_total > 0 else 0
    fake_accuracy = (fake_correct / fake_total * 100) if fake_total > 0 else 0

    return {
        "total_predictions": total_predictions,
        "correct_predictions": correct_predictions,
        "overall_accuracy": overall_accuracy,
        "real_total": real_total,
        "real_correct": real_correct,
        "real_accuracy": real_accuracy,
        "fake_total": fake_total,
        "fake_correct": fake_correct,
        "fake_accuracy": fake_accuracy,
        "incorrect_predictions": total_predictions - correct_predictions,
    }


def generate_summary_report(stats):
    """Generate a formatted summary report."""
    report = []
    report.append("=" * 60)
    report.append("PREDICTION ACCURACY SUMMARY")
    report.append("=" * 60)
    report.append("")

    # Overall statistics
    report.append("OVERALL PERFORMANCE:")
    report.append(f"  Total Images Processed: {stats['total_predictions']}")
    report.append(f"  Correct Predictions:    {stats['correct_predictions']}")
    report.append(f"  Incorrect Predictions:  {stats['incorrect_predictions']}")
    report.append(f"  Overall Accuracy:       {stats['overall_accuracy']:.2f}%")
    report.append("")

    # Real images breakdown
    report.append("REAL IMAGES:")
    report.append(f"  Total Real Images:      {stats['real_total']}")
    report.append(f"  Correctly Classified:   {stats['real_correct']}")
    report.append(f"  Accuracy:               {stats['real_accuracy']:.2f}%")
    report.append("")

    # Fake images breakdown
    report.append("FAKE IMAGES:")
    report.append(f"  Total Fake Images:      {stats['fake_total']}")
    report.append(f"  Correctly Classified:   {stats['fake_correct']}")
    report.append(f"  Accuracy:               {stats['fake_accuracy']:.2f}%")
    report.append("")

    # Error analysis
    report.append("ERROR ANALYSIS:")
    report.append(
        f"  False Positives (Real → Fake): {stats['real_total'] - stats['real_correct']}"
    )
    report.append(
        f"  False Negatives (Fake → Real): {stats['fake_total'] - stats['fake_correct']}"
    )
    report.append("")

    # Summary line
    report.append("=" * 60)
    report.append(f"SUMMARY: Model achieved {stats['overall_accuracy']:.2f}% accuracy")
    report.append("=" * 60)

    return "\n".join(report)


def main():
    """Main execution function."""
    input_file = Path("inference_results.txt")
    output_file = Path("prediction_summary.txt")

    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        return

    print(f"Analyzing {input_file}...")

    # Parse results
    stats = parse_inference_results(input_file)

    # Generate report
    report = generate_summary_report(stats)

    # Write to output file
    with open(output_file, "w") as f:
        f.write(report)

    # Also print to console
    print("\n" + report)
    print(f"\nSummary saved to: {output_file}")


if __name__ == "__main__":
    main()
