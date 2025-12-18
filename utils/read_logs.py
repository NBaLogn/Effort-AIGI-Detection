"""Script to read pickle files and TensorBoard event files from logs directory.

Outputs summaries to files in the specified output directory.
"""

import argparse
import csv
import json
import pickle
import pprint
from pathlib import Path
from typing import Any, TextIO

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def process_pickle_file(file_path: Path, output_file: TextIO[str]) -> None:
    """Process a single pickle file and write content to output."""
    try:
        with file_path.open("rb") as f:
            obj = pickle.load(f)  # noqa: S301
        output_file.write(f"{file_path.relative_to(Path.cwd())}:\n")
        pprint.pprint(obj, stream=output_file)
        output_file.write("\n")
    except (OSError, pickle.PickleError, ValueError) as e:
        output_file.write(f"Error reading {file_path}: {e}\n")


def process_event_file(
    file_path: Path,
    csv_writer: csv.DictWriter[str],
    json_data: list[dict[str, Any]],
) -> None:
    """Process a single TensorBoard event file and write data."""
    try:
        ea = EventAccumulator(str(file_path))
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        for tag in tags:
            scalars = ea.Scalars(tag)
            for scalar in scalars:
                row = {
                    "file": str(file_path.relative_to(Path.cwd())),
                    "tag": tag,
                    "step": scalar.step,
                    "value": scalar.value,
                    "wall_time": scalar.wall_time,
                }
                csv_writer.writerow(row)
                json_data.append(row)
    except (OSError, ValueError) as e:
        print(f"Error reading {file_path}: {e}")


def main() -> None:
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Read pickle and TensorBoard files from logs directory.",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="DeepfakeBench/training/logs",
        help="Directory containing logs (default: DeepfakeBench/training/logs)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs_output",
        help="Directory to save output files (default: logs_output)",
    )
    args = parser.parse_args()

    logs_path = Path(args.logs_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)

    # Output files
    pickle_summary_file = output_path / "pickle_summaries.txt"
    metrics_csv_file = output_path / "metrics.csv"
    metrics_json_file = output_path / "metrics.json"

    # Collect all pickle files
    pickle_files = list(logs_path.rglob("*.pickle"))

    # Collect all event files
    event_files = list(logs_path.rglob("events.out.tfevents.*"))

    # Process pickle files
    with pickle_summary_file.open("w") as f:
        for pickle_file in pickle_files:
            process_pickle_file(pickle_file, f)

    # Process event files
    json_data = []
    with metrics_csv_file.open("w", newline="") as csvfile:
        fieldnames = ["file", "tag", "step", "value", "wall_time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for event_file in event_files:
            process_event_file(event_file, writer, json_data)

    # Write JSON
    with metrics_json_file.open("w") as jsonfile:
        json.dump(json_data, jsonfile, indent=2)

    print(
        f"Processed {len(pickle_files)} pickle files and {len(event_files)} event files.",
    )
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    main()
