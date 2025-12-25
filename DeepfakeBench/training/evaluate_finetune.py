#!/usr/bin/env python3
"""Evaluation script for fine-tuned deepfake detection models.

Calculates frame-level and video-level metrics using predictions from the model.
"""

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Any

import numpy as np
import torch
import yaml
from dataset.factory import DatasetFactory
from detectors import DETECTOR
from logger import create_logger
from metrics.utils import get_test_metrics
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Fine-Tuned Effort Model")

    parser.add_argument(
        "--detector_config",
        type=str,
        default="/Users/logan/Developer/WORK/DEEPFAKE_DETECTION/Effort-AIGI-Detection/DeepfakeBench/training/config/detector/effort_finetune.yaml",
        help="YAML configuration file path",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to fine-tuned model weights",
    )
    parser.add_argument(
        "--test_dataset",
        nargs="+",
        required=True,
        help="Test dataset path(s) - raw file directories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, mps, cpu)",
    )

    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Set up the appropriate device for evaluation."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_model(
    config: dict[str, Any], weights_path: str, device: torch.device
) -> torch.nn.Module:
    """Load the fine-tuned model."""
    logging.info(f"Loading model from {weights_path}")

    try:
        # Initialize model
        model_class = DETECTOR[config["model_name"]]
        model = model_class(config).to(device)

        # Load weights
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Handle DDP prefixes
        model_state_dict = model.state_dict()
        cleaned_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("module."):
                new_key = key[7:]  # Remove DDP prefix
            else:
                new_key = key

            if (
                new_key in model_state_dict
                and value.shape == model_state_dict[new_key].shape
            ):
                cleaned_state_dict[new_key] = value

        # Load state dict
        model.load_state_dict(cleaned_state_dict, strict=False)
        model.eval()

        logging.info("Model loaded successfully")
        return model

    except Exception as e:
        logging.exception(f"Failed to load model: {e}")
        raise


def prepare_test_data(
    config: dict[str, Any], test_dataset_paths: list[str], batch_size: int
) -> DataLoader:
    """Prepare test data loader."""
    # Update config for test dataset
    test_config = config.copy()
    test_config["test_batchSize"] = batch_size

    logging.info(f"Preparing test data for paths: {test_dataset_paths}")

    # For raw file datasets, we don't need JSON folder or label_dict fallbacks
    # The RawFileDataset will handle label inference automatically

    test_set = DatasetFactory.create_dataset(
        config=test_config,
        mode="test",
        raw_data_root=test_dataset_paths,
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=test_config.get("workers", 0),
        collate_fn=test_set.collate_fn,
        drop_last=False,
    )

    logging.info(f"Test loader prepared with {len(test_loader)} batches")
    return test_loader


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    dataset_name: str,
) -> dict[str, Any]:
    """Evaluate model on test dataset."""
    logging.info(f"Evaluating on {dataset_name}")

    model.eval()
    all_labels = []
    all_logits = []
    all_probs = []
    all_names = []

    with torch.no_grad():
        for batch_idx, data_dict in enumerate(test_loader):
            # Move data to device
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.to(device)

            # Forward pass
            with torch.autocast(
                device_type=device.type if device.type != "cpu" else "cpu",
            ):
                pred_dict = model(data_dict, inference=True)

            # Collect results
            labels = data_dict["label"].cpu().detach().numpy()
            logits = pred_dict["cls"].cpu().detach().numpy()
            probs = pred_dict["prob"].cpu().detach().numpy()
            names = data_dict["image"]

            all_labels.extend(labels.tolist())
            all_logits.extend(logits.tolist())
            all_probs.extend(probs.tolist())
            all_names.extend(names)

            if (batch_idx + 1) % 50 == 0:
                logging.info(f"Processed batch {batch_idx + 1}/{len(test_loader)}")

    # Calculate comprehensive metrics using get_test_metrics
    # This ensures consistency with the training/validation logic
    metric_results = get_test_metrics(
        y_pred=np.array(all_probs),
        y_true=np.array(all_labels),
        img_names=all_names,
    )

    # Extract primary metrics
    auc = metric_results["auc"]
    eer = metric_results["eer"]
    acc = metric_results["acc"]
    ap = metric_results.get("ap", 0.0)

    # For sklearn metrics, we need discrete predictions
    all_preds = (np.array(all_probs) > 0.5).astype(int)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    metrics = {
        "auc": float(auc),
        "eer": float(eer),
        "accuracy": float(acc),
        "ap": float(ap),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "sample_count": len(all_labels),
    }

    # Add video-level metrics if available
    if "video_auc" in metric_results:
        metrics["video_auc"] = float(metric_results["video_auc"])
    if "video_eer" in metric_results:
        metrics["video_eer"] = float(metric_results["video_eer"])
    if "video_acc" in metric_results:
        metrics["video_acc"] = float(metric_results["video_acc"])

    logging.info(f"Evaluation results for {dataset_name}:")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (float, int)):
            logging.info(f"  {metric_name}: {metric_value:.4f}")
        else:
            logging.info(f"  {metric_name}: {metric_value}")

    return metrics


def save_results(
    metrics: dict[str, Any], dataset_name: str, output_dir: str, config: dict[str, Any]
) -> None:
    """Save evaluation results to file."""
    os.makedirs(output_dir, exist_ok=True)

    # Create results filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{dataset_name}_eval_{timestamp}.json"

    # Include configuration info
    result_data = {
        "dataset": dataset_name,
        "timestamp": timestamp,
        "metrics": metrics,
        "config": {
            "model_name": config["model_name"],
            "backbone_name": config["backbone_name"],
            "resolution": config["resolution"],
            "frame_num": config["frame_num"],
        },
    }

    # Save to JSON
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=2)

    logging.info(f"Saved evaluation results to {output_path}")
    return output_path


def main():
    """Main evaluation function."""
    args = parse_arguments()

    # Set up device
    device = setup_device(args.device)
    logging.info(f"Using device: {device}")

    # Load configuration
    with open(args.detector_config) as f:
        config = yaml.safe_load(f)

    # Store test dataset paths
    test_dataset_paths = args.test_dataset
    logging.info(f"Using test dataset paths: {test_dataset_paths}")

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Set up logging
    log_file = os.path.join(output_dir, "evaluation.log")
    logger = create_logger(log_file)
    logger.info(f"Evaluation log: {log_file}")

    # Log configuration
    logger.info("--------------- Evaluation Configuration ---------------")
    for key, value in config.items():
        logger.info(f"{key}: {value}")

    # Load model
    model = load_model(config, args.weights, device)

    # Prepare test data and evaluate
    all_results = {}

    # For multiple test datasets, evaluate each one separately
    for i, dataset_path in enumerate(test_dataset_paths):
        try:
            dataset_name = f"dataset_{i + 1}_{os.path.basename(dataset_path)}"

            # Prepare test loader for this specific dataset
            test_loader = prepare_test_data(config, [dataset_path], args.batch_size)

            # Evaluate
            metrics = evaluate_model(model, test_loader, device, dataset_name)

            # Save results
            save_results(metrics, dataset_name, output_dir, config)

            # Store results
            all_results[dataset_name] = metrics

        except Exception as e:
            logger.exception(f"Failed to evaluate on {dataset_path}: {e}")
            continue

    # Save summary
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {"datasets": all_results, "timestamp": datetime.now().isoformat()},
            f,
            indent=2,
        )

    logger.info(f"Evaluation completed! Summary saved to {summary_path}")
    logger.info("Fine-tuned model evaluation finished successfully")


if __name__ == "__main__":
    main()
