#!/usr/bin/env python3
# author: Kilo Code
# date: 2025-12-10
# description: Evaluation script for fine-tuned Effort models

import argparse
import json
import logging
import os
from datetime import datetime

import torch
import yaml
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR
from logger import create_logger
from metrics.base_metrics_class import calculate_metrics_for_train
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO)


def parse_arguments():
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
        help="Test dataset(s)",
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
        "--lmdb_path",
        type=str,
        default=None,
        help="Path to LMDB database directory for LMDB dataset processing",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, mps, cpu)",
    )

    return parser.parse_args()


def setup_device(device_arg):
    """Set up the appropriate device for evaluation."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_model(config, weights_path, device):
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


def prepare_test_data(config, test_dataset, batch_size):
    """Prepare test data loader."""
    # Update config for test dataset
    test_config = config.copy()
    test_config["test_dataset"] = test_dataset
    test_config["test_batchSize"] = batch_size

    logging.info(f"Preparing test data for {test_dataset}")

    # Add fallback for dataset_json_folder if not in config
    if "dataset_json_folder" not in test_config:
        test_config["dataset_json_folder"] = (
            "/Users/logan/Developer/WORK/DEEPFAKE_DETECTION/Effort-AIGI-Detection/DeepfakeBench/preprocessing/dataset_json"
        )

    # Add fallback for label_dict if not in config
    if "label_dict" not in test_config:
        # Add basic label_dict for common datasets
        test_config["label_dict"] = {
            "FF-real": 0,
            "FF-F2F": 1,
            "FF-DF": 1,
            "FF-FS": 1,
            "FF-NT": 1,
            "CelebDFv2_real": 0,
            "CelebDFv2_fake": 1,
            "UADFV_Real": 0,
            "UADFV_Fake": 1,
        }

    test_set = DeepfakeAbstractBaseDataset(
        config=test_config,
        mode="test",
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


def evaluate_model(model, test_loader, device, dataset_name):
    """Evaluate model on test dataset."""
    logging.info(f"Evaluating on {dataset_name}")

    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

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
            labels = data_dict["label"].cpu()
            preds = pred_dict["cls"].cpu()
            probs = pred_dict["prob"].cpu()

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.argmax(dim=1).tolist())
            all_probs.extend(probs.tolist())

            if (batch_idx + 1) % 50 == 0:
                logging.info(f"Processed batch {batch_idx + 1}/{len(test_loader)}")

    # Calculate metrics
    labels_tensor = torch.tensor(all_labels)
    probs_tensor = torch.tensor(all_probs)

    # Calculate comprehensive metrics
    auc, eer, acc, ap = calculate_metrics_for_train(
        labels_tensor,
        probs_tensor.unsqueeze(1),
    )

    # Additional metrics
    from sklearn.metrics import f1_score, precision_score, recall_score

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

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

    logging.info(f"Evaluation results for {dataset_name}:")
    for metric_name, metric_value in metrics.items():
        logging.info(f"  {metric_name}: {metric_value:.4f}")

    return metrics


def save_results(metrics, dataset_name, output_dir, config):
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

    # Update config with test datasets
    config["test_dataset"] = args.test_dataset

    # Set LMDB path if provided
    if args.lmdb_path:
        config["lmdb_path"] = args.lmdb_path
        logging.info(f"Using LMDB dataset mode with path: {args.lmdb_path}")

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

    for dataset in args.test_dataset:
        try:
            # Prepare test loader
            test_loader = prepare_test_data(config, dataset, args.batch_size)

            # Evaluate
            metrics = evaluate_model(model, test_loader, device, dataset)

            # Save results
            save_results(metrics, dataset, output_dir, config)

            # Store results
            all_results[dataset] = metrics

        except Exception as e:
            logger.exception(f"Failed to evaluate on {dataset}: {e}")
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
