#!/usr/bin/env python3
# author: Kilo Code
# date: 2025-12-10
# description: Fine-tuning script for Effort model with SVD decomposition

import argparse
import logging
import os
import random
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.parallel
import torch.utils.data
import yaml
from dataset.factory import DatasetFactory
from detectors import DETECTOR
from logger import RankFilter, create_logger
from metrics.utils import parse_metric_for_print
from optimizor.LinearLR import LinearDecayLR
from optimizor.SAM import SAM
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from trainer.trainer import Trainer

# Set up logging
logging.basicConfig(level=logging.INFO)


def parse_arguments():
    """Parse command line arguments for fine-tuning."""
    parser = argparse.ArgumentParser(description="Effort Model Fine-Tuning")

    parser.add_argument(
        "--detector_config",
        type=str,
        default="/Users/logan/Developer/WORK/DEEPFAKE_DETECTION/Effort-AIGI-Detection/DeepfakeBench/training/config/detector/effort_finetune.yaml",
        help="YAML configuration file path",
    )
    parser.add_argument("--train_dataset", nargs="+", help="training dataset(s)")
    parser.add_argument("--test_dataset", nargs="+", help="testing dataset(s)")
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        help="path to pretrained weights for fine-tuning",
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default=None,
        help="Path to root directory with real/fake folders for raw file processing",
    )
    parser.add_argument(
        "--no-save_ckpt",
        dest="save_ckpt",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--no-save_feat",
        dest="save_feat",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        default=False,
        help="use distributed data parallel",
    )
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for DDP")

    return parser.parse_args()


def init_seed(config):
    """Initialize random seeds for reproducibility."""
    if config["manualSeed"] is None:
        config["manualSeed"] = random.randint(1, 10000)
    random.seed(config["manualSeed"])
    if config.get("cuda", False):
        torch.manual_seed(config["manualSeed"])
        torch.cuda.manual_seed_all(config["manualSeed"])


def prepare_training_data(config, raw_data_root=None):
    """Prepare training data loader with fine-tuning specific settings."""
    logging.info("Preparing training data for fine-tuning")

    train_set = DatasetFactory.create_dataset(
        config=config,
        mode="train",
        raw_data_root=raw_data_root,
    )

    if config["ddp"]:
        sampler = DistributedSampler(train_set)
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config["train_batchSize"],
            num_workers=int(config["workers"]),
            collate_fn=train_set.collate_fn,
            sampler=sampler,
        )
    else:
        # For RawFileDataset, don't shuffle since it already provides interleaved balanced batches
        from dataset.raw_file_dataset import RawFileDataset

        shuffle_train = not isinstance(train_set, RawFileDataset)
        if isinstance(train_set, RawFileDataset):
            logging.info(
                "Using RawFileDataset with pre-interleaved samples, disabling DataLoader shuffle for balanced batches",
            )
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config["train_batchSize"],
            shuffle=shuffle_train,
            num_workers=int(config["workers"]),
            collate_fn=train_set.collate_fn,
        )

    logging.info(f"Training data loader prepared with {len(train_data_loader)} batches")
    return train_data_loader


def prepare_testing_data(config, raw_data_root=None):
    """Prepare testing data loaders."""

    def get_test_data_loader(config, test_name, raw_data_root=None):
        config_copy = config.copy()
        config_copy["test_dataset"] = test_name

        test_set = DatasetFactory.create_dataset(
            config=config_copy,
            mode="test",
            raw_data_root=raw_data_root,
        )

        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=config["test_batchSize"],
            shuffle=False,
            num_workers=int(config["workers"]),
            collate_fn=test_set.collate_fn,
            drop_last=(test_name == "DeepFakeDetection"),
        )

        return test_data_loader

    test_data_loaders = {}
    for test_name in config["test_dataset"]:
        test_data_loaders[test_name] = get_test_data_loader(
            config,
            test_name,
            raw_data_root,
        )

    logging.info(f"Prepared {len(test_data_loaders)} test data loaders")
    return test_data_loaders


def load_pretrained_weights(model, pretrained_path, config):
    """Load pretrained weights for fine-tuning."""
    if not pretrained_path or not os.path.exists(pretrained_path):
        logging.warning(
            # f"No pretrained weights found at {pretrained_path}, starting from scratch",
            "No pretrained weights found at {pretrained_path}, starting from scratch",
            extra={"pretrained_path": pretrained_path},
        )
        return model

    logging.info(
        "Loading pretrained weights from {pretrained_path}",
        extra={"pretrained_path": pretrained_path},
    )

    try:
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Handle potential key mismatches
        model_state_dict = model.state_dict()
        pretrained_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("module."):
                # Remove "module." prefix from DDP training
                new_key = key[7:]
            else:
                new_key = key

            if (
                new_key in model_state_dict
                and value.shape == model_state_dict[new_key].shape
            ):
                pretrained_state_dict[new_key] = value
            else:
                logging.debug(
                    f"Skipping weight {key} due to shape mismatch or missing key",
                )

        # Load compatible weights
        model_state_dict.update(pretrained_state_dict)
        model.load_state_dict(model_state_dict, strict=False)

        # Log loading statistics
        loaded_params = len(pretrained_state_dict)
        total_params = len(model_state_dict)
        logging.info(f"Loaded {loaded_params}/{total_params} pretrained parameters")

        return model

    except Exception as e:
        logging.exception(f"Failed to load pretrained weights: {e}")
        return model


def configure_fine_tuning(model, config):
    """Configure model for fine-tuning based on configuration."""
    logging.info("Configuring model for fine-tuning")

    # Apply fine-tuning settings
    if config.get("freeze_backbone", True):
        logging.info("Freezing backbone SVD main components")
        for name, param in model.named_parameters():
            if "weight_main" in name:
                param.requires_grad = False
            elif (
                any(x in name for x in ["S_residual", "U_residual", "V_residual"])
                or "head" in name
            ):
                param.requires_grad = True

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info(
        f"Trainable parameters: {trainable_params}/{total_params} ({100 * trainable_params / total_params:.2f}%)",
    )

    return model


def choose_optimizer(model, config):
    """Choose optimizer based on configuration."""
    opt_name = config["optimizer"]["type"]

    if opt_name == "sgd":
        optimizer = optim.SGD(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["optimizer"][opt_name]["lr"],
            momentum=config["optimizer"][opt_name]["momentum"],
            weight_decay=config["optimizer"][opt_name]["weight_decay"],
        )
        return optimizer

    if opt_name == "adam":
        optimizer = optim.Adam(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["optimizer"][opt_name]["lr"],
            weight_decay=config["optimizer"][opt_name]["weight_decay"],
            betas=(
                config["optimizer"][opt_name]["beta1"],
                config["optimizer"][opt_name]["beta2"],
            ),
            eps=config["optimizer"][opt_name]["eps"],
            amsgrad=config["optimizer"][opt_name]["amsgrad"],
        )
        return optimizer

    if opt_name == "sam":
        optimizer = SAM(
            filter(lambda p: p.requires_grad, model.parameters()),
            optim.SGD,
            lr=config["optimizer"][opt_name]["lr"],
            momentum=config["optimizer"][opt_name]["momentum"],
        )
        return optimizer

    raise NotImplementedError(f"Optimizer {opt_name} is not implemented")


def choose_scheduler(config, optimizer):
    """Choose learning rate scheduler."""
    if config["lr_scheduler"] is None:
        return None

    if config["lr_scheduler"] == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["lr_step"],
            gamma=config["lr_gamma"],
        )
        return scheduler

    if config["lr_scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["lr_T_max"],
            eta_min=config["lr_eta_min"],
        )
        return scheduler

    if config["lr_scheduler"] == "linear":
        scheduler = LinearDecayLR(
            optimizer,
            config["nEpochs"],
            int(config["nEpochs"] / 4),
        )
        return scheduler

    raise NotImplementedError(f"Scheduler {config['lr_scheduler']} is not implemented")


def choose_metric(config):
    """Choose evaluation metric."""
    metric_scoring = config["metric_scoring"]
    if metric_scoring not in ["eer", "auc", "acc", "ap"]:
        raise NotImplementedError(f"Metric {metric_scoring} is not implemented")
    return metric_scoring


def main():
    """Main fine-tuning function."""
    args = parse_arguments()

    # Set up device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )
    logging.info(f"Using device: {device}")

    # Load configuration
    with open(args.detector_config) as f:
        config = yaml.safe_load(f)
    with open(
        "/Users/logan/Developer/WORK/DEEPFAKE_DETECTION/Effort-AIGI-Detection/DeepfakeBench/training/config/train_config.yaml",
    ) as f:
        config.update(yaml.safe_load(f))

    config["local_rank"] = args.local_rank
    config["save_ckpt"] = args.save_ckpt
    config["save_feat"] = args.save_feat
    config["ddp"] = args.ddp

    # Override with command line arguments
    if args.train_dataset:
        config["train_dataset"] = args.train_dataset
    if args.test_dataset:
        config["test_dataset"] = args.test_dataset

    # Set pretrained weights path
    if args.pretrained_weights:
        config["pretrained_checkpoint"] = args.pretrained_weights

    # Set raw data root from command line or config file
    raw_data_root = args.raw_data_dir or config.get("raw_data_root")
    if raw_data_root:
        logging.info(f"Using raw data mode with root: {raw_data_root}")

    # Create logger
    logger_path = config["log_dir"]
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, "finetuning.log"))
    logger.info(f"Save log to {logger_path}")

    # Log configuration
    logger.info("--------------- Fine-Tuning Configuration ---------------")
    for key, value in config.items():
        logger.info(f"{key}: {value}")

    # Initialize seed
    init_seed(config)

    # Set up DDP if enabled
    if config["ddp"]:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
        logger.addFilter(RankFilter(0))

    # Prepare data loaders
    dataset_class = DatasetFactory.get_dataset_class_name(config, raw_data_root)
    logging.info(f"Using {dataset_class} for data loading")
    if raw_data_root:
        logging.info(f"Raw data directory: {raw_data_root}")

    train_data_loader = prepare_training_data(config, raw_data_root)
    test_data_loaders = prepare_testing_data(config, raw_data_root)

    # Initialize model
    logger.info("Initializing Effort model")
    model_class = DETECTOR[config["model_name"]]
    model = model_class(config).to(device)

    # Load pretrained weights if available
    model = load_pretrained_weights(model, config.get("pretrained_checkpoint"), config)

    # Configure for fine-tuning
    model = configure_fine_tuning(model, config)

    # Prepare optimizer and scheduler
    optimizer = choose_optimizer(model, config)
    scheduler = choose_scheduler(config, optimizer)
    metric_scoring = choose_metric(config)

    # Initialize trainer
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring)

    # Training loop
    logger.info("Starting fine-tuning process")
    best_metric = None

    for epoch in range(config["start_epoch"], config["nEpochs"] + 1):
        trainer.model.epoch = epoch

        # Train for one epoch
        epoch_metric = trainer.train_epoch(
            epoch=epoch,
            train_data_loader=train_data_loader,
            test_data_loaders=test_data_loaders,
        )

        if epoch_metric is not None:
            best_metric = epoch_metric
            logger.info(
                f"===> Epoch[{epoch}] completed with {metric_scoring}: {parse_metric_for_print(epoch_metric)}!",
            )

        # Step scheduler if available
        if scheduler is not None:
            scheduler.step()

    logger.info(
        f"Fine-tuning completed! Best {metric_scoring}: {parse_metric_for_print(best_metric)}",
    )

    # Clean up
    for writer in trainer.writers.values():
        writer.close()

    logger.info("Fine-tuning process finished successfully")


if __name__ == "__main__":
    main()
