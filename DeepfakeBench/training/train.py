# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: training code.

import argparse
import os
import random
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.parallel
import torch.utils.data
import yaml
import secrets # Added for MPS seed
from pathlib import Path # Added for robust path handling
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR
from logger import RankFilter, create_logger
from metrics.utils import parse_metric_for_print
from optimizor.LinearLR import LinearDecayLR
from optimizor.SAM import SAM
from torch import optim
from torch.backends import cudnn
from torch.utils.data.distributed import DistributedSampler
from trainer.trainer import Trainer

parser = argparse.ArgumentParser(description="Process some paths.")
parser.add_argument(
    "--detector_path",
    type=str,
    default="/Users/logan/Developer/WORK/DEEPFAKE_DETECTION/Effort-AIGI-Detection/DeepfakeBench/training/config/detector/effort.yaml",
    help="path to detector YAML file",
)
print(f"[WARNING] Using hardcoded default path for detector_path: /Users/logan/Developer/WORK/DEEPFAKE_DETECTION/Effort-AIGI-Detection/DeepfakeBench/training/config/detector/effort.yaml")
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
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
parser.add_argument("--ddp", action="store_true", default=False)
parser.add_argument("--dry_run", action="store_true", default=False, help="Run a single epoch for testing")
parser.add_argument("--local_rank", type=int, default=0)
args = parser.parse_args()
# torch.cuda.set_device(args.local_rank) # Removed hardcoded CUDA set_device


def init_seed(config):
    if config["manualSeed"] is None:
        config["manualSeed"] = 1 + secrets.randbelow(10000)
    random.seed(config["manualSeed"])
    
    if torch.backends.mps.is_available():
        torch.manual_seed(config["manualSeed"])
        torch.mps.manual_seed(config["manualSeed"])
    elif config["cuda"]:
        torch.manual_seed(config["manualSeed"])
        torch.cuda.manual_seed_all(config["manualSeed"])


def prepare_training_data(config):
    # Only use the blending dataset class in training
    train_set = DeepfakeAbstractBaseDataset(
        config=config,
        mode="train",
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
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config["train_batchSize"],
            shuffle=True,
            num_workers=int(config["workers"]),
            collate_fn=train_set.collate_fn,
        )
    return train_data_loader


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = (
            config.copy()
        )  # create a copy of config to avoid altering the original one
        config["test_dataset"] = test_name  # specify the current test dataset

        test_set = DeepfakeAbstractBaseDataset(
            config=config,
            mode="test",
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
    for one_test_name in config["test_dataset"]:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_optimizer(model, config):
    opt_name = config["optimizer"]["type"]
    if opt_name == "sgd":
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config["optimizer"][opt_name]["lr"],
            momentum=config["optimizer"][opt_name]["momentum"],
            weight_decay=config["optimizer"][opt_name]["weight_decay"],
        )
        return optimizer
    if opt_name == "adam":
        optimizer = optim.Adam(
            params=model.parameters(),
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
            model.parameters(),
            optim.SGD,
            lr=config["optimizer"][opt_name]["lr"],
            momentum=config["optimizer"][opt_name]["momentum"],
        )
    else:
        raise NotImplementedError(
            "Optimizer {} is not implemented".format(config["optimizer"]),
        )
    return optimizer


def choose_scheduler(config, optimizer):
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
    else:
        raise NotImplementedError(
            "Scheduler {} is not implemented".format(config["lr_scheduler"]),
        )


def choose_metric(config):
    metric_scoring = config["metric_scoring"]
    if metric_scoring not in ["eer", "auc", "acc", "ap"]:
        raise NotImplementedError(f"metric {metric_scoring} is not implemented")
    return metric_scoring


def main():
    # parse options and load config
    with open(args.detector_path) as f:
        config = yaml.safe_load(f)
    
    # Resolve train_config.yaml relative to this script
    script_dir = Path(__file__).resolve().parent
    train_config_path = script_dir / "config" / "train_config.yaml"
    
    with open(train_config_path) as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    config["local_rank"] = args.local_rank
    if config["dry_run"]:
        config["nEpochs"] = 0
        config["save_feat"] = False
    # If arguments are provided, they will overwrite the yaml settings
    if args.train_dataset:
        config["train_dataset"] = args.train_dataset
    if args.test_dataset:
        config["test_dataset"] = args.test_dataset
    config["save_feat"] = args.save_feat
    if args.dry_run:
        config["dry_run"] = True
        config["nEpochs"] = 0
        config["save_feat"] = False
    
    if config["lmdb"]:
        config["dataset_json_folder"] = (
            "/Users/logan/Developer/WORK/DEEPFAKE_DETECTION/Effort-AIGI-Detection/DeepfakeBench/preprocessing/dataset_json"
        )
        print(f"[WARNING] Using hardcoded path for dataset_json_folder: {config['dataset_json_folder']}")
    # create logger
    logger_path = config["log_dir"]
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, "training.log"))
    logger.info(f"Save log to {logger_path}")
    config["ddp"] = args.ddp
    # print configuration
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += f"{key}: {value}" + "\n"
    logger.info(params_string)

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config.get("cudnn", False):
        cudnn.benchmark = True
    if config["ddp"]:
        # dist.init_process_group(backend='gloo')
        if torch.backends.mps.is_available():
             backend = "gloo"
        else:
             backend = "nccl"
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=30),
        )
        logger.addFilter(RankFilter(0))
    # prepare the training data loader
    train_data_loader = prepare_training_data(config)

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)

    # prepare the model (detector)
    model_class = DETECTOR[config["model_name"]]
    model = model_class(config)

    # prepare the optimizer
    optimizer = choose_optimizer(model, config)

    # prepare the scheduler
    scheduler = choose_scheduler(config, optimizer)

    # prepare the metric
    metric_scoring = choose_metric(config)

    # prepare the trainer
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring)

    # start training
    for epoch in range(config["start_epoch"], config["nEpochs"] + 1):
        trainer.model.epoch = epoch
        best_metric = trainer.train_epoch(
            epoch=epoch,
            train_data_loader=train_data_loader,
            test_data_loaders=test_data_loaders,
        )
        if best_metric is not None:
            logger.info(
                f"===> Epoch[{epoch}] end with testing {metric_scoring}: {parse_metric_for_print(best_metric)}!",
            )
    logger.info(
        f"Stop Training on best Testing metric {parse_metric_for_print(best_metric)}",
    )
    # update
    if "svdd" in config["model_name"]:
        model.update_R(epoch)
    if scheduler is not None:
        scheduler.step()

    # close the tensorboard writers
    for writer in trainer.writers.values():
        writer.close()


if __name__ == "__main__":
    main()
