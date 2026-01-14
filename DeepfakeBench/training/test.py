"""eval pretained model."""

import argparse
import random

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import yaml
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR
from metrics.utils import get_test_metrics
from torch.backends import cudnn
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Process some paths.")
parser.add_argument(
    "--detector_path",
    type=str,
    default="./training/config/detector/effort.yaml",
    help="path to detector YAML file",
)
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument("--weights_path", type=str, default="./weights/effort_ckpt.pth")
# parser.add_argument("--lmdb", action='store_true', default=False)
args = parser.parse_args()

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise RuntimeError("No MPS or CUDA device found. CPU validation is not supported.")


def init_seed(config):
    if config.get("manualSeed") is None:
        config["manualSeed"] = random.randint(1, 10000)
    random.seed(config["manualSeed"])
    torch.manual_seed(config["manualSeed"])
    if config.get("cuda", False):
        torch.cuda.manual_seed_all(config["manualSeed"])


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
            drop_last=False,
        )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config["test_dataset"]:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config["metric_scoring"]
    if metric_scoring not in ["eer", "auc", "acc", "ap"]:
        raise NotImplementedError(f"metric {metric_scoring} is not implemented")
    return metric_scoring


def test_one_dataset(model, data_loader):
    prediction_lists = []
    feature_lists = []
    label_lists = []
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = (
            data_dict["image"],
            data_dict["label"],
            data_dict["mask"],
            data_dict["landmark"],
        )
        label = torch.where(data_dict["label"] != 0, 1, 0)
        # move data to GPU
        data_dict["image"], data_dict["label"] = data.to(device), label.to(device)
        if mask is not None:
            data_dict["mask"] = mask.to(device)
        if landmark is not None:
            data_dict["landmark"] = landmark.to(device)

        # Log batch labels for debugging (only if needed for troubleshooting)
        # batch_labels = data_dict["label"].cpu().detach().numpy()
        # unique_labels, counts = np.unique(batch_labels, return_counts=True)
        # tqdm.write(
        #     f"Batch {i + 1}: labels distribution - {dict(zip(unique_labels, counts))}",
        # )

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)
        label_lists += list(data_dict["label"].cpu().detach().numpy())
        prediction_lists += list(predictions["prob"].cpu().detach().numpy())
        feature_lists += list(predictions["feat"].cpu().detach().numpy())

    # Log overall labels distribution
    all_labels = np.array(label_lists)
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    tqdm.write(f"Overall labels distribution: {dict(zip(unique_labels, counts))}")

    return np.array(prediction_lists), np.array(label_lists), np.array(feature_lists)


def test_epoch(model, test_data_loaders):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict
        # compute loss for each dataset
        predictions_nps, label_nps, feat_nps = test_one_dataset(
            model,
            test_data_loaders[key],
        )

        # compute metric for each dataset
        metric_one_dataset = get_test_metrics(
            y_pred=predictions_nps,
            y_true=label_nps,
            img_names=data_dict["image"],
        )
        metrics_all_datasets[key] = metric_one_dataset

        # info for each dataset
        tqdm.write(f"dataset: {key}")
        for k, v in metric_one_dataset.items():
            tqdm.write(f"{k}: {v}")

    return metrics_all_datasets


@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def main():
    # parse options and load config
    with open(args.detector_path) as f:
        config = yaml.safe_load(f)
    with open(
        "/Users/logan/Developer/WORK/DEEPFAKE_DETECTION/Effort-AIGI-Detection/DeepfakeBench/training/config/test_config.yaml",
    ) as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    # on_2060 logic removed, using default paths but logging them
    config["workers"] = 8
    config["lmdb_dir"] = r"/Volumes/Crucial/AI/DATASETS/DFB/lmdbs"
    print(f"[WARNING] Using hardcoded path for lmdb_dir: {config['lmdb_dir']}")
    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config["test_dataset"] = args.test_dataset
    if args.weights_path:
        config["weights_path"] = args.weights_path
        weights_path = args.weights_path

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config.get("cudnn", False):
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)

    # prepare the model (detector)
    model_class = DETECTOR[config["model_name"]]
    model = model_class(config).to(device)
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(
        f"Total number of trainable parameters in the model: {total_trainable_params}",
    )
    epoch = 0
    if weights_path:
        try:
            epoch = int(weights_path.split("/")[-1].split(".")[0].split("_")[2])
        except:
            epoch = 0
        ckpt = torch.load(weights_path, map_location=device)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        # # 加载模型的状态字典
        # model_dict = model.state_dict()
        # new_ckpt={}
        # for key in ckpt.keys():
        #     # 替换键
        #     new_key = key.replace('common_encoder_f','student_encoder')
        #     # 将旧的值复制到新的键下
        #     new_ckpt[new_key] = ckpt[key]
        # # 获取ckpt和model的key集合
        # ckpt_keys = set(new_ckpt.keys())
        # model_keys = set(model_dict.keys())
        #
        # # 找出共同的key
        # common_keys = ckpt_keys & model_keys
        # print("Common keys:")
        # for key in common_keys:
        #     print(key)
        #
        # # 找出只在ckpt中的key
        # ckpt_unique_keys = ckpt_keys - model_keys
        # print("\nKeys only in ckpt:")
        # for key in ckpt_unique_keys:
        #     print(key)
        #
        # # 找出只在model中的key
        # model_unique_keys = model_keys - ckpt_keys
        # print("\nKeys only in model:")
        # for key in model_unique_keys:
        #     print(key)

        # 创建一个新的字典，删除module前缀
        new_weights = {}
        for key, value in ckpt.items():
            new_key = key.replace("module.", "")  # 删除module前缀
            # new_key = 'backbone.' + new_key  # 删除module前缀
            #  if 'base_model.' in new_key:
            #      new_key = new_key.replace('base_model.', 'backbone.')
            #  if 'classifier.' in new_key:
            #      new_key = new_key.replace('classifier.', 'head.')
            new_weights[new_key] = value

        model.load_state_dict(new_weights, strict=True) # Normalized Strict loading
        print("===> Load checkpoint done!")
    else:
        print("Fail to load the pre-trained weights")

    #   clip_rank_results = analyze_clip_effective_rank(model.backbone)

    #   for layer_name, rank in clip_rank_results.items():
    #       print(f"Layer: {layer_name}, Effective Rank: {rank}")

    # start testing
    best_metric = test_epoch(model, test_data_loaders)
    print("===> Test Done!")


if __name__ == "__main__":
    main()
