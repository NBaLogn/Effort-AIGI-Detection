import logging

logging.basicConfig(level=logging.INFO)

import sys

sys.path.insert(0, ".")

from DeepfakeBench.training.dataset.lmdb_dataset import LMDBDataset

config = {
    "lmdb_path": "/Volumes/Crucial/Large_Downloads/AI/DATASETS/lmdb",
    "resolution": 224,
    "frame_num": {"train": 1},
    "with_mask": False,
    "with_landmark": False,
    "use_data_augmentation": True,
    "mean": [0.48145466, 0.4578275, 0.40821073],
    "std": [0.26862954, 0.26130258, 0.27577711],
}

ds = LMDBDataset(config, "train")
print(f"Dataset length: {len(ds)}")
