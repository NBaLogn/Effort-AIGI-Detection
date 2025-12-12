"""Will unpickle logs."""

import pickle
from pathlib import Path

with Path(
    "/Users/logan/Developer/WORK/DEEPFAKE_DETECTION/Effort-AIGI-Detection/DeepfakeBench/training/logs/effort_2025-12-12-10-21-14/train/UADFV/data_dict_train.pickle",
).open("rb") as f:
    obj = pickle.load(f)
print(obj)

# import pandas as pd

# df = pd.read_pickle(
#     "/Users/logan/Developer/WORK/DEEPFAKE_DETECTION/Effort-AIGI-Detection/DeepfakeBench/training/logs/effort_2025-12-12-10-21-14/test/UADFV/data_dict_test.pickle",
# )
# print(df)
