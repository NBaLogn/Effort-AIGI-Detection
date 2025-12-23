"""Debug Script to Inspect Pickle Files.

This is a simple debug utility to inspect the contents of pickle files
from training logs. It loads and prints the Python object stored in a pickle file.

Note: This is a debug script with a hardcoded path. Modify the path as needed.

Usage:
    python inspect_pickle_file.py
"""

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
