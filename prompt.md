### problem
while running 
```
uv run 'DeepfakeBench/training/evaluate_finetune.py' \
	--detector_config \
		'DeepfakeBench/training/config/detector/effort_finetune.yaml' \
	--weights \
		'DeepfakeBench/training/weights/finetuned/newBatchFaces.pth' \
	--test_dataset \
		'/Volumes/Crucial/AI/DATASETS/TalkingHead_processed' \
	--output_dir 'evaluation_results'
```

I get 
```
ERROR:root:Failed to evaluate on /Volumes/Crucial/AI/DATASETS/TalkingHead_processed: No valid images found in /Volumes/Crucial/AI/DATASETS/TalkingHead_processed
Traceback (most recent call last):
  File "/Users/logan/Developer/vibes/Effort-AIGI-Detection/DeepfakeBench/training/evaluate_finetune.py", line 324, in main
    test_loader = prepare_test_data(config, [dataset_path], args.batch_size)
  File "/Users/logan/Developer/vibes/Effort-AIGI-Detection/DeepfakeBench/training/evaluate_finetune.py", line 138, in prepare_test_data
    test_set = DatasetFactory.create_dataset(
        config=test_config,
        mode="test",
        raw_data_root=test_dataset_paths,
    )
  File "/Users/logan/Developer/vibes/Effort-AIGI-Detection/DeepfakeBench/training/dataset/factory.py", line 59, in create_dataset
    return RawFileDataset(config, mode, raw_data_root[0])
  File "/Users/logan/Developer/vibes/Effort-AIGI-Detection/DeepfakeBench/training/dataset/raw_file_dataset.py", line 280, in __init__
    self._discover_and_build_index()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/Users/logan/Developer/vibes/Effort-AIGI-Detection/DeepfakeBench/training/dataset/raw_file_dataset.py", line 440, in _discover_and_build_index
    raise ValueError(f"No valid images found in {self.raw_data_root}")
ValueError: No valid images found in /Volumes/Crucial/AI/DATASETS/TalkingHead_processed
2026-01-16 08:11:13,585 - ERROR - Failed to evaluate on /Volumes/Crucial/AI/DATASETS/TalkingHead_processed: No valid images found in /Volumes/Crucial/AI/DATASETS/TalkingHead_processed
Traceback (most recent call last):
  File "/Users/logan/Developer/vibes/Effort-AIGI-Detection/DeepfakeBench/training/evaluate_finetune.py", line 324, in main
    test_loader = prepare_test_data(config, [dataset_path], args.batch_size)
  File "/Users/logan/Developer/vibes/Effort-AIGI-Detection/DeepfakeBench/training/evaluate_finetune.py", line 138, in prepare_test_data
    test_set = DatasetFactory.create_dataset(
        config=test_config,
        mode="test",
        raw_data_root=test_dataset_paths,
    )
  File "/Users/logan/Developer/vibes/Effort-AIGI-Detection/DeepfakeBench/training/dataset/factory.py", line 59, in create_dataset
    return RawFileDataset(config, mode, raw_data_root[0])
  File "/Users/logan/Developer/vibes/Effort-AIGI-Detection/DeepfakeBench/training/dataset/raw_file_dataset.py", line 280, in __init__
    self._discover_and_build_index()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/Users/logan/Developer/vibes/Effort-AIGI-Detection/DeepfakeBench/training/dataset/raw_file_dataset.py", line 440, in _discover_and_build_index
    raise ValueError(f"No valid images found in {self.raw_data_root}")
ValueError: No valid images found in /Volumes/Crucial/AI/DATASETS/TalkingHead_processed
```

### Run these commands to see the folder tree:

tree -d -L 5 /Volumes/Crucial/AI/DATASETS/TalkingHead_processed | head -n 1000
tree -L 5 /Volumes/Crucial/AI/DATASETS/TalkingHead_processed | head -n 1000

### what to do
The error seems to be because the dataset structure is more complicated. 
Originally, i wanted all datasets to follow the structure of [DATASET]/[Methods]/[train|val|test]/[fake|real]/<file> or [DATASET]/[train|val|test]/[fake|real]/<file>.
Now i want to also be able to take in datasets that follow the structure of [DATASET]/[Methods]/[train|val|test]/[fake|real]/[VIDEONAMES]/<file> and [DATASET]/[train|val|test]/[fake|real]/[VIDEONAMES]/<file>. 
Implement this change. Make sure to also update the metrics to handle the new dataset structure.