# Image Sampling Script

This script samples a specified number of images from 'ai' and 'nature' folders across multiple methods in a deepfake detection dataset.

## Overview

The script creates a new directory with the same structure as the source but with sampled images. It's designed to work with the following directory structure:

```
root/
├── Method1/
│   ├── train/
│   │   ├── ai/
│   │   └── nature/
│   └── val/
│       ├── ai/
│       └── nature/
├── Method2/
│   └── ...
...
└── Method8/
    └── ...
```

## Features

- **Random Sampling**: Uses `random.sample()` for reproducible image selection
- **Flexible Limits**: Configurable sample size per category (default: 500)
- **Comprehensive Logging**: Detailed progress reporting and warnings
- **Error Handling**: Gracefully handles missing directories and insufficient images
- **Safe Operations**: Copies files preserving metadata, doesn't modify originals
- **Standard Library Only**: No external dependencies required

## Usage

### Basic Usage

```bash
uv run sample_images.py --source_dir /path/to/dataset
```

### Advanced Usage

```bash
# Sample 300 images per category
uv run sample_images.py --source_dir /path/to/dataset --sample_size 300

# Sample 1000 images per category
uv run sample_images.py --source_dir /path/to/dataset --sample_size 1000
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--source_dir` | Path to the source dataset directory | Required |
| `--sample_size` | Number of images to sample per category | 500 |

## Output Structure

The script creates a new directory with `_sampled` suffix:

```
root_sampled/
├── Method1/
│   ├── train/
│   │   ├── ai/ (sampled images)
│   │   └── nature/ (sampled images)
│   └── val/
│       ├── ai/ (sampled images)
│       └── nature/ (sampled images)
...
```

## Examples

### Example 1: Default Sampling (500 images)

```bash
uv run sample_images.py --source_dir /data/deepfake_dataset
```

This will:
- Sample 500 images from each 'ai' and 'nature' folder
- Create `/data/deepfake_dataset_sampled` with the same structure
- Log progress and any warnings for insufficient images

### Example 2: Custom Sample Size

```bash
uv run sample_images.py --source_dir /data/dataset --sample_size 200
```

This will sample 200 images per category instead of the default 500.

### Example 3: With Logging

The script automatically logs to stdout with timestamps:

```
2025-12-16 13:48:52,502 - INFO - Processing dataset from /data/dataset
2025-12-16 13:48:52,502 - INFO - Output directory: /data/dataset_sampled
2025-12-16 13:48:52,502 - INFO - Sample size per category: 500
2025-12-16 13:48:52,502 - INFO - Processing Method1
2025-12-16 13:48:52,503 - INFO - Sampled 500 images from 1000 available
2025-12-16 13:48:52,505 - INFO - Sampled 500 images from 800 available
2025-12-16 13:48:52,506 - WARNING - Only 300 images found in Method1/train/nature, using all available
2025-12-16 13:48:52,508 - WARNING - Method directory /data/dataset/Method2 does not exist
2025-12-16 13:48:52,508 - INFO - Image sampling completed
```

## Supported Image Formats

The script automatically detects and processes the following image formats:
- `.png`, `.PNG`
- `.jpg`, `.jpeg`, `.JPG`, `.JPEG`

## Error Handling

The script handles various edge cases:

1. **Missing Directories**: Warns if Method directories don't exist
2. **Insufficient Images**: Warns and uses all available images if fewer than requested
3. **Missing Categories**: Warns if 'ai' or 'nature' folders are missing
4. **Empty Directories**: Warns if directories exist but contain no images

## Requirements

- Python 3.8+
- Standard library only (no external dependencies)

## Installation

No installation required. Simply run the script with uv:

```bash
uv run sample_images.py [arguments]
```

## Tips

1. **Reproducibility**: The script uses Python's built-in `random.sample()` which is deterministic for a given seed
2. **Memory Efficiency**: The script processes images in memory without loading them, making it memory-efficient
3. **Preservation**: All file metadata (timestamps, permissions) is preserved during copying
4. **Scalability**: Can handle large datasets efficiently due to minimal memory usage

## Troubleshooting

### Common Issues

1. **Permission Errors**: Ensure you have read access to source directory and write access to parent directory
2. **Path Issues**: Use absolute paths to avoid relative path confusion
3. **Large Datasets**: For very large datasets, consider using smaller sample sizes to reduce processing time

### Getting Help

If you encounter issues:
1. Check the logging output for specific error messages
2. Verify your directory structure matches the expected format
3. Ensure you have sufficient disk space for the output directory