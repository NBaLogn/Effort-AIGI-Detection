# Face Detection Filter Script

A Python script that recursively processes all images in a folder, filters out images containing human faces, and copies them to another folder while preserving the directory structure.

## Features

- **Recursive Processing**: Automatically walks through all subdirectories
- **Fast Face Detection**: Uses OpenCV's Haar cascades for efficient detection
- **Multiple Image Formats**: Supports JPG, PNG, JPEG, GIF, BMP, TIFF, WebP
- **Directory Structure Preservation**: Maintains folder hierarchy in destination
- **Comprehensive Logging**: Detailed logs with timestamps and statistics
- **Configurable Parameters**: Adjustable detection sensitivity and thresholds
- **Dry Run Mode**: Preview what would be copied without making changes
- **Error Handling**: Graceful handling of corrupted images and edge cases

## Requirements

- Python 3.8+
- OpenCV (`opencv-python>=4.11.0.86`) - already included in your project

## Installation

The script uses OpenCV which is already listed in your [`pyproject.toml`](pyproject.toml). To ensure it's installed:

```bash
uv pip install opencv-python
```

## Usage

### Basic Usage

```bash
python face_detection_filter.py -s /Users/name/Pictures -d /Users/name/Faces_Only
```

### Advanced Usage

```bash
# With custom detection parameters
python face_detection_filter.py -s /path/to/source -d /path/to/destination \
    --min-size 80 --scale-factor 1.2 --min-neighbors 3

# Dry run to see what would be copied
python face_detection_filter.py -s /path/to/source -d /path/to/destination --dry-run

# Verbose logging
python face_detection_filter.py -s /path/to/source -d /path/to/destination --log-level DEBUG
```

## Command Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--source` | `-s` | Source directory containing images | Required |
| `--destination` | `-d` | Destination directory for filtered images | Required |
| `--min-size` | | Minimum face size to detect (pixels) | 50 |
| `--scale-factor` | | Scale factor for face detection | 1.1 |
| `--min-neighbors` | | Minimum neighbors for face detection | 5 |
| `--log-level` | | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |
| `--dry-run` | | Show what would be done without copying | False |

## Examples

### Example 1: Basic Face Filtering
```bash
# Filter faces from Pictures folder
python face_detection_filter.py -s ~/Pictures -d ~/Faces_Only
```

### Example 2: Strict Face Detection
```bash
# Only detect larger faces with higher confidence
python face_detection_filter.py -s /photos/archive -d /photos/with_faces \
    --min-size 100 --scale-factor 1.3 --min-neighbors 6
```

### Example 3: Preview Before Running
```bash
# See what would be copied without making changes
python face_detection_filter.py -s /Users/name/Photos -d /Users/name/Faces \
    --dry-run
```

### Example 4: Detailed Logging
```bash
# Get detailed logs for troubleshooting
python face_detection_filter.py -s /path/to/source -d /path/to/destination \
    --log-level DEBUG
```

## Output

### Console Output
```
2025-12-17 10:15:30 - INFO - Logging to: logs/face_detection_20251217_101530.log
2025-12-17 10:15:30 - INFO - Successfully loaded Haar cascade classifier
2025-12-17 10:15:30 - INFO - Initialized FaceDetectionFilter:
2025-12-17 10:15:30 - INFO -   Source: /Users/name/Pictures
2025-12-17 10:15:30 - INFO -   Destination: /Users/name/Faces_Only
2025-12-17 10:15:30 - INFO -   Min face size: 50px
2025-12-17 10:15:30 - INFO -   Scale factor: 1.1
2025-12-17 10:15:30 - INFO -   Min neighbors: 5
2025-12-17 10:15:30 - INFO - Starting directory processing...
2025-12-17 10:15:30 - INFO - Source directory: /Users/name/Pictures
2025-12-17 10:15:30 - INFO - Destination directory: /Users/name/Faces_Only
2025-12-17 10:15:30 - INFO - Dry run mode: False
2025-12-17 10:15:45 - INFO - ============================================================
2025-12-17 10:15:45 - INFO - PROCESSING SUMMARY
2025-12-17 10:15:45 - INFO - ============================================================
2025-12-17 10:15:45 - INFO - Total images processed: 1542
2025-12-17 10:15:45 - INFO - Images with faces: 324
2025-12-17 10:15:45 - INFO - Images without faces: 1218
2025-12-17 10:15:45 - INFO - Skipped files (non-images): 45
2025-12-17 10:15:45 - INFO - Errors encountered: 2
2025-12-17 10:15:45 - INFO - Face detection rate: 21.0%
2025-12-17 10:15:45 - INFO - ============================================================
```

### Log Files
Detailed logs are saved to `logs/face_detection_YYYYMMDD_HHMMSS.log` with:
- Timestamped entries
- Processing statistics
- Error details
- Debug information (if enabled)

## Directory Structure Preservation

The script maintains the original directory structure in the destination:

```
Source:
├── vacation/
│   ├── beach.jpg
│   └── family/
│       └── group_photo.png
└── work/
    └── meeting.jpeg

Destination (after filtering):
├── vacation/
│   ├── beach.jpg (if faces detected)
│   └── family/
│       └── group_photo.png (if faces detected)
└── work/
    └── meeting.jpeg (if faces detected)
```

## Performance Tips

1. **Adjust Detection Sensitivity**:
   - Higher `--min-size` values detect only larger faces (faster)
   - Higher `--scale-factor` values are faster but less thorough
   - Higher `--min-neighbors` values are more strict (fewer false positives)

2. **Use Dry Run First**:
   ```bash
   python face_detection_filter.py -s /path/to/source -d /path/to/destination --dry-run
   ```

3. **Monitor Logs**:
   Check `logs/` directory for detailed processing information and any errors.

## Troubleshooting

### Common Issues

1. **"Could not load face cascade classifier"**
   - Ensure OpenCV is properly installed: `uv pip install opencv-python`
   - Check that cascade files are available in OpenCV installation

2. **"Permission denied" errors**
   - Ensure write permissions to destination directory
   - Check if destination directory exists or can be created

3. **High false positive rate**
   - Increase `--min-neighbors` (try 6-8)
   - Increase `--min-size` for your use case
   - Increase `--scale-factor` (try 1.2-1.3)

4. **Missing faces**
   - Decrease `--min-neighbors` (try 3-4)
   - Decrease `--min-size` for smaller faces
   - Decrease `--scale-factor` (try 1.05-1.1)

### Debug Mode
Enable debug logging for detailed information:
```bash
python face_detection_filter.py -s /path/to/source -d /path/to/destination --log-level DEBUG
```

## Testing

Run the validation script to ensure everything works correctly:
```bash
python validate_script.py
```

This tests:
- ✅ Script import and dependencies
- ✅ OpenCV and cascade files availability
- ✅ Command line argument parsing
- ✅ Logging configuration

## Technical Details

### Face Detection Algorithm
- Uses OpenCV's Haar cascade classifier (`haarcascade_frontalface_default.xml`)
- Fast and lightweight compared to deep learning approaches
- Configurable sensitivity via command line parameters

### Image Processing
- Converts images to grayscale for detection
- Uses `detectMultiScale()` with configurable parameters
- Handles various image formats automatically

### File Operations
- Preserves directory structure using `pathlib`
- Creates destination directories as needed
- Uses `shutil.copy2()` to preserve metadata

## License

This script is part of the Effort-AIGI-Detection project. See the project's LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Enable debug logging
3. Review log files in the `logs/` directory
4. Create an issue with relevant log files and error messages