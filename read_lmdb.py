"""Script to read and inspect LMDB database contents.

Can list keys, read specific entries, and optionally export data back to files.

Author: Kilo Code
"""

import argparse
import io
import logging
import sys
from pathlib import Path

import cv2
import lmdb
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def list_keys(lmdb_path: str, limit: int | None = None) -> list[str]:
    """List all keys in the LMDB database."""
    keys = []
    with (
        lmdb.open(str(lmdb_path), readonly=True, lock=False) as env,
        env.begin() as txn,
    ):
        cursor = txn.cursor()
        for key, _ in cursor:
            keys.append(key.decode())
            if limit and len(keys) >= limit:
                break
    return keys


def read_entry(lmdb_path: str, key: str) -> bytes | None:
    """Read a specific entry from LMDB."""
    with (
        lmdb.open(str(lmdb_path), readonly=True, lock=False) as env,
        env.begin() as txn,
    ):
        return txn.get(key.encode())


def decode_image(data: bytes) -> Image.Image | None:
    """Decode binary data as an image."""
    try:
        image_buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img)
    except Exception as e:
        logger.warning(f"Failed to decode as image: {e}")
    return None


def decode_landmarks(data: bytes) -> np.ndarray | None:
    """Decode binary data as landmarks."""
    try:
        return np.frombuffer(data, dtype=np.uint32).reshape((81, 2))
    except Exception as e:
        logger.warning(f"Failed to decode as landmarks: {e}")
    return None


def decode_numpy(data: bytes) -> np.ndarray | None:
    """Decode binary data as numpy array."""
    try:
        return np.load(io.BytesIO(data))
    except Exception as e:
        logger.warning(f"Failed to decode as numpy array: {e}")
    return None


def inspect_entry(key: str, data: bytes) -> dict:
    """Inspect an entry and return information about it."""
    info = {
        "key": key,
        "size": len(data),
        "type": "unknown",
        "shape": None,
        "dtype": None,
    }

    # Try to decode as different types
    if key.endswith((".jpg", ".jpeg", ".png", ".gif")):
        img = decode_image(data)
        if img:
            info["type"] = "image"
            info["shape"] = (*img.size, len(img.getbands()))
    elif key.endswith(".npy"):
        # Try landmarks first
        landmarks = decode_landmarks(data)
        if landmarks is not None:
            info["type"] = "landmarks"
            info["shape"] = landmarks.shape
            info["dtype"] = landmarks.dtype
        else:
            # Try general numpy array
            arr = decode_numpy(data)
            if arr is not None:
                info["type"] = "numpy_array"
                info["shape"] = arr.shape
                info["dtype"] = arr.dtype

    return info


def export_entry(key: str, data: bytes, output_dir: str) -> None:
    """Export an entry to a file."""
    output_path = Path(output_dir) / key
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if key.endswith((".jpg", ".jpeg", ".png", ".gif")):
            img = decode_image(data)
            if img:
                img.save(output_path)
                logger.info(f"Exported image to {output_path}")
        elif key.endswith(".npy"):
            # Try landmarks first
            landmarks = decode_landmarks(data)
            if landmarks is not None:
                np.save(output_path, landmarks)
                logger.info(f"Exported landmarks to {output_path}")
            else:
                # Try general numpy array
                arr = decode_numpy(data)
                if arr is not None:
                    np.save(output_path, arr)
                    logger.info(f"Exported numpy array to {output_path}")
        else:
            # Save as binary file
            with open(output_path, "wb") as f:
                f.write(data)
            logger.info(f"Exported binary data to {output_path}")
    except Exception as e:
        logger.error(f"Failed to export {key}: {e}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Read and inspect LMDB database contents",
    )
    parser.add_argument(
        "lmdb_path",
        help="Path to LMDB database directory",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List keys command
    list_parser = subparsers.add_parser("list", help="List all keys in the database")
    list_parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of keys to list",
    )

    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect specific keys")
    inspect_parser.add_argument(
        "keys",
        nargs="+",
        help="Keys to inspect",
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export entries to files")
    export_parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for exported files",
    )
    export_parser.add_argument(
        "keys",
        nargs="+",
        help="Keys to export",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "list":
            keys = list_keys(args.lmdb_path, args.limit)
            print(f"Found {len(keys)} keys:")
            for key in keys:
                print(f"  {key}")

        elif args.command == "inspect":
            for key in args.keys:
                data = read_entry(args.lmdb_path, key)
                if data is None:
                    print(f"Key '{key}' not found")
                    continue
                info = inspect_entry(key, data)
                print(f"Key: {info['key']}")
                print(f"  Size: {info['size']} bytes")
                print(f"  Type: {info['type']}")
                if info["shape"]:
                    print(f"  Shape: {info['shape']}")
                if info["dtype"]:
                    print(f"  Dtype: {info['dtype']}")
                print()

        elif args.command == "export":
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            for key in args.keys:
                data = read_entry(args.lmdb_path, key)
                if data is None:
                    logger.warning(f"Key '{key}' not found")
                    continue
                export_entry(key, data, args.output_dir)

        logger.info("Script completed successfully")

    except Exception as e:
        logger.exception(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
