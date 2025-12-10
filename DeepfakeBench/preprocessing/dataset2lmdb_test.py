import argparse
import logging
import os

import lmdb
import numpy as np
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def file_to_binary(file_path):
    """Convert image to binary data"""
    try:
        if file_path.endswith(".npy"):
            data = np.load(file_path)
            file_binary = data.tobytes()
        else:
            with open(file_path, "rb") as f:
                file_binary = f.read()
        return file_binary
    except (OSError, ValueError) as e:
        logging.exception(f"Error reading file {file_path}: {e}")
        raise


def create_lmdb_dataset(source_folder, lmdb_path, dataset_name, map_size):
    """Create LMDB dataset"""
    try:
        # Open LMDB file, create database
        with lmdb.open(lmdb_path, map_size=map_size) as db:
            with db.begin(write=True) as txn:
                for root, dirs, files in os.walk(source_folder, followlinks=True):
                    logging.debug(f"Processing directory: {root}")
                    if "video" in root:
                        continue
                    for file in files:
                        logging.debug(f"Processing file: {file}")
                        image_path = os.path.join(root, file)
                        # Generate relative path key
                        relative_path = os.path.join(
                            dataset_name,
                            os.path.relpath(image_path, source_folder),
                        ).replace(
                            "\\",
                            "/",
                        )  # Ensure forward slashes for cross-platform
                        key = relative_path.encode("utf-8")
                        value = file_to_binary(image_path)

                        # Write to database
                        txn.put(key, value)
        logging.info(f"LMDB dataset created at {lmdb_path}")
    except lmdb.Error as e:
        logging.exception(f"LMDB error: {e}")
        raise
    except Exception as e:
        logging.exception(f"Unexpected error in create_lmdb_dataset: {e}")
        raise


def read_lmdb(
    lmdb_dir_path,
    key="npy_test/000_003/000.npy",
    dtype=np.uint32,
    shape=(81, 2),
):
    """Validate the key and value in the generated LMDB"""
    try:
        with lmdb.open(lmdb_dir_path) as env, env.begin(write=False) as txn:
            # Key for validation
            binary = txn.get(key.encode())
            if binary is None:
                logging.error(f"Key '{key}' not found in LMDB")
                return None
            data = np.frombuffer(binary, dtype=dtype).reshape(shape)
            logging.info(f"Successfully read data for key '{key}': shape {data.shape}")
            return data
    except lmdb.Error as e:
        logging.exception(f"LMDB error in read_lmdb: {e}")
        raise
    except ValueError as e:
        logging.exception(f"Data reshaping error: {e}")
        raise
    except Exception as e:
        logging.exception(f"Unexpected error in read_lmdb: {e}")
        raise


# Create ArgumentParser object
parser = argparse.ArgumentParser(description="Process some inputs.")

# Add --dataset_size parameter
parser.add_argument(
    "--dataset_size",
    type=int,
    default=25,
    help="LMDB requires pre-specifying the total dataset size (GB)",
)

# Parse arguments
args = parser.parse_args()

if __name__ == "__main__":
    # Validate arguments
    if args.dataset_size <= 0:
        logging.error("dataset_size must be a positive integer")
        exit(1)

    # from config.yaml load parameters
    yaml_path = "/Users/logan/Developer/WORK/DEEPFAKE_DETECTION/Effort-AIGI-Detection/DeepfakeBench/preprocessing/config.yaml"
    try:
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        logging.exception(f"Error loading YAML config: {e}")
        exit(1)

    if "to_lmdb" not in config:
        logging.error("Missing 'to_lmdb' section in config")
        exit(1)

    config = config["to_lmdb"]
    dataset_name = config["dataset_name"]["default"]
    dataset_size = args.dataset_size
    dataset_root_path = config["dataset_root_path"]["default"]
    output_lmdb_dir = config["output_lmdb_dir"]["default"]

    try:
        os.makedirs(output_lmdb_dir, exist_ok=True)
    except OSError as e:
        logging.exception(f"Error creating output directory: {e}")
        exit(1)

    dataset_dir_path = f"{dataset_root_path}/{dataset_name}"
    lmdb_path = f"{output_lmdb_dir}/{dataset_name}_lmdb"

    try:
        create_lmdb_dataset(
            dataset_dir_path,
            lmdb_path,
            dataset_name,
            map_size=int(dataset_size) * 1024 * 1024 * 1024,
        )
        # Uncomment to test reading
        # read_lmdb(lmdb_path)
    except Exception as e:
        logging.exception(f"Error during LMDB creation: {e}")
        exit(1)
