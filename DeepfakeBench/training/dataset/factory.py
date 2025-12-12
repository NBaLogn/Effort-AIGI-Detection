#!/usr/bin/env python3
# author: Kilo Code
# date: 2025-12-11
# description: Dataset factory for unified dataset creation

import logging

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset

try:
    from dataset.raw_file_dataset import RawFileDataset
except ImportError:
    RawFileDataset = None

logger = logging.getLogger(__name__)


class DatasetFactory:
    """Factory class for creating appropriate dataset instances based on configuration."""

    @staticmethod
    def create_dataset(
        config: dict,
        mode: str,
        raw_data_root: str | None = None,
    ) -> DeepfakeAbstractBaseDataset | RawFileDataset:
        """Create the appropriate dataset based on configuration and available data.

        Args:
            config: Configuration dictionary
            mode: "train" or "test" mode
            raw_data_root: Optional path to raw data directory

        Returns:
            Appropriate dataset instance

        Raises:
            ValueError: If neither JSON nor raw file approach can be used
        """
        # Check if raw data processing is requested and available
        if raw_data_root:
            if RawFileDataset is None:
                msg = (
                    "Raw file dataset support is not available. "
                    "Please ensure raw_file_dataset.py is accessible."
                )
                raise ImportError(
                    msg,
                )

            logger.info("Using RawFileDataset with data from %s", raw_data_root)
            return RawFileDataset(config, mode, raw_data_root)

        # Fall back to traditional JSON-based approach
        logger.info("Using traditional JSON-based DeepfakeAbstractBaseDataset")
        return DeepfakeAbstractBaseDataset(config, mode)

    @staticmethod
    def get_dataset_class_name(
        config: dict,
        raw_data_root: str | None = None,
    ) -> str:
        """Get the name of the dataset class that would be created."""
        if raw_data_root and RawFileDataset is not None:
            return "RawFileDataset"
        return "DeepfakeAbstractBaseDataset"
