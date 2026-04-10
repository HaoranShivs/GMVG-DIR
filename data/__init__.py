"""Data modules for infrared small target detection datasets.

This package provides dataset classes and utility functions for loading,
preprocessing, and managing infrared small target detection (IRSTD) datasets.
"""

from data.sirst import (
    IRSTD_Dataset,
    generate_and_save_point_labels,
    organize_dataset,
    resize_and_save_dataset,
    split_dataset_by_index_with_mask_prefix_match,
)
from data.utils import (
    AugmentTransform,
    Augmentation,
    RandomResize,
    Rotate4DTransform,
    mask2point,
    mask2point_n,
)

__all__ = [
    "IRSTD_Dataset",
    "AugmentTransform",
    "Augmentation",
    "RandomResize",
    "Rotate4DTransform",
    "mask2point",
    "mask2point_n",
    "organize_dataset",
    "generate_and_save_point_labels",
    "resize_and_save_dataset",
    "split_dataset_by_index_with_mask_prefix_match",
]