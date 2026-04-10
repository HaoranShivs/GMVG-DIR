"""Infrared Small Target Detection (IRSTD) Dataset Modules.

This module provides dataset classes for loading and preprocessing
infrared small target detection datasets including IRSTD-1k, NUDT-SIRST,
MDFA, and SIRST.
"""

import os
import os.path as osp
import shutil
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import torch
import torch.utils.data as Data

from data.utils import AugmentTransform as Augment_transform, mask2point

# Constants
DEFAULT_BASE_SIZE: int = 256
DEFAULT_OFFSET: int = 0
PIXEL_NORMALIZATION_FACTOR: float = 255.0


class IRSTD_Dataset(Data.Dataset):
    """IRSTD dataset for infrared small target detection.

    This dataset loads single-channel infrared images and their corresponding
    masks. Supports pseudo-labels, predicted labels, and point label generation.

    Attributes:
        data_dir: Path to the dataset directory (trainval or test).
        mode: Dataset mode, either 'train' or 'test'.
        base_size: Base size for image resizing.
        pt_label: Whether to generate point labels from masks.
        offset: Offset for point label generation.
        pseudo_label: Whether to load pseudo labels.
        predicted_label: Whether to load predicted labels.
        augment: Whether to apply data augmentation.
        turn_num: Turn number for pseudo/predicted label subdirectories.
        target_mix: Whether to apply target mixing (currently unused).
        names: List of image filenames.
        aug_transformer: Augmentation transformer instance.
    """

    def __init__(
        self,
        base_dir: str,
        mode: str = "train",
        base_size: int = DEFAULT_BASE_SIZE,
        pt_label: bool = False,
        offset: int = DEFAULT_OFFSET,
        pseudo_label: bool = False,
        predicted_label: bool = False,
        augment: bool = True,
        turn_num: str = "",
        target_mix: bool = False,
        file_name: str = "",
    ) -> None:
        """Initialize the IRSTD dataset.

        Args:
            base_dir: Root directory of the dataset.
            mode: Dataset mode ('train' or 'test').
            base_size: Base size for image resizing.
            pt_label: Whether to generate point labels from masks.
            offset: Offset for point label generation.
            pseudo_label: Whether to load pseudo labels.
            predicted_label: Whether to load predicted labels.
            augment: Whether to apply data augmentation.
            turn_num: Turn number for pseudo/predicted label subdirectories.
            target_mix: Whether to apply target mixing (currently unused).
            file_name: Subdirectory suffix for images folder.
        """
        if mode not in ["train", "test"]:
            raise ValueError(f"Mode must be 'train' or 'test', got '{mode}'")

        # Set data directory based on mode
        split_dir = "trainval" if mode == "train" else "test"
        self.data_dir = osp.join(base_dir, split_dir)

        # Store configuration
        self.mode = mode
        self.base_size = base_size
        self.pt_label = pt_label
        self.offset = offset
        self.pseudo_label = pseudo_label
        self.predicted_label = predicted_label
        self.augment = augment
        self.turn_num = turn_num
        self.target_mix = target_mix

        # Load image filenames
        self.names: List[str] = []
        images_dir = osp.join(self.data_dir, f"images{file_name}")
        if not osp.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        for filename in os.listdir(images_dir):
            if filename.endswith(".png"):
                self.names.append(filename)

        # Initialize augmentation transformer
        aug_mode = mode if augment else "test"
        self.aug_transformer = Augment_transform(base_size, aug_mode)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            A tuple of (image, mask) tensors. Image is single channel [1, H, W].
            Mask may contain multiple channels if pseudo/predicted labels are loaded.
        """
        name = self.names[index]
        img_path = osp.join(self.data_dir, "images", name)
        pseudo_label_path = osp.join(
            self.data_dir, f"pixel_pseudo_label{self.turn_num}_", name
        )
        predicted_label_path = osp.join(
            self.data_dir, f"preded_label/{self.turn_num}", name
        )
        label_path = osp.join(self.data_dir, "masks", name)

        # Load image and mask as single-channel tensors
        img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask_np = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if img_np is None or mask_np is None:
            raise FileNotFoundError(
                f"Failed to load image or mask for {name}. "
                f"Image path: {img_path}, Mask path: {label_path}"
            )

        img = torch.from_numpy(img_np).unsqueeze(0).float()
        mask = torch.from_numpy(mask_np).unsqueeze(0).float()

        # Load pseudo label if enabled
        if self.pseudo_label:
            pseudo_label_np = cv2.imread(pseudo_label_path, cv2.IMREAD_GRAYSCALE)
            if pseudo_label_np is None:
                raise FileNotFoundError(f"Pseudo label not found: {pseudo_label_path}")
            pseudo_label = torch.from_numpy(pseudo_label_np).unsqueeze(0).float()
            mask = torch.cat([mask, pseudo_label], dim=0)

        # Load predicted label if enabled
        if self.predicted_label:
            predicted_label_np = cv2.imread(predicted_label_path, cv2.IMREAD_GRAYSCALE)
            if predicted_label_np is None:
                raise FileNotFoundError(f"Predicted label not found: {predicted_label_path}")
            predicted_label = torch.from_numpy(predicted_label_np).unsqueeze(0).float()
            mask = torch.cat([mask, predicted_label], dim=0)

        # Apply augmentation
        img, mask = self.aug_transformer(img, mask)

        # Normalize to [0, 1]
        img = img / PIXEL_NORMALIZATION_FACTOR
        mask = mask / PIXEL_NORMALIZATION_FACTOR

        # Convert mask to point label if enabled
        if self.pt_label:
            pt_label = mask2point(mask[0].unsqueeze(0), img, self.offset)
            mask[0] = pt_label

        return img, mask

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.names)


def organize_dataset(training_dir: str, target_base_path: str) -> None:
    """Organize dataset by separating images and masks.

    Copies files ending with '_1' as images and '_2' as masks to separate
    subdirectories, removing the suffix from filenames.

    Args:
        training_dir: Source directory containing mixed image/mask files.
        target_base_path: Base directory for organized output.
    """
    image_target = osp.join(target_base_path, "image")
    mask_target = osp.join(target_base_path, "mask")

    # Create target directories
    os.makedirs(image_target, exist_ok=True)
    os.makedirs(mask_target, exist_ok=True)

    # Process all files in training directory
    for filename in os.listdir(training_dir):
        file_path = osp.join(training_dir, filename)

        if not osp.isfile(file_path):
            continue  # Skip non-file items

        # Split filename and extension
        name, ext = osp.splitext(filename)

        # Copy based on suffix pattern
        if name.endswith("_1"):
            # Image file: remove '_1' suffix
            new_name = name[:-2] + ext
            dest = osp.join(image_target, new_name)
            shutil.copy2(file_path, dest)

        elif name.endswith("_2"):
            # Mask file: remove '_2' suffix
            new_name = name[:-2] + ext
            dest = osp.join(mask_target, new_name)
            shutil.copy2(file_path, dest)


def generate_and_save_point_labels(
    dataset_class,
    base_dir: str,
    output_dir: str,
    base_size: int = 256,
) -> None:
    """Generate point labels with offset=0 and save as PNG files.

    Args:
        dataset_class: Dataset class to use (e.g., IRSTD_Dataset).
        base_dir: Root directory of the dataset.
        output_dir: Directory to save point labels.
        base_size: Input size for resizing.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize dataset with pt_label enabled and offset=0
    dataset = dataset_class(
        base_dir=base_dir,
        mode="train",
        base_size=base_size,
        pt_label=True,      # Enable point label generation
        offset=0,           # Zero offset
        pseudo_label=False,
        predicted_label=False,
        augment=False,      # No augmentation to preserve point positions
    )

    print(f"Processing {len(dataset)} images...")

    for i in range(len(dataset)):
        name = dataset.names[i]
        _, mask_with_pt = dataset[i]  # mask_with_pt[0] is the point label

        # Extract point label (first channel)
        pt_label = mask_with_pt[0]  # shape: [H, W], values 0 or 1 (normalized)

        # Convert back to 0-255 uint8 format
        pt_label_np = (pt_label * 255).byte().cpu().numpy()

        # Save path
        save_path = os.path.join(output_dir, name)
        cv2.imwrite(save_path, pt_label_np)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(dataset)}")

    print(f"Point labels saved to: {output_dir}")


def resize_and_save_dataset(
    dataset_class,
    src_base_dir: str,
    dst_base_dir: str,
    base_size: int = 256
) -> None:
    """Resize dataset images and masks to base_size and save to new directory.

    Directory structure:
        dst_base_dir/
            ├── trainval/
            │   ├── images/
            │   └── masks/
            └── test/
                ├── images/
                └── masks/

    Args:
        dataset_class: Dataset class to use.
        src_base_dir: Source dataset root directory.
        dst_base_dir: Destination dataset root directory.
        base_size: Target size for resizing.
    """
    for mode in ["train"]:
        print(f"\nProcessing {mode} set...")

        # Create target directories
        if mode == "train":
            dst_img_dir = os.path.join(dst_base_dir, "trainval", "images")
            dst_mask_dir = os.path.join(dst_base_dir, "trainval", "masks")
        else:
            dst_img_dir = os.path.join(dst_base_dir, "test", "images")
            dst_mask_dir = os.path.join(dst_base_dir, "test", "masks")

        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_mask_dir, exist_ok=True)

        # Initialize dataset
        dataset = dataset_class(
            base_dir=src_base_dir,
            mode=mode,
            base_size=base_size,
            pt_label=False,
            offset=0,
            pseudo_label=False,
            predicted_label=False,
            augment=False,
        )

        print(f"Processing {len(dataset)} images...")

        for i in range(len(dataset)):
            name = dataset.names[i]
            img, mask = dataset[i]  # img: [1, H, W], mask: [1, H, W] (values 0~1)

            mask = (mask.float() > 0.5).float()

            # Convert to numpy uint8 (0-255)
            img_np = (img.squeeze(0).cpu().numpy() * 255).astype('uint8')
            mask_np = (mask.squeeze(0).cpu().numpy() * 255).astype('uint8')

            # Save
            cv2.imwrite(os.path.join(dst_img_dir, name), img_np)
            cv2.imwrite(os.path.join(dst_mask_dir, name), mask_np)

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(dataset)}")

    print(f"\nAll data saved to: {dst_base_dir}")


def split_dataset_by_index_with_mask_prefix_match(root_dir: str) -> None:
    """Split dataset by index with mask prefix matching.

    Matches image filenames with mask files by prefix (e.g., Misc_25.png
    matches Misc_25_pixels0.png), then splits by index parity and renames
    masks to match image filenames.

    Expected structure:
        - images/: Misc_25.png
        - masks/: Misc_25_pixels0.png, Misc_25_pixels1.png, etc.

    Args:
        root_dir: Root directory containing 'images' and 'masks' folders.
    """
    root = Path(root_dir)
    img_dir = root / "images"
    mask_dir = root / "masks"

    if not (img_dir.exists() and mask_dir.exists()):
        raise FileNotFoundError(
            f"Please ensure 'images' and 'masks' folders exist in {root}"
        )

    # Get all image files (.png), sorted
    image_files = sorted(
        [f for f in img_dir.glob("*.png")], key=lambda x: x.name
    )
    if not image_files:
        raise ValueError("No .png files found in images folder!")

    # Build mask filename mapping: mask_stem -> full_path
    mask_files = list(mask_dir.glob("*.png"))
    mask_stem_to_path = {}
    for mf in mask_files:
        stem = mf.stem
        # Simple strategy: truncate at "_pixels" if present
        if "_pixels" in stem:
            base_stem = stem.split("_pixels")[0]
        else:
            base_stem = stem
        mask_stem_to_path[stem] = mf

    # Pair list: [(img_path, mask_path), ...]
    paired_files = []
    for img_path in image_files:
        img_stem = img_path.stem  # e.g., "Misc_25"
        img_name = img_path.name  # e.g., "Misc_25.png"

        # Find mask files whose stem starts with img_stem
        matched_masks = []
        for mask_stem, mask_path in mask_stem_to_path.items():
            if mask_stem.startswith(img_stem):
                # Ensure exact match or "_pixelsX" suffix
                if mask_stem == img_stem or mask_stem.startswith(img_stem + "_"):
                    matched_masks.append(mask_path)

        if len(matched_masks) == 0:
            print(f"Warning: No matching mask found for {img_name}, skipping.")
            continue
        elif len(matched_masks) > 1:
            print(
                f"Warning: Multiple masks found for {img_name}, "
                f"using first: {[m.name for m in matched_masks]}"
            )

        mask_path = matched_masks[0]
        paired_files.append((img_path, mask_path))

    if not paired_files:
        raise ValueError("No image-mask pairs found!")

    print(f"Paired {len(paired_files)} image-mask pairs.")

    # Create output directories
    for split in ["trainval", "test"]:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "masks").mkdir(parents=True, exist_ok=True)

    # Split by index parity
    for idx, (img_path, mask_path) in enumerate(paired_files):
        target_split = "trainval" if idx % 2 == 0 else "test"
        target_img_dir = root / target_split / "images"
        target_mask_dir = root / target_split / "masks"

        # Copy image (keep original name)
        shutil.copy2(img_path, target_img_dir / img_path.name)

        # Copy mask, but rename to match image filename
        new_mask_name = img_path.name
        shutil.copy2(mask_path, target_mask_dir / new_mask_name)

    print("Split completed!")
    print(f"   trainval: {len(paired_files[::2])} pairs")
    print(f"   test:     {len(paired_files[1::2])} pairs")
