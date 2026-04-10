"""
Infrared Small Target Detection (IRSTD) Dataset Modules.

This module provides dataset classes for loading and preprocessing
infrared small target detection datasets including IRSTD-1k, NUDT-SIRST,
MDFA, and SIRST.
"""

import os
import os.path as osp
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms

from data.utils import Augment_transform, mask2point

# Constants
DEFAULT_BASE_SIZE = 256
DEFAULT_OFFSET = 0
PIXEL_NORMALIZATION_FACTOR = 255.0


__all__ = [
    "SIRSTDataset",
    "IRSTD1kDataset",
    "NUDTDataset",
    "MDFADataset",
]


class IRSTD1kDataset(Data.Dataset):
    """IRSTD-1k dataset for infrared small target detection.

    This dataset loads single-channel infrared images and their corresponding
    masks. Supports pseudo-labels, predicted labels, and point label generation.

    Attributes:
        data_dir (str): Path to the dataset directory (trainval or test).
        mode (str): Dataset mode, either 'train' or 'test'.
        base_size (int): Base size for image resizing.
        pt_label (bool): Whether to generate point labels from masks.
        offset (int): Offset for point label generation.
        pseudo_label (bool): Whether to load pseudo labels.
        preded_label (bool): Whether to load predicted labels.
        augment (bool): Whether to apply data augmentation.
        turn_num (str): Turn number for pseudo/predicted label subdirectories.
        target_mix (bool): Whether to apply target mixing (currently unused).
        names (List[str]): List of image filenames.
        aug_transformer: Augmentation transformer instance.
    """

    def __init__(
        self,
        base_dir: str = r"W:/DataSets/Infraid_datasets/IRSTD-1k",
        mode: str = "train",
        base_size: int = DEFAULT_BASE_SIZE,
        pt_label: bool = False,
        offset: int = DEFAULT_OFFSET,
        pseudo_label: bool = False,
        preded_label: bool = False,
        augment: bool = True,
        turn_num: str = "",
        target_mix: bool = False,
        file_name: str = "",
    ) -> None:
        """Initialize the IRSTD-1k dataset.

        Args:
            base_dir: Root directory of the dataset.
            mode: Dataset mode ('train' or 'test').
            base_size: Base size for image resizing.
            pt_label: Whether to generate point labels from masks.
            offset: Offset for point label generation.
            pseudo_label: Whether to load pseudo labels.
            preded_label: Whether to load predicted labels.
            augment: Whether to apply data augmentation.
            turn_num: Turn number for pseudo/predicted label subdirectories.
            target_mix: Whether to apply target mixing (currently unused).
            file_name: Subdirectory suffix for images folder.
        """
        assert mode in ["train", "test"], f"Mode must be 'train' or 'test', got '{mode}'"

        # Set data directory based on mode
        split_dir = "trainval" if mode == "train" else "test"
        self.data_dir = osp.join(base_dir, split_dir)

        # Store configuration
        self.mode = mode
        self.base_size = base_size
        self.pt_label = pt_label
        self.offset = offset
        self.pseudo_label = pseudo_label  # Fixed typo: was 'pesudo_label'
        self.preded_label = preded_label
        self.augment = augment  # Fixed: was 'aug'
        self.turn_num = turn_num
        self.target_mix = target_mix

        # Load image filenames
        self.names: List[str] = []
        images_dir = osp.join(self.data_dir, f"images{file_name}")
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
        preded_label_path = osp.join(
            self.data_dir, f"preded_label/{self.turn_num}", name
        )
        label_path = osp.join(self.data_dir, "masks", name)

        # Load image and mask as single-channel tensors
        img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask_np = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        img = torch.from_numpy(img_np).unsqueeze(0).float()
        mask = torch.from_numpy(mask_np).unsqueeze(0).float()

        # Load pseudo label if enabled
        if self.pseudo_label:
            pseudo_label_np = cv2.imread(pseudo_label_path, cv2.IMREAD_GRAYSCALE)
            pseudo_label = torch.from_numpy(pseudo_label_np).unsqueeze(0).float()
            mask = torch.cat([mask, pseudo_label], dim=0)

        # Load predicted label if enabled
        if self.preded_label:
            preded_label_np = cv2.imread(preded_label_path, cv2.IMREAD_GRAYSCALE)
            preded_label = torch.from_numpy(preded_label_np).unsqueeze(0).float()
            mask = torch.cat([mask, preded_label], dim=0)

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


class NUDTDataset(Data.Dataset):
    """NUDT-SIRST dataset for infrared small target detection.

    This dataset loads single-channel infrared images and their corresponding
    masks. Supports pseudo-labels, predicted labels, and point label generation.

    Attributes:
        data_dir (str): Path to the dataset directory (trainval or test).
        mode (str): Dataset mode, either 'train' or 'test'.
        base_size (int): Base size for image resizing.
        pt_label (bool): Whether to generate point labels from masks.
        offset (int): Offset for point label generation.
        pseudo_label (bool): Whether to load pseudo labels.
        preded_label (bool): Whether to load predicted labels.
        augment (bool): Whether to apply data augmentation.
        turn_num (str): Turn number for pseudo/predicted label subdirectories.
        target_mix (bool): Whether to apply target mixing (currently unused).
        names (List[str]): List of image filenames.
        aug_transformer: Augmentation transformer instance.
    """

    def __init__(
        self,
        base_dir: str = r"W:/DataSets/Infraid_datasets/NUDT-SIRST",
        mode: str = "train",
        base_size: int = DEFAULT_BASE_SIZE,
        pt_label: bool = False,
        offset: int = DEFAULT_OFFSET,
        pseudo_label: bool = False,
        preded_label: bool = False,
        augment: bool = True,
        turn_num: str = "",
        target_mix: bool = False,
        file_name: str = "",
    ) -> None:
        """Initialize the NUDT-SIRST dataset.

        Args:
            base_dir: Root directory of the dataset.
            mode: Dataset mode ('train' or 'test').
            base_size: Base size for image resizing.
            pt_label: Whether to generate point labels from masks.
            offset: Offset for point label generation.
            pseudo_label: Whether to load pseudo labels.
            preded_label: Whether to load predicted labels.
            augment: Whether to apply data augmentation.
            turn_num: Turn number for pseudo/predicted label subdirectories.
            target_mix: Whether to apply target mixing (currently unused).
            file_name: Subdirectory suffix for images folder.
        """
        assert mode in ["train", "test"], f"Mode must be 'train' or 'test', got '{mode}'"

        # Set data directory based on mode
        split_dir = "trainval" if mode == "train" else "test"
        self.data_dir = osp.join(base_dir, split_dir)

        # Store configuration
        self.mode = mode
        self.base_size = base_size
        self.pt_label = pt_label
        self.offset = offset
        self.pseudo_label = pseudo_label  # Fixed typo: was 'pesudo_label'
        self.preded_label = preded_label
        self.augment = augment  # Fixed: was 'aug'
        self.turn_num = turn_num
        self.target_mix = target_mix

        # Load image filenames
        self.names: List[str] = []
        images_dir = osp.join(self.data_dir, f"images{file_name}")
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
        preded_label_path = osp.join(
            self.data_dir, f"preded_label/{self.turn_num}", name
        )
        label_path = osp.join(self.data_dir, "masks", name)

        # Load image and mask as single-channel tensors
        img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask_np = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        img = torch.from_numpy(img_np).unsqueeze(0).float()
        mask = torch.from_numpy(mask_np).unsqueeze(0).float()

        # Load pseudo label if enabled
        if self.pseudo_label:
            pseudo_label_np = cv2.imread(pseudo_label_path, cv2.IMREAD_GRAYSCALE)
            pseudo_label = torch.from_numpy(pseudo_label_np).unsqueeze(0).float()
            mask = torch.cat([mask, pseudo_label], dim=0)

        # Load predicted label if enabled
        if self.preded_label:
            preded_label_np = cv2.imread(preded_label_path, cv2.IMREAD_GRAYSCALE)
            preded_label = torch.from_numpy(preded_label_np).unsqueeze(0).float()
            mask = torch.cat([mask, preded_label], dim=0)

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
    

class MDFADataset(Data.Dataset):
    """MDFA dataset for infrared small target detection.

    This dataset loads single-channel infrared images and their corresponding
    masks. Supports pseudo-labels, predicted labels, and point label generation.
    Note: Pseudo labels are resized to match mask dimensions if needed.

    Attributes:
        data_dir (str): Path to the dataset directory (trainval or test).
        mode (str): Dataset mode, either 'train' or 'test'.
        base_size (int): Base size for image resizing.
        pt_label (bool): Whether to generate point labels from masks.
        offset (int): Offset for point label generation.
        pseudo_label (bool): Whether to load pseudo labels.
        preded_label (bool): Whether to load predicted labels.
        augment (bool): Whether to apply data augmentation.
        turn_num (str): Turn number for pseudo/predicted label subdirectories.
        target_mix (bool): Whether to apply target mixing (currently unused).
        names (List[str]): List of image filenames.
        aug_transformer: Augmentation transformer instance.
    """

    def __init__(
        self,
        base_dir: str = r"W:/DataSets/Infraid_datasets/MDFA",
        mode: str = "train",
        base_size: int = DEFAULT_BASE_SIZE,
        pt_label: bool = False,
        offset: int = DEFAULT_OFFSET,
        pseudo_label: bool = False,
        preded_label: bool = False,
        augment: bool = True,
        turn_num: str = "",
        target_mix: bool = False,
        file_name: str = "",
    ) -> None:
        """Initialize the MDFA dataset.

        Args:
            base_dir: Root directory of the dataset.
            mode: Dataset mode ('train' or 'test').
            base_size: Base size for image resizing.
            pt_label: Whether to generate point labels from masks.
            offset: Offset for point label generation.
            pseudo_label: Whether to load pseudo labels.
            preded_label: Whether to load predicted labels.
            augment: Whether to apply data augmentation.
            turn_num: Turn number for pseudo/predicted label subdirectories.
            target_mix: Whether to apply target mixing (currently unused).
            file_name: Subdirectory suffix for images folder.
        """
        assert mode in ["train", "test"], f"Mode must be 'train' or 'test', got '{mode}'"

        # Set data directory based on mode
        split_dir = "trainval" if mode == "train" else "test"
        self.data_dir = osp.join(base_dir, split_dir)

        # Store configuration
        self.mode = mode
        self.base_size = base_size
        self.pt_label = pt_label
        self.offset = offset
        self.pseudo_label = pseudo_label  # Fixed typo: was 'pesudo_label'
        self.preded_label = preded_label
        self.augment = augment  # Fixed: was 'aug'
        self.turn_num = turn_num
        self.target_mix = target_mix

        # Load image filenames
        self.names: List[str] = []
        images_dir = osp.join(self.data_dir, f"images{file_name}")
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
            self.data_dir, f"pixel_pseudo_label{self.turn_num}", name
        )
        preded_label_path = osp.join(
            self.data_dir, f"preded_label/{self.turn_num}", name
        )
        label_path = osp.join(self.data_dir, "masks", name)

        # Load image and mask as single-channel tensors
        img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask_np = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        img = torch.from_numpy(img_np).unsqueeze(0).float()
        mask = torch.from_numpy(mask_np).unsqueeze(0).float()

        # Load and resize pseudo label if enabled
        if self.pseudo_label:
            pseudo_label_np = cv2.imread(pseudo_label_path, cv2.IMREAD_GRAYSCALE)
            pseudo_label = torch.from_numpy(pseudo_label_np).unsqueeze(0).float()
            # Resize pseudo label to match mask dimensions
            pseudo_label = transforms.functional.resize(
                pseudo_label, (mask.shape[-2], mask.shape[-1])
            )
            mask = torch.cat([mask, pseudo_label], dim=0)

        # Load predicted label if enabled
        if self.preded_label:
            preded_label_np = cv2.imread(preded_label_path, cv2.IMREAD_GRAYSCALE)
            preded_label = torch.from_numpy(preded_label_np).unsqueeze(0).float()
            mask = torch.cat([mask, preded_label], dim=0)

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


class SIRSTDataset(Data.Dataset):
    """SIRST dataset for infrared small target detection.

    This dataset loads single-channel infrared images and their corresponding
    masks. Supports pseudo-labels, predicted labels, and point label generation.
    Note: Pseudo labels are resized to match mask dimensions if needed.

    Attributes:
        data_dir (str): Path to the dataset directory (trainval or test).
        mode (str): Dataset mode, either 'train' or 'test'.
        base_size (int): Base size for image resizing.
        pt_label (bool): Whether to generate point labels from masks.
        offset (int): Offset for point label generation.
        pseudo_label (bool): Whether to load pseudo labels.
        preded_label (bool): Whether to load predicted labels.
        augment (bool): Whether to apply data augmentation.
        turn_num (str): Turn number for pseudo/predicted label subdirectories.
        target_mix (bool): Whether to apply target mixing (currently unused).
        names (List[str]): List of image filenames.
        aug_transformer: Augmentation transformer instance.
    """

    def __init__(
        self,
        base_dir: str = r"W:/DataSets/Infraid_datasets/SIRST",
        mode: str = "train",
        base_size: int = DEFAULT_BASE_SIZE,
        pt_label: bool = False,
        offset: int = DEFAULT_OFFSET,
        pseudo_label: bool = False,
        preded_label: bool = False,
        augment: bool = True,
        turn_num: str = "",
        target_mix: bool = False,
        file_name: str = "",
    ) -> None:
        """Initialize the SIRST dataset.

        Args:
            base_dir: Root directory of the dataset.
            mode: Dataset mode ('train' or 'test').
            base_size: Base size for image resizing.
            pt_label: Whether to generate point labels from masks.
            offset: Offset for point label generation.
            pseudo_label: Whether to load pseudo labels.
            preded_label: Whether to load predicted labels.
            augment: Whether to apply data augmentation.
            turn_num: Turn number for pseudo/predicted label subdirectories.
            target_mix: Whether to apply target mixing (currently unused).
            file_name: Subdirectory suffix for images folder.
        """
        assert mode in ["train", "test"], f"Mode must be 'train' or 'test', got '{mode}'"

        # Set data directory based on mode
        split_dir = "trainval" if mode == "train" else "test"
        self.data_dir = osp.join(base_dir, split_dir)

        # Store configuration
        self.mode = mode
        self.base_size = base_size
        self.pt_label = pt_label
        self.offset = offset
        self.pseudo_label = pseudo_label  # Fixed typo: was 'pesudo_label'
        self.preded_label = preded_label
        self.augment = augment  # Fixed: was 'aug'
        self.turn_num = turn_num
        self.target_mix = target_mix

        # Load image filenames
        self.names: List[str] = []
        images_dir = osp.join(self.data_dir, f"images{file_name}")
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
        preded_label_path = osp.join(
            self.data_dir, f"preded_label/{self.turn_num}", name
        )
        label_path = osp.join(self.data_dir, "masks", name)

        # Load image and mask as single-channel tensors
        img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask_np = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        img = torch.from_numpy(img_np).unsqueeze(0).float()
        mask = torch.from_numpy(mask_np).unsqueeze(0).float()

        # Load and resize pseudo label if enabled
        if self.pseudo_label:
            pseudo_label_np = cv2.imread(pseudo_label_path, cv2.IMREAD_GRAYSCALE)
            pseudo_label = torch.from_numpy(pseudo_label_np).unsqueeze(0).float()
            # Resize pseudo label to match mask dimensions
            pseudo_label = transforms.functional.resize(
                pseudo_label, (mask.shape[-2], mask.shape[-1])
            )
            mask = torch.cat([mask, pseudo_label], dim=0)

        # Load predicted label if enabled
        if self.preded_label:
            preded_label_np = cv2.imread(preded_label_path, cv2.IMREAD_GRAYSCALE)
            preded_label = torch.from_numpy(preded_label_np).unsqueeze(0).float()
            mask = torch.cat([mask, preded_label], dim=0)

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


def __organize_dataset(training_dir: str, target_base_path: str) -> None:
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


def __generate_and_save_point_labels(
    dataset_class: NUDTDataset,
    base_dir: str,
    output_dir: str,
    base_size: int = 256,
):
    """
    生成 offset=0 的点标签，并保存为与原图同名的 PNG 文件。
    
    Args:
        base_dir (str): NUDT 数据集根目录，如 "W:/DataSets/Infraid_datasets/NUDT-SIRST"
        output_dir (str): 点标签保存路径
        base_size (int): 输入尺寸
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 初始化 dataset，启用 pt_label 且 offset=0
    dataset = dataset_class(
        base_dir=base_dir,
        mode="train",
        base_size=base_size,
        pt_label=True,      # 启用点标签生成
        offset=0,           # 0偏移
        pseudo_label=False,
        preded_label=False,
        augment=False,      # 不要数据增强，否则点位置会变
    )

    print(f"共 {len(dataset)} 张图像待处理...")

    for i in range(len(dataset)):
        name = dataset.names[i]
        img, mask_with_pt = dataset[i]  # mask_with_pt[0] 是点标签

        # 提取点标签（第一个通道）
        pt_label = mask_with_pt[0]  # shape: [H, W], 值为 0 或 1（因为 /255.0 了）

        # 转回 0-255 的 uint8 格式
        pt_label_np = (pt_label * 255).byte().cpu().numpy()

        # 保存路径
        save_path = os.path.join(output_dir, name)
        cv2.imwrite(save_path, pt_label_np)

        if (i + 1) % 50 == 0:
            print(f"已处理 {i + 1}/{len(dataset)}")

    print(f"✅ 所有点标签已保存至: {output_dir}")

def __resize_and_save_dataset(
    dataset_class: NUDTDataset,
    src_base_dir: str,
    dst_base_dir: str,
    base_size: int = 256
):
    """
    将 NUDT 数据集的 trainval 和 test 的 images/masks resize 到 base_size，并保存到新目录。
    
    目录结构示例：
    dst_base_dir/
        ├── trainval/
        │   ├── images/
        │   └── masks/
        └── test/
            ├── images/
            └── masks/
    """
    # for mode in ["train", "test"]:
    for mode in ["train"]:
        print(f"\n🔄 正在处理 {mode} 集...")

        # 创建目标目录
        if mode == "train":
            dst_img_dir = os.path.join(dst_base_dir, "trainval", "images")
            dst_mask_dir = os.path.join(dst_base_dir, "trainval", "masks")
        else:
            dst_img_dir = os.path.join(dst_base_dir, "test", "images")
            dst_mask_dir = os.path.join(dst_base_dir, "test", "masks")
        
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_mask_dir, exist_ok=True)

        # 初始化 dataset，启用 pt_label 且 offset=0
        dataset = dataset_class(base_dir=src_base_dir,
                                mode=mode,
                                base_size=base_size,
                                pt_label=False,      # 启用点标签生成
                                offset=0,           # 0偏移
                                pseudo_label=False,
                                preded_label=False,
                                augment=False,      # 不要数据增强，否则点位置会变
                                )

        print(f"共 {len(dataset)} 张图像待处理...")

        for i in range(len(dataset)):
            name = dataset.names[i]
            img, mask = dataset[i]  # img: [1, H, W], mask: [1, H, W] (值为 0~1)

            mask = (mask.float() > 0.5).float()

            # 转为 numpy uint8 (0-255)
            img_np = (img.squeeze(0).cpu().numpy() * 255).astype('uint8')
            mask_np = (mask.squeeze(0).cpu().numpy() * 255).astype('uint8')

            # 保存
            cv2.imwrite(os.path.join(dst_img_dir, name), img_np)
            cv2.imwrite(os.path.join(dst_mask_dir, name), mask_np)

            if (i + 1) % 50 == 0:
                print(f"  已处理 {i + 1}/{len(dataset)}")

    print(f"\n✅ 全部数据已保存至: {dst_base_dir}")


def __split_dataset_by_index_with_mask_prefix_match(root_dir: str):
    """
    根据 images 文件名，在 masks 中查找前缀匹配的文件（如 Misc_25.png ↔ Misc_25_pixels0.png），
    按索引奇偶划分，并将 mask 重命名为与 image 一致。
    
    要求：
      - images/ 下: Misc_25.png
      - masks/  下: Misc_25_pixels0.png, Misc_25_pixels1.png 等（但应只存在一个匹配项）
    """
    root = Path(root_dir)
    img_dir = root / "images"
    mask_dir = root / "masks"

    if not (img_dir.exists() and mask_dir.exists()):
        raise FileNotFoundError(f"请确保 {root} 下存在 'images' 和 'masks' 文件夹！")

    # 获取所有 image 文件（.png），排序
    image_files = sorted([f for f in img_dir.glob("*.png")], key=lambda x: x.name)
    if not image_files:
        raise ValueError("images 文件夹中没有 .png 文件！")

    # 构建 mask 文件名映射：mask_stem -> full_path
    # 例如： "Misc_25" -> Path("masks/Misc_25_pixels0.png")
    mask_files = list(mask_dir.glob("*.png"))
    mask_stem_to_path = {}
    for mf in mask_files:
        # 去掉所有可能的后缀变体，只保留主干（如 "Misc_25_pixels0" -> 尝试匹配 "Misc_25"）
        # 策略：从后往前尝试去掉 "_pixels..." 等部分
        stem = mf.stem
        # 简单策略：如果包含 "_pixels"，则截断
        if "_pixels" in stem:
            base_stem = stem.split("_pixels")[0]
        else:
            # 否则尝试通用方式：保留主干（可根据实际命名调整）
            base_stem = stem
        # 也可以更通用：假设 image 的 stem 就是 mask stem 的前缀
        mask_stem_to_path[stem] = mf  # 先保留原始 stem 到路径

    # 配对列表：[(img_path, mask_path), ...]
    paired_files = []
    for img_path in image_files:
        img_stem = img_path.stem  # e.g., "Misc_25"
        img_name = img_path.name  # e.g., "Misc_25.png"

        # 在 mask 中查找：是否有 mask 文件的 stem 以 img_stem 开头？
        matched_masks = []
        for mask_stem, mask_path in mask_stem_to_path.items():
            if mask_stem.startswith(img_stem):
                # 进一步确保不是误匹配（比如 Misc_25 不匹配 Misc_250）
                # 要求：mask_stem == img_stem 或 mask_stem == img_stem + "_pixelsX"
                if mask_stem == img_stem or mask_stem.startswith(img_stem + "_"):
                    matched_masks.append(mask_path)

        if len(matched_masks) == 0:
            print(f"⚠️ 警告: 未找到与 {img_name} 匹配的 mask 文件，跳过。")
            continue
        elif len(matched_masks) > 1:
            print(f"⚠️ 警告: 找到多个 mask 匹配 {img_name}，使用第一个: {[m.name for m in matched_masks]}")
        
        mask_path = matched_masks[0]
        paired_files.append((img_path, mask_path))

    if not paired_files:
        raise ValueError("未找到任何 image-mask 配对！")

    print(f"共配对 {len(paired_files)} 对图像与标签。")

    # 创建输出目录
    for split in ["trainval", "test"]:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "masks").mkdir(parents=True, exist_ok=True)

    # 按索引奇偶划分
    for idx, (img_path, mask_path) in enumerate(paired_files):
        target_split = "trainval" if idx % 2 == 0 else "test"
        target_img_dir = root / target_split / "images"
        target_mask_dir = root / target_split / "masks"

        # 复制 image（保持原名）
        shutil.copy2(img_path, target_img_dir / img_path.name)

        # 复制 mask，但**重命名为 image 的文件名**
        new_mask_name = img_path.name  # 关键：让 mask 和 image 同名
        shutil.copy2(mask_path, target_mask_dir / new_mask_name)

    print(f"✅ 划分完成！")
    print(f"   trainval: {len(paired_files[::2])} 对")
    print(f"   test:     {len(paired_files[1::2])} 对")


if __name__ == "__main__":
    # TRAINING_DIR = "W:/DataSets/ISTD/MDvsFA_cGAN-master/data/training"   # 原始文件夹路径
    TARGET_PATH = "W:/DataSets/ISTD/SIRST"        # 目标根路径
    # SIRST_PATH = "W:/DataSets/ISTD/SIRST"
    # base_dir = "W:/DataSets/ISTD/NUDT-SIRST"
    # base_dir = "W:/DataSets/ISTD/IRSTD-1k"
    # pt_label_dir =  base_dir + "/trainval/point_label"

    # __split_dataset_by_index_with_mask_prefix_match(SIRST_PATH)
    # __organize_dataset(TRAINING_DIR, TARGET_PATH)
    # __generate_and_save_point_labels(IRSTD1kDataset, base_dir, pt_label_dir, 512)
    __resize_and_save_dataset(MDFADataset, TARGET_PATH, TARGET_PATH, 256)