"""Utility functions and transforms for IRSTD datasets.

This module provides data augmentation transforms and utility functions
for processing infrared small target detection datasets.
"""

import random
from typing import Tuple

import numpy as np
import scipy.ndimage
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


class Rotate4DTransform:
    """Random rotation transform with 4 discrete angles (0, 90, 180, 270 degrees)."""

    def __init__(self) -> None:
        self.angles = [0, 90, 180, 270]

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random rotation to image.

        Args:
            img: Input tensor of shape (C, H, W).

        Returns:
            Rotated tensor of same shape.
        """
        random_idx = torch.randint(0, 1000, (1,))
        angle = self.angles[random_idx % 4]
        return self._rotate(img, angle)

    def _rotate(self, img: torch.Tensor, angle: int) -> torch.Tensor:
        """Rotate image by specified angle.

        Args:
            img: Input tensor of shape (C, H, W).
            angle: Rotation angle in degrees.

        Returns:
            Rotated tensor.
        """
        if angle == 90:
            img = torch.transpose(img, -1, -2)
            img = torch.flip(img, dims=(-1,))
        elif angle == 180:
            img = torch.flip(img, dims=(-1, -2))
        elif angle == 270:
            img = torch.transpose(img, -1, -2)
            img = torch.flip(img, dims=(-2,))
        return img


class Augmentation:
    """Data augmentation with flips and transpose."""

    def __call__(
        self, input_data: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random horizontal/vertical flips and transpose.

        Args:
            input_data: Input array of shape (H, W, C) or (H, W).
            target: Target array of same spatial dimensions.

        Returns:
            Augmented input and target arrays.
        """
        if random.random() < 0.5:
            input_data = input_data[::-1, :]
            target = target[::-1, :]
        if random.random() < 0.5:
            input_data = input_data[:, ::-1]
            target = target[:, ::-1]
        if random.random() < 0.5:
            input_data = input_data.transpose(1, 0)
            target = target.transpose(1, 0)
        return input_data.copy(), target.copy()


class RandomResize:
    """Random resize transform."""

    def __init__(self, min_size: int, max_size: int) -> None:
        """Initialize random resize.

        Args:
            min_size: Minimum resize size.
            max_size: Maximum resize size.
        """
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random resize.

        Args:
            img: Input tensor.

        Returns:
            Resized tensor.
        """
        target_size = random.randint(self.min_size, self.max_size)
        resize_transform = transforms.Resize((target_size, target_size))
        return resize_transform(img)


class AugmentTransform:
    """Combined augmentation transform with crop, affine, and flip."""

    def __init__(
        self,
        base_size: int = 256,
        mode: str = "train",
        crop_resize_scale: Tuple[float, float] = (0.8, 1.0),
        affine_degrees: Tuple[float, float] = (-180, 180),
        affine_translate: Tuple[float, float] = (0.3, 0.3),
    ) -> None:
        """Initialize augmentation transform.

        Args:
            base_size: Target size for resizing.
            mode: Either 'train' or 'test'.
            crop_resize_scale: Scale range for random resized crop.
            affine_degrees: Angle range for random affine.
            affine_translate: Translation range for random affine.
        """
        self.base_size = base_size
        self.mode = mode
        self.crop_resize_scale = crop_resize_scale
        self.crop_ratio = (1.0, 1.0)
        self.affine_degrees = affine_degrees
        self.affine_translate = affine_translate
        self.affine_scale = (1.0, 1.0)
        self.affine_shear = (0, 0)

    def __call__(
        self, img: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentation to image and mask.

        Args:
            img: Image tensor of shape (C1, H, W).
            mask: Mask tensor of shape (C2, H, W).

        Returns:
            Augmented image and mask tensors.
        """
        if self.mode == "train":
            # RandomResizedCrop with consistent parameters
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img, scale=self.crop_resize_scale, ratio=self.crop_ratio
            )
            transformed_image = F.resized_crop(
                img, i, j, h, w, self.base_size,
                interpolation=InterpolationMode.BILINEAR, antialias=True
            )
            transformed_mask = F.resized_crop(
                mask, i, j, h, w, self.base_size,
                interpolation=InterpolationMode.NEAREST
            )

            # RandomAffine with consistent parameters
            center = (self.base_size // 2, self.base_size // 2)
            angle, translations, scale_factor, shear_values = (
                transforms.RandomAffine.get_params(
                    degrees=self.affine_degrees,
                    translate=self.affine_translate,
                    scale_ranges=self.affine_scale,
                    shears=self.affine_shear,
                    img_size=(self.base_size, self.base_size)
                )
            )
            transformed_image = F.affine(
                transformed_image,
                angle=angle,
                translate=translations,
                scale=scale_factor,
                shear=shear_values,
                interpolation=InterpolationMode.BILINEAR,
                center=center
            )
            transformed_mask = F.affine(
                transformed_mask,
                angle=angle,
                translate=translations,
                scale=scale_factor,
                shear=shear_values,
                interpolation=InterpolationMode.NEAREST,
                center=center
            )

            # Random horizontal flip
            if torch.rand(1).item() < 0.5:
                transformed_image = F.hflip(transformed_image)
                transformed_mask = F.hflip(transformed_mask)

        elif self.mode == "test":
            transformed_image = F.resize(
                img, [self.base_size, self.base_size], antialias=True
            )
            transformed_mask = F.resize(
                mask, [self.base_size, self.base_size],
                interpolation=InterpolationMode.BILINEAR, antialias=True
            )
        else:
            raise ValueError(f"Mode must be 'train' or 'test', got '{self.mode}'")

        return transformed_image, transformed_mask


def mask2point_n(mask: torch.Tensor, offset: int = 3) -> torch.Tensor:
    """Convert mask to point labels (simplified version without image guidance).

    Args:
        mask: Binary mask tensor of shape (1, H, W) or (H, W).
        offset: Offset distance for point perturbation.

    Returns:
        Point label tensor of same shape as mask.
    """
    base_size = mask.shape[-1]
    mask_array = np.array(mask.cpu())

    # Connected component analysis
    labels, num_features = scipy.ndimage.label(mask_array > 0.1)
    pts_label = torch.zeros_like(mask, dtype=torch.float32)

    for label_id in range(1, num_features + 1):
        target_mask = labels == label_id
        coords = np.argwhere(target_mask)

        if len(coords) == 0:
            continue

        masked_coords = coords

        # Calculate centroid
        centroid = np.mean(coords, axis=0).astype(np.int64)
        if not target_mask[centroid[0], centroid[1]]:
            coord_diff = coords - centroid
            min_dist_idx = np.argmin(np.sum(coord_diff**2, axis=1))
            centroid = coords[min_dist_idx]

        # Find nearest point to centroid
        dists = np.linalg.norm(masked_coords - centroid, axis=1)
        nearest_point = masked_coords[np.argmin(dists)]
        if nearest_point.ndim > 1:
            nearest_point = nearest_point[0]

        point_y_x = nearest_point.copy()

        # Apply offset in random direction
        if offset > 0:
            theta = np.random.uniform(0, 2 * np.pi)
            ideal_y = point_y_x[0] + offset * np.sin(theta)
            ideal_x = point_y_x[1] + offset * np.cos(theta)
            ideal_point = np.array([ideal_y, ideal_x])
            ideal_point = np.clip(ideal_point, 0, base_size - 1)

            dists = np.linalg.norm(masked_coords - ideal_point, axis=1)
            nearest_point_candidate = masked_coords[np.argmin(dists)]

            if not np.array_equal(nearest_point_candidate, point_y_x):
                point_y_x = nearest_point_candidate

        pts_label[point_y_x[0], point_y_x[1]] = 1.0

    return pts_label


def mask2point(
    mask: torch.Tensor, img: torch.Tensor, offset: int = 3
) -> torch.Tensor:
    """Convert mask to point labels using image intensity guidance.

    Finds the brightest point in each connected component and optionally
    applies a small offset.

    Args:
        mask: Binary mask tensor of shape (1, H, W).
        img: Image tensor of shape (1, H, W) for intensity guidance.
        offset: Maximum offset distance for point perturbation.

    Returns:
        Point label tensor of shape (1, H, W).
    """
    base_size = mask.shape[-1]
    mask_array = np.array(mask[0].cpu())
    img_array = np.array(img[0].cpu())

    # Connected component analysis with high threshold
    labels, num_features = scipy.ndimage.label(mask_array > 0.9)
    pts_label = torch.zeros_like(mask, dtype=torch.float32)

    for label_id in range(1, num_features + 1):
        target_mask = labels == label_id
        coords = np.argwhere(target_mask)

        if len(coords) == 0:
            continue

        # Get image values within the mask region
        masked_img_vals = img_array[target_mask]
        masked_coords = coords

        # Find bright pixels (top 50%)
        brightness = masked_img_vals.flatten()
        threshold = np.percentile(brightness, 50)
        bright_coords = masked_coords[brightness >= threshold]

        if len(bright_coords) == 0:
            continue

        # Calculate centroid
        centroid = np.mean(coords, axis=0).astype(np.int64)
        if not target_mask[centroid[0], centroid[1]]:
            coord_diff = coords - centroid
            min_dist_idx = np.argmin(np.sum(coord_diff**2, axis=1))
            centroid = coords[min_dist_idx]

        # Find nearest bright point to centroid
        dists = np.linalg.norm(bright_coords - centroid, axis=1)
        nearest_point = bright_coords[np.argmin(dists)]
        if nearest_point.ndim > 1:
            nearest_point = nearest_point[0]

        point_y_x = nearest_point.copy()

        # Try to find alternative point with offset
        attempt_count = 0
        while attempt_count < 10 and offset > 0:
            offset_y_x = np.random.uniform(-offset, offset, (2,)).astype(np.int64)
            new_point = point_y_x + offset_y_x
            new_point = np.clip(new_point, 0, np.array([base_size - 1, base_size - 1]))

            dists = np.linalg.norm(bright_coords - new_point, axis=1)
            nearest_point_candidate = bright_coords[np.argmin(dists)]

            if np.array_equal(nearest_point_candidate, point_y_x):
                offset -= 1
                attempt_count += 1
                continue
            else:
                point_y_x = nearest_point_candidate
                break

        # Set final point label
        pts_label[0, point_y_x[0], point_y_x[1]] = 1.0

    return pts_label
