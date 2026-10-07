import os
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import logging

from .indices import append_indices

logger = logging.getLogger(__name__)

class GlacierDataset(Dataset):
    """
    PyTorch Dataset for Multi-Spectral GeoTIFFs (Glacial Lake/Ice Segmentation).
    Supports arbitrary N-channel inputs, on-the-fly NDWI/NDSI computation,
    handles missing/nodata values, and albumentations augmentations.
    """
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        config: Dict[str, Any],
        transform: Optional[Callable] = None,
        is_train: bool = True
    ):
        """
        Initialize the Dataset.

        Args:
            image_dir (str): Path to the directory containing image patch GeoTIFFs.
            mask_dir (str): Path to the directory containing mask patch GeoTIFFs.
            config (Dict): Configuration dictionary (e.g., from Hydra/PyYAML).
            transform (Callable, optional): Albumentations compose object.
            is_train (bool): Flag to indicate training mode.
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.config = config
        self.transform = transform
        self.is_train = is_train
        
        # Grab only valid files
        self.image_paths = sorted([
            f for f in self.image_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in ['.tif', '.tiff', '.png']
        ])
        
        if not self.image_paths:
            logger.warning(f"No GeoTIFF patches found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / img_path.name
        
        # 1. Read Multi-Spectral Image (C, H, W)
        with rasterio.open(img_path) as src:
            image = src.read()
            # Handle nodata to 0 or another stable representation
            nodata = src.nodata
            if nodata is not None:
                image[image == nodata] = 0
                
        # 2. Append Spectral Indices (C+k, H, W)
        data_config = self.config.get("data", {})
        image = append_indices(image, data_config)
        
        # 3. Normalize to [0, 1] range. 
        # If it's standard 8-bit PNG, divide by 255.
        if image.dtype == np.uint8 or image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        else:
            # Sentinel-2/Landsat Surface Reflectance L2A products are typically scaled by 10000.
            image = image.astype(np.float32) / 10000.0
        # Clip max bounds
        image = np.clip(image, 0.0, 1.0)
        
        # 4. Read Mask (1, H, W)
        if mask_path.exists():
            with rasterio.open(mask_path) as src:
                mask = src.read(1) # Read the first binary band
        else:
            # If no mask available (e.g., predicting on unannotated data)
            mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8)
            
        mask = mask.astype(np.float32)
        # Ensure mask is strictly 0 and 1 (binary cross entropy fails with NaNs if max is 255)
        if mask.max() > 1.0:
            mask = mask / 255.0
        # Binarize just in case there are intermediate values due to resizing
        mask = (mask > 0.5).astype(np.float32)
        
        # 5. Apply Albumentations Transforms
        if self.transform:
            # Albumentations expects dimensions in (H, W, C) for multi-channel images
            image_hwc = np.transpose(image, (1, 2, 0))
            
            augmented = self.transform(image=image_hwc, mask=mask)
            # ToTensorV2 converts the image back to (C, H, W) PyTorch standard
            image_t = augmented['image'] 
            # Masks return as (H, W) usually, so we add channel dim back for PyTorch BCE/Dice Loss
            mask_t = augmented['mask'].unsqueeze(0)
        else:
            image_t = torch.from_numpy(image)
            mask_t = torch.from_numpy(mask).unsqueeze(0)
            
        return {
            "image": image_t,
            "mask": mask_t,
            "image_id": img_path.name
        }
