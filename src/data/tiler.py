"""
Raster tiler module for large-scale GeoTIFFs.
"""

import os
from pathlib import Path
from typing import Union, List, Tuple, Optional
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeoTiler:
    """
    A robust sliding window tiler that takes raw full-scene GeoTIFF rasters and binary
    vector/raster ground truth masks and extracts uniform patches (e.g., 512x512 pixels).
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        tile_size: int = 512,
        overlap: int = 128,
        nodata_threshold: float = 0.5,
        mask_threshold: float = 0.01
    ):
        """
        Initialize the GeoTiler.

        Args:
            output_dir (Union[str, Path]): Directory to save the extracted tiles.
            tile_size (int): Size of the square tile (width and height).
            overlap (int): Number of pixels to overlap between adjacent tiles.
            nodata_threshold (float): Maximum allowed ratio of nodata pixels in a tile.
                If nodata > threshold, the tile is discarded.
            mask_threshold (float): Minimum ratio of positive pixels in the mask.
                Used to filter out background-only tiles if desired (set to 0 to keep all).
        """
        self.output_dir = Path(output_dir)
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = self.tile_size - self.overlap
        self.nodata_threshold = nodata_threshold
        self.mask_threshold = mask_threshold

        self.img_out_dir = self.output_dir / "images"
        self.mask_out_dir = self.output_dir / "masks"
        
        self.img_out_dir.mkdir(parents=True, exist_ok=True)
        self.mask_out_dir.mkdir(parents=True, exist_ok=True)

    def _get_windows(self, width: int, height: int) -> List[Tuple[Window, int, int]]:
        """
        Calculate overlapping windows for a given raster dimension.

        Args:
            width (int): Raster width in pixels.
            height (int): Raster height in pixels.

        Returns:
            List[Tuple[Window, int, int]]: List of rasterio Window objects and their (x, y) origin.
        """
        windows = []
        for y in range(0, height, self.stride):
            for x in range(0, width, self.stride):
                # Adjust for edge cases where window exceeds raster bounds
                win_x = x
                win_y = y
                
                if win_x + self.tile_size > width:
                    win_x = max(0, width - self.tile_size)
                if win_y + self.tile_size > height:
                    win_y = max(0, height - self.tile_size)
                    
                windows.append((Window(win_x, win_y, self.tile_size, self.tile_size), win_x, win_y))
                
                if x + self.tile_size >= width:
                    break
            if y + self.tile_size >= height:
                break
                
        # Remove duplicates from edge adjustment
        unique_windows = list(dict.fromkeys(windows))
        return unique_windows

    def _save_tile(self, data: np.ndarray, meta: dict, path: Path):
        """Save a numpy array as a GeoTIFF."""
        with rasterio.open(path, 'w', **meta) as dst:
            dst.write(data)

    def process_scene(
        self,
        image_path: Union[str, Path],
        mask_path: Union[str, Path],
        scene_id: str
    ) -> int:
        """
        Process a single large satellite scene and its corresponding mask.

        Args:
            image_path (Union[str, Path]): Path to the multi-spectral GeoTIFF.
            mask_path (Union[str, Path]): Path to the corresponding ground truth mask GeoTIFF.
            scene_id (str): Identifier for the scene, used for output filenames.

        Returns:
            int: Number of valid tiles extracted.
        """
        valid_tiles_count = 0
        
        with rasterio.open(image_path) as src_img, rasterio.open(mask_path) as src_mask:
            if src_img.crs != src_mask.crs:
                logger.warning(f"CRS mismatch: image ({src_img.crs}) vs mask ({src_mask.crs}). Proceeding with image CRS.")
            
            if src_img.shape != src_mask.shape:
                raise ValueError(f"Shape mismatch: Image {src_img.shape} vs Mask {src_mask.shape}")
                
            height, width = src_img.shape
            windows = self._get_windows(width, height)
            
            nodata_val = src_img.nodata if src_img.nodata is not None else 0
            
            # Prepare metadata for tiles
            img_meta = src_img.meta.copy()
            img_meta.update({
                "driver": "GTiff",
                "height": self.tile_size,
                "width": self.tile_size,
                "compress": "lzw"
            })
            
            mask_meta = src_mask.meta.copy()
            mask_meta.update({
                "driver": "GTiff",
                "height": self.tile_size,
                "width": self.tile_size,
                "compress": "lzw"
            })

            logger.info(f"Extracting tiles from scene {scene_id} ({width}x{height})")
            
            for win, x_off, y_off in tqdm(windows, desc=f"Tiling {scene_id}"):
                img_data = src_img.read(window=win)
                mask_data = src_mask.read(window=win)
                
                # Handle potential out of bounds reading with padding (if raster is smaller than tile_size)
                _, h, w = img_data.shape
                if h < self.tile_size or w < self.tile_size:
                    pad_h = self.tile_size - h
                    pad_w = self.tile_size - w
                    img_data = np.pad(img_data, ((0,0), (0,pad_h), (0,pad_w)), mode='constant', constant_values=nodata_val)
                    mask_data = np.pad(mask_data, ((0,0), (0,pad_h), (0,pad_w)), mode='constant', constant_values=0)
                
                # Check nodata threshold (assuming 0 is nodata for imagery if not specified)
                nodata_mask = (img_data[0] == nodata_val)
                nodata_ratio = np.sum(nodata_mask) / (self.tile_size * self.tile_size)
                
                if nodata_ratio > self.nodata_threshold:
                    continue
                    
                # Check mask threshold to discard patches with no glacial lakes (optional config)
                pos_ratio = np.sum(mask_data > 0) / (self.tile_size * self.tile_size)
                if pos_ratio < self.mask_threshold and self.mask_threshold > 0:
                    continue
                
                # Calculate window transform and update spatial reference metadata for the specific tile
                win_transform = src_img.window_transform(win)
                img_meta.update({"transform": win_transform})
                mask_meta.update({"transform": win_transform})
                
                # Save tiles
                tile_name = f"{scene_id}_x{x_off}_y{y_off}.tif"
                img_tile_path = self.img_out_dir / tile_name
                mask_tile_path = self.mask_out_dir / tile_name
                
                self._save_tile(img_data, img_meta, img_tile_path)
                self._save_tile(mask_data, mask_meta, mask_tile_path)
                
                valid_tiles_count += 1
                
        logger.info(f"Finished scene {scene_id}. Extracted {valid_tiles_count} valid tiles.")
        return valid_tiles_count
