import numpy as np
from typing import Dict, Any

def compute_ndwi(image: np.ndarray, green_idx: int, nir_idx: int) -> np.ndarray:
    """
    Compute Normalized Difference Water Index (NDWI).
    
    Args:
        image (np.ndarray): Multi-spectral image array of shape (C, H, W).
        green_idx (int): Channel index for the Green band.
        nir_idx (int): Channel index for the Near Infrared (NIR) band.
        
    Returns:
        np.ndarray: NDWI array of shape (1, H, W).
    """
    green = image[green_idx].astype(np.float32)
    nir = image[nir_idx].astype(np.float32)
    
    # Add a small epsilon to prevent division by zero
    ndwi = (green - nir) / (green + nir + 1e-8)
    return np.expand_dims(ndwi, axis=0)

def compute_ndsi(image: np.ndarray, green_idx: int, swir1_idx: int) -> np.ndarray:
    """
    Compute Normalized Difference Snow Index (NDSI) for glacier differentiation.
    
    Args:
        image (np.ndarray): Multi-spectral image array of shape (C, H, W).
        green_idx (int): Channel index for the Green band.
        swir1_idx (int): Channel index for the SWIR1 band.
        
    Returns:
        np.ndarray: NDSI array of shape (1, H, W).
    """
    green = image[green_idx].astype(np.float32)
    swir1 = image[swir1_idx].astype(np.float32)
    
    ndsi = (green - swir1) / (green + swir1 + 1e-8)
    return np.expand_dims(ndsi, axis=0)

def append_indices(image: np.ndarray, data_config: Dict[str, Any]) -> np.ndarray:
    """
    Compute and append spectral indices as additional channels.
    
    Args:
        image (np.ndarray): Original image array of shape (C, H, W).
        data_config (Dict): Configuration dictionary containing index settings.
        
    Returns:
        np.ndarray: Augmented image array of shape (C+k, H, W).
    """
    indices_config = data_config.get("indices", {})
    channels = [image]
    
    if data_config.get("compute_ndwi", False):
        ndwi = compute_ndwi(
            image, 
            green_idx=indices_config.get("green_idx", 1),
            nir_idx=indices_config.get("nir_idx", 3)
        )
        channels.append(ndwi)
        
    if data_config.get("compute_ndsi", False):
        ndsi = compute_ndsi(
            image,
            green_idx=indices_config.get("green_idx", 1),
            swir1_idx=indices_config.get("swir1_idx", 4)
        )
        channels.append(ndsi)
        
    if len(channels) > 1:
        return np.concatenate(channels, axis=0)
    
    return image
