import os
from pathlib import Path
from typing import Union, Optional
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import logging

logger = logging.getLogger(__name__)

def raster_to_vector(
    raster_path: Union[str, Path], 
    output_path: Union[str, Path],
    threshold: float = 0.5,
    min_area: Optional[float] = None
) -> None:
    """
    Converts a predicted binary raster mask into a fully georeferenced vector 
    file (GeoJSON/Shapefile).
    
    Args:
        raster_path (Union[str, Path]): Path to the input predicted GeoTIFF raster.
        output_path (Union[str, Path]): Path where the output vector file will be saved.
                                        Suffix determines format (.geojson or .shp).
        threshold (float): Threshold to binarize probability maps if needed.
        min_area (Optional[float]): Minimum area (in CRS units) to keep a polygon.
                                    Crucial for removing small pixel-noise artifacts.
    """
    raster_path = Path(raster_path)
    output_path = Path(output_path)
    
    with rasterio.open(raster_path) as src:
        image = src.read(1)
        transform = src.transform
        crs = src.crs
        
    # Binarize if it's a probability map
    if image.dtype in [np.float32, np.float64]:
        mask = (image > threshold).astype(np.uint8)
    else:
        mask = image.astype(np.uint8)
        
    # Extract shapes using rasterio
    # mask=mask ensures we only vectorize the foreground (1), not the background (0)
    polygon_generator = shapes(mask, mask=mask, transform=transform)
    
    geometries = []
    values = []
    
    for geom, val in polygon_generator:
        poly = shape(geom)
        
        # Filter out micro-artifacts by area
        if min_area is not None:
            if poly.area < min_area:
                continue
                
        geometries.append(poly)
        values.append(int(val))
        
    if not geometries:
        logger.warning(f"No polygons found in {raster_path}. Creating empty vector file.")
        gdf = gpd.GeoDataFrame(columns=['geometry', 'class_id'], geometry='geometry', crs=crs)
    else:
        gdf = gpd.GeoDataFrame({'class_id': values, 'geometry': geometries}, crs=crs)
        
    # Save to disk
    driver = "GeoJSON" if output_path.suffix.lower() == ".geojson" else "ESRI Shapefile"
    
    # Ensure parent dir exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    gdf.to_file(output_path, driver=driver)
    logger.info(f"Successfully vectorized {len(geometries)} polygons to {output_path}")

def blend_tiles_gaussian(
    tile: np.ndarray,
    global_raster: np.ndarray,
    global_weight: np.ndarray,
    x_off: int,
    y_off: int,
    gaussian_weights: np.ndarray
):
    """
    Utility for smooth overlapping tile inference. 
    Adds a predicted tile into a global memory buffer using a gaussian weighting scheme
    to elegantly eliminate sharp edge artifacts at tile boundaries.
    
    Args:
        tile (np.ndarray): Predicted tile probabilities (H, W).
        global_raster (np.ndarray): Global probability accumulator array.
        global_weight (np.ndarray): Global weight accumulator array.
        x_off (int): X offset in the global raster.
        y_off (int): Y offset in the global raster.
        gaussian_weights (np.ndarray): Precomputed 2D gaussian window (H, W).
    """
    h, w = tile.shape
    global_raster[y_off:y_off+h, x_off:x_off+w] += (tile * gaussian_weights)
    global_weight[y_off:y_off+h, x_off:x_off+w] += gaussian_weights

def get_gaussian_window(size: int, sigma: float = 0.5) -> np.ndarray:
    """
    Generates a 2D Gaussian window to be used for smoothing sliding-window inference overlap.
    """
    center = size // 2
    x, y = np.mgrid[0:size, 0:size]
    
    # Scale coordinates to [-1, 1] for stable gaussian
    x = (x - center) / center
    y = (y - center) / center
    
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g
