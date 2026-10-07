import argparse
import yaml
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from tqdm import tqdm

from src.data.indices import append_indices
from src.models.unet import build_model
from src.utils.geo_utils import get_gaussian_window, blend_tiles_gaussian, raster_to_vector

def predict_scene(config_path: str, checkpoint_path: str, input_raster: str, output_dir: str):
    """
    Run full-scene sliding window inference on a raw multi-spectral GeoTIFF.
    Includes Gaussian smoothing of overlapping tiles and automatic shapefile vectorization.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on: {device}")
    
    # Build and load Model
    model = build_model(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    tile_size = config['data']['tile_size']
    overlap = config['data']['overlap']
    stride = tile_size - overlap
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_raster_path = out_dir / f"pred_{Path(input_raster).name}"
    
    # Pre-compute gaussian window for blend weight
    gaussian_weights = get_gaussian_window(tile_size, sigma=0.5)
    
    with rasterio.open(input_raster) as src:
        height, width = src.shape
        meta = src.meta.copy()
        
        # Accumulators for the entire scene
        global_prob = np.zeros((height, width), dtype=np.float32)
        global_weight = np.zeros((height, width), dtype=np.float32)
        
        # Calculate sliding windows
        windows = []
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                win_x, win_y = x, y
                
                # Shift start if window exceeds raster boundary
                if win_x + tile_size > width:
                    win_x = max(0, width - tile_size)
                if win_y + tile_size > height:
                    win_y = max(0, height - tile_size)
                    
                windows.append((Window(win_x, win_y, tile_size, tile_size), win_x, win_y))
        
        windows = list(dict.fromkeys(windows)) # Remove duplicates near boundaries
        
        for win, x_off, y_off in tqdm(windows, desc="Sliding Window Inference"):
            image_patch = src.read(window=win)
            
            # Pad if raster is smaller than a single tile (edge case)
            _, h, w = image_patch.shape
            if h < tile_size or w < tile_size:
                pad_h = tile_size - h
                pad_w = tile_size - w
                image_patch = np.pad(image_patch, ((0,0), (0,pad_h), (0,pad_w)), mode='constant')
                
            # Pre-processing pipeline (Indices + Normalization)
            image_patch = append_indices(image_patch, config.get("data", {}))
            if image_patch.dtype == np.uint8 or image_patch.max() > 1.0:
                image_patch = image_patch.astype(np.float32) / 255.0
            else:
                image_patch = image_patch.astype(np.float32) / 10000.0
            image_patch = np.clip(image_patch, 0.0, 1.0)
            
            # Predict (1, C, H, W)
            tensor = torch.from_numpy(image_patch).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy() # (H, W)
                
            # Handle potential cropping if we padded
            probs_cropped = probs[:h, :w]
            gaussian_cropped = gaussian_weights[:h, :w]
            
            # Accumulate predictions with gaussian blend
            blend_tiles_gaussian(probs_cropped, global_prob, global_weight, x_off, y_off, gaussian_cropped)
            
    # Normalize accumulated probabilities by accumulated gaussian weights
    final_prob = np.divide(global_prob, global_weight, out=np.zeros_like(global_prob), where=global_weight!=0)
    
    # Save probabilistic raster
    meta.update({
        "count": 1,
        "dtype": "float32",
        "compress": "lzw"
    })
    
    with rasterio.open(out_raster_path, 'w', **meta) as dst:
        dst.write(final_prob, 1)
        
    print(f"Saved predicted probabilistic raster to {out_raster_path}")
    
    # Vectorize strictly predicted pixels to GeoJSON polygons
    out_vector_path = out_dir / f"pred_{Path(input_raster).stem}.geojson"
    print(f"Vectorizing binary mask to {out_vector_path}...")
    raster_to_vector(out_raster_path, out_vector_path, threshold=0.5, min_area=50.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full-Scene Glacial Lake Segmentation Inference")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to best_model.pth")
    parser.add_argument('--input', type=str, required=True, help="Path to unseen full-scene GeoTIFF")
    parser.add_argument('--output_dir', type=str, default='predictions', help="Output directory")
    args = parser.parse_args()
    
    predict_scene(args.config, args.checkpoint, args.input, args.output_dir)
