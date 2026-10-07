# Glacial Lake Semantic Segmentation

A production-grade PyTorch-based multi-spectral semantic segmentation pipeline designed to automatically delineate glacial lake and ice boundaries using Sentinel-2 L2A / Landsat satellite imagery.

## Core Features

- **Multi-Spectral Native**: Supports $N$-channel inputs (e.g., 6 bands) natively without artificially compressing reflectance data into 3-channel RGB imagery.
- **Dynamic Indices**: On-the-fly, numerically stable computation of spectral indices like Normalized Difference Water Index (NDWI) and Snow Index (NDSI).
- **Extreme Class Imbalance Handling**: Utilizes a highly robust hybrid `DiceFocalLoss` specifically tuned to heavily penalize false-negatives on sparse target segmentations.
- **Georeferenced Inference**: Full-scene sliding window inference utilizing a 2D Gaussian memory-buffer to eliminate edge artifacts. Binarized outputs are automatically converted into georeferenced GeoJSON/Shapefiles via `geopandas`.

## Installation

We highly recommend using a fresh virtual environment.
```bash
pip install -r requirements.txt
```

## Pipeline Execution

### 1. Data Preparation (Tiling)
Extract overlapping 512x512 patches from massive raw scenes into a dataset directory:
```python
from src.data.tiler import GeoTiler

tiler = GeoTiler(output_dir="data/processed/tiles", tile_size=512, overlap=128)
tiler.process_scene("raw/scene_1.tif", "raw/mask_1.tif", "scene_1")
```

### 2. Training
Adjust your bands, paths, and hyperparameters in `configs/default_config.yaml`, then execute the training loop:
```bash
python train.py --config configs/default_config.yaml
```
Monitor real-time training metrics (IoU, F1) via TensorBoard:
```bash
tensorboard --logdir runs/glacier_seg
```

### 3. Evaluation
Evaluate the optimal checkpoint against hold-out test tiles:
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

### 4. Full Scene Inference & Vectorization
Run the model over an entire unseen satellite scene (gigapixel scale). The script automatically outputs a stitched probabilistic GeoTIFF and a topologically correct `.geojson` vector polygon map.
```bash
python predict.py --checkpoint checkpoints/best_model.pth --input data/raw_unseen_scene.tif --output_dir predictions/
```
