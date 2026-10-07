# Automated Glacial Lake & Ice Semantic Segmentation (B.Tech Project)

A deep learning and geospatial remote sensing pipeline for high-resolution semantic segmentation and boundary vectorization of glacial lakes and ice bodies using satellite imagery (Sentinel-2 L2A / Landsat).

---

## 📁 Repository & Project Architecture

```text
B.Tech Project/
├── GLID/                          # Raw satellite image scenes & ground truth
├── val/                           # Raw validation scenes
└── glacier_seg/                   # Main Deep Learning & Geospatial Codebase
    ├── configs/
    │   └── default_config.yaml    # Hyperparameters, spectral bands, and dataset paths
    ├── src/                       # Modular source code
    │   ├── data/
    │   │   ├── dataset.py         # Multi-spectral PyTorch Dataset & normalization
    │   │   ├── indices.py         # On-the-fly NDWI / NDSI index calculation
    │   │   ├── tiler.py           # Large-scale satellite imagery tiler
    │   │   └── transforms.py      # Spatial & radiometric augmentations
    │   ├── models/
    │   │   ├── unet.py            # U-Net architecture with pretrained backbones
    │   │   └── losses.py          # DiceFocalLoss for severe class imbalance
    │   └── utils/
    │       ├── geo_utils.py       # Raster-to-vector polygon polygonization
    │       └── metrics.py         # IoU, F1 (Dice), Precision, Recall metrics
    ├── checkpoints/               # Trained model weights (e.g., best_model.pth)
    ├── predictions/               # Output GeoTIFF rasters & GeoJSON polygons
    ├── train.py                   # Model training and validation loop
    ├── evaluate.py                # Quantitative metrics evaluation on test/val set
    ├── predict.py                 # Full-scene sliding-window inference & vectorization
    ├── prepare_local_data.py      # Utility for dataset splitting
    ├── requirements.txt           # Python package dependencies
    └── README.md                  # Project documentation
```

---

## ⚡ Core Features

- **Multi-Spectral Native**: Accommodates multi-spectral band stacks (B2, B3, B4, B8, B11, B12) and standard RGB imagery.
- **Dynamic Spectral Indices**: Calculates indices like NDWI (Normalized Difference Water Index) and NDSI on the fly.
- **Extreme Class Imbalance Handling**: Optimized using a hybrid `DiceFocalLoss` to accurately detect sparse glacial features against rocky/snowy mountainous terrain.
- **Artifact-Free Inference**: Sliding-window inference with 2D Gaussian apodization to eliminate boundary stitching seams.
- **Automated GIS Vectorization**: Converts continuous raster probability predictions directly into georeferenced GeoJSON / Shapefile polygons for GIS tools (QGIS, ArcGIS).

---

## 🚀 Execution & Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Model Training
```bash
python train.py --config configs/default_config.yaml
```

### 3. Quantitative Model Evaluation
Evaluates the model on holdout images and computes **IoU, F1-Score, Precision, and Recall**:
```bash
python evaluate.py \
    --config configs/default_config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --output predictions/evaluation_report.txt
```

### 4. Scene Inference & Vectorization
Run prediction on any new satellite image:
```bash
python predict.py \
    --checkpoint checkpoints/best_model.pth \
    --input /path/to/satellite_scene.png \
    --output_dir predictions/
```
Outputs:
- `pred_<name>.tif`: Predicted continuous confidence raster.
- `pred_<name>.geojson`: Vectorized boundary polygons ready for GIS mapping.

