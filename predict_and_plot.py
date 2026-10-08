import os
import argparse
import glob
import numpy as np
import rasterio
from PIL import Image
import matplotlib.pyplot as plt

from predict import predict_scene

def predict_and_visualize(
    image_identifier: str,
    checkpoint: str = "checkpoints/best_model.pth",
    config_path: str = "configs/default_config.yaml",
    output_dir: str = "predictions",
    val_root: str = "../val",
    show: bool = False
):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Resolve image and label paths
    if os.path.isfile(image_identifier):
        img_path = image_identifier
        base_name = os.path.splitext(os.path.basename(img_path))[0]
    else:
        # Search by ID / number
        img_id = image_identifier.replace(".png", "").replace(".tif", "")
        img_matches = glob.glob(f"{val_root}/**/images/{img_id}.png", recursive=True)
        if not img_matches:
            img_matches = glob.glob(f"{val_root}/**/{img_id}.png", recursive=True)
            img_matches = [p for p in img_matches if "labels" not in p and "masks" not in p]
            
        if not img_matches:
            raise FileNotFoundError(f"Could not find image '{image_identifier}' in '{val_root}'.")
        img_path = img_matches[0]
        base_name = img_id

    # Find matching label if available
    label_matches = glob.glob(f"{val_root}/**/labels/{base_name}.png", recursive=True)
    label_path = label_matches[0] if label_matches else None
    
    print(f"=== Predicting on Image: {img_path} ===")
    
    # 2. Run Sliding Window Model Inference
    predict_scene(
        config_path=config_path,
        checkpoint_path=checkpoint,
        input_raster=img_path,
        output_dir=output_dir
    )
    
    # Output file paths
    pred_raster_path = os.path.join(output_dir, f"pred_{base_name}.tif")
    pred_geojson_path = os.path.join(output_dir, f"pred_{base_name}.geojson")
    
    # 3. Load for 4-Panel Visualization
    sat_img = Image.open(img_path)
    with rasterio.open(pred_raster_path) as src:
        pred_prob = src.read(1)
        
    pred_mask = (pred_prob > 0.5).astype(np.uint8)
    
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    
    # Panel 1: Satellite Image
    axes[0].imshow(sat_img)
    axes[0].set_title(f"1. Satellite Image ({base_name})", fontsize=13)
    axes[0].axis("off")
    
    # Panel 2: Ground Truth Label
    if label_path and os.path.exists(label_path):
        ground_truth = Image.open(label_path)
        axes[1].imshow(ground_truth, cmap="gray")
        axes[1].set_title("2. Ground Truth (Label)", fontsize=13)
    else:
        axes[1].text(0.5, 0.5, "No Label Available", ha='center', va='center')
        axes[1].set_title("2. Ground Truth", fontsize=13)
    axes[1].axis("off")
    
    # Panel 3: Prediction Confidence Heatmap
    im2 = axes[2].imshow(pred_prob, cmap="coolwarm", vmin=0, vmax=1)
    axes[2].set_title("3. Prediction Confidence Heatmap", fontsize=13)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Panel 4: Satellite Overlay with Boundary
    axes[3].imshow(sat_img)
    if pred_mask.sum() > 0:
        axes[3].contour(pred_mask, levels=[0.5], colors=['cyan'], linewidths=2)
        axes[3].set_title("4. Satellite + Predicted Boundary (Cyan)", fontsize=13)
    else:
        axes[3].set_title("4. Satellite (No Glacier Detected)", fontsize=13)
    axes[3].axis("off")
    
    plt.tight_layout()
    
    # Save High-Res 300 DPI Figure
    save_fig_path = os.path.join(output_dir, f"sample_{base_name}_result.png")
    plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    print(f"\n[Done] Saved Figure: {save_fig_path}")
    print(f"[Done] Saved Raster: {pred_raster_path}")
    print(f"[Done] Saved Vector: {pred_geojson_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict and plot glacier segmentation locally")
    parser.add_argument('--image', type=str, default="1000", help="Image ID (e.g. 1000) or path to image")
    parser.add_argument('--checkpoint', type=str, default="checkpoints/best_model.pth", help="Path to checkpoint")
    parser.add_argument('--config', type=str, default="configs/default_config.yaml", help="Path to config")
    parser.add_argument('--output_dir', type=str, default="predictions", help="Directory to save predictions")
    parser.add_argument('--show', action='store_true', default=False, help="Display popup interactive window")
    args = parser.parse_args()
    
    predict_and_visualize(args.image, args.checkpoint, args.config, args.output_dir, show=args.show)
