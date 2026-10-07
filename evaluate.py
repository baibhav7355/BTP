import os
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import GlacierDataset
from src.data.transforms import get_val_transforms
from src.models.unet import build_model
from src.utils.metrics import SegmentationMetrics

def evaluate(
    config_path: str, 
    checkpoint_path: str, 
    image_dir: str = None, 
    mask_dir: str = None,
    output_path: str = None
):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    
    # Resolve image and mask directories
    val_img_dir = image_dir or config['data'].get('val_image_dir')
    val_msk_dir = mask_dir or config['data'].get('val_mask_dir')
    
    # Auto-fallback for Colab environment if default path doesn't exist
    if not os.path.exists(val_img_dir):
        colab_img_candidates = [
            "/content/data/val/images",
            "/content/Val/val/images",
            "/content/Val/images"
        ]
        colab_msk_candidates = [
            "/content/data/val/labels",
            "/content/Val/val/labels",
            "/content/Val/labels"
        ]
        for c_img, c_msk in zip(colab_img_candidates, colab_msk_candidates):
            if os.path.exists(c_img) and os.path.exists(c_msk):
                val_img_dir = c_img
                val_msk_dir = c_msk
                print(f"Auto-detected dataset paths at: {val_img_dir}")
                break
                
    if not os.path.exists(val_img_dir):
        raise FileNotFoundError(f"Validation image directory not found: {val_img_dir}")
        
    print(f"Evaluating on dataset:")
    print(f"  Images: {val_img_dir}")
    print(f"  Labels: {val_msk_dir}")
    
    test_dataset = GlacierDataset(
        image_dir=val_img_dir,
        mask_dir=val_msk_dir,
        config=config,
        transform=get_val_transforms(),
        is_train=False
    )
    
    print(f"Found {len(test_dataset)} validation samples.")
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['training'].get('batch_size', 4), 
        shuffle=False, 
        num_workers=config['training'].get('num_workers', 2),
        pin_memory=(device.type == 'cuda')
    )
    
    # Load Model and State Dict
    print(f"Loading checkpoint: {checkpoint_path}")
    model = build_model(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    metrics = SegmentationMetrics(device)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on Test Set"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            logits = model(images)
            metrics.update(logits, masks)
            
    results = metrics.compute()
    
    report = [
        "=" * 45,
        "    B.TECH PROJECT: GLACIER SEGMENTATION    ",
        "         FINAL EVALUATION METRICS           ",
        "=" * 45,
        f" Model Backbone : {config['model'].get('architecture', 'UNet')} ({config['model'].get('encoder_name', 'resnet34')})",
        f" Total Samples  : {len(test_dataset)}",
        "-" * 45,
        f"  IoU (Jaccard) : {results['iou']:.4f}  ({results['iou']*100:.2f}%)",
        f"  F1 (Dice)     : {results['f1']:.4f}  ({results['f1']*100:.2f}%)",
        f"  Precision     : {results['precision']:.4f}  ({results['precision']*100:.2f}%)",
        f"  Recall        : {results['recall']:.4f}  ({results['recall']*100:.2f}%)",
        "=" * 45,
    ]
    
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report_text + "\n")
        print(f"\nSaved evaluation metrics report to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model on test hold-out set")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to best_model.pth")
    parser.add_argument('--image_dir', type=str, default=None, help="Optional override for validation images directory")
    parser.add_argument('--mask_dir', type=str, default=None, help="Optional override for validation masks directory")
    parser.add_argument('--output', type=str, default=None, help="Optional path to save evaluation report text file")
    args = parser.parse_args()
    
    evaluate(args.config, args.checkpoint, args.image_dir, args.mask_dir, args.output)
