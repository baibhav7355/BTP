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

def evaluate(config_path: str, checkpoint_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    
    test_dataset = GlacierDataset(
        image_dir=os.path.join(config['data']['processed_dir'], 'test/images'),
        mask_dir=os.path.join(config['data']['processed_dir'], 'test/masks'),
        config=config,
        transform=get_val_transforms(),
        is_train=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=config['training'].get('num_workers', 4)
    )
    
    # Load Model and State Dict
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
    
    print("\n--- Final Evaluation Results ---")
    for k, v in results.items():
        print(f"{k.capitalize()}: {v:.4f}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model on test hold-out set")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to best_model.pth")
    args = parser.parse_args()
    
    evaluate(args.config, args.checkpoint)
