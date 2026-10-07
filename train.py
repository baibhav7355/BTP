import os
import yaml
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import GlacierDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.unet import build_model
from src.models.losses import build_loss
from src.utils.metrics import SegmentationMetrics

def train(config_path: str):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Initialize Datasets and Dataloaders
    train_dataset = GlacierDataset(
        image_dir=config['data']['train_image_dir'],
        mask_dir=config['data']['train_mask_dir'],
        config=config,
        transform=get_train_transforms(),
        is_train=True
    )
    
    val_dataset = GlacierDataset(
        image_dir=config['data']['val_image_dir'],
        mask_dir=config['data']['val_mask_dir'],
        config=config,
        transform=get_val_transforms(),
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    # Initialize Model, Loss, Optimizer
    model = build_model(config).to(device)
    criterion = build_loss(config).to(device)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=float(config['training']['learning_rate']), 
        weight_decay=float(config['training']['weight_decay'])
    )
    
    # Setup Metrics & TensorBoard Logging
    metrics = SegmentationMetrics(device)
    writer = SummaryWriter(log_dir='runs/glacier_seg')
    
    best_iou = 0.0
    checkpoint_dir = config.get('training', {}).get('checkpoint_dir', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    max_epochs = config['training']['max_epochs']
    
    for epoch in range(max_epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]")
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader)
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        metrics.reset()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Val]"):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item()
                
                metrics.update(logits, masks)
                
        val_loss /= len(val_loader)
        val_metrics = metrics.compute()
        
        # --- LOGGING ---
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Metrics/IoU', val_metrics['iou'], epoch)
        writer.add_scalar('Metrics/F1', val_metrics['f1'], epoch)
        
        print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f} | IoU: {val_metrics['iou']:.4f} | F1: {val_metrics['f1']:.4f}")
        
        # Checkpoint Saving
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            save_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best model (IoU: {best_iou:.4f}) to {save_path}")
            
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Glacier Segmentation Model")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help="Path to config file")
    args = parser.parse_args()
    
    train(args.config)
