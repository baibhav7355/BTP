import os
import shutil
from pathlib import Path
import random
import argparse

def setup_local_data(glid_dir, val_dir, output_dir, split_ratio=0.8):
    """
    Automatically splits GLID (images) and Val (masks) into Train, Val, and Test
    folders required by the training pipeline.
    """
    glid_dir = Path(glid_dir)
    val_dir = Path(val_dir)
    output_dir = Path(output_dir)
    
    print(f"Reading images from {glid_dir}...")
    print(f"Reading masks from {val_dir}...")
    
    # Create necessary subdirectories
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'masks').mkdir(parents=True, exist_ok=True)
        
    images = sorted([f for f in glid_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.tif', '.tiff']])
    
    if not images:
        print("Error: No .tif images found in GLID folder!")
        return
        
    # Shuffle for random split
    random.seed(42)
    random.shuffle(images)
    
    # 80% Train, 10% Val, 10% Test
    train_split = int(len(images) * split_ratio)
    val_split = int(len(images) * 0.9) 
    
    train_imgs = images[:train_split]
    val_imgs = images[train_split:val_split]
    test_imgs = images[val_split:]
    
    def copy_files(img_list, split_name):
        copied = 0
        for img_path in img_list:
            mask_path = val_dir / img_path.name
            if mask_path.exists():
                shutil.copy(img_path, output_dir / split_name / 'images' / img_path.name)
                shutil.copy(mask_path, output_dir / split_name / 'masks' / mask_path.name)
                copied += 1
            else:
                print(f"Warning: Mask not found for {img_path.name}. Skipping.")
        return copied
                
    print(f"Organizing {len(train_imgs)} images into Training set...")
    t = copy_files(train_imgs, 'train')
    
    print(f"Organizing {len(val_imgs)} images into Validation set...")
    v = copy_files(val_imgs, 'val')
    
    print(f"Organizing {len(test_imgs)} images into Test set...")
    te = copy_files(test_imgs, 'test')
    
    print(f"\nDone! Successfully copied {t+v+te} images/masks to {output_dir}")
    print("You are ready to run: python train.py --config configs/default_config.yaml")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split GLID and Val into train/val/test")
    parser.add_argument('--glid', type=str, required=True, help="Path to your GLID folder")
    parser.add_argument('--val', type=str, required=True, help="Path to your Val folder")
    parser.add_argument('--out', type=str, default='data/processed/tiles', help="Output directory")
    args = parser.parse_args()
    
    setup_local_data(args.glid, args.val, args.out)
