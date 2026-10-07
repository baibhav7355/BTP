import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    """
    Spatial augmentations for multi-spectral remote sensing imagery.
    Note: Standard color jitter and brightness changes are avoided out-of-the-box 
    since they operate on assumptions of 3-channel RGB imagery in [0, 255].
    Instead, we perform purely geometric and spatial augmentations to preserve 
    the multi-spectral physical reflectance values.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        ToTensorV2() # Automatically converts (H, W, C) numpy to (C, H, W) PyTorch Tensor
    ])

def get_val_transforms():
    """
    Validation transforms (usually just to tensor).
    """
    return A.Compose([
        ToTensorV2()
    ])

def get_predict_transforms():
    """
    Inference transforms.
    """
    return A.Compose([
        ToTensorV2()
    ])
