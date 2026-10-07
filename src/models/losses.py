import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.
    Excellent for spatial overlap maximization.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid to get probabilities from logits
        probs = torch.sigmoid(logits)
        
        # Flatten tensors
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class FocalLoss(nn.Module):
    """
    Focal Loss for handling severe class imbalance.
    Scales the cross entropy loss to focus on hard, misclassified examples.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calculate numerically stable binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # pt is the probability of the true class
        pt = torch.exp(-bce_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceFocalLoss(nn.Module):
    """
    Hybrid loss combining Dice and Focal Loss.
    Highly recommended for tasks like sparse glacial lake segmentation where 
    background completely dominates the field of view.
    """
    def __init__(self, alpha: float = 0.5, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        """
        Args:
            alpha (float): Weight for Dice loss (1 - alpha weight for Focal Loss).
        """
        super().__init__()
        self.alpha = alpha
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(logits, targets)
        focal_loss = self.focal(logits, targets)
        
        return self.alpha * dice_loss + (1 - self.alpha) * focal_loss
        
def build_loss(config: dict) -> nn.Module:
    """Factory function to instantiate the correct loss function from config."""
    model_cfg = config.get("model", {})
    loss_name = model_cfg.get("loss", "DiceFocalLoss")
    
    if loss_name == "DiceFocalLoss":
        return DiceFocalLoss()
    elif loss_name == "BCE":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_name}")
