import torch
import torchmetrics
from typing import Dict

class SegmentationMetrics:
    """
    Wrapper around torchmetrics for seamlessly computing binary segmentation metrics
    on batches during the training/validation loop.
    Calculates IoU (Jaccard Index), F1 Score (Dice), Precision, and Recall.
    """
    def __init__(self, device: torch.device):
        self.device = device
        
        # We use binary metrics since classes=1 (glacier lake/ice vs background)
        self.iou = torchmetrics.JaccardIndex(task='binary').to(self.device)
        self.f1 = torchmetrics.F1Score(task='binary').to(self.device)
        self.precision = torchmetrics.Precision(task='binary').to(self.device)
        self.recall = torchmetrics.Recall(task='binary').to(self.device)
        
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Update metric states with new predictions and targets.
        
        Args:
            logits (torch.Tensor): Raw model outputs (B, 1, H, W)
            targets (torch.Tensor): Ground truth labels (B, 1, H, W)
        """
        # Convert logits to probabilities then binarize
        preds = (torch.sigmoid(logits) > 0.5).int()
        targets = targets.int()
        
        self.iou.update(preds, targets)
        self.f1.update(preds, targets)
        self.precision.update(preds, targets)
        self.recall.update(preds, targets)
        
    def compute(self) -> Dict[str, float]:
        """
        Compute and return the metrics for logging (e.g. to Tensorboard/WandB).
        """
        return {
            "iou": self.iou.compute().item(),
            "f1": self.f1.compute().item(),
            "precision": self.precision.compute().item(),
            "recall": self.recall.compute().item()
        }
        
    def reset(self):
        """
        Reset all metric states at the end of an epoch.
        """
        self.iou.reset()
        self.f1.reset()
        self.precision.reset()
        self.recall.reset()
