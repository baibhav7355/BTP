import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional

class MultiSpectralUnet(nn.Module):
    """
    U-Net wrapper around segmentation_models_pytorch (SMP) that cleanly supports
    N-channel multi-spectral inputs. 
    SMP handles encoder weight modification for `in_channels != 3` by cleverly
    reusing pre-trained weights to bootstrap learning.
    """
    def __init__(
        self, 
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 7, 
        classes: int = 1,
        activation: Optional[str] = None
    ):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output logits/probabilities of shape (B, Classes, H, W)
        """
        return self.model(x)

def build_model(config: dict) -> nn.Module:
    """Factory function to build model from config."""
    model_cfg = config.get("model", {})
    
    architecture = model_cfg.get("architecture", "Unet")
    if architecture != "Unet":
        raise NotImplementedError(f"Architecture {architecture} not yet implemented.")
        
    model = MultiSpectralUnet(
        encoder_name=model_cfg.get("encoder_name", "resnet34"),
        encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
        in_channels=model_cfg.get("in_channels", 7),
        classes=model_cfg.get("classes", 1),
        activation=None  # We output logits so we can use numerically stable BCEWithLogits loss functions
    )
    return model
