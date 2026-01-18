import torch
import torch.nn as nn
from torchvision.models import resnet18


class CraterRefiner(nn.Module):
    """
    CNN that refines crater ellipse geometry.
    Input : (B, 1, H, W) grayscale crop
    Output: (B, 5) -> [dx, dy, da, db, dtheta]
    """

    def __init__(self):
        super().__init__()

        # Backbone
        self.backbone = resnet18(weights=None)

        # Change first conv to accept 1-channel (grayscale)
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace classifier head
        self.backbone.fc = nn.Linear(512, 5)

        # Initialize last layer small (important for stability)
        nn.init.normal_(self.backbone.fc.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.backbone.fc.bias)

    def forward(self, x):
        """
        x: torch.Tensor, shape (B, 1, H, W)
        returns: torch.Tensor, shape (B, 5)
        """
        return self.backbone(x)
