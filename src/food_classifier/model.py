from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from food_classifier.data import CLASSES


def build_resnet18(pretrained: bool) -> nn.Module:
    """ResNet-18 with a 12-way head, either randomly initialized or from ImageNet weights."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model
