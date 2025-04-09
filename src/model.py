import torch.nn as nn
from torchvision import models

def create_model(num_classes):
    """
    Creates and returns a ResNet50 model with a custom classification head.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: The ResNet50 model.
    """
    model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),  # Regularization
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model