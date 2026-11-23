import torch.nn as nn
from torchvision import models


class SmallCNN(nn.Module):
    """4 x [Conv-BN-ReLU x2, MaxPool] -> global average pooling -> dropout -> linear. Trained from scratch."""
    def __init__(self, n):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, n))
    def forward(self, x): return self.head(self.features(x))


def build_model(arch, n, pretrained=True):
    if arch == "scratch": return SmallCNN(n)
    m = models.resnet18(weights="DEFAULT" if pretrained else None)      # ImageNet weights (needs internet once)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, n))
    return m


def set_backbone_trainable(model, trainable):
    if isinstance(model, SmallCNN): return
    for p in model.parameters(): p.requires_grad = trainable
    for p in model.fc.parameters(): p.requires_grad = True


def cam_layer(model):
    return model.layer4[-1] if hasattr(model, "layer4") else model.features[-1]
