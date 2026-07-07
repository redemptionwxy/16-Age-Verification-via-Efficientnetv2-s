"""
model.py — transfer-learning age regressor.

All backbones are torchvision BSD-3-Clause models, safe for commercial use.

Available backbones:
  resnet18         — 11M params, fast, baseline
  resnet50         — 25M, stronger features
  mobilenet        — 5M, fastest inference (MobileNetV3-Small)
  efficientnet-b3  — 12M, best accuracy-per-parameter
  efficientnetv2-s — 21M, best overall (recommended for production)
"""
import torch.nn as nn
from torchvision import models


def build_model(backbone="efficientnetv2-s", pretrained=True):
    """
    Return a pretrained backbone with the classifier/FC head replaced by
    a single linear output = predicted age.
    """
    if backbone == "resnet18":
        net = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_f = net.fc.in_features
        net.fc = nn.Linear(in_f, 1)

    elif backbone == "resnet50":
        net = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_f = net.fc.in_features
        net.fc = nn.Linear(in_f, 1)

    elif backbone == "mobilenet":
        net = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        in_f = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_f, 1)

    elif backbone == "efficientnet-b3":
        net = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.DEFAULT if pretrained else None)
        in_f = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_f, 1)

    elif backbone == "efficientnetv2-s":
        net = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
        in_f = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_f, 1)

    else:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Choices: resnet18, resnet50, mobilenet, "
            f"efficientnet-b3, efficientnetv2-s")

    return net
