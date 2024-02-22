from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

import torch
import torch.nn as nn
from enum import Enum


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class MLPHead(nn.Module):
    def __init__(self, in_features=512, hidden_features=[512, 512], out_classes=10, mode="classifier"):
        super(MLPHead, self).__init__()
        layers = []
        last_layer_size = in_features
        for mlp_size in [in_features] + hidden_features:
            layers.append(nn.Linear(last_layer_size, mlp_size))
            layers.append(nn.ReLU())
            last_layer_size = mlp_size
        self.mlp = nn.Sequential(*layers)
        self.fc = nn.Linear(last_layer_size, out_classes)
        # TODO: make this configurable
        self.act = nn.Softmax()
        self.mode = mode

    def forward(self, x):
        x = self.mlp(x)
        if self.mode == "classifier":
            x = self.fc(x)
            x = self.act(x)
        elif self.mode == "embeddings":
            x = x
        return x


class SingleHeadFrozenBackboneClassifier(nn.Module):
    def __init__(self, backbone, head):
        super(SingleHeadFrozenBackboneClassifier, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        x = self.head(x)
        return x


def load_resnet50_imagenet_backbone():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = Identity()
    return model


def load_resnet18_imagenet_backbone():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = Identity()
    return model


backbones = {
    "resnet50": {
        "load": load_resnet50_imagenet_backbone,
        "out_features": 2048,
    },
    "resnet18": {
        "load": load_resnet18_imagenet_backbone,
        "out_features": 512,
    },
}

def make_model(
    backbone="resnet50",
    hidden_features=[1024, 512],
    out_classes=1000,
):
    if backbone not in backbones:
        raise ValueError(f"backbone {backbone} not found")
    backbone_config = backbones[backbone]
    backbone = backbone_config["load"]()
    head = MLPHead(in_features=backbone_config["out_features"], hidden_features=hidden_features, out_classes=out_classes)
    model = SingleHeadFrozenBackboneClassifier(backbone, head)
    return model
