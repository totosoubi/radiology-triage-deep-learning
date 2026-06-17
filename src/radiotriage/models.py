from __future__ import annotations

import math

import torch
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 14, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ResNet18MultiLabel(nn.Module):
    def __init__(self, num_classes: int = 14, in_channels: int = 3, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)
        if in_channels == 1:
            old = self.model.conv1
            self.model.conv1 = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
            with torch.no_grad():
                self.model.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if not name.startswith("fc."):
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TinyViT(nn.Module):
    def __init__(
        self,
        num_classes: int = 14,
        in_channels: int = 1,
        image_size: int = 64,
        patch_size: int = 8,
        dim: int = 96,
        depth: int = 2,
        heads: int = 4,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        n_patches = (image_size // patch_size) ** 2
        self.patch = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            batch_first=True,
            dropout=0.1,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.patch(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(x.size(0), -1, -1)
        z = torch.cat([cls, z], dim=1) + self.pos
        z = self.encoder(z)
        return self.head(self.norm(z[:, 0]))


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 1, latent_channels: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, latent_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 96):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = (tokens != 0).float().unsqueeze(-1)
        emb = self.embedding(tokens) * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        return self.norm(emb.sum(dim=1) / denom)


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalClassifier(nn.Module):
    def __init__(self, mode: str, vocab_size: int, num_classes: int = 14, in_channels: int = 1, dim: int = 96):
        super().__init__()
        if mode not in {"image", "text", "fusion"}:
            raise ValueError("mode must be image, text or fusion")
        self.mode = mode
        self.image_encoder = ImageEncoder(in_channels, dim) if mode in {"image", "fusion"} else None
        self.text_encoder = TextEncoder(vocab_size, dim) if mode in {"text", "fusion"} else None
        head_dim = dim * 2 if mode == "fusion" else dim
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(head_dim, num_classes))

    def forward(self, image: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.image_encoder is not None:
            parts.append(self.image_encoder(image))
        if self.text_encoder is not None:
            parts.append(self.text_encoder(tokens))
        return self.head(torch.cat(parts, dim=1))


def build_supervised_model(name: str, num_classes: int, in_channels: int, image_size: int, pretrained: bool) -> nn.Module:
    if name == "cnn":
        return SimpleCNN(num_classes=num_classes, in_channels=in_channels)
    if name == "resnet18":
        return ResNet18MultiLabel(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)
    if name == "vit":
        return TinyViT(num_classes=num_classes, in_channels=in_channels, image_size=image_size)
    raise ValueError(f"Unknown model: {name}")
