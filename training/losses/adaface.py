"""AdaFace: Quality Adaptive Margin for Face Recognition (CVPR 2022)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AdaFace(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.4,
        scale: float = 64.0,
        h: float = 0.333,
        t_alpha: float = 0.01,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale
        self.h = h
        self.t_alpha = t_alpha

        # EMA статистика по нормам эмбеддингов для более стабильного margin scaling.
        self.register_buffer("batch_mean", torch.ones(1) * 20.0)
        self.register_buffer("batch_std", torch.ones(1) * 100.0)

    def inference_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True).clamp(min=1e-6)
        normalized_embeddings = embeddings / norms
        normalized_weight = F.normalize(self.weight)
        cosine = F.linear(normalized_embeddings, normalized_weight).clamp(
            -1.0 + 1e-7,
            1.0 - 1e-7,
        )
        return cosine * self.scale

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True).clamp(min=1e-6)
        normalized_embeddings = embeddings / norms
        normalized_weight = F.normalize(self.weight)
        cosine = F.linear(normalized_embeddings, normalized_weight).clamp(
            -1.0 + 1e-7,
            1.0 - 1e-7,
        )

        if self.training:
            safe_norms = norms.detach().squeeze()
            mean = safe_norms.mean()
            std = safe_norms.std(unbiased=False).clamp(min=1e-3)
            self.batch_mean.mul_(1.0 - self.t_alpha).add_(mean * self.t_alpha)
            self.batch_std.mul_(1.0 - self.t_alpha).add_(std * self.t_alpha)

        safe_norms = norms.detach().squeeze()
        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + 1e-3)
        margin_scaler = margin_scaler * self.h
        margin_scaler = margin_scaler.clamp(-1.0, 1.0)

        g_angular = self.margin * margin_scaler * -1.0
        g_additive = self.margin + (self.margin * margin_scaler)

        theta = torch.acos(cosine)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        adjusted_cosine = (
            torch.cos(theta + g_angular.unsqueeze(1) * one_hot)
            - g_additive.unsqueeze(1) * one_hot
        )

        output = self.scale * adjusted_cosine
        return output
