"""AdaFace: Quality Adaptive Margin for Face Recognition (CVPR 2022).

Implements adaptive margin that adjusts based on image quality (approximated
by the embedding norm *before* normalization).  Low-quality samples get a
smaller margin so the model does not overfit to noisy data; high-quality
samples get a larger margin for tighter clusters.

Reference: Kim et al., "AdaFace: Quality Adaptive Margin for Face
Recognition", CVPR 2022.  https://github.com/mk-minchul/AdaFace
"""

from __future__ import annotations

import math

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

        # Exponential moving average of ||z|| (norm before L2 norm)
        self.register_buffer("t", torch.zeros(1))
        self.register_buffer("batch_mean", torch.ones(1) * 20.0)
        self.register_buffer("batch_std", torch.ones(1) * 100.0)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Compute norms before normalization (quality proxy)
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True).clamp(min=1e-6)
        # Normalize
        normalized_embeddings = embeddings / norms
        normalized_weight = F.normalize(self.weight)

        # Cosine similarity
        cosine = F.linear(normalized_embeddings, normalized_weight).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Update running stats of norms
        if self.training:
            safe_norms = norms.detach().squeeze()
            mean = safe_norms.mean()
            std = safe_norms.std()
            self.batch_mean = mean
            self.batch_std = std
            self.t = (1 - self.t_alpha) * self.t + self.t_alpha * mean

        # Compute adaptive margin concentration
        safe_norms = norms.detach().squeeze()
        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + 1e-3)
        margin_scaler = margin_scaler * self.h
        margin_scaler = margin_scaler.clamp(-1.0, 1.0)

        # Compute g_angular and g_additive
        g_angular = self.margin * margin_scaler * -1.0
        g_additive = self.margin + (self.margin * margin_scaler)

        # ArcFace-style with adaptive margins
        theta = torch.acos(cosine)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Apply margins only to the target class
        # theta_m = theta + m1 (angular margin) for target class
        adjusted_cosine = torch.cos(theta + g_angular.unsqueeze(1) * one_hot) - g_additive.unsqueeze(1) * one_hot

        output = self.scale * adjusted_cosine
        return output
