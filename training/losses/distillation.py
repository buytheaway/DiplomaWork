from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CosineEmbeddingDistillationLoss(nn.Module):
    """Cosine embedding distillation loss for normalized teacher/student vectors."""

    def forward(self, student_embeddings: torch.Tensor, teacher_embeddings: torch.Tensor) -> torch.Tensor:
        student = F.normalize(student_embeddings.float(), p=2, dim=1)
        teacher = F.normalize(teacher_embeddings.float(), p=2, dim=1)
        return 1.0 - F.cosine_similarity(student, teacher, dim=1).mean()
