"""
Tests for training model and embeddings
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.models.ir_resnet import build_model
from training.utils import load_config, set_seed


class TestModelArchitecture:
    """Test model architecture and forward pass"""

    @pytest.fixture
    def config(self):
        """Load config"""
        return load_config("training/config.yaml")

    @pytest.fixture
    def model(self, config):
        """Build model"""
        model = build_model(config["model"]["arch"], config["model"]["embedding_dim"])
        return model

    def test_model_builds_ir50(self, config):
        """Test that ir50 model builds successfully"""
        model = build_model(config["model"]["arch"], config["model"]["embedding_dim"])
        assert model is not None
        assert config["model"]["arch"] == "ir50"

    def test_model_forward_pass(self, model):
        """Test forward pass with dummy batch"""
        batch_size = 4
        input_size = 112
        x = torch.randn(batch_size, 3, input_size, input_size)
        
        model.eval()
        with torch.no_grad():
            embeddings = model(x)
        
        assert embeddings.shape == (batch_size, 512)
        assert embeddings.dtype == torch.float32

    def test_embedding_dimension(self, model, config):
        """Test embedding dimension matches config"""
        batch_size = 2
        input_size = 112
        x = torch.randn(batch_size, 3, input_size, input_size)
        
        model.eval()
        with torch.no_grad():
            embeddings = model(x)
        
        assert embeddings.shape[1] == config["model"]["embedding_dim"]

    def test_normalization_properties(self, model):
        """Test that embeddings can be normalized"""
        batch_size = 4
        input_size = 112
        x = torch.randn(batch_size, 3, input_size, input_size)
        
        model.eval()
        with torch.no_grad():
            embeddings = model(x)
            normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Check that normalized embeddings have unit norm
        norms = torch.norm(normalized, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-6)


class TestCheckpointLoading:
    """Test checkpoint loading and model inference"""

    @pytest.fixture
    def config(self):
        """Load config"""
        return load_config("training/config.yaml")

    @pytest.fixture
    def latest_checkpoint(self, config):
        """Find latest checkpoint"""
        output_dir = Path(config["train"]["output_dir"])
        checkpoints = sorted(output_dir.glob("checkpoint_epoch_*.pth"))
        assert len(checkpoints) > 0, f"No checkpoints found in {output_dir}"
        return checkpoints[-1]

    def test_checkpoint_exists(self, config):
        """Test that at least one checkpoint file exists"""
        output_dir = Path(config["train"]["output_dir"])
        checkpoints = list(output_dir.glob("checkpoint_epoch_*.pth"))
        assert len(checkpoints) > 0, f"No checkpoints found in {output_dir}"

    def test_checkpoint_loads_successfully(self, config, latest_checkpoint):
        """Test loading checkpoint"""
        device = torch.device("cpu")  # Use CPU for testing
        
        model = build_model(config["model"]["arch"], config["model"]["embedding_dim"]).to(device)
        state = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(state.get("state_dict", state), strict=False)
        
        assert model is not None

    def test_checkpoint_inference(self, config, latest_checkpoint):
        """Test model inference on loaded checkpoint"""
        device = torch.device("cpu")
        
        model = build_model(config["model"]["arch"], config["model"]["embedding_dim"]).to(device)
        state = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(state.get("state_dict", state), strict=False)
        model.eval()
        
        # Test forward pass
        batch_size = 2
        input_size = 112
        x = torch.randn(batch_size, 3, input_size, input_size).to(device)
        
        with torch.no_grad():
            embeddings = model(x)
        
        assert embeddings.shape == (batch_size, 512)
        assert not torch.isnan(embeddings).any()


class TestEmbeddingProperties:
    """Test embedding properties and quality"""

    @pytest.fixture
    def config(self):
        """Load config"""
        return load_config("training/config.yaml")

    @pytest.fixture
    def latest_checkpoint(self, config):
        """Find latest checkpoint"""
        output_dir = Path(config["train"]["output_dir"])
        checkpoints = sorted(output_dir.glob("checkpoint_epoch_*.pth"))
        if not checkpoints:
            pytest.skip("No checkpoints available for testing")
        return checkpoints[-1]

    @pytest.fixture
    def model_with_weights(self, config, latest_checkpoint):
        """Load model with trained weights"""
        device = torch.device("cpu")
        
        model = build_model(config["model"]["arch"], config["model"]["embedding_dim"]).to(device)
        state = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(state.get("state_dict", state), strict=False)
        model.eval()
        return model

    def test_embeddings_are_finite(self, model_with_weights):
        """Test that embeddings don't contain NaN or Inf"""
        batch_size = 8
        input_size = 112
        x = torch.randn(batch_size, 3, input_size, input_size)
        
        model_with_weights.eval()
        with torch.no_grad():
            embeddings = model_with_weights(x)
        
        assert torch.isfinite(embeddings).all()

    def test_embeddings_normalized_range(self, model_with_weights):
        """Test that embeddings are in reasonable range when normalized"""
        batch_size = 8
        input_size = 112
        x = torch.randn(batch_size, 3, input_size, input_size)
        
        model_with_weights.eval()
        with torch.no_grad():
            embeddings = model_with_weights(x)
            normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Normalized vectors should be in range [-1, 1]
        assert normalized.min() >= -1.0
        assert normalized.max() <= 1.0

    def test_embedding_consistency(self, model_with_weights):
        """Test that same input produces same embedding"""
        input_size = 112
        x = torch.randn(1, 3, input_size, input_size)
        
        model_with_weights.eval()
        with torch.no_grad():
            emb1 = model_with_weights(x)
            emb2 = model_with_weights(x)
        
        assert torch.allclose(emb1, emb2, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
