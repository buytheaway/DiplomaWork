"""Tests for the face embedding model on real face images."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.models.ir_resnet import build_model  # noqa: E402
from training.utils import load_config  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class TestModelOnRealFaces:
    """Test model behavior on a configured real-face dataset."""

    @pytest.fixture
    def config(self):
        return load_config("training/config.yaml")

    @pytest.fixture
    def latest_checkpoint(self, config):
        output_dir = Path(config["train"]["output_dir"])
        checkpoints = sorted(output_dir.glob("checkpoint_epoch_*.pth"))
        if not checkpoints:
            pytest.skip("No checkpoints available for testing")
        return checkpoints[-1]

    @pytest.fixture
    def model(self, config, latest_checkpoint):
        device = torch.device("cpu")
        model = build_model(config["model"]["arch"], config["model"]["embedding_dim"]).to(device)
        state = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(state.get("state_dict", state), strict=False)
        model.eval()
        return model

    @pytest.fixture
    def transform(self, config):
        input_size = int(config["data"]["input_size"])
        return transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    @pytest.fixture
    def face_images_dir(self, config):
        dataset_path = Path(config["data"]["train_dir"])
        if not dataset_path.exists():
            pytest.skip(f"Dataset not found at {dataset_path}")
        return dataset_path

    def get_person_dirs(self, face_images_dir: Path, max_count: int | None = None) -> list[Path]:
        person_dirs = sorted(path for path in face_images_dir.iterdir() if path.is_dir())
        return person_dirs[:max_count] if max_count is not None else person_dirs

    def get_images_from_person(self, person_dir: Path, max_count: int = 5) -> list[Path]:
        return [
            path
            for path in sorted(person_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ][:max_count]

    def load_face_image(self, image_path: Path, transform) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        return transform(image)

    def first_person_dir(self, face_images_dir: Path) -> Path:
        person_dirs = self.get_person_dirs(face_images_dir, max_count=1)
        if not person_dirs:
            pytest.skip("No person directories found in dataset")
        return person_dirs[0]

    def test_dataset_exists(self, face_images_dir):
        assert face_images_dir.exists()
        assert self.get_person_dirs(face_images_dir), "No person directories found in dataset"

    def test_load_face_images(self, face_images_dir, transform):
        person_dir = self.first_person_dir(face_images_dir)
        images = self.get_images_from_person(person_dir, max_count=3)
        assert images, f"No images found in {person_dir}"

        for img_path in images:
            img_tensor = self.load_face_image(img_path, transform)
            assert img_tensor.shape == (3, 112, 112), f"Unexpected image shape: {img_tensor.shape}"

    def test_extract_embeddings_from_real_faces(self, model, face_images_dir, transform):
        person_dir = self.first_person_dir(face_images_dir)
        images = self.get_images_from_person(person_dir, max_count=5)
        assert images, f"No images found in {person_dir}"

        embeddings = []
        with torch.no_grad():
            for img_path in images:
                img_tensor = self.load_face_image(img_path, transform).unsqueeze(0)
                embeddings.append(model(img_tensor).cpu().numpy())

        for embedding in embeddings:
            assert embedding.shape == (1, 512)

    def test_same_person_embeddings_similar(self, model, face_images_dir, transform):
        person_dir = self.first_person_dir(face_images_dir)
        images = self.get_images_from_person(person_dir, max_count=5)
        if len(images) < 2:
            pytest.skip("Need at least 2 images of the same person")

        embeddings = []
        with torch.no_grad():
            for img_path in images:
                img_tensor = self.load_face_image(img_path, transform).unsqueeze(0)
                emb_norm = torch.nn.functional.normalize(model(img_tensor), p=2, dim=1)
                embeddings.append(emb_norm.cpu().numpy()[0])

        ref_emb = embeddings[0]
        similarities = [np.dot(ref_emb, emb) for emb in embeddings[1:]]
        avg_similarity = np.mean(similarities)
        assert avg_similarity > 0.2, f"Low similarity for same person: {avg_similarity}"
        print(f"OK Same person similarity: {avg_similarity:.4f}")

    def test_different_persons_embeddings_are_valid(self, model, face_images_dir, transform):
        person_dirs = self.get_person_dirs(face_images_dir, max_count=2)
        if len(person_dirs) < 2:
            pytest.skip("Need at least 2 person directories")

        img_0 = self.get_images_from_person(person_dirs[0], max_count=1)[0]
        img_1 = self.get_images_from_person(person_dirs[1], max_count=1)[0]

        with torch.no_grad():
            img_0_tensor = self.load_face_image(img_0, transform).unsqueeze(0)
            img_1_tensor = self.load_face_image(img_1, transform).unsqueeze(0)
            emb_0 = torch.nn.functional.normalize(model(img_0_tensor), p=2, dim=1).cpu().numpy()[0]
            emb_1 = torch.nn.functional.normalize(model(img_1_tensor), p=2, dim=1).cpu().numpy()[0]

        similarity = np.dot(emb_0, emb_1)
        assert -1.0 <= similarity <= 1.0, f"Invalid similarity value: {similarity}"
        print(f"OK Different person similarity: {similarity:.4f}")

    def test_batch_face_processing(self, model, face_images_dir, transform):
        person_dir = self.first_person_dir(face_images_dir)
        images = self.get_images_from_person(person_dir, max_count=4)
        if len(images) < 2:
            pytest.skip("Need at least 2 images")

        batch_tensor = torch.stack([self.load_face_image(path, transform) for path in images])
        with torch.no_grad():
            embeddings = model(batch_tensor)

        assert embeddings.shape == (len(images), 512)
        assert not torch.isnan(embeddings).any()

    def test_embedding_reproducibility_real_faces(self, model, face_images_dir, transform):
        person_dir = self.first_person_dir(face_images_dir)
        images = self.get_images_from_person(person_dir, max_count=1)
        if not images:
            pytest.skip(f"No images found in {person_dir}")

        img_tensor = self.load_face_image(images[0], transform).unsqueeze(0)
        with torch.no_grad():
            emb_1 = model(img_tensor)
            emb_2 = model(img_tensor)

        assert torch.allclose(emb_1, emb_2, atol=1e-6)

    def test_multiple_persons_embeddings(self, model, face_images_dir, transform):
        person_dirs = self.get_person_dirs(face_images_dir, max_count=3)
        if len(person_dirs) < 2:
            pytest.skip("Need at least 2 person directories")

        all_embeddings = {}
        with torch.no_grad():
            for person_dir in person_dirs:
                embeddings = []
                for img_path in self.get_images_from_person(person_dir, max_count=2):
                    img_tensor = self.load_face_image(img_path, transform).unsqueeze(0)
                    emb = torch.nn.functional.normalize(model(img_tensor), p=2, dim=1)
                    embeddings.append(emb.cpu().numpy()[0])
                all_embeddings[person_dir.name] = embeddings

        for person_id, embeddings in all_embeddings.items():
            if len(embeddings) <= 1:
                continue
            within_sim = np.dot(embeddings[0], embeddings[1])
            other_person_ids = [pid for pid in all_embeddings if pid != person_id]
            if not other_person_ids or not all_embeddings[other_person_ids[0]]:
                continue

            cross_sim = np.dot(embeddings[0], all_embeddings[other_person_ids[0]][0])
            assert -1.0 <= within_sim <= 1.0
            assert -1.0 <= cross_sim <= 1.0
            print(f"OK Person {person_id}: within={within_sim:.4f} vs cross={cross_sim:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
