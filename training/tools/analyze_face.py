"""
Test custom face from attachment against model
"""
from __future__ import annotations

import sys
from pathlib import Path
import io
import base64

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.models.ir_resnet import build_model
from training.utils import load_config


def get_latest_checkpoint(config) -> Path:
    """Get latest checkpoint"""
    output_dir = Path(config["train"]["output_dir"])
    checkpoints = sorted(output_dir.glob("checkpoint_epoch_*.pth"))
    if not checkpoints:
        raise RuntimeError("No checkpoints found")
    return checkpoints[-1]


def extract_embedding_from_image(img: Image.Image, config) -> tuple[np.ndarray, np.ndarray]:
    """Extract embedding from PIL Image"""
    device = torch.device("cpu")
    
    # Load model
    model = build_model(config["model"]["arch"], config["model"]["embedding_dim"]).to(device)
    checkpoint = get_latest_checkpoint(config)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("state_dict", state), strict=False)
    model.eval()
    
    # Process image
    input_size = int(config["data"]["input_size"])
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    
    img_rgb = img.convert("RGB")
    img_tensor = transform(img_rgb).unsqueeze(0)
    
    # Extract embedding
    with torch.no_grad():
        embedding = model(img_tensor).cpu().numpy()[0]
        embedding_normalized = embedding / np.linalg.norm(embedding)
    
    return embedding, embedding_normalized, checkpoint


def compare_with_dataset(test_embedding: np.ndarray) -> dict:
    """Compare test embedding with dataset embeddings"""
    config = load_config("training/config.yaml")
    device = torch.device("cpu")
    
    # Load model
    model = build_model(config["model"]["arch"], config["model"]["embedding_dim"]).to(device)
    checkpoint = get_latest_checkpoint(config)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("state_dict", state), strict=False)
    model.eval()
    
    # Load dataset images
    dataset_path = Path("datasets/digiface1m_small/train")
    person_dirs = sorted(dataset_path.glob("*/"))[:3]  # Use 3 persons
    
    input_size = int(config["data"]["input_size"])
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    
    results = {}
    test_emb_norm = test_embedding / np.linalg.norm(test_embedding)
    
    for person_dir in person_dirs:
        person_id = person_dir.name
        images = sorted(person_dir.glob("*.png"))[:3]  # Use first 3 images per person
        
        similarities = []
        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0)
            
            with torch.no_grad():
                emb = model(img_tensor).cpu().numpy()[0]
                emb_norm = emb / np.linalg.norm(emb)
                similarity = np.dot(test_emb_norm, emb_norm)
                similarities.append(float(similarity))
        
        results[person_id] = {
            "similarities": similarities,
            "avg_similarity": float(np.mean(similarities)),
            "max_similarity": float(np.max(similarities)),
            "min_similarity": float(np.min(similarities))
        }
    
    return results


def test_face_from_image(img: Image.Image) -> dict:
    """Test face from PIL Image object"""
    config = load_config("training/config.yaml")
    
    print("=" * 70)
    print("🔍 FACE EMBEDDING TEST")
    print("=" * 70)
    
    # Extract embedding
    print("\n📸 Processing face...")
    embedding, embedding_norm, checkpoint = extract_embedding_from_image(img, config)
    
    print(f"✅ Embedding extracted successfully!")
    print(f"   Checkpoint: {Path(checkpoint).name}")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Embedding norm: {np.linalg.norm(embedding):.4f}")
    print(f"   Sample values: {embedding[:5]}")
    
    # Compare with dataset
    print("\n📊 Comparing with dataset faces...")
    comparisons = compare_with_dataset(embedding)
    
    for person_id, stats in comparisons.items():
        print(f"\n   👤 Person {person_id}:")
        print(f"      Average similarity: {stats['avg_similarity']:.4f}")
        print(f"      Max similarity:     {stats['max_similarity']:.4f}")
        print(f"      Min similarity:     {stats['min_similarity']:.4f}")
        sims_str = ", ".join([f"{s:.4f}" for s in stats['similarities']])
        print(f"      Individual sims:    [{sims_str}]")
    
    # Find best match
    best_match = max(comparisons.items(), key=lambda x: x[1]["avg_similarity"])
    print(f"\n🎯 BEST MATCH: Person {best_match[0]} (similarity: {best_match[1]['avg_similarity']:.4f})")
    
    print("\n" + "=" * 70)
    
    return {
        "embedding": embedding,
        "embedding_normalized": embedding_norm,
        "comparisons": comparisons,
        "best_match": best_match[0],
        "best_similarity": best_match[1]["avg_similarity"]
    }


if __name__ == "__main__":
    # This will be called with an image
    print("Import this module and call test_face_from_image(pil_image)")
