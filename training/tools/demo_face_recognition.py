"""
Test face recognition model - demonstrate with dataset face and compare
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[0]
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


def main():
    """Main test function - use a face from dataset"""
    config = load_config("training/config.yaml")
    
    # Use person 0, image 0 as test face
    test_face_path = Path("datasets/digiface1m_small/train/0/0.png")
    
    if not test_face_path.exists():
        print(f"❌ Test face not found: {test_face_path}")
        return
    
    print("=" * 70)
    print("🔍 FACE EMBEDDING & RECOGNITION TEST")
    print("=" * 70)
    
    # Load test face image
    test_img = Image.open(test_face_path)
    print(f"\n📸 Test face loaded: {test_face_path}")
    print(f"   Image size: {test_img.size}")
    
    # Extract embedding
    print("\n⚙️ Extracting embedding from test face...")
    embedding, embedding_norm, checkpoint = extract_embedding_from_image(test_img, config)
    
    print("✅ Embedding extracted successfully!")
    print(f"   Checkpoint: {Path(checkpoint).name}")
    print(f"   Embedding dimension: {embedding.shape[0]}")
    print(f"   Embedding L2 norm: {np.linalg.norm(embedding):.4f}")
    
    # Now compare with other faces
    print("\n" + "=" * 70)
    print("🔎 SEARCHING FOR MATCHES IN DATASET")
    print("=" * 70)
    
    device = torch.device("cpu")
    model = build_model(config["model"]["arch"], config["model"]["embedding_dim"]).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("state_dict", state), strict=False)
    model.eval()
    
    input_size = int(config["data"]["input_size"])
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    
    dataset_path = Path("datasets/digiface1m_small/train")
    person_dirs = sorted(dataset_path.glob("*/"))[:5]  # Compare with 5 persons
    
    all_results = {}
    test_emb_norm = embedding / np.linalg.norm(embedding)
    
    for person_dir in person_dirs:
        person_id = person_dir.name
        images = sorted(person_dir.glob("*.png"))[:5]  # Use first 5 images per person
        
        similarities = []
        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0)
            
            with torch.no_grad():
                emb = model(img_tensor).cpu().numpy()[0]
                emb_norm = emb / np.linalg.norm(emb)
                similarity = np.dot(test_emb_norm, emb_norm)
                similarities.append(float(similarity))
        
        avg_sim = float(np.mean(similarities))
        all_results[person_id] = {
            "similarities": similarities,
            "avg_similarity": avg_sim,
            "max_similarity": float(np.max(similarities)),
            "min_similarity": float(np.min(similarities))
        }
        
        # Visual bar chart
        bar_length = int(avg_sim * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"\n👤 Person {person_id}: [{bar}] {avg_sim:.4f}")
        print(f"   Max: {all_results[person_id]['max_similarity']:.4f}, "
              f"Min: {all_results[person_id]['min_similarity']:.4f}")
    
    # Find best matches
    print("\n" + "=" * 70)
    print("🏆 TOP MATCHES")
    print("=" * 70)
    
    sorted_results = sorted(all_results.items(), 
                          key=lambda x: x[1]["avg_similarity"], 
                          reverse=True)
    
    for rank, (person_id, stats) in enumerate(sorted_results, 1):
        print(f"\n{rank}️⃣ Person {person_id}: {stats['avg_similarity']:.4f}")
        print(f"   Range: {stats['min_similarity']:.4f} - {stats['max_similarity']:.4f}")
    
    best_person = sorted_results[0][0]
    best_sim = sorted_results[0][1]["avg_similarity"]
    
    print("\n" + "=" * 70)
    print(f"🎯 BEST MATCH: Person {best_person} (similarity: {best_sim:.4f})")
    
    # Analysis
    if best_person == "0":
        print("✅ CORRECT! Test face matches with its own person!")
    else:
        print(f"⚠️ Test face from Person 0, but best match is Person {best_person}")
        print("   (This may indicate the model needs more training)")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
