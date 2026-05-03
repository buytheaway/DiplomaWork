"""
Test custom face image with the trained model
Usage: python test_my_face.py <path_to_image>
Example: python test_my_face.py my_photo.jpg
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
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


def extract_embedding_from_image(img: Image.Image, config) -> tuple[np.ndarray, np.ndarray, Path]:
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


def run_custom_face(image_path: str) -> dict:
    """Test custom face against dataset"""
    config = load_config("training/config.yaml")
    
    # Verify image exists
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"❌ ERROR: Image not found: {image_path}")
        print(f"   Current directory: {Path.cwd()}")
        return {}
    
    print("=" * 70)
    print("🔍 CUSTOM FACE RECOGNITION TEST")
    print("=" * 70)
    
    # Load test face image
    try:
        test_img = Image.open(image_file)
        print(f"\n📸 Test face loaded: {image_file.name}")
        print(f"   Full path: {image_file.absolute()}")
        print(f"   Image size: {test_img.size}")
        print(f"   Image mode: {test_img.mode}")
    except Exception as e:
        print(f"❌ ERROR: Could not load image: {e}")
        return {}
    
    # Extract embedding
    print("\n⚙️ Extracting embedding from test face...")
    try:
        embedding, embedding_norm, checkpoint = extract_embedding_from_image(test_img, config)
        print("✅ Embedding extracted successfully!")
        print(f"   Checkpoint: {checkpoint.name}")
        print(f"   Embedding dimension: {embedding.shape[0]}")
        print(f"   Embedding L2 norm: {np.linalg.norm(embedding):.6f}")
    except Exception as e:
        print(f"❌ ERROR extracting embedding: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    # Now compare with dataset
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
    if not dataset_path.exists():
        print(f"❌ ERROR: Dataset not found at {dataset_path}")
        return {}
    
    person_dirs = sorted(dataset_path.glob("*/"))
    print(f"Found {len(person_dirs)} persons in dataset")
    
    all_results = {}
    test_emb_norm = embedding / np.linalg.norm(embedding)
    
    for person_dir in person_dirs:
        person_id = person_dir.name
        images = sorted(person_dir.glob("*.png"))
        
        if not images:
            continue
        
        similarities = []
        for img_path in images:
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0)
                
                with torch.no_grad():
                    emb = model(img_tensor).cpu().numpy()[0]
                    emb_norm = emb / np.linalg.norm(emb)
                    similarity = np.dot(test_emb_norm, emb_norm)
                    similarities.append(float(similarity))
            except Exception as e:
                print(f"Warning: Could not process {img_path}: {e}")
                continue
        
        if similarities:
            avg_sim = float(np.mean(similarities))
            all_results[person_id] = {
                "similarities": similarities,
                "avg_similarity": avg_sim,
                "max_similarity": float(np.max(similarities)),
                "min_similarity": float(np.min(similarities))
            }
    
    # Display results
    print(f"\n📊 Comparison Results ({len(all_results)} persons):\n")
    
    sorted_results = sorted(all_results.items(), 
                          key=lambda x: x[1]["avg_similarity"], 
                          reverse=True)
    
    for person_id, stats in sorted_results[:10]:  # Show top 10
        avg_sim = stats['avg_similarity']
        bar_length = int(avg_sim * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"  Person {person_id}: [{bar}] {avg_sim:.4f}")
    
    # Top matches
    print("\n" + "=" * 70)
    print("🏆 TOP 5 MATCHES")
    print("=" * 70)
    
    for rank, (person_id, stats) in enumerate(sorted_results[:5], 1):
        print(f"\n{rank}. Person {person_id}")
        print(f"   Similarity: {stats['avg_similarity']:.4f}")
        print(f"   Range: {stats['min_similarity']:.4f} - {stats['max_similarity']:.4f}")
        print(f"   Compared {len(stats['similarities'])} images")
    
    best_person = sorted_results[0][0]
    best_sim = sorted_results[0][1]["avg_similarity"]
    
    print("\n" + "=" * 70)
    print(f"✨ BEST MATCH: Person {best_person} (similarity: {best_sim:.4f})")
    print("=" * 70)
    
    return {
        "embedding": embedding,
        "embedding_normalized": embedding_norm,
        "results": all_results,
        "best_match": best_person,
        "best_similarity": best_sim
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_my_face.py <path_to_image>")
        print("\nExample:")
        print("  python test_my_face.py my_photo.jpg")
        print("  python test_my_face.py C:\\Users\\MyName\\Pictures\\face.png")
        print("\nSupported formats: jpg, jpeg, png, gif, bmp")
        sys.exit(1)
    
    image_path = sys.argv[1]
    run_custom_face(image_path)
