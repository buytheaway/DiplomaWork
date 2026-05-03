"""
Prepare digiface1m dataset for training
"""
import zipfile
from pathlib import Path


def prepare_digiface1m():
    """Extract and organize digiface1m dataset"""
    dataset_root = Path("datasets/digiface1m")
    zips_dir = dataset_root / "zips"
    images_dir = dataset_root / "images"
    
    print("=" * 70)
    print("📦 PREPARING DIGIFACE1M DATASET")
    print("=" * 70)
    
    # Check zips
    zip_files = list(zips_dir.glob("*.zip"))
    print(f"\n📋 Found {len(zip_files)} zip files")
    
    if not zip_files:
        print("❌ No zip files found in datasets/digiface1m/zips/")
        return False
    
    # Create images directory if it doesn't exist
    images_dir.mkdir(exist_ok=True)
    
    # Extract zip files
    print("\n🔓 Extracting zip files...")
    for zf in zip_files[:3]:  # Extract first 3 zips for now
        print(f"  Extracting {zf.name}...")
        try:
            with zipfile.ZipFile(zf, 'r') as zip_ref:
                zip_ref.extractall(images_dir)
            print("    ✅ Done")
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    # Count images
    all_images = list(images_dir.rglob("*.png")) + list(images_dir.rglob("*.jpg"))
    persons = set()
    for img in all_images:
        person_id = img.parent.name
        persons.add(person_id)
    
    print("\n📊 Dataset Statistics:")
    print(f"   Total images: {len(all_images)}")
    print(f"   Unique persons: {len(persons)}")
    
    if len(all_images) > 0:
        print("\n✅ Dataset ready for training!")
        return True
    else:
        print("\n❌ No images extracted")
        return False


if __name__ == "__main__":
    prepare_digiface1m()
