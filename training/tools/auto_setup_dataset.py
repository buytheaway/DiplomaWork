"""
Automatically extract and prepare digiface1m dataset
"""
import zipfile
from pathlib import Path
from tqdm import tqdm
import os
import shutil


def extract_dataset():
    """Extract digiface1m dataset from zips"""
    
    zips_dir = Path("datasets/digiface1m/zips")
    images_dir = Path("datasets/digiface1m/images")
    
    print("=" * 70)
    print("📦 EXTRACTING DIGIFACE1M DATASET")
    print("=" * 70)
    
    # Create images directory
    images_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {images_dir.absolute()}")
    
    # Get all zip files
    zip_files = sorted(zips_dir.glob("*.zip"))
    total_size_gb = sum(f.stat().st_size for f in zip_files) / (1024**3)
    
    print(f"\n📦 Found {len(zip_files)} zip files")
    print(f"   Total size: {total_size_gb:.1f} GB")
    
    # Extract each zip
    for zip_file in zip_files:
        print(f"\n🔓 Extracting {zip_file.name}...")
        size_mb = zip_file.stat().st_size / (1024**2)
        print(f"   Size: {size_mb:.1f} MB")
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Get list of files
                file_list = zip_ref.namelist()
                
                # Extract with progress bar
                for file_info in tqdm(zip_ref.infolist(), desc="   Extracting", leave=False):
                    zip_ref.extract(file_info, images_dir)
            
            print(f"   ✅ Extraction completed")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    # Verify extraction
    print("\n" + "=" * 70)
    print("📊 VERIFICATION")
    print("=" * 70)
    
    # Count images
    all_images = list(images_dir.rglob("*.png")) + list(images_dir.rglob("*.jpg"))
    print(f"\n📸 Total images extracted: {len(all_images)}")
    
    # Count persons (unique directories)
    person_dirs = {}
    for img in all_images:
        person_id = img.parent.name
        person_dirs[person_id] = person_dirs.get(person_id, 0) + 1
    
    print(f"👥 Unique persons: {len(person_dirs)}")
    
    if person_dirs:
        avg_imgs = len(all_images) / len(person_dirs)
        print(f"📈 Average images per person: {avg_imgs:.1f}")
        
        # Show distribution
        sorted_persons = sorted(person_dirs.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop 10 persons by image count:")
        for person_id, count in sorted_persons[:10]:
            print(f"   Person {person_id}: {count} images")
    
    print("\n✅ Dataset extraction complete!")
    print("=" * 70)
    
    return len(all_images), len(person_dirs)


def prepare_train_val_split(train_split=0.8):
    """Split dataset into train and validation sets"""
    
    images_dir = Path("datasets/digiface1m/images")
    train_dir = Path("datasets/digiface1m/train")
    val_dir = Path("datasets/digiface1m/val")
    
    print("\n" + "=" * 70)
    print("🔀 PREPARING TRAIN/VAL SPLIT")
    print("=" * 70)
    
    # Create directories
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all person directories
    person_dirs = sorted([d for d in images_dir.glob("*") if d.is_dir()])
    
    if not person_dirs:
        print("❌ No person directories found!")
        return
    
    print(f"\n📁 Found {len(person_dirs)} person directories")
    print(f"   Train split: {train_split*100:.0f}%")
    print(f"   Val split: {(1-train_split)*100:.0f}%")
    
    # Copy images
    total_train = 0
    total_val = 0
    
    for person_dir in tqdm(person_dirs, desc="Organizing dataset"):
        person_id = person_dir.name
        images = sorted(person_dir.glob("*.png")) + sorted(person_dir.glob("*.jpg"))
        
        # Split images
        split_idx = int(len(images) * train_split)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Create person directories
        train_person_dir = train_dir / person_id
        val_person_dir = val_dir / person_id
        train_person_dir.mkdir(exist_ok=True)
        val_person_dir.mkdir(exist_ok=True)
        
        # Copy train images
        for img in train_images:
            shutil.copy2(img, train_person_dir / img.name)
            total_train += 1
        
        # Copy val images
        for img in val_images:
            shutil.copy2(img, val_person_dir / img.name)
            total_val += 1
    
    print(f"\n✅ Dataset split complete!")
    print(f"   Train images: {total_train}")
    print(f"   Val images: {total_val}")
    print(f"   Persons: {len(person_dirs)}")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    # Extract dataset
    num_images, num_persons = extract_dataset()
    
    if num_images > 0:
        # Ask about train/val split
        print("\n" + "=" * 70)
        response = input("Create train/val split? (y/n): ").strip().lower()
        if response == 'y':
            prepare_train_val_split(train_split=0.8)
    
    print("\n🎉 Done!")
