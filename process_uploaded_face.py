"""
Process uploaded face image and test it
"""
import sys
from pathlib import Path

# The image was attached - let's use PIL to detect and process it
from PIL import Image
import io

# Try to find any image files in the workspace
workspace = Path("c:\\Users\\mukha\\OneDrive\\Documents\\GitHub\\DiplomaWork")

# Check for common image files
image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp"]
found_images = []

for pattern in image_patterns:
    found_images.extend(workspace.glob(pattern))

print("Found images:", found_images)

# If we found test_face.jpg, try to reload it
test_face = workspace / "test_face.jpg"
if test_face.exists():
    print(f"Found test face: {test_face}")
    print(f"File size: {test_face.stat().st_size} bytes")
    
    # Try to open it
    try:
        img = Image.open(test_face)
        print(f"Image loaded successfully!")
        print(f"Size: {img.size}")
        print(f"Format: {img.format}")
        print(f"Mode: {img.mode}")
        
        # Now use it
        from analyze_face import test_face_from_image
        result = test_face_from_image(img)
        
    except Exception as e:
        print(f"Error loading image: {e}")
else:
    print("test_face.jpg not found")
    print(f"Checking workspace: {workspace}")
    print(f"Contents: {list(workspace.glob('*'))[:10]}")
