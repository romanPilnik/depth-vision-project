import os
import glob
import random

def create_split_file(output_path, pairs):
    with open(output_path, 'w') as f:
        for img, depth in pairs:
            f.write(f"{img},{depth}\n")
    print(f"✅ Created {output_path} with {len(pairs)} samples.")

# --- 1. Process FLSea (Whatever is currently downloaded) ---
flsea_root = "data/FLSea"
flsea_pairs = []

print("Scanning FLSea folders...")
# Flexible search for images
flsea_images = glob.glob(os.path.join(flsea_root, "**/*.jpg"), recursive=True) + \
               glob.glob(os.path.join(flsea_root, "**/*.jpeg"), recursive=True)

for img_path in flsea_images:
    # Logic: Look for depth map with similar name in a parallel folder or nearby
    # This tries standard patterns: .jpg -> .tif, .jpg -> .png
    # And 'images' -> 'depths' or 'depth_maps'
    
    # 1. Guess the depth filename
    base_name = os.path.basename(img_path)
    depth_name_tif = base_name.replace(".jpg", ".tif").replace(".jpeg", ".tif")
    depth_name_png = base_name.replace(".jpg", ".png").replace(".jpeg", ".png")
    
    # 2. Search for it
    # We construct a likely path. You might need to adjust 'depths' if your folder is named 'depth_maps'
    parent = os.path.dirname(os.path.dirname(img_path)) # Go up two levels usually
    
    # Check common locations
    potential_depths = [
        img_path.replace("images", "depths").replace(".jpg", ".tif"), 
        img_path.replace("images", "depth_maps").replace(".jpg", ".tif"),
        img_path.replace("images", "depths").replace(".jpg", ".png"),
        img_path.replace("images", "depth_maps").replace(".jpg", ".png")
    ]
    
    found = False
    for p in potential_depths:
        if os.path.exists(p):
            flsea_pairs.append((img_path, p))
            found = True
            break

print(f"   -> Found {len(flsea_pairs)} FLSea pairs (so far).")

# --- 2. Process NYUv2 (The stable data) ---
# Adjust this path if needed based on your earlier ls command
nyu_root = "data/NYUv2" 
nyu_pairs = []

print("Scanning NYUv2 folders...")
for img_path in glob.glob(os.path.join(nyu_root, "**/*.jpg"), recursive=True):
    # NYU usually has matched filenames with png extension
    depth_path = img_path.replace(".jpg", ".png")
    
    # Some versions have separate folders
    if not os.path.exists(depth_path):
        depth_path = img_path.replace("images", "depths").replace(".jpg", ".png")

    if os.path.exists(depth_path):
        nyu_pairs.append((img_path, depth_path))

print(f"   -> Found {len(nyu_pairs)} NYUv2 pairs.")

# --- 3. Combine and Save ---
all_pairs = flsea_pairs + nyu_pairs
random.shuffle(all_pairs)

if len(all_pairs) == 0:
    print("❌ ERROR: No pairs found. Check your 'data' folder structure.")
    print("   Make sure NYUv2 is unzipped in data/NYUv2")
else:
    # 80% Train, 20% Validation
    split_idx = int(len(all_pairs) * 0.8)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    os.makedirs("data/splits", exist_ok=True)
    create_split_file("data/splits/train_files.txt", train_pairs)
    create_split_file("data/splits/val_files.txt", val_pairs)