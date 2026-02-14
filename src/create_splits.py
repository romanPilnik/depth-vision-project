import os
import glob
import random

def create_split_file(output_path, pairs):
    with open(output_path, 'w') as f:
        for img, depth in pairs:
            f.write(f"{img},{depth}\n")
    print(f"Created {output_path} with {len(pairs)} samples.")

flsea_root = "data/FLSea"
flsea_pairs = []

print("Scanning FLSea folders...")
flsea_images = glob.glob(os.path.join(flsea_root, "**/*.jpg"), recursive=True) + \
               glob.glob(os.path.join(flsea_root, "**/*.jpeg"), recursive=True) + \
               glob.glob(os.path.join(flsea_root, "**/*.tiff"), recursive=True)

flsea_images = [p for p in flsea_images if "/depth/" not in p and "/seaErra/" not in p]

for img_path in flsea_images:
    base_name = os.path.basename(img_path)
    name_no_ext = os.path.splitext(base_name)[0]

    potential_depths = [
        img_path.replace("images", "depths").replace(".jpg", ".tif"),
        img_path.replace("images", "depth_maps").replace(".jpg", ".tif"),
        img_path.replace("images", "depths").replace(".jpg", ".png"),
        img_path.replace("images", "depth_maps").replace(".jpg", ".png"),
        os.path.join(
            os.path.dirname(img_path).replace("imgs", "depth"),
            name_no_ext + "_SeaErra_abs_depth.tif"
        ),
    ]

    for p in potential_depths:
        if os.path.exists(p):
            flsea_pairs.append((img_path, p))
            break

print(f"Found {len(flsea_pairs)} FLSea pairs.")

nyu_root = "data/NYUv2"
nyu_pairs = []

print("Scanning NYUv2 folders...")
for img_path in glob.glob(os.path.join(nyu_root, "**/*.jpg"), recursive=True):
    depth_path = img_path.replace(".jpg", ".png")

    if not os.path.exists(depth_path):
        depth_path = img_path.replace("images", "depths").replace(".jpg", ".png")

    if os.path.exists(depth_path):
        nyu_pairs.append((img_path, depth_path))

print(f"Found {len(nyu_pairs)} NYUv2 pairs.")

all_pairs = flsea_pairs + nyu_pairs
random.shuffle(all_pairs)

if len(all_pairs) == 0:
    print("ERROR: No pairs found. Check your 'data' folder structure.")
else:
    split_idx = int(len(all_pairs) * 0.8)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    os.makedirs("data/splits", exist_ok=True)
    create_split_file("data/splits/train_files.txt", train_pairs)
    create_split_file("data/splits/val_files.txt", val_pairs)