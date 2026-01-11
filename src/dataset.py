import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from transformers import AutoImageProcessor

class UnderwaterDataset(Dataset):
    def __init__(self, split_file, processor_name="LiheYoung/depth-anything-small-hf"):
        """
        Args:
            split_file (str): Path to the .txt file (e.g., 'data/splits/train_files.txt')
            processor_name (str): Hugging Face model ID for transforms
        """
        # Read the text file containing "image_path,depth_path" lines
        with open(split_file, "r") as f:
            self.lines = [line.strip().split(",") for line in f.readlines()]
        
        # Load the official ViT pre-processor 
        # This handles resizing (to 518x518) and normalization automatically
        self.processor = AutoImageProcessor.from_pretrained(processor_name)

    def apply_clahe(self, image):
        """
        Applies CLAHE to the L-channel of the image to enhance local contrast.
        Essential for removing underwater haze/scattering effects.
        """
        # 1. Convert RGB to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # 2. Split channels
        l, a, b = cv2.split(lab)
        
        # 3. Apply CLAHE to L (Lightness) channel
        # clipLimit: Threshold for contrast limiting (2.0 is standard)
        # tileGridSize: Size of grid for histogram equalization (8x8 is standard)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_clahe = clahe.apply(l)
        
        # 4. Merge and convert back to RGB
        merged = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        img_path, depth_path = self.lines[idx]
        
        # --- 1. Load Image ---
        # OpenCV loads as BGR by default, convert to RGB
        image = cv2.imread(img_path)
        if image is None:
            # Fallback if a file is corrupt/missing (prevents training crash)
            print(f"Warning: Image not found {img_path}")
            return self.__getitem__((idx + 1) % len(self))
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # --- 2. Preprocessing: CLAHE ---
        image = self.apply_clahe(image)
        
        # --- 3. Load Depth ---
        # Load as-is (unchanged flags) to preserve float values or 16-bit int
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"Warning: Depth not found {depth_path}")
            return self.__getitem__((idx + 1) % len(self))
        
        # Handle different depth formats
        if depth.ndim == 3: 
            depth = depth[:, :, 0] # Take one channel if saved as RGB
        
        # Convert to float32
        depth = depth.astype(np.float32)
        
        # --- 4. Prepare for Model ---
        # The processor handles resizing the IMAGE to 518x518
        inputs = self.processor(images=image, return_tensors="pt")
        
        # We must manually resize the DEPTH to match the model's standard size (518x518)
        # This ensures we can stack them into batches without shape errors.
        h, w = 518, 518
        depth_tensor = torch.tensor(depth).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
        depth_resized = torch.nn.functional.interpolate(depth_tensor, size=(h, w), mode="nearest")
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0), # [3, 518, 518]
            "depth_labels": depth_resized.squeeze(0)           # [1, 518, 518]
        }