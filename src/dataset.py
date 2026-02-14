import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from transformers import AutoImageProcessor

class UnderwaterDataset(Dataset):
    def __init__(self, split_file, processor_name="LiheYoung/depth-anything-small-hf"):
        with open(split_file, "r") as f:
            self.lines = [line.strip().split(",") for line in f.readlines()]

        self.processor = AutoImageProcessor.from_pretrained(processor_name)
        self.target_size = self.processor.size["height"]

    def apply_clahe(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_clahe = clahe.apply(l)
        merged = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        img_path, depth_path = self.lines[idx]

        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Image not found {img_path}")
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.apply_clahe(image)

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"Warning: Depth not found {depth_path}")
            return self.__getitem__((idx + 1) % len(self))

        if depth.ndim == 3:
            depth = depth[:, :, 0]

        depth = depth.astype(np.float32)

        image = cv2.resize(image, (self.target_size, self.target_size))
        inputs = self.processor(images=image, return_tensors="pt", do_resize=False)

        h, w = self.target_size, self.target_size
        depth_tensor = torch.tensor(depth).unsqueeze(0).unsqueeze(0)
        depth_resized = torch.nn.functional.interpolate(depth_tensor, size=(h, w), mode="nearest")

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "depth_labels": depth_resized.squeeze(0)
        }