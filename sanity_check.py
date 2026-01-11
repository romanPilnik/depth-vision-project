import torch
from torch.utils.data import DataLoader
from src.dataset import UnderwaterDataset
from src.model import UnderwaterDepthModel
from src.utils import plot_batch_predictions
from torch.optim import AdamW
from src.loss import ScaleInvariantLogLoss

# 1. Load just 4 images
dataset = UnderwaterDataset(split_file="data/splits/train_files.txt")
loader = DataLoader(dataset, batch_size=4, shuffle=True)
batch = next(iter(loader)) # Grab one batch

# 2. Setup Model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = UnderwaterDepthModel(freeze_encoder=True).to(device)
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3) # High LR for fast overfitting
criterion = ScaleInvariantLogLoss()

print("🧠 Overfitting on a single batch (Sanity Check)...")

# 3. Train on SAME batch 50 times
model.train()
pixel_values = batch["pixel_values"].to(device)
depth_labels = batch["depth_labels"].to(device)

for i in range(50):
    optimizer.zero_grad()
    preds = model(pixel_values)
    
    # Resize to match label
    preds = torch.nn.functional.interpolate(
        preds.unsqueeze(1), size=depth_labels.shape[-2:], mode='bilinear', align_corners=False
    )
    
    loss = criterion(preds, depth_labels)
    loss.backward()
    optimizer.step()
    
    if (i+1) % 10 == 0:
        print(f"Step {i+1}/50 | Loss: {loss.item():.4f}")

# 4. Visualize
print("🎨 Generating result image...")
plot_batch_predictions(pixel_values, depth_labels, preds, save_path="sanity_check_result.png")