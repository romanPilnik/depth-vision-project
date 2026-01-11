import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import os
import argparse
from tqdm import tqdm
import numpy as np

# Import our custom modules
from src.dataset import UnderwaterDataset
from src.model import UnderwaterDepthModel
from src.loss import ScaleInvariantLogLoss

def get_device():
    """Auto-select the best available device (MPS for Mac, CUDA for Nvidia)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def train(args):
    # 1. Setup Device & Folders
    device = get_device()
    print(f"🚀 Training on device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Load Data
    print("📂 Loading datasets...")
    # We mix both datasets (or just FLSea if you prefer)
    # Ideally, you'd concatenate them, but for this start, let's load FLSea
    # Adjust 'batch_size' based on your memory (4-8 is safe for M-series chips)
    train_dataset = UnderwaterDataset(split_file="data/splits/train_files.txt")
    val_dataset = UnderwaterDataset(split_file="data/splits/val_files.txt")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 3. Load Model
    # We freeze the encoder to avoid destroying the pre-trained weights
    model = UnderwaterDepthModel(freeze_encoder=True)
    model.to(device)

    # 4. Optimizer & Loss
    # AdamW is standard for Transformers. 
    # Learning rate 1e-4 is good for fine-tuning decoders.
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = ScaleInvariantLogLoss(lam=0.5)

    best_val_loss = float('inf')

    # 5. Training Loop
    print(f"🔥 Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        # Progress Bar
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for batch in loop:
            # Move to device
            pixel_values = batch["pixel_values"].to(device) # [B, 3, H, W]
            depth_labels = batch["depth_labels"].to(device) # [B, 1, H, W]

            optimizer.zero_grad()

            # Forward Pass
            # Model outputs raw depth prediction
            preds = model(pixel_values)
            
            # Ensure dimensions match: 
            # Model output might be [B, H, W], Target is [B, 1, H, W]
            if preds.dim() == 3:
                preds = preds.unsqueeze(1)
            
            # Resize prediction to match target EXACTLY if they differ
            # (Though our dataset class usually handles this, safety first)
            if preds.shape[-2:] != depth_labels.shape[-2:]:
                preds = F.interpolate(preds, size=depth_labels.shape[-2:], mode='bilinear', align_corners=False)

            # Calculate Loss
            loss = criterion(preds, depth_labels)
            
            # Backward Pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # 6. Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                pixel_values = batch["pixel_values"].to(device)
                depth_labels = batch["depth_labels"].to(device)

                preds = model(pixel_values)
                if preds.dim() == 3: preds = preds.unsqueeze(1)
                
                loss = criterion(preds, depth_labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        print(f"📉 Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 7. Save Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ New Best Model saved to {save_path}")

        # Save last epoch anyway
        torch.save(model.state_dict(), os.path.join(args.output_dir, "last_model.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (reduce if OOM)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
    parser.add_argument("--output_dir", type=str, default="output", help="Where to save weights")
    
    args = parser.parse_args()
    train(args)