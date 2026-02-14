import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import argparse
from tqdm import tqdm
import numpy as np

from src.dataset import UnderwaterDataset
from src.model import UnderwaterDepthModel
from src.loss import ScaleInvariantLogLoss

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def train(args):
    device = get_device()
    print(f"Training on device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset = UnderwaterDataset(split_file="data/splits/train_files.txt")
    val_dataset = UnderwaterDataset(split_file="data/splits/val_files.txt")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = UnderwaterDepthModel(freeze_encoder=True)
    model.to(device)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = ScaleInvariantLogLoss(lam=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    start_epoch = 0

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        print(f"Resumed at epoch {start_epoch + 1}, best val loss: {best_val_loss:.4f}")

    print(f"Starting training for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")

        for batch in loop:
            pixel_values = batch["pixel_values"].to(device)
            depth_labels = batch["depth_labels"].to(device)

            optimizer.zero_grad()

            preds = model(pixel_values)

            if preds.dim() == 3:
                preds = preds.unsqueeze(1)

            if preds.shape[-2:] != depth_labels.shape[-2:]:
                preds = F.interpolate(preds, size=depth_labels.shape[-2:], mode='bilinear', align_corners=False)

            loss = criterion(preds, depth_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                pixel_values = batch["pixel_values"].to(device)
                depth_labels = batch["depth_labels"].to(device)

                preds = model(pixel_values)
                if preds.dim() == 3: preds = preds.unsqueeze(1)
                if preds.shape[-2:] != depth_labels.shape[-2:]:
                    preds = F.interpolate(preds, size=depth_labels.shape[-2:], mode='bilinear', align_corners=False)

                loss = criterion(preds, depth_labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        scheduler.step()

        print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_data["best_val_loss"] = best_val_loss
            save_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(checkpoint_data, save_path)
            print(f"New best model saved to {save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter}/{args.patience} epochs.")

        torch.save(checkpoint_data, os.path.join(args.output_dir, "last_model.pth"))

        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()
    train(args)