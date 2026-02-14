import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import argparse
import json
import numpy as np
from tqdm import tqdm

from src.dataset import UnderwaterDataset
from src.model import UnderwaterDepthModel
from src.utils import plot_batch_predictions


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def apply_median_scaling(pred, gt):
    pred_sq = pred.squeeze()
    gt_sq = gt.squeeze()

    valid_mask = gt_sq > 0
    if valid_mask.sum() == 0:
        return pred

    median_gt = torch.median(gt_sq[valid_mask])
    median_pred = torch.median(pred_sq[valid_mask])

    if median_pred.abs() < 1e-8:
        return pred

    scale = median_gt / median_pred
    return pred * scale


def compute_depth_metrics(pred, gt):
    valid_mask = gt > 0
    if valid_mask.sum() == 0:
        return None

    pred_valid = pred[valid_mask].clamp(min=1e-6)
    gt_valid = gt[valid_mask]

    abs_rel = torch.mean(torch.abs(pred_valid - gt_valid) / gt_valid)

    rmse_log = torch.sqrt(torch.mean((torch.log(pred_valid) - torch.log(gt_valid)) ** 2))

    ratio = torch.max(pred_valid / gt_valid, gt_valid / pred_valid)
    delta1 = (ratio < 1.25).float().mean()
    delta2 = (ratio < 1.25 ** 2).float().mean()
    delta3 = (ratio < 1.25 ** 3).float().mean()

    return {
        "abs_rel": abs_rel.item(),
        "rmse_log": rmse_log.item(),
        "delta1": delta1.item(),
        "delta2": delta2.item(),
        "delta3": delta3.item(),
    }


def evaluate(args):
    device = get_device()
    print(f"Evaluating on device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    val_dataset = UnderwaterDataset(split_file=args.split_file)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = UnderwaterDepthModel(freeze_encoder=True)

    if args.baseline:
        print("Running baseline evaluation (pretrained Depth Anything, no fine-tuning)")
        model_label = "Baseline (pretrained)"
    else:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        trained_epoch = checkpoint.get("epoch", -1) + 1
        best_val_loss = checkpoint.get("best_val_loss", None)
        print(f"Loaded checkpoint: epoch {trained_epoch}, best val loss: {best_val_loss:.4f}" if best_val_loss else f"Loaded checkpoint: epoch {trained_epoch}")
        model_label = f"Fine-tuned (epoch {trained_epoch})"

    model.to(device)
    model.eval()

    all_metrics = []
    vis_samples = []
    samples_collected = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            depth_labels = batch["depth_labels"].to(device)

            preds = model(pixel_values)
            if preds.dim() == 3:
                preds = preds.unsqueeze(1)
            if preds.shape[-2:] != depth_labels.shape[-2:]:
                preds = F.interpolate(preds, size=depth_labels.shape[-2:], mode='bilinear', align_corners=False)

            scaled_preds = []
            for i in range(pixel_values.shape[0]):
                scaled_pred = apply_median_scaling(preds[i], depth_labels[i])
                scaled_preds.append(scaled_pred)

                metrics = compute_depth_metrics(scaled_pred.squeeze(), depth_labels[i].squeeze())
                if metrics is not None:
                    all_metrics.append(metrics)

            if samples_collected < args.num_vis:
                scaled_batch = torch.stack(scaled_preds)
                remaining = args.num_vis - samples_collected
                take = min(pixel_values.shape[0], remaining)
                vis_samples.append((
                    pixel_values[:take].cpu(),
                    depth_labels[:take].cpu(),
                    scaled_batch[:take].cpu(),
                ))
                samples_collected += take

    if len(all_metrics) == 0:
        print("No valid images found for evaluation.")
        return

    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))

    print(f"\n{'=' * 50}")
    print(f"  Evaluation Results")
    print(f"  Model: {model_label}")
    print(f"  Split: {args.split_file} ({len(val_dataset)} images)")
    print(f"{'=' * 50}")
    print(f"\n  {'Metric':<25} {'Value':>10}")
    print(f"  {'-' * 40}")
    print(f"  {'Abs Rel':<25} {avg_metrics['abs_rel']:>10.4f}")
    print(f"  {'RMSE log':<25} {avg_metrics['rmse_log']:>10.4f}")
    print(f"  {'delta < 1.25':<25} {avg_metrics['delta1']:>10.4f}")
    print(f"  {'delta < 1.25^2':<25} {avg_metrics['delta2']:>10.4f}")
    print(f"  {'delta < 1.25^3':<25} {avg_metrics['delta3']:>10.4f}")
    print(f"  {'-' * 40}\n")

    print(f"Saving {samples_collected} sample visualizations...")
    vis_idx = 0
    for pv_chunk, gt_chunk, pred_chunk in vis_samples:
        save_path = os.path.join(args.output_dir, f"vis_{vis_idx:03d}.png")
        plot_batch_predictions(pv_chunk, gt_chunk, pred_chunk, save_path=save_path)
        vis_idx += 1

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(avg_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate depth estimation model")
    parser.add_argument("--checkpoint", type=str, default="output/best_model.pth")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--split_file", type=str, default="data/splits/val_flsea.txt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="output/eval")
    parser.add_argument("--num_vis", type=int, default=16)

    args = parser.parse_args()
    evaluate(args)