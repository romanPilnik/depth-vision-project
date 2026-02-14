import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

def colorize_depth(depth_map, cmap='magma'):
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.detach().cpu().numpy()

    depth_map = np.squeeze(depth_map)

    valid_mask = depth_map > 0
    if valid_mask.sum() == 0:
        return np.zeros((*depth_map.shape, 3), dtype=np.uint8)

    d_min = depth_map[valid_mask].min()
    d_max = depth_map[valid_mask].max()

    if d_max - d_min > 1e-6:
        depth_norm = (depth_map - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_map)

    colormap = plt.get_cmap(cmap)
    depth_color = (colormap(depth_norm)[:, :, :3] * 255).astype(np.uint8)

    return depth_color

def plot_batch_predictions(pixel_values, target_depth, pred_depth, save_path=None):
    batch_size = pixel_values.shape[0]

    imgs = pixel_values.permute(0, 2, 3, 1).cpu().numpy()
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())

    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))
    if batch_size == 1: axes = [axes]

    for i in range(batch_size):
        ax = axes[i][0] if batch_size > 1 else axes[0]
        ax.imshow(imgs[i])
        ax.set_title("Input (CLAHE)")
        ax.axis('off')

        ax = axes[i][1] if batch_size > 1 else axes[1]
        gt_viz = colorize_depth(target_depth[i])
        ax.imshow(gt_viz)
        ax.set_title("Ground Truth")
        ax.axis('off')

        ax = axes[i][2] if batch_size > 1 else axes[2]
        pred_viz = colorize_depth(pred_depth[i])
        ax.imshow(pred_viz)
        ax.set_title("Prediction")
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")

    plt.show()
    plt.close()