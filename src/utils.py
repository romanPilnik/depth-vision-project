import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

def colorize_depth(depth_map, cmap='magma'):
    """
    Normalizes a depth map (H, W) to 0-255 and applies a colormap.
    """
    # 1. Handle Tensor inputs
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.detach().cpu().numpy()
        
    # 2. Remove singleton dimensions if any (e.g. 1xHxW -> HxW)
    depth_map = np.squeeze(depth_map)
    
    # 3. Normalize to 0-1 for visualization
    valid_mask = depth_map > 0
    if valid_mask.sum() == 0:
        return np.zeros((*depth_map.shape, 3), dtype=np.uint8)
        
    d_min = depth_map[valid_mask].min()
    d_max = depth_map[valid_mask].max()
    
    # Safe normalize
    if d_max - d_min > 1e-6:
        depth_norm = (depth_map - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_map)
        
    # 4. Apply Colormap (Magma is great for depth: Black=Close, Bright=Far)
    # Matplotlib returns RGBA float 0-1, we want RGB uint8 0-255
    colormap = plt.get_cmap(cmap)
    depth_color = (colormap(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    
    return depth_color

def plot_batch_predictions(pixel_values, target_depth, pred_depth, save_path=None):
    """
    Plots a grid of [Input Image | Ground Truth | Prediction]
    """
    batch_size = pixel_values.shape[0]
    
    # Normalize input image for display (undo ImageNet normalization if needed)
    # For simplicity, we just clip standard normalization roughly
    imgs = pixel_values.permute(0, 2, 3, 1).cpu().numpy()
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min()) # Simple 0-1 norm
    
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))
    if batch_size == 1: axes = [axes] # Handle single item case
    
    for i in range(batch_size):
        # 1. Input Image
        ax = axes[i][0] if batch_size > 1 else axes[0]
        ax.imshow(imgs[i])
        ax.set_title("Input (CLAHE)")
        ax.axis('off')
        
        # 2. Ground Truth
        ax = axes[i][1] if batch_size > 1 else axes[1]
        gt_viz = colorize_depth(target_depth[i])
        ax.imshow(gt_viz)
        ax.set_title("Ground Truth")
        ax.axis('off')

        # 3. Prediction
        ax = axes[i][2] if batch_size > 1 else axes[2]
        pred_viz = colorize_depth(pred_depth[i])
        ax.imshow(pred_viz)
        ax.set_title("Prediction")
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    
    plt.show() # If in notebook
    plt.close() # Free memory