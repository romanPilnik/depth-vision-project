import torch
import torch.nn as nn

class ScaleInvariantLogLoss(nn.Module):
    """
    Scale-Invariant Loss as defined by Eigen et al. (2014).
    Penalizes relative depth errors rather than absolute differences.
    Essential for monocular depth estimation where absolute scale is unknown.
    """
    def __init__(self, lam=0.5, eps=1e-6):
        """
        Args:
            lam (float): Lambda weighting factor (0.5 is standard).
                         lam=0 reduces to standard log-MSE.
                         lam=1 focuses entirely on scale consistency.
            eps (float): Small epsilon to prevent log(0).
        """
        super().__init__()
        self.lam = lam
        self.eps = eps

    def forward(self, prediction, target):
        """
        Args:
            prediction: (Batch, H, W) or (Batch, 1, H, W)
            target: (Batch, H, W) or (Batch, 1, H, W)
        """
        # 1. Flatten tensors to (Batch, N)
        pred_flat = prediction.view(prediction.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        # 2. Mask invalid pixels (where GT depth <= 0)
        # FLSea/NYU often use 0 or -1 for "no data"
        valid_mask = target_flat > 0
        
        # We only compute loss on valid pixels
        # Since batches might have different counts of valid pixels, we iterate or use masking carefully.
        # For efficiency, we can compute over the whole batch if we accept a slight approximation,
        # but masking is safer.
        
        loss_sum = 0.0
        batch_size = prediction.size(0)

        for i in range(batch_size):
            mask = valid_mask[i]
            if mask.sum() == 0:
                continue # Skip images with no valid depth (rare)

            p = pred_flat[i][mask]
            t = target_flat[i][mask]

            # 3. Compute Log Difference
            # log(prediction) - log(target)
            log_diff = torch.log(p + self.eps) - torch.log(t + self.eps)

            # 4. The Loss Formula
            # L = Mean(d^2) - (Lambda * (Mean(d))^2)
            # This subtracts the mean error, making the loss shift-invariant (scale-invariant in log space)
            mse_term = torch.mean(log_diff ** 2)
            var_term = (self.lam * (torch.mean(log_diff) ** 2))
            
            loss_sum += (mse_term - var_term)

        return loss_sum / batch_size