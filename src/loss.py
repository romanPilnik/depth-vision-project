import torch
import torch.nn as nn

class ScaleInvariantLogLoss(nn.Module):
    def __init__(self, lam=0.5, eps=1e-6):
        super().__init__()
        self.lam = lam
        self.eps = eps

    def forward(self, prediction, target):
        pred_flat = prediction.view(prediction.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        valid_mask = target_flat > 0

        loss_sum = 0.0
        batch_size = prediction.size(0)

        for i in range(batch_size):
            mask = valid_mask[i]
            if mask.sum() == 0:
                continue

            p = pred_flat[i][mask]
            t = target_flat[i][mask]

            log_diff = torch.log(p + self.eps) - torch.log(t + self.eps)

            mse_term = torch.mean(log_diff ** 2)
            var_term = (self.lam * (torch.mean(log_diff) ** 2))

            loss_sum += (mse_term - var_term)

        return loss_sum / batch_size