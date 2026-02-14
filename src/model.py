import torch
import torch.nn as nn
from transformers import AutoModelForDepthEstimation, AutoConfig

class UnderwaterDepthModel(nn.Module):
    def __init__(self, checkpoint="LiheYoung/depth-anything-small-hf", freeze_encoder=True):
        super().__init__()

        self.config = AutoConfig.from_pretrained(checkpoint)
        self.model = AutoModelForDepthEstimation.from_pretrained(checkpoint)

        if freeze_encoder:
            for name, param in self.model.backbone.named_parameters():
                param.requires_grad = False

        self._print_trainable_parameters()

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.predicted_depth

    def _print_trainable_parameters(self):
        trainable_params = 0
        all_params = 0
        for _, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_params:,} || "
            f"trainable%: {100 * trainable_params / all_params:.2f}%"
        )