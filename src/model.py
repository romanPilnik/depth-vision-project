import torch
import torch.nn as nn
from transformers import AutoModelForDepthEstimation, AutoConfig

class UnderwaterDepthModel(nn.Module):
    def __init__(self, checkpoint="LiheYoung/depth-anything-small-hf", freeze_encoder=True):
        """
        Args:
            checkpoint (str): Hugging Face model ID.
            freeze_encoder (bool): If True, locks the ViT backbone weights.
        """
        super().__init__()
        
        print(f"Loading foundation model: {checkpoint}...")
        # Load the configuration and model from Hugging Face
        self.config = AutoConfig.from_pretrained(checkpoint)
        self.model = AutoModelForDepthEstimation.from_pretrained(checkpoint)

        # --- ENCODER FREEZING LOGIC ---
        if freeze_encoder:
            print("❄️  Freezing Encoder (Backbone) parameters...")
            # In HF models, the backbone is usually stored under 'backbone' or 'encoder'
            # We iterate through named parameters to be safe
            for name, param in self.model.backbone.named_parameters():
                param.requires_grad = False
        
        # Log trainable parameters to verify we aren't retraining everything
        self._print_trainable_parameters()

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: Tensor [Batch, 3, H, W] - Preprocessed images
        Returns:
            predicted_depth: Tensor [Batch, H, W] - Raw depth predictions
        """
        # Forward pass through the HF model
        outputs = self.model(pixel_values=pixel_values)
        
        # Extract the depth tensor from the output object
        # Note: This is usually smaller than the original image (e.g. 518x518)
        # We will resize it to match the Ground Truth in the training loop/loss function
        return outputs.predicted_depth

    def _print_trainable_parameters(self):
        """
        Helper to visualize how many params we are actually training.
        """
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