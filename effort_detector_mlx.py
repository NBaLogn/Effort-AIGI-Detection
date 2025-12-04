"""EffortDetector implementation in MLX.

Combines CLIP Vision Transformer with SVD modifications and linear classification head.
"""

import mlx.core as mx
from mlx import nn

from clip_vision_mlx import CLIP_VIT_L_14_CONFIG, CLIPVisionTransformer
from svd_residual_mlx import apply_svd_residual_to_self_attn


class EffortDetector(nn.Module):
    """EffortDetector with CLIP vision backbone and linear head."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or CLIP_VIT_L_14_CONFIG
        self.backbone = CLIPVisionTransformer(self.config)
        self.head = nn.Linear(self.config["hidden_size"], 2)

        # Apply SVD to self-attention layers
        self.backbone = apply_svd_residual_to_self_attn(
            self.backbone, r=1023
        )  # 1024 - 1

    def features(self, data_dict: dict) -> mx.array:
        """Extract features from image."""
        pixel_values = data_dict["image"]
        hidden_states = self.backbone(pixel_values)
        # Use the class token (first token)
        feat = hidden_states[:, 0, :]  # (batch_size, hidden_size)
        return feat

    def classifier(self, features: mx.array) -> mx.array:
        """Classify features."""
        return self.head(features)

    def forward(self, data_dict: dict, inference=False) -> dict:
        """Forward pass."""
        features = self.features(data_dict)
        pred = self.classifier(features)

        # Get probabilities
        prob = mx.softmax(pred, axis=1)[:, 1]

        pred_dict = {"cls": pred, "prob": prob, "feat": features}
        return pred_dict

    def __call__(self, data_dict: dict, inference=False) -> dict:
        """Alias for forward."""
        return self.forward(data_dict, inference)
