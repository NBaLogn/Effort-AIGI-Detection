"""SVD Residual Linear layer implementation in MLX.

Based on the PyTorch SVDResidualLinear for efficient parameter reduction.
"""

import math

import mlx.core as mx
from mlx import nn


class SVDResidualLinear(nn.Module):
    """Linear layer with SVD-based residual parameterization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        bias: bool = True,
        init_weight: mx.array | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r

        # Original weights (fixed)
        self.weight_main = mx.zeros((out_features, in_features))
        if init_weight is not None:
            self.weight_main = init_weight.copy()
        else:
            # Kaiming uniform initialization
            scale = math.sqrt(1 / in_features)
            self.weight_main = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(out_features, in_features),
            )

        # Bias
        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None

        # Residual components (trainable)
        self.S_residual: mx.array | None = None
        self.U_residual: mx.array | None = None
        self.V_residual: mx.array | None = None

        # Fixed components
        self.S_r: mx.array | None = None
        self.U_r: mx.array | None = None
        self.V_r: mx.array | None = None

        # For loss computation
        self.weight_original_fnorm: float | None = None

    def apply_svd(self):
        """Apply SVD to initialize the residual components."""
        # Perform SVD on the original weight
        U, S, Vh = mx.linalg.svd(self.weight_main, full_matrices=False)

        # Determine r
        r = min(self.r, len(S))

        # Keep top r singular components (main weight)
        U_r = U[:, :r]  # Shape: (out_features, r)
        S_r = S[:r]  # Shape: (r,)
        Vh_r = Vh[:r, :]  # Shape: (r, in_features)

        # Reconstruct the main weight (fixed)
        weight_main = U_r @ mx.diag(S_r) @ Vh_r

        # Set the main weight
        self.weight_main = weight_main

        # Calculate Frobenius norm of main weight
        self.weight_original_fnorm = float(mx.linalg.norm(weight_main))

        # Residual components (trainable)
        if len(S) > r:
            U_residual = U[:, r:]  # Shape: (out_features, n - r)
            S_residual = S[r:]  # Shape: (n - r,)
            Vh_residual = Vh[r:, :]  # Shape: (n - r, in_features)

            self.S_residual = S_residual
            self.U_residual = U_residual
            self.V_residual = Vh_residual

            self.S_r = S_r
            self.U_r = U_r
            self.V_r = Vh_r
        else:
            self.S_residual = None
            self.U_residual = None
            self.V_residual = None

            self.S_r = None
            self.U_r = None
            self.V_r = None

    def compute_current_weight(self) -> mx.array:
        """Compute the current weight matrix."""
        if self.S_residual is not None:
            residual_weight = (
                self.U_residual @ mx.diag(self.S_residual) @ self.V_residual
            )
            return self.weight_main + residual_weight
        return self.weight_main

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass."""
        weight = self.compute_current_weight()
        return mx.linear(x, weight, self.bias)

    def compute_orthogonal_loss(self) -> float:
        """Compute orthogonal regularization loss."""
        if self.S_residual is not None:
            UUT = mx.concatenate([self.U_r, self.U_residual], axis=1)
            UUT = UUT @ UUT.T

            VVT = mx.concatenate([self.V_r.T, self.V_residual.T], axis=0)
            VVT = VVT @ VVT.T

            identity_U = mx.eye(UUT.shape[0])
            identity_V = mx.eye(VVT.shape[0])

            loss_U = mx.mean((UUT - identity_U) ** 2)
            loss_V = mx.mean((VVT - identity_V) ** 2)

            return float(loss_U + loss_V)
        return 0.0

    def compute_keepsv_loss(self) -> float:
        """Compute loss to maintain singular values."""
        if self.S_residual is not None and self.weight_original_fnorm is not None:
            weight_current = self.compute_current_weight()
            weight_current_fnorm = float(mx.linalg.norm(weight_current))

            # Loss is the squared difference in Frobenius norms
            loss = (weight_current_fnorm**2 - self.weight_original_fnorm**2) ** 2
            return loss
        return 0.0


def apply_svd_residual_to_self_attn(model, r: int):
    """Apply SVD residual to self-attention layers in CLIP vision model."""
    for name, module in model.named_modules():
        if "self_attn" in name:
            # Replace Linear layers in this module
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    # Get parent module
                    parent_module = module
                    sub_module_names = sub_name.split(".")
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)

                    # Replace with SVDResidualLinear
                    svd_layer = SVDResidualLinear(
                        in_features=sub_module.weight.shape[1],
                        out_features=sub_module.weight.shape[0],
                        r=r,
                        bias=sub_module.bias is not None,
                        init_weight=sub_module.weight,
                    )
                    svd_layer.apply_svd()

                    if sub_module.bias is not None:
                        svd_layer.bias = sub_module.bias.copy()

                    setattr(parent_module, sub_module_names[-1], svd_layer)

    return model
