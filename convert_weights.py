"""Convert PyTorch CLIP weights to MLX format."""

import argparse

import numpy as np
import torch


def map_pytorch_to_mlx(
    key: str, value: torch.Tensor
) -> tuple[str | None, np.ndarray | None]:
    """Map PyTorch weight keys to MLX format."""
    # Remove vision_model prefix
    key = key.removeprefix("vision_model.")

    # Handle embeddings
    if key == "embeddings.class_embedding":
        return "embeddings.class_embedding", value.numpy()
    if key == "embeddings.patch_embedding.weight":
        return "embeddings.patch_embedding.weight", value.numpy()
    if key == "embeddings.position_embedding.weight":
        return "embeddings.position_embedding", value.numpy()

    # Handle encoder layers
    if key.startswith("encoder.layers."):
        # Replace layer indices and map attention/mlp components
        key = key.replace("encoder.layers.", "encoder.layers.")
        key = key.replace("self_attn.k_proj.", "self_attn.k_proj.")
        key = key.replace("self_attn.v_proj.", "self_attn.v_proj.")
        key = key.replace("self_attn.q_proj.", "self_attn.q_proj.")
        key = key.replace("self_attn.out_proj.", "self_attn.out_proj.")
        key = key.replace("mlp.fc1.", "mlp.fc1.")
        key = key.replace("mlp.fc2.", "mlp.fc2.")
        key = key.replace("layer_norm1.", "layer_norm1.")
        key = key.replace("layer_norm2.", "layer_norm2.")
        return key, value.numpy()

    # Handle post layernorm
    if key == "post_layernorm.weight":
        return "post_layernorm.weight", value.numpy()
    if key == "post_layernorm.bias":
        return "post_layernorm.bias", value.numpy()

    # Skip other keys
    return None, None


def convert_checkpoint(pytorch_checkpoint: str, output_file: str):
    """Convert PyTorch checkpoint to MLX format."""
    # Load PyTorch checkpoint
    ckpt = torch.load(pytorch_checkpoint, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    # Filter vision model weights
    vision_weights = {}
    for key, value in state_dict.items():
        if key.startswith("backbone.vision_model."):
            vision_weights[key[len("backbone.vision_model.") :]] = value

    # Map to MLX format
    mlx_weights = {}
    for key, value in vision_weights.items():
        mlx_key, mlx_value = map_pytorch_to_mlx(key, value)
        if mlx_key is not None:
            mlx_weights[mlx_key] = mlx_value

    # Save as NPZ
    np.savez(output_file, **mlx_weights)
    print(f"Converted weights saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch CLIP weights to MLX")
    parser.add_argument("pytorch_checkpoint", help="Path to PyTorch checkpoint")
    parser.add_argument("output_file", help="Output NPZ file path")
    args = parser.parse_args()

    convert_checkpoint(args.pytorch_checkpoint, args.output_file)
