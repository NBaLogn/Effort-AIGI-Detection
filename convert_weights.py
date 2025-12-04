"""Convert PyTorch CLIP weights to MLX format."""

import argparse

import numpy as np
import torch


def map_pytorch_to_mlx(
    key: str,
    value: torch.Tensor,
) -> tuple[str | None, np.ndarray | None]:
    """Map PyTorch weight keys to MLX format."""
    # Add backbone prefix since MLX model has backbone.clip_vision_model
    mlx_key = "backbone." + key

    # Handle embeddings
    if key == "embeddings.class_embedding":
        return "backbone.embeddings.class_embedding", value.numpy()
    if key == "embeddings.patch_embedding.weight":
        return "backbone.embeddings.patch_embedding.weight", value.numpy()
    if key == "embeddings.position_embedding.weight":
        return "backbone.embeddings.position_embedding", value.numpy()

    # Handle encoder layers
    if key.startswith("encoder.layers."):
        # Replace layer indices and map attention/mlp components
        mlx_key = "backbone." + key.replace("encoder.layers.", "encoder.layers.")
        return mlx_key, value.numpy()

    # Handle post layernorm
    if key == "post_layernorm.weight":
        return "backbone.post_layernorm.weight", value.numpy()
    if key == "post_layernorm.bias":
        return "backbone.post_layernorm.bias", value.numpy()

    # Handle classifier
    if key == "classifier.weight":
        return "head.weight", value.numpy()
    if key == "classifier.bias":
        return "head.bias", value.numpy()

    # Skip other keys
    return None, None


def convert_checkpoint(pytorch_checkpoint: str, output_file: str) -> None:
    """Convert PyTorch checkpoint to MLX format."""
    # Load PyTorch checkpoint
    ckpt = torch.load(pytorch_checkpoint, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    # Filter vision model and classifier weights
    vision_weights = {}
    for key, value in state_dict.items():
        if key.startswith("backbone.vision_model."):
            vision_weights[key[len("backbone.vision_model.") :]] = value
        elif key == "classifier.weight":
            vision_weights["classifier.weight"] = value
        elif key == "classifier.bias":
            vision_weights["classifier.bias"] = value

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
