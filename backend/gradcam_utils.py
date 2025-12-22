import cv2
import numpy as np
import torch


def reshape_transform(tensor, height=None, width=None):
    """
    Reshape the tensor from (Batch, Seq, Channels) to (Batch, Channels, Height, Width)
    for ViT Grad-CAM.
    """
    # Exclude the class token (first token)
    seq_len = tensor.size(1) - 1

    if height is None or width is None:
        # Assume square grid
        side = int(seq_len**0.5)
        height = width = side

    if height * width != seq_len:
        msg = f"Sequence length {seq_len} cannot be reshaped to {height}x{width}"
        raise ValueError(msg)

    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Permute to (Batch, Channels, Height, Width)
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def show_cam_on_image(
    img: np.ndarray,
    mask: np.ndarray,
    use_rgb: bool = False,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlays the CAM mask on the image.

    Args:
        img: The base image (float32, [0, 1]).
        mask: The CAM mask (float32, [0, 1]).
        use_rgb: Whether to use RGB for the output.
        colormap: The cv2 colormap to use.

    Returns:
        The image with the CAM overlay (uint8, [0, 255]).
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should be float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
