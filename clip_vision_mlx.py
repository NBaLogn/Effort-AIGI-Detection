"""CLIP Vision Transformer implementation in MLX.

Based on OpenAI CLIP ViT-L/14 architecture with SVD modifications.
"""

import mlx.core as mx
from mlx import nn


class CLIPVisionEmbeddings(nn.Module):
    """CLIP Vision Embeddings with patch and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config["hidden_size"]
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]

        self.class_embedding = mx.zeros((self.embed_dim,))
        self.patch_embedding = nn.Conv2d(
            in_channels=config["num_channels"],
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = mx.zeros((self.num_positions, self.embed_dim))

    def __call__(self, pixel_values: mx.array) -> mx.array:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(
            pixel_values
        )  # (batch_size, embed_dim, height, width)
        patch_embeds = mx.transpose(
            patch_embeds, (0, 2, 3, 1)
        )  # (batch_size, height, width, embed_dim)
        patch_embeds = mx.reshape(
            patch_embeds, (batch_size, -1, self.embed_dim)
        )  # (batch_size, num_patches, embed_dim)

        # Add class embedding
        class_embeds = mx.broadcast_to(
            self.class_embedding, (batch_size, 1, self.embed_dim)
        )
        embeddings = mx.concatenate([class_embeds, patch_embeds], axis=1)

        # Add position embeddings
        embeddings = embeddings + self.position_embedding

        return embeddings


class CLIPAttention(nn.Module):
    """Multi-head attention for CLIP."""

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.head_dim = self.embed_dim // self.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: mx.array, seq_len: int, bsz: int):
        return mx.reshape(
            tensor, (bsz, seq_len, self.num_attention_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        bsz, tgt_len, embed_dim = hidden_states.shape

        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), tgt_len, bsz)
        value_states = self._shape(self.v_proj(hidden_states), tgt_len, bsz)

        proj_shape = (bsz * self.num_attention_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).reshape(proj_shape)
        key_states = key_states.reshape(proj_shape)
        value_states = value_states.reshape(proj_shape)

        src_len = tgt_len
        attn_weights = mx.matmul(query_states, key_states.transpose(0, 2, 1))

        if attn_weights.shape != (bsz * self.num_attention_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_attention_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.shape}",
            )

        attn_weights = mx.softmax(attn_weights, axis=-1)

        attn_output = mx.matmul(attn_weights, value_states)

        attn_output = mx.reshape(
            attn_output, (bsz, self.num_attention_heads, tgt_len, self.head_dim)
        )
        attn_output = mx.transpose(attn_output, (0, 2, 1, 3))
        attn_output = mx.reshape(attn_output, (bsz, tgt_len, embed_dim))

        attn_output = self.out_proj(attn_output)

        return attn_output


class CLIPMLP(nn.Module):
    """MLP for CLIP."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.fc2 = nn.Linear(config["intermediate_size"], config["hidden_size"])

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    """Transformer encoder layer for CLIP."""

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config["hidden_size"]
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config["layer_norm_eps"])
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config["layer_norm_eps"])

    def __call__(self, hidden_states: mx.array) -> mx.array:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(nn.Module):
    """Transformer encoder for CLIP."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = [
            CLIPEncoderLayer(config) for _ in range(config["num_hidden_layers"])
        ]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class CLIPVisionTransformer(nn.Module):
    """CLIP Vision Transformer."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = CLIPVisionEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(
            config["hidden_size"], eps=config["layer_norm_eps"]
        )

    def __call__(self, pixel_values: mx.array) -> mx.array:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.post_layernorm(hidden_states)
        return hidden_states


# CLIP ViT-L/14 configuration
CLIP_VIT_L_14_CONFIG = {
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "num_channels": 3,
    "image_size": 224,
    "patch_size": 14,
    "layer_norm_eps": 1e-5,
    "attention_dropout": 0.0,
    "initializer_range": 0.02,
}
