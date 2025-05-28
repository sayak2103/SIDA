# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
SAM Model Builder

This module provides functions to build different variants of the Segment Anything Model (SAM)
using Vision Transformers (ViT) as the image encoder backbone. It supports 'huge', 'large', and 'base' variants.

Key Functions:
    - build_sam_vit_h: Build SAM with ViT-Huge backbone.
    - build_sam_vit_l: Build SAM with ViT-Large backbone.
    - build_sam_vit_b: Build SAM with ViT-Base backbone.
    - _build_sam: Internal function to construct a SAM model with given parameters.

References:
    - .modeling: Imports core SAM components (ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer).
"""
from functools import partial

import torch

# Import core SAM components from the local modeling.py file
from .modeling import (
    ImageEncoderViT,  # Vision Transformer encoder for images
    MaskDecoder,      # Decoder to produce segmentation masks
    PromptEncoder,    # Encodes prompts (points, boxes, text, etc.)
    Sam,              # The main SAM model class
    TwoWayTransformer # Transformer block used in the mask decoder
)


def build_sam_vit_h(checkpoint=None):
    """
    Build a SAM model with a ViT-Huge backbone.

    Args:
        checkpoint (str, optional): Path to a model checkpoint to load weights from.

    Returns:
        Sam: An instance of the SAM model with ViT-Huge backbone.
    """
    return _build_sam(
        encoder_embed_dim=1280,         # Embedding dimension for ViT-H
        encoder_depth=32,               # Number of transformer layers
        encoder_num_heads=16,           # Number of attention heads
        encoder_global_attn_indexes=[7, 15, 23, 31],  # Layers with global attention
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    """
    Build a SAM model with a ViT-Large backbone.

    Args:
        checkpoint (str, optional): Path to a model checkpoint to load weights from.

    Returns:
        Sam: An instance of the SAM model with ViT-Large backbone.
    """
    return _build_sam(
        encoder_embed_dim=1024,         # Embedding dimension for ViT-L
        encoder_depth=24,               # Number of transformer layers
        encoder_num_heads=16,           # Number of attention heads
        encoder_global_attn_indexes=[5, 11, 17, 23],  # Layers with global attention
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    """
    Build a SAM model with a ViT-Base backbone.

    Args:
        checkpoint (str, optional): Path to a model checkpoint to load weights from.

    Returns:
        Sam: An instance of the SAM model with ViT-Base backbone.
    """
    return _build_sam(
        encoder_embed_dim=768,          # Embedding dimension for ViT-B
        encoder_depth=12,               # Number of transformer layers
        encoder_num_heads=12,           # Number of attention heads
        encoder_global_attn_indexes=[2, 5, 8, 11],  # Layers with global attention
        checkpoint=checkpoint,
    )

# Registry for easy lookup of SAM model builders by name
sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    """
    Internal function to construct a SAM model with specified transformer parameters.

    Args:
        encoder_embed_dim (int): Embedding dimension for the ViT encoder.
        encoder_depth (int): Number of transformer layers.
        encoder_num_heads (int): Number of attention heads.
        encoder_global_attn_indexes (list): Indices of layers with global attention.
        checkpoint (str, optional): Path to a model checkpoint to load weights from.

    Returns:
        Sam: An instance of the SAM model.
    """
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    # Instantiate the SAM model with the specified ViT backbone and components
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        # Normalization mean and standard deviation (ImageNet statistics)
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict, strict=False)
    return sam
