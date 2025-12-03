import os
import math
import datetime
import logging
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

from transformers import CLIPModel, AutoProcessor

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='huggingface_clip')
class HuggingFaceCLIPDetector(nn.Module):
    def __init__(self, config=None):
        super(HuggingFaceCLIPDetector, self).__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.head = nn.Linear(1024, 2)  # CLIP ViT-L/14 has 1024-dim features
        self.loss_func = nn.CrossEntropyLoss()
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

        # Initialize processor for inference
        self.processor = None

    def build_backbone(self, config):
        """Load CLIP model from HuggingFace Hub"""
        # Get model name from config or use default
        model_name = getattr(config, 'huggingface_model_name', 'openai/clip-vit-large-patch14')

        print(f"Loading HuggingFace model: {model_name}")
        model = CLIPModel.from_pretrained(model_name)

        # Apply SVD modifications if specified in config
        if hasattr(config, 'use_svd') and config.use_svd:
            r = getattr(config, 'svd_rank', 1024-1)  # Default to 1023 for ViT-L/14
            model.vision_model = apply_svd_residual_to_self_attn(model.vision_model, r=r)

        return model.vision_model

    def features(self, data_dict: dict) -> torch.tensor:
        """Extract features from CLIP vision model"""
        # Get image input - should be tensors from preprocess_face
        images = data_dict['image']

        # Convert tensor to PIL images for CLIP processing
        from torchvision.transforms import ToPILImage
        to_pil = ToPILImage()
        pil_images = []

        # Handle batch dimension
        if images.dim() == 4:  # [batch, channels, height, width]
            for i in range(images.shape[0]):
                # Denormalize from CLIP normalization for PIL conversion
                img_tensor = images[i].clone()
                # CLIP normalization constants
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(images.device)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(images.device)
                # Denormalize: img = img * std + mean
                img_tensor = img_tensor * std + mean
                # Clamp to [0, 1] range
                img_tensor = torch.clamp(img_tensor, 0, 1)
                pil_img = to_pil(img_tensor)
                pil_images.append(pil_img)
        else:
            # Single image case
            img_tensor = images.clone()
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(images.device)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(images.device)
            img_tensor = img_tensor * std + mean
            img_tensor = torch.clamp(img_tensor, 0, 1)
            pil_images = [to_pil(img_tensor)]

        # Initialize processor if not done yet
        if self.processor is None:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                getattr(self.config, 'huggingface_model_name', 'openai/clip-vit-large-patch14')
            )

        # Process images
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(images.device) for k, v in inputs.items()}

        # Get features
        outputs = self.backbone(**inputs)
        feat = outputs.pooler_output

        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}

        return pred_dict


# SVD-related functions (copied from effort_detector.py)
class SVDResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super(SVDResidualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r  # Number of top singular values to exclude

        # Original weights (fixed)
        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def compute_current_weight(self):
        if self.S_residual is not None:
            return self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        else:
            return self.weight_main

    def forward(self, x):
        if hasattr(self, 'U_residual') and hasattr(self, 'V_residual') and self.S_residual is not None:
            # Reconstruct the residual weight
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            # Total weight is the fixed main weight plus the residual
            weight = self.weight_main + residual_weight
        else:
            # If residual components are not set, use only the main weight
            weight = self.weight_main

        return F.linear(x, weight, self.bias)


def apply_svd_residual_to_self_attn(model, r):
    """Apply SVD residual modifications to self-attention layers"""
    for name, module in model.named_children():
        if 'self_attn' in name:
            # Replace nn.Linear layers in this module
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    # Get parent module within self_attn
                    parent_module = module
                    sub_module_names = sub_name.split('.')
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)
                    # Replace the nn.Linear layer with SVDResidualLinear
                    setattr(parent_module, sub_module_names[-1], replace_with_svd_residual(sub_module, r))
        else:
            # Recursively apply to child modules
            apply_svd_residual_to_self_attn(module, r)
    # After replacing, set requires_grad for residual components
    for param_name, param in model.named_parameters():
        if any(x in param_name for x in ['S_residual', 'U_residual', 'V_residual']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def replace_with_svd_residual(module, r):
    """Replace a linear layer with SVD residual version"""
    if isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None

        # Create SVDResidualLinear module
        new_module = SVDResidualLinear(in_features, out_features, r, bias=bias, init_weight=module.weight.data.clone())

        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)

        new_module.weight_original_fnorm = torch.norm(module.weight.data, p='fro')

        # Perform SVD on the original weight
        U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)

        # Determine r based on the rank of the weight matrix
        r = min(r, len(S))  # Ensure r does not exceed the number of singular values

        # Keep top r singular components (main weight)
        U_r = U[:, :r]      # Shape: (out_features, r)
        S_r = S[:r]         # Shape: (r,)
        Vh_r = Vh[:r, :]    # Shape: (r, in_features)

        # Reconstruct the main weight (fixed)
        weight_main = U_r @ torch.diag(S_r) @ Vh_r

        # Calculate the frobenius norm of main weight
        new_module.weight_main_fnorm = torch.norm(weight_main.data, p='fro')

        # Set the main weight
        new_module.weight_main.data.copy_(weight_main)

        # Residual components (trainable)
        U_residual = U[:, r:]    # Shape: (out_features, n - r)
        S_residual = S[r:]       # Shape: (n - r,)
        Vh_residual = Vh[r:, :]  # Shape: (n - r, in_features)

        if len(S_residual) > 0:
            new_module.S_residual = nn.Parameter(S_residual.clone())
            new_module.U_residual = nn.Parameter(U_residual.clone())
            new_module.V_residual = nn.Parameter(Vh_residual.clone())

            new_module.S_r = nn.Parameter(S_r.clone(), requires_grad=False)
            new_module.U_r = nn.Parameter(U_r.clone(), requires_grad=False)
            new_module.V_r = nn.Parameter(Vh_r.clone(), requires_grad=False)
        else:
            new_module.S_residual = None
            new_module.U_residual = None
            new_module.V_residual = None

            new_module.S_r = None
            new_module.U_r = None
            new_module.V_r = None

        return new_module
    else:
        return module