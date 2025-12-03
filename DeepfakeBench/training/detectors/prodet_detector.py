import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

from .base_detector import AbstractDetector
from detectors import DETECTOR
from metrics.base_metrics_class import calculate_metrics_for_train


@DETECTOR.register_module(module_name='prodet')
class ProDetDetector(AbstractDetector):
    def __init__(self, config=None, load_param: Union[bool, str] = False):
        super(ProDetDetector, self).__init__(config, load_param)
        self.config = config
        self.backbone = self.build_backbone(config)
        self.classifier = self.build_classifier(config)
        self.loss_func = self.build_loss(config)
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

    def build_backbone(self, config):
        """Build the backbone network for ProDet"""
        # ProDet typically uses a CNN backbone like ResNet or EfficientNet
        # Since we don't know the exact architecture, we'll create a flexible backbone
        # that can be loaded from the checkpoint

        # For now, create a simple CNN backbone that matches common deepfake detector architectures
        # This will be replaced when the actual weights are loaded
        backbone = nn.Sequential(
            # Input: 3x224x224
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Residual blocks (simplified)
            self._make_layer(64, 64, 2),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),

            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        return backbone

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """Create a residual layer"""
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def build_classifier(self, config):
        """Build the classifier head"""
        # ProDet typically outputs features that are then classified
        # We'll use a simple linear layer for binary classification
        return nn.Linear(512, 2)

    def build_loss(self, config):
        """Build the loss function"""
        return nn.CrossEntropyLoss()

    def features(self, data_dict: dict) -> torch.tensor:
        """Extract features from the backbone"""
        x = data_dict['image']
        features = self.backbone(x)
        return features

    def classifier(self, features: torch.tensor) -> torch.tensor:
        """Classify the features"""
        return self.classifier(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        """Compute losses"""
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)

        loss_dict = {'overall': loss}
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        """Compute training metrics"""
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def get_test_metrics(self):
        """Compute test metrics (placeholder)"""
        # This would typically compute metrics on accumulated predictions
        # For now, return empty dict as this is mainly used during training
        return {}

    def forward(self, data_dict: dict, inference=False) -> dict:
        """Forward pass"""
        # get the features by backbone
        features = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}

        return pred_dict