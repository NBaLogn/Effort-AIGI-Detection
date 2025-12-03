import torch
import torch.nn as nn
from transformers import CLIPModel, AutoProcessor
from torchvision.transforms import ToPILImage

from detectors import DETECTOR


@DETECTOR.register_module(module_name='huggingface_clip')
class HuggingFaceCLIPDetector(nn.Module):
    """Simplified HuggingFace CLIP detector for deepfake detection"""

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.model_name = getattr(config, 'huggingface_model_name', 'openai/clip-vit-base-patch32')

        # Load CLIP model and processor
        print(f"Loading {self.model_name}...")
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        # Simple classification head
        self.classifier = nn.Linear(768 if 'base' in self.model_name else 1024, 2)
        self.loss_func = nn.CrossEntropyLoss()

    def features(self, data_dict):
        """Extract CLIP features from preprocessed images"""
        images = data_dict['image']

        # Convert normalized tensors back to PIL images for CLIP
        to_pil = ToPILImage()
        pil_images = []

        # Handle batch dimension
        if images.dim() == 4:
            for i in range(images.shape[0]):
                # Denormalize CLIP normalization
                img = images[i] * torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(images.device)
                img = img + torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(images.device)
                img = torch.clamp(img, 0, 1)
                pil_images.append(to_pil(img))
        else:
            img = images * torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(images.device)
            img = img + torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(images.device)
            img = torch.clamp(img, 0, 1)
            pil_images = [to_pil(img)]

        # Process with CLIP
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(images.device) for k, v in inputs.items()}

        outputs = self.model.vision_model(**inputs)
        return outputs.pooler_output

    def get_losses(self, data_dict, pred_dict):
        """Compute loss"""
        labels = data_dict['label']
        preds = pred_dict['cls']
        return {'overall': self.loss_func(preds, labels)}

    def get_train_metrics(self, data_dict, pred_dict):
        """Compute training metrics (simplified)"""
        return {'acc': 0.5, 'auc': 0.5, 'eer': 0.5, 'ap': 0.5}

    def forward(self, data_dict, inference=False):
        """Forward pass"""
        features = self.features(data_dict)
        logits = self.classifier(features)
        probs = torch.softmax(logits, dim=1)

        return {
            'cls': logits,
            'prob': probs[:, 1],
            'feat': features
        }