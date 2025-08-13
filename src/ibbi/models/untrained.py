# src/ibbi/models/untrained.py

"""
Untrained models for feature extraction.
"""

import timm
import torch
from PIL import Image
from timm.data.config import resolve_model_data_config
from timm.data.transforms_factory import create_transform

from ._registry import register_model


class UntrainedFeatureExtractor:
    """A wrapper class for using pretrained timm models for feature extraction."""

    def __init__(self, model_name: str):
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.eval().to(self.device)
        self.data_config = resolve_model_data_config(self.model)
        self.transforms = create_transform(**self.data_config, is_training=False)
        print(f"{model_name} model loaded on device: {self.device}")

    def predict(self, image, **kwargs):
        raise NotImplementedError("This model is for feature extraction only and does not support prediction.")

    def extract_features(self, image, **kwargs):
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise TypeError("Image must be a PIL Image or a file path.")

        # Prepare the input tensor. The create_transform function has a complex
        # return type (Union) that pyright struggles to resolve, so we ignore the type error here.
        input_tensor = self.transforms(img).unsqueeze(0).to(self.device)  # type: ignore

        # Pyright cannot statically determine that the dynamically created timm model has
        # forward_features and forward_head methods, so we ignore the type errors.
        features = self.model.forward_features(input_tensor)  # type: ignore
        output = self.model.forward_head(features, pre_logits=True)  # type: ignore

        return output.detach().cpu().numpy()

    def get_classes(self) -> list[str]:
        raise NotImplementedError("This model is for feature extraction only and does not have classes.")


@register_model
def dinov2_vitl14_lvd142m_features_model(pretrained: bool = True, **kwargs):
    return UntrainedFeatureExtractor(model_name="vit_large_patch14_dinov2.lvd142m")


@register_model
def eva02_base_patch14_224_mim_in22k_features_model(pretrained: bool = True, **kwargs):
    return UntrainedFeatureExtractor(model_name="eva02_base_patch14_224.mim_in22k")


@register_model
def convformer_b36_features_model(pretrained: bool = True, **kwargs):
    return UntrainedFeatureExtractor(model_name="caformer_b36")
