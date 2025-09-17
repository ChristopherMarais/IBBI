# src/ibbi/models/feature_extractors.py

"""
Untrained models for feature extraction.
"""

import numpy as np
import timm
import torch
from PIL import Image
from timm.data.config import resolve_model_data_config
from timm.data.transforms_factory import create_transform
from transformers import pipeline

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

        if not callable(self.transforms):
            raise TypeError("The transform object is not callable. Check the 'separate' argument in create_transform.")

        transformed_img = self.transforms(img)
        input_tensor = torch.as_tensor(transformed_img).unsqueeze(0).to(self.device)

        features = self.model.forward_features(input_tensor)  # type: ignore
        output = self.model.forward_head(features, pre_logits=True)  # type: ignore

        return output.detach()

    def get_classes(self) -> list[str]:
        raise NotImplementedError("This model is for feature extraction only and does not have classes.")


class HuggingFaceFeatureExtractor:
    """A wrapper class for using pretrained Hugging Face models for feature extraction."""

    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        device_id = 0 if self.device == "cuda" else -1
        self.feature_extractor = pipeline(task="image-feature-extraction", model=model_name, device=device_id)
        print(f"{model_name} model loaded successfully using the pipeline on device: {self.device}")

    def predict(self, image, **kwargs):
        raise NotImplementedError("This model is for feature extraction only and does not support prediction.")

    def extract_features(self, image, **kwargs):
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise TypeError("Image must be a PIL Image or a file path.")

        embedding = self.feature_extractor(img, **kwargs)
        global_features = np.array(embedding)[0, 0, :]
        return torch.tensor(global_features).to(self.device)

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
    return UntrainedFeatureExtractor(model_name="caformer_b36.sail_in22k_ft_in1k_384")


@register_model
def dinov3_vitl16_lvd1689m_features_model(pretrained: bool = True, **kwargs):
    return HuggingFaceFeatureExtractor(model_name="IBBI-bio/dinov3-vitl16-pretrain-lvd1689m")
