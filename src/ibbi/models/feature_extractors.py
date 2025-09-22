# src/ibbi/models/feature_extractors.py

"""
This module provides models for feature extraction, which are designed to convert
images into dense numerical representations (embeddings) without being trained for a
specific classification or detection task. These embeddings are useful for a variety
of downstream applications, such as clustering, similarity search, or as input
features for other machine learning models.

The module includes two primary wrapper classes:
- `UntrainedFeatureExtractor`: For using pretrained models from the `timm` library.
- `HuggingFaceFeatureExtractor`: For using pretrained models from the Hugging Face Hub
  via the `transformers` pipeline.

Additionally, it provides several factory functions, decorated with `@register_model`,
to easily instantiate specific, recommended feature extraction models.
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
    """A wrapper class for using pretrained `timm` models for feature extraction.

    This class provides a standardized interface for loading and using models from the
    PyTorch Image Models (`timm`) library for the purpose of feature extraction.
    It handles model loading, device placement, and the necessary image transformations.

    Args:
        model_name (str): The name of the `timm` model to be loaded.
    """

    def __init__(self, model_name: str):
        """Initializes the UntrainedFeatureExtractor.

        Args:
            model_name (str): The name of the model to load from the `timm` library.
                              The model is always loaded with pretrained weights.
        """
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.eval().to(self.device)
        self.data_config = resolve_model_data_config(self.model)
        self.transforms = create_transform(**self.data_config, is_training=False)
        print(f"{model_name} model loaded on device: {self.device}")

    def predict(self, image, **kwargs):
        """This method is not implemented for this class.

        Raises:
            NotImplementedError: This model is for feature extraction only.
        """
        raise NotImplementedError("This model is for feature extraction only and does not support prediction.")

    def extract_features(self, image, **kwargs):
        """Extracts deep feature embeddings from an image.

        Args:
            image (Union[str, Image.Image]): The input image, which can be a file path or a PIL Image object.
            **kwargs: Additional keyword arguments (not used in this implementation but included for API consistency).

        Returns:
            torch.Tensor: A tensor containing the extracted feature embedding.
        """
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
        """This method is not applicable to feature extraction models.

        Raises:
            NotImplementedError: Feature extractors do not have a fixed set of classes.
        """
        raise NotImplementedError("This model is for feature extraction only and does not have classes.")


class HuggingFaceFeatureExtractor:
    """A wrapper class for using pretrained Hugging Face models for feature extraction.

    This class uses the `transformers` pipeline to provide an easy-to-use interface for
    extracting features from models hosted on the Hugging Face Hub.

    Args:
        model_name (str): The name of the Hugging Face model to be loaded.
    """

    def __init__(self, model_name: str):
        """Initializes the HuggingFaceFeatureExtractor.

        Args:
            model_name (str): The model identifier from the Hugging Face Hub.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        device_id = 0 if self.device == "cuda" else -1
        self.feature_extractor = pipeline(task="image-feature-extraction", model=model_name, device=device_id)
        print(f"{model_name} model loaded successfully using the pipeline on device: {self.device}")

    def predict(self, image, **kwargs):
        """This method is not implemented for this class.

        Raises:
            NotImplementedError: This model is for feature extraction only.
        """
        raise NotImplementedError("This model is for feature extraction only and does not support prediction.")

    def extract_features(self, image, **kwargs):
        """Extracts deep feature embeddings from an image.

        Args:
            image (Union[str, Image.Image]): The input image, which can be a file path or a PIL Image object.
            **kwargs: Additional keyword arguments to be passed to the feature extraction pipeline.

        Returns:
            torch.Tensor: A tensor containing the extracted feature embedding.
        """
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
        """This method is not applicable to feature extraction models.

        Raises:
            NotImplementedError: Feature extractors do not have a fixed set of classes.
        """
        raise NotImplementedError("This model is for feature extraction only and does not have classes.")


@register_model
def dinov2_vitl14_lvd142m_features_model(pretrained: bool = True, **kwargs):
    """Factory function for the DINOv2 ViT-L/14 feature extractor.

    Args:
        pretrained (bool, optional): This argument is ignored as the model is always pretrained. Defaults to True.
        **kwargs: Additional keyword arguments (not used).

    Returns:
        UntrainedFeatureExtractor: An instance of the DINOv2 ViT-L/14 model.
    """
    return UntrainedFeatureExtractor(model_name="vit_large_patch14_dinov2.lvd142m")


@register_model
def eva02_base_patch14_224_mim_in22k_features_model(pretrained: bool = True, **kwargs):
    """Factory function for the EVA-02 Base feature extractor.

    Args:
        pretrained (bool, optional): This argument is ignored as the model is always pretrained. Defaults to True.
        **kwargs: Additional keyword arguments (not used).

    Returns:
        UntrainedFeatureExtractor: An instance of the EVA-02 Base model.
    """
    return UntrainedFeatureExtractor(model_name="eva02_base_patch14_224.mim_in22k")


@register_model
def convformer_b36_features_model(pretrained: bool = True, **kwargs):
    """Factory function for the ConvFormer-B36 feature extractor.

    Args:
        pretrained (bool, optional): This argument is ignored as the model is always pretrained. Defaults to True.
        **kwargs: Additional keyword arguments (not used).

    Returns:
        UntrainedFeatureExtractor: An instance of the ConvFormer-B36 model.
    """
    return UntrainedFeatureExtractor(model_name="caformer_b36.sail_in22k_ft_in1k_384")


@register_model
def dinov3_vitl16_lvd1689m_features_model(pretrained: bool = True, **kwargs):
    """Factory function for the DINOv3 ViT-L/16 feature extractor.

    Args:
        pretrained (bool, optional): This argument is ignored as the model is always pretrained. Defaults to True.
        **kwargs: Additional keyword arguments (not used).

    Returns:
        HuggingFaceFeatureExtractor: An instance of the DINOv3 ViT-L/16 model.
    """
    return HuggingFaceFeatureExtractor(model_name="IBBI-bio/dinov3-vitl16-pretrain-lvd1689m")
