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
  via the `transformers` library.

Additionally, it provides several factory functions, decorated with `@register_model`,
to easily instantiate specific, recommended feature extraction models.
"""

import timm
import torch
from PIL import Image
from timm.data.config import resolve_model_data_config
from timm.data.transforms_factory import create_transform
from transformers import AutoModel, AutoProcessor

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

    def extract_features(self, image, pool: bool = True, **kwargs):
        """Extracts deep features from an image, offering both pooled embeddings and raw feature maps.

        This method processes an image and returns either a single, flat feature vector (embedding)
        or the raw feature map from the model's backbone. The pooled embedding is
        useful for tasks like classification or clustering, while the raw feature map retains
        spatial information required for tasks like object detection.

        Args:
            image (Union[str, Image.Image]): The input image, which can be a file path or a PIL Image object.
            pool (bool, optional): If True (the default), returns a flat, pooled feature vector (embedding)
                                   by passing the features through the model's head. If False, returns
                                   the raw, un-pooled feature map directly from the model's backbone.
                                   Defaults to True.
            **kwargs: Additional keyword arguments (not used in this implementation but included for API consistency).

        Returns:
            torch.Tensor: A tensor containing the extracted features. Its shape will be either a
                          flat vector (if `pool=True`) or a multi-dimensional feature map (if `pool=False`).
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

        if pool:
            # Current behavior: return the flat embedding vector
            output = self.model.forward_head(features, pre_logits=True)  # type: ignore
            return output.detach()
        else:
            # New behavior: return the raw feature map.
            # For ViT-style models, this is a 3D tensor [batch, num_tokens, dim].
            # We remove the [CLS] token as it is for classification.
            if (
                features.dim() == 3
                and hasattr(self.model, "num_prefix_tokens")
                and isinstance(self.model.num_prefix_tokens, int)
                and self.model.num_prefix_tokens > 0
            ):
                return features[:, self.model.num_prefix_tokens :, :].detach()
            return features.detach()

    def get_classes(self) -> list[str]:
        """This method is not applicable to feature extraction models.

        Raises:
            NotImplementedError: Feature extractors do not have a fixed set of classes.
        """
        raise NotImplementedError("This model is for feature extraction only and does not have classes.")


class HuggingFaceFeatureExtractor:
    """A wrapper class for using pretrained Hugging Face models for feature extraction.

    This class uses the `transformers` library to provide a standardized interface for
    extracting features from models hosted on the Hugging Face Hub. It loads the model
    and processor directly to allow flexible access to different types of outputs.

    Args:
        model_name (str): The name of the Hugging Face model to be loaded.
    """

    def __init__(self, model_name: str):
        """Initializes the HuggingFaceFeatureExtractor.

        Args:
            model_name (str): The model identifier from the Hugging Face Hub.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        print(f"{model_name} model loaded successfully on device: {self.device}")

    def predict(self, image, **kwargs):
        """This method is not implemented for this class.

        Raises:
            NotImplementedError: This model is for feature extraction only.
        """
        raise NotImplementedError("This model is for feature extraction only and does not support prediction.")

    def extract_features(self, image, pool: bool = True, **kwargs):
        """Extracts deep features from an image, offering both pooled embeddings and raw patch embeddings.

        This method processes an image and returns either a single, flat feature vector (embedding)
        representing the whole image, or the full sequence of patch embeddings (feature map) from
        the transformer's last hidden state. The pooled embedding is useful for classification, while
        the sequence of patch embeddings retains spatial information for detection tasks.

        Args:
            image (Union[str, Image.Image]): The input image, which can be a file path or a PIL Image object.
            pool (bool, optional): If True (the default), returns the `pooler_output`, which is a
                                   single feature vector summarizing the image (often derived from the
                                   [CLS] token). If False, returns the `last_hidden_state` (excluding
                                   the CLS token), which is the sequence of patch embeddings.
                                   Defaults to True.
            **kwargs: Additional keyword arguments (not used in this implementation).

        Returns:
            torch.Tensor: A tensor containing the extracted features. Its shape will be a
                          2D tensor (batch, features) if `pool=True` or a 3D tensor
                          (batch, num_patches, features) if `pool=False`.
        """
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise TypeError("Image must be a PIL Image or a file path.")

        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # --- ✅ MODIFICATION: Conditionally return pooled or raw features ---
        if pool:
            # Current behavior: return the pooled output, typically from the [CLS] token
            return outputs.pooler_output.detach()
        else:
            # New behavior: return the full sequence of patch embeddings (raw features)
            # We slice [:, 1:, :] to remove the [CLS] token, which is not needed for detection.
            return outputs.last_hidden_state[:, 1:, :].detach()

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
