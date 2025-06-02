from ultralytics import YOLO

from ..utils.hub import download_from_hf_hub
from ._registry import register_model


@register_model
def yolov10x_beetle_detector(pretrained: bool = False, **kwargs):
    """
    Factory function for the YOLOv10x beetle detector.
    """
    # For YOLO models, the model structure and weights are in the same .pt file.
    # For this package, we assume pretrained is the primary use case.
    if pretrained:
        # This is where we link to our Hugging Face model
        repo_id = "ChristopherMarais/ibbi_yolov10_od_20250601"
        filename = "model.pt"

        # Download the weights file and get its local path
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)

        # Load the model directly from the downloaded weights file
        model = YOLO(local_weights_path)
    else:
        # Load a base, non-fine-tuned model from Ultralytics
        model = YOLO("yolov10x.pt")

    return model
