# src/ibbi/models/codino.py

"""
Co-DINO arthropod detector (Co-DETR with an EVA-02-L backbone, Objects365-pretrained, trained on the IBBI arthropod
detection corpus): a larger, more accurate alternative to the YOLO11x arthropod detector.

The model runs on plain PyTorch (`ibbi.models._codino`, no MMDetection / MMCV needed). It is large (348 M parameters,
images resized to at most 2048 x 1280 px) and is meant for a GPU; on a CPU one image takes minutes.
"""

from typing import Any

import numpy as np
import torch

from ..utils.hub import HF_ORG, download_from_hf_hub, get_model_config_from_hub
from ._common import ImageInput, empty_result, is_batch, load_image, resolve_device
from ._registry import register_model

REPO = "ibbi_codino_arthropod_detector"


class CoDINODetector:
    """Single-class Co-DINO detector ("arthropod").

    Args:
        weights_path (str): Path to the `model.safetensors` state dict.
        config (dict): The repository's `config.json` (architecture, preprocessing, defaults).
        device (str | None): Device; defaults to the best available.
        name (str | None): Registry name.

    `operating_conf` (0.65) is the confidence at which the detector makes at most 0.2 false alarms per image on the
    arthropod corpus validation sample.
    """

    is_species_level = False

    def __init__(self, weights_path: str, config: dict[str, Any], device: str | None = None, name: str | None = None):
        from safetensors.torch import load_file

        from ._codino import CoDINO

        self.config = config
        self.name = name or "codino_arthropod_detector"
        self.device = resolve_device(device)
        arch = config["architecture"]
        self.model = CoDINO(
            num_classes=1,
            num_queries=int(arch["num_queries"]),
            window_block_indexes=arch["window_block_indexes"],
            embed_dim=int(arch.get("embed_dim", 1024)),
            depth=int(arch.get("depth", 24)),
            num_heads=int(arch.get("num_heads", 16)),
        )
        state = load_file(weights_path)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Co-DINO weights do not match the model: missing {missing[:5]}, unexpected {unexpected[:5]}")
        self.model.eval().to(self.device)
        pre = config["preprocessing"]
        self.scale = tuple(int(v) for v in pre["scale"])  # (long side, short side)
        self.size_divisor = int(pre["size_divisor"])
        self.mean = np.asarray(pre["mean"], dtype=np.float32)
        self.std = np.asarray(pre["std"], dtype=np.float32)
        post = config["postprocessing"]
        self.soft_nms_iou = float(post["soft_nms_iou"])
        self.max_per_img = int(post["max_per_img"])
        self.classes = ["arthropod"]
        self.operating_conf = float(config.get("operating_conf", 0.65))
        self.inference_defaults = dict(config.get("inference_defaults", {"conf": 0.25, "max_det": 300}))
        self.benchmark_kwargs = dict(config.get("benchmark_inference", {"conf": 0.001}))
        print(f"{self.name} loaded on device: {self.device}")

    # ------------------------------------------------------------------------------------------------------------
    def _preprocess(self, img: np.ndarray):
        """Keep-ratio resize to fit (2048, 1280) as MMCV `imrescale` (bilinear), normalise, pad to a multiple of 32."""
        import cv2

        h, w = img.shape[:2]
        sf = min(max(self.scale) / max(h, w), min(self.scale) / min(h, w))
        nw, nh = int(w * sf + 0.5), int(h * sf + 0.5)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = (resized.astype(np.float32) - self.mean) / self.std
        d = self.size_divisor
        ph, pw = int(np.ceil(nh / d)) * d, int(np.ceil(nw / d)) * d
        canvas = np.zeros((ph, pw, 3), dtype=np.float32)
        canvas[:nh, :nw] = x
        tensor = torch.from_numpy(canvas).permute(2, 0, 1)[None].to(self.device)
        mask = torch.ones((1, ph, pw), device=self.device)
        mask[:, :nh, :nw] = 0
        return tensor, mask, (nh, nw), (nw / w, nh / h)

    @torch.no_grad()
    def _detect(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """All detections of one RGB image after soft-NMS: boxes [N, 4] (xyxy, original pixels), scores [N]."""
        from ._codino import soft_nms_linear

        x, mask, (nh, nw), (sx, sy) = self._preprocess(img)
        logits, coords = self.model(x, mask)
        scores = logits[0, :, 0].sigmoid()
        scores, order = scores.sort(descending=True)
        c = coords[0][order]
        boxes = torch.stack([c[:, 0] - c[:, 2] / 2, c[:, 1] - c[:, 3] / 2, c[:, 0] + c[:, 2] / 2, c[:, 1] + c[:, 3] / 2], -1)
        boxes[:, 0::2] = (boxes[:, 0::2] * nw).clamp(0, nw)
        boxes[:, 1::2] = (boxes[:, 1::2] * nh).clamp(0, nh)
        boxes = boxes / boxes.new_tensor([sx, sy, sx, sy])
        b, s = boxes.cpu().numpy(), scores.cpu().numpy()
        keep, new_scores = soft_nms_linear(b, s, iou_threshold=self.soft_nms_iou)
        keep, new_scores = keep[: self.max_per_img], new_scores[: self.max_per_img]
        return b[keep].astype(np.float32), new_scores

    # ------------------------------------------------------------------------------------------------------------
    def predict(self, image, conf: float | None = None, max_det: int | None = None, **kwargs):
        """Detects arthropods in one image or a list of images.

        Args:
            image (str | Path | np.ndarray | PIL.Image | list): Image(s) to process.
            conf (float | None): Confidence threshold (default 0.25; `operating_conf` = 0.65 for ≤ 0.2 false alarms
                per image on the validation sample).
            max_det (int | None): Maximum detections per image (default 300).

        Returns:
            dict | list[dict]: Per image, {"boxes" (xyxy), "scores", "labels", "class_ids"}.
        """
        conf = float(self.inference_defaults.get("conf", 0.25) if conf is None else conf)
        max_det = int(self.inference_defaults.get("max_det", 300) if max_det is None else max_det)
        images = list(image) if is_batch(image) else [image]
        outs = []
        for im in images:
            boxes, scores = self._detect(np.asarray(load_image(im)))
            sel = scores >= conf
            boxes, scores = boxes[sel][:max_det], scores[sel][:max_det]
            out = empty_result()
            out["boxes"] = boxes.tolist()
            out["scores"] = scores.tolist()
            out["labels"] = ["arthropod"] * len(scores)
            out["class_ids"] = [0] * len(scores)
            outs.append(out)
        return outs if is_batch(image) else outs[0]

    def predict_proba(self, images: list[ImageInput], **kwargs) -> np.ndarray:
        """Per image, the highest detection confidence: array [N, 1] (used by LIME / SHAP)."""
        out = np.zeros((len(images), 1), dtype=np.float32)
        for i, im in enumerate(images):
            _, s = self._detect(np.asarray(load_image(im)))
            out[i, 0] = float(s.max()) if len(s) else 0.0
        return out

    @torch.no_grad()
    def extract_features(self, image: ImageInput, **kwargs) -> torch.Tensor:
        """Image-level embedding: the mean of the EVA-02 backbone's last feature map, shape [1, 1024]."""
        x, _, _, _ = self._preprocess(np.asarray(load_image(image)))
        return self.model.backbone(x).mean(dim=(2, 3))

    def get_classes(self) -> list[str]:
        return self.classes


@register_model
def codino_arthropod_detector(pretrained: bool = True, device: str | None = None, revision: str | None = None, **kwargs):
    """Universal arthropod detector, Co-DINO with an EVA-02-L backbone (single class "arthropod").

    More accurate than `yolo11x_arthropod_detector` on the arthropod corpus test sets, at a much higher compute cost
    (GPU recommended). Weights: https://huggingface.co/IBBI-bio/ibbi_codino_arthropod_detector

    Args:
        pretrained (bool): Must be True (there is no generic checkpoint to fall back to).
        device (str | None): Device; defaults to the best available.
        revision (str | None): Hub revision of the weights.
    """
    if not pretrained:
        raise ValueError("codino_arthropod_detector is only available with its trained weights (pretrained=True).")
    repo_id = f"{HF_ORG}/{REPO}"
    cfg = get_model_config_from_hub(repo_id, revision=revision)
    path = download_from_hf_hub(repo_id, "model.safetensors", revision=revision)
    return CoDINODetector(path, cfg, device=device, name="codino_arthropod_detector")
