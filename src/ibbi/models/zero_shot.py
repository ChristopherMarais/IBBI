# src/ibbi/models/zero_shot.py

"""
Zero-shot (open-vocabulary) detectors: they find objects described by text prompts, without training on beetles.

One model per open-vocabulary family, with the released weights of their authors:

    grounding_dino_zero_shot_detector  Grounding DINO base (grounded transformer), IDEA-Research/grounding-dino-base
    owlv2_zero_shot_detector           OWLv2 large ensemble (CLIP-style), google/owlv2-large-patch14-ensemble
    yoloworld_zero_shot_detector       YOLO-World v2-X (real-time YOLO), Ultralytics yolov8x-worldv2.pt
    sam3_zero_shot_detector            SAM 3 (segment anything with concepts), facebook/sam3 (gated: accept the licence
                                       on the Hub and log in with `hf auth login`)

Defaults (prompt set, sliding-window tiling) are the settings each model scored best with on the validation sample of
the IBBI arthropod detection corpus (1,813 images across lab, trap, camera-trap and field imagery); they were never
tuned on test data or on the beetle benchmark. With `tile=1024` the image is processed in 1024 px windows with 20%
overlap plus the whole image, merged with non-maximum suppression, which helps on small specimens and large images.
"""

from typing import Any

import numpy as np
import torch
from PIL import Image

from ._common import ImageInput, empty_result, is_batch, load_image, nms, resolve_device
from ._registry import register_model

THIRTEEN_TAXA = ["insect", "spider", "beetle", "moth", "fly", "bee", "ant", "wasp", "butterfly", "caterpillar", "mite", "springtail", "bug"]


class ZeroShotDetector:
    """Base class: prompt handling, tiling, NMS and the common output format.

    Args:
        prompts (list[str]): Default text prompts (one class per prompt).
        tile (int): Sliding-window size in px; 0 processes the whole image only.
        device (str | None): Device; defaults to the best available.
        operating_conf (float): Confidence of the model's operating point on the detector-corpus validation sample
            (at most 0.2 false alarms per image).
        name (str): Registry name.
    """

    is_species_level = False
    nms_iou = 0.6

    def __init__(self, prompts: list[str], tile: int = 0, device: str | None = None, operating_conf: float = 0.5, name: str = ""):
        self.device = resolve_device(device)
        self.prompts = list(prompts)
        self.tile = int(tile)
        self.operating_conf = float(operating_conf)
        self.name = name or type(self).__name__
        self.score_floor = 0.05
        self.benchmark_kwargs = {"conf": 0.02}

    # -- prompts --------------------------------------------------------------------------------------------------
    def set_classes(self, classes: list[str] | str) -> None:
        """Sets the text prompts. A string is split on "." (e.g. "beetle . insect")."""
        if isinstance(classes, str):
            classes = [c.strip() for c in classes.split(".") if c.strip()]
        self.prompts = list(classes)

    def get_classes(self) -> list[str]:
        return list(self.prompts)

    # -- inference ------------------------------------------------------------------------------------------------
    def _detect(self, img: Image.Image, conf: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Whole-image detection: (boxes [N,4] xyxy, scores [N], prompt index [N])."""
        raise NotImplementedError

    def _detect_tiled(self, img: Image.Image, conf: float, tile: int, overlap: float = 0.2):
        W, H = img.size
        if not tile or max(W, H) <= tile:
            return self._detect(img, conf)
        step = int(tile * (1 - overlap))
        xs = sorted({max(0, v) for v in [*range(0, max(W - tile, 0) + 1, step), W - tile]})
        ys = sorted({max(0, v) for v in [*range(0, max(H - tile, 0) + 1, step), H - tile]})
        B, S, L = [], [], []
        for y in ys:
            for x in xs:
                b, s, lab = self._detect(img.crop((x, y, min(x + tile, W), min(y + tile, H))), conf)
                if len(b):
                    b = b.copy()
                    b[:, [0, 2]] += x
                    b[:, [1, 3]] += y
                    B.append(b), S.append(s), L.append(lab)
        b, s, lab = self._detect(img, conf)  # plus the whole image, for large specimens
        B.append(b), S.append(s), L.append(lab)
        b, s, lab = np.concatenate(B), np.concatenate(S), np.concatenate(L)
        k = nms(b, s, self.nms_iou)
        return b[k], s[k], lab[k]

    def predict(self, image, text_prompt: str | list[str] | None = None, conf: float | None = None, tile: int | None = None, **kwargs):
        """Detects the prompted objects in one image or a list of images.

        Args:
            image: Path, URL, RGB array or PIL image, or a list of them.
            text_prompt (str | list[str] | None): New prompts (sets the classes). Defaults to the current prompts.
            conf (float | None): Minimum score. Defaults to 0.05.
            tile (int | None): Sliding-window size (0 = whole image). Defaults to the model's chosen setting.

        Returns:
            dict | list[dict]: Per image, {"boxes" (xyxy), "scores", "labels" (the matching prompt)}.
        """
        for k in ("verbose", "include_full_probabilities", "box_threshold", "text_threshold"):
            kwargs.pop(k, None)
        if text_prompt:
            self.set_classes(text_prompt)
        conf = self.score_floor if conf is None else float(conf)
        tile = self.tile if tile is None else int(tile)
        images = list(image) if is_batch(image) else [image]
        outs = []
        for im in images:
            img = load_image(im)
            W, H = img.size
            with torch.inference_mode():
                b, s, lab = self._detect_tiled(img, conf, tile)
            if len(b):
                b[:, [0, 2]] = b[:, [0, 2]].clip(0, W)
                b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
            o = empty_result()
            o["boxes"], o["scores"] = b.tolist(), s.tolist()
            o["labels"] = [self.prompts[int(i)] if 0 <= int(i) < len(self.prompts) else "object" for i in lab]
            outs.append(o)
        return outs if is_batch(image) else outs[0]

    def predict_proba(self, images: list[ImageInput], **kwargs) -> np.ndarray:
        """Per image, the highest score for each prompt: array [N, n_prompts]."""
        out = np.zeros((len(images), len(self.prompts)), dtype=np.float32)
        for i, r in enumerate(self.predict(list(images), conf=kwargs.get("conf", 0.01), tile=kwargs.get("tile", 0))):
            for lab, s in zip(r["labels"], r["scores"]):
                if lab in self.prompts:
                    j = self.prompts.index(lab)
                    out[i, j] = max(out[i, j], s)
        return out

    def extract_features(self, image: ImageInput, **kwargs) -> torch.Tensor | None:
        raise NotImplementedError(f"{self.name} does not provide image embeddings.")


class GroundingDINOModel(ZeroShotDetector):
    """Grounding DINO (transformers)."""

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-base", **kwargs):
        super().__init__(**kwargs)
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()
        print(f"{self.name} loaded on device: {self.device}")

    def _detect(self, img, conf):
        W, H = img.size
        text = " . ".join(p.lower() for p in self.prompts) + " ."
        inp = self.processor(images=img, text=text, return_tensors="pt").to(self.device)
        out = self.model(**inp)
        res = self.processor.post_process_grounded_object_detection(out, inp.input_ids, threshold=conf, text_threshold=conf, target_sizes=[(H, W)])[0]
        lows = [p.lower() for p in self.prompts]
        labels = res.get("text_labels", res.get("labels", []))
        idx = np.array(
            [lows.index(str(t).strip().lower()) if str(t).strip().lower() in lows else _best_prompt(str(t), lows) for t in labels], dtype=int
        )
        b, s = res["boxes"].cpu().numpy(), res["scores"].cpu().numpy()
        k = nms(b, s, self.nms_iou)
        return b[k], s[k], idx[k] if len(idx) else idx

    def extract_features(self, image: ImageInput, text_prompt: str = "insect", **kwargs) -> torch.Tensor | None:
        """Mean of the last vision-encoder state, conditioned on `text_prompt`."""
        img = load_image(image)
        inp = self.processor(images=img, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model(**inp)
        v = getattr(out, "encoder_last_hidden_state_vision", None)
        return v.mean(dim=1).detach() if v is not None else None


def _best_prompt(text: str, prompts: list[str]) -> int:
    """Grounding DINO can return a span joining several prompt words; pick the first prompt it mentions."""
    for i, p in enumerate(prompts):
        if p in text.lower():
            return i
    return -1


class OWLv2Model(ZeroShotDetector):
    """OWLv2 (transformers)."""

    def __init__(self, model_id: str = "google/owlv2-large-patch14-ensemble", **kwargs):
        super().__init__(**kwargs)
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device).eval()
        print(f"{self.name} loaded on device: {self.device}")

    def _detect(self, img, conf):
        W, H = img.size
        inp = self.processor(text=[self.prompts], images=img, return_tensors="pt").to(self.device)
        out = self.model(**inp)
        side = max(W, H)  # OWLv2 pads to a square: boxes are in the padded frame
        res = self.processor.post_process_grounded_object_detection(out, threshold=conf, target_sizes=[(side, side)])[0]
        b, s, lab = res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), res["labels"].cpu().numpy().astype(int)
        k = nms(b, s, self.nms_iou)
        return b[k], s[k], lab[k]


class YOLOWorldModel(ZeroShotDetector):
    """YOLO-World v2-X (Ultralytics). The text encoder is installed by Ultralytics on first use."""

    def __init__(self, weights: str = "yolov8x-worldv2.pt", imgsz: int = 1024, **kwargs):
        super().__init__(**kwargs)
        from ultralytics import YOLOWorld

        self.model = YOLOWorld(weights)
        self.imgsz = imgsz
        self._set_on_model(self.prompts)
        print(f"{self.name} loaded on device: {self.device}")

    def _set_on_model(self, prompts):
        if hasattr(self.model, "clip_model") and self.model.clip_model is not None:
            self.model.clip_model.eval()
        with torch.no_grad():
            self.model.set_classes(list(prompts))

    def set_classes(self, classes):
        super().set_classes(classes)
        self._set_on_model(self.prompts)

    def _detect(self, img, conf):
        r = self.model.predict(img, conf=conf, iou=self.nms_iou, max_det=300, imgsz=self.imgsz, agnostic_nms=True, verbose=False, device=self.device)[
            0
        ]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)

    def extract_features(self, image: ImageInput, **kwargs) -> torch.Tensor | None:
        feats = self.model.embed(load_image(image), verbose=False)
        return feats[0] if feats else None


class SAM3Model(ZeroShotDetector):
    """SAM 3 (transformers). Boxes come from its instance predictions, one prompt at a time."""

    def __init__(self, model_id: str = "facebook/sam3", **kwargs):
        super().__init__(**kwargs)
        try:
            from transformers import Sam3Model, Sam3Processor
        except ImportError as e:  # pragma: no cover
            raise ImportError("SAM 3 needs transformers >= 5.0.") from e
        try:
            self.processor = Sam3Processor.from_pretrained(model_id)
            self.model = Sam3Model.from_pretrained(model_id).to(self.device).eval()
        except OSError as e:
            raise OSError(
                f"Could not load '{model_id}'. SAM 3 is gated: accept its licence at https://huggingface.co/{model_id} "
                "and log in with `hf auth login`."
            ) from e
        print(f"{self.name} loaded on device: {self.device}")

    def _detect(self, img, conf):
        W, H = img.size
        B, S, L = [], [], []
        # post-process at <= 1536 px and scale back: full-resolution masks are only needed for boxes and can need
        # >100 GB on very large scanner images; SAM 3 itself sees a 1008 px input
        sc = min(1.0, 1536 / max(H, W))
        for i, prompt in enumerate(self.prompts):
            inp = self.processor(images=img, text=prompt, return_tensors="pt").to(self.device)
            out = self.model(**inp)
            res = self.processor.post_process_instance_segmentation(
                out, threshold=conf, mask_threshold=0.5, target_sizes=[(max(1, round(H * sc)), max(1, round(W * sc)))]
            )[0]
            if len(res["scores"]):
                B.append(res["boxes"].cpu().numpy() / sc)
                S.append(res["scores"].cpu().numpy())
                L.append(np.full(len(res["scores"]), i, dtype=int))
        if not B:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
        b, s, lab = np.concatenate(B), np.concatenate(S), np.concatenate(L)
        k = nms(b, s, self.nms_iou)
        return b[k], s[k], lab[k]


# ----------------------------------------------------------------------------------------------------------------------
def _kw(kwargs: dict[str, Any], prompts: list[str], tile: int, op: float, name: str) -> dict[str, Any]:
    return {
        "prompts": kwargs.pop("prompts", prompts),
        "tile": kwargs.pop("tile", tile),
        "device": kwargs.pop("device", None),
        "operating_conf": kwargs.pop("operating_conf", op),
        "name": name,
    }


@register_model
def grounding_dino_zero_shot_detector(pretrained: bool = True, **kwargs):
    """Grounding DINO base. Default prompts: 13 arthropod taxa; tiling 1024 px (best setting on the validation sample).

    Keyword args: `prompts` (list[str]), `tile` (int, 0 = off), `device`, `model_id`.
    """
    model_id = kwargs.pop("model_id", "IDEA-Research/grounding-dino-base")
    return GroundingDINOModel(model_id=model_id, **_kw(kwargs, THIRTEEN_TAXA, 1024, 0.6, "grounding_dino_zero_shot_detector"))


@register_model
def owlv2_zero_shot_detector(pretrained: bool = True, **kwargs):
    """OWLv2 large ensemble. Default prompt "a photo of an insect"; tiling 1024 px (best setting on the validation sample).

    Keyword args: `prompts`, `tile`, `device`, `model_id`.
    """
    model_id = kwargs.pop("model_id", "google/owlv2-large-patch14-ensemble")
    return OWLv2Model(model_id=model_id, **_kw(kwargs, ["a photo of an insect"], 1024, 0.725, "owlv2_zero_shot_detector"))


@register_model
def yoloworld_zero_shot_detector(pretrained: bool = True, **kwargs):
    """YOLO-World v2-X at 1024 px. Default prompts: 13 arthropod taxa; tiling 1024 px (best setting on the validation sample).

    Keyword args: `prompts`, `tile`, `device`, `weights`, `imgsz`.
    """
    weights = kwargs.pop("weights", "yolov8x-worldv2.pt")
    imgsz = kwargs.pop("imgsz", 1024)
    return YOLOWorldModel(weights=weights, imgsz=imgsz, **_kw(kwargs, THIRTEEN_TAXA, 1024, 0.5, "yoloworld_zero_shot_detector"))


@register_model
def sam3_zero_shot_detector(pretrained: bool = True, **kwargs):
    """SAM 3 (gated on the Hub). Default prompts: insect, spider, arthropod; tiling 1024 px (best setting on the validation sample).

    Keyword args: `prompts`, `tile`, `device`, `model_id`.
    """
    model_id = kwargs.pop("model_id", "facebook/sam3")
    return SAM3Model(model_id=model_id, **_kw(kwargs, ["insect", "spider", "arthropod"], 1024, 0.95, "sam3_zero_shot_detector"))
