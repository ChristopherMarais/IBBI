# src/ibbi/models/detectors.py

"""
Detectors trained by the IBBI project (Ultralytics YOLO and RT-DETR checkpoints).

* Species detectors: one model per architecture (YOLOv8x, YOLOv9e, YOLOv10x, YOLO11x, YOLO12x, RT-DETR-X), each
  trained on the `train` split of the Bark and Ambrosia Beetle Detection Benchmark v2.0.1 to detect and name the 65
  trainable species in one step. The shipped seed of each architecture is the one with the best validation fitness.
* Arthropod detector: YOLO11x trained on the IBBI arthropod detection corpus (307k images from 14 sources, lab, trap,
  camera-trap and field imagery) to find any arthropod (single class "arthropod"). It is the first stage of the
  two-stage identification pipeline (`ibbi.create_pipeline`).

Weights live in the IBBI-bio organisation on the Hugging Face Hub; see each repository's model card for training
details, benchmark results and licence terms.
"""

from typing import Any

import numpy as np
import torch

from ..utils.hub import HF_ORG, download_from_hf_hub, get_model_config_from_hub
from ._common import ImageInput, empty_result, is_batch, load_image, resolve_device
from ._registry import register_model


def _check_ultralytics_version() -> None:
    """The IBBI detectors were trained and benchmarked with Ultralytics 8.3. Ultralytics 8.4 loads 8.3 YOLOv10
    checkpoints without their NMS-free head and changes RT-DETR post-processing, which changes their predictions
    (benchmark iid_test AP 0.561 -> 0.505 for YOLOv10x, 0.535 -> 0.599 for RT-DETR-X); the other architectures are
    unaffected. ibbi therefore requires ultralytics < 8.4."""
    import warnings

    import ultralytics

    major_minor = tuple(int(x) for x in ultralytics.__version__.split(".")[:2])
    if major_minor >= (8, 4):
        warnings.warn(
            f"ultralytics {ultralytics.__version__} is installed; ibbi's detectors were validated with ultralytics 8.3 "
            "(pip install 'ultralytics>=8.3.139,<8.4'). YOLOv10 and RT-DETR predictions differ under 8.4.",
            stacklevel=3,
        )


class UltralyticsDetector:
    """Common wrapper for Ultralytics YOLO / RT-DETR checkpoints.

    Args:
        model_path (str): Path to the `.pt` checkpoint.
        config (dict | None): The repository's `config.json` (image size, defaults, class mapping).
        device (str | None): "cuda", "cpu", "mps", ... Defaults to the best available device.
        name (str | None): Registry name of the model.
        fast (bool | None): Letterbox on the GPU and pass Ultralytics a ready tensor (default: True on CUDA). Same
            geometry as Ultralytics' CPU letterbox; resize rounding differs slightly. Set `detector.fast = False` for
            the reference path that produced the published benchmark numbers.
    """

    is_species_level = False

    def __init__(
        self, model_path: str, config: dict[str, Any] | None = None, device: str | None = None, name: str | None = None, fast: bool | None = None
    ):
        from ultralytics import RTDETR, YOLO

        _check_ultralytics_version()
        self.config = config or {}
        loader = RTDETR if self.config.get("loader") == "RTDETR" else YOLO
        self.model = loader(model_path)
        self.device = resolve_device(device)
        self.model.to(self.device)
        self.name = name or type(self).__name__
        self.classes = [self.model.names[i] for i in range(len(self.model.names))]
        self.imgsz = int(self.config.get("imgsz", 640))
        self.inference_defaults = dict(self.config.get("inference_defaults", {"conf": 0.25, "iou": 0.7, "max_det": 300}))
        bench = dict(self.config.get("benchmark_inference", {"conf": 0.001}))
        bench.pop("imgsz", None)
        self.benchmark_kwargs = bench
        # fast path: letterbox on the GPU and give Ultralytics a ready tensor (see models/_gpu.py); CUDA only
        self.fast = str(self.device).startswith("cuda") if fast is None else bool(fast)
        self.accepts_tensors = True
        print(f"{self.name} loaded on device: {self.device}")

    def _run(self, images: list, **kwargs) -> list:
        kw = {"imgsz": self.imgsz, "verbose": False, "device": self.device, **self.inference_defaults, **kwargs}
        if self.fast:
            return [self._run_tensor(im, kw) for im in images]
        # Ultralytics expects BGR numpy arrays or paths; PIL images are converted by it.
        return self.model.predict([load_image(im) if not isinstance(im, torch.Tensor) else _to_pil(im) for im in images], **kw)

    def _run_tensor(self, image, kw: dict) -> Any:
        """One image through the GPU letterbox; boxes are mapped back to the original image like Ultralytics does."""
        from ultralytics.utils import ops

        from ._gpu import load_tensor, ultralytics_letterbox

        t = load_tensor(image, self.device)
        stride = int(max(getattr(self.model.model, "stride", torch.tensor([32])).max().item(), 32))
        x = ultralytics_letterbox(t, int(kw["imgsz"]), stride)
        res = self.model.predict(x, **kw)[0]
        if res.boxes is not None and len(res.boxes):
            from ultralytics.engine.results import Boxes

            data = res.boxes.data.clone()
            data[:, :4] = ops.scale_boxes(x.shape[2:], data[:, :4], tuple(t.shape[1:]))
            res.boxes = Boxes(data, tuple(t.shape[1:]))
        return res

    @staticmethod
    def _to_dict(res, names: list[str]) -> dict[str, list]:
        out = empty_result()
        out["class_ids"] = []
        if res is None or res.boxes is None or len(res.boxes) == 0:
            return out
        b = res.boxes
        out["boxes"] = b.xyxy.cpu().numpy().tolist()
        out["scores"] = b.conf.cpu().numpy().tolist()
        out["class_ids"] = b.cls.cpu().numpy().astype(int).tolist()
        out["labels"] = [names[c] for c in out["class_ids"]]
        return out

    def predict(self, image, **kwargs):
        """Detects objects in one image or a list of images.

        Args:
            image (str | Path | np.ndarray | PIL.Image | list): Image(s) to process (path, URL, RGB array or PIL).
            **kwargs: Passed to Ultralytics `predict` (e.g. `conf`, `iou`, `max_det`, `imgsz`, `agnostic_nms`).

        Returns:
            dict | list[dict]: Per image, {"boxes" (xyxy), "scores", "labels", "class_ids"}.
        """
        kwargs.pop("include_full_probabilities", None)
        images = list(image) if is_batch(image) else [image]
        if not self.fast and any(isinstance(im, torch.Tensor) for im in images):
            images = [_to_pil(im) if isinstance(im, torch.Tensor) else im for im in images]
        results = self._run(images, **kwargs)
        outs = [self._to_dict(r, self.classes) for r in results]
        return outs if is_batch(image) else outs[0]

    def predict_proba(self, images: list[ImageInput], **kwargs) -> np.ndarray:
        """Per image, the highest detection confidence of every class: array [N, n_classes] (used by LIME / SHAP)."""
        kwargs.setdefault("conf", 0.01)
        out = np.zeros((len(images), len(self.classes)), dtype=np.float32)
        for i, r in enumerate(self.predict(list(images), **kwargs)):
            for c, s in zip(r["class_ids"], r["scores"]):
                out[i, c] = max(out[i, c], s)
        return out

    def extract_features(self, image: ImageInput, **kwargs) -> torch.Tensor | None:
        """Image-level embedding from the detector backbone (Ultralytics `embed`)."""
        feats = self.model.embed(load_image(image), imgsz=kwargs.pop("imgsz", self.imgsz), verbose=False, **kwargs)
        return feats[0] if feats else None

    def get_classes(self) -> list[str]:
        """Class names, in class-index order."""
        return self.classes


def _to_pil(t: torch.Tensor):
    from PIL import Image

    return Image.fromarray(t.permute(1, 2, 0).cpu().numpy())


class SpeciesDetector(UltralyticsDetector):
    """Detects and names the 65 trainable species of the benchmark in one step.

    `predict(..., level="genus")` reports labels at a coarser taxonomic level (subfamily, tribe or genus) by mapping
    each predicted species to its lineage.
    """

    is_species_level = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from ..utils.data import get_taxonomy

        t = get_taxonomy().drop_duplicates("scientificName").set_index("scientificName")
        self.lineage = {sp: {lvl: t.at[sp, lvl] for lvl in ("subfamily", "tribe", "genus")} for sp in self.classes if sp in t.index}

    def predict(self, image, level: str = "species", **kwargs):
        """Like `UltralyticsDetector.predict`; `level` in {"species", "genus", "tribe", "subfamily"} sets the label level.

        The predicted species is always also returned under "species".
        """
        outs = super().predict(image, **kwargs)
        for o in outs if isinstance(outs, list) else [outs]:
            o["species"] = list(o["labels"])
            if level != "species":
                o["labels"] = [self.lineage.get(s, {}).get(level, s) for s in o["species"]]
        return outs


class ArthropodDetector(UltralyticsDetector):
    """Single-class detector that finds any arthropod ("arthropod").

    `operating_conf` (0.70) is the confidence at which the detector makes at most 0.2 false alarms per image on its
    validation sample; use `predict(image, conf=detector.operating_conf)` for a high-precision operating point.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operating_conf = float(self.config.get("operating_conf", 0.7))


# ----------------------------------------------------------------------------------------------------------------------
SPECIES_REPOS = {
    "yolov8x_species_detector": ("ibbi_yolov8x_species_detector", "yolov8x.pt"),
    "yolov9e_species_detector": ("ibbi_yolov9e_species_detector", "yolov9e.pt"),
    "yolov10x_species_detector": ("ibbi_yolov10x_species_detector", "yolov10x.pt"),
    "yolo11x_species_detector": ("ibbi_yolo11x_species_detector", "yolo11x.pt"),
    "yolo12x_species_detector": ("ibbi_yolo12x_species_detector", "yolo12x.pt"),
    "rtdetrx_species_detector": ("ibbi_rtdetrx_species_detector", "rtdetr-x.pt"),
}


def _load(cls, name: str, repo: str, base: str, pretrained: bool, device: str | None, revision: str | None, loader: str = "YOLO"):
    if pretrained:
        repo_id = f"{HF_ORG}/{repo}"
        cfg = get_model_config_from_hub(repo_id, revision=revision)
        path = download_from_hf_hub(repo_id, "model.pt", revision=revision)
        return cls(path, config=cfg, device=device, name=name)
    print(f"pretrained=False: loading the generic Ultralytics checkpoint '{base}' (COCO classes, not trained on beetles).")
    return UltralyticsDetector(base, config={"loader": loader}, device=device, name=name)


def _species_factory(name: str):
    repo, base = SPECIES_REPOS[name]
    loader = "RTDETR" if name.startswith("rtdetr") else "YOLO"

    def factory(pretrained: bool = True, device: str | None = None, revision: str | None = None, **kwargs):
        return _load(SpeciesDetector, name, repo, base, pretrained, device, revision, loader)

    factory.__name__ = name
    factory.__qualname__ = name
    factory.__doc__ = (
        f"Species detector ({repo.split('_')[1]}) for the 65 trainable bark and ambrosia beetle species "
        f"(weights: https://huggingface.co/{HF_ORG}/{repo}).\n\n"
        "Args:\n    pretrained (bool): Load the IBBI weights (default). False loads the generic Ultralytics COCO checkpoint.\n"
        "    device (str | None): Device; defaults to the best available.\n    revision (str | None): Hub revision of the weights.\n"
    )
    return register_model(factory)


yolov8x_species_detector = _species_factory("yolov8x_species_detector")
yolov9e_species_detector = _species_factory("yolov9e_species_detector")
yolov10x_species_detector = _species_factory("yolov10x_species_detector")
yolo11x_species_detector = _species_factory("yolo11x_species_detector")
yolo12x_species_detector = _species_factory("yolo12x_species_detector")
rtdetrx_species_detector = _species_factory("rtdetrx_species_detector")


@register_model
def yolo11x_arthropod_detector(pretrained: bool = True, device: str | None = None, revision: str | None = None, **kwargs):
    """Universal arthropod detector (YOLO11x, single class "arthropod", 1024 px input).

    Trained on the IBBI arthropod detection corpus (lab, trap, camera-trap and field imagery from 14 sources).
    Weights: https://huggingface.co/IBBI-bio/ibbi_yolo11x_arthropod_detector

    Args:
        pretrained (bool): Load the IBBI weights (default). False loads the generic Ultralytics COCO checkpoint.
        device (str | None): Device; defaults to the best available.
        revision (str | None): Hub revision of the weights.
    """
    return _load(ArthropodDetector, "yolo11x_arthropod_detector", "ibbi_yolo11x_arthropod_detector", "yolo11x.pt", pretrained, device, revision)
