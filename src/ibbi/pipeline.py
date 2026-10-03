# src/ibbi/pipeline.py

"""
Two-stage identification: a detector finds every arthropod, a hierarchical classifier names each one.

    pipe = ibbi.create_pipeline()                 # yolo11x_arthropod_detector + dinov3_hierarchical_classifier
    result = pipe.predict("plate.jpg")
    for box, rec in zip(result["boxes"], result["classifications"]):
        print(box, rec["reported"], rec["species"]["prob"])

The pipeline's labels are the reported identifications ("Xyleborus affinis", "Euwallacea sp. (species undetermined)",
"unrecognised (...)"); "species" holds the classifier's best species for every detection, which is what the
benchmark scores.
"""

from typing import Any

import numpy as np

from .models._common import ImageInput, is_batch, load_image


class IdentificationPipeline:
    """Detector + hierarchical classifier.

    Args:
        detector: A detector created with `ibbi.create_model` (normally "yolo11x_arthropod_detector").
        classifier: A hierarchical classifier created with `ibbi.create_model`.
        det_conf (float | None): Detector confidence threshold. Defaults to 0.25, the detector's general setting;
            use `detector.operating_conf` (0.70) for fewer false alarms.
        operating_point (str | None): Classifier operating point ("0.90", "0.95", "0.99", "gallery").
    """

    is_species_level = True

    def __init__(self, detector: Any, classifier: Any, det_conf: float | None = None, operating_point: str | None = None):
        self.detector = detector
        self.classifier = classifier
        self.det_conf = 0.25 if det_conf is None else float(det_conf)
        self.operating_point = operating_point
        self.name = f"{getattr(detector, 'name', 'detector')}+{getattr(classifier, 'name', 'classifier')}"
        # benchmark protocol: a low detector floor so most of the precision-recall curve is scored, capped so the
        # classifier does not have to name hundreds of near-zero-confidence boxes per image
        self.benchmark_kwargs = {"det_conf": 0.05, "max_det": 100}

    def predict(self, image, det_conf: float | None = None, operating_point: str | None = None, **kwargs):
        """Detects and classifies every specimen in one image or a list of images.

        Returns:
            dict | list[dict]: Per image {"boxes" (xyxy), "det_scores", "scores" (detector confidence x calibrated
            species probability), "labels" (reported identification), "species" (best species), "classifications"
            (full per-level records)}.
        """
        conf = self.det_conf if det_conf is None else float(det_conf)
        op = operating_point or self.operating_point
        images = list(image) if is_batch(image) else [image]
        outs = []
        for im in images:
            img = load_image(im)
            det = self.detector.predict(img, conf=conf, **kwargs)
            recs = self.classifier.predict(img, boxes=det["boxes"], operating_point=op) if det["boxes"] else []
            outs.append(
                {
                    "boxes": det["boxes"],
                    "det_scores": det["scores"],
                    "scores": [float(s) * r["species"]["prob"] for s, r in zip(det["scores"], recs)],
                    "labels": [r["reported"] for r in recs],
                    "species": [r["species"]["taxon"] for r in recs],
                    "classifications": recs,
                }
            )
        return outs if is_batch(image) else outs[0]

    def predict_proba(self, images: list[ImageInput], **kwargs) -> np.ndarray:
        """Per image, the highest (detector confidence x species probability) of every species: [N, 65]."""
        classes = self.classifier.get_classes()
        out = np.zeros((len(images), len(classes)), dtype=np.float32)
        for i, im in enumerate(images):
            img = load_image(im)
            det = self.detector.predict(img, conf=self.det_conf)
            if not det["boxes"]:
                continue
            crops = [self.classifier.crop(img, b) for b in det["boxes"]]
            p = self.classifier.predict_proba(crops) * np.asarray(det["scores"])[:, None]
            out[i] = p.max(0)
        return out

    def extract_features(self, image: ImageInput, **kwargs):
        """Classifier embedding of the whole image (treated as one specimen)."""
        return self.classifier.extract_features(image, **kwargs)

    def get_classes(self) -> list[str]:
        return self.classifier.get_classes()


def create_pipeline(
    detector: str | Any = "arthropod_detector",
    classifier: str | Any = "hierarchical_classifier",
    det_conf: float | None = None,
    operating_point: str | None = None,
    device: str | None = None,
) -> IdentificationPipeline:
    """Builds the two-stage identification pipeline.

    Args:
        detector (str | model): Model name/alias or an instantiated detector. Defaults to the arthropod detector.
        classifier (str | model): Model name/alias or an instantiated hierarchical classifier. Defaults to the DINOv3
            classifier.
        det_conf (float | None): Detector confidence threshold (default 0.25).
        operating_point (str | None): Classifier operating point.
        device (str | None): Device for models created here.

    Returns:
        IdentificationPipeline
    """
    from . import create_model

    det = create_model(detector, device=device) if isinstance(detector, str) else detector
    clf = create_model(classifier, device=device) if isinstance(classifier, str) else classifier
    return IdentificationPipeline(det, clf, det_conf=det_conf, operating_point=operating_point)
