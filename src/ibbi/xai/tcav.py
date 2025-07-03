# src/ibbi/xai/tcav.py
"""
ROI-aware TCAV utilities for IBBI multi-class YOLO detectors
-----------------------------------------------------------
This re-implementation keeps **spatial information inside each detected
bounding box** when discovering concepts and computing TCAV scores. The
pipeline is now:

1. Detect objects -> store **box coordinates** (no cropping).
2. Forward once per image, capture full neck feature maps.
3. ROI-align inside each box, global-average-pool per level -> 1 vector/ROI.
4. Cluster target-class vectors (HDBSCAN) -> concepts.
5. Learn one CAV per concept (SGD vs. background vectors).
6. TCAV test: directional derivative of the *box-specific* logit along the CAV.

As a result, every concept and every TCAV score is conditioned on the exact
pixels of the beetle inside its bounding box, enabling body-part analysis.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import hdbscan
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from torchvision.ops import roi_align
from tqdm.auto import tqdm

from ibbi.models.multi_class_detection import (
    YOLOBeetleMultiClassDetector,
)

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------
#           Constants & ROI-pool helper
# --------------------------------------------------------------
# Neck strides for YOLOv8/9/10 P3, P4, P5 feature maps.
STRIDES: list[int] = [8, 16, 32]


def _to_tensor_and_pad(pil_img: Image.Image, divisor: int = 32) -> torch.Tensor:
    """Convert PIL -> Tensor and right/bottom-pad so height & width are multiples of *divisor*."""
    t = transforms.ToTensor()(pil_img)
    _, h, w = t.shape
    new_h = math.ceil(h / divisor) * divisor
    new_w = math.ceil(w / divisor) * divisor
    pad_h = new_h - h
    pad_w = new_w - w
    if pad_h or pad_w:
        # pad format: (left, right, top, bottom)
        t = F.pad(t, (0, pad_w, 0, pad_h))
    return t


def roi_pool_and_flatten(
    fmap_list: list[torch.Tensor],
    boxes_xyxy: torch.Tensor,
) -> torch.Tensor:
    """ROI-align each feature-map level inside *boxes* and GAP-pool.

    * `fmap_list` - list[C x H x W] feature maps from the neck.
    * `boxes_xyxy` - `[N,4]` boxes in image-pixel coordinates.

    Returns a tensor `[N, ΣC]` (concatenated pooled vectors).
    """
    device = fmap_list[0].device
    pooled: list[torch.Tensor] = []
    for lvl, fmap in enumerate(fmap_list):
        stride = STRIDES[lvl]
        rois = torch.cat(
            [
                torch.zeros(len(boxes_xyxy), 1, device=device),  # batch idx 0
                boxes_xyxy / stride,
            ],
            dim=1,
        )
        # 1x1 output gives global avg inside ROI
        feat = roi_align(
            fmap.unsqueeze(0),
            rois,
            output_size=(1, 1),
            spatial_scale=1.0,
            aligned=True,
        )  # type: ignore
        pooled.append(feat.flatten(start_dim=2))  # [N, C, 1x1] -> [N, C]
    return torch.cat(pooled, dim=1).squeeze(-1)  # [N, ΣC]


# ------------------------------------------------------------------
#                   Feature-vector helpers
# ------------------------------------------------------------------


def pool_and_flatten_features(feature_maps_list: list[list[torch.Tensor]]) -> torch.Tensor:
    """Global-average-pool every fmap, flatten, then stack."""
    pooled = []
    for fmap_set in feature_maps_list:
        if not fmap_set:
            continue
        pooled.append(torch.cat([torch.nn.functional.adaptive_avg_pool2d(fm, (1, 1)).flatten() for fm in fmap_set]))
    return torch.stack(pooled) if pooled else torch.empty(0)


# ------------------------------------------------------------------
#                   Robust raw-prediction flattening
# ------------------------------------------------------------------


def _stack_raw_preds(raw_outputs: Any) -> torch.Tensor:
    """Fuse heterogeneous Ultralytics head outputs to `[bs,N,5+nc]`."""

    def _add_flat(t: torch.Tensor, pool: list[torch.Tensor]):
        bs = t.shape[0]
        pool.append(t.view(bs, -1, t.shape[-1]))

    def _fuse_and_add(
        cls_t: torch.Tensor, box_t: torch.Tensor | None, obj_t: torch.Tensor | None, pool: list[torch.Tensor]
    ):
        bs, *spatial, _ = cls_t.shape
        device = cls_t.device
        if box_t is None:
            box_t = cls_t.new_zeros(bs, *spatial, 4, device=device)
        if obj_t is None:
            obj_t = cls_t.new_zeros(bs, *spatial, 1, device=device)
        elif obj_t.dim() == cls_t.dim() - 1:
            obj_t = obj_t.unsqueeze(-1)
        fused = torch.cat([box_t, obj_t, cls_t], dim=-1)
        _add_flat(fused, pool)

    # normalise to iterable
    if isinstance(raw_outputs, (torch.Tensor, dict)):
        raw_outputs = [raw_outputs]
    elif not isinstance(raw_outputs, (list, tuple)):
        raise TypeError(f"Unsupported raw output type: {type(raw_outputs)}")

    pool: list[torch.Tensor] = []
    for chunk in raw_outputs:
        if isinstance(chunk, torch.Tensor):
            if chunk.shape[-1] == 6:  # YOLO post-NMS tensor
                continue
            _add_flat(chunk, pool)
            continue
        if not isinstance(chunk, dict):
            raise TypeError(f"Unsupported scale output type: {type(chunk)}")

        box = chunk.get("box") or chunk.get("boxes") or chunk.get("bbox") or chunk.get("reg")
        obj = chunk.get("obj") or chunk.get("objectness") or chunk.get("score") or chunk.get("confidence")
        cls = chunk.get("cls") or chunk.get("class") or chunk.get("scores") or chunk.get("prob") or chunk.get("probs")

        if isinstance(cls, list):
            for i, cls_t in enumerate(cls):
                bx = box[i] if isinstance(box, list) and i < len(box) else None
                ob = obj[i] if isinstance(obj, list) and i < len(obj) else None
                _fuse_and_add(cls_t, bx, ob, pool)
            continue
        if isinstance(cls, torch.Tensor):
            _fuse_and_add(cls, box, obj, pool)
            continue
        # fallback: any tensor >=5 chans
        for v in chunk.values():
            cands = v if isinstance(v, list) else [v]
            for t in cands:
                if isinstance(t, torch.Tensor) and t.shape[-1] >= 5:
                    _add_flat(t, pool)
                    break

    if not pool:
        raise ValueError("Could not stack raw predictions - no suitable tensors found.")
    widths = [p.shape[-1] for p in pool]
    target_w = max(widths)
    pool = [p for p in pool if p.shape[-1] == target_w]
    return torch.cat(pool, dim=1)


# ------------------------------------------------------------------
#                   Public TCAV entry-point
# ------------------------------------------------------------------


def auto_tcav_explain(
    model: YOLOBeetleMultiClassDetector,
    dataset: Any,
    target_class: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """End-to-end ROI-aware TCAV pipeline."""
    device = model.device

    # ---------------- Sanity checks ----------------
    class_names = model.get_classes()
    if target_class not in class_names:
        raise ValueError(f"{target_class!r} not in model classes.")
    tgt_idx = class_names.index(target_class)

    batch = kwargs.get("detection_batch_size", 16)
    conf_th = kwargs.get("crop_confidence", 0.5)
    min_cluster = kwargs.get("min_cluster_size", 4)

    # ---------------- Stage 1: detection (store boxes) -------------
    print("STAGE 1: Detecting instances...")
    boxes_by_cls: dict[str, list[tuple[int, torch.Tensor]]] = {}
    images: list[Image.Image] = []

    for start in tqdm(range(0, len(dataset), batch), desc="Detecting"):
        items = [dataset[j] for j in range(start, min(start + batch, len(dataset)))]
        imgs = [it["image"] for it in items]
        images.extend(imgs)

        for img_i, res in enumerate(model.predict(imgs, stream=False, verbose=False)):
            if getattr(res, "boxes", None) is None:
                continue
            for box in res.boxes:
                if box.conf.item() < conf_th:
                    continue
                cls_name = class_names[int(box.cls)]
                boxes_by_cls.setdefault(cls_name, []).append((start + img_i, box.xyxy[0].cpu()))

    if target_class not in boxes_by_cls:
        raise RuntimeError("No target-class detections - TCAV aborted.")

    # ---------------- Stage 2: feature extraction & ROI vectors ----
    print("\nSTAGE 2: ROI-pooling neck feature maps...")
    pre_full = _to_tensor_and_pad

    # forward once per image to cache feature maps
    raw_feats: list[list[torch.Tensor]] = []
    for pil in tqdm(images, desc="Forward"):
        tensor = pre_full(pil).unsqueeze(0).to(device)
        fmaps = model.extract_raw_features(tensor)
        raw_feats.append([fm.squeeze(0).cpu().detach() for fm in fmaps])

    roi_vecs: list[torch.Tensor] = []
    roi_labels: list[int] = []
    roi_img_ids: list[int] = []
    roi_boxes: list[torch.Tensor] = []

    # iterate over *all* classes so background vectors are present
    for cls_name, roi_list in boxes_by_cls.items():
        is_target = int(cls_name == target_class)
        for img_id, xyxy in roi_list:
            vec = roi_pool_and_flatten([fm.to(device) for fm in raw_feats[img_id]], xyxy.to(device).unsqueeze(0))
            roi_vecs.append(vec.squeeze(0).cpu())
            roi_labels.append(is_target)
            roi_img_ids.append(img_id)
            roi_boxes.append(xyxy)

    roi_vecs = torch.stack(roi_vecs)  # type: ignore
    roi_labels_np = np.array(roi_labels)

    print(f"Collected {roi_labels_np.sum()} target and {(1 - roi_labels_np).sum()} background ROIs.")

    # ---------------- Stage 3: clustering -------------------------
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster)
    tgt_mask = roi_labels_np == 1
    labels = clusterer.fit_predict(roi_vecs[tgt_mask].numpy())
    concepts = sorted({c for c in labels if c != -1})
    print(f"Identified {len(concepts)} concepts.")

    results: dict[str, Any] = {}
    for cid in concepts:
        cname = f"concept_{cid}"
        print(f"\n--- Analysing {cname} ---")
        idxs = np.where(labels == cid)[0]

        X = np.concatenate([roi_vecs[tgt_mask][idxs], roi_vecs[~tgt_mask]])
        y = np.concatenate([np.ones(len(idxs)), np.zeros((~tgt_mask).sum())])
        X_scaled = StandardScaler().fit_transform(X)
        # MODIFIED: Assign the classifier to a variable to inspect its coef_
        classifier = SGDClassifier(max_iter=1000, tol=1e-3).fit(X_scaled, y)

        # ADDED: Check if coef_ is None or empty before accessing it
        if classifier.coef_ is None or classifier.coef_.shape[0] == 0:
            print(
                f"Warning: SGDClassifier coef_ is None or empty for concept {cid}. "
                f"This usually means the classifier could not converge or received single-class data. "
                f"Skipping CAV calculation for this concept."
            )
            continue  # Skip this concept as a valid CAV cannot be formed

        cav_vec = torch.as_tensor(
            classifier.coef_[0],  # Now safe to access coef_[0]
            dtype=torch.float32,
            device=device,
        )

        score = _tcav_score_roi(
            model,
            cav_vec,
            roi_img_ids=[roi_img_ids[i] for i in idxs],
            roi_boxes=[roi_boxes[i] for i in idxs],
            raw_feats=raw_feats,
            target_class_index=tgt_idx,
        )
        print(f"TCAV Score: {score:.4f}")

        results[cname] = {
            "score": score,
            "images": [images[roi_img_ids[i]].crop(tuple(roi_boxes[i].tolist())) for i in idxs],
        }
    return results


# ------------------------------------------------------------------
#                   TCAV directional derivative test
# ------------------------------------------------------------------


def _tcav_score_roi(
    model: YOLOBeetleMultiClassDetector,
    cav: torch.Tensor,
    *,
    roi_img_ids: list[int],
    roi_boxes: list[torch.Tensor],
    raw_feats: list[list[torch.Tensor]],
    target_class_index: int,
) -> float:
    """Fraction of ROIs where directional derivative > 0 along CAV."""
    device = cav.device

    positives = 0
    for img_id, xyxy in zip(roi_img_ids, roi_boxes):
        fmap_list = [fm.to(device).clone().requires_grad_(True) for fm in raw_feats[img_id]]
        roi_feat = roi_pool_and_flatten(fmap_list, xyxy.to(device).unsqueeze(0))
        roi_feat.retain_grad()

        with torch.enable_grad():
            raw_out = model.predict_from_features([fm.unsqueeze(0) for fm in fmap_list])
        preds = _stack_raw_preds(raw_out)

        # keep boxes whose centre lies inside xyxy
        cxcy = preds[..., :2] + preds[..., 2:4] / 2.0
        mask = (
            (cxcy[..., 0] >= xyxy[0])
            & (cxcy[..., 0] <= xyxy[2])
            & (cxcy[..., 1] >= xyxy[1])
            & (cxcy[..., 1] <= xyxy[3])
        )
        if not mask.any():
            continue
        sliced_preds = preds[0, mask, 5 + target_class_index]
        assert isinstance(sliced_preds, torch.Tensor)  # Assert for pyright type narrowing
        logit = sliced_preds.max().to(device)

        assert hasattr(model, "model")
        model.model.zero_grad()
        logit.backward()

        grad_val = roi_feat.grad
        if grad_val is not None:
            grad = grad_val.view(-1)
            if torch.dot(grad, cav).item() > 0:
                positives += 1

    if not roi_img_ids:
        return 0.0
    return positives / len(roi_img_ids)
