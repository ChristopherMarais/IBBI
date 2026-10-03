# src/ibbi/models/classifiers.py

"""
Hierarchical bark and ambrosia beetle classifiers with per-level confidence and per-level "known / unsure" decisions.

For a specimen crop the classifier predicts the subfamily, tribe, genus and species, each with

    taxon      top-down prediction (best subfamily, then its best tribe, ...)
    prob       calibrated probability of that taxon (per-level temperature scaling)
    score      novelty score in [0, 1], higher = more familiar (percentile among known validation beetles)
    known      score >= the level's threshold at the chosen operating point

and reports the deepest level such that every level up to it is known: "Species", "Genus sp. (species undetermined)",
"Tribe (genus undetermined)", "Subfamily (tribe undetermined)" or "unrecognised".

Model: a fine-tuned ViT backbone (DINOv3-L or BioCLIP 2), LayerNorm and one linear layer that emits a conditional
logit per taxonomy node ("hierarchical softmax"): P(node | parent) is a softmax over siblings, the species probability
is the product along its lineage and every level's probability is the sum over its species, so the four levels are
always consistent. Trained on the benchmark's 65 trainable species with hierarchical cross-entropy, logit adjustment,
non-beetle outlier exposure and synthetic outliers. Novelty: max calibrated probability at subfamily, tribe and genus,
and its rank-average with the cosine similarity to the 5th nearest training embedding at species. No real unknown
beetles were used to train the model or to set its thresholds.
"""

import json
from typing import Any, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file

from ..utils.hub import HF_ORG, download_from_hf_hub
from ._common import ImageInput, is_batch, load_image, resolve_device
from ._registry import register_model

LEVELS = ("subfamily", "tribe", "genus", "species")


class Taxonomy:
    """Tree over the classifier's species and the tree-factorised head maths (inference only)."""

    def __init__(self, d: dict[str, Any]):
        self.nodes: dict[str, list[str]] = d["nodes"]
        self.parent: dict[str, list[int]] = d["parent"]
        self.path = torch.tensor(d["path"], dtype=torch.long)  # [S, 4] node index per level
        self.n = {lvl: len(self.nodes[lvl]) for lvl in LEVELS}
        self.species = self.nodes["species"]
        self.sib = {}
        self.desc = {}
        for li, lvl in enumerate(LEVELS):
            par = torch.tensor(self.parent[lvl])
            self.sib[lvl] = par[:, None] == par[None, :]
            self.desc[lvl] = self.path[:, li][None, :] == torch.arange(self.n[lvl])[:, None]

    def marginals(self, logits: dict[str, torch.Tensor], temps: dict[str, float]) -> dict[str, torch.Tensor]:
        """Raw conditional logits -> calibrated log-probability of every node at every level."""
        cond = {}
        for lvl in LEVELS:
            z = logits[lvl].float() / temps[lvl]
            neg = torch.zeros_like(self.sib[lvl], dtype=z.dtype).masked_fill(~self.sib[lvl], float("-inf"))
            cond[lvl] = z - torch.logsumexp(z[:, None, :] + neg[None], dim=-1)
        joint = 0
        for li, lvl in enumerate(LEVELS):
            joint = joint + cond[lvl][:, self.path[:, li]]
        out = {}
        for lvl in LEVELS:
            m = torch.zeros_like(self.desc[lvl], dtype=joint.dtype).masked_fill(~self.desc[lvl], float("-inf"))
            out[lvl] = torch.logsumexp(joint[:, None, :] + m[None], dim=-1)
        return out

    def table(self) -> pd.DataFrame:
        """One row per species with its subfamily, tribe and genus."""
        rows = []
        for s, p in enumerate(self.path.tolist()):
            rows.append({lvl: self.nodes[lvl][p[li]] for li, lvl in enumerate(LEVELS)} | {"scientificName": self.species[s]})
        return pd.DataFrame(rows)


class _Backbone(nn.Module):
    def __init__(self, spec: dict[str, Any], res: int):
        super().__init__()
        self.kind = spec["kind"]
        if self.kind == "timm":
            import timm

            self.net = timm.create_model(spec["name"], pretrained=False, num_classes=0, img_size=res)
            self.num_prefix = getattr(self.net, "num_prefix_tokens", 1)
        else:
            import open_clip

            mc = spec.get("open_clip_model_cfg")
            if mc is None:
                from huggingface_hub import hf_hub_download

                with open(hf_hub_download(spec["name"].replace("hf-hub:", ""), "open_clip_config.json")) as f:
                    mc = json.load(f)["model_cfg"]
                mc["vision_cfg"] = dict(mc["vision_cfg"], image_size=res)
            self.net = open_clip.model.CLIP(**mc).visual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "timm":
            t = self.net.forward_features(x)
            return torch.cat([t[:, 0], t[:, self.num_prefix :].mean(1)], dim=-1)
        return self.net(x)


class _HierNet(nn.Module):
    def __init__(self, spec: dict[str, Any], res: int, dim: int, sizes: list[int]):
        super().__init__()
        self.backbone = _Backbone(spec, res)
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, sum(sizes))
        self.sizes = sizes

    def forward(self, x):
        emb = self.norm(self.backbone(x))
        return emb, dict(zip(LEVELS, torch.split(self.fc(emb), self.sizes, dim=-1)))


def _letterbox(im: Image.Image, res: int, fill: tuple[int, int, int]) -> Image.Image:
    w, h = im.size
    s = res / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    im = im.resize((nw, nh), Image.BICUBIC, reducing_gap=2.0)
    canvas = Image.new("RGB", (res, res), fill)
    canvas.paste(im, ((res - nw) // 2, (res - nh) // 2))
    return canvas


class HierarchicalClassifier:
    """Specimen-crop classifier at subfamily, tribe, genus and species with per-level abstention.

    Args:
        config (dict): The repository's config.json.
        weights_path (str): model.safetensors.
        deploy_path (str): deploy.safetensors (kNN reference bank and the validation score distributions).
        device (str | None): Device; defaults to the best available.
        operating_point (str | None): "0.90", "0.95", "0.99" (share of known validation beetles accepted at every
            level) or "gallery" (0.99 at subfamily, 0.95 below; the default).
        name (str | None): Registry name.
    """

    is_species_level = True

    def __init__(
        self,
        config: dict[str, Any],
        weights_path: str,
        deploy_path: str,
        device: str | None = None,
        operating_point: str | None = None,
        name: str | None = None,
    ):
        self.config = config
        self.name = name or type(self).__name__
        self.device = resolve_device(device)
        self.tax = Taxonomy(config["taxonomy"])
        self.res = int(config["res"])
        self.pad = float(config.get("crop_pad", 0.05))
        spec = config["backbone"]
        self.net = _HierNet(spec, self.res, int(config["embedding_dim"]), [self.tax.n[lvl] for lvl in LEVELS])
        self.net.load_state_dict(load_file(weights_path), strict=True)
        self.net.to(self.device).eval()
        self.mean = torch.tensor(spec["mean"], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(spec["std"], device=self.device).view(1, 3, 1, 1)
        self.fill = tuple(int(round(255 * m)) for m in spec["mean"])
        self.temps = {lvl: float(config["temperatures"][lvl]) for lvl in LEVELS}
        nov = config["novelty"]
        self.members: dict[str, list[str]] = nov["members"]
        self.thresholds: dict[str, dict[str, float]] = nov["thresholds"]
        self.operating_point = self._resolve_op(operating_point or nov.get("default_op", "gallery"))
        dep = load_file(deploy_path)
        self.bank = dep["knn_bank"].to(self.device).float()
        self.valsorted = {(m, lvl): dep[f"valsorted.{m}.{lvl}"].numpy() for lvl in LEVELS for m in self.members[lvl]}
        self.benchmark_kwargs: dict[str, Any] = {}
        print(f"{self.name} loaded on device: {self.device}")

    # -- helpers --------------------------------------------------------------------------------------------------
    @property
    def taxonomy_table(self) -> pd.DataFrame:
        return self.tax.table()

    def get_classes(self) -> list[str]:
        """The 65 species the classifier knows, in output order."""
        return list(self.tax.species)

    def crop(self, image: ImageInput, box_xyxy, pad: float | None = None) -> Image.Image:
        """Crops a detection box grown by `pad` (default: the classifier's own 5%) on every side."""
        im = load_image(image)
        W, H = im.size
        x0, y0, x1, y1 = (float(v) for v in box_xyxy)
        p = self.pad if pad is None else pad
        w, h = x1 - x0, y1 - y0
        x0, y0, x1, y1 = max(0.0, x0 - p * w), max(0.0, y0 - p * h), min(float(W), x1 + p * w), min(float(H), y1 + p * h)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return im
        return im.crop((int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))))

    def _tensor(self, crops: list[Image.Image]) -> torch.Tensor:
        arr = np.stack([np.asarray(_letterbox(load_image(c), self.res, self.fill), dtype=np.uint8) for c in crops])
        x = torch.from_numpy(arr).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        return (x - self.mean) / self.std

    @torch.no_grad()
    def _forward(self, crops: list[Image.Image]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self._tensor(crops)
        if self.device.startswith("cuda"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb, z = self.net(x)
        else:
            emb, z = self.net(x)
        return emb.float(), {lvl: z[lvl].float().cpu() for lvl in LEVELS}

    def _novelty(self, emb: torch.Tensor, marg: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        raw: dict[str, dict[str, np.ndarray]] = {"msp": {lvl: np.exp(marg[lvl]).max(1) for lvl in LEVELS}}
        if any("knn5" in m for m in self.members.values()):
            sim = torch.nn.functional.normalize(emb, dim=1) @ self.bank.T
            k5 = sim.topk(5, dim=1).values[:, 4].cpu().numpy()
            raw["knn5"] = dict.fromkeys(LEVELS, k5)
        out = {}
        for lvl in LEVELS:
            pct = [np.searchsorted(self.valsorted[(m, lvl)], raw[m][lvl], side="right") / len(self.valsorted[(m, lvl)]) for m in self.members[lvl]]
            out[lvl] = np.mean(pct, axis=0)
        return out

    # -- public API -----------------------------------------------------------------------------------------------
    def _resolve_op(self, op: str | float) -> str:
        """Maps an operating point to its stored key, so that "0.90", "0.9" and 0.9 all name the same one."""
        key = str(op)
        if key in self.thresholds:
            return key
        try:
            value = float(key)
        except ValueError:
            value = None
        for k in self.thresholds:
            try:
                if value is not None and abs(float(k) - value) < 1e-9:
                    return k
            except ValueError:
                continue
        raise ValueError(f"Unknown operating point '{op}'. Available: {list(self.thresholds)}")

    def classify_crops(self, crops: list[ImageInput], operating_point: str | None = None, batch_size: int = 32) -> list[dict[str, Any]]:
        """Classifies specimen crops (one specimen per image). Returns one record per crop (see module docstring).

        Each record also has "depth" (0-4), "reported" (the human-readable result), "depth_by_op" (the depth under
        every stored operating point) and "embedding" is not included (use `extract_features`).
        """
        op = self._resolve_op(operating_point or self.operating_point)
        thr = self.thresholds[op]
        out = []
        for i in range(0, len(crops), batch_size):
            batch = [load_image(c) for c in crops[i : i + batch_size]]
            emb, z = self._forward(batch)
            marg_t = self.tax.marginals(z, self.temps)
            marg = {lvl: marg_t[lvl].numpy() for lvl in LEVELS}
            nov = self._novelty(emb, marg)
            for j in range(len(batch)):
                out.append(self._record(marg, nov, j, thr))
        return out

    def _record(self, marg, nov, j, thr) -> dict[str, Any]:
        rec: dict[str, Any] = {}
        prev, depth, alive = None, 0, True
        for lvl in LEVELS:
            m = marg[lvl][j].copy()
            if prev is not None:
                m[np.asarray(self.tax.parent[lvl]) != prev] = -np.inf  # stay inside the predicted parent
            k = int(m.argmax())
            prev = k
            score = float(nov[lvl][j])
            known = score >= thr[lvl]
            alive = alive and known
            depth += int(alive)
            p_all = np.exp(marg[lvl][j])
            top = np.argsort(-p_all)[:3]
            rec[lvl] = {
                "taxon": self.tax.nodes[lvl][k],
                "prob": float(min(1.0, p_all[k])),
                "score": score,
                "known": bool(known),
                "threshold": float(thr[lvl]),
                "entropy": float(max(0.0, -(p_all * marg[lvl][j]).sum())),
                "top3": [(self.tax.nodes[lvl][int(t)], float(min(1.0, p_all[t]))) for t in top],
            }
        rec["depth_by_op"] = {}
        for tag, th in self.thresholds.items():
            a, d = True, 0
            for lvl in LEVELS:
                a = a and rec[lvl]["score"] >= th[lvl]
                d += int(a)
            rec["depth_by_op"][tag] = d
        rec["depth"] = depth
        rec["reported"] = describe(rec, depth)
        return rec

    def predict(self, image, boxes=None, operating_point: str | None = None, batch_size: int = 32, **kwargs):
        """Classifies a specimen image, or the given boxes in an image.

        Args:
            image: A crop of one specimen (path, URL, array or PIL), or a list of crops. With `boxes`, a full image.
            boxes (list | None): xyxy boxes to crop from `image` (e.g. from a detector) and classify.
            operating_point (str | None): Override the operating point for this call.
            batch_size (int): Crops per forward pass.

        Returns:
            dict | list[dict]: One record for a single crop; a list for a list of crops or for `boxes`.
        """
        if boxes is not None:
            im = load_image(image)
            return self.classify_crops([self.crop(im, b) for b in boxes], operating_point, batch_size)
        if is_batch(image):
            return self.classify_crops(list(image), operating_point, batch_size)
        return self.classify_crops([image], operating_point, batch_size)[0]

    def predict_proba(self, images: list[ImageInput], level: str = "species", **kwargs) -> np.ndarray:
        """Calibrated probabilities of every taxon at `level` for each image: array [N, n_taxa]."""
        out = []
        for i in range(0, len(images), 32):
            _, z = self._forward([load_image(c) for c in images[i : i + 32]])
            out.append(np.exp(self.tax.marginals(z, self.temps)[level].numpy()))
        return np.concatenate(out).clip(0, 1)

    def extract_features(self, image: ImageInput, **kwargs) -> torch.Tensor:
        """The classifier embedding of a crop (LayerNorm output), shape [1, D]."""
        emb, _ = self._forward([load_image(image)])
        return emb.cpu()


def describe(rec: dict[str, Any], depth: int) -> str:
    if depth == 0:
        return "unrecognised (not a known bark or ambrosia beetle)"
    if depth == 4:
        return rec["species"]["taxon"]
    if depth == 3:
        return f"{rec['genus']['taxon']} sp. (species undetermined)"
    if depth == 2:
        return f"{rec['tribe']['taxon']} (genus undetermined)"
    return f"{rec['subfamily']['taxon']} (tribe undetermined)"


def _load_classifier(name: str, repo: str, device: str | None, revision: str | None, operating_point: str | None) -> HierarchicalClassifier:
    repo_id = f"{HF_ORG}/{repo}"
    with open(download_from_hf_hub(repo_id, "config.json", revision=revision)) as f:
        cfg = json.load(f)
    w = download_from_hf_hub(repo_id, "model.safetensors", revision=revision)
    d = download_from_hf_hub(repo_id, "deploy.safetensors", revision=revision)
    return HierarchicalClassifier(cfg, w, d, device=device, operating_point=operating_point, name=name)


def _check_pretrained(pretrained: bool, name: str) -> None:
    if not pretrained:
        raise ValueError(f"{name} is only available with its trained weights (pretrained=True); ibbi does not include training code.")


@register_model
def dinov3_hierarchical_classifier(
    pretrained: bool = True, device: str | None = None, revision: str | None = None, operating_point: str | None = None, **kwargs
) -> HierarchicalClassifier:
    """Hierarchical classifier on a fine-tuned DINOv3 ViT-L/16 at 336 px (the IBBI default classifier).

    Weights: https://huggingface.co/IBBI-bio/ibbi_dinov3l_hierarchical_classifier (DINOv3 License).

    Args:
        pretrained (bool): Must be True.
        device (str | None): Device; defaults to the best available.
        revision (str | None): Hub revision of the weights.
        operating_point (str | None): "0.90", "0.95", "0.99" or "gallery" (default).
    """
    _check_pretrained(pretrained, "dinov3_hierarchical_classifier")
    return _load_classifier("dinov3_hierarchical_classifier", "ibbi_dinov3l_hierarchical_classifier", device, revision, operating_point)


@register_model
def bioclip2_hierarchical_classifier(
    pretrained: bool = True, device: str | None = None, revision: str | None = None, operating_point: str | None = None, **kwargs
) -> HierarchicalClassifier:
    """Hierarchical classifier on a fine-tuned BioCLIP 2 (ViT-L/14) at 224 px.

    Weights: https://huggingface.co/IBBI-bio/ibbi_bioclip2_hierarchical_classifier

    Args:
        pretrained (bool): Must be True.
        device (str | None): Device; defaults to the best available.
        revision (str | None): Hub revision of the weights.
        operating_point (str | None): "0.90", "0.95", "0.99" or "gallery" (default).
    """
    _check_pretrained(pretrained, "bioclip2_hierarchical_classifier")
    return _load_classifier("bioclip2_hierarchical_classifier", "ibbi_bioclip2_hierarchical_classifier", device, revision, operating_point)


ClassifierInput = Union[ImageInput, list[ImageInput]]
