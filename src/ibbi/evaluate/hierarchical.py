# src/ibbi/evaluate/hierarchical.py

"""
Metrics for hierarchical classifiers with per-level abstention, computed on benchmark specimen crops.

For a crop of true species s with lineage (subfamily, tribe, genus, species), level L is *known* when the classifier's
label space contains that lineage down to L. The *ideal depth* is the number of leading known levels (4 for a trained
species; 3 for a held-out species of a trained genus; 2 for a new genus of a trained tribe; 1 for a new tribe). A
prediction *over-commits* when its reported depth exceeds the ideal depth (it names a taxon that cannot be right).
"""

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from ..utils.data import TAXONOMY_LEVELS, get_taxonomy


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative), ties counted half."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank(method="average").to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def evaluate_hierarchical_records(records: list[dict[str, Any]], truths: list[dict[str, Any]], known_table: pd.DataFrame) -> dict[str, Any]:
    """Computes per-level accuracy, calibration, novelty separation and depth metrics.

    Args:
        records (list[dict]): Classifier outputs (`HierarchicalClassifier.classify_crops`), one per crop.
        truths (list[dict]): `{"split": str, "species": str}` per crop, same order.
        known_table (pd.DataFrame): The classifier's species with columns subfamily, tribe, genus, scientificName.

    Returns:
        dict: `per_split` (accuracy / ECE / depth metrics), `novelty` (per-level AUROC, FPR@95TPR), `headline` (flat).
    """
    tax = get_taxonomy().drop_duplicates("scientificName").set_index("scientificName")
    known_keys = {lvl: set() for lvl in TAXONOMY_LEVELS}
    for _, r in known_table.iterrows():
        names = (r["subfamily"], r["tribe"], r["genus"], r["scientificName"])
        for li, lvl in enumerate(TAXONOMY_LEVELS):
            known_keys[lvl].add(names[: li + 1])

    rows = []
    for rec, t in zip(records, truths):
        sp = t["species"]
        if sp not in tax.index:
            continue
        lin = (tax.at[sp, "subfamily"], tax.at[sp, "tribe"], tax.at[sp, "genus"], sp)
        known = [lin[: li + 1] in known_keys[lvl] for li, lvl in enumerate(TAXONOMY_LEVELS)]
        ideal = 0
        for k in known:
            if not k:
                break
            ideal += 1
        row = {"split": t["split"], "species": sp, "band": tax.at[sp, "distance_band_vs_trainable"], "ideal_depth": ideal, "depth": rec["depth"]}
        prefix_ok = True
        for li, lvl in enumerate(TAXONOMY_LEVELS):
            row[f"known_{lvl}"] = known[li]
            row[f"correct_{lvl}"] = rec[lvl]["taxon"] == lin[li]
            row[f"prob_{lvl}"] = rec[lvl]["prob"]
            row[f"score_{lvl}"] = rec[lvl]["score"]
            if li < rec["depth"]:
                prefix_ok = prefix_ok and row[f"correct_{lvl}"]
        row["reported_correct"] = prefix_ok
        rows.append(row)
    df = pd.DataFrame(rows)
    out: dict[str, Any] = {"per_split": {}, "novelty": {}, "n_crops": len(df)}
    flat: dict[str, float] = {}
    if df.empty:
        out["headline"] = flat
        return out

    for split, g in df.groupby("split"):
        s: dict[str, Any] = {"n": len(g)}
        for lvl in TAXONOMY_LEVELS:
            k = g[g[f"known_{lvl}"]]
            if len(k):
                s[f"acc_{lvl}"] = float(k[f"correct_{lvl}"].mean())
                s[f"ece_{lvl}"] = _ece(k[f"prob_{lvl}"].to_numpy(), k[f"correct_{lvl}"].to_numpy(float))
        s["mean_depth"] = float(g["depth"].mean())
        s["depth_hist"] = np.bincount(g["depth"], minlength=5).tolist()
        s["over_commit_rate"] = float((g["depth"] > g["ideal_depth"]).mean())
        s["correct_at_reported_depth"] = float(g["reported_correct"].mean())
        s["right_depth_and_taxon"] = float(((g["depth"] == g["ideal_depth"]) & g["reported_correct"]).mean())
        if (g["ideal_depth"] == 4).any():
            k = g[g["ideal_depth"] == 4]
            s["known_species_named_correctly"] = float(((k["depth"] == 4) & k["correct_species"]).mean())
        if split == "semantic_ood":
            s["by_band"] = {}
            for band, b in g.groupby("band"):
                s["by_band"][band] = {
                    "n": len(b),
                    "ideal_depth": float(b["ideal_depth"].mean()),
                    "over_commit_rate": float((b["depth"] > b["ideal_depth"]).mean()),
                    "mean_depth": float(b["depth"].mean()),
                    "correct_genus_when_known": float(b.loc[b["known_genus"], "correct_genus"].mean()) if b["known_genus"].any() else float("nan"),
                    "correct_tribe_when_known": float(b.loc[b["known_tribe"], "correct_tribe"].mean()) if b["known_tribe"].any() else float("nan"),
                }
        out["per_split"][split] = s
        for key in (
            "acc_subfamily",
            "acc_tribe",
            "acc_genus",
            "acc_species",
            "ece_species",
            "over_commit_rate",
            "right_depth_and_taxon",
            "known_species_named_correctly",
        ):
            if key in s:
                flat[f"{split}.{key}"] = s[key]
        for band, b in s.get("by_band", {}).items():
            flat[f"{split}.{band}.over_commit_rate"] = b["over_commit_rate"]

    for lvl in TAXONOMY_LEVELS:
        pos = df.loc[df[f"known_{lvl}"], f"score_{lvl}"].to_numpy()
        neg = df.loc[~df[f"known_{lvl}"], f"score_{lvl}"].to_numpy()
        if len(pos) and len(neg):
            thr = np.quantile(pos, 0.05)
            out["novelty"][lvl] = {"auroc": _auroc(pos, neg), "fpr_at_95tpr": float((neg >= thr).mean()), "n_known": len(pos), "n_unknown": len(neg)}
            flat[f"novelty.{lvl}.auroc"] = out["novelty"][lvl]["auroc"]
    out["headline"] = flat
    out["rows"] = df
    return out


def summarize_by(df: pd.DataFrame, key: str) -> dict[str, Any]:
    """Small helper: mean of the boolean / numeric columns of `rows` grouped by `key`."""
    res: dict[str, Any] = defaultdict(dict)
    for k, g in df.groupby(key):
        res[str(k)] = g.select_dtypes(include=["number", "bool"]).mean().to_dict()
    return dict(res)
