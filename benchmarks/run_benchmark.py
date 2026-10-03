#!/usr/bin/env python3
"""
Benchmark one ibbi model on the Bark and Ambrosia Beetle Detection Benchmark v2.0.1 (inference only).

    python benchmarks/run_benchmark.py --model yolo12x_species_detector --out results/
    python benchmarks/run_benchmark.py --model pipeline:dinov3_hierarchical_classifier --out results/
    python benchmarks/run_benchmark.py --model dinov3_hierarchical_classifier --out results/

What is run depends on the model:
  species detectors, pipelines   Evaluator.benchmark (the benchmark's crowd-aware reference evaluator)
  arthropod / zero-shot detectors Evaluator.benchmark (class-agnostic AP/AR, recall and precision at the operating conf)
  hierarchical classifiers        Evaluator.hierarchical_classification on ground-truth crops (scored specimens, and
                                  all specimens including crowd) and Evaluator.embeddings on iid_test specimens

Outputs in <out>/<model>/: predictions per split, the evaluator's files, and results.json (headline metrics, timing,
environment). Every setting used is the model's default; nothing is tuned on the benchmark.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import torch

import ibbi

SPLITS = ["iid_test", "inat_test", "semantic_ood"]


def env_info():
    import importlib.metadata as md

    pk = {}
    for p in ("ibbi", "torch", "ultralytics", "transformers", "timm", "open_clip_torch", "pycocotools"):
        try:
            pk[p] = md.version(p)
        except md.PackageNotFoundError:
            pk[p] = None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"python": platform.python_version(), "packages": pk, "gpu": gpu}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if hasattr(o, "item") and not hasattr(o, "shape"):
        return o.item()
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if hasattr(o, "tolist"):
        return o.tolist()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="registry name, or pipeline:<classifier name> for arthropod detector + classifier")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-dir", default=None, help="benchmark root (default: download to the ibbi cache)")
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--max-images", type=int, default=None, help="quick test only")
    args = ap.parse_args()

    name = args.model.replace(":", "__")
    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.model.startswith("pipeline:"):
        model = ibbi.create_pipeline("yolo11x_arthropod_detector", args.model.split(":", 1)[1])
    else:
        model = ibbi.create_model(args.model)
    res = {"model": args.model, "env": env_info(), "dataset_revision": ibbi.utils.data.BENCHMARK_REVISION, "splits": args.splits}
    ev = ibbi.Evaluator(model)
    if hasattr(model, "classify_crops"):
        h = ev.hierarchical_classification(splits=args.splits, dataset_dir=args.dataset_dir, max_images=args.max_images)
        rows = h.pop("rows")
        rows.to_csv(out / "crops_scored.csv.gz", index=False)
        res["hierarchical_scored"] = h
        hc = ev.hierarchical_classification(
            splits=[s for s in args.splits if s != "semantic_ood"], dataset_dir=args.dataset_dir, include_crowd=True, max_images=args.max_images
        )
        hc.pop("rows")
        res["hierarchical_all_known_specimens"] = hc
        ds = ibbi.get_dataset("iid_test", local_dir=args.dataset_dir) if args.dataset_dir else ibbi.get_dataset("iid_test")
        if args.max_images:
            ds = ds.select(range(min(args.max_images, len(ds))))
        emb = ev.embeddings(ds, evaluation_level="object", use_umap=args.max_images is None, include_crowd=True)
        res["embeddings_iid_test"] = {k: v for k, v in emb.items() if k not in ("sample_results", "per_class_centroids")}
        res["headline"] = dict(h["headline"])
        mc = emb.get("mantel_correlation") or {}
        if mc:
            res["headline"]["embeddings.iid_test.mantel_r_taxonomic"] = mc["r"]
    else:
        r = ev.benchmark(splits=args.splits, dataset_dir=args.dataset_dir, max_images=args.max_images, output_dir=out, model_name=name)
        res["benchmark"] = r
        res["headline"] = r["headline"]
        res["benchmark_kwargs"] = getattr(model, "benchmark_kwargs", {})
        res["operating_conf"] = getattr(model, "operating_conf", None)
    res["seconds"] = time.time() - t0
    (out / "results.json").write_text(json.dumps(jsonable(res), indent=1))
    print(json.dumps(jsonable(res["headline"]), indent=1))


if __name__ == "__main__":
    main()
