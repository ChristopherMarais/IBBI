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

Slow detectors can be split over several GPUs (images i, i+n, i+2n, ... of every split go to shard i):

    python benchmarks/run_benchmark.py --model sam3_zero_shot_detector --out results/ --shard 0/6    # one job per shard
    python benchmarks/run_benchmark.py --model sam3_zero_shot_detector --out results/ --merge 6       # scores all shards
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
    ap.add_argument("--shard", default=None, help="i/n: only predict shard i of n (detectors); scored later with --merge n")
    ap.add_argument("--merge", type=int, default=None, help="n: merge the n shards written by --shard and score them")
    args = ap.parse_args()

    name = args.model.replace(":", "__")
    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)
    if args.merge:
        return merge_shards(args, name, out)
    t0 = time.time()
    if args.model.startswith("pipeline:"):
        model = ibbi.create_pipeline("yolo11x_arthropod_detector", args.model.split(":", 1)[1])
    else:
        model = ibbi.create_model(args.model)
    res = {"model": args.model, "env": env_info(), "dataset_revision": ibbi.utils.data.BENCHMARK_REVISION, "splits": args.splits}
    ev = ibbi.Evaluator(model)
    if args.shard:
        return predict_shard(args, model, ev, out, t0)
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


def predict_shard(args, model, ev, out: Path, t0: float):
    """Predict shard i of n of every split and write <out>/shards/<split>.<i>of<n>.json (image ids + predictions)."""
    i, n = (int(v) for v in args.shard.split("/"))
    shard_dir = out / "shards"
    shard_dir.mkdir(exist_ok=True)
    for split in args.splits:
        ds = ibbi.get_dataset(split, local_dir=args.dataset_dir) if args.dataset_dir else ibbi.get_dataset(split)
        part = ds.select(list(range(i, len(ds), n)))
        ts = time.time()
        preds = ev.predict_split(part)
        rec = {
            "model": args.model,
            "split": split,
            "shard": [i, n],
            "image_ids": [r["image_id"] for r in part.records()],
            "predictions": preds,
            "is_species_level": bool(getattr(model, "is_species_level", False)),
            "operating_conf": getattr(model, "operating_conf", None),
            "benchmark_kwargs": getattr(model, "benchmark_kwargs", {}),
            "env": env_info(),
            "seconds": time.time() - ts,
        }
        (shard_dir / f"{split}.{i}of{n}.json").write_text(json.dumps(jsonable(rec)))
        print(f"{split}: shard {i}/{n}, {len(part)} images, {len(preds)} predictions, {rec['seconds']:.0f} s", flush=True)
    print(f"shard done in {time.time() - t0:.0f} s")


def merge_shards(args, name: str, out: Path):
    """Check that the shards cover every image exactly once, then score them like a single run."""
    from ibbi.evaluate.benchmark import evaluate_class_agnostic, evaluate_predictions

    n = args.merge
    root = Path(args.dataset_dir) if args.dataset_dir else ibbi.download_benchmark(args.splits, images=False)
    preds, meta, seconds = {}, None, 0.0
    for split in args.splits:
        recs = [json.loads((out / "shards" / f"{split}.{i}of{n}.json").read_text()) for i in range(n)]
        ids = [x for r in recs for x in r["image_ids"]]
        expected = {im["id"] for im in json.loads((root / "detection" / "annotations_coco" / f"{split}.json").read_text())["images"]}
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise SystemExit(f"{split}: shards cover {len(set(ids))} of {len(expected)} images ({len(ids) - len(set(ids))} duplicates)")
        preds[split] = [p for r in recs for p in r["predictions"]]
        (out / f"{split}_predictions.json").write_text(json.dumps(preds[split]))
        meta = recs[0]
        seconds += sum(r["seconds"] for r in recs)
    res = {
        "model": args.model,
        "env": meta["env"],
        "dataset_revision": ibbi.utils.data.BENCHMARK_REVISION,
        "splits": args.splits,
        "sharded": n,
        "seconds": seconds,
        "benchmark_kwargs": meta["benchmark_kwargs"],
        "operating_conf": meta["operating_conf"],
    }
    if meta["is_species_level"]:
        r = evaluate_predictions(preds, root, output_dir=out, model_name=name)
    else:
        r = evaluate_class_agnostic(preds, root, operating_conf=meta["operating_conf"])
    res["benchmark"] = r
    res["headline"] = r["headline"]
    (out / "results.json").write_text(json.dumps(jsonable(res), indent=1))
    print(json.dumps(jsonable(res["headline"]), indent=1))


if __name__ == "__main__":
    main()
