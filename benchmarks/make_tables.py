#!/usr/bin/env python3
"""
Build the benchmark tables of docs/benchmark.md and the summary in README.md from run_benchmark.py outputs.

    python benchmarks/make_tables.py --results <dir with <model>/results.json> [--suite <benchmark suite dir>]

Also copies each model's headline metrics to benchmarks/results/<model>.json (small files kept in the repository).
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECIES = [
    "yolov8x_species_detector",
    "yolov9e_species_detector",
    "yolov10x_species_detector",
    "yolo11x_species_detector",
    "yolo12x_species_detector",
    "rtdetrx_species_detector",
    "pipeline__dinov3_hierarchical_classifier",
    "pipeline__bioclip2_hierarchical_classifier",
]
CLASS_AGNOSTIC = [
    "yolo11x_arthropod_detector",
    "grounding_dino_zero_shot_detector",
    "owlv2_zero_shot_detector",
    "yoloworld_zero_shot_detector",
    "sam3_zero_shot_detector",
]
CLASSIFIERS = ["dinov3_hierarchical_classifier", "bioclip2_hierarchical_classifier"]
SUITE_ARCH = {
    "yolov8x_species_detector": "yolov8x",
    "yolov9e_species_detector": "yolov9e",
    "yolov10x_species_detector": "yolov10x",
    "yolo11x_species_detector": "yolo11x",
    "yolo12x_species_detector": "yolo12x",
    "rtdetrx_species_detector": "rtdetr-x",
}
NICE = {
    "pipeline__dinov3_hierarchical_classifier": "pipeline: arthropod detector + DINOv3 classifier",
    "pipeline__bioclip2_hierarchical_classifier": "pipeline: arthropod detector + BioCLIP 2 classifier",
}


def f(v, nd=3):
    return "–" if v is None else f"{v:.{nd}f}"


def load(results: Path, name: str):
    p = results / name / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def species_table(results):
    cols = [
        ("iid_test.AP_50_95", "iid AP"),
        ("iid_test.AP_50", "iid AP50"),
        ("iid_test.detection_recall", "iid det. recall"),
        ("iid_test.acc_species_given_det", "iid species acc. \\| det."),
        ("iid_test.species_recall", "iid species recall"),
        ("iid_test.acc_genus_given_det", "iid genus acc. \\| det."),
        ("inat_test.AP_50", "iNat AP50"),
        ("inat_test.detection_recall", "iNat det. recall"),
        ("semantic_ood.class_agnostic_AR_100", "OOD class-agn. AR@100"),
        ("semantic_ood.acc_genus_given_det", "OOD genus acc. \\| det."),
        ("summary.novelty_det_AUROC", "novelty AUROC"),
    ]
    out = ["| Model | " + " | ".join(c for _, c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for m in SPECIES:
        r = load(results, m)
        if r:
            h = r["headline"]
            out.append(f"| {NICE.get(m, m)} | " + " | ".join(f(h.get(k)) for k, _ in cols) + " |")
    return "\n".join(out)


def class_agnostic_table(results):
    out = ["| Model | Split | AP | AP50 | AR@100 | max recall@0.5 | op. conf | recall@op | precision@op | false alarms / img |", "|---" * 10 + "|"]
    for m in CLASS_AGNOSTIC:
        r = load(results, m)
        if not r:
            continue
        h, op = r["headline"], r.get("operating_conf")
        for s in ("iid_test", "inat_test", "semantic_ood"):
            g = lambda k, h=h, s=s: h.get(f"{s}.class_agnostic_{k}")  # noqa: E731
            if g("AP_50") is None:
                continue
            out.append(
                f"| {m} | {s} | {f(g('AP_50_95'))} | {f(g('AP_50'))} | {f(g('AR_100'))} | {f(g('max_recall_50'))} | {f(op, 2)} | "
                f"{f(g('recall_50_at_op'))} | {f(g('precision_50_at_op'))} | {f(g('fp_per_image_at_op'), 2)} |"
            )
    return "\n".join(out)


def classifier_tables(results):
    acc = [
        "| Classifier | Split | n | subfamily | tribe | genus | species | species ECE | mean depth | over-commit | right depth & taxon |",
        "|---" * 11 + "|",
    ]
    nov = ["| Classifier | subfamily AUROC | tribe AUROC | genus AUROC | species AUROC | genus FPR@95 | species FPR@95 |", "|---" * 7 + "|"]
    band = [
        "| Classifier | Band | n | ideal depth | mean reported depth | over-commit | genus correct (when known) | tribe correct (when known) |",
        "|---" * 8 + "|",
    ]
    allk = ["| Classifier | Split | n (scored + crowd) | genus | species | known species named correctly |", "|---" * 6 + "|"]
    emb = ["| Classifier | Mantel r (embedding vs taxonomic distance) | p | ARI | NMI |", "|---" * 5 + "|"]
    for m in CLASSIFIERS:
        r = load(results, m)
        if not r:
            continue
        h = r["hierarchical_scored"]
        for s, p in h["per_split"].items():
            acc.append(
                f"| {m} | {s} | {p['n']} | {f(p.get('acc_subfamily'))} | {f(p.get('acc_tribe'))} | {f(p.get('acc_genus'))} | {f(p.get('acc_species'))} | "
                f"{f(p.get('ece_species'))} | {f(p.get('mean_depth'), 2)} | {f(p.get('over_commit_rate'))} | {f(p.get('right_depth_and_taxon'))} |"
            )
            for b, v in p.get("by_band", {}).items():
                band.append(
                    f"| {m} | {b} | {v['n']} | {f(v['ideal_depth'], 2)} | {f(v['mean_depth'], 2)} | {f(v['over_commit_rate'])} | "
                    f"{f(v.get('correct_genus_when_known'))} | {f(v.get('correct_tribe_when_known'))} |"
                )
        n = h["novelty"]
        g = lambda lvl, k, n=n: n.get(lvl, {}).get(k)  # noqa: E731
        nov.append(
            f"| {m} | {f(g('subfamily', 'auroc'))} | {f(g('tribe', 'auroc'))} | {f(g('genus', 'auroc'))} | {f(g('species', 'auroc'))} | "
            f"{f(g('genus', 'fpr_at_95tpr'))} | {f(g('species', 'fpr_at_95tpr'))} |"
        )
        for s, p in r.get("hierarchical_all_known_specimens", {}).get("per_split", {}).items():
            allk.append(
                f"| {m} | {s} | {p['n']} | {f(p.get('acc_genus'))} | {f(p.get('acc_species'))} | {f(p.get('known_species_named_correctly'))} |"
            )
        e = r.get("embeddings_iid_test", {})
        mc = e.get("mantel_correlation") or {}
        ext = e.get("external_cluster_validation") or {}
        ari = _first(ext, "ARI")
        nmi = _first(ext, "NMI")
        emb.append(f"| {m} | {f(mc.get('r'))} | {f(mc.get('p_value'))} | {f(ari)} | {f(nmi)} |")
    return "\n\n".join(["\n".join(acc), "\n".join(nov), "\n".join(band), "\n".join(allk), "\n".join(emb)])


def _first(d, key):
    """External validation is stored as a one-row DataFrame dict ({col: {0: v}}) or a flat dict."""
    v = d.get(key)
    if isinstance(v, dict):
        v = next(iter(v.values()), None)
    return v


def seed_table(suite: Path):
    out = ["| Architecture | Seed | iid AP | iid species acc. \\| det. | OOD class-agn. AR@100 | shipped |", "|---" * 6 + "|"]
    sel = json.loads((ROOT / "benchmarks" / "species_detector_selection.json").read_text())
    for m, arch in SUITE_ARCH.items():
        for seed in range(3):
            p = suite / arch / "evaluations" / f"baseline_seed{seed}" / f"baseline_seed{seed}_summary.json"
            if not p.exists():
                continue
            s = json.loads(p.read_text())["per_split"]
            shipped = "yes" if sel.get(arch, {}).get("chosen", {}).get("seed") == seed else ""
            out.append(
                f"| {arch} | {seed} | {f(s['iid_test']['coco']['AP_50_95'])} | {f(s['iid_test']['decomposition']['acc_species_given_det'])} | "
                f"{f(s['semantic_ood']['class_agnostic']['class_agnostic_AR_100'])} | {shipped} |"
            )
    return "\n".join(out)


def summary_table(results):
    out = ["| Model | Headline (benchmark v2.0.1) |", "|---|---|"]
    for m in SPECIES:
        r = load(results, m)
        if r:
            h = r["headline"]
            out.append(
                f"| {NICE.get(m, m)} | iid AP {f(h.get('iid_test.AP_50_95'))}, detection recall {f(h.get('iid_test.detection_recall'))}, "
                f"species accuracy given detection {f(h.get('iid_test.acc_species_given_det'))}; unseen-species detection recall (AR@100) "
                f"{f(h.get('semantic_ood.class_agnostic_AR_100'))} |"
            )
    for m in CLASS_AGNOSTIC:
        r = load(results, m)
        if r:
            h = r["headline"]
            out.append(
                f"| {m} | class-agnostic AP: iid {f(h.get('iid_test.class_agnostic_AP_50_95'))}, iNat {f(h.get('inat_test.class_agnostic_AP_50_95'))}, "
                f"unseen species {f(h.get('semantic_ood.class_agnostic_AP_50_95'))} |"
            )
    for m in CLASSIFIERS:
        r = load(results, m)
        if r:
            p = r["hierarchical_scored"]["per_split"]
            n = r["hierarchical_scored"]["novelty"]
            out.append(
                f"| {m} | iid species / genus accuracy {f(p['iid_test'].get('acc_species'))} / {f(p['iid_test'].get('acc_genus'))}; "
                f"unseen species named at an impossible depth {f(p.get('semantic_ood', {}).get('over_commit_rate'))}; genus novelty AUROC "
                f"{f(n.get('genus', {}).get('auroc'))} |"
            )
    return "\n".join(out)


def replace_block(path: Path, tag: str, text: str):
    s = path.read_text()
    pat = re.compile(rf"(<!-- {tag}_START -->\n).*?(\n<!-- {tag}_END -->)", re.S)
    if not pat.search(s):
        raise SystemExit(f"marker {tag} not found in {path}")
    path.write_text(pat.sub(lambda m: m.group(1) + text + m.group(2), s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--suite", type=Path, default=Path("/blue/hulcr/gmarais/PhD/benchmark_testing/full_benchmark/benchmark_suite_b200"))
    a = ap.parse_args()
    keep = ROOT / "benchmarks" / "results"
    keep.mkdir(exist_ok=True)
    for d in sorted(a.results.iterdir()):
        r = load(a.results, d.name)
        if r:
            slim = {
                k: r[k] for k in ("model", "env", "dataset_revision", "splits", "headline", "seconds", "operating_conf", "benchmark_kwargs") if k in r
            }
            for k in ("hierarchical_scored", "hierarchical_all_known_specimens"):
                if k in r:
                    slim[k] = {kk: vv for kk, vv in r[k].items() if kk != "rows"}
            if "embeddings_iid_test" in r:
                slim["embeddings_iid_test"] = {
                    k: r["embeddings_iid_test"].get(k) for k in ("mantel_correlation", "external_cluster_validation", "internal_cluster_validation")
                }
            (keep / f"{d.name}.json").write_text(json.dumps(slim, indent=1))
    doc = ROOT / "docs" / "benchmark.md"
    replace_block(doc, "SPECIES_TABLE", species_table(a.results))
    replace_block(doc, "CLASS_AGNOSTIC_TABLE", class_agnostic_table(a.results))
    replace_block(doc, "CLASSIFIER_TABLES", classifier_tables(a.results))
    replace_block(doc, "SEED_TABLE", seed_table(a.suite))
    replace_block(ROOT / "README.md", "BENCHMARK_SUMMARY", summary_table(a.results))
    print("tables written")


if __name__ == "__main__":
    main()
