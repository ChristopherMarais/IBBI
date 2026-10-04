# Benchmark results

Every model in `ibbi` v0.3 was run through `ibbi.Evaluator` on the
[Bark and Ambrosia Beetle Detection Benchmark v2.0.1](https://huggingface.co/datasets/IBBI-bio/bark-ambrosia-beetle-benchmark)
(pinned revision `8dc5a58e`). The scripts are in [`benchmarks/`](https://github.com/ChristopherMarais/IBBI/tree/main/benchmarks):
`run_benchmark.py` runs one model, `make_tables.py` builds the tables below, and `benchmarks/results/` holds the headline
numbers of every run with the package versions and GPU used.

## Protocol

* **Splits.** `iid_test`: held-out specimens of the 65 trainable species (650 scored specimens; 4,367 more on the same
  images are crowd regions and are ignored). `inat_test`: iNaturalist field photographs of 8 trained species (80 scored).
  `semantic_ood`: 46,924 specimens of 110 species never seen in training, banded by taxonomic distance: `near_genus`
  (a trained genus), `mid_tribe` (a trained tribe, new genus), `far_tribe` (a new tribe).
* **Species-level models** (species detectors and pipelines) are scored with the benchmark's own crowd-aware reference
  evaluator (`evaluation/evaluate.py` of the dataset, vendored unchanged in `ibbi.evaluate`). Inference follows the
  benchmark protocol: detectors at confidence 0.001; pipelines with the arthropod detector at confidence 0.05 and at most
  100 detections per image (every detection is classified), each box scored as detector confidence × calibrated species
  probability and labelled with the classifier's best species.
* **Class-agnostic detectors** (arthropod detector, zero-shot detectors) are scored class-agnostically with
  pycocotools (crowd regions ignored), plus recall, precision and false alarms per image at IoU 0.5 at each model's
  operating confidence. Zero-shot models use the prompts and tiling chosen on the detector corpus validation sample;
  detections are kept down to confidence 0.02.
* **Hierarchical classifiers** are scored on ground-truth specimen crops (box + 5%), at the default `gallery` operating
  point. A level is *known* for a specimen when the classifier's label space contains its lineage down to that level;
  the *ideal depth* is the number of leading known levels; *over-commit* = reported depth > ideal depth (naming a taxon
  that cannot be right). Novelty AUROC per level: novelty score of specimens known at that level versus specimens unknown
  at that level, all splits pooled. Embedding metrics use all `iid_test` specimens (scored and crowd).
* **Nothing was tuned on the benchmark test splits** for any model in this table: settings are the models' defaults,
  chosen on validation data (with the exception noted for the species detectors below).

Metric names: *det. recall* = share of specimens found (IoU ≥ 0.5, any confidence); *species acc. | det.* = share of
found specimens given the right species; *species recall* = found and correctly named; *AR@100* = COCO average recall
with 100 detections per image (IoU 0.50:0.95); *AP* = COCO AP over IoU 0.50:0.95; *novelty AUROC* = how well the
detection confidence separates known from unseen species (reference evaluator).

## Species-level detection and identification

<!-- SPECIES_TABLE_START -->
| Model | iid AP | iid AP50 | iid det. recall | iid species acc. \| det. | iid species recall | iid genus acc. \| det. | iNat AP50 | iNat det. recall | OOD class-agn. AR@100 | OOD genus acc. \| det. | novelty AUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| yolov8x_species_detector | 0.497 | 0.514 | 0.929 | 0.586 | 0.545 | 0.627 | 0.079 | 0.163 | 0.732 | 0.169 | 0.692 |
| yolov9e_species_detector | 0.519 | 0.538 | 0.929 | 0.613 | 0.569 | 0.656 | 0.113 | 0.388 | 0.735 | 0.175 | 0.668 |
| yolov10x_species_detector | 0.561 | 0.582 | 0.932 | 0.604 | 0.563 | 0.655 | 0.166 | 0.450 | 0.724 | 0.163 | 0.664 |
| yolo11x_species_detector | 0.485 | 0.501 | 0.929 | 0.575 | 0.534 | 0.629 | 0.077 | 0.163 | 0.727 | 0.141 | 0.693 |
| yolo12x_species_detector | 0.490 | 0.508 | 0.931 | 0.577 | 0.537 | 0.628 | 0.156 | 0.312 | 0.723 | 0.129 | 0.702 |
| rtdetrx_species_detector | 0.535 | 0.557 | 0.937 | 0.647 | 0.606 | 0.701 | 0.035 | 0.388 | 0.739 | 0.221 | 0.569 |
| pipeline: arthropod detector + DINOv3 classifier | 0.718 | 0.743 | 0.932 | 0.858 | 0.800 | 0.919 | 0.336 | 0.350 | 0.762 | 0.322 | 0.820 |
| pipeline: arthropod detector + BioCLIP 2 classifier | 0.691 | 0.712 | 0.932 | 0.828 | 0.772 | 0.884 | 0.336 | 0.362 | 0.762 | 0.266 | 0.710 |
<!-- SPECIES_TABLE_END -->

## Class-agnostic detection (arthropod and zero-shot detectors)

<!-- CLASS_AGNOSTIC_TABLE_START -->
| Model | Split | AP | AP50 | AR@100 | max recall@0.5 | op. conf | recall@op | precision@op | false alarms / img |
|---|---|---|---|---|---|---|---|---|---|
| yolo11x_arthropod_detector | iid_test | 0.744 | 0.774 | 0.909 | 0.932 | 0.70 | 0.922 | 0.435 | 1.25 |
| yolo11x_arthropod_detector | inat_test | 0.932 | 0.996 | 0.945 | 1.000 | 0.70 | 0.963 | 0.987 | 0.01 |
| yolo11x_arthropod_detector | semantic_ood | 0.632 | 0.670 | 0.763 | 0.797 | 0.70 | 0.784 | 0.795 | 1.23 |
| grounding_dino_zero_shot_detector | iid_test | 0.294 | 0.395 | 0.801 | 0.931 | 0.60 | 0.391 | 0.419 | 0.57 |
| grounding_dino_zero_shot_detector | inat_test | 0.595 | 0.701 | 0.887 | 1.000 | 0.60 | 0.688 | 0.556 | 0.59 |
| grounding_dino_zero_shot_detector | semantic_ood | 0.243 | 0.448 | 0.518 | 0.789 | 0.60 | 0.237 | 0.637 | 0.82 |
| owlv2_zero_shot_detector | iid_test | 0.365 | 0.529 | 0.719 | 0.928 | 0.72 | 0.366 | 0.625 | 0.23 |
| owlv2_zero_shot_detector | inat_test | 0.531 | 0.771 | 0.764 | 1.000 | 0.72 | 0.662 | 0.716 | 0.28 |
| owlv2_zero_shot_detector | semantic_ood | 0.207 | 0.419 | 0.470 | 0.789 | 0.72 | 0.154 | 0.572 | 0.70 |
| yoloworld_zero_shot_detector | iid_test | 0.114 | 0.207 | 0.678 | 0.926 | 0.50 | 0.266 | 0.265 | 0.77 |
| yoloworld_zero_shot_detector | inat_test | 0.482 | 0.664 | 0.736 | 0.975 | 0.50 | 0.225 | 0.947 | 0.01 |
| yoloworld_zero_shot_detector | semantic_ood | 0.252 | 0.474 | 0.518 | 0.787 | 0.50 | 0.363 | 0.636 | 1.27 |
| sam3_zero_shot_detector | iid_test | 0.218 | 0.341 | 0.716 | 0.926 | 0.95 | 0.762 | 0.229 | 2.68 |
| sam3_zero_shot_detector | inat_test | 0.362 | 0.569 | 0.731 | 1.000 | 0.95 | 0.725 | 0.487 | 0.82 |
| sam3_zero_shot_detector | semantic_ood | 0.202 | 0.424 | 0.460 | 0.785 | 0.95 | 0.667 | 0.385 | 6.50 |
<!-- CLASS_AGNOSTIC_TABLE_END -->

## Hierarchical classifiers (ground-truth crops)

Tables: accuracy per level and depth metrics per split; novelty separation per level; behaviour on held-out species by
taxonomic band; accuracy over all known specimens (scored and crowd, 8× more specimens than the scored set); embeddings.

<!-- CLASSIFIER_TABLES_START -->
| Classifier | Split | n | subfamily | tribe | genus | species | species ECE | mean depth | over-commit | right depth & taxon |
|---|---|---|---|---|---|---|---|---|---|---|
| dinov3_hierarchical_classifier | iid_test | 650 | 0.989 | 0.949 | 0.891 | 0.835 | 0.043 | 3.45 | 0.000 | 0.718 |
| dinov3_hierarchical_classifier | inat_test | 80 | 0.925 | 0.562 | 0.487 | 0.362 | 0.210 | 1.65 | 0.000 | 0.200 |
| dinov3_hierarchical_classifier | semantic_ood | 46924 | 0.979 | 0.619 | 0.515 | – | – | 2.57 | 0.497 | 0.102 |
| bioclip2_hierarchical_classifier | iid_test | 650 | 0.994 | 0.945 | 0.871 | 0.814 | 0.048 | 3.45 | 0.000 | 0.718 |
| bioclip2_hierarchical_classifier | inat_test | 80 | 1.000 | 0.637 | 0.475 | 0.338 | 0.239 | 1.60 | 0.000 | 0.150 |
| bioclip2_hierarchical_classifier | semantic_ood | 46924 | 0.966 | 0.585 | 0.428 | – | – | 2.47 | 0.471 | 0.136 |

| Classifier | subfamily AUROC | tribe AUROC | genus AUROC | species AUROC | genus FPR@95 | species FPR@95 |
|---|---|---|---|---|---|---|
| dinov3_hierarchical_classifier | – | 0.772 | 0.561 | 0.739 | 0.881 | 0.925 |
| bioclip2_hierarchical_classifier | – | 0.811 | 0.625 | 0.653 | 0.904 | 0.978 |

| Classifier | Band | n | ideal depth | mean reported depth | over-commit | genus correct (when known) | tribe correct (when known) |
|---|---|---|---|---|---|---|---|
| dinov3_hierarchical_classifier | far_tribe | 6034 | 1.00 | 1.96 | 0.326 | nan | nan |
| dinov3_hierarchical_classifier | mid_tribe | 17099 | 2.00 | 2.37 | 0.449 | nan | 0.258 |
| dinov3_hierarchical_classifier | near_genus | 23791 | 3.00 | 2.88 | 0.575 | 0.515 | 0.878 |
| bioclip2_hierarchical_classifier | far_tribe | 6034 | 1.00 | 1.28 | 0.096 | nan | nan |
| bioclip2_hierarchical_classifier | mid_tribe | 17099 | 2.00 | 2.32 | 0.443 | nan | 0.254 |
| bioclip2_hierarchical_classifier | near_genus | 23791 | 3.00 | 2.89 | 0.585 | 0.428 | 0.822 |

| Classifier | Split | n (scored + crowd) | genus | species | known species named correctly |
|---|---|---|---|---|---|
| dinov3_hierarchical_classifier | iid_test | 5017 | 0.860 | 0.820 | 0.733 |
| dinov3_hierarchical_classifier | inat_test | 88 | 0.443 | 0.330 | 0.182 |
| bioclip2_hierarchical_classifier | iid_test | 5017 | 0.832 | 0.794 | 0.727 |
| bioclip2_hierarchical_classifier | inat_test | 88 | 0.455 | 0.330 | 0.136 |

| Classifier | Mantel r (embedding vs taxonomic distance) | p | ARI | NMI |
|---|---|---|---|---|
| dinov3_hierarchical_classifier | 0.542 | 0.001 | 0.885 | 0.886 |
| bioclip2_hierarchical_classifier | 0.739 | 0.001 | 0.878 | 0.881 |
<!-- CLASSIFIER_TABLES_END -->

## Seeds of the species detectors

Each architecture was trained with three seeds in the benchmark suite; the package ships the seed with the best
Ultralytics validation fitness. **The validation split of that suite was `iid_test`**, so the choice of the best epoch and
of the seed used `iid_test`, and the shipped seed's `iid_test` numbers are slightly optimistic. The table (reference
evaluator, suite runs at batch 16) shows the seed-to-seed spread so the effect can be judged. Numbers through `ibbi`
(batch 1, the table above) differ from the suite's by up to about ±0.015 AP because Ultralytics pads batched images
differently.

<!-- SEED_TABLE_START -->
| Architecture | Seed | iid AP | iid species acc. \| det. | OOD class-agn. AR@100 | shipped |
|---|---|---|---|---|---|
| yolov8x | 0 | 0.477 | 0.597 | 0.738 |  |
| yolov8x | 1 | 0.505 | 0.590 | 0.733 | yes |
| yolov8x | 2 | 0.481 | 0.569 | 0.732 |  |
| yolov9e | 0 | 0.503 | 0.607 | 0.737 | yes |
| yolov9e | 1 | 0.492 | 0.600 | 0.740 |  |
| yolov9e | 2 | 0.505 | 0.613 | 0.740 |  |
| yolov10x | 0 | 0.548 | 0.603 | 0.733 | yes |
| yolov10x | 1 | 0.541 | 0.597 | 0.732 |  |
| yolov10x | 2 | 0.523 | 0.570 | 0.737 |  |
| yolo11x | 0 | 0.494 | 0.584 | 0.742 |  |
| yolo11x | 1 | 0.477 | 0.571 | 0.737 | yes |
| yolo11x | 2 | 0.493 | 0.587 | 0.730 |  |
| yolo12x | 0 | 0.503 | 0.599 | 0.731 | yes |
| yolo12x | 1 | 0.504 | 0.603 | 0.740 |  |
| yolo12x | 2 | 0.491 | 0.581 | 0.738 |  |
| rtdetr-x | 0 | 0.501 | 0.612 | 0.730 |  |
| rtdetr-x | 1 | 0.509 | 0.620 | 0.738 |  |
| rtdetr-x | 2 | 0.530 | 0.645 | 0.739 | yes |
<!-- SEED_TABLE_END -->

## Caveats

* **The arthropod detector has seen most benchmark images.** About 99% of the benchmark's images come from the Bark and
  Ambrosia Gallery, which is one of the sources of the arthropod detection corpus, many of them in its training split.
  Its detection recall on the benchmark, and therefore the pipelines' detection recall, is optimistic. Classification
  given a detection is not affected: the classifiers never trained on `iid_test` or `semantic_ood`.
* **Operating confidences do not transfer exactly.** The arthropod and zero-shot detectors' operating confidences were
  set for at most ~0.2 false alarms per image on the detector corpus validation sample. On the benchmark most of them
  give more (1.2 per image for the arthropod detector on `iid_test` and `semantic_ood`; up to 6.5 for SAM 3 on
  `semantic_ood`), so set `conf` for your own images when false alarms matter. AP and AR do not depend on this choice.
* **Field photographs** (`inat_test`) are a weak domain for every model, and the split is small (80 specimens).
* **Unseen genera are the hard case** for the hierarchical classifiers: no novelty score reaches a high AUROC at genus
  level, so unseen species and genera are still named at an impossible depth a large share of the time. The non-beetle
  case is easy (not measurable on this benchmark, which contains only beetles; see the model cards).
* **Ultralytics version.** Results were produced with Ultralytics 8.3.253, the version range the package requires.
  Ultralytics 8.4 changes YOLOv10 and RT-DETR predictions (iid AP 0.561 → 0.505 and 0.535 → 0.599 respectively).
