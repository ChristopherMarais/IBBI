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
_Pending._
<!-- SPECIES_TABLE_END -->

## Class-agnostic detection (arthropod and zero-shot detectors)

<!-- CLASS_AGNOSTIC_TABLE_START -->
_Pending._
<!-- CLASS_AGNOSTIC_TABLE_END -->

## Hierarchical classifiers (ground-truth crops)

Tables: accuracy per level and depth metrics per split; novelty separation per level; behaviour on held-out species by
taxonomic band; accuracy over all known specimens (scored and crowd, 8× more specimens than the scored set); embeddings.

<!-- CLASSIFIER_TABLES_START -->
_Pending._
<!-- CLASSIFIER_TABLES_END -->

## Seeds of the species detectors

Each architecture was trained with three seeds in the benchmark suite; the package ships the seed with the best
Ultralytics validation fitness. **The validation split of that suite was `iid_test`**, so the choice of the best epoch and
of the seed used `iid_test`, and the shipped seed's `iid_test` numbers are slightly optimistic. The table (reference
evaluator, suite runs at batch 16) shows the seed-to-seed spread so the effect can be judged. Numbers through `ibbi`
(batch 1, the table above) differ from the suite's by up to about ±0.015 AP because Ultralytics pads batched images
differently.

<!-- SEED_TABLE_START -->
_Pending._
<!-- SEED_TABLE_END -->

## Caveats

* **The arthropod detector has seen most benchmark images.** About 99% of the benchmark's images come from the Bark and
  Ambrosia Gallery, which is one of the sources of the arthropod detection corpus, many of them in its training split.
  Its detection recall on the benchmark, and therefore the pipelines' detection recall, is optimistic. Classification
  given a detection is not affected: the classifiers never trained on `iid_test` or `semantic_ood`.
* **Field photographs** (`inat_test`) are a weak domain for every model, and the split is small (80 specimens).
* **Unseen genera are the hard case** for the hierarchical classifiers: no novelty score reaches a high AUROC at genus
  level, so unseen species and genera are still named at an impossible depth a large share of the time. The non-beetle
  case is easy (not measurable on this benchmark, which contains only beetles; see the model cards).
* **Ultralytics version.** Results were produced with Ultralytics 8.3.253, the version range the package requires.
  Ultralytics 8.4 changes YOLOv10 and RT-DETR predictions (iid AP 0.561 → 0.505 and 0.535 → 0.599 respectively).
