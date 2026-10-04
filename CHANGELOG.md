# Changelog

## v0.3.0

A full update of the package around the Bark and Ambrosia Beetle Detection Benchmark v2.0.1. This release breaks
compatibility with v0.2.

### Data
* `ibbi.get_dataset(split)` loads a split (`train`, `iid_test`, `inat_test`, `semantic_ood`) of
  [IBBI-bio/bark-ambrosia-beetle-benchmark](https://huggingface.co/datasets/IBBI-bio/bark-ambrosia-beetle-benchmark),
  pinned to the v2.0.1 revision; only the requested split is downloaded and downloads resume through Hub rate limits.
  Items keep the `image` / `objects` layout and add `iscrowd`, category ids and the lineage of every specimen.
* New: `ibbi.download_benchmark()`, `ibbi.get_taxonomy()`, `ibbi.utils.data.taxonomic_distance_matrix()`.
* Deprecated datasets: `ibbi_test_data`, `ibbi_ood_data`, `ibbi_shap_dataset`. `get_ood_dataset()` returns
  `semantic_ood` with a `DeprecationWarning`; `get_shap_background_dataset()` samples the benchmark's `train` split.

### Models
* All v0.2 models are removed. New models (weights in new IBBI-bio repositories):
  * species detectors for 65 species: `yolov8x_`, `yolov9e_`, `yolov10x_`, `yolo11x_`, `yolo12x_`,
    `rtdetrx_species_detector`;
  * `yolo11x_arthropod_detector`, a universal single-class arthropod detector, and `codino_arthropod_detector`, a
    larger and more accurate one (Co-DINO with an EVA-02-L backbone, run on plain PyTorch; CC BY-NC 4.0);
  * `dinov3_hierarchical_classifier` and `bioclip2_hierarchical_classifier`: subfamily / tribe / genus / species with
    calibrated probabilities and per-level "known / unsure" decisions;
  * zero-shot detectors `grounding_dino_`, `owlv2_`, `yoloworld_`, `sam3_zero_shot_detector`.
* New `ibbi.create_pipeline()`: arthropod detector + hierarchical classifier.
* Every model implements `predict`, `predict_proba` and `extract_features`. `create_model()` loads trained weights by
  default (`pretrained=True`).
* Aliases: `arthropod_detector`, `species_detector`, `hierarchical_classifier` added; `beetle_detector`,
  `species_classifier`, `feature_extractor`, `zero_shot_detector` now point to the new models.
* `IBBI_MODELS_DIR` loads weights from a local folder.

### Evaluation and explainability
* Crowd-aware evaluation: the benchmark's reference evaluator is vendored (`ibbi.evaluate.benchmark`).
  `Evaluator.benchmark()` runs and scores any model on the benchmark; `Evaluator.hierarchical_classification()` scores
  the classifiers on specimen crops (per-level accuracy, calibration, novelty AUROC, over-commit);
  `Evaluator.embeddings()` uses taxonomic distance for the Mantel test.
* `Evaluator.object_classification()` is deprecated (its evaluator ignored crowd regions).
* LIME and SHAP work with every model through `predict_proba`.
* Benchmark results of every model: `docs/benchmark.md`, scripts in `benchmarks/`.

### Dependencies
* `ultralytics>=8.3.139,<8.4` (8.4 changes YOLOv10 and RT-DETR predictions), `transformers>=5`, `timm>=1.0.20`,
  `open-clip-torch`, `pycocotools`, `safetensors` added; `datasets`, `ipywidgets`, `hf-xet`, `slicer`, `tabulate`
  dropped.

### Licences
* Package code stays MIT. Model weights carry their own licences (AGPL-3.0 for the Ultralytics-trained detectors,
  DINOv3 License for the DINOv3 classifier, MIT for the BioCLIP 2 classifier); see the README and model cards.
