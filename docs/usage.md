# Usage guide

## Installation

IBBI needs Python ≥ 3.11 and PyTorch. Install PyTorch for your hardware first
([pytorch.org](https://pytorch.org/get-started/locally/)), then install the package.

**conda + pip**

```bash
conda create -n ibbi python=3.11
conda activate ibbi
# install PyTorch following pytorch.org for your system, then:
pip install ibbi
```

**pixi**

```bash
pixi init ibbi-project && cd ibbi-project
pixi add python=3.11 pip
pixi run pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # or the CPU wheel
pixi run pip install ibbi
pixi run python
```

#### Requirements and notes

* A CUDA GPU with ≥ 8 GB of memory is recommended; CPU inference works but is slow.
* `ultralytics` is restricted to `>=8.3.139,<8.4`: Ultralytics 8.4 changes the predictions of the YOLOv10 and RT-DETR
  checkpoints.
* SAM 3 is gated: accept its licence at https://huggingface.co/facebook/sam3 and run `hf auth login`.
* Files are cached in `~/.cache/ibbi` (`IBBI_CACHE_DIR` to change, `ibbi.clean_cache()` to delete). To use weights
  offline, put each model repository in a folder and set `IBBI_MODELS_DIR` to it.

---

## Models

```python
import ibbi

df = ibbi.list_models(as_df=True)          # one row per model, with benchmark numbers
df[df["Task"].str.contains("Zero-shot")]
```

Create a model by name or alias. Weights are downloaded from the [IBBI-bio](https://huggingface.co/IBBI-bio)
organisation the first time.

```python
detector   = ibbi.create_model("arthropod_detector")        # yolo11x_arthropod_detector
species    = ibbi.create_model("species_detector")          # yolo12x_species_detector
classifier = ibbi.create_model("hierarchical_classifier")   # dinov3_hierarchical_classifier
zero_shot  = ibbi.create_model("zero_shot_detector")        # grounding_dino_zero_shot_detector
other      = ibbi.create_model("rtdetrx_species_detector", device="cpu")
```

All models have the same core methods:

| Method | Returns |
|---|---|
| `predict(image or list)` | detectors: `{"boxes" (xyxy pixels), "scores", "labels"}`; classifier: a hierarchical record |
| `predict_proba(list of images)` | `[N, n_classes]` class scores (used by LIME / SHAP) |
| `extract_features(image)` | an embedding tensor |
| `get_classes()` | class names (species, "arthropod" or the current prompts) |

Images can be file paths, URLs, RGB numpy arrays or PIL images.

---

## Two-stage identification (recommended)

```python
pipe = ibbi.create_pipeline()                      # arthropod detector + DINOv3 hierarchical classifier
res = pipe.predict("plate.jpg")

res["boxes"]            # [[x1, y1, x2, y2], ...]
res["det_scores"]       # detector confidence
res["labels"]           # reported identification, e.g. "Xyleborus volvulus", "Euwallacea sp. (species undetermined)"
res["species"]          # the classifier's best species (always one of its 65)
res["classifications"]  # full record per detection (below)
```

Options: `ibbi.create_pipeline(detector="arthropod_detector", classifier="bioclip2_hierarchical_classifier",
det_conf=0.7, operating_point="0.95")`. `det_conf=detector.operating_conf` (0.70) gives at most ~0.2 false alarms per
image on the detector's validation sample; the default 0.25 finds more specimens.

### The hierarchical record

```python
rec = res["classifications"][0]
rec["reported"]          # human-readable answer
rec["depth"]             # 0 unrecognised, 1 subfamily, 2 tribe, 3 genus, 4 species
rec["genus"]             # {"taxon", "prob", "score", "known", "threshold", "entropy", "top3"}
rec["depth_by_op"]       # the depth under every operating point: {"0.90": 3, "0.95": 3, "0.99": 4, "gallery": 3}
```

* `prob` is the calibrated probability of the predicted taxon at that level; the levels are consistent
  (P(subfamily) ≥ P(tribe) ≥ P(genus) ≥ P(species) along the predicted lineage).
* `score` is the novelty score: the percentile of the specimen's familiarity among known validation beetles (higher =
  more familiar). `known` is `score >= threshold`.
* The reported depth is the deepest level such that every level above it is known.

### Operating points

| Operating point | Meaning |
|---|---|
| `gallery` (default) | accept 99% of known validation beetles at subfamily, 95% at tribe, genus and species |
| `0.90`, `0.95`, `0.99` | accept that share of known validation beetles at every level |

Higher acceptance means fewer abstentions on known species and more over-commitment on unknown ones. Choose with
`ibbi.create_model("hierarchical_classifier", operating_point="0.99")` or per call (`predict(..., operating_point=...)`).

---

## Hierarchical classifier on its own

```python
clf = ibbi.create_model("hierarchical_classifier")
rec = clf.predict("crop.jpg")                                   # one specimen per image
recs = clf.predict(["a.jpg", "b.jpg"])                          # several crops
recs = clf.predict("photo.jpg", boxes=[[10, 20, 300, 400]])     # crop boxes from a larger image (box + 5%)
probs = clf.predict_proba(["a.jpg"], level="genus")             # [1, 38] calibrated genus probabilities
clf.taxonomy_table                                              # the 65 species with subfamily / tribe / genus
```

## Species detectors

```python
det = ibbi.create_model("yolov10x_species_detector")
r = det.predict("plate.jpg", conf=0.25)
r["labels"], r["species"]                         # species names
det.predict("plate.jpg", level="genus")["labels"] # same boxes, labels mapped to genus
```

These always answer with one of their 65 species, also for species they have never seen; use the pipeline when
unknown species are possible.

## Zero-shot detectors

```python
zs = ibbi.create_model("owlv2_zero_shot_detector")              # default prompt "a photo of an insect", 1024 px tiles
zs.predict("trap.jpg")
zs.predict("trap.jpg", text_prompt="beetle . fly . moth")       # new prompts (labels = matching prompt)
zs.predict("trap.jpg", tile=0, conf=0.3)                        # whole image only, higher threshold
ibbi.create_model("sam3_zero_shot_detector", prompts=["beetle"])
```

## Embeddings

```python
fx = ibbi.create_model("feature_extractor")       # the DINOv3 classifier's embedding (2048-d)
v = fx.extract_features("crop.jpg")
```

---

## The benchmark dataset

```python
train = ibbi.get_dataset("train")                 # downloads only that split (resumes through Hub rate limits)
test = ibbi.get_dataset("iid_test")
ood = ibbi.get_dataset("semantic_ood")
item = test[0]
item["image"]                     # PIL image
item["objects"]["bbox"]           # [[x, y, w, h], ...] absolute pixels (COCO)
item["objects"]["category"]       # species names
item["objects"]["iscrowd"]        # 1 = real specimen that is not scored
item["objects"]["genus"]          # lineage of every specimen
small = test.select(range(20))    # subsets; test.shuffle(seed=0)
ibbi.get_taxonomy()               # all 175 species with their benchmark role and distance band
ibbi.download_benchmark()         # the whole dataset (~27 GB) at the pinned v2.0.1 revision
```

---

## Evaluation

```python
ev = ibbi.Evaluator(ibbi.create_model("species_detector"))
res = ev.benchmark()                                    # iid_test, inat_test, semantic_ood
res["headline"]                                         # flat dict of the key metrics
res["per_split"]["iid_test"]["decomposition"]           # detection recall, species accuracy given detection, ...
ev.benchmark(splits=["iid_test"], max_images=50)        # quick check (not comparable to published numbers)
ev.benchmark(output_dir="results/")                     # predictions + the reference evaluator's reports
```

* Species-level models (species detectors, pipelines) are scored with the benchmark's reference evaluator.
* Class-agnostic detectors (arthropod, zero-shot) get class-agnostic AP/AR and recall / precision / false alarms per
  image at their operating confidence.
* Scoring your own COCO predictions: `ibbi.evaluate.benchmark.evaluate_predictions({"iid_test": preds}, root)` and
  `evaluate_class_agnostic(...)`.

```python
clf_ev = ibbi.Evaluator(ibbi.create_model("hierarchical_classifier"))
h = clf_ev.hierarchical_classification()                # ground-truth crops of every split
h["per_split"]["iid_test"]["acc_species"]
h["per_split"]["semantic_ood"]["by_band"]               # over-commit on held-out species by taxonomic band
h["novelty"]["genus"]["auroc"]

emb = ibbi.Evaluator(ibbi.create_model("feature_extractor")).embeddings(ibbi.get_dataset("iid_test"))
emb["mantel_correlation"]                               # embedding distance vs taxonomic distance
```

---

## Explainability

```python
model = ibbi.create_model("hierarchical_classifier")
explainer = ibbi.Explainer(model)

# LIME: superpixels that raised or lowered the top classes of one image
explanation, image = explainer.with_lime(crop, image_size=(336, 336), num_samples=500, top_labels=3)
ibbi.plot_lime_explanation(explanation, image, top_k=1)

# SHAP: a background set is needed to simulate missing features
background = ibbi.get_shap_background_dataset(image_size=(336, 336), n_images=32)
values = explainer.with_shap([{"image": crop}], background, num_explain_samples=1, image_size=(336, 336), max_evals=500)
ibbi.plot_shap_explanation(values[0], model, top_k=3)
```

Detectors are explained through their highest confidence per class, zero-shot detectors per prompt
(`text_prompt=` sets the prompts), classifiers through their calibrated species probabilities.
