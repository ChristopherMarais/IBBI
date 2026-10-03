# Models

Every model is created with `ibbi.create_model(name)`. The IBBI-trained weights are hosted in the
[IBBI-bio](https://huggingface.co/IBBI-bio) organisation; each repository has a model card with training details,
benchmark results and licence terms. Benchmark numbers: [benchmark.md](benchmark.md).

## Species detectors (one step: find and name 65 species)

| Name | Architecture | Parameters | Weights |
|---|---|---|---|
| `yolov8x_species_detector` | YOLOv8x | 68.2 M | [ibbi_yolov8x_species_detector](https://huggingface.co/IBBI-bio/ibbi_yolov8x_species_detector) |
| `yolov9e_species_detector` | YOLOv9e | 58.2 M | [ibbi_yolov9e_species_detector](https://huggingface.co/IBBI-bio/ibbi_yolov9e_species_detector) |
| `yolov10x_species_detector` | YOLOv10x | 31.8 M | [ibbi_yolov10x_species_detector](https://huggingface.co/IBBI-bio/ibbi_yolov10x_species_detector) |
| `yolo11x_species_detector` | YOLO11x | 56.9 M | [ibbi_yolo11x_species_detector](https://huggingface.co/IBBI-bio/ibbi_yolo11x_species_detector) |
| `yolo12x_species_detector` | YOLO12x | 59.2 M | [ibbi_yolo12x_species_detector](https://huggingface.co/IBBI-bio/ibbi_yolo12x_species_detector) |
| `rtdetrx_species_detector` | RT-DETR-X | 67.4 M | [ibbi_rtdetrx_species_detector](https://huggingface.co/IBBI-bio/ibbi_rtdetrx_species_detector) |

* **Training:** the benchmark's `train` split (6,096 images, 53,040 specimens, 65 species), 640 px, 100 epochs with a
  fixed budget, from the Ultralytics COCO checkpoints, Ultralytics 8.3.139.
* **Selection:** three seeds per architecture; the seed with the best Ultralytics validation fitness is shipped. The
  validation split of that training suite was the benchmark's `iid_test`, so the shipped seeds' `iid_test` numbers are
  slightly optimistic (the seed spread is shown in [benchmark.md](benchmark.md#seeds-of-the-species-detectors)).
* **Limits:** they always name one of their 65 species, including for species they have never seen.
* **Licence:** AGPL-3.0 (Ultralytics).

## Arthropod detector

`yolo11x_arthropod_detector` (alias `arthropod_detector`, `beetle_detector`): YOLO11x, single class `arthropod`,
1024 px input, 56.9 M parameters.

* **Training:** the IBBI arthropod detection corpus v3: 307,421 images and 889,283 boxes from 14 datasets in five
  environment categories (lab specimens and scans, light traps, sticky cards and pitfall trays, camera traps over
  vegetation, field photographs); training split 239,477 images with square-root source balancing; 20 epochs, batch 32,
  fp32.
* **Selection:** epoch 20, chosen before any test by the highest recall at ≤ 0.2 false alarms per image on the corpus
  validation sample.
* **Operating confidence:** `operating_conf = 0.70` (≤ 0.2 false alarms per image on exhaustively labelled validation
  images). The default `conf` of `predict` is 0.25.
* **Licence:** AGPL-3.0 (Ultralytics). Training data include iNaturalist 2017 (non-commercial research and education
  only) and IP102 (academic use only): the weights are released for non-commercial research.

## Hierarchical classifiers

| Name | Backbone | Input | Parameters | Weights | Licence |
|---|---|---|---|---|---|
| `dinov3_hierarchical_classifier` (default) | DINOv3 ViT-L/16 (LVD-1689M) | 336 px | 303 M | [ibbi_dinov3l_hierarchical_classifier](https://huggingface.co/IBBI-bio/ibbi_dinov3l_hierarchical_classifier) | DINOv3 License |
| `bioclip2_hierarchical_classifier` | BioCLIP 2 ViT-L/14 | 224 px | 304 M | [ibbi_bioclip2_hierarchical_classifier](https://huggingface.co/IBBI-bio/ibbi_bioclip2_hierarchical_classifier) | MIT |

* **Output:** for each crop, the subfamily, tribe, genus and species with a calibrated probability, a novelty score and
  a known / unsure flag per level, and the reported identification at the deepest trusted level.
* **Head:** LayerNorm + one linear layer giving one conditional logit per taxonomy node; P(node | parent) is a softmax
  over the siblings, the species probability is the product along the lineage, and each level's probability is the sum
  over its species, so the levels are always consistent.
* **Training:** benchmark `train` crops of the 65 species (minus a validation set of whole specimen groups), hierarchical
  cross-entropy with logit adjustment and a square-root sampler, non-beetle outlier exposure (arthropod crops from the
  detector corpus). The DINOv3 model also uses jointly trained synthetic outliers (NPOS) and a hierarchical supervised
  contrastive loss; it is the classifier deployed on the Bark and Ambrosia Gallery.
* **Calibration and novelty:** per-level temperatures fitted on the validation set. Novelty score per level = mean
  percentile, among known validation beetles, of the max calibrated probability (subfamily, tribe, genus) and of the
  max calibrated probability and the cosine similarity to the 5th nearest training embedding (species). Thresholds are
  set on known validation beetles only; no real unknown beetles were used.
* **Selection:** the DINOv3 configuration (336 px, NPOS, outlier exposure, contrastive loss, seed 0) was chosen by a
  validation rule among 17 trained runs; BioCLIP 2 is the best run of the second backbone family.
* **Licences:** the DINOv3 classifier is a derivative of DINOv3 and is distributed under the DINOv3 License (text in its
  repository; publications must acknowledge DINOv3). The BioCLIP 2 classifier is MIT. Training images include CC BY-NC
  benchmark images and iNaturalist 2017 / IP102 non-beetle crops; non-commercial research use is recommended.

## Zero-shot detectors

| Name | Model | Parameters | Default prompts | Tiling | Operating conf. | Upstream licence |
|---|---|---|---|---|---|---|
| `grounding_dino_zero_shot_detector` (alias `zero_shot_detector`) | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base) | 232 M | 13 arthropod taxa | 1024 px | 0.60 | Apache-2.0 |
| `owlv2_zero_shot_detector` | [google/owlv2-large-patch14-ensemble](https://huggingface.co/google/owlv2-large-patch14-ensemble) | 438 M | "a photo of an insect" | 1024 px | 0.725 | Apache-2.0 |
| `yoloworld_zero_shot_detector` | Ultralytics yolov8x-worldv2 (1024 px) | 224 M | 13 arthropod taxa | 1024 px | 0.50 | AGPL-3.0 |
| `sam3_zero_shot_detector` | [facebook/sam3](https://huggingface.co/facebook/sam3) (gated) | 840 M | insect, spider, arthropod | 1024 px | 0.95 | SAM License |

The 13 taxa: insect, spider, beetle, moth, fly, bee, ant, wasp, butterfly, caterpillar, mite, springtail, bug. Prompt set
and tiling were chosen per model on the 1,813-image validation sample of the arthropod detection corpus (best AP); the
operating confidence gives ≤ 0.2 false alarms per image there. Weights are downloaded from their authors.
"Tiling" means the image is also processed in overlapping 1024 px tiles (plus the whole image) and the boxes merged
with non-maximum suppression, so that small specimens on large trays are found; `tile=0` turns it off. Parameter
counts include the text encoders.

## Removed in v0.3

The v0.2 models (`*_bb_detect_model`, `*_bb_multi_class_detect_model`, `grounding_dino_detect_model`,
`yoloworldv2_bb_detect_model` and the untrained `*_features_model` extractors) were trained on or evaluated with the
retired IBBI test data and are no longer part of the package. Their repositories remain on the Hub for older versions.
