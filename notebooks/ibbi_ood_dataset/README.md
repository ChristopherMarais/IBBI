---
license: mit
dataset_info:
  features:
  - name: image
    dtype: image
  - name: objects
    sequence:
    - name: bbox
      sequence: float64
    - name: category
      dtype: string
  splits:
  - name: train
    num_examples: 2418
  # Note: num_bytes and download_size are updated automatically by the Hub
  # after the upload is complete. These are placeholder estimates.
  download_size: 6432490239
  dataset_size: 7697416830
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# Dataset Card for IBBI Out-of-Distribution (OOD) Dataset

## Dataset Summary

This dataset contains **out-of-distribution (OOD)** images of bark and ambrosia beetles, intended for evaluating the robustness and generalization capabilities of models from the `ibbi` Python package. The **121 species** included in this dataset were not part of the original training or in-distribution test sets for the models in the `ibbi` package.

This dataset is crucial for testing how well models can:
- **Reject unknown classes** (for multi-class classifiers) by assigning low confidence scores.
- **Generalize detection** to beetle species they have not been explicitly trained on (for single-class detectors).

The dataset can be loaded directly using the `ibbi` package:
```python
import ibbi
ood_dataset = ibbi.get_dataset(repo_id="IBBI-bio/ibbi_ood_dataset")
````

## Supported Tasks

  - **Out-of-Distribution Detection**: The primary purpose is to evaluate how models respond to unseen species.
  - **Object Detection**: Can be used to benchmark the generalization performance of beetle detectors.

## Dataset Structure

### Data Instances

Each instance in the dataset consists of an image and a corresponding set of object annotations.

**Example:**

```json
{
  "image": "<PIL.Image.Image image>",
  "objects": {
    "bbox": [
      [656.56, 1335.47, 80.53, 75.35]
    ],
    "category": [
      "Dendroctonus_frontalis"
    ]
  }
}
```

### Data Fields

  * `image`: A PIL Image object of a beetle specimen.
  * `objects`: A dictionary containing the annotation information for the image.
      * `bbox`: A list of bounding boxes. Each box is in the format `[x_min, y_min, width, height]`.
      * `category`: A list of string labels corresponding to the species in each bounding box.

### Data Splits

The dataset contains a single split, named `train` for compatibility with Hugging Face's default structure. This split serves as the official out-of-distribution evaluation set.

  * **`train`**: 2,418 images with annotations.

## Dataset Statistics

| Metric                         | Value         |
| ------------------------------ | ------------- |
| Total Images with Annotations  | 2,418         |
| Unique Species                 | 121           |
| Total Bounding Box Annotations | 12,884        |
| Average Bboxes per Image       | 5.33          |
| Average Image Dimensions (WxH) | 2146 x 1702   |

### Species Distribution

| Species                             | Annotation Count |
| ----------------------------------- | ---------------- |
| Pseudopityophthorus\_minutissimus    | 1220             |
| Dactylotrypes\_longicollis           | 1090             |
| Carphoborus\_bifurcus                | 1077             |
| Scolytus\_rugulosus                  | 867              |
| Hypothenemus\_seriatus               | 840              |
| Anisandrus\_maiche                   | 643              |
| Scolytus\_carpini                    | 620              |
| Cryphalus\_mangiferae                | 612              |
| Eidophelus\_fagi                     | 611              |
| Cryptocarenus\_seriatus              | 529              |
| Hylurgus\_micklitzi                  | 423              |
| Pityogenes\_hopkinsi                 | 418              |
| Anisandrus\_obesus                   | 363              |
| Pityophthorus\_pityographus          | 314              |
| Scolytus\_intricatus                 | 284              |
| Dendroctonus\_adjunctus              | 274              |
| Pityogenes\_japonicus                | 218              |
| Ernoporus\_tiliae                    | 213              |
| Hylocurus\_langstoni                 | 169              |
| Tomicus\_brevipilosus                | 72               |
| Xyleborus\_monographus               | 58               |
| Tomicus\_piniperda                   | 51               |
| Trypodendron\_lineatum               | 50               |
| Dendroctonus\_frontalis              | 44               |
| Xyleborus\_bispinatus                | 41               |
| Xylocleptes\_bispinus                | 39               |
| Trypodendron\_signatum               | 38               |
| Hypothenemus\_eruditus               | 38               |
| Polygraphus\_poligraphus             | 36               |
| Hypothenemus\_birmanus               | 36               |
| Ambrosiodmus\_rubricollis            | 35               |
| Chaetoptelius\_mundulus              | 34               |
| Xyleborus\_perforans                 | 34               |
| Euwallacea\_kuroshio                 | 33               |
| Xyleborus\_volvulus                  | 32               |
| Hylastes\_ater                       | 31               |
| Xyleborinus\_andrewesi               | 29               |
| Euwallacea\_interjectus              | 29               |
| Xyloterinus\_politus                 | 28               |
| Xyleborus\_pubescens                 | 28               |
| Gnathotrichus\_materiarius           | 27               |
| Dendroctonus\_ponderosae             | 25               |
| Xyleborinus\_attenuatus              | 25               |
| Ambrosiodmus\_lewisi                 | 24               |
| Cyclorhipidion\_bodoanum             | 23               |
| Xyleborus\_horridus                  | 23               |
| Hylastes\_gracilis                   | 22               |
| Crossotarsus\_mnizsechi              | 22               |
| Ips\_pini                            | 22               |
| Xyleborinus\_schaufussi              | 21               |
| Premnobius\_cavipennis               | 21               |
| Xyleborinus\_gracilis                | 20               |
| Xyleborinus\_artestriatus            | 19               |
| Stegomerus\_pygmaeus                 | 19               |
| Truncaudum\_agnatum                  | 19               |
| Euwallacea\_funereus                 | 18               |
| Dinoplatypus\_pallidus               | 18               |
| Scolytus\_ratzeburgi                 | 18               |
| Euwallacea\_wallacei                 | 18               |
| Dryocoetes\_confusus                 | 18               |
| Pityoborus\_comatus                  | 18               |
| Eccoptopterus\_spinosus              | 18               |
| Euwallacea\_similis                  | 18               |
| Hypothenemus\_obscurus               | 17               |
| Hypothenemus\_dissimilis             | 17               |
| Euwallacea\_posticus                 | 17               |
| Diuncus\_justus                      | 17               |
| Xyleborus\_spinulosus                | 17               |
| Leptoxyleborus\_sordicauda           | 17               |
| Ambrosiodmus\_hagedorni              | 16               |
| Hypothenemus\_javanus                | 16               |
| Microperus\_alpha                    | 16               |
| Cnestus\_bimaculatus                 | 16               |
| Ambrosiodmus\_obliquus               | 16               |
| Hadrodemius\_globus                  | 16               |
| Webbia\_pabo                         | 15               |
| Monarthrum\_lobatum                  | 15               |
| Dendroterus\_defectus                | 15               |
| Ambrosiodmus\_devexulus              | 15               |
| Heteroborips\_seriatus               | 15               |
| Diuncus\_papatrae                    | 15               |
| Debus\_pumilus                       | 15               |
| Hypothenemus\_atomus                 | 15               |
| Ambrosiodmus\_asperatus              | 14               |
| Platypus\_selysi                     | 14               |
| Hylastes\_cunicularius               | 14               |
| Diuncus\_haberkorni                  | 14               |
| Metacorthylus\_velutinus             | 14               |
| Monarthrum\_laterale                 | 14               |
| Scolytus\_mundus                     | 14               |
| Coptoborus\_pseudotenuis             | 14               |
| Monarthrum\_dentigerum               | 14               |
| Coptoborus\_coartatus                | 14               |
| Scolytus\_aztecus                    | 14               |
| Truncaudum\_impexum                  | 14               |
| Cnesinus\_strigicollis               | 13               |
| Tricosa\_metacuneolus                | 13               |
| Procryphalus\_mucronatus             | 13               |
| Debus\_emarginatus                   | 13               |
| Xyleborus\_bidentatus                | 13               |
| Xyleborus\_impressus                 | 13               |
| Scolytus\_dimidiatus                 | 13               |
| Ambrosiodmus\_tachygraphus           | 13               |
| Crypturgus\_hispidulus               | 13               |
| Carphoborus\_bicornis                | 13               |
| Hypothenemus\_crudiae                | 13               |
| Microperus\_diversicolor             | 13               |
| Dendroctonus\_pseudotsugae           | 13               |
| Xyleborus\_xylographus               | 13               |
| Monarthrum\_huachucae                | 13               |
| Crossotarsus\_lacordairei            | 12               |
| Xyleborinus\_exiguus                 | 12               |
| Xyleborus\_intrusus                  | 12               |
| Pycnarthrum\_hispidum                | 12               |
| Microperus\_popondettae              | 12               |
| Pseudowebbia\_trepanicauda           | 12               |
| Pseudopityophthorus\_pruinosus       | 12               |
| Beaverium\_insulindicus              | 11               |
| Hylastes\_tenuis                     | 11               |
| Wallacellus\_piceus                  | 10               |
| Euwallacea\_destruens                | 10               |

## Dataset Creation

The images and annotations were curated by the Forest Entomology Lab at the University of Florida. Specifically, this dataset is composed of images that were filtered out during the creation of the primary `ibbi` multi-class training dataset. The species included here are those that had either fewer than 50 total bounding box annotations or were represented by fewer than 50 images, making them ideal for out-of-distribution testing. Bounding boxes and species labels were provided by expert annotators.

## Citation

If you use this dataset in your research, please cite the associated paper for the `ibbi` project:

```bibtex
@article{marais2025progress,
  title={Progress in developing a bark beetle identification tool},
  author={Marais, G Christopher and Stratton, Isabelle C and Johnson, Andrew J and Hulcr, Jiri},
  journal={PLoS One},
  volume={20},
  number={6},
  pages={e0310716},
  year={2025},
  publisher={Public Library of Science San Francisco, CA USA}
}
```
