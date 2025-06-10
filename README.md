# Intelligent Bark Beetle Identifier (IBBI)

<!-- [![JOSS submission](https://joss.theoj.org/papers/10.21105/joss.01234/status.svg)](https://joss.theoj.org/papers/10.21105/joss.01234) -->
[![PyPI version](https://badge.fury.io/py/ibbi.svg)](https://badge.fury.io/py/ibbi)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**IBBI** is a Python package that provides a simple and unified interface for detecting and classifying bark and ambrosia beetles from images using state-of-the-art computer vision models.

This package is designed to support entomological research by automating the laborious task of beetle identification, enabling high-throughput data analysis for ecological studies, pest management, and biodiversity monitoring. The core models are built on multiple different architectures and are made easily accessible through a simple Python API.

### Motivation

The ability to accurately identify bark and ambrosia beetles is critical for forest health and pest management. However, traditional methods face significant challenges:

* **They are slow and time-consuming.**
* **They require highly specialized expertise.**
* **They create a bottleneck for large-scale research.**

The IBBI package provides a powerful, modern solution to overcome these obstacles:

* It uses **pre-trained, open-source models** for rapid analysis.
* It **automates both detection and classification** from images.
* It **lowers the barrier to entry**, enabling faster and more extensive data collection for all researchers.

---

## Table of Contents

- [Intelligent Bark Beetle Identifier (IBBI)](#intelligent-bark-beetle-identifier-ibbi)
    - [Motivation](#motivation)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Available Models](#available-models)
  - [Model Training Workflow](#model-training-workflow)
  - [How to Contribute](#how-to-contribute)
  - [License](#license)

---

## Installation

This package requires PyTorch. For compatibility with your specific hardware (e.g., CUDA-enabled GPU), please install PyTorch *before* installing `ibbi`.

**1. Install PyTorch**

Follow the official instructions at **[pytorch.org](https://pytorch.org/get-started/locally/)** to install the correct version for your system (OS, package manager, and CUDA version).

**2. Install IBBI**

Once PyTorch is installed, you can install the package directly from PyPI:

```bash
pip install ibbi
````

-----

## Quick Start

Using IBBI is straightforward. You can load a pre-trained model for either detection or classification and immediately use it for inference on your images.

```python
import ibbi
from PIL import Image

# Load an image
image = Image.open("path/to/your/beetle_image.jpg")

# 1. Load a pretrained object detection model or a classification model
detector = ibbi.create_model("yolov10x_bb_detect_model", pretrained=True)
classifier = ibbi.create_model("yolov10x_bb_classify_model", pretrained=True)

# 2. Run prediction to get class probabilities and/or bounding boxes
# The results will be the detected bounding box coordinates, confidence scores, and class labels
detection_results = detector.predict(image)
classification_results = classifier.predict(image)

# 3. You can also extract deep features from all models for other tasks
# The results will be a tensor of features
features = classifier.extract_features(image)

```

For a more detailed, hands-on demonstration, please see the example notebook located in the repository: `notebooks/example.ipynb`.

-----

## Available Models

The package provides a factory function `create_model()` to access the following pre-trained models from Huggingface Hub:

| Model Name                 | Task             | Pretrained Weights Repository              | Model Size (Params) | mAP@0.5 | mAP@[.5:.95] |
|----------------------------|------------------|--------------------------------------------|---------------------|---------|--------------|
| yolov10x_bb_detect_model   | Object Detection | ChristopherMarais/ibbi_yolov10_od_20250601 | 29.5M                | N/A     | N/A          |
| yolov10x_bb_classify_model | Classification   | ChristopherMarais/ibbi_yolov10_c_20250608  | 29.5M                 | N/A     | N/A          |

A detailed list of available models and their Hugging Face repositories can be found in the [ibbi_model_summary.csv](./docs/assets/data/ibbi_model_summary.csv) file.

-----

## Model Training Workflow

The models included in this package were trained using a standardized data flow on a dataset of bark and ambrosia beetle images from multiple different sources to include 63 different species. A list of all the species the current set of classification models were trained on can be found in the [ibbi_species_table.csv](./docs/assets/data/ibbi_species_table.csv) file.

<p align="center">
  <img src="./docs/assets/images/data_flow_ibbi.png" alt="My training workflow">
</p>

**1. Data Collection & Aggregation:**

  * An initial dataset of 54,421 images was compiled from diverse sources, including field photography from [barkbeetles.info](https://www.barkbeetles.info), lab-based specimen photography, and images from iNaturalist.
  * This aggregated dataset contained a mix of labeled and unlabeled images. Initially, 17,689 images had species-level labels, while 36,732 had no annotations.

**2. Annotation with Human-in-the-Loop:**

  * To create high-quality localization data, a zero-shot object detection model (GroundingDINO) was first used to generate preliminary bounding boxes for the beetles in the images.
  * Crucially, these automated annotations were then manually reviewed and refined by experts to ensure their accuracy and consistency, creating a reliable ground truth for training.

**3. Dataset Preparation and Splitting:**

  * A dedicated test set of 2,031 images was created by selecting images only from species with at least 50 representatives. This ensures a balanced and fair evaluation.
  * The remaining annotated data was split into two distinct training sets based on the task:
    * Object Detection Training Set (35,274 images): All images with verified bounding boxes (excluding the test set) were used to train the general beetle detection models. This larger dataset helps the models learn to accurately localize beetles under various conditions.
    * Classification Training Set (11,507 images): A filtered subset containing images with both verified bounding boxes and species-level labels was used to train the fine-grained classification models.

**4. Model Training and Fine-Tuning:**

  * Pre-trained model architectures were fine-tuned for each specific task:
    * Object Detection models were trained on the larger localization dataset to become expert beetle detectors.
    * Classification models were trained on the fully labeled dataset to specialize in identifying different beetle species.
  * Data augmentation techniques, including random rotations, scaling, color jitter, and mosaic augmentation, were used throughout training to improve model robustness and prevent over-fitting.

**5. Evaluation and Deployment:**

  * The performance of all trained models was rigorously measured against the held-out test set to ensure high accuracy for both detection and classification tasks.
  * The final, best-performing model weights were saved and uploaded to Hugging Face Hub, from where the `ibbi` package automatically downloads them, making them easily accessible to researchers via a simple API.

-----

## How to Contribute

Contributions are welcome\! If you would like to improve IBBI, please follow these steps:

1.  Clone this repository.
2.  Create a Conda environment and activate it:
    ```bash
    conda env create -f environment.yml
    conda activate IBBI
    ```
3.  Install dependencies using Poetry and set up pre-commit hooks:
    ```bash
    pip install torch torchvision torchaudio # Ensure PyTorch is installed first
    poetry config virtualenvs.create false --local
    poetry install
    poetry run pre-commit install
    ```
4.  Create a new branch for your feature or bug fix.
5.  Commit your changes and open a pull request.

To add new dependencies, use `poetry add <package-name>` for the main package or `poetry add --group dev <package-name>` for development dependencies.

-----

<!-- ## Citing IBBI

If you use IBBI in your research, please cite the JOSS paper.

**(Placeholder) To be added upon acceptance:**

> Marais, C., et al., (2025). IBBI: Intelligent Bark Beetle Identifier. Journal of Open Source Software, X(XX), XXXX. https://www.google.com/search?q=https://doi.org/XX.XXXXX/joss.XXXXX

You can also cite the specific version of the software archive using the DOI provided by Zenodo/figshare.

----- -->

## License

This project is licensed under the terms of the MIT License. See the `LICENSE` file for details.
