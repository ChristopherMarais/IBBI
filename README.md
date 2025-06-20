# Intelligent Bark Beetle Identifier (IBBI)

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
  - [Workflow: How the Models Were Built](#workflow-how-the-models-were-built)
  - [Package API and Usage](#package-api-and-usage)
    - [Usage Examples](#usage-examples)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Available Models](#available-models)
  - [How to Contribute](#how-to-contribute)
  - [License](#license)

---

## Workflow: How the Models Were Built

The models in the `ibbi` package are the result of a comprehensive data collection, annotation, and training pipeline. The following diagram illustrates the major components of this workflow.

<p align="center">
  <img src="docs/assets/images/data_flow_ibbi.png" alt="IBBI Data Flow" width="800">
</p>

1.  **Data Collection and Zero-Shot Detection:** The process begins with data collection from various sources. A zero-shot detection model is used to perform initial localization of beetles in the images. This is followed by a human-in-the-loop verification process to ensure the accuracy of the bounding box annotations. The data is then split into training and testing sets.

2.  **Model Training Data Curation:** The annotated dataset is curated to create specific training sets for different models:
    * **Single-class Object Detection Models:** All images with verified localization annotations are used to train the object detection models. This provides a large and diverse dataset for learning to accurately locate bark beetles.
    * **Multi-class Object Detection Models:** Only images with both localization annotations and species-level labels are used for training the multi-class models. Furthermore, to ensure robust training and evaluation, species with fewer than 50 images are excluded. This results in a smaller, but more precisely labeled, dataset for the fine-grained classification task.

3.  **Evaluation and Deployment:**
    * The held-out **testing data** is used to evaluate the performance of all trained models. You can view the performance metrics for each model using the `ibbi.list_models()` function.
    * The final models are made available through the package, allowing users to easily download them and the testing dataset for their own research and applications.

---

## Package API and Usage

The `ibbi` package is designed to be simple and intuitive. The core functionalities, inputs, and outputs are summarized in the diagram below.

<p align="center">
  <img src="docs/assets/images/ibbi_inputs_outputs.png" alt="IBBI Inputs and Outputs" width="800">
</p>

The main components of the package API are:

* **Inputs**: The primary inputs are images (as file paths, URLs, or PIL/numpy objects) and the name of the desired model as a string.
* **`model.predict()`**: This is the main prediction function. For detection models, it returns bounding boxes, and for classification models, it returns species classifications.
* **`model.extract_features()`**: This method allows you to extract deep feature embeddings from the images, which can be used for tasks like clustering or similarity analysis.
* **Dataset and Model Functions**: The package includes helper functions to `get_dataset()` and `list_models()`, which provides a table of all available models and their performance metrics.

### Usage Examples

Here are some visual examples of what you can do with `ibbi`.

| Input Image                                        | Object Detection (`detector.predict()`)                  | Object Classification (`classifier.predict()`)            | Zero-Shot Classification (`zs_classifier.predict()`)          |
| -------------------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------- |
| ![Beetles](docs/assets/images/beetles.png) | ![Object Detection](docs/assets/images/beetles_od.png) | ![Object Classification](docs/assets/images/beetles_oc.png) | ![Zero-Shot Classification](docs/assets/images/beetles_zsoc.png) |

---

## Installation

This package requires PyTorch. For compatibility with your specific hardware (e.g., CUDA-enabled GPU), please install PyTorch *before* installing `ibbi`.

**1. Install PyTorch**

Follow the official instructions at **[pytorch.org](https://pytorch.org/get-started/locally/)** to install the correct version for your system (OS, package manager, and CUDA version).

**2. Install IBBI**

Once PyTorch is installed, you can install the package directly from PyPI:

```bash
pip install ibbi
```

---

## Quick Start

Using IBBI is straightforward. You can load a pre-trained model for either detection or classification and immediately use it for inference on your images.

```python
import ibbi
from PIL import Image

# Load an image
# image = Image.open("path/to/your/beetle_image.jpg")
# Or use a URL
image = "[https://raw.githubusercontent.com/christopher-marais/IBBI/main/docs/assets/images/beetles.png](https://raw.githubusercontent.com/christopher-marais/IBBI/main/docs/assets/images/beetles.png)"


# 1. Load a pretrained object detection model or a classification model
detector = ibbi.create_model("yolov10x_bb_detect_model", pretrained=True)
classifier = ibbi.create_model("yolov10x_bb_classify_model", pretrained=True)

# 2. Run prediction to get class probabilities and/or bounding boxes
# The results will be the detected bounding box coordinates, confidence scores, and class labels
detection_results = detector.predict(image)
classification_results = classifier.predict(image)

print("Detection Results:")
print(detection_results)


print("\nClassification Results:")
print(classification_results)


# 3. You can also extract deep features from all models for other tasks
# The results will be a tensor of features
features = classifier.extract_features(image)
print(f"\nExtracted feature vector shape: {features.shape}")

```

For a more detailed, hands-on demonstration, please see the example notebook located in the repository: [`notebooks/example.ipynb`](notebooks/example.ipynb).

---

## Available Models

The package provides a factory function `create_model()` to access pre-trained models from Hugging Face Hub. A detailed list of available models and their Hugging Face repositories can be found in the [`src/ibbi/data/ibbi_model_summary.csv`](src/ibbi/data/ibbi_model_summary.csv) file.

To see a list of available models directly from your Python environment, you can run:

```python
import ibbi
ibbi.list_models()
```

---

## How to Contribute

Contributions are welcome! If you would like to improve IBBI, please see the [Contributing Guide](docs/CONTRIBUTING.md).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
