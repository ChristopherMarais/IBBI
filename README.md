# Intelligent Bark Beetle Identifier (IBBI)

[![PyPI version](https://badge.fury.io/py/ibbi.svg)](https://badge.fury.io/py/ibbi)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://gcmarais.com/IBBI/)

**IBBI** is a Python package that provides a simple and unified interface for detecting and classifying bark and ambrosia beetles from images using state-of-the-art computer vision models.

This package is designed to support entomological research by automating the laborious task of beetle identification, enabling high-throughput data analysis for ecological studies, pest management, and biodiversity monitoring.

### Motivation

The ability to accurately identify bark and ambrosia beetles is critical for forest health and pest management. However, traditional methods face significant challenges:

* **They are slow and time-consuming.**
* **They require highly specialized expertise.**
* **They create a bottleneck for large-scale research.**

The IBBI package provides a powerful, modern solution to overcome these obstacles by using pre-trained, open-source models to automate detection and classification from images, lowering the barrier to entry for researchers.

### Key Features

* **Simple API:** Access powerful detection and classification models with a single function call: `ibbi.create_model()`.
* **Multiple Model Types:**
    * **Single-Class Detection:** Detect the presence of *any* beetle in an image.
    * **Multi-Class Species Detection:** Identify the species of a beetle from an image.
    * **Zero-Shot Detection:** Detect objects using a text prompt (e.g., "insect"), without prior training on that specific class.
* **Pre-trained Models:** Leverages pre-trained models hosted on the Hugging Face Hub for immediate use.
* **Model Explainability:** Understand model predictions using SHAP (SHapley Additive exPlanations) to visualize which parts of an image contribute to the identification.
* **Extensible:** Designed to easily incorporate new model architectures in the future.

---

## Table of Contents

- [Intelligent Bark Beetle Identifier (IBBI)](#intelligent-bark-beetle-identifier-ibbi)
    - [Motivation](#motivation)
    - [Key Features](#key-features)
  - [Table of Contents](#table-of-contents)
  - [Workflow: How the Models Were Built](#workflow-how-the-models-were-built)
  - [Package API and Usage](#package-api-and-usage)
    - [Usage Examples](#usage-examples)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Advanced Usage](#advanced-usage)
    - [Feature Extraction](#feature-extraction)
    - [Model Explainability with SHAP](#model-explainability-with-shap)
  - [Available Models](#available-models)
  - [How to Contribute](#how-to-contribute)
  - [License](#license)

---

## Workflow: How the Models Were Built

The models in `ibbi` are the result of a comprehensive data collection, annotation, and training pipeline.

<p align="center">
  <img src="docs/assets/images/data_flow_ibbi.png" alt="IBBI Data Flow" width="800">
</p>

1.  **Data Collection and Curation:** The process begins with data collection from various sources. A zero-shot detection model performs initial beetle localization, followed by human-in-the-loop verification to ensure accurate bounding box annotations.
2.  **Model-Specific Training Data:** The annotated dataset is curated for different model types:
    * **Single-Class Detection:** Models are trained on all images with verified beetle localizations.
    * **Multi-Class Species Detection:** Models are trained only on images with both verified localizations and species-level labels. To ensure robustness, species with fewer than 50 images are excluded.
3.  **Evaluation and Deployment:** A held-out test set is used to evaluate all models. Performance metrics can be viewed with `ibbi.list_models()`. Final models are deployed to the Hugging Face Hub for easy access.

---

## Package API and Usage

The `ibbi` package is designed to be simple and intuitive.

<p align="center">
  <img src="docs/assets/images/ibbi_inputs_outputs.png" alt="IBBI Inputs and Outputs" width="800">
</p>

The main components are:

* **Inputs**: Images (file paths, URLs, PIL/numpy objects) and a model name string.
* **`model.predict()`**: The main prediction function. Returns bounding boxes, scores, and class labels.
* **`model.extract_features()`**: Extracts deep feature embeddings from images for downstream tasks.
* **Helper Functions**: `get_dataset()` to load the training/test data, `list_models()` to see available models, and `explain_with_shap()` to visualize model predictions using SHAP.

### Usage Examples

Here are visual examples of what you can do with `ibbi`.

<table style="width: 100%; border: none;">
  <thead>
    <tr>
      <th style="width: 34%; text-align: center;">Input Image</th>
      <th style="width: 22%; text-align: center;">Single-Class Detection<br>(<code>yolov10x_bb_detect_model</code>)</th>
      <th style="width: 22%; text-align: center;">Multi-Class Species Detection<br>(<code>yolov10x_bb_multi_class_detect_model</code>)</th>
      <th style="width: 22%; text-align: center;">Zero-Shot Detection<br>(<code>grounding_dino_detect_model</code>)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center;"><img src="docs/assets/images/beetles.png" alt="Beetles" style="max-width: 100%;"></td>
      <td style="text-align: center;"><img src="docs/assets/images/beetles_od.png" alt="Object Detection" style="max-width: 100%;"></td>
      <td style="text-align: center;"><img src="docs/assets/images/beetles_oc.png" alt="Object Classification" style="max-width: 100%;"></td>
      <td style="text-align: center;"><img src="docs/assets/images/beetles_zsoc.png" alt="Zero-Shot Classification" style="max-width: 100%;"></td>
    </tr>
  </tbody>
</table>

---

## Installation

This package requires PyTorch. For compatibility with your specific hardware (e.g., CUDA-enabled GPU), please install PyTorch *before* installing `ibbi`.

**1. Install PyTorch**

Follow the official instructions at **[pytorch.org](https://pytorch.org/get-started/locally/)** to install the correct version for your system.

**2. Install IBBI**

Once PyTorch is installed, install the package from PyPI:

```bash
pip install ibbi
```

---

## Quick Start

Using `ibbi` is straightforward. Load a pre-trained model and immediately use it for inference.

```python
import ibbi
from PIL import Image

# Use a URL or local path
image_source = "IBBI/docs/assets/images/beetles.png"

# --- 1. Load All Model Types ---
species_detector = ibbi.create_model("yolov10x_bb_multi_class_detect_model", pretrained=True)
detector = ibbi.create_model("yolov10x_bb_detect_model", pretrained=True)
zs_detector = ibbi.create_model("grounding_dino_detect_model", pretrained=True)

# --- 2. Run Prediction for Each Model ---
species_results = species_detector.predict(image_source)
detection_results = detector.predict(image_source)
zs_results = zs_detector.predict(image_source, text_prompt="insect")

# --- 3. Print Results with Name and Confidence ---

# Example for Multi-Class Species Detection
print("--- Multi-Class Species Detection ---")
for box in species_results[0].boxes:
    class_name = species_results[0].names[int(box.cls)]
    confidence = float(box.conf)
    print(f"  - Detected '{class_name}' with confidence {confidence:.2f}")

# Example for Single-Class Detection
print("\n--- Single-Class Detection ---")
for box in detection_results[0].boxes:
    class_name = detection_results[0].names[int(box.cls)]
    confidence = float(box.conf)
    print(f"  - Detected '{class_name}' with confidence {confidence:.2f}")

# Example for Zero-Shot Detection
print("\n--- Zero-Shot Detection (prompt: 'insect') ---")
for box in zs_results[0].boxes:
    class_name = zs_results[0].names[int(box.cls)]
    confidence = float(box.conf)
    print(f"  - Detected '{class_name}' with confidence {confidence:.2f}")

# To visualize the bounding boxes, you can uncomment the following lines:
# species_results[0].show()
# detection_results[0].show()
# zs_results[0].show()

```

For more detailed, hands-on demonstrations, please see the example notebooks located in the [`notebooks/`](notebooks/) folder of the repository.

---

## Advanced Usage

### Feature Extraction

All models can extract deep feature embeddings from an image. These vectors are useful for downstream tasks like clustering or similarity analysis.

```python
# Assuming 'species_detector' is a loaded model
features = species_detector.extract_features(image_source)
print(f"Extracted feature vector shape: {features.shape}")
```

### Model Explainability with SHAP

Understand *why* a model made a certain prediction using SHAP. This is crucial for building trust and interpreting the model's decisions by highlighting which pixels were most influential.

```python
import ibbi

# Load a model
model = ibbi.create_model("yolov10x_bb_multi_class_detect_model", pretrained=True)

# Get images to explain and a background dataset
# This can be any dataset the model is applied to
explain_data = ibbi.get_dataset(split="train", streaming=True).take(5)
background_data = ibbi.get_dataset(split="train", streaming=True).skip(5).take(10)

# Generate explanations (this is computationally intensive)
shap_explanation = ibbi.explain_with_shap(
    model=model,
    explain_dataset=list(explain_data),
    background_dataset=list(background_data),
    num_explain_samples=1,
    num_background_samples=5
)

# Plot the explanation for the first image
ibbi.plot_shap_explanation(shap_explanation[0], model)
```

---

## Available Models

To see a list of available models and their performance metrics directly from Python, run:

```python
import ibbi

# Returns a pandas DataFrame
models_df = ibbi.list_models(as_df=True)
print(models_df)
```

A detailed list can also be found in the [`src/ibbi/data/ibbi_model_summary.csv`](src/ibbi/data/ibbi_model_summary.csv) file.

---

## How to Contribute

Contributions are welcome! If you would like to improve IBBI, please see the [Contributing Guide](docs/CONTRIBUTING.md).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
