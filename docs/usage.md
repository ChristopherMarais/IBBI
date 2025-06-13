# Usage Guide

This guide will walk you through the primary functionalities of the `ibbi` package, from installation to prediction and feature extraction.

## Installation

First, ensure you have PyTorch installed according to the official instructions at [pytorch.org](https://pytorch.org/get-started/locally/). Then, install `ibbi` from PyPI:

```bash
pip install ibbi
```

---

## Core Functions

The `ibbi` package is designed for simplicity. Here are the core functions you will use.

### Listing Available Models

To see all available models and their performance metrics, use `ibbi.list_models()`.

```python
import ibbi

# Get the model list as a pandas DataFrame
models_df = ibbi.list_models(as_df=True)
print(models_df)
```

### Creating a Model

Load any model using `ibbi.create_model()` by passing its name.

```python
# Load an object detection model
detector = ibbi.create_model("yolov10x_bb_detect_model", pretrained=True)

# Load a species classification model
classifier = ibbi.create_model("yolov10x_bb_classify_model", pretrained=True)
```

---

## Prediction Examples

You can perform inference on images from a file path, a URL, or a PIL Image object.

### Example 1: Object Detection

Object detection models find the location (bounding boxes) of beetles in an image.

**Input Image:**
![Beetles](assets/images/beetles.png)

```python
import ibbi
from PIL import Image

detector = ibbi.create_model("yolov10x_bb_detect_model", pretrained=True)

# Use an image URL or local path
image_source = "[https://raw.githubusercontent.com/christopher-marais/IBBI/main/docs/assets/images/beetles.png](https://raw.githubusercontent.com/christopher-marais/IBBI/main/docs/assets/images/beetles.png)"

# Get bounding box predictions
results = detector.predict(image_source)

# The 'results' object contains bounding boxes, scores, and labels.
# To display the image with bounding boxes:
results[0].show()
```

**Detection Output:**
![Object Detection Example](assets/images/beetles_od.png)

### Example 2: Species Classification

Classification models predict the species of a beetle in an image. These models work best on images cropped around a single beetle.

```python
classifier = ibbi.create_model("yolov10x_bb_classify_model", pretrained=True)

# This assumes you have an image of a single, cropped beetle
# For this example, we'll use the same multi-beetle image
results = classifier.predict(image_source)

# The 'results' object contains class probabilities.
# To see the top prediction:
top_prediction_index = results[0].probs.top1
predicted_species = results[0].names[top_prediction_index]
print(f"Predicted Species: {predicted_species}")

# To display the top 5 predictions and their confidences:
results[0].show()
```

**Classification Output:**
![Object Classification Example](assets/images/beetles_oc.png)

### Example 3: Zero-Shot Detection

Zero-shot models can detect objects based on a text description, without being explicitly trained on that class.

```python
# Load the zero-shot detection model
zs_detector = ibbi.create_model("grounding_dino_detect_model", pretrained=True)

# Predict with a text prompt
results = zs_detector.predict(image_source, text_prompt="a beetle")
results[0].show()
```

**Zero-Shot Output:**
![Zero-shot Classification Example](assets/images/beetles_zs_oc.png)

---

## Advanced Usage

### Feature Extraction

All models can extract deep feature embeddings from an image. These are useful for downstream tasks like clustering or similarity analysis.

```python
# Assuming 'classifier' is a loaded model
features = classifier.extract_features(image_source)

print(f"Extracted feature vector shape: {features.shape}")
```

### Downloading the Test Dataset

The test dataset used to evaluate the models can be downloaded for your own research and validation.

```python
import ibbi

# Download the dataset (it will be saved to a local cache)
dataset_path = ibbi.download_dataset()
print(f"Dataset downloaded to: {dataset_path}")
```
