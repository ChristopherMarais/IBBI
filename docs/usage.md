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

To see all available models and their performance metrics, use `ibbi.list_models()`. This is useful for choosing the best model for your needs.

```python
import ibbi

# Get the model list as a pandas DataFrame
models_df = ibbi.list_models(as_df=True)
print(models_df)
```

### Creating a Model

Load any model from the list using `ibbi.create_model()` by passing its name.

```python
# Load a single-class object detection model
detector = ibbi.create_model("yolov10x_bb_detect_model", pretrained=True)

# Load a multi-class species detection model
classifier = ibbi.create_model("yolov10x_bb_multi_class_detect_model", pretrained=True)

# Load a zero-shot detection model
zs_detector = ibbi.create_model("grounding_dino_detect_model", pretrained=True)
```

---

## Prediction Examples

You can perform inference on images from a file path, a URL, or a PIL Image object.

### Example 1: Bark Beetle Detection (Single Class)

These models find the location (bounding boxes) of any beetle in an image, without identifying the species.

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

### Example 2: Species Detection (Multi-Class)

These models simultaneously find the location of beetles and predict their species.

```python
classifier = ibbi.create_model("yolov10x_bb_multi_class_detect_model", pretrained=True)

# Use the same multi-beetle image
results = classifier.predict(image_source)

# The 'results' object contains bounding boxes with predicted species.
# To see the top prediction for the first detected beetle:
first_box = results[0].boxes[0]
predicted_species_index = int(first_box.cls)
predicted_species = results[0].names[predicted_species_index]
confidence = float(first_box.conf)

print(f"Predicted Species: {predicted_species} with confidence: {confidence:.2f}")

# To display the image with bounding boxes and class labels:
results[0].show()
```

**Classification Output:**
![Object Classification Example](assets/images/beetles_oc.png)

### Example 3: Zero-Shot Detection

Zero-shot models can detect objects based on a text description, without being explicitly trained on that class. This is powerful for detecting objects not in the training data.

```python
# Load the zero-shot detection model
zs_detector = ibbi.create_model("grounding_dino_detect_model", pretrained=True)

# Predict with a text prompt
results = zs_detector.predict(image_source, text_prompt="a beetle")
results[0].show()
```

**Zero-Shot Output:**
![Zero-shot Classification Example](assets/images/beetles_zsoc.png)

---

## Advanced Usage

### Feature Extraction

All models can extract deep feature embeddings from an image. These vectors are useful for downstream tasks like clustering, similarity analysis, or training other machine learning models.

```python
# Assuming 'classifier' is a loaded model
features = classifier.extract_features(image_source)

print(f"Extracted feature vector shape: {features.shape}")
```

### Model Explainability with SHAP

Understand *why* a model made a certain prediction using SHAP (SHapley Additive exPlanations). This is crucial for building trust and interpreting the model's decisions by highlighting which pixels were most influential.

```python
import ibbi

# Load a model
model = ibbi.create_model("yolov10x_bb_multi_class_detect_model", pretrained=True)

# Get a few images to explain and a background dataset
# Note: Using more images for background_dataset provides better explanations
explain_data = ibbi.get_dataset(split="train", streaming=True).take(5)
background_data = ibbi.get_dataset(split="train", streaming=True).skip(5).take(10)

# Generate explanations (this is computationally intensive)
shap_explanation = ibbi.explain_with_shap(
    model=model,
    explain_dataset=list(explain_data),
    background_dataset=list(background_data),
    num_explain_samples=1, # Number of images to explain
    num_background_samples=5 # Number of background images to use
)

# Plot the explanation for the first image
ibbi.plot_shap_explanation(shap_explanation[0], model)
```

### Loading the Dataset

The dataset used to train and evaluate the models can be loaded for your own research and validation.

```python
import ibbi

# Load the dataset (it will be downloaded and cached locally)
# Set streaming=False to download the full dataset
dataset = ibbi.get_dataset(split="test", streaming=False)
print(f"Dataset loaded: {dataset}")

# You can also iterate through it without downloading everything
streaming_dataset = ibbi.get_dataset(split="train", streaming=True)
print(next(iter(streaming_dataset)))
