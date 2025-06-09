# Usage Guide

This guide will walk you through the primary functionalities of the `ibbi` package.

## Installation

First, ensure you have PyTorch installed according to the official instructions at [pytorch.org](https://pytorch.org/get-started/locally/). Then, install `ibbi`:

```bash
pip install ibbi
```

## Basic Usage: Detection and Classification

The core of the package is the `ibbi.create_model()` function. You can use it to load either an object detection model or a species classification model.

### Object Detection

Object detection models are used to find the location (bounding box) of beetles in an image.

```python
import ibbi
from PIL import Image

# Load the pretrained object detection model
detector = ibbi.create_model("yolov10x_bb_detect_model", pretrained=True)

# Load an image
image = Image.open("path/to/your/beetle_image.jpg")

# Get bounding box predictions
results = detector.predict(image)

# The 'results' object contains the bounding boxes, scores, and labels.
# You can plot the results:
results[0].show()
```

### Species Classification

Classification models are used to predict the species of a beetle in a given image. For best results, these models expect images that are already cropped to the beetle itself.

```python
import ibbi
from PIL import Image

# Load the pretrained classification model
classifier = ibbi.create_model("yolov10x_bb_classify_model", pretrained=True)

# Load an image of a single beetle
image = Image.open("path/to/cropped_beetle.jpg")

# Get species predictions
results = classifier.predict(image)

# The 'results' object contains the class probabilities.
# You can see the top 5 predictions:
results[0].show()
print(results[0].names[results[0].probs.top5[0]])
```

## Advanced Usage: Feature Extraction

Both detector and classifier models can also be used to extract deep features (embeddings) from an image. These feature vectors can be used for other machine learning tasks, such as clustering morphologically similar species or building a visual search engine.

```python
# Assuming 'classifier' is a loaded model and 'image' is a loaded PIL Image
features = classifier.extract_features(image)

print(f"Extracted feature vector shape: {features.shape}")
# Example output: Extracted feature vector shape: torch.Size([1, 768])
```
This allows for advanced analyses beyond simple classification.
