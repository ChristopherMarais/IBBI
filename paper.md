---
title: 'IBBI: A Python package for the detection and classification of bark and ambrosia beetles'
tags:
  - Python
  - PyTorch
  - computer vision
  - deep learning
  - entomology
  - object detection
  - classification
  - forest health
authors:
  - name: Christopher Marais
    orcid: 0000-0003-3409-0473
    affiliation: 1
  - name: Eric Kuo
    orcid: 0000-xxxx-xxxx-xxxx
    affiliation: 1
  - name: Raquel Dias
    orcid: 0000-0003-3680-6834
    affiliation: 2
  - name: Jiri Hulcr
    orcid: 0000-0002-8706-4618
    affiliation: 1
affiliations:
 - name: School of Forest, Fisheries, and Geomatics Sciences, University of Florida, Gainesville, FL, USA
   index: 1
 - name: Microbiology and Cell Science, University of Florida, Gainesville, FL, USA
   index: 2
date: 13 June 2025
bibliography: paper.bib
---

# Summary

The Intelligent Bark Beetle Identifier (IBBI) is an open-source Python package that provides a simple, unified interface for the automated detection and classification of bark and ambrosia beetles (Coleoptera: Curculionidae: Scolytinae and Platypodinae) from images. This package addresses a critical need in forest health and biodiversity research by leveraging state-of-the-art computer vision models to streamline a traditionally labor-intensive and expert-dependent task. The target audience for `ibbi` includes entomologists, ecologists, forest health professionals, and biodiversity researchers who require high-throughput identification of beetle specimens from field or lab images. By providing programmatic access to a suite of pre-trained models for object detection and species classification, `ibbi` empowers researchers to accelerate data analysis for ecological studies, invasive species monitoring, and biodiversity assessments. The core functionality is accessible through a single factory function, `create_model()`, which simplifies model loading and inference, making advanced deep learning techniques accessible to users without requiring specialized expertise in computer vision.

# Statement of Need

The accurate and timely identification of bark and ambrosia beetle species is fundamental to forest health monitoring and pest management worldwide. Many species are cryptic or morphologically similar (i.e., exhibit high homoplasy), making identification a significant bottleneck that requires highly specialized taxonomic expertise [@kirkendall2015; @hulcr2015morphology]. Traditional morphological identification methods are slow, not scalable for large ecological datasets, and can be prohibitively expensive, hindering rapid responses to invasive species outbreaks and broad-scale biodiversity research.

While computer vision offers a promising solution, a critical gap exists in the available software ecosystem. There is a lack of specialized, accessible, and benchmarked tools for bark beetle identification. `ibbi` fills this gap by providing a robust, easy-to-install Python package that:
1.  Focuses specifically on the taxonomically challenging and economically critical bark and ambrosia beetle groups.
2.  Offers a range of pre-trained, benchmarked models for different tasks (detection, classification, zero-shot detection).
3.  Standardizes model access through a simple, high-level API, lowering the barrier to entry for researchers.

By consolidating these functionalities into a single package, `ibbi` provides a foundational tool that enables reproducible, high-throughput research that was previously infeasible.

# State of the Field

Automated species identification using computer vision has seen significant progress, with deep learning models achieving expert-level accuracy in various biological domains [@valan2019; @hoye2021]. Large-scale projects, often powered by citizen-science datasets like iNaturalist, have successfully developed generalist models for identifying a wide range of common taxa [@horn2017]. However, tools specialized for the fine-grained identification of economically and ecologically critical insect groups like bark beetles are far less common.

While some studies have developed bespoke computer vision models for bark beetle classification, these models are typically created for a specific research question and are not released as accessible, reusable, or maintained software. Consequently, they are not universally benchmarked against one another, and their performance on different datasets is unknown. `ibbi` is novel in that it is, to our knowledge, the first software package to:
1.  Aggregate multiple deep learning architectures for bark beetle identification into a single, cohesive framework.
2.  Train these models on one of the largest and most taxonomically diverse datasets of bark and ambrosia beetles available.
3.  Provide a standardized method for benchmarking these models, allowing for direct performance comparisons.
4.  Distribute the models and inference code as an easy-to-install Python package, promoting transparency, reusability, and extensibility.

# Functionality

The `ibbi` package is designed for ease of use and flexibility. Its core functionality revolves around a factory function, `create_model()`, which handles model instantiation and the downloading of pre-trained weights from Hugging Face Hub.

### Core Tasks and API
The package supports three primary tasks:
- **Object Detection**: Identifies and localizes beetles in an image, returning bounding box coordinates. This is ideal for processing raw images from traps or collection sheets.
- **Object Classification**: Predicts the species of a beetle from an image, returning class probabilities. This works best on images cropped to a single specimen.
- **Feature Extraction**: Generates deep-learning feature embeddings (vectors) from an image, which can be used for downstream tasks like clustering, similarity analysis, or visual search engines.

A typical workflow is demonstrated below:
```python
import ibbi
from PIL import Image

# 1. List available models to see choices
ibbi.list_models()

# 2. Load a pre-trained model
# Use "yolov10x_bb_detect_model" for detection
# or "yolov10x_bb_classify_model" for classification
model = ibbi.create_model("yolov10x_bb_classify_model", pretrained=True)

# 3. Prepare an image (from path, URL, or PIL/Numpy object)
image = Image.open("path/to/beetle_image.jpg")

# 4. Run inference
results = model.predict(image) # Returns a list of prediction objects

# 5. Extract features
features = model.extract_features(image) # Returns a torch.Tensor
```

### Inputs, Outputs, and Hardware
- **Inputs**: The `predict` and `extract_features` methods accept image inputs as a file path, a URL, a `PIL.Image` object, or a `numpy.ndarray`.
- **Outputs**: The `predict` method returns a list of custom `Results` objects, which contain easily accessible attributes such as bounding boxes (`.boxes`), probability scores (`.probs`), and class names (`.names`).
- **Hardware & Limitations**: While `ibbi` can run on a standard CPU, a CUDA-enabled GPU is highly recommended for faster inference, especially for batch processing. The performance of the models is dependent on the quality and resolution of the input images. Classification models achieve the highest accuracy when used on images that are well-lit and tightly cropped around the specimen.

# Future Development Roadmap

`ibbi` is under active development, with a commitment to long-term support and expansion for at least the next two years. The future roadmap includes:
- **Model Expansion**: Integrating additional state-of-the-art model architectures (e.g., transformers-based vision models) into the package to provide users with more options for performance and speed.
- **Dataset Growth**: Continuously expanding the training dataset with more images and species to improve model accuracy and taxonomic coverage.
- **Advanced Evaluation**: Implementing explainable AI (XAI) techniques to better understand model predictions and performing clustering analyses on feature embeddings to explore morphological relationships between species.
- **Hierarchical Classification**: Building models that incorporate expert domain knowledge from beetle taxonomy to improve accuracy and handle out-of-distribution samples more gracefully.

# Acknowledgements

We thank the numerous collaborators, students, and technicians in the Hulcr lab at the University of Florida who have contributed to the collection, curation, and annotation of the image dataset that made this work possible. We also acknowledge [Funding Agency, Grant Number, etc. - to be added by authors] for their financial support.
