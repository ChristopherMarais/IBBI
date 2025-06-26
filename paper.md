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
  - explainable AI
  - XAI
authors:
  - name: Christopher Marais
    orcid: 0000-0003-3409-0473
    affiliation: 1
  - name: Eric Kuo
    orcid: 0009-0003-4748-4241
    affiliation: 1
  - name: Jiri Hulcr
    orcid: 0000-0002-8706-4618
    affiliation: 1
  - name: Raquel Dias
    orcid: 0000-0003-3680-6834
    affiliation: 2
affiliations:
 - name: School of Forest, Fisheries, and Geomatics Sciences, University of Florida, Gainesville, FL, USA
   index: 1
 - name: Microbiology and Cell Science, University of Florida, Gainesville, FL, USA
   index: 2
date: 25 June 2025
bibliography: paper.bib
---

# Summary

The Intelligent Bark Beetle Identifier (IBBI) is an open-source Python package that provides a simple, unified interface for the automated detection, classification, and interpretation of bark and ambrosia beetles (Coleoptera: Curculionidae: Scolytinae and Platypodinae) from images. This package addresses a critical need in forest health and biodiversity research by leveraging multiple state-of-the-art computer vision models to streamline a traditionally labor-intensive and expert-dependent task. The target audience for `ibbi` includes entomologists, ecologists, forest health professionals, and biodiversity researchers who require high-throughput identification of beetle specimens. By providing programmatic access to a suite of pre-trained models, utility functions for data exploration, and Explainable AI (XAI) tools for model interpretation, `ibbi` empowers researchers to accelerate data analysis, understand model behavior, and build trust in automated identification systems. The core functionality is accessible through a single factory function, `create_model()`, which simplifies model loading, inference, and analysis, making advanced deep learning techniques accessible to users without requiring specialized expertise.

# Statement of Need

The ability to accurately and rapidly identify bark and ambrosia beetles is critical for forest health, biosecurity, and pest management. These insects are important, abundant, and diverse, and they also include many high-impact tree pests and invasive species [@hulcr2024platform]. Their identification is challenging because of the sheer diversity of species (>6,000 spp.), their very small size (mean length ~2mm), and rampant homoplasy—the evolution of similar morphologies in unrelated taxa [@kirkendall2015; @hulcr2015morphology]. Traditional morphological identification methods face significant challenges: they are time-consuming, require highly specialized and underfunded expertise, and create a "taxonomic impediment" that bottlenecks large-scale research and prevents rapid responses to invasive pest outbreaks.

`ibbi` provides a modern solution by using pre-trained, open-source models to automate both detection and classification from images. It lowers the barrier to entry for researchers, enabling faster and more extensive data collection. The package is trained on the largest available set of images of bark and ambrosia beetles, far surpassing the holdings of any individual museum collection. `ibbi` fills a critical gap in the software ecosystem by providing a robust, easy-to-install Python package that is specifically focused on these taxonomically challenging and economically critical beetle groups, and by including tools for model transparency and interpretability.

# State of the Field

Automated species identification using computer vision has seen significant progress, with deep learning models achieving expert-level accuracy in various biological domains [@valan2019; @hoye2021]. While large-scale projects like iNaturalist have successfully developed generalist models [@horn2017], these models often struggle with specialized groups like bark and ambrosia beetles. Some bespoke models for bark beetle classification have been developed, but they are not typically released as reusable or maintained software, and often operate as "black boxes," making it difficult to understand their decision-making process. `ibbi` is novel in that it is, to our knowledge, the first software package to:

1.  Aggregate multiple deep learning architectures for bark beetle identification into a single, cohesive framework.
2.  Train these models on one of the largest and most taxonomically diverse datasets of bark and ambrosia beetles available.
3.  Provide a standardized method for benchmarking these models, allowing for direct performance comparisons.
4.  Integrate Explainable AI (XAI) techniques to provide visual explanations of model predictions, fostering trust and enabling error analysis.
5.  Distribute the models, inference code, and utility functions as an easy-to-install Python package, promoting transparency, reusability, and extensibility.

# Functionality

The `ibbi` package is designed for ease of use, flexibility, and transparency. Its functionality extends beyond simple prediction to include tools for data inspection, model understanding, and advanced analysis.

### Model Curation and Datasets

The models provided in `ibbi` are the result of a comprehensive data pipeline. The process involved collecting 54,421 images from diverse sources, including museum collections (barkbeetles.info), iNaturalist, and the USDA Forest Service's Early Detection and Rapid Response program. A zero-shot detection model (GroundingDINO, [@liu2023grounding]) was used for initial beetle localization, followed by a human-in-the-loop process to verify annotations.

A dedicated test set of 2,031 images was created, and the remaining data was split into an object detection training set (35,274 images) and a classification training set (11,507 images). The classification models were trained on 63 different species.

### Model and Data Transparency

To promote transparency and reproducibility, `ibbi` includes several helper functions that allow users to inspect the package's data and models directly. Users can programmatically view all available models and their performance metrics using `ibbi.list_models()`, which returns a table of model names, tasks, and validation scores. This allows for informed model selection based on the user's specific needs. Additionally, a complete list of the 63 species used for training the multi-class models is accessible via the `ibbi.get_species_table()` function.

### Core Tasks and API

The package's primary functionality is accessed via the `create_model()` factory function. A typical workflow involves loading a model, preparing an image, and running inference.

```python
import ibbi
from PIL import Image

# 1. List available models to see choices and performance metrics
print(ibbi.list_models())

# 2. Choose and load a pre-trained model by name
model_name = "yolov10x_bb_multi_class_detect_model"
model = ibbi.create_model(model_name, pretrained=True)

# 3. Prepare an image (from path, URL, or PIL/Numpy object)
image = Image.open("path/to/beetle_image.jpg")

# 4. Run inference
results = model.predict(image)

# 5. The Results object contains rich information
print(results.boxes)
print(results.probs)
print(results.names)
```

The package supports four primary tasks:

-   **Single-class Object Detection**: Identifies and localizes any bark and ambrosia beetles in an image, returning bounding box coordinates for each detected instance.
-   **Multi-class Object Detection**: Detects and classifies beetles to the species level from an image, returning bounding boxes, species names, and class probabilities. This currently supports 63 species.
-   **Zero-Shot Object Detection**: Detects arbitrary objects in an image based on a user-provided text prompt, using the GroundingDINO model [@liu2023grounding]. This is particularly useful for locating objects for which the models were not explicitly trained.
-   **Feature Extraction**: Generates deep-learning feature embeddings (vectors) from an image via the `extract_features()` method. These vectors can be used for downstream tasks like clustering, similarity analysis, or building custom classifiers.

### Explainable AI (XAI) for Model Interpretability

Understanding *why* a model makes a certain prediction is crucial for building trust and for scientific applications. `ibbi` integrates Explainable AI (XAI) techniques to provide insight into the decision-making process of its models. The package includes a module for generating SHAP (SHapley Additive exPlanations) visualizations [@lundberg2017]. These visualizations produce a heatmap over the input image, highlighting the pixels that contributed most to the model's prediction. This allows researchers to verify that the model is focusing on relevant morphological features (e.g., the beetle's elytra or pronotum) rather than confounding artifacts in the image background.

```python
# Continuing from the previous example...
# 6. Generate a SHAP explanation for the prediction
shap_explainer = ibbi.xai.SHAPExplainer(model)
shap_values = shap_explainer.explain(image)
shap_explainer.plot(shap_values) # Displays the explanation plot
```

### Inputs, Outputs, and Hardware
**Inputs**: The `predict` and `extract_features` methods accept image inputs as a file path, a URL, a `PIL.Image` object, or a `numpy.ndarray`.
**Outputs**: The `predict` method returns a list of custom `Results` objects, which contain easily accessible attributes such as bounding boxes (`.boxes`), probability scores (`.probs`), and class names (`.names`).
**Hardware & Limitations**: While `ibbi` can run on a standard CPU, a CUDA-enabled GPU is highly recommended for faster inference, especially for the XAI features. Classification models achieve the highest accuracy on well-lit images that are tightly cropped around the specimen.

### Future Development Roadmap
`ibbi` is under active development. The future roadmap includes expanding the model repository with more architectures, augmenting the training dataset with more images and species, and enhancing the package's evaluation framework to serve as a standardized benchmarking tool for other models in the field. We also plan to incorporate additional XAI techniques to further improve model transparency.

### Licensing, Authorship, and Conflicts of Interest

#### License
This software is licensed under the OSI-approved MIT License.

#### Authorship
All authors have made substantial contributions to the design and implementation of the software. C.M. led the software development and paper writing. E.K. assisted with data curation and model training. J.H. and R.D. provided the domain expertise, image datasets, and project supervision. All authors contributed to the editing of the final manuscript.

#### Conflicts of Interest
The authors declare that they have no conflicts of interest.

### Acknowledgements
We thank the numerous collaborators, students, and technicians in the Hulcr lab at the University of Florida who have contributed to the collection, curation, and annotation of the image dataset that made this work possible. The training set, and the project personnel, were supported by the National Science Foundation, the USDA Forest Service, and the Florida Forest Service.
