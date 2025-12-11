---
title: 'IBBI: A Python package for the detection and classification of bark and ambrosia beetles'
tags:
  - Python
  - PyTorch
  - computer vision
  - deep learning
  - entomology
  - object detection
  - object classification
  - forest health
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
date: 13 June 2025
bibliography: paper.bib
---

# Summary

The Intelligent Bark Beetle Identifier (IBBI) is an open-source Python package that provides a simple, unified interface for the automated detection and classification of bark and ambrosia beetles (Coleoptera: Curculionidae: Scolytinae and Platypodinae) from images. This package addresses a critical need in forest health and biodiversity research by leveraging multiple state-of-the-art computer vision models to streamline a traditionally labor-intensive and expert-dependent task. The target audience for `ibbi` includes entomologists, ecologists, forest health professionals, and biodiversity researchers who require high-throughput identification of beetle specimens from field or lab images. By providing programmatic access to a suite of pre-trained models for object-detection and species classification, `ibbi` empowers researchers to accelerate data analysis for ecological studies, invasive species monitoring, and biodiversity assessments. The core functionality is accessible through a single factory function, `create_model()`, which simplifies model loading and inference, making deep learning techniques accessible to users without requiring specialized expertise in computer vision.

# Statement of Need

The accurate and timely identification of bark and ambrosia beetle species is fundamental to forest health monitoring, pest management, and quarantine efforts worldwide [@hulcr2017ambrosia; @piper2019prospects; @ramsfield2016forest]. Many species are cryptic or morphologically similar (i.e., exhibit high homoplasy), making identification a significant bottleneck that requires highly specialized taxonomic expertise [@kirkendall2015; @hulcr2015morphology]. Traditional morphological identification methods are slow, not scalable for large ecological datasets, and can be prohibitively expensive, hindering rapid responses to invasive species outbreaks and broad-scale biodiversity research.

While computer vision offers a promising solution, a critical gap exists in the available software ecosystem. There is a lack of specialized, accessible, and benchmarked tools for bark beetle identification. `ibbi` fills this gap by providing a robust, easy-to-install Python package that:
1.  Focuses specifically on the taxonomically challenging and economically critical bark and ambrosia beetle groups.
2.  Offers a range of pre-trained, benchmarked models for different tasks (object-detection and object-classification, zero-shot object-detection).
3.  Standardizes model access through a simple, high-level API, lowering the barrier to entry for researchers.

By consolidating these functionalities into a single package, `ibbi` provides a foundational tool that enables reproducible, high-throughput research that was previously infeasible.

# State of the Field

Automated species identification using computer vision has seen significant progress, with deep learning models achieving expert-level accuracy in various biological domains [@valan2019; @hoye2021]. This progress has been largely driven by two trends: broad-scale generalist models and bespoke academic models.

Large-scale projects, often powered by citizen-science datasets like iNaturalist, have successfully developed generalist models for identifying a wide range of common taxa [@horn2018]. Similarly, comprehensive toolkits like Pytorch-Wildlife offer powerful, pre-trained models, such as MegaDetector, for identifying a broad spectrum of wildlife in camera trap images [@hernandez2024pytorch]. However, these generalist models, while powerful for common categories, often struggle with the fine-grained distinctions required for less common or cryptic species, particularly in specialized groups like bark and ambrosia beetles, where morphological similarities can lead to misidentification [@kirkendall2015].

This has led to a growing trend of developing specialized tools for specific taxonomic groups of high economic or ecological importance. For example, various research initiatives have focused on creating models for identifying bees to monitor pollinators or for classifying mosquito species to track disease vectors [@buschbacher2020image, @goodwin2021mosquito]. These efforts demonstrate the recognized need for domain-specific tools, but the resulting models are rarely distributed in an accessible, unified software package.

On the other end of the spectrum, numerous studies have developed bespoke computer vision models for bark beetle classification [@marais2025progress, @sun2024intelligent]. While valuable, these models are typically created for a specific research question and are not released as accessible, reusable, or maintained software. Consequently, they are not universally benchmarked, and their performance on different datasets is unknown.

ibbi is novel in that it bridges this gap. It is, to our knowledge, the first software package to:

1. Aggregate multiple deep learning architectures specifically for bark and ambrosia beetle identification into a single, cohesive framework.
2. Train these models on one of the largest and most taxonomically diverse datasets of bark and ambrosia beetles available.
3. Provide a standardized method for benchmarking these models, allowing for direct performance comparisons.
4. Distribute the models and inference code as an easy-to-install Python package, promoting transparency, reusability, and extensibility.

The first version of ibbi is made possible by powerful open-source libraries that provide state-of-the-art model implementations, such as timm [@rw2021timm] and ultralytics [@jocher2020yolov5], and frameworks such as HuggingFace for model and data sharing [@wolf2019huggingface].

By making these changes, you clearly position ibbi not as a replacement for tools like MegaDetector but as a necessary, specialized tool that addresses a gap those generalist models cannot fill, while also improving on the inaccessible nature of bespoke academic models.

# Functionality

The `ibbi` package is designed for ease of use and flexibility. Its core functionality revolves around a factory function, `create_model()`, which handles model instantiation and the downloading of pre-trained weights from Hugging Face Hub [@wolf2019huggingface].

### Model Curation and Datasets
The models provided in `ibbi` are the result of a comprehensive data pipeline (Figure 1). The process involved collecting images from diverse sources, using a zero-shot detection model for initial beetle localization, and employing a human-in-the-loop process to verify annotations. The curated data was then used to train the object-detection and object-classification models. To promote transparency and reproducibility, the package provides a helper function to access the testing dataset used for evaluation. Furthermore, users can programmatically view all available models and their performance metrics using the `ibbi.list_models()` function, allowing for informed model selection.

![Overview of the data collection, curation, and model training workflow for the `ibbi` package.](https://media.githubusercontent.com/media/ChristopherMarais/IBBI/main/docs/assets/images/data_flow_ibbi.png)
*Figure 1: Overview of the data collection, curation, and model training workflow for the `ibbi` package.*

### Core Tasks and API
The package supports four primary tasks:
- **Single-class Object-Detection**: Identifies and localizes any bark and ambrosia beetles in an image, returning bounding box coordinates. This is ideal for processing raw images from traps or collection sheets and is based on fine-tuned high-performance architectures like YOLO [@redmon2016] and detection transformers [@lv2023rtdetr].
- **Zero-Shot Object-Detection**: Detects arbitrary objects in an image based on a user-provided text prompt. This powerful feature uses the GroundingDINO model [@liu2023grounding] and allows for flexible analysis beyond predefined categories. However, this model has not been fine-tuned specifically on bark and ambrosia beetle images.
- **Multi-class Object-Detection**: Predicts the species of a beetle from an image, returning species class probabilities. This currently supports 63 species and is also based on fine-tuned high-performance architectures like YOLO [@redmon2016] and detection transformers [@lv2023rtdetr]. However, it is important to note that the models are not fine tuned on a dataset as large as the one used for single-class object-detection, and therefore may not perform as well on images that are not tightly cropped around the specimen. This task is the main focus for the future roadmap of the package, with plans to expand the number of species supported and improve model performance.
- **Feature Extraction**: Generates deep-learning feature embeddings (vectors) from an image. This task is available for models of all tasks. These feature embeddings can be used for downstream tasks like clustering, similarity analysis, or visual search engines.

### A typical Workflow:
```python
import ibbi
from PIL import Image

# 1. List available models to see choices and performance metrics
ibbi.list_models()

# 2. Choose and load a pre-trained model by name
# For this example, we'll use a species detection and classification model
model_name = "yolov10x_bb_multi_class_detect_model"
model = ibbi.create_model(model_name, pretrained=True)

# 3. Prepare an image (from path, URL, or PIL/Numpy object)
image = Image.open("path/to/beetle_image.jpg")

# 4. Run inference
results = model.predict(image)

# 5. Extract features
features = model.extract_features(image)
```

### Inputs, Outputs, and Hardware
- **Inputs**: The `predict` and `extract_features` methods accept image inputs as a file path, a URL, a `PIL.Image` object, or a `numpy.ndarray`.

- **Outputs**: The `predict` method returns a list of custom `Results` objects, which contain easily accessible attributes such as bounding boxes (`.boxes`), probability scores (`.probs`), and class names (`.names`).

- **Hardware & Limitations**: While `ibbi` can run on a standard CPU, a CUDA-enabled GPU is highly recommended for faster inference, especially for batch processing. The performance of the models is dependent on the quality and resolution of the input images. Object-classification models achieve the highest accuracy when used on images that are well-lit and tightly cropped around the specimen.

# Future Development Roadmap

`ibbi` is under active development with a commitment to long-term support and expansion. The future roadmap is centered on two key areas: expanding the core platform and developing a novel, specialized identification model.

### Platform and Ecosystem Expansion
- **Model repository expansion**: We are committed to continually expanding the `ibbi` model repository by fine-tuning additional state-of-the-art architectures on our dataset.
- **Dataset Collection Growth**: The training dataset itself will be perpetually augmented with more images and species to improve taxonomic coverage.
- **Enhanced Evaluation Framework**: We will expand the evaluation capabilities of the package to serve as a standardized benchmarking tool for other models in the field. This will includes integrating explainable AI (XAI) techniques for better model interpretability and enhancing the ability to perform clustering analyses on feature embeddings.
- **Specialized Identification Model Development**: We will develop a new, specialized identification model that is optimized for bark beetle identification, moving beyond the generalist models currently available. This model will be modular and incrementally developed, with new capabilities released through package updates.

# Licensing, Authorship, and Conflicts of Interest

### License
This software is licensed under the OSI-approved MIT License.

### Authorship
All authors have made substantial contributions to the design of the software.

### Conflicts of Interest
The authors declare that they have no conflicts of interest.

# Acknowledgements

We thank the numerous collaborators, students, and technicians in the Hulcr lab at the University of Florida who have contributed to the collection, curation, and annotation of the image dataset that made this work possible. We also acknowledge the National Science Foundation, the USDA Forest Service, and the Florida Forest Service for their financial support.
