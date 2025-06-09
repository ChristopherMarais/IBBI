---
title: 'IBBI: A Python package for the detection and classification of bark and ambrosia beetles'
tags:
  - Python
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
date: 9 June 2025
bibliography: paper.bib

---

# Summary

The Intelligent Bark Beetle Identifier (IBBI) is an open-source Python package that provides a simple, unified interface for the automated detection and classification of bark and ambrosia beetles (Coleoptera: Curculionidae: Scolytinae and Platypodinae) from images. Leveraging state-of-the-art computer vision models, `ibbi` is designed to streamline a traditionally labor-intensive and expert-dependent task. By providing programmatic access to pre-trained models for both object detection and species classification, the package serves as a critical tool for entomological research, enabling high-throughput data analysis for ecological studies, invasive species monitoring, and biodiversity assessments. The core functionality is accessible through a single factory function, `create_model()`, which simplifies model loading and inference.

# Statement of need

The accurate and timely identification of bark and ambrosia beetle species is fundamental to forest health monitoring and pest management. Many species are morphologically similar (homoplastic), making identification a significant bottleneck that requires highly specialized taxonomic expertise [@kirkendall2015; @hulcr2015morphology]. Traditional methods are slow and not scalable for large ecological datasets, hindering rapid responses to invasive species outbreaks and broad-scale biodiversity research.

`ibbi` addresses this challenge by providing an accessible, programmatic solution that automates the identification process. It lowers the barrier to entry for researchers without deep expertise in computer vision or taxonomy, enabling faster and more extensive data collection and analysis. By wrapping complex deep learning models in a simple API, `ibbi` empowers entomologists and ecologists to integrate automated identification into their research workflows, accelerating research and improving the capacity for pest management and conservation efforts.

# State of the field

Automated species identification using computer vision has seen significant progress, with deep learning models achieving expert-level accuracy in various contexts [@valan2019; @hoye2021]. Several projects have focused on general insect classification using large, citizen-science datasets like iNaturalist [@horn2017]. However, tools specialized for the fine-grained identification of economically and ecologically critical groups like bark beetles are less common. Existing research often involves bespoke models trained for specific studies, which are not easily accessible or reusable by the broader community.

`ibbi` differentiates itself by:
1.  **Focusing specifically** on the challenging bark and ambrosia beetle taxa.
2.  Providing both **detection and classification** models, allowing for flexible workflows from raw field images.
3.  **Standardizing access** to multiple, pre-trained model architectures through a consistent API.
4.  **Distributing the software** as an easy-to-install Python package, promoting reusability and extensibility.

While other platforms might offer broader taxonomic coverage, `ibbi` provides a specialized, high-performance tool tailored to the needs of forest entomology, filling a critical gap in the available software ecosystem.

# Key Functionality

The primary entry point for using `ibbi` is the `create_model()` factory function. This function abstracts away the model loading and weight-downloading process, allowing users to instantiate a model with a single line of code.

```python
import ibbi
from PIL import Image

# Load a pre-trained object detection or classification model
detector = ibbi.create_model("yolov10x_bb_detect_model", pretrained=True)
classifier = ibbi.create_model("yolov10x_bb_classify_model", pretrained=True)

# Load an image
image = Image.open("path/to/beetle_image.jpg")

# Run inference to get bounding boxes or class predictions
detection_results = detector.predict(image)
classification_results = classifier.predict(image)

# Extract deep features for other downstream tasks
features = classifier.extract_features(image)
