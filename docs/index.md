# Welcome to IBBI

**Intelligent Bark Beetle Identifier (IBBI)** is a Python package that provides a simple and unified interface for detecting and classifying bark and ambrosia beetles from images using state-of-the-art computer vision models.

This package is designed to support entomological research by automating the laborious task of beetle identification, enabling high-throughput data analysis for ecological studies, pest management, and biodiversity monitoring.

![My training workflow](assets/images/data_flow_ibbi.png)

## The Need for Automation

The accurate and timely identification of bark and ambrosia beetle species is fundamental to forest health monitoring and pest management. Many species are morphologically similar, making identification a significant bottleneck that requires highly specialized taxonomic expertise.

IBBI addresses this challenge by providing an accessible, programmatic solution that automates the identification process.

## Key Features

- **Simple API:** Access powerful detection and classification models with a single function call: `ibbi.create_model()`.
- **Pre-trained Models:** Leverages pre-trained YOLOv10 models hosted on the Hugging Face Hub for immediate use.
- **Extensible:** Designed to easily incorporate new model architectures in the future.
- **Research-Focused:** Aims to accelerate ecological research by automating beetle identification.

Ready to get started? Check out the [**Usage Guide**](./usage.md).
