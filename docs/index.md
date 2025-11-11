# Welcome to the Intelligent Bark Beetle Identifier (IBBI)

**IBBI** is a Python package that provides a simple and unified interface for detecting and classifying bark and ambrosia beetles from images using state-of-the-art computer vision models.

This package is designed to support entomological research by automating the laborious task of beetle identification, enabling high-throughput data analysis for ecological studies, pest management, and biodiversity monitoring.

### The Need for Automation

The accurate and timely identification of bark and ambrosia beetle species is fundamental to forest health monitoring. However, many species are morphologically similar, making identification a significant bottleneck that requires highly specialized taxonomic expertise. IBBI addresses this challenge by providing an accessible, programmatic solution that automates the identification process.

---

### Key Features

* **Simple API:** Access powerful detection and classification models with a single function call: `ibbi.create_model()`.
* **Multiple Model Types:**
    * **Single-Class Detection:** Detect the presence of any beetle in an image.
    * **Multi-Class Classification:** Identify the species of a beetle from a cropped image.
    * **Zero-Shot Detection:** Detect beetles using a text prompt, without prior training on that specific class.
* **Pre-trained Models:** Leverages pre-trained models hosted on the Hugging Face Hub for immediate use.
* **Model Explainability:** Understand model predictions using SHAP (SHapley Additive exPlanations) to visualize which parts of an image contribute to the identification.
* **Extensible:** Designed to easily incorporate new model architectures in the future.
* **Research-Focused:** Aims to accelerate ecological research by automating beetle identification.

---

### Workflow: How the Models Were Built

The models in `ibbi` are trained using a detailed workflow, from data collection and annotation to model evaluation. This process ensures high-quality, reliable models for your research.

![IBBI Workflow](https://media.githubusercontent.com/media/ChristopherMarais/IBBI/main/docs/assets/images/data_flow_ibbi.png)

---

### Package API and Usage

The `ibbi` package is designed for ease of use. The main functions, inputs, and outputs are summarized below.

![IBBI Inputs and Outputs](https://media.githubusercontent.com/media/ChristopherMarais/IBBI/main/docs/assets/images/ibbi_inputs_outputs.png)

Ready to get started? Check out the **[Usage Guide](usage.md)**.
