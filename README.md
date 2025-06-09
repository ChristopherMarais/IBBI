# Intelligent Bark Beetle Identifier (IBBI)

* Wrapper for detection and/or classification of bark and ambrosia beetles with multiple computer vision models as they get developed.

# Installation

This package requires PyTorch to be installed separately before you install `ibbi`. This is to ensure that the PyTorch version matches your specific hardware (CPU or GPU) and CUDA version.

### 1. Install PyTorch

First, install PyTorch by following the official instructions on their website. This will ensure you get the correct build for your system.

- **[PyTorch Official Installation Guide](https://pytorch.org/get-started/locally/)**

Choose the options that match your OS, package manager (`pip` or `conda`), and CUDA version if you have an NVIDIA GPU.

### 2. Install the IBBI Package

Once PyTorch is installed, you can install the `ibbi` package using pip:

```bash
pip install ibbi
```


# How to contribute:
Clone this repository and create a new branch for your changes:
```bash
cd <path/to/cloned/repository/IBBI>
conda env create -f environment.yml
conda activate IBBI
conda install cudatoolkit -y
pip install torch torchvision torchaudio
poetry config virtualenvs.create false --local
poetry install
poetry run pre-commit install
```

To add a new dependency for the IBBI package use:
```bash
poetry add <new-package-name>
```

To add a new development specific package use:
```bash
poetry add --group dev <new-package-name>
```
