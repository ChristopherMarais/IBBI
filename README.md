# Intelligent Bark Beetle Identifier (IBBI)

* Wrapper for detection and/or classification of bark and ambrosia beetles with multiple computer vision models as they get developed.


# How to contribute:
Clone this repository and create a new branch for your changes:
```bash
cd <path/to/cloned/repository/IBBI>
conda env create -f environment.yml
conda activate IBBI
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
