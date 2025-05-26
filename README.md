# Intelligent Bark Beetle Identifier (IBBI)

* Wrapper for detection and/or classification with mutliple models as they get developed.

Examples:
- https://github.com/Heldenkombinat/Logodetect
- https://github.com/PetervanLunteren/AddaxAI
- https://github.com/martibosch/detectree


# How to contribute:
```bash
conda env create -f environment.yml
conda activate IBBI
cd <path/to/cloned/repository/IBBI>
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
