# Intelligent Bark Beetle Identifier (IBBI)

* Wrapper for detection and/or classification with mutliple models as they get developed.

Examples:
- https://github.com/Heldenkombinat/Logodetect
- https://github.com/PetervanLunteren/AddaxAI
- https://github.com/martibosch/detectree


# How to contribute:
```bash
1. conda env create -f environment.yml
2. conda activate IBBI
3. cd <path/to/cloned/repository/IBBI>
4. poetry config virtualenvs.create false --local
5. poetry install
6. poetry run pre-commit install
```

To add a new dependency for the IBBI package use:
```bash
poetry add <new-package-name>
```

To add a new development specific package use:
```bash
poetry add --group dev <new-package-name>
```
