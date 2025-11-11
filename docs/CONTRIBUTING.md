# Contributing to IBBI

First off, thank you for considering contributing to IBBI! Your help is greatly appreciated. This document provides guidelines for contributing to the project.

## Seeking Support

If you have a general question about how to use `ibbi`, are not sure about a feature, or are encountering a bug, the best way to get help is by opening an issue on our [GitHub Issue Tracker](https://github.com/ChristopherMarais/ibbi/issues).

This is the preferred method for getting support, as it allows the community and maintainers to track and respond to your query in one central place.

## How Can I Contribute?

There are many ways to contribute, from writing tutorials to implementing new models. Here are a few ideas:

- **Reporting Bugs:** If you find a bug, please open an issue on our [GitHub issue tracker](https://github.com/ChristopherMarais/ibbi/issues). Describe the issue in detail, including steps to reproduce it.
- **Suggesting Enhancements:** Have an idea for a new feature or an improvement to an existing one? Open an issue to start a discussion.
- **Writing Documentation:** Good documentation is key. If you find parts of our docs unclear or want to add a new tutorial, please let us know or submit a pull request.
- **Adding New Models:** If you have trained a new model that would be a good fit for IBBI, we'd love to hear about it.
- **Submitting Pull Requests:** If you've fixed a bug or implemented a new feature, you can submit a pull request.

## Setting Up Your Development Environment

To get started with development, please follow these steps.

1.  **Clone the repository:**
    This downloads the project source code to your local machine.
```bash
git clone https://github.com/ChristopherMarais/ibbi.git
cd ibbi
```

2.  **Create a Conda environment:**
    We recommend using Conda to manage your Python environment to avoid conflicts with other projects. This command creates an environment named `ibbi` with Python 3.11.
```bash
conda env create -f environment.yml
conda activate IBBI
```

1.  **Install dependencies with Poetry:**
    This project uses Poetry for dependency management. The `environment.yml` file sets up Python and pip, and then we use Poetry to install the project dependencies.

2.  **Install dependencies with Poetry:**
    This project uses Poetry for dependency management. These commands will install all the necessary packages for running and developing `ibbi`.
```bash
# Install PyTorch first, as its installation can be system-specific (CPU/GPU)
# See https://pytorch.org/get-started/locally/ for the correct command
pip install torch torchvision torchaudio

# Configure Poetry to use the existing Conda environment
poetry config virtualenvs.create false --local

# Install all other project dependencies, including development tools
poetry install --with dev
```

4.  **Set up pre-commit hooks:**
    We use `pre-commit` to automatically run code formatters and linters before each commit. This ensures code quality and a consistent style across the project.
```bash
pre-commit install
```
The hooks will now run automatically every time you make a commit.

## Pull Request Process

1.  Create a new branch for your feature or bug fix (e.g., `git checkout -b feature/my-new-feature`).
2.  Make your changes and commit them. Make sure your commit messages are clear and descriptive.
3.  Ensure all tests pass and that the pre-commit hooks run without errors.
4.  Push your branch to your fork on GitHub.
5.  Open a pull request from your branch to the `main` branch of the IBBI repository.
6.  In the pull request description, clearly describe the changes you've made and why. If it fixes an existing issue, please reference it (e.g., "Fixes #123").

Thank you again for your interest in contributing!
