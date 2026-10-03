# API reference

Generated from the docstrings of `ibbi`. The everyday entry points are at the top; the model, evaluation and data
classes follow.

## Top-level functions

::: ibbi.create_model

::: ibbi.create_pipeline

::: ibbi.list_models

::: ibbi.utils.data.get_dataset

::: ibbi.utils.data.download_benchmark

::: ibbi.utils.data.get_taxonomy

::: ibbi.utils.data.get_shap_background_dataset

::: ibbi.utils.data.taxonomic_distance_matrix

::: ibbi.utils.data.get_ood_dataset

::: ibbi.utils.cache.get_cache_dir

::: ibbi.utils.cache.clean_cache

## Pipeline

::: ibbi.pipeline.IdentificationPipeline

## Models

::: ibbi.models.detectors.UltralyticsDetector

::: ibbi.models.detectors.SpeciesDetector

::: ibbi.models.detectors.ArthropodDetector

::: ibbi.models.classifiers.HierarchicalClassifier

::: ibbi.models.zero_shot.ZeroShotDetector

::: ibbi.models.zero_shot.GroundingDINOModel

::: ibbi.models.zero_shot.OWLv2Model

::: ibbi.models.zero_shot.YOLOWorldModel

::: ibbi.models.zero_shot.SAM3Model

## Data

::: ibbi.utils.data.BenchmarkDataset

## Evaluation

::: ibbi.evaluate.Evaluator

::: ibbi.evaluate.benchmark
    options:
      members:
        - evaluate_predictions
        - evaluate_class_agnostic
        - to_coco_results

::: ibbi.evaluate.hierarchical.evaluate_hierarchical_records

::: ibbi.evaluate.embeddings.EmbeddingEvaluator

## Explainability

::: ibbi.explain.Explainer

::: ibbi.explain.lime.plot_lime_explanation

::: ibbi.explain.shap.plot_shap_explanation
