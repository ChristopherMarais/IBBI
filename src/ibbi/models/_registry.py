"""Implements the model registry for the ibbi package.

This module provides a simple decorator-based registry pattern to dynamically
discover and manage available models. All model factory functions are decorated
with `register_model`, which adds them to a central `model_registry` dictionary.
This allows the main `ibbi.create_model` function to easily instantiate models
by their string name without needing to import them directly.
"""

# 1. The registry itself: a dictionary to hold your models.
model_registry = {}


def register_model(fn):
    """A decorator to register a model factory function in the model registry.

    This function is used as a decorator on model factory functions (e.g.,
    `yolov10x_bb_detect_model`). It takes the decorated function and adds it to the
    global `model_registry` dictionary, using the function's name as the key.
    This allows for easy lookup and instantiation of models by name.

    Args:
        fn (Callable): The model-creating function to register. It is expected that this
                       function accepts a `pretrained: bool` argument and `**kwargs`.

    Returns:
        Callable: The original, unmodified function after it has been registered.

    Raises:
        ValueError: If a model with the same name is already present in the registry.
    """
    model_name = fn.__name__
    if model_name in model_registry:
        raise ValueError(f"Model {model_name} is already registered.")

    model_registry[model_name] = fn
    return fn
