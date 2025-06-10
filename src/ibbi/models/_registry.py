# 1. The registry itself: a dictionary to hold your models.
model_registry = {}


def register_model(fn):
    """
    # 2. A decorator function to easily add models to the registry.

    This function takes a model-creating function (like your
    `yolov10x_bb_detect_model` function) and adds it to the
    `model_registry` dictionary. The function's name becomes the key.

    Args:
        fn: The model-creating function to register.

    Returns:
        The original function, after it has been registered.
    """
    model_name = fn.__name__
    if model_name in model_registry:
        raise ValueError(f"Model {model_name} is already registered.")

    model_registry[model_name] = fn
    return fn
