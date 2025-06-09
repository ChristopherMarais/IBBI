# --- IMPORTANT ---
# Import the model definition files first.
# This ensures that the @register_model decorator runs and populates
# the model_registry before we try to use it.
from .models import classification, detection  # noqa: F401

# Now, import the registry that has been populated.
from .models._registry import model_registry


def create_model(model_name: str, pretrained: bool = False, **kwargs):
    """
    Creates a model from a name.

    This factory function is the main entry point for users of the package.

    Args:
        model_name (str): Name of the model to create.
        pretrained (bool): Whether to load pretrained weights.
        **kwargs: Extra arguments to pass to the model-creating function.

    Returns:
        An instance of the requested model.
    """
    if model_name not in model_registry:
        available = ", ".join(model_registry.keys())
        raise KeyError(f"Model '{model_name}' not found. Available models: [{available}]")

    # Look up the factory function in the registry and call it
    model_factory = model_registry[model_name]
    model = model_factory(pretrained=pretrained, **kwargs)

    return model
