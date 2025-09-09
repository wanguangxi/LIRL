"""LIRL - Learning with Implicit Reinforcement Learning

A package for managing and listing available models.
"""

__version__ = "0.1.0"
__author__ = "LIRL Team"

from .models import (
    ModelRegistry, list_available_models, get_model_info, 
    get_available_model_types, register_custom_model
)

__all__ = [
    "ModelRegistry", "list_available_models", "get_model_info",
    "get_available_model_types", "register_custom_model"
]