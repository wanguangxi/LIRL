"""Models module for LIRL - Available Models Registry System."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    description: str
    model_type: str
    version: str
    parameters: Optional[Dict[str, Any]] = None
    requirements: Optional[List[str]] = None
    status: str = "available"  # available, experimental, deprecated
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "model_type": self.model_type,
            "version": self.version,
            "parameters": self.parameters or {},
            "requirements": self.requirements or [],
            "status": self.status
        }


class ModelRegistry:
    """Registry for managing available models."""
    
    def __init__(self):
        """Initialize the model registry."""
        self._models: Dict[str, ModelInfo] = {}
        self._load_default_models()
    
    def _load_default_models(self):
        """Load default/built-in models."""
        # Add some example models that would typically be available
        default_models = [
            ModelInfo(
                name="DQN",
                description="Deep Q-Network for discrete action spaces",
                model_type="reinforcement_learning",
                version="1.0.0",
                parameters={
                    "learning_rate": 0.001,
                    "gamma": 0.99,
                    "epsilon_decay": 0.995,
                    "target_update": 100
                },
                requirements=["torch", "numpy"],
                status="available"
            ),
            ModelInfo(
                name="PPO",
                description="Proximal Policy Optimization",
                model_type="reinforcement_learning",
                version="2.0.0",
                parameters={
                    "learning_rate": 3e-4,
                    "n_steps": 2048,
                    "batch_size": 64,
                    "clip_range": 0.2
                },
                requirements=["torch", "numpy"],
                status="available"
            ),
            ModelInfo(
                name="A3C",
                description="Asynchronous Advantage Actor-Critic",
                model_type="reinforcement_learning",
                version="1.5.0",
                parameters={
                    "learning_rate": 1e-4,
                    "entropy_coef": 0.01,
                    "value_loss_coef": 0.5
                },
                requirements=["torch", "numpy", "multiprocessing"],
                status="experimental"
            ),
            ModelInfo(
                name="LIRL-Base",
                description="Base Learning with Implicit Reinforcement Learning model",
                model_type="implicit_rl",
                version="0.1.0",
                parameters={
                    "hidden_dim": 256,
                    "num_layers": 3,
                    "learning_rate": 1e-3
                },
                requirements=["torch", "numpy"],
                status="available"
            ),
            ModelInfo(
                name="LIRL-Advanced",
                description="Advanced LIRL model with attention mechanisms",
                model_type="implicit_rl",
                version="0.2.0",
                parameters={
                    "hidden_dim": 512,
                    "num_layers": 4,
                    "attention_heads": 8,
                    "learning_rate": 5e-4
                },
                requirements=["torch", "numpy", "transformers"],
                status="experimental"
            )
        ]
        
        for model in default_models:
            self._models[model.name] = model
    
    def register_model(self, model_info: ModelInfo):
        """Register a new model."""
        self._models[model_info.name] = model_info
    
    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get model information by name."""
        return self._models.get(name)
    
    def list_models(self, model_type: Optional[str] = None, status: Optional[str] = None) -> List[ModelInfo]:
        """List available models with optional filtering."""
        models = list(self._models.values())
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if status:
            models = [m for m in models if m.status == status]
        
        return sorted(models, key=lambda x: x.name)
    
    def get_model_names(self, model_type: Optional[str] = None, status: Optional[str] = None) -> List[str]:
        """Get list of model names."""
        models = self.list_models(model_type, status)
        return [model.name for model in models]
    
    def to_json(self, indent: int = 2) -> str:
        """Export all models to JSON format."""
        models_dict = {name: model.to_dict() for name, model in self._models.items()}
        return json.dumps(models_dict, indent=indent)


# Global registry instance
_registry = ModelRegistry()


def list_available_models(model_type: Optional[str] = None, status: Optional[str] = None) -> List[str]:
    """List available model names.
    
    Args:
        model_type: Filter by model type (e.g., 'reinforcement_learning', 'implicit_rl')
        status: Filter by status ('available', 'experimental', 'deprecated')
    
    Returns:
        List of model names
    """
    return _registry.get_model_names(model_type, status)


def get_model_info(name: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific model.
    
    Args:
        name: Model name
        
    Returns:
        Model information dictionary or None if not found
    """
    model = _registry.get_model(name)
    return model.to_dict() if model else None


def get_available_model_types() -> List[str]:
    """Get list of available model types."""
    models = _registry.list_models()
    return sorted(list(set(model.model_type for model in models)))


def register_custom_model(name: str, description: str, model_type: str, version: str, **kwargs):
    """Register a custom model.
    
    Args:
        name: Model name
        description: Model description
        model_type: Type of model
        version: Model version
        **kwargs: Additional parameters like 'parameters', 'requirements', 'status'
    """
    model_info = ModelInfo(
        name=name,
        description=description,
        model_type=model_type,
        version=version,
        parameters=kwargs.get('parameters'),
        requirements=kwargs.get('requirements'),
        status=kwargs.get('status', 'available')
    )
    _registry.register_model(model_info)