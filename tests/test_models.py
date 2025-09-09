"""Tests for LIRL models functionality."""

import pytest
import json
from lirl.models import (
    ModelInfo, ModelRegistry, list_available_models, 
    get_model_info, get_available_model_types, register_custom_model
)


class TestModelInfo:
    """Test ModelInfo class."""
    
    def test_model_info_creation(self):
        """Test creating a ModelInfo instance."""
        model = ModelInfo(
            name="TestModel",
            description="A test model",
            model_type="test",
            version="1.0.0",
            status="available"
        )
        
        assert model.name == "TestModel"
        assert model.description == "A test model"
        assert model.model_type == "test"
        assert model.version == "1.0.0"
        assert model.status == "available"
        assert model.parameters is None
        assert model.requirements is None
    
    def test_model_info_to_dict(self):
        """Test converting ModelInfo to dictionary."""
        model = ModelInfo(
            name="TestModel",
            description="A test model",
            model_type="test",
            version="1.0.0",
            parameters={"lr": 0.001},
            requirements=["numpy"],
            status="available"
        )
        
        result = model.to_dict()
        expected = {
            "name": "TestModel",
            "description": "A test model",
            "model_type": "test",
            "version": "1.0.0",
            "parameters": {"lr": 0.001},
            "requirements": ["numpy"],
            "status": "available"
        }
        
        assert result == expected


class TestModelRegistry:
    """Test ModelRegistry class."""
    
    def test_registry_initialization(self):
        """Test registry initialization with default models."""
        registry = ModelRegistry()
        models = registry.list_models()
        
        # Should have some default models
        assert len(models) > 0
        
        # Check for expected default models
        model_names = [m.name for m in models]
        assert "DQN" in model_names
        assert "PPO" in model_names
        assert "LIRL-Base" in model_names
    
    def test_register_custom_model(self):
        """Test registering a custom model."""
        registry = ModelRegistry()
        
        custom_model = ModelInfo(
            name="CustomModel",
            description="A custom test model",
            model_type="custom",
            version="1.0.0"
        )
        
        registry.register_model(custom_model)
        retrieved = registry.get_model("CustomModel")
        
        assert retrieved is not None
        assert retrieved.name == "CustomModel"
        assert retrieved.description == "A custom test model"
    
    def test_list_models_with_filters(self):
        """Test listing models with filters."""
        registry = ModelRegistry()
        
        # Filter by type
        rl_models = registry.list_models(model_type="reinforcement_learning")
        assert len(rl_models) > 0
        assert all(m.model_type == "reinforcement_learning" for m in rl_models)
        
        # Filter by status
        available_models = registry.list_models(status="available")
        assert len(available_models) > 0
        assert all(m.status == "available" for m in available_models)
        
        # Filter by both
        available_rl = registry.list_models(model_type="reinforcement_learning", status="available")
        assert all(m.model_type == "reinforcement_learning" and m.status == "available" for m in available_rl)
    
    def test_to_json(self):
        """Test JSON export."""
        registry = ModelRegistry()
        json_str = registry.to_json()
        
        # Should be valid JSON
        data = json.loads(json_str)
        assert isinstance(data, dict)
        assert len(data) > 0
        
        # Check structure
        for model_name, model_data in data.items():
            assert "name" in model_data
            assert "description" in model_data
            assert "model_type" in model_data
            assert "version" in model_data


class TestModuleFunctions:
    """Test module-level functions."""
    
    def test_list_available_models(self):
        """Test list_available_models function."""
        models = list_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        
        # Test filtering
        rl_models = list_available_models(model_type="reinforcement_learning")
        assert len(rl_models) > 0
        
        available_models = list_available_models(status="available")
        assert len(available_models) > 0
    
    def test_get_model_info(self):
        """Test get_model_info function."""
        # Test existing model
        info = get_model_info("DQN")
        assert info is not None
        assert info["name"] == "DQN"
        assert "description" in info
        assert "model_type" in info
        
        # Test non-existing model
        info = get_model_info("NonExistentModel")
        assert info is None
    
    def test_get_available_model_types(self):
        """Test get_available_model_types function."""
        types = get_available_model_types()
        assert isinstance(types, list)
        assert len(types) > 0
        assert "reinforcement_learning" in types
        assert "implicit_rl" in types
    
    def test_register_custom_model(self):
        """Test register_custom_model function."""
        register_custom_model(
            name="TestCustomModel",
            description="Test custom model",
            model_type="test",
            version="1.0.0",
            parameters={"test_param": 1},
            requirements=["test_lib"],
            status="experimental"
        )
        
        info = get_model_info("TestCustomModel")
        assert info is not None
        assert info["name"] == "TestCustomModel"
        assert info["model_type"] == "test"
        assert info["status"] == "experimental"
        assert info["parameters"]["test_param"] == 1
        assert "test_lib" in info["requirements"]


if __name__ == "__main__":
    pytest.main([__file__])