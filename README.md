# LIRL - Learning with Implicit Reinforcement Learning

一个用于管理和查看可用模型的系统。A system for managing and viewing available models.

## 我现在可以使用的模型 (Available Models)

This repository provides a comprehensive system to list and manage available models for the LIRL project.

### 快速开始 (Quick Start)

#### 安装 (Installation)
```bash
pip install -e .
```

#### 查看可用模型 (View Available Models)

**Python API:**
```python
import lirl

# 列出所有可用模型
models = lirl.list_available_models()
print("Available models:", models)

# 获取模型详细信息
model_info = lirl.get_model_info("DQN")
print("DQN info:", model_info)

# 按类型筛选
rl_models = lirl.list_available_models(model_type="reinforcement_learning")
print("RL models:", rl_models)

# 按状态筛选
stable_models = lirl.list_available_models(status="available")
print("Stable models:", stable_models)
```

**命令行工具 (Command Line Tool):**
```bash
# 列出所有模型
python -m lirl.cli

# 显示详细信息
python -m lirl.cli --details

# 筛选强化学习模型
python -m lirl.cli --type reinforcement_learning

# 显示特定模型信息
python -m lirl.cli --info DQN

# JSON格式输出
python -m lirl.cli --json

# 显示模型类型
python -m lirl.cli --types
```

### 可用模型类型 (Available Model Types)

1. **强化学习模型 (Reinforcement Learning Models)**:
   - **DQN**: Deep Q-Network for discrete action spaces
   - **PPO**: Proximal Policy Optimization
   - **A3C**: Asynchronous Advantage Actor-Critic (实验性 experimental)

2. **隐式强化学习模型 (Implicit RL Models)**:
   - **LIRL-Base**: Base Learning with Implicit Reinforcement Learning model
   - **LIRL-Advanced**: Advanced LIRL model with attention mechanisms (实验性 experimental)

### 模型状态 (Model Status)
- **available** (可用): 稳定可用的模型
- **experimental** (实验性): 正在开发中的模型
- **deprecated** (已弃用): 不建议使用的旧模型

### 添加自定义模型 (Adding Custom Models)

```python
from lirl.models import register_custom_model

# 注册自定义模型
register_custom_model(
    name="MyCustomModel",
    description="My custom reinforcement learning model",
    model_type="reinforcement_learning",
    version="1.0.0",
    parameters={
        "learning_rate": 0.001,
        "batch_size": 32
    },
    requirements=["torch", "numpy"],
    status="experimental"
)
```

### API 文档 (API Documentation)

#### 主要函数 (Main Functions)

- `list_available_models(model_type=None, status=None)`: 列出可用模型名称
- `get_model_info(name)`: 获取特定模型的详细信息
- `get_available_model_types()`: 获取可用的模型类型列表
- `register_custom_model(...)`: 注册自定义模型

#### 模型信息结构 (Model Info Structure)

```python
{
    "name": "模型名称",
    "description": "模型描述",
    "model_type": "模型类型",
    "version": "版本号",
    "parameters": {"参数": "值"},
    "requirements": ["依赖包"],
    "status": "状态"
}
```

### 开发和测试 (Development and Testing)

```bash
# 安装开发依赖
pip install -e .[dev]

# 运行测试
pytest tests/

# 运行特定测试
pytest tests/test_models.py -v
```

### 示例用法 (Example Usage)

```python
#!/usr/bin/env python3
"""示例：如何使用LIRL模型系统"""

import lirl
import json

def main():
    print("=== LIRL 可用模型系统示例 ===\n")
    
    # 1. 列出所有可用模型
    print("1. 所有可用模型:")
    all_models = lirl.list_available_models()
    for i, model in enumerate(all_models, 1):
        print(f"   {i}. {model}")
    
    # 2. 按类型筛选
    print("\n2. 强化学习模型:")
    rl_models = lirl.list_available_models(model_type="reinforcement_learning")
    for model in rl_models:
        print(f"   - {model}")
    
    # 3. 获取模型详细信息
    print("\n3. DQN 模型详细信息:")
    dqn_info = lirl.get_model_info("DQN")
    if dqn_info:
        print(json.dumps(dqn_info, indent=2, ensure_ascii=False))
    
    # 4. 查看模型类型
    print("\n4. 可用模型类型:")
    types = lirl.get_available_model_types()
    for model_type in types:
        print(f"   - {model_type}")

if __name__ == "__main__":
    main()
```

## 许可证 (License)

MIT License - 详见 LICENSE 文件