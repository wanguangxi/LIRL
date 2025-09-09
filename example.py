#!/usr/bin/env python3
"""示例：如何使用LIRL模型系统"""

import sys
import json

# Add the lirl package to path for local testing
sys.path.insert(0, '.')

import lirl


def main():
    """主函数演示LIRL模型系统的使用"""
    print("=== LIRL 可用模型系统示例 ===\n")
    
    try:
        # 1. 列出所有可用模型
        print("1. 所有可用模型 (All Available Models):")
        all_models = lirl.list_available_models()
        for i, model in enumerate(all_models, 1):
            print(f"   {i:2d}. {model}")
        
        print(f"\n   总计: {len(all_models)} 个模型")
        
        # 2. 按类型筛选
        print("\n2. 强化学习模型 (Reinforcement Learning Models):")
        rl_models = lirl.list_available_models(model_type="reinforcement_learning")
        for model in rl_models:
            print(f"   - {model}")
        
        print("\n   隐式强化学习模型 (Implicit RL Models):")
        implicit_models = lirl.list_available_models(model_type="implicit_rl")
        for model in implicit_models:
            print(f"   - {model}")
        
        # 3. 按状态筛选
        print("\n3. 稳定可用模型 (Stable Available Models):")
        stable_models = lirl.list_available_models(status="available")
        for model in stable_models:
            print(f"   - {model}")
        
        print("\n   实验性模型 (Experimental Models):")
        experimental_models = lirl.list_available_models(status="experimental")
        for model in experimental_models:
            print(f"   - {model}")
        
        # 4. 获取模型详细信息
        print("\n4. DQN 模型详细信息:")
        print("-" * 50)
        dqn_info = lirl.get_model_info("DQN")
        if dqn_info:
            print(f"名称: {dqn_info['name']}")
            print(f"描述: {dqn_info['description']}")
            print(f"类型: {dqn_info['model_type']}")
            print(f"版本: {dqn_info['version']}")
            print(f"状态: {dqn_info['status']}")
            print("参数:")
            for key, value in dqn_info['parameters'].items():
                print(f"  {key}: {value}")
            print("依赖:")
            for req in dqn_info['requirements']:
                print(f"  - {req}")
        
        # 5. LIRL-Base 模型信息
        print("\n5. LIRL-Base 模型详细信息:")
        print("-" * 50)
        lirl_info = lirl.get_model_info("LIRL-Base")
        if lirl_info:
            print(f"名称: {lirl_info['name']}")
            print(f"描述: {lirl_info['description']}")
            print(f"类型: {lirl_info['model_type']}")
            print(f"版本: {lirl_info['version']}")
            print(f"状态: {lirl_info['status']}")
            print("参数:")
            for key, value in lirl_info['parameters'].items():
                print(f"  {key}: {value}")
        
        # 6. 查看所有可用模型类型
        print("\n6. 可用模型类型 (Available Model Types):")
        types = lirl.get_available_model_types()
        for model_type in types:
            count = len(lirl.list_available_models(model_type=model_type))
            print(f"   - {model_type}: {count} 个模型")
        
        # 7. 演示注册自定义模型
        print("\n7. 注册自定义模型示例:")
        from lirl.models import register_custom_model
        
        register_custom_model(
            name="MyCustomDQN",
            description="我的自定义DQN模型",
            model_type="reinforcement_learning",
            version="1.0.0",
            parameters={
                "learning_rate": 0.0005,
                "batch_size": 64,
                "memory_size": 10000
            },
            requirements=["torch", "numpy", "gym"],
            status="experimental"
        )
        
        print("   已注册自定义模型: MyCustomDQN")
        
        # 验证注册的模型
        custom_info = lirl.get_model_info("MyCustomDQN")
        if custom_info:
            print(f"   验证成功: {custom_info['name']} - {custom_info['description']}")
        
        # 更新后的模型总数
        updated_models = lirl.list_available_models()
        print(f"\n   更新后总计: {len(updated_models)} 个模型")
        
        print("\n=== 示例完成 ===")
        print("\n提示：使用 'python -m lirl.cli' 来运行命令行工具")
        print("     使用 'python -m lirl.cli --help' 查看更多选项")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())