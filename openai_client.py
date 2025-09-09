import os
import json
from openai import OpenAI


def response_openai():
    """
    Generate academic paper using OpenAI API with proper formatting.
    """
    # Use environment variable for API key or fallback to hardcoded (not recommended for production)
    api_key = os.getenv('OPENAI_API_KEY', 'sk-pcuCJOLhp8TO6wmxhhN8U465FuGLVzrYMaYTeuZVK0SsZf4p')
    
    client = OpenAI(
        api_key=api_key,
        base_url='https://api.bianxie.ai/v1'
    )

    try:
        response = client.chat.completions.create(
            model="o3-pro",
            messages=[
                {
                    "role": "user",
                    "content": "利用VLM进行实际状态感知，采用任务分层网络进行机器人复杂任务的分解与编排，解决动态环境下机器人的实时任务规划能力不足的问题。请帮我生成一篇学术论文；包括摘要500字、介绍1000字、相关工作1000字、问题建模1000字、方法1000字、实验1000字、结论800字。"
                }
            ]
        )
        
        # Format and print the response content properly
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            print("=" * 80)
            print("GENERATED ACADEMIC PAPER")
            print("=" * 80)
            print(content)
            print("=" * 80)
            return content
        else:
            print("No response content received")
            return None
            
    except Exception as e:
        print(f"Error occurred while calling OpenAI API: {str(e)}")
        return None


def response_openai_json():
    """
    Alternative function that returns the full JSON response for debugging.
    """
    api_key = os.getenv('OPENAI_API_KEY', 'sk-pcuCJOLhp8TO6wmxhhN8U465FuGLVzrYMaYTeuZVK0SsZf4p')
    
    client = OpenAI(
        api_key=api_key,
        base_url='https://api.bianxie.ai/v1'
    )

    try:
        response = client.chat.completions.create(
            model="o3-pro",
            messages=[
                {
                    "role": "user",
                    "content": "利用VLM进行实际状态感知，采用任务分层网络进行机器人复杂任务的分解与编排，解决动态环境下机器人的实时任务规划能力不足的问题。请帮我生成一篇学术论文；包括摘要500字、介绍1000字、相关工作1000字、问题建模1000字、方法1000字、实验1000字、结论800字。"
                }
            ]
        )
        
        # Convert to dict and print formatted JSON
        response_dict = response.model_dump()
        print(json.dumps(response_dict, indent=2, ensure_ascii=False))
        return response_dict
        
    except Exception as e:
        print(f"Error occurred while calling OpenAI API: {str(e)}")
        return None


if __name__ == "__main__":
    # Run the main function
    result = response_openai()
    if result:
        print("\nFunction executed successfully!")
    else:
        print("\nFunction execution failed!")