"""
配置文件加载模块
从主项目配置文件中读取LLM配置
"""

import yaml
from pathlib import Path
from typing import Dict, Optional


class ConfigLoader:
    """配置文件加载器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置加载器

        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        if config_path is None:
            # 默认使用主项目的配置文件
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """
        加载配置文件

        Returns:
            配置字典
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def get_llm_config(self) -> Dict:
        """
        获取LLM配置

        Returns:
            LLM配置字典，包含provider, model, api_key, base_url等
        """
        llm_config = self.config.get('llm', {})

        if not llm_config.get('enabled', False):
            raise ValueError("LLM is not enabled in configuration")

        return {
            'provider': llm_config.get('provider', 'openai'),
            'model': llm_config.get('model', 'gpt-4'),
            'api_key': llm_config.get('api_key'),
            'base_url': llm_config.get('base_url'),
            'temperature': llm_config.get('temperature', 0.3),
            'max_tokens': llm_config.get('max_tokens', 4096),
            'timeout': llm_config.get('timeout', 60)
        }

    def get_embedding_config(self) -> Dict:
        """
        获取Embedding模型配置

        Returns:
            Embedding配置字典
        """
        llm_config = self.config.get('llm', {})

        return {
            'model': llm_config.get('embedding_model', 'all-MiniLM-L6-v2'),
            'use_modelscope': llm_config.get('use_modelscope', False)
        }

    def get_evaluation_config(self) -> Dict:
        """
        获取评估配置

        Returns:
            评估配置字典
        """
        # 可以根据需要扩展，目前返回空字典
        return {}

    def print_config_summary(self):
        """打印配置摘要"""
        llm_config = self.get_llm_config()
        embedding_config = self.get_embedding_config()

        print("Configuration Summary:")
        print("-" * 60)
        print(f"LLM Provider: {llm_config['provider']}")
        print(f"LLM Model: {llm_config['model']}")
        print(f"Base URL: {llm_config['base_url'] or 'Default (OpenAI)'}")
        print(f"Temperature: {llm_config['temperature']}")
        print(f"Max Tokens: {llm_config['max_tokens']}")
        print(f"Embedding Model: {embedding_config['model']}")
        print("-" * 60)


def main():
    """测试配置加载"""
    try:
        loader = ConfigLoader()
        loader.print_config_summary()

        print("\nLLM Config:")
        import json
        print(json.dumps(loader.get_llm_config(), indent=2, ensure_ascii=False))

        print("\nEmbedding Config:")
        print(json.dumps(loader.get_embedding_config(), indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"Error loading configuration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
