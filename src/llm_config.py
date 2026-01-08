"""
LLM配置管理模块
统一管理LLM相关的配置和客户端初始化

支持的配置文件格式：
- YAML (.yaml, .yml)
- JSON (.json)
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

# YAML支持（可选）
try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)

# 延迟导入LLM库
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None


@dataclass
class LLMConfig:
    """LLM配置数据类"""
    provider: str  # openai, anthropic, local, none
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.3  # 低温度减少幻觉
    max_tokens: int = 500
    timeout: int = 30  # API超时时间（秒）

    # RAG相关配置
    embedding_model: str = 'all-MiniLM-L6-v2'
    use_modelscope: bool = True
    max_context_length: int = 3000

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LLMConfig':
        """从字典创建配置"""
        return cls(
            provider=config_dict.get('llm_provider', 'openai'),
            model=config_dict.get('llm_model', 'gpt-4o-mini'),
            api_key=config_dict.get('llm_api_key'),
            base_url=config_dict.get('llm_base_url'),
            temperature=config_dict.get('temperature', 0.3),
            max_tokens=config_dict.get('max_tokens', 500),
            timeout=config_dict.get('timeout', 30),
            embedding_model=config_dict.get('embedding_model', 'all-MiniLM-L6-v2'),
            use_modelscope=config_dict.get('use_modelscope', True),
            max_context_length=config_dict.get('max_context_length', 3000)
        )

    @classmethod
    def from_file(cls, config_path: str) -> 'LLMConfig':
        """
        从配置文件加载（支持YAML和JSON）

        Args:
            config_path: 配置文件路径 (.yaml, .yml, .json)

        Returns:
            LLMConfig实例
        """
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        # 根据文件扩展名选择解析器
        suffix = config_file.suffix.lower()

        try:
            if suffix in ['.yaml', '.yml']:
                # YAML格式
                if yaml is None:
                    raise ImportError("PyYAML未安装，请运行: pip install pyyaml")

                with open(config_file, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)

                logger.info(f"从YAML文件加载LLM配置: {config_path}")

            elif suffix == '.json':
                # JSON格式
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)

                logger.info(f"从JSON文件加载LLM配置: {config_path}")

            else:
                raise ValueError(f"不支持的配置文件格式: {suffix}，支持 .yaml, .yml, .json")

            # 兼容两种配置格式：
            # 1. 旧格式：llm_provider, llm_model等直接在顶层
            # 2. 新格式：在llm节点下的provider, model等
            if 'llm' in config_dict:
                # 新格式：config/config.yaml
                llm_config = config_dict['llm']
                # 转换为旧格式的键名
                converted_config = {
                    'llm_provider': llm_config.get('provider', 'openai'),
                    'llm_model': llm_config.get('model', 'gpt-4o-mini'),
                    'llm_api_key': llm_config.get('api_key'),
                    'llm_base_url': llm_config.get('base_url'),
                    'temperature': llm_config.get('temperature', 0.3),
                    'max_tokens': llm_config.get('max_tokens', 500),
                    'timeout': llm_config.get('timeout', 30),
                    'embedding_model': llm_config.get('embedding_model', 'all-MiniLM-L6-v2'),
                    'use_modelscope': llm_config.get('use_modelscope', True),
                    'max_context_length': llm_config.get('max_context_length', 3000)
                }
                logger.info(f"  使用新格式配置（llm节点）")
                return cls.from_dict(converted_config)
            else:
                # 旧格式：llm_config.yaml
                logger.info(f"  使用旧格式配置（顶层键）")
                return cls.from_dict(config_dict)

        except Exception as e:
            raise ValueError(f"加载配置文件失败: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'llm_provider': self.provider,
            'llm_model': self.model,
            'llm_api_key': self.api_key,
            'llm_base_url': self.base_url,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'timeout': self.timeout
        }


class LLMClient:
    """LLM客户端封装类"""

    def __init__(self, config: LLMConfig):
        """
        初始化LLM客户端

        Args:
            config: LLM配置对象
        """
        self.config = config
        self.client = None

        # 初始化客户端
        self._init_client()

    def _init_client(self):
        """初始化LLM客户端"""
        if self.config.provider == "none":
            logger.info("LLM功能未启用")
            return

        if self.config.provider == "openai":
            self._init_openai_client()
        elif self.config.provider == "anthropic":
            self._init_anthropic_client()
        elif self.config.provider == "local":
            self._init_local_client()
        else:
            raise ValueError(f"不支持的LLM提供商: {self.config.provider}")

    def _init_openai_client(self):
        """初始化OpenAI客户端"""
        if openai is None:
            raise ImportError("openai包未安装，请运行: pip install openai")

        try:
            if self.config.base_url:
                self.client = openai.OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout
                )
            else:
                self.client = openai.OpenAI(
                    api_key=self.config.api_key,
                    timeout=self.config.timeout
                )

            logger.info(f"✅ OpenAI客户端初始化成功")
            logger.info(f"   模型: {self.config.model}")
            logger.info(f"   Base URL: {self.config.base_url or 'default'}")

        except Exception as e:
            raise RuntimeError(f"初始化OpenAI客户端失败: {e}")

    def _init_anthropic_client(self):
        """初始化Anthropic客户端"""
        if anthropic is None:
            raise ImportError("anthropic包未安装，请运行: pip install anthropic")

        try:
            self.client = anthropic.Anthropic(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            logger.info(f"✅ Anthropic客户端初始化成功 (model: {self.config.model})")

        except Exception as e:
            raise RuntimeError(f"初始化Anthropic客户端失败: {e}")

    def _init_local_client(self):
        """初始化本地LLM客户端（使用OpenAI兼容接口）"""
        if openai is None:
            raise ImportError("openai包未安装，请运行: pip install openai")

        try:
            self.client = openai.OpenAI(
                api_key=self.config.api_key or "not-needed",
                base_url=self.config.base_url or "http://localhost:11434/v1",
                timeout=self.config.timeout
            )
            logger.info(f"✅ 本地LLM客户端初始化成功")
            logger.info(f"   Base URL: {self.config.base_url or 'http://localhost:11434/v1'}")
            logger.info(f"   模型: {self.config.model}")

        except Exception as e:
            raise RuntimeError(f"初始化本地LLM客户端失败: {e}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        调用LLM生成回答

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词（可选）
            temperature: 温度参数（可选，覆盖配置）
            max_tokens: 最大token数（可选，覆盖配置）

        Returns:
            LLM生成的回答
        """
        if self.client is None:
            logger.warning("LLM客户端未初始化")
            return "LLM未配置，无法生成分析"

        # 使用参数或配置中的默认值
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        try:
            if self.config.provider == "anthropic":
                return self._generate_anthropic(prompt, system_prompt, temperature, max_tokens)
            else:
                # OpenAI API（包括本地模型）
                return self._generate_openai(prompt, system_prompt, temperature, max_tokens)

        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return f"LLM调用失败: {str(e)}"

    def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """使用OpenAI API生成"""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content.strip()

    def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """使用Anthropic API生成"""
        messages = []

        # Anthropic的system prompt需要和user message合并
        if system_prompt:
            messages.append({
                "role": "user",
                "content": f"{system_prompt}\n\n{prompt}"
            })
        else:
            messages.append({"role": "user", "content": prompt})

        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages
        )

        return response.content[0].text.strip()


# 便捷函数
def create_llm_client(config_path: str) -> LLMClient:
    """
    从配置文件创建LLM客户端

    Args:
        config_path: 配置文件路径

    Returns:
        LLMClient实例
    """
    config = LLMConfig.from_file(config_path)
    return LLMClient(config)


if __name__ == "__main__":
    # 测试代码
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 测试配置加载
    print("\n" + "="*60)
    print("测试LLM配置管理模块")
    print("="*60)

    # 创建示例配置
    config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        temperature=0.3,
        max_tokens=500
    )

    print(f"\n配置信息:")
    print(f"  Provider: {config.provider}")
    print(f"  Model: {config.model}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Max Tokens: {config.max_tokens}")

    # 测试配置转换
    config_dict = config.to_dict()
    print(f"\n配置字典: {config_dict}")

    print("\n" + "="*60)
    print("✅ 配置管理模块测试完成")
    print("="*60)
