"""
LLM Configuration Management Module
Unified management of LLM-related configurations and client initialization

Supported configuration file formats:
- YAML (.yaml, .yml)
- JSON (.json)
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

# YAML support (optional)
try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)

# Lazy import of LLM libraries
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
    """LLM configuration data class"""
    provider: str  # openai, anthropic, local, none
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.3  # Low temperature reduces hallucinations
    max_tokens: int = 500
    timeout: int = 30  # API timeout (seconds)

    # RAG-related configuration
    embedding_model: str = 'all-MiniLM-L6-v2'
    use_modelscope: bool = True
    max_context_length: int = 3000

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LLMConfig':
        """Create configuration from dictionary"""
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
        Load from configuration file (supports YAML and JSON)

        Args:
            config_path: Configuration file path (.yaml, .yml, .json)

        Returns:
            LLMConfig instance
        """
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

        # Select parser based on file extension
        suffix = config_file.suffix.lower()

        try:
            if suffix in ['.yaml', '.yml']:
                # YAML format
                if yaml is None:
                    raise ImportError("PyYAML not installed, please run: pip install pyyaml")

                with open(config_file, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)

                logger.info(f"Loading LLM configuration from YAML file: {config_path}")

            elif suffix == '.json':
                # JSON format
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)

                logger.info(f"Loading LLM configuration from JSON file: {config_path}")

            else:
                raise ValueError(f"Unsupported configuration file format: {suffix}, supported formats: .yaml, .yml, .json")

            # Compatible with two configuration formats:
            # 1. Old format: llm_provider, llm_model, etc. directly at top level
            # 2. New format: provider, model, etc. under llm node
            if 'llm' in config_dict:
                # New format: config/config.yaml
                llm_config = config_dict['llm']
                # Convert to old format key names
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
                logger.info(f"  Using new format configuration (llm node)")
                return cls.from_dict(converted_config)
            else:
                # Old format: llm_config.yaml
                logger.info(f"  Using old format configuration (top-level keys)")
                return cls.from_dict(config_dict)

        except Exception as e:
            raise ValueError(f"Failed to load configuration file: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
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
    """LLM client wrapper class"""

    def __init__(self, config: LLMConfig):
        """
        Initialize LLM client

        Args:
            config: LLM configuration object
        """
        self.config = config
        self.client = None

        # Initialize client
        self._init_client()

    def _init_client(self):
        """Initialize LLM client"""
        if self.config.provider == "none":
            logger.info("LLM functionality not enabled")
            return

        if self.config.provider == "openai":
            self._init_openai_client()
        elif self.config.provider == "anthropic":
            self._init_anthropic_client()
        elif self.config.provider == "local":
            self._init_local_client()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

    def _init_openai_client(self):
        """Initialize OpenAI client"""
        if openai is None:
            raise ImportError("openai package not installed, please run: pip install openai")

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

            logger.info(f"✅ OpenAI client initialized successfully")
            logger.info(f"   Model: {self.config.model}")
            logger.info(f"   Base URL: {self.config.base_url or 'default'}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    def _init_anthropic_client(self):
        """Initialize Anthropic client"""
        if anthropic is None:
            raise ImportError("anthropic package not installed, please run: pip install anthropic")

        try:
            self.client = anthropic.Anthropic(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            logger.info(f"✅ Anthropic client initialized successfully (model: {self.config.model})")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Anthropic client: {e}")

    def _init_local_client(self):
        """Initialize local LLM client (using OpenAI-compatible interface)"""
        if openai is None:
            raise ImportError("openai package not installed, please run: pip install openai")

        try:
            self.client = openai.OpenAI(
                api_key=self.config.api_key or "not-needed",
                base_url=self.config.base_url or "http://localhost:11434/v1",
                timeout=self.config.timeout
            )
            logger.info(f"✅ Local LLM client initialized successfully")
            logger.info(f"   Base URL: {self.config.base_url or 'http://localhost:11434/v1'}")
            logger.info(f"   Model: {self.config.model}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize local LLM client: {e}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Call LLM to generate response

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Temperature parameter (optional, overrides config)
            max_tokens: Maximum number of tokens (optional, overrides config)

        Returns:
            LLM-generated response
        """
        if self.client is None:
            logger.warning("LLM client not initialized")
            return "LLM not configured, unable to generate analysis"

        # Use parameter or default value from config
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        try:
            if self.config.provider == "anthropic":
                return self._generate_anthropic(prompt, system_prompt, temperature, max_tokens)
            else:
                # OpenAI API (including local models)
                return self._generate_openai(prompt, system_prompt, temperature, max_tokens)

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"LLM call failed: {str(e)}"

    def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate using OpenAI API"""
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
        """Generate using Anthropic API"""
        messages = []

        # Anthropic's system prompt needs to be merged with user message
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


# Convenience function
def create_llm_client(config_path: str) -> LLMClient:
    """
    Create LLM client from configuration file

    Args:
        config_path: Configuration file path

    Returns:
        LLMClient instance
    """
    config = LLMConfig.from_file(config_path)
    return LLMClient(config)


if __name__ == "__main__":
    # Test code
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Test configuration loading
    print("\n" + "="*60)
    print("Testing LLM Configuration Management Module")
    print("="*60)

    # Create example configuration
    config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        temperature=0.3,
        max_tokens=500
    )

    print(f"\nConfiguration information:")
    print(f"  Provider: {config.provider}")
    print(f"  Model: {config.model}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Max Tokens: {config.max_tokens}")

    # Test configuration conversion
    config_dict = config.to_dict()
    print(f"\nConfiguration dictionary: {config_dict}")

    print("\n" + "="*60)
    print("✅ Configuration management module test completed")
    print("="*60)
