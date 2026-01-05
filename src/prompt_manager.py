"""
Prompt Management Module
Unified management and loading of all LLM prompts
"""

import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PromptManager:
    """Prompt Manager"""

    def __init__(self, prompts_dir: str = "./prompts"):
        """
        Initialize prompt manager

        Args:
            prompts_dir: Prompt folder path
        """
        self.prompts_dir = Path(prompts_dir)

        if not self.prompts_dir.exists():
            logger.warning(f"Prompt folder does not exist: {prompts_dir}")
            self.prompts_dir.mkdir(parents=True, exist_ok=True)

        # Cache loaded prompts
        self._prompts_cache: Dict[str, str] = {}

        # Load all prompts
        self._load_prompts()

    def _load_prompts(self):
        """Load all prompt files"""
        logger.info(f"Loading prompts from {self.prompts_dir}...")

        prompt_files = {
            'system': 'system_prompt.txt',
            'problem': 'extract_problem.txt',
            'method': 'extract_contribution.txt',
            'limitation': 'extract_limitation.txt',
            'future_work': 'extract_future_work.txt'
        }

        for key, filename in prompt_files.items():
            file_path = self.prompts_dir / filename

            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self._prompts_cache[key] = f.read().strip()
                    logger.info(f"  Loaded prompt: {key}")
                except Exception as e:
                    logger.error(f"  Failed to load prompt ({key}): {e}")
            else:
                logger.warning(f"  Prompt file does not exist: {filename}")

        logger.info(f"Prompt loading complete, total {len(self._prompts_cache)} prompts")

    def get_prompt(self, key: str) -> Optional[str]:
        """
        Get prompt

        Args:
            key: Prompt key name

        Returns:
            Prompt content, returns None if not exists
        """
        return self._prompts_cache.get(key)

    def get_system_prompt(self) -> str:
        """Get system prompt"""
        return self.get_prompt('system') or "You are a professional academic paper analysis assistant."

    def get_extraction_prompt(self, field: str) -> str:
        """
        Get field extraction prompt

        Args:
            field: Field name (problem, contribution, limitation, future_work)

        Returns:
            Extraction prompt
        """
        prompt = self.get_prompt(field)

        if prompt is None:
            logger.warning(f"Prompt for field {field} not found, using default prompt")
            return f"Please extract {field} related information from the paper."

        return prompt

    def build_full_prompt(self, field: str, context: str) -> str:
        """
        Build complete prompt (context + extraction instruction)

        Args:
            field: Field to extract
            context: Paper content context

        Returns:
            Complete user prompt
        """
        extraction_prompt = self.get_extraction_prompt(field)

        full_prompt = f"""The following is the relevant section content retrieved from an academic paper:

{context}

---

{extraction_prompt}"""

        return full_prompt

    def reload(self):
        """Reload all prompts"""
        logger.info("Reloading prompts...")
        self._prompts_cache.clear()
        self._load_prompts()

    def list_prompts(self) -> Dict[str, int]:
        """
        List all loaded prompts

        Returns:
            {prompt_key: prompt_length}
        """
        return {key: len(prompt) for key, prompt in self._prompts_cache.items()}


# Global prompt manager instance
_global_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager(prompts_dir: str = "./prompts") -> PromptManager:
    """
    Get global prompt manager instance (singleton pattern)

    Args:
        prompts_dir: Prompt folder path

    Returns:
        PromptManager instance
    """
    global _global_prompt_manager

    if _global_prompt_manager is None:
        _global_prompt_manager = PromptManager(prompts_dir)

    return _global_prompt_manager


if __name__ == "__main__":
    # Test code
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*60)
    print("Testing Prompt Management Module")
    print("="*60)

    # Create manager
    manager = PromptManager("../prompts")

    # List all prompts
    prompts = manager.list_prompts()
    print(f"\nLoaded prompts:")
    for key, length in prompts.items():
        print(f"  • {key}: {length} characters")

    # Test getting prompts
    print(f"\nSystem prompt:")
    print(manager.get_system_prompt()[:100] + "...")

    print(f"\nProblem extraction prompt:")
    print(manager.get_extraction_prompt('problem')[:150] + "...")

    # Test building full prompt
    context = "This paper addresses the problem of..."
    full_prompt = manager.build_full_prompt('problem', context)
    print(f"\nFull prompt example:")
    print(full_prompt[:200] + "...")

    print("\n" + "="*60)
    print("Prompt management module test complete")
    print("="*60)
