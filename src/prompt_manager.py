"""
提示词管理模块
统一管理和加载所有LLM提示词
"""

import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PromptManager:
    """提示词管理器"""

    def __init__(self, prompts_dir: str = "./prompts"):
        """
        初始化提示词管理器

        Args:
            prompts_dir: 提示词文件夹路径
        """
        self.prompts_dir = Path(prompts_dir)

        if not self.prompts_dir.exists():
            logger.warning(f"提示词文件夹不存在: {prompts_dir}")
            self.prompts_dir.mkdir(parents=True, exist_ok=True)

        # 缓存加载的提示词
        self._prompts_cache: Dict[str, str] = {}

        # 加载所有提示词
        self._load_prompts()

    def _load_prompts(self):
        """加载所有提示词文件"""
        logger.info(f"从 {self.prompts_dir} 加载提示词...")

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
                    logger.info(f"  ✅ 加载提示词: {key}")
                except Exception as e:
                    logger.error(f"  ❌ 加载提示词失败 ({key}): {e}")
            else:
                logger.warning(f"  ⚠️ 提示词文件不存在: {filename}")

        logger.info(f"提示词加载完成，共 {len(self._prompts_cache)} 个")

    def get_prompt(self, key: str) -> Optional[str]:
        """
        获取提示词

        Args:
            key: 提示词键名

        Returns:
            提示词内容，如果不存在返回None
        """
        return self._prompts_cache.get(key)

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return self.get_prompt('system') or "你是一个专业的学术论文分析助手。"

    def get_extraction_prompt(self, field: str) -> str:
        """
        获取字段提取提示词

        Args:
            field: 字段名（problem, contribution, limitation, future_work）

        Returns:
            提取提示词
        """
        prompt = self.get_prompt(field)

        if prompt is None:
            logger.warning(f"未找到字段 {field} 的提示词，使用默认提示")
            return f"请从论文中提取{field}相关信息。"

        return prompt

    def build_full_prompt(self, field: str, context: str) -> str:
        """
        构建完整的提示词（上下文 + 提取指令）

        Args:
            field: 要提取的字段
            context: 论文内容上下文

        Returns:
            完整的用户提示词
        """
        extraction_prompt = self.get_extraction_prompt(field)

        full_prompt = f"""以下是从一篇学术论文中检索到的相关章节内容：

{context}

---

{extraction_prompt}"""

        return full_prompt

    def reload(self):
        """重新加载所有提示词"""
        logger.info("重新加载提示词...")
        self._prompts_cache.clear()
        self._load_prompts()

    def list_prompts(self) -> Dict[str, int]:
        """
        列出所有已加载的提示词

        Returns:
            {prompt_key: prompt_length}
        """
        return {key: len(prompt) for key, prompt in self._prompts_cache.items()}


# 全局提示词管理器实例
_global_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager(prompts_dir: str = "./prompts") -> PromptManager:
    """
    获取全局提示词管理器实例（单例模式）

    Args:
        prompts_dir: 提示词文件夹路径

    Returns:
        PromptManager实例
    """
    global _global_prompt_manager

    if _global_prompt_manager is None:
        _global_prompt_manager = PromptManager(prompts_dir)

    return _global_prompt_manager


if __name__ == "__main__":
    # 测试代码
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*60)
    print("测试提示词管理模块")
    print("="*60)

    # 创建管理器
    manager = PromptManager("../prompts")

    # 列出所有提示词
    prompts = manager.list_prompts()
    print(f"\n已加载的提示词:")
    for key, length in prompts.items():
        print(f"  • {key}: {length} 字符")

    # 测试获取提示词
    print(f"\n系统提示词:")
    print(manager.get_system_prompt()[:100] + "...")

    print(f"\nProblem提取提示词:")
    print(manager.get_extraction_prompt('problem')[:150] + "...")

    # 测试构建完整提示
    context = "This paper addresses the problem of..."
    full_prompt = manager.build_full_prompt('problem', context)
    print(f"\n完整提示词示例:")
    print(full_prompt[:200] + "...")

    print("\n" + "="*60)
    print("✅ 提示词管理模块测试完成")
    print("="*60)
