"""
Baseline 评估器: Zero-shot LLM 分类
仅使用 Abstract 进行引用关系分类
"""

import json
import logging
from typing import Dict, List, Tuple
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from llm_config import LLMClient, LLMConfig

logger = logging.getLogger(__name__)


class BaselineEvaluator:
    """
    Baseline 评估器
    使用 Zero-shot LLM 仅基于 Abstract 进行分类
    """

    def __init__(self, llm_client: LLMClient, prompts_dir: str = "./prompts"):
        """
        初始化评估器

        Args:
            llm_client: LLM客户端
            prompts_dir: 提示词目录
        """
        self.llm_client = llm_client
        self.prompts_dir = Path(prompts_dir)

        # 加载提示词模板
        self.prompt_template = self._load_baseline_prompt()

        # 支持的关系类型
        self.relation_types = [
            "Overcomes",
            "Realizes",
            "Extends",
            "Alternative",
            "Adapts_to",
            "Baselines"
        ]

    def _load_baseline_prompt(self) -> str:
        """加载 Baseline 提示词模板"""
        prompt_file = self.prompts_dir / "baseline_classification.txt"

        if prompt_file.exists():
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            # 默认提示词
            return self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """获取默认提示词"""
        return """You are an expert in analyzing citation relationships between research papers.

Given two papers (Paper B cites Paper A), classify the citation relationship type based ONLY on their abstracts.

**Citation Relationship Types:**
1. **Overcomes**: Paper B solves a limitation or problem mentioned in Paper A
2. **Realizes**: Paper B implements a future work direction suggested by Paper A
3. **Extends**: Paper B extends or improves the method from Paper A (incremental improvement)
4. **Alternative**: Paper B proposes an alternative approach to solve a similar problem as Paper A
5. **Adapts_to**: Paper B adapts Paper A's method to a different domain or task (cross-domain transfer)
6. **Baselines**: Paper B only uses Paper A as a baseline for comparison (background reference)

**Paper A (Cited Paper):**
Title: {cited_title}
Abstract: {cited_abstract}

**Paper B (Citing Paper):**
Title: {citing_title}
Abstract: {citing_abstract}

**Task:** Classify the citation relationship type.

**Output Format (JSON):**
```json
{{
    "relationship_type": "one of [Overcomes, Realizes, Extends, Alternative, Adapts_to, Baselines]",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}
```

Respond with JSON only."""

    def classify_citation(
        self,
        citing_abstract: str,
        cited_abstract: str,
        citing_title: str = "",
        cited_title: str = ""
    ) -> Dict:
        """
        分类单条引用关系

        Args:
            citing_abstract: 引用论文摘要
            cited_abstract: 被引用论文摘要
            citing_title: 引用论文标题
            cited_title: 被引用论文标题

        Returns:
            分类结果字典
        """
        # 构建提示词
        prompt = self.prompt_template.format(
            citing_title=citing_title or "Unknown",
            citing_abstract=citing_abstract or "N/A",
            cited_title=cited_title or "Unknown",
            cited_abstract=cited_abstract or "N/A"
        )

        # 调用 LLM
        try:
            response = self.llm_client.generate(
                prompt,
                temperature=0.1,  # 低温度提高一致性
                max_tokens=300
            )

            # 解析 JSON 响应
            json_str = self._extract_json_from_response(response)
            result = json.loads(json_str)

            # 验证关系类型
            rel_type = result.get('relationship_type', 'Baselines')
            if rel_type not in self.relation_types:
                logger.warning(f"无效的关系类型: {rel_type}, 默认为 Baselines")
                rel_type = 'Baselines'

            return {
                'relationship_type': rel_type,
                'confidence': result.get('confidence', 0.0),
                'reasoning': result.get('reasoning', '')
            }

        except Exception as e:
            logger.error(f"分类失败: {e}")
            return {
                'relationship_type': 'Baselines',
                'confidence': 0.0,
                'reasoning': f'Error: {str(e)}'
            }

    def evaluate_dataset(
        self,
        dataset: List[Dict]
    ) -> Tuple[List[Dict], Dict]:
        """
        评估整个数据集

        Args:
            dataset: 数据集列表，每项包含:
                - citing_paper_abstract
                - cited_paper_abstract
                - citing_paper_title (optional)
                - cited_paper_title (optional)
                - edge_type (ground truth)

        Returns:
            (predictions, statistics)
        """
        logger.info(f"开始 Baseline 评估，共 {len(dataset)} 条数据...")

        predictions = []
        statistics = {}

        for i, data in enumerate(dataset):
            logger.info(f"处理 {i+1}/{len(dataset)}...")

            result = self.classify_citation(
                citing_abstract=data.get('citing_paper_abstract', ''),
                cited_abstract=data.get('cited_paper_abstract', ''),
                citing_title=data.get('citing_paper_title', ''),
                cited_title=data.get('cited_paper_title', '')
            )

            # 记录预测结果
            prediction = {
                'citing_paper_id': data.get('citing_paper_id', ''),
                'cited_paper_id': data.get('cited_paper_id', ''),
                'predicted_type': result['relationship_type'],
                'confidence': result['confidence'],
                'reasoning': result['reasoning'],
                'ground_truth': data.get('edge_type', '')
            }
            predictions.append(prediction)

            # 统计
            pred_type = result['relationship_type']
            statistics[pred_type] = statistics.get(pred_type, 0) + 1

        logger.info("✅ Baseline 评估完成")
        logger.info("\n📊 预测类型分布:")
        for rel_type, count in sorted(statistics.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(dataset)) * 100 if dataset else 0
            logger.info(f"  • {rel_type}: {count} ({percentage:.1f}%)")

        return predictions, statistics

    def _extract_json_from_response(self, response: str) -> str:
        """从 LLM 响应中提取 JSON 内容"""
        import re

        response = response.strip()

        # 尝试提取 markdown 代码块中的 JSON
        json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        match = re.search(json_block_pattern, response, re.DOTALL)

        if match:
            return match.group(1).strip()

        # 如果没有代码块，直接返回原响应
        return response


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 测试
    print("\n" + "="*80)
    print("Baseline 评估器测试")
    print("="*80)

    # 创建 LLM 客户端
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "config.yaml"
    from llm_config import create_llm_client
    llm_client = create_llm_client(str(config_path))

    # 创建评估器
    evaluator = BaselineEvaluator(llm_client)

    # 测试数据
    test_data = {
        'citing_paper_title': 'Vision Transformer (ViT)',
        'citing_paper_abstract': 'We show that Transformers can be applied to computer vision tasks with minimal modifications.',
        'cited_paper_title': 'Attention Is All You Need',
        'cited_paper_abstract': 'We propose the Transformer, a model architecture based solely on attention mechanisms.',
        'edge_type': 'Adapts_to'
    }

    result = evaluator.classify_citation(
        citing_abstract=test_data['citing_paper_abstract'],
        cited_abstract=test_data['cited_paper_abstract'],
        citing_title=test_data['citing_paper_title'],
        cited_title=test_data['cited_paper_title']
    )

    print(f"\n预测结果:")
    print(f"  关系类型: {result['relationship_type']}")
    print(f"  置信度: {result['confidence']:.2f}")
    print(f"  推理: {result['reasoning']}")
    print(f"  真实标签: {test_data['edge_type']}")

    print("\n" + "="*80)
