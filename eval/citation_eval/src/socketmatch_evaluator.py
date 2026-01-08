"""
SocketMatch 方法评估器
使用 LLM + DeepPaper 提取信息 + SocketMatch 逻辑
"""

import json
import logging
from typing import Dict, List, Tuple
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from citation_type_inferencer import CitationTypeInferencer
from llm_config import LLMClient, LLMConfig

logger = logging.getLogger(__name__)


class SocketMatchEvaluator:
    """
    SocketMatch 方法评估器
    使用深度提取信息 + Socket Matching 逻辑进行分类
    """

    def __init__(self, llm_client: LLMClient, prompts_dir: str = None):
        """
        初始化评估器

        Args:
            llm_client: LLM客户端
            prompts_dir: 提示词目录（如果为None则使用默认路径）
        """
        # 默认提示词目录
        if prompts_dir is None:
            prompts_dir = str(Path(__file__).parent.parent.parent.parent / "prompts")

        # 初始化 CitationTypeInferencer
        self.inferencer = CitationTypeInferencer(
            llm_client=llm_client,
            prompts_dir=prompts_dir
        )

        logger.info("SocketMatch 评估器初始化完成")

    def classify_citation(
        self,
        citing_paper_data: Dict,
        cited_paper_data: Dict
    ) -> Dict:
        """
        使用 SocketMatch 方法分类单条引用关系

        Args:
            citing_paper_data: 引用论文数据，包含:
                - id
                - title
                - abstract
                - rag_problem
                - rag_contribution
                - rag_limitation
                - rag_future_work
            cited_paper_data: 被引用论文数据（格式同上）

        Returns:
            分类结果字典
        """
        # 构建论文对象（符合 CitationTypeInferencer 的输入格式）
        citing_paper = {
            'id': citing_paper_data.get('id', ''),
            'title': citing_paper_data.get('title', ''),
            'abstract': citing_paper_data.get('abstract', ''),
            'deep_analysis': {
                'problem': citing_paper_data.get('rag_problem', ''),
                'contribution': citing_paper_data.get('rag_contribution', ''),
                'limitation': citing_paper_data.get('rag_limitation', ''),
                'future_work': citing_paper_data.get('rag_future_work', '')
            }
        }

        cited_paper = {
            'id': cited_paper_data.get('id', ''),
            'title': cited_paper_data.get('title', ''),
            'abstract': cited_paper_data.get('abstract', ''),
            'deep_analysis': {
                'problem': cited_paper_data.get('rag_problem', ''),
                'contribution': cited_paper_data.get('rag_contribution', ''),
                'limitation': cited_paper_data.get('rag_limitation', ''),
                'future_work': cited_paper_data.get('rag_future_work', '')
            }
        }

        # 使用 CitationTypeInferencer 推断关系类型
        try:
            relationship = self.inferencer.infer_single_edge_type(
                citing_paper=citing_paper,
                cited_paper=cited_paper
            )

            return {
                'relationship_type': relationship.relationship_type,
                'confidence': relationship.confidence,
                'reasoning': relationship.reasoning,
                'evidence': relationship.evidence
            }

        except Exception as e:
            logger.error(f"SocketMatch 分类失败: {e}")
            return {
                'relationship_type': 'Baselines',
                'confidence': 0.0,
                'reasoning': f'Error: {str(e)}',
                'evidence': ''
            }

    def evaluate_dataset(
        self,
        dataset: List[Dict]
    ) -> Tuple[List[Dict], Dict]:
        """
        评估整个数据集

        Args:
            dataset: 数据集列表，每项包含:
                - citing_paper_id
                - citing_paper_title
                - citing_paper_abstract
                - citing_paper_rag_problem
                - citing_paper_rag_contribution
                - citing_paper_rag_limitation
                - citing_paper_rag_future_work
                - cited_paper_* (同上)
                - edge_type (ground truth)

        Returns:
            (predictions, statistics)
        """
        logger.info(f"开始 SocketMatch 评估，共 {len(dataset)} 条数据...")

        predictions = []
        statistics = {}

        for i, data in enumerate(dataset):
            logger.info(f"处理 {i+1}/{len(dataset)}...")

            # 准备引用论文数据
            citing_paper_data = {
                'id': data.get('citing_paper_id', ''),
                'title': data.get('citing_paper_title', ''),
                'abstract': data.get('citing_paper_abstract', ''),
                'rag_problem': data.get('citing_paper_rag_problem', ''),
                'rag_contribution': data.get('citing_paper_rag_contribution', ''),
                'rag_limitation': data.get('citing_paper_rag_limitation', ''),
                'rag_future_work': data.get('citing_paper_rag_future_work', '')
            }

            # 准备被引用论文数据
            cited_paper_data = {
                'id': data.get('cited_paper_id', ''),
                'title': data.get('cited_paper_title', ''),
                'abstract': data.get('cited_paper_abstract', ''),
                'rag_problem': data.get('cited_paper_rag_problem', ''),
                'rag_contribution': data.get('cited_paper_rag_contribution', ''),
                'rag_limitation': data.get('cited_paper_rag_limitation', ''),
                'rag_future_work': data.get('cited_paper_rag_future_work', '')
            }

            # 分类
            result = self.classify_citation(citing_paper_data, cited_paper_data)

            # 记录预测结果
            prediction = {
                'citing_paper_id': citing_paper_data['id'],
                'cited_paper_id': cited_paper_data['id'],
                'predicted_type': result['relationship_type'],
                'confidence': result['confidence'],
                'reasoning': result['reasoning'],
                'ground_truth': data.get('edge_type', '')
            }
            predictions.append(prediction)

            # 统计
            pred_type = result['relationship_type']
            statistics[pred_type] = statistics.get(pred_type, 0) + 1

        logger.info("✅ SocketMatch 评估完成")
        logger.info("\n📊 预测类型分布:")
        for rel_type, count in sorted(statistics.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(dataset)) * 100 if dataset else 0
            logger.info(f"  • {rel_type}: {count} ({percentage:.1f}%)")

        return predictions, statistics

    def batch_classify(
        self,
        papers: List[Dict],
        citation_edges: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str, str]], Dict[str, int]]:
        """
        批量推断引用关系（直接调用 infer_edge_types）

        Args:
            papers: 论文列表
            citation_edges: 引用关系列表 [(citing_id, cited_id), ...]

        Returns:
            (typed_edges, statistics)
        """
        return self.inferencer.infer_edge_types(papers, citation_edges)


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 测试
    print("\n" + "="*80)
    print("SocketMatch 评估器测试")
    print("="*80)

    # 创建 LLM 客户端
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "config.yaml"
    from llm_config import create_llm_client
    llm_client = create_llm_client(str(config_path))

    # 创建评估器
    evaluator = SocketMatchEvaluator(llm_client)

    # 测试数据
    citing_data = {
        'id': 'W2',
        'title': 'Vision Transformer (ViT)',
        'abstract': 'We show that Transformers can be applied to computer vision tasks.',
        'rag_problem': 'Applying Transformer to computer vision tasks',
        'rag_contribution': 'Pure Transformer for image classification',
        'rag_limitation': 'Requires very large datasets',
        'rag_future_work': 'Apply to detection and segmentation'
    }

    cited_data = {
        'id': 'W1',
        'title': 'Attention Is All You Need',
        'abstract': 'We propose the Transformer architecture.',
        'rag_problem': 'Sequence models difficult to parallelize',
        'rag_contribution': 'Transformer based on attention mechanisms',
        'rag_limitation': 'Limited to fixed-length sequences',
        'rag_future_work': 'Explore Transformer applications in computer vision'
    }

    result = evaluator.classify_citation(citing_data, cited_data)

    print(f"\n预测结果:")
    print(f"  关系类型: {result['relationship_type']}")
    print(f"  置信度: {result['confidence']:.2f}")
    print(f"  推理: {result['reasoning'][:200]}...")

    print("\n" + "="*80)
