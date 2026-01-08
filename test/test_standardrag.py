"""
测试 Standard RAG 创意生成
对比 Standard RAG vs 基于知识图谱的方法
"""

import json
import logging
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from idea_eval.StandardRAG import StandardRAGIdeaGenerator
from test_idea_generation import load_full_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_standard_rag(
    papers_file: str,
    topic: str = "Natural Language Processing",
    num_ideas: int = 10,
    retrieve_k: int = 20,
    config_path: str = 'config/config.yaml',
    output_file: str = 'standardrag_ideas.json'
):
    """
    测试 Standard RAG 创意生成

    Args:
        papers_file: 论文数据文件
        topic: 研究主题
        num_ideas: 生成创意数量
        retrieve_k: 检索论文数量
        config_path: 配置文件路径
        output_file: 输出文件路径
    """
    logger.info("=" * 80)
    logger.info("测试 Standard RAG 创意生成")
    logger.info("=" * 80)

    # 加载配置
    config = load_full_config(config_path)

    # 初始化生成器
    logger.info("\n🔧 初始化 Standard RAG 生成器")
    generator = StandardRAGIdeaGenerator(config=config)

    # 加载论文
    logger.info(f"\n📂 加载论文数据: {papers_file}")
    generator.load_papers_from_json(papers_file)

    # 构建向量索引
    logger.info("\n🔨 构建向量索引")
    cache_file = Path(papers_file).stem + "_vector_cache.npy"
    generator.build_index(use_cache=True, cache_file=cache_file)

    # 生成创意
    logger.info(f"\n💡 生成创意 (主题: {topic})")
    ideas = generator.generate_ideas(
        topic=topic,
        num_ideas=num_ideas,
        retrieve_k=retrieve_k
    )

    # 保存结果
    result = {
        'topic': topic,
        'generation_method': 'StandardRAG',
        'ideas': ideas,
        'total_ideas': len(ideas),
        'successful_ideas': len([i for i in ideas if i.get('status') == 'success'])
    }

    logger.info(f"\n💾 保存结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("✅ 测试完成")
    logger.info("=" * 80)
    logger.info(f"生成创意数: {result['total_ideas']}")
    logger.info(f"成功创意数: {result['successful_ideas']}")
    logger.info(f"结果文件: {output_file}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='测试 Standard RAG 创意生成')
    parser.add_argument(
        '--papers',
        type=str,
        default='226_papers_natural_language_processing_20251218_032842.json',
        help='论文数据JSON文件'
    )
    parser.add_argument(
        '--topic',
        type=str,
        default='Natural Language Processing',
        help='研究主题'
    )
    parser.add_argument(
        '--num-ideas',
        type=int,
        default=10,
        help='生成创意数量'
    )
    parser.add_argument(
        '--retrieve-k',
        type=int,
        default=20,
        help='检索论文数量'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='standardrag_ideas.json',
        help='输出文件路径'
    )

    args = parser.parse_args()

    # 检查文件是否存在
    if not Path(args.papers).exists():
        logger.error(f"❌ 论文数据文件不存在: {args.papers}")
        sys.exit(1)

    try:
        result = test_standard_rag(
            papers_file=args.papers,
            topic=args.topic,
            num_ideas=args.num_ideas,
            retrieve_k=args.retrieve_k,
            config_path=args.config,
            output_file=args.output
        )

        logger.info(f"\n🎉 测试成功！")

    except Exception as e:
        logger.error(f"❌ 测试失败: {e}", exc_info=True)
        sys.exit(1)
