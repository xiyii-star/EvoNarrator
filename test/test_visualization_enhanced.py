"""
增强版测试可视化生成脚本
使用完整的 graph_data.json + deep_survey + research_ideas 生成交互式知识图谱
"""

import json
import logging
from pathlib import Path
from src.knowledge_graph import CitationGraph

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_json_file(file_path: str) -> dict:
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"✓ 成功加载: {Path(file_path).name}")
        return data
    except Exception as e:
        logger.error(f"✗ 加载失败 {file_path}: {e}")
        return {}


def merge_papers_data(papers_list: list, graph_data: dict) -> list:
    """
    合并 papers.json 和 graph_data.json 中的论文信息
    papers_list: 从 papers.json 加载的完整论文列表
    graph_data: 从 graph_data.json 加载的图谱数据（包含节点的分析结果）
    """
    # 创建 paper_id 到 paper 数据的映射
    papers_dict = {p['id']: p for p in papers_list}

    # 从 graph_data 的 nodes 中获取分析结果
    merged_papers = []

    if 'nodes' in graph_data:
        for node in graph_data['nodes']:
            paper_id = node.get('id', '')

            # 获取基础论文数据
            if paper_id in papers_dict:
                paper = papers_dict[paper_id].copy()
            else:
                # 如果 papers_list 中没有，使用 node 中的基础信息
                paper = {
                    'id': paper_id,
                    'title': node.get('title', 'Unknown'),
                    'authors': node.get('authors', []),
                    'year': node.get('year', 2020),
                    'cited_by_count': node.get('cited_by_count', 0),
                    'venue': node.get('venue', ''),
                    'is_open_access': node.get('is_open_access', False),
                    'is_seed': node.get('is_seed', False)
                }

            # 添加来自 graph_data nodes 的分析结果
            if 'deep_analysis' in node:
                paper['deep_analysis'] = node['deep_analysis']

            if 'rag_analysis' in node:
                paper['rag_analysis'] = node['rag_analysis']

            if 'ai_analysis' in node:
                paper['ai_analysis'] = node['ai_analysis']

            # 添加其他节点属性
            if 'analysis_method' in node:
                paper['analysis_method'] = node['analysis_method']

            if 'sections_extracted' in node:
                paper['sections_extracted'] = node['sections_extracted']

            merged_papers.append(paper)

    logger.info(f"✓ 合并论文数据: {len(merged_papers)} 篇论文")
    return merged_papers


def extract_citations_from_graph_data(graph_data: dict) -> list:
    """从 graph_data.json 中提取引用关系"""
    citations = []

    if 'edges' in graph_data:
        for edge in graph_data['edges']:
            from_id = edge.get('from', '')
            to_id = edge.get('to', '')
            edge_type = edge.get('edge_type', 'Baselines')

            if from_id and to_id:
                citations.append((from_id, to_id, edge_type))

    logger.info(f"✓ 从图谱数据中提取了 {len(citations)} 个引用关系")
    return citations


def main():
    """主函数"""
    logger.info("=" * 70)
    logger.info("增强版可视化测试 - 使用完整图谱数据")
    logger.info("=" * 70)

    # 文件路径
    papers_file = "/home/lexy/下载/CLwithRAG/KGdemo/226_papers_natural_language_processing_20251218_032842.json"
    graph_data_path = "/home/lexy/下载/CLwithRAG/KGdemo/226_graph_data_natural_language_processing_20251218_032842.json"
    deep_survey_path = "/home/lexy/下载/CLwithRAG/KGdemo/output/deep_survey/Natural_Language_Processing_20251221_213646.json"
    research_ideas_path = "/home/lexy/下载/CLwithRAG/KGdemo/test_idea_generation_result.json"
    output_viz_path = "/home/lexy/下载/CLwithRAG/KGdemo/test_visualization_papers.html"

    # 1. 加载数据
    logger.info("\n📂 步骤 1: 加载数据文件")
    papers_list = load_json_file(papers_file)
    graph_data = load_json_file(graph_data_path)
    deep_survey_data = load_json_file(deep_survey_path)
    research_ideas_data = load_json_file(research_ideas_path)

    if not graph_data:
        logger.error("❌ 图谱数据加载失败，退出")
        return

    # 如果 papers_list 不是列表，尝试转换
    if isinstance(papers_list, dict):
        logger.warning("papers_file 是字典格式，尝试提取论文列表")
        papers_list = papers_list.get('papers', [])

    # 2. 合并论文数据和提取引用关系
    logger.info("\n🔍 步骤 2: 合并论文数据和提取引用关系")
    papers = merge_papers_data(papers_list, graph_data)
    citations = extract_citations_from_graph_data(graph_data)

    if not papers:
        logger.error("❌ 未能提取到任何论文数据，退出")
        return

    # 统计分析方法类型
    analysis_methods = {}
    for paper in papers:
        method = paper.get('analysis_method', 'unknown')
        analysis_methods[method] = analysis_methods.get(method, 0) + 1

    # 3. 创建知识图谱
    logger.info("\n🏗️  步骤 3: 构建知识图谱")
    topic = graph_data.get('metadata', {}).get('topic', 'Natural Language Processing')
    citation_graph = CitationGraph(topic=topic)

    # 构建引用网络
    citation_graph.build_citation_network(papers, citations)

    # 4. 生成可视化
    logger.info("\n🎨 步骤 4: 生成交互式可视化")
    max_nodes = 300  # 显示最多300个节点（足够包含所有226篇论文）

    viz_file = citation_graph.visualize_graph(
        output_path=output_viz_path,
        max_nodes=max_nodes,
        deep_survey_report=deep_survey_data,
        research_ideas=research_ideas_data
    )

    # 5. 统计引用关系类型
    edge_types = {}
    for citation in citations:
        if len(citation) >= 3:
            edge_type = citation[2]
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

    # 6. 输出详细统计
    logger.info("\n" + "=" * 70)
    logger.info("✅ 可视化生成完成!")
    logger.info("=" * 70)

    logger.info("\n📊 数据统计:")
    logger.info(f"  • 论文总数: {len(papers)}")
    logger.info(f"  • 种子论文: {sum(1 for p in papers if p.get('is_seed', False))}")
    logger.info(f"  • 引用关系: {len(citations)}")
    logger.info(f"  • 显示节点: {min(len(papers), max_nodes)}")

    logger.info("\n🔬 分析方法分布:")
    for method, count in sorted(analysis_methods.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  • {method}: {count} 篇")

    logger.info("\n🔗 引用关系类型分布:")
    for edge_type, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  • {edge_type}: {count} 个")

    logger.info("\n📝 综述与创意:")
    logger.info(f"  • 研究演化路径: {len(deep_survey_data.get('evolutionary_paths', []))}")
    logger.info(f"  • 关键转折论文: {len(deep_survey_data.get('pivotal_papers', []))}")
    logger.info(f"  • 科研创意总数: {research_ideas_data.get('total_ideas', 0)}")
    logger.info(f"  • 可行创意数量: {research_ideas_data.get('successful_ideas', 0)}")

    logger.info("\n📄 输出文件:")
    logger.info(f"  {viz_file}")

    logger.info("\n💡 使用提示:")
    logger.info("  1. 在浏览器中打开 HTML 文件查看交互式可视化")
    logger.info("  2. 点击节点查看论文详情和 RAG 分析结果")
    logger.info("  3. 切换标签页查看深度综述和科研创意")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
