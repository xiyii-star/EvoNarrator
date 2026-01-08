#!/usr/bin/env python3
"""
重新生成深度综述脚本
使用已有的论文数据和知识图谱，重新生成综述报告
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from deep_survey_analyzer import DeepSurveyAnalyzer
from knowledge_graph import CitationGraph

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理datetime对象"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def generate_markdown_from_result(result: dict, json_path: Path) -> str:
    """
    从分析结果生成Markdown格式报告

    Args:
        result: 深度综述分析结果字典
        json_path: JSON文件路径

    Returns:
        生成的Markdown文件路径
    """
    topic = result['topic']
    evolutionary_paths = result['evolutionary_paths']
    survey_report = result.get('survey_report', {})

    # 生成Markdown内容
    md_lines = []

    # 标题
    md_lines.append(f"# Deep Survey: {topic}")
    md_lines.append("")
    md_lines.append(f"**生成时间**: {result.get('timestamp', '')}")
    md_lines.append("")

    # 综述摘要（来自 survey_report）
    if survey_report and survey_report.get('abstract'):
        md_lines.append("## 综述摘要")
        md_lines.append("")
        md_lines.append(survey_report['abstract'])
        md_lines.append("")

    # 统计信息
    md_lines.append("## 统计概览")
    md_lines.append("")
    md_lines.append("| 指标 | 数值 |")
    md_lines.append("|------|------|")
    md_lines.append(f"| 演化路径数 | {len(evolutionary_paths)} |")

    # 计算总论文数（去重）
    all_paper_ids = set()
    for path in evolutionary_paths:
        for paper in path.get('papers', []):
            all_paper_ids.add(paper['paper_id'])
    total_papers = len(all_paper_ids)
    md_lines.append(f"| 相关论文总数 | {total_papers} |")

    # 计算总引用数
    total_citations = sum(path.get('total_citations', 0) for path in evolutionary_paths)
    md_lines.append(f"| 总引用数 | {total_citations} |")

    # 剪枝统计
    if 'pruning_stats' in result:
        stats = result['pruning_stats']
        md_lines.append(f"| 原始论文数 | {stats.get('original_papers', 'N/A')} |")
        md_lines.append(f"| 剪枝保留率 | {stats.get('retention_rate', 0)*100:.1f}% |")

    md_lines.append("")

    # 演化路径详细分析
    md_lines.append("## 演化路径详细分析")
    md_lines.append("")

    for i, path in enumerate(evolutionary_paths, 1):
        pattern_type = path.get('pattern_type', 'Unknown')
        md_lines.append(f"### 路径 {i}: {pattern_type}")
        md_lines.append("")

        # 路径标题
        if path.get('title'):
            md_lines.append(f"**{path['title']}**")
            md_lines.append("")

        # 路径概览
        md_lines.append("#### 📊 路径概览")
        md_lines.append("")
        md_lines.append(f"- **模式类型**: {pattern_type}")
        md_lines.append(f"- **论文数量**: {len(path.get('papers', []))}")
        md_lines.append(f"- **总引用数**: {path.get('total_citations', 0)}")
        md_lines.append(f"- **结构**: {path.get('visual_structure', 'N/A')}")
        md_lines.append("")

        # Star 类型特有信息
        if path.get('thread_type') == 'star':
            if path.get('center_paper'):
                md_lines.append(f"- **中心论文**: {path['center_paper']}")
            if path.get('routes_count'):
                md_lines.append(f"- **分支路线数**: {path['routes_count']}")
            md_lines.append("")

        # 演化叙事
        if path.get('narrative'):
            md_lines.append("#### 📝 演化叙事")
            md_lines.append("")
            md_lines.append(path['narrative'])
            md_lines.append("")

        # Chain 类型：显示关系链
        if path.get('thread_type') == 'chain' and path.get('relation_chain'):
            md_lines.append("#### 🔗 演化关系链")
            md_lines.append("")
            relation_chain = path['relation_chain']
            for j, rel in enumerate(relation_chain):
                from_paper = rel.get('from_paper', {})
                to_paper = rel.get('to_paper', {})
                relation = rel.get('narrative_relation', rel.get('relation_type', 'Unknown'))

                from_title = from_paper.get('title', 'Unknown')[:50]
                to_title = to_paper.get('title', 'Unknown')[:50]
                from_year = from_paper.get('year', '?')
                to_year = to_paper.get('year', '?')

                md_lines.append(f"{j+1}. **{from_title}** ({from_year}) --{relation}--> **{to_title}** ({to_year})")
            md_lines.append("")

        # Star 类型：显示各条路线
        if path.get('thread_type') == 'star' and path.get('routes'):
            md_lines.append("#### 🌟 分支路线")
            md_lines.append("")
            for route_idx, route in enumerate(path['routes'], 1):
                relation_type = route.get('relation_type', 'Unknown')
                route_papers = route.get('papers', [])
                md_lines.append(f"**路线 {route_idx}** ({relation_type}):")
                for paper in route_papers[:3]:  # 只显示前3篇
                    paper_title = paper.get('title', 'Unknown')[:60]
                    paper_year = paper.get('year', '?')
                    md_lines.append(f"  - {paper_title} ({paper_year})")
                if len(route_papers) > 3:
                    md_lines.append(f"  - ... 共 {len(route_papers)} 篇论文")
            md_lines.append("")

        # 核心论文列表
        if path.get('papers'):
            md_lines.append("#### ⭐ 核心论文")
            md_lines.append("")
            md_lines.append("| 标题 | 年份 | 引用数 | 论文ID |")
            md_lines.append("|------|------|--------|--------|")
            for paper in path['papers'][:10]:  # 最多显示10篇
                title = paper.get('title', 'Unknown')
                title = title[:60] + "..." if len(title) > 60 else title
                year = paper.get('year', 'N/A')
                citations = paper.get('cited_by_count', 0)
                paper_id = paper.get('paper_id', 'N/A')
                role = paper.get('role', '')
                role_marker = f" ({role})" if role else ""
                md_lines.append(f"| {title}{role_marker} | {year} | {citations} | `{paper_id}` |")
            md_lines.append("")

        md_lines.append("---")
        md_lines.append("")

    # 研究趋势与展望
    md_lines.append("## 研究趋势与展望")
    md_lines.append("")

    if len(evolutionary_paths) >= 1:
        md_lines.append("### 演化模式分析")
        md_lines.append("")

        # 统计不同类型的路径
        chain_count = sum(1 for p in evolutionary_paths if p.get('thread_type') == 'chain')
        star_count = sum(1 for p in evolutionary_paths if p.get('thread_type') == 'star')

        if chain_count > 0:
            md_lines.append(f"- **线性链条模式** ({chain_count} 条): 体现了技术的渐进式演化，后续研究逐步克服前人局限")
        if star_count > 0:
            md_lines.append(f"- **星型爆发模式** ({star_count} 个): 展示了基础性工作如何激发多个研究方向")
        md_lines.append("")

    # 方法论说明
    md_lines.append("## 方法论说明")
    md_lines.append("")
    md_lines.append("1. **关系剪枝**: 基于论文间的语义关系（Overcomes、Extends、Adapts等）进行图谱剪枝")
    md_lines.append("2. **演化路径识别**: 识别线性链条和星型爆发两种核心演化模式")
    md_lines.append("3. **LLM辅助叙事**: 使用大语言模型生成流畅的演化叙事描述")
    md_lines.append("4. **质量筛选**: 基于引用数和关系强度筛选高质量演化路径")
    md_lines.append("")

    # 结论
    md_lines.append("## 结论")
    md_lines.append("")
    md_lines.append(f"本综述通过知识图谱分析识别出 {topic} 领域的 {len(evolutionary_paths)} 条关键演化路径，")
    md_lines.append(f"涵盖 {total_papers} 篇高质量论文，总引用数达 {total_citations}。")
    md_lines.append(f"这些演化路径揭示了该领域的技术演进脉络和多元化发展趋势。")
    md_lines.append("")

    # 生成Markdown文本
    markdown_text = "\n".join(md_lines)

    # 保存文件
    md_file = json_path.parent / json_path.name.replace('.json', '.md')
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(markdown_text)

    return str(md_file)


def regenerate_survey(
    papers_file: str,
    graph_file: str,
    topic: str,
    output_file: str = None
):
    """
    重新生成深度综述

    Args:
        papers_file: 论文数据JSON文件路径
        graph_file: 知识图谱JSON文件路径
        topic: 研究主题
        output_file: 输出文件路径（可选）
    """
    logger.info("="*80)
    logger.info("🔄 开始重新生成深度综述")
    logger.info("="*80)

    # 1. 加载论文数据
    logger.info(f"\n📂 加载论文数据: {papers_file}")
    with open(papers_file, 'r', encoding='utf-8') as f:
        papers_data = json.load(f)
    logger.info(f"  ✅ 加载了 {len(papers_data)} 篇论文")

    # 2. 加载知识图谱数据
    logger.info(f"\n📂 加载知识图谱: {graph_file}")
    with open(graph_file, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

    # 3. 重建知识图谱
    logger.info("\n🔨 重建知识图谱...")
    citation_graph = CitationGraph()

    # 添加节点
    for paper in papers_data:
        # 提取deep_analysis中的信息
        deep_analysis = paper.get('deep_analysis', {})

        # 添加deep_analysis信息（兼容旧版本字段名）
        if deep_analysis:
            paper['rag_analysis'] = {
                'problem': deep_analysis.get('problem', ''),
                'method': deep_analysis.get('method', ''),
                'contribution': deep_analysis.get('contribution', ''),
                'limitation': deep_analysis.get('limitation', ''),
                'future_work': deep_analysis.get('future_work', ''),
            }

        citation_graph.add_paper_node(paper)

    # 添加边
    edges_data = graph_data.get('edges', [])
    for edge in edges_data:
        # 兼容不同的边数据结构
        source = edge.get('from') or edge.get('source')
        target = edge.get('to') or edge.get('target')
        edge_type = edge.get('edge_type') or edge.get('type', 'CITES')

        if source and target:
            citation_graph.add_citation_edge(source, target, edge_type=edge_type)

    logger.info(f"  ✅ 知识图谱重建完成")
    logger.info(f"     - 节点数: {citation_graph.graph.number_of_nodes()}")
    logger.info(f"     - 边数: {citation_graph.graph.number_of_edges()}")

    # 4. 初始化深度综述分析器（使用优化后的版本）
    logger.info("\n🔧 初始化深度综述分析器...")
    config = {
        'embedding_model': 'all-MiniLM-L6-v2',
        'use_modelscope': True,
        'relevance_threshold': 0.3,  # 使用新的相关性过滤功能
        'llm_config_path': './config/config.yaml'
    }

    analyzer = DeepSurveyAnalyzer(config=config)
    logger.info("  ✅ 分析器初始化完成")

    # 5. 执行深度综述分析
    logger.info(f"\n📊 开始分析主题: '{topic}'")
    logger.info("="*80)

    try:
        result = analyzer.analyze(citation_graph.graph, topic)

        logger.info("\n✅ 深度综述分析完成!")
        logger.info("="*80)

        # 6. 显示结果摘要
        logger.info("\n📈 分析结果摘要:")
        logger.info(f"  - 原始论文数: {result['summary']['original_papers']}")
        logger.info(f"  - 剪枝后论文数: {result['summary']['pruned_papers']}")
        logger.info(f"  - 识别演化路径数: {result['summary']['total_threads']}")

        # 显示剪枝统计
        if 'pruning_stats' in result:
            stats = result['pruning_stats']
            logger.info(f"\n📊 图谱剪枝统计:")
            logger.info(f"  - 剪枝模式: {stats.get('pruning_mode', 'N/A')}")
            logger.info(f"  - 保留率: {stats.get('retention_rate', 0)*100:.1f}%")
            logger.info(f"  - 强关系边: {stats.get('strong_edges', 0)}")
            logger.info(f"  - 剔除弱关系边: {stats.get('weak_edges_removed', 0)}")
            if stats.get('strong_components_count', 0) > 0:
                logger.info(f"  - 强关系连通分量数: {stats['strong_components_count']}")
                logger.info(f"  - 最大连通分量大小: {stats.get('largest_component_size', 0)}")

        # 显示演化路径详情
        logger.info("\n📚 演化路径详情:")
        for i, path in enumerate(result.get('evolutionary_paths', []), 1):
            logger.info(f"\n  Thread {i}: {path.get('pattern_type', 'Unknown')}")
            logger.info(f"    📝 标题: {path.get('title', 'N/A')}")
            logger.info(f"    📄 论文数量: {len(path.get('papers', []))}")
            logger.info(f"    📈 总引用数: {path.get('total_citations', 0)}")
            logger.info(f"    🔗 结构: {path.get('visual_structure', 'N/A')}")

            # 显示代表性论文
            papers = path.get('papers', [])[:3]
            if papers:
                logger.info(f"    ⭐ 代表性论文:")
                for paper in papers:
                    logger.info(f"       - {paper.get('title', 'N/A')[:60]}... ({paper.get('year', 'N/A')})")

        # 显示综述报告摘要
        if 'survey_report' in result and result['survey_report']:
            report = result['survey_report']
            logger.info(f"\n📖 综述报告:")
            logger.info(f"  - 标题: {report.get('title', 'N/A')}")
            if 'abstract' in report:
                abstract = report['abstract']
                logger.info(f"  - 摘要: {abstract[:200]}..." if len(abstract) > 200 else f"  - 摘要: {abstract}")

        # 7. 保存结果
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            topic_safe = topic.replace(' ', '_').replace('/', '_')
            output_file = f"output/deep_survey/{topic_safe}_{timestamp}.json"

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)

        logger.info(f"\n💾 深度综述报告已保存至: {output_path}")

        # 8. 生成Markdown格式报告
        logger.info("\n📄 生成Markdown格式报告...")
        md_file = generate_markdown_from_result(result, output_path)
        logger.info(f"  ✅ Markdown报告已保存至: {md_file}")

        logger.info("="*80)

        return result

    except Exception as e:
        logger.error(f"❌ 深度综述生成失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """主函数"""
    # 配置参数
    papers_file = "/home/lexy/下载/CLwithRAG/KGdemo/226_papers_natural_language_processing_20251218_032842.json"
    graph_file = "/home/lexy/下载/CLwithRAG/KGdemo/226_graph_data_natural_language_processing_20251218_032842.json"
    topic = "Natural Language Processing"

    # 检查文件是否存在
    if not Path(papers_file).exists():
        logger.error(f"❌ 论文数据文件不存在: {papers_file}")
        return

    if not Path(graph_file).exists():
        logger.error(f"❌ 知识图谱文件不存在: {graph_file}")
        return

    # 重新生成综述
    try:
        result = regenerate_survey(papers_file, graph_file, topic)
        logger.info("\n🎉 综述重新生成完成!")
    except Exception as e:
        logger.error(f"\n❌ 程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
