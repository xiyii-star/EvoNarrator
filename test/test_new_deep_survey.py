#!/usr/bin/env python3
"""
测试新版 DeepSurveyAnalyzer
基于关系剪枝 + 演化路径识别的方法
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
    从新版分析结果生成Markdown格式报告

    Args:
        result: 深度综述分析结果字典（新版格式：基于Thread）
        json_path: JSON文件路径

    Returns:
        生成的Markdown文件路径
    """
    topic = result['topic']
    survey_report = result['survey_report']
    threads = survey_report.get('threads', [])
    pruning_stats = result.get('pruning_stats', {})

    # 生成Markdown内容
    md_lines = []

    # 标题
    md_lines.append(f"# {survey_report['title']}")
    md_lines.append("")
    md_lines.append(f"**生成时间**: {result.get('timestamp', '')}")
    md_lines.append("")

    # 摘要
    md_lines.append("## 📋 摘要")
    md_lines.append("")
    md_lines.append(survey_report.get('abstract', ''))
    md_lines.append("")

    # 统计概览
    md_lines.append("## 📊 统计概览")
    md_lines.append("")
    md_lines.append("### 图谱剪枝统计")
    md_lines.append("")
    md_lines.append("| 指标 | 数值 |")
    md_lines.append("|------|------|")
    md_lines.append(f"| 原始论文数 | {pruning_stats.get('original_papers', 0)} |")
    md_lines.append(f"| Seed Papers | {pruning_stats.get('seed_papers', 0)} |")
    md_lines.append(f"| 剪枝后论文数 | {pruning_stats.get('pruned_papers', 0)} |")
    md_lines.append(f"| 保留率 | {pruning_stats.get('retention_rate', 0)*100:.1f}% |")
    md_lines.append(f"| 强关系边数 | {pruning_stats.get('strong_edges', 0)} |")
    md_lines.append(f"| 剔除弱关系边 | {pruning_stats.get('weak_edges_removed', 0)} |")
    md_lines.append("")

    # 关系类型分布
    relation_dist = pruning_stats.get('relation_type_distribution', {})
    if relation_dist:
        md_lines.append("### 关系类型分布")
        md_lines.append("")
        md_lines.append("| 关系类型 | 数量 | 占比 |")
        md_lines.append("|---------|------|------|")
        total_relations = sum(relation_dist.values())
        for rel_type, count in sorted(relation_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_relations * 100 if total_relations > 0 else 0
            md_lines.append(f"| {rel_type} | {count} | {percentage:.1f}% |")
        md_lines.append("")

    md_lines.append("### 演化路径")
    md_lines.append("")
    md_lines.append("| 指标 | 数值 |")
    md_lines.append("|------|------|")
    md_lines.append(f"| 演化故事数 (Threads) | {len(threads)} |")

    chain_count = sum(1 for t in threads if t.get('pattern_type', '').startswith('The Chain'))
    star_count = sum(1 for t in threads if t.get('pattern_type', '').startswith('The Star'))
    md_lines.append(f"| 线性链条 (Chain) | {chain_count} |")
    md_lines.append(f"| 星型爆发 (Star) | {star_count} |")
    md_lines.append("")

    # 演化路径详细分析
    md_lines.append("## 🔗 关键演化路径 (Critical Evolutionary Paths)")
    md_lines.append("")
    md_lines.append("这里完全不用担心它们是不连通的，每个Thread都是一个独立的关键故事。")
    md_lines.append("")

    for thread in threads:
        thread_id = thread['thread_id']
        pattern_type = thread['pattern_type']
        title = thread['title']
        narrative = thread['narrative']
        papers = thread['papers']
        total_citations = thread.get('total_citations', 0)
        visual_structure = thread.get('visual_structure', '')
        relation_stats = thread.get('relation_stats', {})

        md_lines.append(f"### Thread {thread_id}: {pattern_type}")
        md_lines.append("")
        md_lines.append(f"**{title}**")
        md_lines.append("")

        # 可视化结构
        if visual_structure:
            md_lines.append("**演化结构**:")
            md_lines.append("")
            md_lines.append(f"```")
            md_lines.append(visual_structure)
            md_lines.append(f"```")
            md_lines.append("")

        # 关系统计
        if relation_stats:
            md_lines.append("**关系统计**:")
            md_lines.append("")
            md_lines.append(f"- 总关系数: {relation_stats.get('total_relations', 0)}")
            md_lines.append(f"- 主导关系: {relation_stats.get('dominant_relation', 'Unknown')}")

            rel_dist = relation_stats.get('relation_distribution', {})
            if rel_dist:
                md_lines.append("- 分布: " + ", ".join([f"{k}({v})" for k, v in rel_dist.items()]))
            md_lines.append("")

        # 详细关系链（新增）
        relation_chain = thread.get('relation_chain', [])
        if relation_chain:
            md_lines.append("**详细关系链**:")
            md_lines.append("")

            if thread.get('thread_type') == 'chain':
                # 线性链条：显示论文演进路径
                md_lines.append("| 源论文 | 关系类型 | 目标论文 |")
                md_lines.append("|--------|----------|----------|")
                for rel in relation_chain:
                    from_title = rel['from_paper']['title'][:50] + "..." if len(rel['from_paper']['title']) > 50 else rel['from_paper']['title']
                    to_title = rel['to_paper']['title'][:50] + "..." if len(rel['to_paper']['title']) > 50 else rel['to_paper']['title']
                    from_year = rel['from_paper']['year']
                    to_year = rel['to_paper']['year']
                    relation = rel['relation_type']

                    md_lines.append(f"| {from_title} ({from_year}) | **{relation}** | {to_title} ({to_year}) |")

            elif thread.get('thread_type') == 'star':
                # 星型结构：显示中心论文到各路线的关系
                md_lines.append("| 路线 | 中心论文 | 关系类型 | 目标论文 |")
                md_lines.append("|------|----------|----------|----------|")
                for rel in relation_chain:
                    route_id = rel.get('route_id', '?')
                    from_title = rel['from_paper']['title'][:40] + "..." if len(rel['from_paper']['title']) > 40 else rel['from_paper']['title']
                    to_title = rel['to_paper']['title'][:40] + "..." if len(rel['to_paper']['title']) > 40 else rel['to_paper']['title']
                    from_year = rel['from_paper']['year']
                    to_year = rel['to_paper']['year']
                    relation = rel['relation_type']

                    md_lines.append(f"| 路线{route_id} | {from_title} ({from_year}) | **{relation}** | {to_title} ({to_year}) |")

            md_lines.append("")

        # 叙事文本
        md_lines.append("**演化叙事**:")
        md_lines.append("")
        md_lines.append(narrative)
        md_lines.append("")

        # 涉及论文
        md_lines.append("**涉及论文**:")
        md_lines.append("")
        md_lines.append(f"- 论文数量: {len(papers)}")
        md_lines.append(f"- 总引用数: {total_citations}")
        md_lines.append("")

        md_lines.append("**代表性论文列表**:")
        md_lines.append("")
        md_lines.append("| 标题 | 年份 | 引用数 |")
        md_lines.append("|------|------|--------|")
        for paper in papers[:5]:  # 只显示前5篇
            title_short = paper['title'][:60] + "..." if len(paper['title']) > 60 else paper['title']
            md_lines.append(f"| {title_short} | {paper.get('year', 'N/A')} | {paper.get('cited_by_count', 0)} |")
        md_lines.append("")

        md_lines.append("---")
        md_lines.append("")

    # 方法论说明
    md_lines.append("## 🔬 方法论说明")
    md_lines.append("")
    md_lines.append("### 第一步：基于关系的图谱剪枝 (Relation-Based Pruning)")
    md_lines.append("")
    md_lines.append("- ✅ 保留所有 Seed Papers")
    md_lines.append("- ✅ 通过强逻辑关系（Overcomes, Realizes, Extends, Alternative, Adapts_to）进行连通性分析")
    md_lines.append("- ✅ 剔除仅由弱关系（Baselines）连接的论文")
    md_lines.append("- ✅ 极大提升数据纯度")
    md_lines.append("")

    md_lines.append("### 第二步：关键演化路径识别 (Critical Evolutionary Paths)")
    md_lines.append("")
    md_lines.append("**识别两种核心演化模式：**")
    md_lines.append("")
    md_lines.append("1. **线性链条 (The Chain)** - 技术迭代故事")
    md_lines.append("   - 结构：A -> (Overcomes) -> B -> (Extends) -> C")
    md_lines.append("   - 叙事模板：起因 → 转折 → 发展")
    md_lines.append("")
    md_lines.append("2. **星型爆发 (The Star)** - 百家争鸣故事")
    md_lines.append("   - 结构：Seed -> (Overcomes) -> A, Seed -> (Alternative) -> B, Seed -> (Extends) -> C")
    md_lines.append("   - 叙事模板：焦点 → 分歧 → 对比")
    md_lines.append("")

    md_lines.append("### 第三步：结构化 Deep Survey 报告")
    md_lines.append("")
    md_lines.append("- 📊 Thread 形式展示各个演化故事")
    md_lines.append("- 📈 配合可视化图和文字解读")
    md_lines.append("- 🎯 每个Thread是独立的关键故事，互不连通也清晰")
    md_lines.append("")

    # 结论
    md_lines.append("## 🎯 结论")
    md_lines.append("")
    md_lines.append(f"本综述基于知识图谱剪枝技术，从 {pruning_stats.get('original_papers', 0)} 篇论文中")
    md_lines.append(f"筛选出 {pruning_stats.get('pruned_papers', 0)} 篇高质量论文，")
    md_lines.append(f"并识别出 {len(threads)} 条关键演化路径（{chain_count} 条线性链条 + {star_count} 个星型爆发）。")
    md_lines.append("")
    md_lines.append("通过关系类型分析和演化路径识别，完整呈现了该领域的技术演进脉络和多元化发展趋势。")
    md_lines.append("")

    # 生成Markdown文本
    markdown_text = "\n".join(md_lines)

    # 保存文件
    md_file = json_path.parent / json_path.name.replace('.json', '.md')
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(markdown_text)

    return str(md_file)


def test_new_deep_survey(
    papers_file: str,
    graph_file: str,
    topic: str,
    output_file: str = None
):
    """
    测试新版深度综述分析器

    Args:
        papers_file: 论文数据JSON文件路径
        graph_file: 知识图谱JSON文件路径
        topic: 研究主题
        output_file: 输出文件路径（可选）
    """
    logger.info("="*80)
    logger.info("🧪 开始测试新版 DeepSurveyAnalyzer")
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
        citation_graph.add_paper_node(paper)

    # 添加边
    edges_data = graph_data.get('edges', [])
    for edge in edges_data:
        source = edge.get('from') or edge.get('source')
        target = edge.get('to') or edge.get('target')
        edge_type = edge.get('edge_type') or edge.get('type', 'CITES')

        if source and target:
            citation_graph.add_citation_edge(source, target, edge_type=edge_type)

    logger.info(f"  ✅ 知识图谱重建完成")
    logger.info(f"     - 节点数: {citation_graph.graph.number_of_nodes()}")
    logger.info(f"     - 边数: {citation_graph.graph.number_of_edges()}")

    # 4. 初始化新版深度综述分析器
    logger.info("\n🔧 初始化新版深度综述分析器...")
    config = {
        'llm_config_path': './config/config.yaml',
        'min_chain_length': 3,  # 最小链条长度
        'max_threads': 5,       # 最多保留5个Thread
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
        logger.info(f"  - 演化路径数 (Threads): {result['summary']['total_threads']}")

        # 显示演化路径信息
        threads = result['survey_report']['threads']
        logger.info("\n🔗 演化路径详情:")
        for thread in threads:
            logger.info(f"\n  Thread {thread['thread_id']}: {thread['pattern_type']}")
            logger.info(f"    📝 标题: {thread['title']}")
            logger.info(f"    📄 论文数: {len(thread['papers'])}")
            logger.info(f"    📊 总引用数: {thread['total_citations']}")
            logger.info(f"    🔗 结构: {thread['visual_structure']}")

            # 关系统计
            rel_stats = thread.get('relation_stats', {})
            if rel_stats:
                logger.info(f"    🔀 关系统计:")
                logger.info(f"       - 总关系: {rel_stats.get('total_relations', 0)}")
                logger.info(f"       - 主导关系: {rel_stats.get('dominant_relation', 'Unknown')}")
                rel_dist = rel_stats.get('relation_distribution', {})
                if rel_dist:
                    logger.info(f"       - 分布: {rel_dist}")

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

    # 测试新版分析器
    try:
        result = test_new_deep_survey(papers_file, graph_file, topic)
        logger.info("\n🎉 测试完成!")
    except Exception as e:
        logger.error(f"\n❌ 测试失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
