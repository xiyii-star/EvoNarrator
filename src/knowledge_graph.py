"""
知识图谱构建和可视化模块
构建论文引用关系图谱并进行可视化
"""

import networkx as nx
import json
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理datetime对象"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class CitationGraph:
    """
    论文引用知识图谱构建器
    """

    def __init__(self, topic: str = ""):
        self.graph = nx.DiGraph()
        self.papers = {}  # 存储论文详细信息
        self.topic = topic  # 研究主题
        self.deep_survey_report = {}  # 存储深度综述报告
        self.research_ideas = {}  # 存储科研创意
        logger.info("知识图谱构建器初始化完成")

    # 结点
    def add_paper_node(self, paper: Dict) -> None:
        """
        添加论文节点到图中

        Args:
            paper: 论文信息字典
        """
        paper_id = paper['id']

        # 将论文信息存储
        self.papers[paper_id] = paper

        # 提取节点属性
        node_attrs = {
            'title': paper.get('title', 'Unknown'),
            'authors': paper.get('authors', []),
            'year': paper.get('year', 0),
            'cited_by_count': paper.get('cited_by_count', 0),
            'venue': paper.get('venue', ''),
            'is_open_access': paper.get('is_open_access', False),
            'is_seed': paper.get('is_seed', False)  # 添加种子节点标记
        }

        # 如果有AI分析结果，添加到节点属性（兼容ai_analysis和deep_analysis两种字段）
        analysis = paper.get('deep_analysis') or paper.get('ai_analysis')
        if analysis:
            # 兼容两种数据结构
            # deep_analysis结构: {problem: {content: ...}, method: {content: ...}, ...}
            # ai_analysis结构: {research_problem: ..., solution: ..., ...}

            # 提取problem/research_problem
            problem = ''
            if isinstance(analysis.get('problem'), dict):
                problem = analysis['problem'].get('content', '')
            else:
                problem = analysis.get('research_problem', '')

            # 提取method/contributions
            method = ''
            contributions_list = []
            if isinstance(analysis.get('method'), dict):
                method = analysis['method'].get('content', '')
            else:
                contributions_list = analysis.get('contributions', [])

            # 提取limitation/limitations
            limitation = ''
            limitations_list = []
            if isinstance(analysis.get('limitation'), dict):
                limitation = analysis['limitation'].get('content', '')
            else:
                limitations_list = analysis.get('limitations', [])

            # 提取future_work
            future_work = ''
            if isinstance(analysis.get('future_work'), dict):
                future_work = analysis['future_work'].get('content', '')
            else:
                future_work = analysis.get('future_work', '')

            node_attrs.update({
                'research_problem': problem,
                'solution': analysis.get('solution', ''),
                'key_techniques': analysis.get('key_techniques', []),
                'contributions': contributions_list or [method] if method else [],
                'limitations': limitations_list or [limitation] if limitation else [],
                'deep_analysis': analysis  # 保存完整的deep_analysis结构
            })

        # 如果有 rag_analysis 结果，添加到节点属性（用于碎片池化）
        rag_analysis = paper.get('rag_analysis')
        if rag_analysis:
            node_attrs.update({
                'rag_limitation': rag_analysis.get('limitation', ''),
                'rag_method': rag_analysis.get('method', ''),
                'rag_problem': rag_analysis.get('problem', ''),
                'rag_future_work': rag_analysis.get('future_work', '')
            })

        self.graph.add_node(paper_id, **node_attrs)
        logger.debug(f"添加论文节点: {paper_id}")

    def _calculate_node_size(self, cited_count: int, graph) -> float:
        """
        计算节点大小，基于引用数的相对重要性

        Args:
            cited_count: 该论文的引用数
            graph: 当前图谱（用于计算相对重要性）

        Returns:
            节点大小
        """
        # 获取图中所有论文的引用数
        all_citations = []
        for node in graph.nodes():
            paper = self.papers.get(node, {})
            citations = paper.get('cited_by_count', 0)
            all_citations.append(citations)

        if not all_citations:
            return 20  # 默认大小

        # 计算统计量
        min_citations = min(all_citations)
        max_citations = max(all_citations)
        avg_citations = sum(all_citations) / len(all_citations)

        # 定义大小范围
        min_size = 8   # 最小节点大小
        max_size = 60  # 最大节点大小

        if max_citations == min_citations:
            # 所有论文引用数相同，返回中等大小
            return (min_size + max_size) / 2

        # 使用对数缩放，让差异更明显但不会过于极端
        import math
        if cited_count <= 0:
            return min_size

        # 对数缩放公式
        log_cited = math.log(cited_count + 1)
        log_max = math.log(max_citations + 1)
        log_min = math.log(min_citations + 1) if min_citations > 0 else 0

        if log_max == log_min:
            return (min_size + max_size) / 2

        # 线性映射到目标范围
        normalized = (log_cited - log_min) / (log_max - log_min)
        size = min_size + normalized * (max_size - min_size)

        # 额外的分层逻辑
        if cited_count >= avg_citations * 3:
            # 超高引用论文，额外加大
            size *= 1.2
        elif cited_count >= avg_citations * 2:
            # 高引用论文，适度加大
            size *= 1.1
        elif cited_count < avg_citations * 0.3:
            # 低引用论文，适度缩小
            size *= 0.8

        return max(min_size, min(max_size, size))

    def _get_node_color(self, year):
        """根据年份获取节点颜色"""
        if not isinstance(year, int) or year < 1900:
            return '#808080'  # 灰色，未知年份

        # 将年份映射到颜色值 (1990-2024)
        min_year, max_year = 1990, 2024
        normalized = (year - min_year) / (max_year - min_year)
        normalized = max(0, min(1, normalized))  # 确保在0-1范围内

        # 使用HSV色彩空间：从蓝色到红色
        hue = (1 - normalized) * 240  # 240度是蓝色，0度是红色
        return f'hsl({hue:.0f}, 70%, 60%)'

    # 边
    def add_citation_edge(self, from_paper_id: str, to_paper_id: str, edge_type: str = "CITES") -> None:
        """
        添加引用关系边

        Args:
            from_paper_id: 引用论文ID
            to_paper_id: 被引用论文ID
            edge_type: 引用关系类型
        """
        if from_paper_id in self.graph and to_paper_id in self.graph:
            self.graph.add_edge(
                from_paper_id,
                to_paper_id,
                edge_type=edge_type,
                weight=1
            )
            logger.debug(f"添加引用边: {from_paper_id} -> {to_paper_id} ({edge_type})")

    def _infer_edge_type(self, citing_id: str, cited_id: str) -> str:
        """
        阶段四为进行引用关系推断时执行

        Args:
            citing_id: 引用论文ID
            cited_id: 被引用论文ID

        Returns:
            引用关系类型
        """
        if citing_id not in self.papers or cited_id not in self.papers:
            return "CITES"

        citing_paper = self.papers[citing_id]
        cited_paper = self.papers[cited_id]

        # 获取基本信息
        citing_year = citing_paper.get('year', 0)
        cited_year = cited_paper.get('year', 0)
        citing_citations = citing_paper.get('cited_by_count', 0)
        cited_citations = cited_paper.get('cited_by_count', 0)

        # 获取论文标题和关键技术（如果有AI分析结果）
        citing_title = citing_paper.get('title', '').lower()
        cited_title = cited_paper.get('title', '').lower()

        citing_techniques = []
        cited_techniques = []
        if 'ai_analysis' in citing_paper:
            citing_techniques = [tech.lower() for tech in citing_paper['ai_analysis'].get('key_techniques', [])]
        if 'ai_analysis' in cited_paper:
            cited_techniques = [tech.lower() for tech in cited_paper['ai_analysis'].get('key_techniques', [])]

        # 1. 基于时间的推断
        year_diff = citing_year - cited_year if citing_year > 0 and cited_year > 0 else 0

        # 2. 基于影响力的推断
        citation_ratio = citing_citations / max(1, cited_citations) if cited_citations > 0 else 1

        # 3. 基于技术相似性的推断
        common_techniques = set(citing_techniques) & set(cited_techniques)
        technique_similarity = len(common_techniques) / max(1, len(set(citing_techniques) | set(cited_techniques)))

        # 4. 基于标题关键词的推断
        title_similarity = self._calculate_title_similarity(citing_title, cited_title)

        # 复杂推断逻辑
        if year_diff >= 10:
            if cited_citations > 1000:
                return "CLASSIC_REFERENCE"  # 引用经典文献
            else:
                return "HISTORICAL_REFERENCE"  # 历史参考

        elif year_diff >= 5:
            if technique_similarity > 0.5:
                return "BUILDS_ON"  # 基于相关工作扩展
            else:
                return "BACKGROUND_REFERENCE"  # 背景知识引用

        elif abs(year_diff) <= 2:
            if technique_similarity > 0.7:
                return "DIRECT_COMPARISON"  # 直接比较
            elif technique_similarity > 0.3:
                return "RELATED_WORK"  # 相关工作
            elif citation_ratio > 2:
                return "CHALLENGES"  # 挑战已有工作
            else:
                return "CONTEMPORARY_REFERENCE"  # 同期参考

        elif year_diff < 0:  # 引用更新的论文（少见但可能）
            return "FORWARD_REFERENCE"  # 前向引用

        else:
            # 综合判断
            if title_similarity > 0.5 or technique_similarity > 0.5:
                return "METHODOLOGICAL_REFERENCE"  # 方法学引用
            elif cited_citations > citing_citations * 5:
                return "AUTHORITATIVE_REFERENCE"  # 权威引用
            else:
                return "GENERAL_REFERENCE"  # 一般引用

    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """
        计算标题相似性

        Args:
            title1: 第一个标题
            title2: 第二个标题

        Returns:
            相似度分数 (0-1)
        """
        if not title1 or not title2:
            return 0.0

        # 简单的关键词重叠度计算
        words1 = set(title1.split())
        words2 = set(title2.split())

        # 过滤常见停用词
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'using', 'based', 'via', 'through'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _create_edge_traces(self, graph, pos) -> list:
        """
        创建不同类型的边trace

        Args:
            graph: NetworkX图对象
            pos: 节点位置字典

        Returns:
            边trace列表
        """
        # 定义不同引用类型的样式
        edge_styles = {
            'CLASSIC_REFERENCE': {'color': '#FF6B6B', 'width': 2.5, 'dash': 'solid'},      # 红色，粗线
            'BUILDS_ON': {'color': '#4ECDC4', 'width': 2.0, 'dash': 'solid'},             # 青色，中等
            'DIRECT_COMPARISON': {'color': '#45B7D1', 'width': 2.0, 'dash': 'dash'},       # 蓝色，虚线
            'METHODOLOGICAL_REFERENCE': {'color': '#96CEB4', 'width': 1.5, 'dash': 'solid'}, # 绿色
            'AUTHORITATIVE_REFERENCE': {'color': '#FFEAA7', 'width': 2.0, 'dash': 'solid'},  # 黄色
            'RELATED_WORK': {'color': '#DDA0DD', 'width': 1.5, 'dash': 'dot'},            # 紫色，点线
            'CONTEMPORARY_REFERENCE': {'color': '#FFB6C1', 'width': 1.0, 'dash': 'solid'}, # 粉色，细线
            'GENERAL_REFERENCE': {'color': '#D3D3D3', 'width': 1.0, 'dash': 'solid'},      # 灰色，默认
            'HISTORICAL_REFERENCE': {'color': '#CD853F', 'width': 1.0, 'dash': 'dot'},    # 棕色，点线
            'BACKGROUND_REFERENCE': {'color': '#C0C0C0', 'width': 0.8, 'dash': 'solid'}   # 银色，细线
        }

        # 按边类型分组
        edges_by_type = {}
        for edge in graph.edges(data=True):
            from_node, to_node, attrs = edge
            edge_type = attrs.get('edge_type', 'GENERAL_REFERENCE')

            if edge_type not in edges_by_type:
                edges_by_type[edge_type] = []
            edges_by_type[edge_type].append((from_node, to_node))

        # 为每种类型创建trace
        traces = []
        for edge_type, edges in edges_by_type.items():
            style = edge_styles.get(edge_type, edge_styles['GENERAL_REFERENCE'])

            edge_x = []
            edge_y = []

            for from_node, to_node in edges:
                if from_node in pos and to_node in pos:
                    x0, y0 = pos[from_node]
                    x1, y1 = pos[to_node]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

            if edge_x:  # 只有当有边时才创建trace
                # 定义边类型的中文名称和描述
                edge_type_names = {
                    'CLASSIC_REFERENCE': '经典引用 (10年+ 高引用)',
                    'BUILDS_ON': '技术扩展 (基于相关工作)',
                    'DIRECT_COMPARISON': '直接比较 (技术相似度高)',
                    'METHODOLOGICAL_REFERENCE': '方法学引用',
                    'AUTHORITATIVE_REFERENCE': '权威引用',
                    'RELATED_WORK': '相关工作',
                    'CONTEMPORARY_REFERENCE': '同期参考',
                    'GENERAL_REFERENCE': '一般引用',
                    'HISTORICAL_REFERENCE': '历史参考',
                    'BACKGROUND_REFERENCE': '背景引用'
                }

                display_name = edge_type_names.get(edge_type, edge_type)

                traces.append(go.Scatter(
                    x=edge_x, y=edge_y,
                    mode='lines',
                    line=dict(
                        width=style['width'],
                        color=style['color'],
                        dash=style['dash']
                    ),
                    hoverinfo='none',
                    showlegend=True,  # 启用图例
                    name=display_name,
                    legendgroup=edge_type  # 分组显示
                ))

        return traces


    # 整体页面布局
    def _generate_interactive_html_page(self, subgraph, pos) -> str:
        """
        生成完整的交互式HTML页面

        Args:
            subgraph: 网络图子图
            pos: 节点位置字典

        Returns:
            完整的HTML页面内容
        """
        # 准备节点数据
        nodes_data = []
        for node in subgraph.nodes():
            paper = self.papers.get(node, {})
            authors = paper.get('authors', [])
            first_author = authors[0] if authors else "Unknown"
            # 提取姓氏（假设是最后一个单词）
            first_author_surname = first_author.split()[-1] if first_author != "Unknown" else "Unknown"
            year = paper.get('year', 'Unknown')

            # 提取RAG分析结果
            rag_analysis = paper.get('rag_analysis', {})

            x, y = pos[node]
            nodes_data.append({
                'id': node,
                'x': x,
                'y': y,
                'title': paper.get('title', 'Unknown'),
                'authors': authors,
                'first_author': first_author,
                'first_author_surname': first_author_surname,
                'year': year,
                'cited_by_count': paper.get('cited_by_count', 0),
                'venue': paper.get('venue', ''),
                'size': self._calculate_node_size(paper.get('cited_by_count', 0), subgraph),
                'color': self._get_node_color(year),
                'label': f"{first_author_surname} ,{year}",
                # 添加RAG分析结果
                'rag_problem': rag_analysis.get('problem', ''),
                'rag_method': rag_analysis.get('method', ''),
                'rag_limitation': rag_analysis.get('limitation', ''),
                'rag_future_work': rag_analysis.get('future_work', ''),
                'analysis_method': paper.get('analysis_method', ''),
                'sections_extracted': paper.get('sections_extracted', 0)
            })

        # 准备边数据
        edges_data = []

        # Socket Matching 关系类型样式（Tech Tree Schema - 6种核心类型）
        edge_styles = {
            # === Socket Matching 核心类型（6种）===
            'Overcomes': {
                'color': '#E74C3C',      # 红色 - 主干路径（攻克/优化）
                'width': 3.0,
                'dash': 'solid',
                'description': '攻克/优化 - B解决了A的局限性（纵向深化）'
            },
            'Realizes': {
                'color': '#9B59B6',      # 紫色 - 前瞻验证（实现愿景）
                'width': 2.5,
                'dash': 'solid',
                'description': '实现愿景 - B实现了A的未来工作建议（科研传承）'
            },
            'Extends': {
                'color': '#2ECC71',      # 绿色 - 方法扩展（微创新）
                'width': 2.0,
                'dash': 'solid',
                'description': '方法扩展 - B在A的方法基础上做增量改进（微创新）'
            },
            'Alternative': {
                'color': '#E67E22',      # 橙色 - 另辟蹊径（颠覆创新）
                'width': 2.0,
                'dash': 'dot',
                'description': '另辟蹊径 - B用完全不同的范式解决问题（颠覆创新）'
            },
            'Adapts_to': {
                'color': '#3498DB',      # 蓝色 - 分支扩散（迁移/应用）
                'width': 2.0,
                'dash': 'dash',
                'description': '迁移/应用 - B将A的方法应用到新领域（横向扩散）'
            },
            'Baselines': {
                'color': '#95A5A6',      # 灰色 - 背景噪音（基线对比）
                'width': 1.0,
                'dash': 'solid',
                'description': '基线对比 - B仅把A作为对比对象（无直接继承）'
            },

            # === 传统类型（向后兼容）===
            'CLASSIC_REFERENCE': {'color': '#FF6B6B', 'width': 2.5, 'dash': 'solid', 'description': '经典引用'},
            'BUILDS_ON': {'color': '#4ECDC4', 'width': 2.0, 'dash': 'solid', 'description': '技术扩展'},
            'DIRECT_COMPARISON': {'color': '#45B7D1', 'width': 2.0, 'dash': 'dash', 'description': '直接比较'},
            'METHODOLOGICAL_REFERENCE': {'color': '#96CEB4', 'width': 1.5, 'dash': 'solid', 'description': '方法学引用'},
            'AUTHORITATIVE_REFERENCE': {'color': '#FFEAA7', 'width': 2.0, 'dash': 'solid', 'description': '权威引用'},
            'RELATED_WORK': {'color': '#DDA0DD', 'width': 1.5, 'dash': 'dot', 'description': '相关工作'},
            'CONTEMPORARY_REFERENCE': {'color': '#FFB6C1', 'width': 1.0, 'dash': 'solid', 'description': '同期引用'},
            'GENERAL_REFERENCE': {'color': '#CCCCCC', 'width': 1.0, 'dash': 'solid', 'description': '一般引用'},
            'HISTORICAL_REFERENCE': {'color': '#CD853F', 'width': 1.0, 'dash': 'dot', 'description': '历史引用'},
            'BACKGROUND_REFERENCE': {'color': '#C0C0C0', 'width': 0.8, 'dash': 'solid', 'description': '背景引用'}
        }

        for edge in subgraph.edges(data=True):
            from_node, to_node, attrs = edge
            edge_type = attrs.get('edge_type', 'Baselines')  # 默认使用Baselines
            style = edge_styles.get(edge_type, edge_styles.get('Baselines', {'color': '#CCCCCC', 'width': 1.0, 'dash': 'solid', 'description': '未知类型'}))

            if from_node in pos and to_node in pos:
                edges_data.append({
                    'from': from_node,
                    'to': to_node,
                    'type': edge_type,
                    'color': style['color'],  # 直接使用对应颜色
                    'original_color': style['color'],
                    'width': style['width'],
                    'dash': style['dash'],
                    'description': style.get('description', edge_type)
                })

        # 获取年份范围
        years = [node['year'] for node in nodes_data if isinstance(node['year'], int)]
        min_year, max_year = (min(years), max(years)) if years else (2000, 2020)

        # 生成HTML内容
        # 构建标题
        title = f"{self.topic} - 交互式论文引用知识图谱" if self.topic else "交互式论文引用知识图谱"

        html_template = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    height: 100vh;
                    overflow: hidden;
                }}
                .container {{
                    display: flex;
                    width: 100%;
                    height: 100vh;
                    background: white;
                    overflow: hidden;
                }}
                .graph-section {{
                    width: 70%;
                    display: flex;
                    flex-direction: column;
                }}
                .graph-container {{
                    flex: 1;
                    padding: 10px;
                }}
                .legend-container {{
                    padding: 15px;
                    background: #f8f9fa;
                    border-top: 1px solid #dee2e6;
                    min-height: 180px;
                    max-height: 200px;
                }}
                .details-section {{
                    width: 30%;
                    background: #f8f9fa;
                    border-left: 1px solid #dee2e6;
                    display: flex;
                    flex-direction: column;
                }}
                .details-header {{
                    padding: 15px 20px;
                    background: #6c757d;
                    color: white;
                    font-weight: bold;
                    font-size: 18px;
                }}
                .details-content {{
                    padding: 20px;
                    flex: 1;
                    overflow-y: auto;
                }}
                .paper-info {{
                    margin-bottom: 15px;
                }}
                .paper-info h3 {{
                    color: #2c3e50;
                    margin: 0 0 10px 0;
                    font-size: 16px;
                    line-height: 1.4;
                }}
                .paper-info p {{
                    margin: 5px 0;
                    color: #5a5a5a;
                    font-size: 14px;
                }}
                .legend-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 10px;
                }}
                .legend-item {{
                    display: flex;
                    align-items: center;
                    padding: 8px;
                    background: white;
                    border-radius: 5px;
                    font-size: 12px;
                }}
                .legend-color {{
                    width: 20px;
                    height: 3px;
                    margin-right: 8px;
                    border-radius: 2px;
                }}
                .stats {{
                    background: #e9ecef;
                    padding: 15px;
                    border-radius: 8px;
                    margin-bottom: 15px;
                }}
                .stat-item {{
                    display: flex;
                    justify-content: space-between;
                    margin: 5px 0;
                    font-size: 14px;
                }}
                .placeholder {{
                    text-align: center;
                    color: #6c757d;
                    font-style: italic;
                    margin-top: 50px;
                }}
                .title {{
                    text-align: center;
                    color: #2c3e50;
                    margin-bottom: 20px;
                    font-size: 24px;
                    font-weight: bold;
                }}
                /* 标签页样式 */
                .tabs {{
                    display: flex;
                    background: #dee2e6;
                    border-bottom: 2px solid #6c757d;
                }}
                .tab {{
                    flex: 1;
                    padding: 12px 10px;
                    text-align: center;
                    cursor: pointer;
                    border: none;
                    background: #dee2e6;
                    color: #495057;
                    font-size: 14px;
                    font-weight: 500;
                    transition: all 0.3s;
                }}
                .tab:hover {{
                    background: #c4c8cc;
                }}
                .tab.active {{
                    background: #6c757d;
                    color: white;
                }}
                .tab-content {{
                    display: none;
                    padding: 20px;
                    overflow-y: auto;
                    height: calc(100vh - 130px);
                }}
                .tab-content.active {{
                    display: block;
                }}
                .epoch-card {{
                    background: white;
                    border-left: 4px solid #3498DB;
                    padding: 15px;
                    margin-bottom: 15px;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .epoch-card h4 {{
                    margin: 0 0 10px 0;
                    color: #2c3e50;
                    font-size: 15px;
                }}
                .epoch-card p {{
                    margin: 5px 0;
                    font-size: 13px;
                    color: #555;
                }}
                .idea-card {{
                    background: white;
                    border-left: 4px solid #2ECC71;
                    padding: 15px;
                    margin-bottom: 15px;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .idea-card h4 {{
                    margin: 0 0 10px 0;
                    color: #2c3e50;
                    font-size: 15px;
                }}
                .idea-card .status-badge {{
                    display: inline-block;
                    padding: 3px 8px;
                    border-radius: 3px;
                    font-size: 11px;
                    font-weight: bold;
                    margin-left: 8px;
                }}
                .status-success {{
                    background: #d4edda;
                    color: #155724;
                }}
                .status-incompatible {{
                    background: #f8d7da;
                    color: #721c24;
                }}
                .pivot-paper {{
                    background: #fff3cd;
                    padding: 10px;
                    margin: 8px 0;
                    border-radius: 4px;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <!-- 左侧图谱部分 (70%) -->
                <div class="graph-section">
                    <div class="title">{title}</div>
                    <div class="graph-container">
                        <div id="graph" style="width:100%; height:100%;"></div>
                    </div>
                    <div class="legend-container">
                        <h4 style="margin-top:0; color:#2c3e50;">🔌 Socket Matching 引用关系类型（6种核心类型）</h4>
                        <div class="legend-grid">
                            <!-- Socket Matching 核心类型（6种）-->
                            <div class="legend-item" style="border-left: 3px solid #E74C3C;">
                                <div class="legend-color" style="background-color:#E74C3C; height:3px;"></div>
                                <span><strong>Overcomes</strong> - 攻克/优化（纵向深化）</span>
                            </div>
                            <div class="legend-item" style="border-left: 3px solid #9B59B6;">
                                <div class="legend-color" style="background-color:#9B59B6; height:3px;"></div>
                                <span><strong>Realizes</strong> - 实现愿景（科研传承）</span>
                            </div>
                            <div class="legend-item" style="border-left: 3px solid #2ECC71;">
                                <div class="legend-color" style="background-color:#2ECC71; height:2px;"></div>
                                <span><strong>Extends</strong> - 方法扩展（微创新）</span>
                            </div>
                            <div class="legend-item" style="border-left: 3px solid #E67E22;">
                                <div class="legend-color" style="background-color:#E67E22; border: 2px dotted #E67E22; height:1px;"></div>
                                <span><strong>Alternative</strong> - 另辟蹊径（颠覆创新）</span>
                            </div>
                            <div class="legend-item" style="border-left: 3px solid #3498DB;">
                                <div class="legend-color" style="background-color:#3498DB; border: 2px dashed #3498DB; height:1px;"></div>
                                <span><strong>Adapts_to</strong> - 迁移/应用（横向扩散）</span>
                            </div>
                            <div class="legend-item" style="border-left: 3px solid #95A5A6;">
                                <div class="legend-color" style="background-color:#95A5A6; height:1px;"></div>
                                <span><strong>Baselines</strong> - 基线对比（背景噪音）</span>
                            </div>
                        </div>
                        <p style="margin-top:10px; font-size:11px; color:#666;">
                            💡 <strong>逻辑对接矩阵 (4个Match → 6种类型)</strong>: Match1(Limitation→Problem) → Overcomes | Match2(FutureWork→Problem) → Realizes | Match3(Method→Method) → Extends/Alternative | Match4(Problem跨域) → Adapts_to | 无匹配 → Baselines
                        </p>
                    </div>
                </div>

                <!-- 右侧详情部分 (30%) -->
                <div class="details-section">
                    <!-- 标签页导航 -->
                    <div class="tabs">
                        <div class="tab active" onclick="switchTab(event, 'paper-tab')">📄 论文详情</div>
                        <div class="tab" onclick="switchTab(event, 'survey-tab')">📝 深度综述</div>
                        <div class="tab" onclick="switchTab(event, 'ideas-tab')">💡 科研创意</div>
                    </div>

                    <!-- 论文详情标签页 -->
                    <div id="paper-tab" class="tab-content active">
                        <div class="stats">
                            <h4 style="margin-top:0;">图谱统计</h4>
                            <div class="stat-item">
                                <span>论文总数:</span>
                                <span>{len(nodes_data)}</span>
                            </div>
                            <div class="stat-item">
                                <span>引用关系:</span>
                                <span>{len(edges_data)}</span>
                            </div>
                            <div class="stat-item">
                                <span>时间跨度:</span>
                                <span>{min_year} - {max_year}</span>
                            </div>
                        </div>
                        <div class="placeholder">
                            👆 点击图谱中的节点查看论文详细信息
                        </div>
                    </div>

                    <!-- 深度综述标签页 -->
                    <div id="survey-tab" class="tab-content"></div>

                    <!-- 科研创意标签页 -->
                    <div id="ideas-tab" class="tab-content"></div>
                </div>
            </div>

            <script>
                // ========== 数据初始化 ==========
                const nodesData = {json.dumps(nodes_data, ensure_ascii=False, indent=2)};
                const edgesData = {json.dumps(edges_data, ensure_ascii=False, indent=2)};

                // Deep Survey数据
                const deepSurveyData = {json.dumps(self.deep_survey_report, ensure_ascii=False, indent=2, cls=DateTimeEncoder)};

                // Research Ideas数据
                const researchIdeasData = {json.dumps(self.research_ideas, ensure_ascii=False, indent=2, cls=DateTimeEncoder)};

                // 按类型分组边数据
                const edgesByType = {{}};
                edgesData.forEach(edge => {{
                    if (!edgesByType[edge.type]) {{
                        edgesByType[edge.type] = [];
                    }}
                    edgesByType[edge.type].push(edge);
                }});

                // 创建节点轨迹
                const nodeTrace = {{
                    x: nodesData.map(n => n.x),
                    y: nodesData.map(n => n.y),
                    mode: 'markers+text',
                    marker: {{
                        size: nodesData.map(n => n.size),
                        color: nodesData.map(n => n.color),
                        line: {{ width: 2, color: 'white' }},
                        colorscale: 'Viridis'
                    }},
                    text: nodesData.map(n => n.label),
                    textposition: 'middle center',
                    textfont: {{ size: 10, color: 'black' }},
                    customdata: nodesData,
                    hovertemplate: '<b>%{{customdata.title}}</b><extra></extra>',
                    type: 'scatter',
                    name: '论文节点'
                }};

                // 图表布局配置
                const layout = {{
                    title: '',
                    showlegend: false,
                    hovermode: 'closest',
                    margin: {{ l: 0, r: 0, b: 40, t: 0 }},
                    xaxis: {{
                        title: '发表年份',
                        showgrid: true,
                        gridcolor: 'lightgray',
                        range: [{min_year - 1}, {max_year + 1}]
                    }},
                    yaxis: {{
                        title: '论文分布',
                        showgrid: true,
                        gridcolor: 'lightgray',
                        showticklabels: false
                    }},
                    plot_bgcolor: 'white',
                    paper_bgcolor: 'white'
                }};

                // ========== 工具函数 ==========
                // 创建边轨迹（通用函数，消除重复逻辑）
                function createEdgeTraces(styleMap) {{
                    const traces = [];
                    Object.keys(edgesByType).forEach(type => {{
                        const edges = edgesByType[type];
                        const style = styleMap.get(type) || {{
                            color: edges[0].color,
                            width: edges[0].width,
                            dash: 'solid'
                        }};

                        const edgeX = [];
                        const edgeY = [];

                        edges.forEach(edge => {{
                            const fromNode = nodesData.find(n => n.id === edge.from);
                            const toNode = nodesData.find(n => n.id === edge.to);
                            if (fromNode && toNode) {{
                                edgeX.push(fromNode.x, toNode.x, null);
                                edgeY.push(fromNode.y, toNode.y, null);
                            }}
                        }});

                        if (edgeX.length > 0) {{
                            traces.push({{
                                x: edgeX,
                                y: edgeY,
                                mode: 'lines',
                                line: {{
                                    width: style.width,
                                    color: style.color,
                                    dash: style.dash
                                }},
                                hoverinfo: 'none',
                                showlegend: false,
                                type: 'scatter'
                            }});
                        }}
                    }});
                    return traces;
                }}

                // 更新图表（通用函数）
                function updateGraph(edgeTraces, nodeColors, nodeLineStyle = null) {{
                    const nodeUpdate = {{
                        ...nodeTrace,
                        marker: {{
                            ...nodeTrace.marker,
                            color: nodeColors,
                            line: nodeLineStyle || {{ width: 2, color: 'white' }}
                        }}
                    }};

                    Plotly.react('graph', [...edgeTraces, nodeUpdate], layout);
                }}

                // ========== 初始化图表 ==========
                // 创建初始边样式（使用原始定义的颜色）
                const initialEdgeStyle = new Map();
                Object.keys(edgesByType).forEach(type => {{
                    const firstEdge = edgesByType[type][0];
                    initialEdgeStyle.set(type, {{
                        color: firstEdge.original_color,
                        width: firstEdge.width,
                        dash: firstEdge.dash
                    }});
                }});

                const initialEdgeTraces = createEdgeTraces(initialEdgeStyle);
                updateGraph(initialEdgeTraces, nodesData.map(n => n.color));

                // ========== 事件处理器 ==========
                document.getElementById('graph').on('plotly_click', function(data) {{
                    if (data.points?.[0]?.customdata) {{
                        const node = data.points[0].customdata;
                        const nodeIndex = data.points[0].pointIndex;
                        showPaperDetails(node);
                        highlightClickedNodeAndEdges(nodeIndex, node);
                    }}
                }});


                // ========== 功能函数 ==========
                // 标签页切换函数
                function switchTab(event, tabId) {{
                    // 移除所有active类
                    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

                    // 添加active类到当前标签
                    event.target.classList.add('active');
                    document.getElementById(tabId).classList.add('active');

                    // 切换到"论文详情"或"科研创意"标签时，重置图谱高亮
                    if (tabId === 'paper-tab' || tabId === 'ideas-tab') {{
                        resetGraphHighlight();
                        console.log(`切换到 ${{tabId}}，已重置图谱高亮`);
                    }}

                    // 根据标签页ID加载相应内容
                    if (tabId === 'survey-tab') {{
                        renderDeepSurvey();
                    }} else if (tabId === 'ideas-tab') {{
                        renderResearchIdeas();
                    }}
                }}

                // 渲染深度综述 (新版数据结构)
                function renderDeepSurvey() {{
                    const surveyTab = document.getElementById('survey-tab');

                    if (!deepSurveyData || Object.keys(deepSurveyData).length === 0) {{
                        surveyTab.innerHTML = '<div class="placeholder">暂无深度综述数据</div>';
                        return;
                    }}

                    let html = '<div style="padding:20px;">';

                    // 摘要信息 (新结构)
                    if (deepSurveyData.summary) {{
                        html += `
                            <div class="stats">
                                <h4 style="margin-top:0;">📊 综述摘要</h4>
                                <div class="stat-item"><span>原始论文:</span><span>${{deepSurveyData.summary.original_papers || 0}} 篇</span></div>
                                <div class="stat-item"><span>筛选后论文:</span><span>${{deepSurveyData.summary.pruned_papers || 0}} 篇</span></div>
                                <div class="stat-item"><span>演化故事线:</span><span>${{deepSurveyData.summary.total_threads || 0}} 条</span></div>
                            </div>
                        `;
                    }}

                    // 剪枝统计信息
                    if (deepSurveyData.pruning_stats) {{
                        const stats = deepSurveyData.pruning_stats;
                        const retentionRate = (stats.retention_rate * 100).toFixed(1);
                        html += `
                            <div class="stats" style="margin-top:15px; background:#fff3cd;">
                                <h4 style="margin-top:0;">✂️ 图谱剪枝统计</h4>
                                <div class="stat-item"><span>保留率:</span><span>${{retentionRate}}%</span></div>
                                <div class="stat-item"><span>种子论文:</span><span>${{stats.seed_papers || 0}} 篇</span></div>
                                <div class="stat-item"><span>强关系边:</span><span>${{stats.strong_edges || 0}} 条</span></div>
                                <div class="stat-item"><span>剔除弱关系边:</span><span>${{stats.weak_edges_removed || 0}} 条</span></div>
                            </div>
                        `;
                    }}

                    // 演化路径 (Threads)
                    const threads = deepSurveyData.survey_report?.threads || deepSurveyData.evolutionary_paths || [];
                    if (threads.length > 0) {{
                        html += `
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:20px;">
                                <h3 style="color:#2c3e50; margin:0;">🧵 关键演化故事线</h3>
                                <button id="resetGraphBtn" onclick="resetGraphHighlight()"
                                    style="padding:5px 12px; background:#6c757d; color:white; border:none; border-radius:4px; cursor:pointer; font-size:12px;">
                                    🔄 重置图谱
                                </button>
                            </div>
                        `;
                        threads.forEach((thread, index) => {{
                            const threadTitle = thread.title || thread.thread_name || `Thread ${{index + 1}}`;
                            const patternType = thread.pattern_type || thread.thread_type || '未知类型';
                            const paperCount = thread.papers ? thread.papers.length : 0;
                            const narrative = thread.narrative || '暂无叙事文本';

                            // 定义丰富的颜色调色板（按故事线索引分配）
                            const colorPalette = [
                                '#E74C3C',  // 红色 - Thread 0
                                '#3498DB',  // 蓝色 - Thread 1
                                '#2ECC71',  // 绿色 - Thread 2
                                '#F39C12',  // 橙色 - Thread 3
                                '#9B59B6',  // 紫色 - Thread 4
                                '#1ABC9C',  // 青色 - Thread 5
                                '#E67E22',  // 深橙色 - Thread 6
                                '#95A5A6',  // 灰色 - Thread 7
                                '#34495E',  // 深蓝灰 - Thread 8
                                '#16A085'   // 深青色 - Thread 9
                            ];

                            // 根据故事线索引分配颜色（保证每条故事线颜色唯一）
                            let borderColor = colorPalette[index % colorPalette.length];
                            let highlightColor = borderColor;

                            // 收集该故事线的所有论文ID
                            const paperIds = thread.papers ? thread.papers.map(p => p.paper_id) : [];

                            html += `
                                <div class="epoch-card" style="border-left-color:${{borderColor}}; cursor:pointer; transition:all 0.3s;"
                                     onclick="highlightThread(${{index}}, '${{highlightColor}}')"
                                     onmouseover="this.style.backgroundColor='#f8f9fa'"
                                     onmouseout="this.style.backgroundColor='white'">
                                    <h4>
                                        ${{threadTitle}}
                                        <span style="float:right; font-size:12px; color:#666; font-weight:normal;">
                                            ${{patternType}}
                                        </span>
                                    </h4>
                                    <p><strong>📚 论文数量:</strong> ${{paperCount}} 篇</p>
                                    ${{thread.total_citations ? `<p><strong>📊 总引用数:</strong> ${{thread.total_citations}}</p>` : ''}}
                                    <p style="font-size:11px; color:#666; margin-top:8px;">
                                        💡 <em>点击此卡片可在左侧图谱中高亮显示该故事线的所有论文</em>
                                    </p>

                                    <details style="margin-top:10px;">
                                        <summary style="cursor:pointer; color:#3498DB; font-weight:bold;">📖 查看演化叙事</summary>
                                        <div style="margin-top:10px; padding:10px; background:#f8f9fa; border-radius:5px; line-height:1.6; white-space:pre-wrap;">
                                            ${{narrative}}
                                        </div>
                                    </details>

                                    ${{thread.papers && thread.papers.length > 0 ? `
                                        <details style="margin-top:10px;">
                                            <summary style="cursor:pointer; color:#2ECC71; font-weight:bold;">📄 查看论文列表</summary>
                                            <ul style="margin-top:10px; padding-left:20px;">
                                                ${{thread.papers.map((p, pIndex) => `
                                                    <li style="margin:5px 0; cursor:pointer; color:#2980b9; transition:color 0.2s;"
                                                        onclick="event.stopPropagation(); showPaperFromThread('${{p.paper_id}}');"
                                                        onmouseover="this.style.color='#3498db'; this.style.textDecoration='underline';"
                                                        onmouseout="this.style.color='#2980b9'; this.style.textDecoration='none';"
                                                        title="点击查看该论文详情并在图谱中高亮">
                                                        <strong>${{p.title}}</strong>
                                                        (${{p.year || 'N/A'}}, 引用: ${{p.cited_by_count || 0}})
                                                    </li>
                                                `).join('')}}
                                            </ul>
                                        </details>
                                    ` : ''}}
                                </div>
                            `;
                        }});
                    }}

                    // 综述报告摘要
                    if (deepSurveyData.survey_report?.abstract) {{
                        html += `
                            <div style="margin-top:20px; padding:15px; background:#e8f4f8; border-left:4px solid #3498DB; border-radius:5px;">
                                <h4 style="margin:0 0 10px 0; color:#2c3e50;">📝 综述摘要</h4>
                                <p style="line-height:1.6; color:#333; margin:0;">${{deepSurveyData.survey_report.abstract}}</p>
                            </div>
                        `;
                    }}

                    html += '</div>';
                    surveyTab.innerHTML = html;
                }}

                // 渲染科研创意
                function renderResearchIdeas() {{
                    const ideasTab = document.getElementById('ideas-tab');

                    if (!researchIdeasData || Object.keys(researchIdeasData).length === 0) {{
                        ideasTab.innerHTML = '<div class="placeholder">暂无科研创意数据</div>';
                        return;
                    }}

                    let html = '<div style="padding:20px;">';

                    // 统计信息
                    html += `
                        <div class="stats">
                            <h4 style="margin-top:0;">💡 创意生成统计</h4>
                            <div class="stat-item"><span>总创意数:</span><span>${{researchIdeasData.total_ideas || 0}}</span></div>
                            <div class="stat-item"><span>可行创意:</span><span>${{researchIdeasData.successful_ideas || 0}}</span></div>
                            ${{researchIdeasData.pools ? `
                                <div class="stat-item"><span>未解决限制:</span><span>${{researchIdeasData.pools.unsolved_limitations || 0}}</span></div>
                                <div class="stat-item"><span>候选方法:</span><span>${{researchIdeasData.pools.candidate_methods || 0}}</span></div>
                            ` : ''}}
                        </div>
                    `;

                    // 创意列表
                    if (researchIdeasData.ideas && researchIdeasData.ideas.length > 0) {{
                        html += '<h3 style="color:#2c3e50; margin-top:20px;">💡 研究创意列表</h3>';
                        researchIdeasData.ideas.forEach((idea, index) => {{
                            const statusClass = idea.status === 'SUCCESS' ? 'status-success' : 'status-incompatible';
                            const statusText = idea.status === 'SUCCESS' ? '✓ 可行' : '✗ 不兼容';

                            html += `
                                <div class="idea-card">
                                    <h4>
                                        创意 ${{index + 1}}: ${{idea.title || '未命名创意'}}
                                        <span class="status-badge ${{statusClass}}">${{statusText}}</span>
                                    </h4>
                                    ${{idea.abstract ? `
                                        <p style="margin:10px 0; line-height:1.6; color:#444;">
                                            <strong>摘要:</strong> ${{idea.abstract}}
                                        </p>
                                    ` : ''}}
                                    ${{idea.modification ? `
                                        <p style="margin:8px 0; padding:10px; background:#f8f9fa; border-radius:4px;">
                                            <strong>🔧 关键创新:</strong> ${{idea.modification}}
                                        </p>
                                    ` : ''}}
                                    ${{idea.reasoning ? `
                                        <details style="margin-top:10px;">
                                            <summary style="cursor:pointer; color:#3498DB;"><strong>查看推理过程</strong></summary>
                                            <p style="margin-top:8px; font-size:12px; color:#666; white-space:pre-wrap;">${{idea.reasoning}}</p>
                                        </details>
                                    ` : ''}}
                                </div>
                            `;
                        }});
                    }}

                    html += '</div>';
                    ideasTab.innerHTML = html;
                }}

                function showPaperDetails(node) {{
                    const authorsText = node.authors.slice(0, 5).join(', ') +
                                      (node.authors.length > 5 ? ' 等' : '');

                    // 构建RAG分析部分的HTML
                    let ragAnalysisHTML = '';
                    if (node.rag_problem || node.rag_method || node.rag_limitation || node.rag_future_work) {{
                        ragAnalysisHTML = `
                            <div class="paper-info" style="background:#e8f4f8; padding:15px; border-radius:8px; margin-top:15px;">
                                <h4 style="margin:0 0 10px 0; color:#1a73e8; font-size:15px;">🧠 多智能体系统深度分析</h4>
                                ${{node.analysis_method ? `<p style="font-size:12px; color:#666; margin-bottom:10px;"><strong>分析方法:</strong> ${{node.analysis_method.toUpperCase()}}</p>` : ''}}
                                ${{node.sections_extracted ? `<p style="font-size:12px; color:#666; margin-bottom:10px;"><strong>提取章节:</strong> ${{node.sections_extracted}} 个</p>` : ''}}
                            </div>
                            ${{node.rag_problem ? `
                            <div class="paper-info" style="border-left:3px solid #FF6B6B; padding-left:10px;">
                                <h4 style="margin:0 0 8px 0; color:#FF6B6B; font-size:14px;">📋 研究问题 (Problem)</h4>
                                <p style="font-size:13px; line-height:1.6; color:#333;">${{node.rag_problem}}</p>
                            </div>
                            ` : ''}}
                            ${{node.rag_method ? `
                            <div class="paper-info" style="border-left:3px solid #4ECDC4; padding-left:10px;">
                                <h4 style="margin:0 0 8px 0; color:#4ECDC4; font-size:14px;">💡 核心方法 (Method)</h4>
                                <p style="font-size:13px; line-height:1.6; color:#333;">${{node.rag_method}}</p>
                            </div>
                            ` : ''}}
                            ${{node.rag_limitation ? `
                            <div class="paper-info" style="border-left:3px solid #FFA500; padding-left:10px;">
                                <h4 style="margin:0 0 8px 0; color:#FFA500; font-size:14px;">⚠️ 局限性 (Limitation)</h4>
                                <p style="font-size:13px; line-height:1.6; color:#333;">${{node.rag_limitation}}</p>
                            </div>
                            ` : ''}}
                            ${{node.rag_future_work ? `
                            <div class="paper-info" style="border-left:3px solid #9B59B6; padding-left:10px;">
                                <h4 style="margin:0 0 8px 0; color:#9B59B6; font-size:14px;">🔮 未来工作 (Future Work)</h4>
                                <p style="font-size:13px; line-height:1.6; color:#333;">${{node.rag_future_work}}</p>
                            </div>
                            ` : ''}}
                        `;
                    }}

                    document.getElementById('paper-tab').innerHTML = `
                        <div class="stats">
                            <h4 style="margin-top:0;">图谱统计</h4>
                            <div class="stat-item"><span>论文总数:</span><span>{len(nodes_data)}</span></div>
                            <div class="stat-item"><span>引用关系:</span><span>{len(edges_data)}</span></div>
                            <div class="stat-item"><span>时间跨度:</span><span>{min_year} - {max_year}</span></div>
                        </div>
                        <div class="paper-info">
                            <h3>${{node.title}}</h3>
                            <p><strong>作者:</strong> ${{authorsText}}</p>
                            <p><strong>年份:</strong> ${{node.year}}</p>
                            <p><strong>引用数:</strong> ${{node.cited_by_count}}</p>
                            <p><strong>期刊/会议:</strong> ${{node.venue || '未知'}}</p>
                            <p><strong>论文ID:</strong> ${{node.id}}</p>
                        </div>
                        ${{ragAnalysisHTML}}
                    `;
                }}

                function highlightClickedNodeAndEdges(nodeIndex, node) {{
                    // 只改变节点颜色 - 被点击的节点高亮
                    const nodeColors = nodesData.map((n, i) =>
                        i === nodeIndex ? '#FF4444' : n.color);

                    // 边保持原始样式不变
                    updateGraph(initialEdgeTraces, nodeColors);
                }}

                // ========== 演化故事线高亮功能 ==========
                function highlightThread(threadIndex, highlightColor) {{
                    // 获取线程数据
                    const threads = deepSurveyData.survey_report?.threads || deepSurveyData.evolutionary_paths || [];
                    if (threadIndex >= threads.length) return;

                    const thread = threads[threadIndex];
                    const threadPaperIds = new Set(thread.papers?.map(p => p.paper_id) || []);

                    console.log(`高亮 Thread ${{threadIndex}}: ${{thread.title}}, 包含 ${{threadPaperIds.size}} 篇论文`);

                    // 更新节点颜色和大小
                    const newColors = [];
                    const newSizes = [];
                    const newLineStyles = [];

                    nodesData.forEach((node, index) => {{
                        if (threadPaperIds.has(node.id)) {{
                            // 高亮显示：保持节点原本颜色，放大1.5倍
                            newColors.push(node.color);
                            newSizes.push(node.size * 1.5);
                            newLineStyles.push({{ width: 3, color: node.color }});
                        }} else {{
                            // 其他节点：变灰，缩小到0.5倍
                            newColors.push('#D3D3D3');
                            newSizes.push(node.size * 0.5);
                            newLineStyles.push({{ width: 1, color: '#CCCCCC' }});
                        }}
                    }});

                    // 更新图谱
                    const highlightedNodeTrace = {{
                        ...nodeTrace,
                        marker: {{
                            ...nodeTrace.marker,
                            size: newSizes,
                            color: newColors,
                            line: newLineStyles
                        }}
                    }};

                    // 边也调整透明度（高亮故事线内的边）
                    const highlightedEdgeTraces = createEdgeTracesWithHighlight(threadPaperIds, highlightColor);

                    Plotly.react('graph', [...highlightedEdgeTraces, highlightedNodeTrace], layout);

                    // 滚动图谱到高亮区域
                    document.getElementById('graph').scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                }}

                function createEdgeTracesWithHighlight(highlightedNodeIds, highlightColor) {{
                    const traces = [];
                    Object.keys(edgesByType).forEach(type => {{
                        const edges = edgesByType[type];
                        const style = initialEdgeStyle.get(type) || {{
                            color: edges[0].color,
                            width: edges[0].width,
                            dash: 'solid'
                        }};

                        // 分离高亮边和非高亮边
                        const highlightedEdgeX = [];
                        const highlightedEdgeY = [];
                        const dimmedEdgeX = [];
                        const dimmedEdgeY = [];

                        edges.forEach(edge => {{
                            const fromNode = nodesData.find(n => n.id === edge.from);
                            const toNode = nodesData.find(n => n.id === edge.to);
                            if (fromNode && toNode) {{
                                // 检查是否是高亮故事线的边
                                const isHighlighted = highlightedNodeIds.has(edge.from) && highlightedNodeIds.has(edge.to);

                                if (isHighlighted) {{
                                    highlightedEdgeX.push(fromNode.x, toNode.x, null);
                                    highlightedEdgeY.push(fromNode.y, toNode.y, null);
                                }} else {{
                                    dimmedEdgeX.push(fromNode.x, toNode.x, null);
                                    dimmedEdgeY.push(fromNode.y, toNode.y, null);
                                }}
                            }}
                        }});

                        // 添加高亮边trace（保持原始颜色，只加粗）
                        if (highlightedEdgeX.length > 0) {{
                            traces.push({{
                                x: highlightedEdgeX,
                                y: highlightedEdgeY,
                                mode: 'lines',
                                line: {{
                                    width: style.width * 1.8,
                                    color: style.color,  // 使用原始颜色，不改变
                                    dash: style.dash
                                }},
                                opacity: 1.0,
                                hoverinfo: 'none',
                                showlegend: false,
                                type: 'scatter'
                            }});
                        }}

                        // 添加变灰边trace
                        if (dimmedEdgeX.length > 0) {{
                            traces.push({{
                                x: dimmedEdgeX,
                                y: dimmedEdgeY,
                                mode: 'lines',
                                line: {{
                                    width: style.width * 0.4,
                                    color: '#E0E0E0',
                                    dash: style.dash
                                }},
                                opacity: 0.2,
                                hoverinfo: 'none',
                                showlegend: false,
                                type: 'scatter'
                            }});
                        }}
                    }});
                    return traces;
                }}

                function resetGraphHighlight() {{
                    console.log('重置图谱高亮');
                    // 恢复原始颜色和大小
                    updateGraph(initialEdgeTraces, nodesData.map(n => n.color));
                }}

                // 从深度综述的论文列表中点击论文，显示详情并高亮
                function showPaperFromThread(paperId) {{
                    console.log('从深度综述点击论文:', paperId);

                    // 查找对应的节点索引
                    const nodeIndex = nodesData.findIndex(n => n.id === paperId);

                    if (nodeIndex === -1) {{
                        console.warn('未在图谱中找到论文:', paperId);
                        alert('该论文不在当前显示的图谱节点中');
                        return;
                    }}

                    const node = nodesData[nodeIndex];

                    // 切换到论文详情标签页
                    const paperTab = document.querySelector('.tab[onclick*="paper-tab"]');
                    if (paperTab) {{
                        paperTab.click();
                    }}

                    // 显示论文详情
                    showPaperDetails(node);

                    // 查找该论文所属的演化故事线
                    const threads = deepSurveyData.survey_report?.threads || deepSurveyData.evolutionary_paths || [];
                    let threadIndex = -1;

                    for (let i = 0; i < threads.length; i++) {{
                        const thread = threads[i];
                        const paperIds = thread.papers?.map(p => p.paper_id) || [];
                        if (paperIds.includes(paperId)) {{
                            threadIndex = i;
                            break;
                        }}
                    }}

                    // 高亮整个故事线（节点保持原色）
                    if (threadIndex !== -1) {{
                        console.log(`该论文属于故事线 ${{threadIndex}}，将高亮整个故事线`);
                        highlightThread(threadIndex, null);  // 传null因为不再需要highlightColor
                    }} else {{
                        // 如果没有找到故事线，回退到单节点高亮
                        console.log('该论文不属于任何故事线，只高亮单个节点');
                        highlightClickedNodeAndEdges(nodeIndex, node);
                    }}

                    console.log('已显示论文详情并高亮节点:', node.title);
                }}


                function updateHoverPosition(event) {{
                    const hoverDiv = document.getElementById('hoverTitle');
                    if (hoverDiv) {{
                        hoverDiv.style.left = (event.clientX + 10) + 'px';
                        hoverDiv.style.top = (event.clientY - 30) + 'px';
                    }}
                }}
            </script>
            
        </body>
        </html>
        """

        return html_template

    def _create_time_based_layout(self, graph) -> Dict:
        """
        创建基于时间的布局，横坐标为时间轴

        Args:
            graph: NetworkX图对象

        Returns:
            节点位置字典 {node_id: (x, y)}
        """
        pos = {}

        # 获取所有论文的年份信息
        papers_with_years = []
        for node in graph.nodes():
            paper = self.papers.get(node, {})
            year = paper.get('year', 2000)  # 默认年份2000
            if not isinstance(year, int) or year < 1900 or year > 2030:
                year = 2000
            papers_with_years.append((node, year))

        # 如果没有论文，返回空布局
        if not papers_with_years:
            return pos

        # 按年份排序
        papers_with_years.sort(key=lambda x: x[1])

        # 获取年份范围
        years = [year for _, year in papers_with_years]
        min_year = min(years)
        max_year = max(years)

        # 按年份分组，计算每年的论文数量
        year_groups = {}
        for node, year in papers_with_years:
            if year not in year_groups:
                year_groups[year] = []
            year_groups[year].append(node)

        # 为每个节点分配位置，确保不重叠
        for year, nodes in year_groups.items():
            # X坐标：直接使用年份作为坐标
            x = year

            # 计算Y坐标分布，确保节点充分分散
            num_papers = len(nodes)
            if num_papers == 1:
                y_positions = [0]
            else:
                # 增大Y轴分布范围以避免重叠
                y_range = max(8, num_papers * 2.5)  # 增大最小范围和间距
                y_positions = np.linspace(-y_range/2, y_range/2, num_papers)

            # 按引用数排序，影响力大的论文放中间
            nodes_with_citations = []
            for node in nodes:
                paper = self.papers.get(node, {})
                cited_count = paper.get('cited_by_count', 0)
                nodes_with_citations.append((node, cited_count))

            # 按引用数排序
            nodes_with_citations.sort(key=lambda x: x[1], reverse=True)

            # 分配Y坐标：影响力大的在中间，其他分散到两边
            for i, (node, cited_count) in enumerate(nodes_with_citations):
                y = y_positions[i]

                # 增大随机偏移避免完全重叠
                y += np.random.uniform(-0.5, 0.5)

                # 增大X坐标偏移，避免水平重叠
                x_offset = np.random.uniform(-0.6, 0.6)

                pos[node] = (x + x_offset, y)

        return pos


    # 主函数
    def build_citation_network(self, papers: List[Dict], citation_data: List) -> None:
        """
        构建完整的引用网络

        Args:
            papers: 论文列表
            citation_data: 引用关系列表
                - 支持二元组: [(citing_id, cited_id), ...]
                - 支持三元组: [(citing_id, cited_id, edge_type), ...]
        """
        logger.info(f"构建引用网络: {len(papers)} 篇论文, {len(citation_data)} 个引用关系")

        # 添加所有论文节点
        for paper in papers:
            self.add_paper_node(paper)

        # 添加引用关系
        for edge_data in citation_data:
            if len(edge_data) == 3:
                # 三元组：(citing_id, cited_id, edge_type)
                citing_id, cited_id, edge_type = edge_data
            elif len(edge_data) == 2:
                # 二元组：(citing_id, cited_id)，需要推断类型
                citing_id, cited_id = edge_data
                if citing_id in self.papers and cited_id in self.papers:
                    edge_type = self._infer_edge_type(citing_id, cited_id)
                else:
                    edge_type = "CITES"
            else:
                logger.warning(f"无效的引用数据格式: {edge_data}")
                continue

            if citing_id in self.papers and cited_id in self.papers:
                self.add_citation_edge(citing_id, cited_id, edge_type)

        logger.info(f"图谱构建完成: {self.graph.number_of_nodes()} 个节点, {self.graph.number_of_edges()} 条边")

    def visualize_graph(self, output_path: str = "./output/graph_visualization.html",
                       max_nodes: int = 50,
                       deep_survey_report: Dict = None,
                       research_ideas: Dict = None) -> str:
        """
        可视化知识图谱 - 新的交互式布局设计

        Args:
            output_path: 输出文件路径
            max_nodes: 最大节点数（避免图太复杂）
            deep_survey_report: 深度综述报告（阶段6结果）
            research_ideas: 科研创意报告（阶段7结果）

        Returns:
            生成的可视化文件路径
        """
        # 保存报告到实例变量
        if deep_survey_report:
            self.deep_survey_report = deep_survey_report
        if research_ideas:
            self.research_ideas = research_ideas

        logger.info(f"生成交互式知识图谱可视化 (最多{max_nodes}个节点)...")

        # 如果节点太多，选择最重要的节点
        if self.graph.number_of_nodes() > max_nodes:
            node_importance = {}
            for node in self.graph.nodes():
                paper = self.papers.get(node, {})
                cited_count = paper.get('cited_by_count', 0)
                in_degree = self.graph.in_degree(node)
                node_importance[node] = cited_count + in_degree * 10

            top_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            selected_nodes = [node for node, _ in top_nodes]
            subgraph = self.graph.subgraph(selected_nodes)
        else:
            subgraph = self.graph

        # 创建时间轴布局
        pos = self._create_time_based_layout(subgraph)

        # 生成带有交互功能的HTML文件
        html_content = self._generate_interactive_html_page(subgraph, pos)

        # 保存文件
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"交互式可视化文件已保存: {output_path}")
        return str(output_path)


    # 图分析
    def compute_metrics(self) -> Dict:
        """
        计算图谱指标

        Returns:
            包含各种图谱指标的字典
        """
        logger.info("计算图谱指标...")

        metrics = {}

        if self.graph.number_of_nodes() > 0:
            # 基本统计
            metrics['total_nodes'] = self.graph.number_of_nodes()
            metrics['total_edges'] = self.graph.number_of_edges()
            metrics['density'] = nx.density(self.graph)

            # PageRank (论文重要性)
            try:
                pagerank = nx.pagerank(self.graph)
                metrics['top_papers_by_pagerank'] = sorted(
                    pagerank.items(), key=lambda x: x[1], reverse=True
                )[:10]
            except:
                metrics['top_papers_by_pagerank'] = []

            # 度中心性
            try:
                in_degree = dict(self.graph.in_degree())
                out_degree = dict(self.graph.out_degree())

                metrics['most_cited_papers'] = sorted(
                    in_degree.items(), key=lambda x: x[1], reverse=True
                )[:10]

                metrics['most_citing_papers'] = sorted(
                    out_degree.items(), key=lambda x: x[1], reverse=True
                )[:10]
            except:
                metrics['most_cited_papers'] = []
                metrics['most_citing_papers'] = []

            # 连通分量
            try:
                if nx.is_weakly_connected(self.graph):
                    metrics['is_connected'] = True
                    metrics['connected_components'] = 1
                else:
                    components = list(nx.weakly_connected_components(self.graph))
                    metrics['is_connected'] = False
                    metrics['connected_components'] = len(components)
                    metrics['largest_component_size'] = max(len(c) for c in components)
            except:
                metrics['is_connected'] = False
                metrics['connected_components'] = 0

        logger.info("图谱指标计算完成")
        return metrics

    def find_research_clusters(self, min_cluster_size: int = 3) -> List[List[str]]:
        """
        发现研究聚类

        Args:
            min_cluster_size: 最小聚类大小

        Returns:
            聚类列表
        """
        logger.info("寻找研究聚类...")

        try:
            # 转换为无向图进行聚类
            undirected_graph = self.graph.to_undirected()

            # 使用Louvain算法进行社区检测
            # 这里使用简单的连通分量作为聚类
            clusters = list(nx.connected_components(undirected_graph))

            # 过滤小聚类
            significant_clusters = [
                list(cluster) for cluster in clusters
                if len(cluster) >= min_cluster_size
            ]

            logger.info(f"发现 {len(significant_clusters)} 个研究聚类")
            return significant_clusters

        except Exception as e:
            logger.error(f"聚类发现失败: {e}")
            return []

    def export_graph_data(self, output_path: str) -> None:
        """
        导出图数据为JSON格式

        Args:
            output_path: 输出文件路径
        """
        logger.info(f"导出图数据到: {output_path}")

        # 计算指标
        logger.info("计算图谱指标...")
        metrics = self.compute_metrics()
        logger.info("图谱指标计算完成")

        # 收集种子节点ID列表
        seed_ids = [
            node_id for node_id, attrs in self.graph.nodes(data=True)
            if attrs.get('is_seed', False)
        ]

        # 准备导出数据
        graph_data = {
            'nodes': [],
            'edges': [],
            'metrics': metrics,
            'metadata': {
                'total_papers': len(self.papers),
                'total_citations': self.graph.number_of_edges(),
                'seed_count': len(seed_ids),
                'seed_ids': seed_ids,  # 添加种子节点ID列表
                'created_at': str(Path().resolve())
            }
        }

        # 导出节点
        for node_id, attrs in self.graph.nodes(data=True):
            node_data = {'id': node_id}
            node_data.update(attrs)
            graph_data['nodes'].append(node_data)

        # 导出边
        for from_node, to_node, attrs in self.graph.edges(data=True):
            edge_data = {
                'from': from_node,
                'to': to_node
            }
            edge_data.update(attrs)
            graph_data['edges'].append(edge_data)

        # 保存到文件
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)

        logger.info(f"图数据导出完成: {output_path}")
        if seed_ids:
            logger.info(f"  包含 {len(seed_ids)} 个种子节点: {seed_ids[:3]}{'...' if len(seed_ids) > 3 else ''}")

    def _get_time_span(self) -> Dict:
        """获取论文时间跨度"""
        years = [paper.get('year', 0) for paper in self.papers.values() if paper.get('year', 0) > 0]
        if years:
            return {
                'earliest': min(years),
                'latest': max(years),
                'span': max(years) - min(years)
            }
        return {'earliest': None, 'latest': None, 'span': 0}

    def _get_research_areas(self) -> List[str]:
        """获取研究领域"""
        all_techniques = []
        for paper in self.papers.values():
            if 'ai_analysis' in paper:
                techniques = paper['ai_analysis'].get('key_techniques', [])
                all_techniques.extend(techniques)

        # 统计频率
        from collections import Counter
        technique_counts = Counter(all_techniques)
        return [tech for tech, count in technique_counts.most_common(10)]

    def _find_influential_papers(self, top_k: int = 5) -> List[Dict]:
        """找到最有影响力的论文"""
        papers_with_scores = []

        for paper_id, paper in self.papers.items():
            score = 0
            score += paper.get('cited_by_count', 0) * 0.7  # 总引用数
            score += self.graph.in_degree(paper_id) * 10   # 图中被引用次数

            papers_with_scores.append({
                'id': paper_id,
                'title': paper.get('title', 'Unknown'),
                'authors': paper.get('authors', []),
                'year': paper.get('year', 0),
                'influence_score': score,
                'cited_by_count': paper.get('cited_by_count', 0)
            })

        # 排序并返回top K
        papers_with_scores.sort(key=lambda x: x['influence_score'], reverse=True)
        return papers_with_scores[:top_k]

    def _analyze_research_trends(self) -> Dict:
        """分析研究趋势"""
        # 按年份统计论文数量
        year_counts = {}
        for paper in self.papers.values():
            year = paper.get('year', 0)
            if year > 0:
                year_counts[year] = year_counts.get(year, 0) + 1

        return {
            'papers_per_year': dict(sorted(year_counts.items())),
            'peak_year': max(year_counts, key=year_counts.get) if year_counts else None,
            'total_years': len(year_counts)
        }


if __name__ == "__main__":
    # 测试代码
    kg = CitationGraph()

    # 创建测试数据
    test_papers = [
        {
            'id': 'W1', 'title': 'Paper A', 'year': 2020, 'cited_by_count': 100,
            'ai_analysis': {'key_techniques': ['Deep Learning', 'CNN']}
        },
        {
            'id': 'W2', 'title': 'Paper B', 'year': 2021, 'cited_by_count': 50,
            'ai_analysis': {'key_techniques': ['Transformer', 'NLP']}
        }
    ]

    test_citations = [('W2', 'W1')]  # W2 引用 W1

    # 构建图谱
    kg.build_citation_network(test_papers, test_citations)

    # 生成可视化
    kg.visualize_graph("./test_graph.html")

    # 生成报告
    report = kg.generate_analysis_report()
    print(f"分析报告: {report['overview']}")