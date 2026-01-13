"""
主题发展脉络分析模块
Topic Evolution Analyzer

负责分析研究主题在知识图谱中的发展脉络，包括：
1. 时间演化分析
2. 关键节点识别（里程碑论文）
3. 研究分支分析（社区检测）
4. 引用链路分析
5. 创新模式分析
6. 关键进化路径提取（Critical Evolutionary Path Extraction）
7. 技术分歧点检测（Technical Bifurcation Detection）
8. 未闭合前沿探测（Open Frontier Detection）
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, Counter
from datetime import datetime
import re

try:
    import networkx as nx
except ImportError:
    raise ImportError("需要安装networkx: pip install networkx")

logger = logging.getLogger(__name__)


# 进化动量权重定义 (Evolutionary Momentum Scores)
EVOLUTIONARY_WEIGHTS = {
    'Overcomes': 3.0,      # 质变：解决了前人的缺陷
    'Realizes': 2.5,       # 填坑：实现了前人的Future Work
    'Extends': 1.0,        # 量变：性能提升
    'Alternative': 1.0,    # 旁支：另辟蹊径
    'Adapts_to': 0.5,      # 迁移：横向扩散
    'Baselines': 0.1,      # 背景噪音：几乎忽略
    'Unknown': 0.3         # 未知类型：低权重
}


class TopicEvolutionAnalyzer:
    """
    主题发展脉络分析器

    基于知识图谱（NetworkX Graph）分析研究主题的演化规律
    """

    def __init__(self, config: Dict = None):
        """
        初始化分析器

        Args:
            config: 配置字典，包含topic_evolution相关配置
        """
        # 默认配置
        self.config = config or {}

        # 提取topic_evolution配置
        evolution_config = self.config.get('topic_evolution', {})

        # 里程碑论文配置
        milestone_config = evolution_config.get('milestone', {})
        self.milestone_top_count = milestone_config.get('top_count', 10)
        self.milestone_citation_weight = milestone_config.get('citation_weight', 0.5)
        self.milestone_pagerank_weight = milestone_config.get('pagerank_weight', 1000)
        self.milestone_betweenness_weight = milestone_config.get('betweenness_weight', 500)
        self.milestone_display_count = milestone_config.get('display_count', 3)

        # 研究分支配置
        branch_config = evolution_config.get('branch', {})
        self.branch_min_size = branch_config.get('min_size', 3)
        self.branch_top_keywords = branch_config.get('top_keywords', 5)
        self.branch_display_count = branch_config.get('display_count', 3)
        self.branch_min_avg_citations = branch_config.get('min_avg_citations', 0)

        # 引用链路配置
        chain_config = evolution_config.get('citation_chain', {})
        self.chain_max_chains = chain_config.get('max_chains', 5)
        self.chain_min_length = chain_config.get('min_length', 3)
        self.chain_start_from_top = chain_config.get('start_from_top_milestones', 5)

        # 时间演化配置
        time_config = evolution_config.get('time_evolution', {})
        self.time_top_papers_per_year = time_config.get('top_papers_per_year', 3)
        self.time_include_citation_types = time_config.get('include_citation_types', True)

        # 创新模式配置
        pattern_config = evolution_config.get('innovation_pattern', {})
        self.pattern_examples_per_type = pattern_config.get('examples_per_type', 3)
        self.pattern_sort_by_count = pattern_config.get('sort_by_count', True)

        # 关键词提取配置
        keyword_config = evolution_config.get('keyword_extraction', {})
        self.keyword_min_length = keyword_config.get('min_word_length', 3)
        self.keyword_remove_stopwords = keyword_config.get('remove_stopwords', True)
        self.keyword_case_sensitive = keyword_config.get('case_sensitive', False)

        # 关键进化路径配置
        evolutionary_path_config = evolution_config.get('evolutionary_path', {})
        self.evol_enabled = evolutionary_path_config.get('enabled', True)
        self.evol_max_paths = evolutionary_path_config.get('max_paths', 3)
        self.evol_min_weight = evolutionary_path_config.get('min_total_weight', 3.0)
        self.evol_time_window_years = evolutionary_path_config.get('time_window_years', None)
        self.evol_custom_weights = evolutionary_path_config.get('custom_weights', {}) or {}

        # 合并自定义权重
        self.evolutionary_weights = EVOLUTIONARY_WEIGHTS.copy()
        if self.evol_custom_weights:
            self.evolutionary_weights.update(self.evol_custom_weights)

        # 技术分歧点检测配置
        bifurcation_config = evolution_config.get('bifurcation', {})
        self.bifur_enabled = bifurcation_config.get('enabled', True)
        self.bifur_max_bifurcations = bifurcation_config.get('max_bifurcations', 5)
        self.bifur_fork_edge_types = bifurcation_config.get('fork_edge_types', ['Alternative', 'Extends'])
        self.bifur_min_children = bifurcation_config.get('min_children', 2)
        self.bifur_method_sim_threshold = bifurcation_config.get('method_similarity_threshold', 0.3)
        self.bifur_problem_sim_threshold = bifurcation_config.get('problem_similarity_threshold', 0.6)
        self.bifur_use_cosine = bifurcation_config.get('use_cosine_similarity', True)

        # 未闭合前沿探测配置
        frontier_config = evolution_config.get('open_frontier', {})
        self.frontier_enabled = frontier_config.get('enabled', True)
        self.frontier_recent_years = frontier_config.get('recent_years', 2)
        self.frontier_max_open_problems = frontier_config.get('max_open_problems', 10)
        self.frontier_max_ideas = frontier_config.get('max_cross_domain_ideas', 5)
        self.frontier_lim_sim_threshold = frontier_config.get('limitation_similarity_threshold', 0.5)
        self.frontier_min_contrib_score = frontier_config.get('min_contribution_score', 0.3)

        logger.info(f"TopicEvolutionAnalyzer初始化完成")
        logger.info(f"  里程碑论文: Top {self.milestone_top_count}, 显示 {self.milestone_display_count}")
        logger.info(f"  研究分支: 最小规模 {self.branch_min_size}, 关键词 {self.branch_top_keywords}")
        logger.info(f"  引用链路: 最大 {self.chain_max_chains} 条, 最小长度 {self.chain_min_length}")
        logger.info(f"  关键进化路径: {'启用' if self.evol_enabled else '禁用'}, 最多 {self.evol_max_paths} 条")
        logger.info(f"  技术分歧点检测: {'启用' if self.bifur_enabled else '禁用'}, 最多 {self.bifur_max_bifurcations} 个")
        logger.info(f"  未闭合前沿探测: {'启用' if self.frontier_enabled else '禁用'}, 最多 {self.frontier_max_open_problems} 个问题")

    def analyze(self, graph: nx.DiGraph, topic: str) -> Dict:
        """
        执行完整的主题发展脉络分析（双核心方向）

        核心方向1: 回溯脉络 (Retrospective Analysis)
        核心方向2: 预测未来 (Future Prediction)

        Args:
            graph: NetworkX有向图，节点包含论文信息
            topic: 研究主题名称

        Returns:
            分析报告字典
        """
        if len(graph.nodes()) == 0:
            logger.warning("知识图谱为空，跳过主题发展脉络分析")
            return {}

        logger.info(f"开始分析主题发展脉络: '{topic}'")
        logger.info(f"  图谱规模: {len(graph.nodes())} 节点, {len(graph.edges())} 边")

        # 基础分析
        year_stats = self._analyze_time_evolution(graph)
        milestone_papers = self._identify_milestone_papers(graph)

        # ========== 核心方向1: 回溯脉络 ==========
        logger.info("\n  🔙 核心方向1: 回溯脉络分析...")

        # 1.1 识别进化主干 vs 旁支修补
        logger.info("    📍 识别进化主干 vs 旁支修补...")
        backbone_analysis = self._analyze_backbone_vs_incremental(graph)

        # 1.2 识别技术分叉口
        logger.info("    🔀 识别技术分叉口...")
        bifurcations = self._detect_technical_bifurcations(graph) if self.bifur_enabled else []

        # 1.3 识别跨界入侵
        logger.info("    🌐 识别跨界入侵...")
        cross_domain_invasions = self._detect_cross_domain_invasions(graph)

        # ========== 核心方向2: 预测未来 ==========
        logger.info("\n  🔮 核心方向2: 预测未来...")

        # 2.1 等级一：捡漏型Idea（未实现的Future Work）
        logger.info("    💡 等级一：捡漏型Idea...")
        low_hanging_fruits = self._detect_low_hanging_fruits(graph, year_stats)

        # 2.2 等级二：攻坚型Idea（未解决的Limitation）
        logger.info("    🔨 等级二：攻坚型Idea...")
        hard_nuts = self._detect_hard_nuts(graph, milestone_papers)

        # 2.3 等级三：创新型Idea（跨域迁移/组合拳）
        logger.info("    🚀 等级三：创新型Idea...")
        innovative_ideas = self._generate_innovative_ideas(graph)

        # 生成报告
        report = {
            'topic': topic,
            'analysis_time': datetime.now().isoformat(),
            'graph_overview': {
                'total_papers': len(graph.nodes()),
                'total_citations': len(graph.edges()),
                'year_range': f"{min(year_stats.keys())}-{max(year_stats.keys())}" if year_stats else "Unknown"
            },

            # 核心方向1: 回溯脉络
            'retrospective_analysis': {
                'backbone_vs_incremental': backbone_analysis,
                'technical_bifurcations': bifurcations,
                'cross_domain_invasions': cross_domain_invasions
            },

            # 核心方向2: 预测未来
            'future_prediction': {
                'level_1_low_hanging_fruits': low_hanging_fruits,
                'level_2_hard_nuts': hard_nuts,
                'level_3_innovative_ideas': innovative_ideas
            },

            # 保留原有分析（兼容性）
            'milestone_papers': milestone_papers,
            'time_evolution': dict(sorted(year_stats.items()))
        }

        # 输出概要信息
        self._log_summary(report)

        return report

    def _analyze_time_evolution(self, graph: nx.DiGraph) -> Dict:
        """
        分析时间演化

        Returns:
            年份统计字典
        """
        year_stats = defaultdict(lambda: {
            'papers': [],
            'citation_types': defaultdict(int),
            'avg_citations': 0
        })

        # 收集每年的论文
        for node_id, node_data in graph.nodes(data=True):
            year = node_data.get('year')
            if year:
                year_stats[year]['papers'].append({
                    'id': node_id,
                    'title': node_data.get('title', ''),
                    'cited_by_count': node_data.get('cited_by_count', 0)
                })

        # 计算每年的统计信息
        for year, stats in year_stats.items():
            if stats['papers']:
                # 平均引用数
                stats['avg_citations'] = sum(
                    p['cited_by_count'] for p in stats['papers']
                ) / len(stats['papers'])

                # 按引用数排序，只保留top N
                stats['papers'] = sorted(
                    stats['papers'],
                    key=lambda x: x['cited_by_count'],
                    reverse=True
                )[:self.time_top_papers_per_year]

        # 统计每年的引用类型分布
        if self.time_include_citation_types:
            for source, target, edge_data in graph.edges(data=True):
                source_year = graph.nodes[source].get('year')
                edge_type = edge_data.get('edge_type', 'Unknown')
                if source_year:
                    year_stats[source_year]['citation_types'][edge_type] += 1

        return year_stats

    def _identify_milestone_papers(self, graph: nx.DiGraph) -> List[Dict]:
        """
        识别里程碑论文

        使用综合评分：引用数 + PageRank + 中介中心性

        Returns:
            里程碑论文列表
        """
        # 计算节点重要性指标
        try:
            pagerank = nx.pagerank(graph, alpha=0.85)
            betweenness = nx.betweenness_centrality(graph)
        except Exception as e:
            logger.warning(f"计算图指标失败: {e}，使用默认值")
            pagerank = {node: 0 for node in graph.nodes()}
            betweenness = {node: 0 for node in graph.nodes()}

        # 综合评分
        milestone_papers = []
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            score = (
                node_data.get('cited_by_count', 0) * self.milestone_citation_weight +
                pagerank.get(node_id, 0) * self.milestone_pagerank_weight +
                betweenness.get(node_id, 0) * self.milestone_betweenness_weight
            )
            milestone_papers.append({
                'id': node_id,
                'title': node_data.get('title', ''),
                'year': node_data.get('year'),
                'cited_by_count': node_data.get('cited_by_count', 0),
                'pagerank': pagerank.get(node_id, 0),
                'betweenness': betweenness.get(node_id, 0),
                'score': score
            })

        # 按综合评分排序
        milestone_papers = sorted(
            milestone_papers,
            key=lambda x: x['score'],
            reverse=True
        )[:self.milestone_top_count]

        return milestone_papers

    def _analyze_research_branches(self, graph: nx.DiGraph) -> List[Dict]:
        """
        分析研究分支（社区检测）

        使用Louvain算法进行社区检测

        Returns:
            研究分支列表
        """
        try:
            # 使用Louvain算法进行社区检测
            communities = nx.community.louvain_communities(graph.to_undirected())

            research_branches = []
            for i, community in enumerate(communities):
                if len(community) < self.branch_min_size:
                    continue

                # 分析该分支的特征
                branch_papers = []
                branch_years = []
                branch_citations = []

                for node_id in community:
                    node_data = graph.nodes[node_id]
                    branch_papers.append({
                        'id': node_id,
                        'title': node_data.get('title', ''),
                        'year': node_data.get('year')
                    })
                    if node_data.get('year'):
                        branch_years.append(node_data.get('year'))
                    branch_citations.append(node_data.get('cited_by_count', 0))

                # 计算平均引用数
                avg_citations = sum(branch_citations) / len(branch_citations) if branch_citations else 0

                # 过滤低质量分支
                if avg_citations < self.branch_min_avg_citations:
                    continue

                # 识别分支的关键词
                branch_keywords = self._extract_keywords(
                    [p['title'] for p in branch_papers],
                    top_k=self.branch_top_keywords
                )

                research_branches.append({
                    'branch_id': i + 1,
                    'size': len(community),
                    'papers': sorted(branch_papers, key=lambda x: x.get('year', 0))[:5],
                    'year_range': f"{min(branch_years)}-{max(branch_years)}" if branch_years else "Unknown",
                    'avg_citations': avg_citations,
                    'keywords': branch_keywords
                })

            # 按规模排序
            research_branches = sorted(
                research_branches,
                key=lambda x: x['size'],
                reverse=True
            )

            return research_branches

        except Exception as e:
            logger.warning(f"社区检测失败: {e}")
            return []

    def _analyze_citation_chains(
        self,
        graph: nx.DiGraph,
        milestone_papers: List[Dict]
    ) -> List[Dict]:
        """
        分析引用链路（引用传承路径）

        从里程碑论文开始，找出最长的引用链

        Args:
            graph: 知识图谱
            milestone_papers: 里程碑论文列表

        Returns:
            引用链路列表
        """
        citation_chains = []

        try:
            # 从top N里程碑论文开始追踪
            for start_node in milestone_papers[:self.chain_start_from_top]:
                start_id = start_node['id']
                if start_id not in graph:
                    continue

                # 找出从该节点出发的最长路径
                lengths = nx.single_source_shortest_path_length(graph, start_id)
                if not lengths:
                    continue

                farthest_node = max(lengths.items(), key=lambda x: x[1])

                # 检查路径长度
                if farthest_node[1] < self.chain_min_length - 1:  # -1 因为长度是边数
                    continue

                # 获取路径
                path = nx.shortest_path(graph, start_id, farthest_node[0])
                if len(path) < self.chain_min_length:
                    continue

                # 构建链路信息
                chain_info = []
                for node in path:
                    node_data = graph.nodes[node]
                    chain_info.append({
                        'id': node,
                        'title': node_data.get('title', '')[:60],
                        'year': node_data.get('year')
                    })

                citation_chains.append({
                    'length': len(path),
                    'chain': chain_info
                })

        except Exception as e:
            logger.warning(f"引用链路分析失败: {e}")

        # 按长度排序，取top N
        citation_chains = sorted(
            citation_chains,
            key=lambda x: x['length'],
            reverse=True
        )[:self.chain_max_chains]

        return citation_chains

    def _analyze_innovation_patterns(self, graph: nx.DiGraph) -> Dict:
        """
        分析创新模式（引用类型统计）

        统计不同引用类型的分布和示例

        Returns:
            创新模式字典
        """
        innovation_patterns = defaultdict(lambda: {
            'count': 0,
            'examples': []
        })

        for source, target, edge_data in graph.edges(data=True):
            edge_type = edge_data.get('edge_type', 'Unknown')
            innovation_patterns[edge_type]['count'] += 1

            # 保存示例
            if len(innovation_patterns[edge_type]['examples']) < self.pattern_examples_per_type:
                source_data = graph.nodes[source]
                target_data = graph.nodes[target]
                innovation_patterns[edge_type]['examples'].append({
                    'from': source_data.get('title', '')[:50],
                    'to': target_data.get('title', '')[:50],
                    'from_year': source_data.get('year'),
                    'to_year': target_data.get('year')
                })

        # 转换为普通字典
        return {k: dict(v) for k, v in innovation_patterns.items()}

    def _extract_keywords(self, titles: List[str], top_k: int = 5) -> List[str]:
        """
        从标题列表中提取关键词

        Args:
            titles: 标题列表
            top_k: 返回top k关键词

        Returns:
            关键词列表
        """
        # 停用词
        stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'using', 'based', 'via',
            'through', 'into', 'over', 'after', 'before', 'between', 'under'
        } if self.keyword_remove_stopwords else set()

        # 提取所有单词
        words = []
        for title in titles:
            # 转小写（如果不区分大小写）
            title_processed = title if self.keyword_case_sensitive else title.lower()

            # 提取单词（最小长度）
            pattern = rf'\b[a-zA-Z]{{{self.keyword_min_length},}}\b'
            if not self.keyword_case_sensitive:
                pattern = rf'\b[a-z]{{{self.keyword_min_length},}}\b'

            words_in_title = re.findall(pattern, title_processed)
            words.extend([w for w in words_in_title if w not in stopwords])

        # 统计频次
        word_counts = Counter(words)

        # 返回top k关键词
        return [word for word, count in word_counts.most_common(top_k)]

    def _analyze_backbone_vs_incremental(self, graph: nx.DiGraph) -> Dict:
        """
        识别进化主干 vs 旁支修补（增强版）

        核心洞察：
        1. 主干路径（Backbone）：只保留 Overcomes 和 Realizes 类型的连接
           - 含金量最高的演化线
           - 每个节点都在解决前人的致命缺陷

        2. 渐进路径（Incremental）：只保留 Extends 类型的连接
           - 内卷/刷榜的演化线
           - 同一方法论下的微调优化

        3. 破局点（Breakthrough）：从 Extends 内卷中突然跳出 Overcomes
           - 这是最有价值的创新点
           - 代表方法论的突破

        Returns:
            详细的主干vs旁支分析报告
        """
        backbone_paths = []
        incremental_paths = []
        breakthrough_points = []

        # 统计每个节点的输入输出边类型
        node_stats = {}

        for node in graph.nodes():
            in_edges = list(graph.in_edges(node, data=True))
            out_edges = list(graph.out_edges(node, data=True))

            in_overcomes = sum(1 for _, _, d in in_edges if d.get('edge_type') == 'Overcomes')
            in_realizes = sum(1 for _, _, d in in_edges if d.get('edge_type') == 'Realizes')
            in_extends = sum(1 for _, _, d in in_edges if d.get('edge_type') == 'Extends')

            out_overcomes = sum(1 for _, _, d in out_edges if d.get('edge_type') == 'Overcomes')
            out_realizes = sum(1 for _, _, d in out_edges if d.get('edge_type') == 'Realizes')
            out_extends = sum(1 for _, _, d in out_edges if d.get('edge_type') == 'Extends')

            node_stats[node] = {
                'in_overcomes': in_overcomes,
                'in_realizes': in_realizes,
                'in_extends': in_extends,
                'out_overcomes': out_overcomes,
                'out_realizes': out_realizes,
                'out_extends': out_extends
            }

        # 构建主干路径和渐进路径
        for source, target, data in graph.edges(data=True):
            edge_type = data.get('edge_type', 'Unknown')

            if edge_type in ['Overcomes', 'Realizes']:
                backbone_paths.append({
                    'from': {
                        'id': source,
                        'title': graph.nodes[source].get('title', '')[:60],
                        'year': graph.nodes[source].get('year')
                    },
                    'to': {
                        'id': target,
                        'title': graph.nodes[target].get('title', '')[:60],
                        'year': graph.nodes[target].get('year')
                    },
                    'type': edge_type
                })
            elif edge_type == 'Extends':
                incremental_paths.append({
                    'from': {
                        'id': source,
                        'title': graph.nodes[source].get('title', '')[:60],
                        'year': graph.nodes[source].get('year')
                    },
                    'to': {
                        'id': target,
                        'title': graph.nodes[target].get('title', '')[:60],
                        'year': graph.nodes[target].get('year')
                    }
                })

        # 识别破局点：从 Extends 内卷中跳出 Overcomes
        # 标准：输入有多个 Extends，输出有 Overcomes/Realizes
        for node, stats in node_stats.items():
            # 判断是否为破局点
            is_breakthrough = False
            breakthrough_score = 0

            # 模式1：从Extends内卷中跳出Overcomes（最经典）
            if stats['in_extends'] >= 2 and stats['out_overcomes'] >= 1:
                is_breakthrough = True
                breakthrough_score = stats['in_extends'] * 1.0 + stats['out_overcomes'] * 3.0

            # 模式2：从纯Extends变为Realizes（填坑型突破）
            elif stats['in_extends'] >= 1 and stats['out_realizes'] >= 1 and stats['out_overcomes'] == 0:
                is_breakthrough = True
                breakthrough_score = stats['in_extends'] * 0.5 + stats['out_realizes'] * 2.0

            if is_breakthrough:
                node_data = graph.nodes[node]
                breakthrough_points.append({
                    'node': node,
                    'title': node_data.get('title', '')[:80],
                    'year': node_data.get('year'),
                    'cited_by_count': node_data.get('cited_by_count', 0),
                    'in_extends': stats['in_extends'],
                    'out_overcomes': stats['out_overcomes'],
                    'out_realizes': stats['out_realizes'],
                    'breakthrough_score': breakthrough_score,
                    'breakthrough_type': 'Overcomes突破' if stats['out_overcomes'] > 0 else 'Realizes填坑'
                })

        # 按突破分数排序
        breakthrough_points = sorted(
            breakthrough_points,
            key=lambda x: x['breakthrough_score'],
            reverse=True
        )[:10]

        # 分析主干路径的连贯性
        backbone_chains = self._extract_backbone_chains(graph, backbone_paths)

        # 分析渐进路径的瓶颈
        incremental_bottlenecks = self._analyze_incremental_bottlenecks(graph, incremental_paths)

        return {
            'summary': {
                'backbone_count': len(backbone_paths),
                'incremental_count': len(incremental_paths),
                'breakthrough_count': len(breakthrough_points),
                'ratio': len(backbone_paths) / len(incremental_paths) if len(incremental_paths) > 0 else 0
            },
            'backbone_paths': backbone_paths[:20],  # 返回前20个主干路径
            'incremental_paths': incremental_paths[:20],  # 返回前20个渐进路径
            'breakthrough_points': breakthrough_points,
            'backbone_chains': backbone_chains,  # 主干连续链
            'incremental_bottlenecks': incremental_bottlenecks  # 渐进瓶颈
        }

    def _extract_backbone_chains(self, graph: nx.DiGraph, backbone_paths: List[Dict]) -> List[Dict]:
        """
        提取主干连续链：连续的Overcomes/Realizes路径

        这些链代表了"硬核攻坚"的演化线
        """
        # 构建只包含主干边的子图
        backbone_graph = nx.DiGraph()

        for path in backbone_paths:
            source = path['from']['id']
            target = path['to']['id']
            edge_type = path['type']
            backbone_graph.add_edge(source, target, edge_type=edge_type)

            # 添加节点属性
            for node_id in [source, target]:
                if node_id in graph.nodes():
                    node_data = graph.nodes[node_id]
                    backbone_graph.add_node(node_id, **node_data)

        # 找出所有简单路径（长度>=3）
        chains = []

        # 找出起点（入度为0的节点）
        source_nodes = [n for n in backbone_graph.nodes() if backbone_graph.in_degree(n) == 0]
        # 找出终点（出度为0的节点）
        target_nodes = [n for n in backbone_graph.nodes() if backbone_graph.out_degree(n) == 0]

        for source in source_nodes:
            for target in target_nodes:
                if source == target:
                    continue

                if nx.has_path(backbone_graph, source, target):
                    # 找最长路径
                    try:
                        path = nx.shortest_path(backbone_graph, source, target)

                        if len(path) >= 3:  # 至少3个节点
                            chain_info = []
                            edge_types = []

                            for i, node in enumerate(path):
                                node_data = graph.nodes[node]
                                chain_info.append({
                                    'id': node,
                                    'title': node_data.get('title', '')[:50],
                                    'year': node_data.get('year')
                                })

                                # 获取边类型
                                if i < len(path) - 1:
                                    edge_data = backbone_graph[path[i]][path[i+1]]
                                    edge_types.append(edge_data.get('edge_type', 'Unknown'))

                            chains.append({
                                'length': len(path),
                                'chain': chain_info,
                                'edge_types': edge_types,
                                'year_span': chain_info[-1]['year'] - chain_info[0]['year'] if chain_info[0].get('year') and chain_info[-1].get('year') else 0
                            })
                    except:
                        continue

        # 按长度排序
        chains = sorted(chains, key=lambda x: x['length'], reverse=True)[:5]

        return chains

    def _analyze_incremental_bottlenecks(self, graph: nx.DiGraph, incremental_paths: List[Dict]) -> List[Dict]:
        """
        分析渐进路径的瓶颈：找到那些被大量Extends引用但没有后续突破的论文

        这些论文代表了"内卷终点"，可能已经接近瓶颈
        """
        # 构建只包含Extends边的子图
        extends_graph = nx.DiGraph()

        for path in incremental_paths:
            source = path['from']['id']
            target = path['to']['id']
            extends_graph.add_edge(source, target)

            # 添加节点属性
            for node_id in [source, target]:
                if node_id in graph.nodes():
                    node_data = graph.nodes[node_id]
                    extends_graph.add_node(node_id, **node_data)

        bottlenecks = []

        for node in extends_graph.nodes():
            in_degree = extends_graph.in_degree(node)

            # 被大量Extends引用（至少3个）
            if in_degree >= 3:
                # 检查是否有后续突破（在原图中）
                has_breakthrough = False

                for pred in graph.predecessors(node):
                    edge_data = graph[pred][node]
                    edge_type = edge_data.get('edge_type', 'Unknown')

                    if edge_type in ['Overcomes', 'Realizes']:
                        has_breakthrough = True
                        break

                # 没有突破的才是瓶颈
                if not has_breakthrough:
                    node_data = graph.nodes[node]
                    bottlenecks.append({
                        'node': node,
                        'title': node_data.get('title', '')[:60],
                        'year': node_data.get('year'),
                        'cited_by_count': node_data.get('cited_by_count', 0),
                        'extends_in_count': in_degree,
                        'reason': f'被{in_degree}个Extends引用但无后续突破，可能已达瓶颈'
                    })

        # 按Extends引用数排序
        bottlenecks = sorted(bottlenecks, key=lambda x: x['extends_in_count'], reverse=True)[:5]

        return bottlenecks

    def _detect_cross_domain_invasions(self, graph: nx.DiGraph) -> List[Dict]:
        """
        识别跨界入侵（Cross-Domain Invasion）- 增强版

        核心思想：
        追踪 Adapts_to 类型的连接，识别方法论的跨域迁移。
        这种节点通常意味着方法论的降维打击，是该Topic的一次重要外部输血。

        增强分析：
        1. 追踪这些节点的前身（原始领域）
        2. 分析迁移的影响力（后续工作数量）
        3. 识别最成功的跨域迁移案例

        Returns:
            跨界入侵列表（包含影响力分析）
        """
        invasions = []

        for source, target, data in graph.edges(data=True):
            if data.get('edge_type') == 'Adapts_to':
                source_data = graph.nodes[source]
                target_data = graph.nodes[target]

                # 分析迁移的影响力
                # 1. 统计source的后续工作数量（迁移后的影响）
                source_descendants = list(nx.descendants(graph, source)) if source in graph else []
                impact_count = len(source_descendants)

                # 2. 统计target的原有影响力（被迁移的基础）
                target_citations = target_data.get('cited_by_count', 0)

                # 3. 判断迁移成功度
                if impact_count > 10:
                    success_level = 'highly_successful'  # 高度成功
                elif impact_count > 5:
                    success_level = 'successful'  # 成功
                elif impact_count > 0:
                    success_level = 'moderate'  # 一般
                else:
                    success_level = 'limited'  # 有限

                # 4. 尝试识别原始领域和目标领域（基于关键词）
                source_keywords = self._extract_domain_keywords(source_data.get('title', ''))
                target_keywords = self._extract_domain_keywords(target_data.get('title', ''))

                invasions.append({
                    'from': {
                        'id': source,
                        'title': source_data.get('title', '')[:80],
                        'year': source_data.get('year'),
                        'domain_keywords': source_keywords
                    },
                    'to': {
                        'id': target,
                        'title': target_data.get('title', '')[:80],
                        'year': target_data.get('year'),
                        'citations': target_citations,
                        'domain_keywords': target_keywords
                    },
                    'impact_analysis': {
                        'descendants_count': impact_count,
                        'success_level': success_level,
                        'year_gap': source_data.get('year', 0) - target_data.get('year', 0) if source_data.get('year') and target_data.get('year') else 0
                    },
                    'cross_domain_story': self._generate_invasion_story(
                        source_keywords,
                        target_keywords,
                        source_data.get('year'),
                        impact_count,
                        success_level
                    )
                })

        # 按影响力排序
        invasions = sorted(invasions, key=lambda x: x['impact_analysis']['descendants_count'], reverse=True)[:15]

        return invasions

    def _extract_domain_keywords(self, title: str) -> List[str]:
        """
        从标题中提取领域关键词

        Args:
            title: 论文标题

        Returns:
            关键词列表
        """
        # 常见领域关键词
        domain_keywords_dict = {
            'nlp': ['language', 'text', 'nlp', 'semantic', 'linguistic', 'dialogue', 'translation', 'sentiment'],
            'cv': ['image', 'vision', 'visual', 'object', 'detection', 'segmentation', 'recognition', 'video'],
            'rl': ['reinforcement', 'policy', 'reward', 'agent', 'environment', 'q-learning'],
            'graph': ['graph', 'node', 'edge', 'network', 'topology'],
            'audio': ['audio', 'speech', 'sound', 'acoustic', 'voice'],
            'generative': ['generation', 'generative', 'gan', 'diffusion', 'synthesis'],
            'representation': ['representation', 'embedding', 'feature', 'encoding']
        }

        title_lower = title.lower()
        found_domains = []

        for domain, keywords in domain_keywords_dict.items():
            for kw in keywords:
                if kw in title_lower:
                    found_domains.append(domain)
                    break  # 找到一个就够了

        # 如果没找到，返回一些从标题中提取的通用关键词
        if not found_domains:
            words = re.findall(r'\b[a-z]{4,}\b', title_lower)
            return words[:3] if words else ['unknown']

        return found_domains

    def _generate_invasion_story(
        self,
        source_keywords: List[str],
        target_keywords: List[str],
        year: Optional[int],
        impact: int,
        success: str
    ) -> str:
        """
        生成跨域迁移的故事描述

        Args:
            source_keywords: 源领域关键词
            target_keywords: 目标领域关键词
            year: 迁移年份
            impact: 影响力（后续工作数）
            success: 成功程度

        Returns:
            描述文本
        """
        source_str = ', '.join(source_keywords[:2]) if source_keywords else '未知领域'
        target_str = ', '.join(target_keywords[:2]) if target_keywords else '未知领域'

        success_desc = {
            'highly_successful': f'产生了{impact}个后续工作，成为该领域的重要突破',
            'successful': f'产生了{impact}个后续工作，获得较好发展',
            'moderate': f'产生了{impact}个后续工作，有一定影响',
            'limited': '后续发展有限'
        }

        year_str = f"在{year}年，" if year else ""

        return f"{year_str}将[{target_str}]领域的方法迁移到[{source_str}]领域，{success_desc.get(success, '影响未知')}"

    def _detect_low_hanging_fruits(self, graph: nx.DiGraph, year_stats: Dict) -> List[Dict]:
        """
        等级一：捡漏型Idea - 寻找未被Realized的Future Work（增强版）

        算法逻辑：
        1. 提取最近3年发表的论文（"前沿"）
        2. 提取它们文本中的 Future_Work 部分
        3. 确认是否已经有后续论文通过 Realizes 连接到它
        4. 如果没有，那么"把这篇论文的Future Work做出来"就是一个现成的Idea

        增强：
        - 按论文影响力（引用数）排序，优先推荐高影响力论文的Future Work
        - 分析Future Work的可行性（长度、具体性）
        - 提供实现难度评估

        Returns:
            捡漏型Idea列表
        """
        if not year_stats:
            return []

        years = sorted(year_stats.keys())
        recent_cutoff = years[-1] - 3 if len(years) > 3 else years[0]

        low_hanging = []

        for node in graph.nodes():
            node_data = graph.nodes[node]
            year = node_data.get('year')

            if not year or year < recent_cutoff:
                continue

            # 从deep_analysis结构中获取future_work
            deep_analysis = node_data.get('deep_analysis', {})
            future_work = deep_analysis.get('future_work', {}).get('content', '')

            if not future_work or len(future_work) < 20:
                continue

            # 检查是否有后续工作通过Realizes实现
            has_realization = False
            for pred in graph.predecessors(node):
                edge_data = graph[pred][node]
                if edge_data.get('edge_type') == 'Realizes':
                    has_realization = True
                    break

            if not has_realization:
                # 评估可行性（基于描述长度和具体性）
                feasibility_score = self._assess_idea_feasibility(future_work)

                # 评估难度
                difficulty = self._assess_implementation_difficulty(future_work, node_data)

                low_hanging.append({
                    'paper': {
                        'id': node,
                        'title': node_data.get('title', '')[:80],
                        'year': year,
                        'cited_by_count': node_data.get('cited_by_count', 0)
                    },
                    'future_work': future_work[:300],
                    'feasibility_score': feasibility_score,
                    'difficulty': difficulty,
                    'priority': node_data.get('cited_by_count', 0) * feasibility_score,  # 综合优先级
                    'recommendation': self._generate_implementation_recommendation(future_work, difficulty)
                })

        # 按优先级排序（引用数 * 可行性）
        low_hanging = sorted(low_hanging, key=lambda x: x['priority'], reverse=True)[:15]

        return low_hanging

    def _assess_idea_feasibility(self, future_work: str) -> float:
        """
        评估Future Work的可行性

        Args:
            future_work: Future Work描述

        Returns:
            可行性分数（0-1）
        """
        score = 0.5  # 基础分

        # 长度越长越具体
        if len(future_work) > 100:
            score += 0.2
        elif len(future_work) > 200:
            score += 0.3

        # 包含具体关键词
        action_keywords = ['apply', 'extend', 'improve', 'combine', 'test', 'evaluate', 'implement']
        for kw in action_keywords:
            if kw in future_work.lower():
                score += 0.1
                break

        # 包含具体方法或数据集
        specific_keywords = ['dataset', 'benchmark', 'algorithm', 'model', 'framework']
        for kw in specific_keywords:
            if kw in future_work.lower():
                score += 0.1
                break

        return min(score, 1.0)

    def _assess_implementation_difficulty(self, future_work: str, paper_data: Dict) -> str:
        """
        评估实现难度

        Args:
            future_work: Future Work描述
            paper_data: 论文数据

        Returns:
            难度等级：'easy', 'medium', 'hard'
        """
        # 基于关键词判断难度
        easy_keywords = ['extend', 'apply', 'test', 'evaluate', 'additional']
        medium_keywords = ['improve', 'enhance', 'combine', 'integrate']
        hard_keywords = ['novel', 'new', 'develop', 'design', 'fundamental', 'theoretical']

        future_lower = future_work.lower()

        hard_count = sum(1 for kw in hard_keywords if kw in future_lower)
        medium_count = sum(1 for kw in medium_keywords if kw in future_lower)
        easy_count = sum(1 for kw in easy_keywords if kw in future_lower)

        if hard_count > medium_count and hard_count > easy_count:
            return 'hard'
        elif medium_count > easy_count:
            return 'medium'
        else:
            return 'easy'

    def _generate_implementation_recommendation(self, future_work: str, difficulty: str) -> str:
        """
        生成实现建议

        Args:
            future_work: Future Work描述
            difficulty: 难度等级

        Returns:
            实现建议文本
        """
        difficulty_desc = {
            'easy': '难度较低，可快速验证',
            'medium': '难度中等，需要一定工程实现',
            'hard': '难度较高，需要深入研究和创新'
        }

        return f"{difficulty_desc.get(difficulty, '难度未知')}。建议优先阅读原论文，理解其核心方法后进行扩展。"

    def _detect_hard_nuts(self, graph: nx.DiGraph, milestone_papers: List[Dict]) -> List[Dict]:
        """
        等级二：攻坚型Idea - 寻找未被Overcomes的Limitation（增强版）

        算法逻辑：
        1. 找到该领域引用量最高、但发布时间较近的几篇"基石论文"
        2. 提取它们的 Limitation
        3. 检查是否有任何新论文通过 Overcomes 连接它
        4. 如果还没有，说明大家虽然都在引用它，但它的核心缺陷依然存在

        增强：
        - 评估Limitation的严重性和影响
        - 分析为何至今未被解决（技术难度、资源要求等）
        - 提供攻坚建议

        你的Idea：专门针对这个未解的Limitation提出新的Method

        Returns:
            攻坚型Idea列表
        """
        hard_nuts = []

        for milestone in milestone_papers[:15]:  # 扩展到前15篇
            node_id = milestone['id']
            if node_id not in graph:
                continue

            node_data = graph.nodes[node_id]

            # 从deep_analysis结构中获取limitation
            deep_analysis = node_data.get('deep_analysis', {})
            limitation = deep_analysis.get('limitation', {}).get('content', '')

            if not limitation or len(limitation) < 20:
                continue

            # 检查是否有后续工作通过Overcomes解决
            has_overcome = False
            overcome_attempts = []  # 记录尝试解决的工作

            for pred in graph.predecessors(node_id):
                edge_data = graph[pred][node_id]
                edge_type = edge_data.get('edge_type', 'Unknown')

                if edge_type == 'Overcomes':
                    has_overcome = True
                    break
                elif edge_type in ['Extends', 'Realizes']:
                    # 虽然不是Overcomes，但有人在尝试改进
                    overcome_attempts.append({
                        'id': pred,
                        'title': graph.nodes[pred].get('title', '')[:60],
                        'type': edge_type
                    })

            if not has_overcome:
                # 评估Limitation的严重性
                severity = self._assess_limitation_severity(limitation)

                # 分析为何未被解决
                unsolved_reason = self._analyze_unsolved_reason(
                    limitation,
                    len(overcome_attempts),
                    node_data
                )

                # 评估攻坚难度
                attack_difficulty = self._assess_attack_difficulty(
                    limitation,
                    node_data.get('cited_by_count', 0),
                    len(overcome_attempts)
                )

                hard_nuts.append({
                    'paper': {
                        'id': node_id,
                        'title': node_data.get('title', '')[:80],
                        'year': node_data.get('year'),
                        'cited_by_count': node_data.get('cited_by_count', 0)
                    },
                    'limitation': limitation[:300],
                    'severity': severity,
                    'unsolved_reason': unsolved_reason,
                    'attack_difficulty': attack_difficulty,
                    'overcome_attempts': overcome_attempts[:3],  # 显示前3个尝试
                    'impact_potential': node_data.get('cited_by_count', 0) * severity,  # 解决后的潜在影响
                    'research_direction': self._suggest_research_direction(limitation, attack_difficulty)
                })

        # 按潜在影响排序
        hard_nuts = sorted(hard_nuts, key=lambda x: x['impact_potential'], reverse=True)[:12]

        return hard_nuts

    def _assess_limitation_severity(self, limitation: str) -> float:
        """
        评估Limitation的严重性

        Args:
            limitation: Limitation描述

        Returns:
            严重性分数（0-1）
        """
        score = 0.5  # 基础分

        # 严重性关键词
        critical_keywords = ['critical', 'major', 'significant', 'fundamental', 'serious']
        moderate_keywords = ['important', 'notable', 'considerable']
        minor_keywords = ['minor', 'small', 'slight']

        limitation_lower = limitation.lower()

        # 检查严重性
        if any(kw in limitation_lower for kw in critical_keywords):
            score += 0.4
        elif any(kw in limitation_lower for kw in moderate_keywords):
            score += 0.2
        elif any(kw in limitation_lower for kw in minor_keywords):
            score -= 0.2

        # 检查影响范围
        scope_keywords = ['all', 'general', 'common', 'widespread', 'universal']
        if any(kw in limitation_lower for kw in scope_keywords):
            score += 0.2

        return min(max(score, 0.1), 1.0)

    def _analyze_unsolved_reason(self, limitation: str, attempt_count: int, paper_data: Dict) -> str:
        """
        分析Limitation为何至今未被解决

        Args:
            limitation: Limitation描述
            attempt_count: 尝试解决的工作数量
            paper_data: 论文数据

        Returns:
            分析结论
        """
        limitation_lower = limitation.lower()

        # 技术难度高
        if any(kw in limitation_lower for kw in ['complex', 'difficult', 'challenging', 'non-trivial']):
            if attempt_count > 0:
                return f"技术难度极高，已有{attempt_count}个工作尝试改进但未能从根本上解决"
            else:
                return "技术难度极高，至今无人敢于挑战"

        # 资源要求高
        if any(kw in limitation_lower for kw in ['expensive', 'costly', 'large-scale', 'computation']):
            return "需要大量计算资源或数据，对研究者要求较高"

        # 理论基础问题
        if any(kw in limitation_lower for kw in ['theoretical', 'fundamental', 'framework']):
            return "涉及理论基础问题，需要方法论突破"

        # 其他
        if attempt_count > 0:
            return f"已有{attempt_count}个工作尝试改进，但核心问题仍未解决"
        else:
            return "尚未引起足够重视，或需要跨学科知识"

    def _assess_attack_difficulty(self, limitation: str, citations: int, attempts: int) -> str:
        """
        评估攻坚难度

        Args:
            limitation: Limitation描述
            citations: 论文引用数
            attempts: 尝试解决的工作数量

        Returns:
            难度等级：'very_hard', 'hard', 'medium'
        """
        # 高引用但无人解决 = 极难
        if citations > 1000 and attempts == 0:
            return 'very_hard'

        # 有人尝试但失败 = 难
        if attempts > 0:
            return 'hard'

        # 其他
        if citations > 500:
            return 'hard'
        else:
            return 'medium'

    def _suggest_research_direction(self, limitation: str, difficulty: str) -> str:
        """
        建议研究方向

        Args:
            limitation: Limitation描述
            difficulty: 难度等级

        Returns:
            研究方向建议
        """
        limitation_lower = limitation.lower()

        # 基于Limitation类型给出建议
        if 'scalability' in limitation_lower or 'scale' in limitation_lower:
            direction = "考虑分布式方法、近似算法或模型压缩技术"
        elif 'generalization' in limitation_lower or 'generalize' in limitation_lower:
            direction = "探索元学习、域适应或迁移学习方法"
        elif 'efficiency' in limitation_lower or 'computation' in limitation_lower:
            direction = "研究加速算法、模型蒸馏或硬件优化"
        elif 'interpretability' in limitation_lower or 'explainability' in limitation_lower:
            direction = "引入可解释性方法、注意力机制或因果推理"
        elif 'data' in limitation_lower and 'require' in limitation_lower:
            direction = "探索少样本学习、数据增强或无监督方法"
        else:
            direction = "建议从方法论创新或跨领域迁移角度入手"

        difficulty_prefix = {
            'very_hard': "⚠️ 极高难度项目，",
            'hard': "🔥 高难度项目，",
            'medium': "💪 中等难度项目，"
        }

        return f"{difficulty_prefix.get(difficulty, '')} {direction}"

    def _generate_innovative_ideas(self, graph: nx.DiGraph) -> Dict:
        """
        等级三：创新型Idea - 跨域迁移和组合拳（完全重构）

        核心思想：
        利用 Adapts_to 或 Alternative 的传递性进行推理

        模式 A（借尸还魂）：
        1. 找到一个在分支 A 中非常成功（被大量 Extends）的 Method_X
        2. 找到分支 B 中目前面临的一个 Problem_Y（有很多 Limitation 没被解决）
        3. 预测：尝试计算 TextSimilarity(Method_X, Problem_Y's context)
        4. 如果逻辑上可行，将 Method_X 迁移过来解决 Problem_Y 就是一个典型的 Adapts_to 创新

        模式 B（组合拳）：
        1. 如果论文 A 和论文 B 是 Alternative 关系（解决同一问题，方法不同）
        2. 检查 A 的 Limitation 是否正好是 B 的优势，反之亦然
        3. 预测：提出一个 Hybrid Method，结合 A 和 B 的优点，这通常能生成一篇强有力的 Overcomes 论文

        Returns:
            创新型Idea字典，包含cross_domain_transfer和hybrid_methods两种类型
        """
        # 模式A：借尸还魂（跨域迁移）
        cross_domain_ideas = self._generate_cross_domain_transfer_ideas(graph)

        # 模式B：组合拳（混合方法）
        hybrid_ideas = self._generate_hybrid_method_ideas(graph)

        return {
            'cross_domain_transfer': cross_domain_ideas,
            'hybrid_methods': hybrid_ideas,
            'summary': {
                'cross_domain_count': len(cross_domain_ideas),
                'hybrid_count': len(hybrid_ideas),
                'total_ideas': len(cross_domain_ideas) + len(hybrid_ideas)
            }
        }

    def _generate_cross_domain_transfer_ideas(self, graph: nx.DiGraph) -> List[Dict]:
        """
        模式A：借尸还魂 - 生成跨域迁移Idea

        算法步骤：
        1. 找到被大量Extends引用的成功Method（证明方法有效）
        2. 找到有未解决Limitation的Problem
        3. 计算Method与Problem的语义匹配度
        4. 推荐高匹配度的跨域迁移方案

        Returns:
            跨域迁移Idea列表
        """
        ideas = []

        # Step 1: 识别成功的Method（被大量Extends的节点）
        successful_methods = []

        for node in graph.nodes():
            # 统计被Extends引用的次数
            in_extends = sum(
                1 for pred in graph.predecessors(node)
                if graph[pred][node].get('edge_type') == 'Extends'
            )

            if in_extends >= 3:  # 被至少3个Extends引用
                node_data = graph.nodes[node]

                # 从deep_analysis结构中获取method
                deep_analysis = node_data.get('deep_analysis', {})
                method = deep_analysis.get('method', {}).get('content', '')
                if not method:
                    method = node_data.get('title', '')

                successful_methods.append({
                    'id': node,
                    'title': node_data.get('title', ''),
                    'year': node_data.get('year'),
                    'extends_count': in_extends,
                    'method': method,
                    'domain': self._extract_domain_keywords(node_data.get('title', ''))
                })

        # 按Extends数量排序
        successful_methods = sorted(successful_methods, key=lambda x: x['extends_count'], reverse=True)[:10]

        # Step 2: 识别有未解决Limitation的Problem
        unsolved_problems = []

        for node in graph.nodes():
            node_data = graph.nodes[node]

            # 从deep_analysis结构中获取limitation
            deep_analysis = node_data.get('deep_analysis', {})
            limitation = deep_analysis.get('limitation', {}).get('content', '')

            if not limitation or len(limitation) < 20:
                continue

            # 检查是否已被Overcomes解决
            has_overcome = any(
                graph[pred][node].get('edge_type') == 'Overcomes'
                for pred in graph.predecessors(node)
            )

            if not has_overcome:
                # 从deep_analysis获取problem字段
                deep_analysis = node_data.get('deep_analysis', {})
                problem = deep_analysis.get('problem', {}).get('content', '')
                if not problem:
                    problem = limitation  # 如果没有problem，使用limitation

                unsolved_problems.append({
                    'id': node,
                    'title': node_data.get('title', ''),
                    'year': node_data.get('year'),
                    'limitation': limitation,
                    'problem': problem,
                    'domain': self._extract_domain_keywords(node_data.get('title', ''))
                })

        # Step 3: 匹配Method与Problem
        for method in successful_methods[:8]:  # 前8个成功方法
            for problem in unsolved_problems[:15]:  # 前15个未解决问题
                # 避免自己匹配自己
                if method['id'] == problem['id']:
                    continue

                # 避免已有引用关系
                if graph.has_edge(method['id'], problem['id']) or graph.has_edge(problem['id'], method['id']):
                    continue

                # 计算语义相似度
                similarity = self._calculate_text_similarity(
                    method['method'],
                    problem['problem']
                )

                # 过滤低相似度
                if similarity < 0.2:
                    continue

                # 检查是否跨域（增加创新性）
                is_cross_domain = len(set(method['domain']) & set(problem['domain'])) == 0

                # 评估迁移可行性
                feasibility = self._assess_transfer_feasibility(
                    method,
                    problem,
                    similarity,
                    is_cross_domain
                )

                ideas.append({
                    'type': 'cross_domain_transfer',
                    'method_paper': {
                        'id': method['id'],
                        'title': method['title'][:80],
                        'year': method['year'],
                        'extends_count': method['extends_count'],
                        'domain': method['domain'][:2]
                    },
                    'target_paper': {
                        'id': problem['id'],
                        'title': problem['title'][:80],
                        'year': problem['year'],
                        'domain': problem['domain'][:2]
                    },
                    'method_description': method['method'][:200],
                    'target_limitation': problem['limitation'][:200],
                    'similarity_score': similarity,
                    'is_cross_domain': is_cross_domain,
                    'feasibility': feasibility,
                    'innovation_story': self._generate_transfer_story(method, problem, similarity, is_cross_domain)
                })

        # 按可行性和相似度排序
        ideas = sorted(ideas, key=lambda x: x['feasibility'] * x['similarity_score'], reverse=True)[:8]

        return ideas

    def _generate_hybrid_method_ideas(self, graph: nx.DiGraph) -> List[Dict]:
        """
        模式B：组合拳 - 生成混合方法Idea

        算法步骤：
        1. 找到Alternative关系对（解决同一问题，方法不同）
        2. 分析A的Limitation是否是B的优势
        3. 推荐结合两者优点的Hybrid Method

        Returns:
            混合方法Idea列表
        """
        ideas = []

        # 收集所有Alternative关系
        alternative_pairs = []

        for source, target, data in graph.edges(data=True):
            if data.get('edge_type') == 'Alternative':
                source_data = graph.nodes[source]
                target_data = graph.nodes[target]

                # 从deep_analysis获取各字段
                source_deep = source_data.get('deep_analysis', {})
                target_deep = target_data.get('deep_analysis', {})

                alternative_pairs.append({
                    'paper_a': {
                        'id': source,
                        'title': source_data.get('title', ''),
                        'year': source_data.get('year'),
                        'method': source_deep.get('method', {}).get('content', '') or source_data.get('title', ''),
                        'limitation': source_deep.get('limitation', {}).get('content', '')
                    },
                    'paper_b': {
                        'id': target,
                        'title': target_data.get('title', ''),
                        'year': target_data.get('year'),
                        'method': target_deep.get('method', {}).get('content', '') or target_data.get('title', ''),
                        'limitation': target_deep.get('limitation', {}).get('content', '')
                    }
                })

        # 分析每对Alternative关系
        for pair in alternative_pairs[:10]:  # 分析前10对
            paper_a = pair['paper_a']
            paper_b = pair['paper_b']

            # 检查是否有足够的信息
            if not paper_a['limitation'] or not paper_b['method']:
                continue

            # 检查A的Limitation与B的Method的互补性
            complementarity_ab = self._calculate_text_similarity(
                paper_a['limitation'],
                paper_b['method']
            )

            complementarity_ba = self._calculate_text_similarity(
                paper_b.get('limitation', ''),
                paper_a.get('method', '')
            ) if paper_b.get('limitation') and paper_a.get('method') else 0

            # 至少一方面具有互补性
            if complementarity_ab < 0.3 and complementarity_ba < 0.3:
                continue

            # 评估混合方法的可行性
            hybrid_feasibility = self._assess_hybrid_feasibility(
                paper_a,
                paper_b,
                complementarity_ab,
                complementarity_ba
            )

            ideas.append({
                'type': 'hybrid_method',
                'paper_a': {
                    'id': paper_a['id'],
                    'title': paper_a['title'][:80],
                    'year': paper_a['year'],
                    'strength': paper_a.get('method', '')[:150],
                    'weakness': paper_a.get('limitation', '')[:150]
                },
                'paper_b': {
                    'id': paper_b['id'],
                    'title': paper_b['title'][:80],
                    'year': paper_b['year'],
                    'strength': paper_b.get('method', '')[:150],
                    'weakness': paper_b.get('limitation', '')[:150]
                },
                'complementarity_scores': {
                    'a_weakness_vs_b_strength': complementarity_ab,
                    'b_weakness_vs_a_strength': complementarity_ba,
                    'overall': max(complementarity_ab, complementarity_ba)
                },
                'hybrid_feasibility': hybrid_feasibility,
                'hybrid_strategy': self._suggest_hybrid_strategy(paper_a, paper_b, complementarity_ab, complementarity_ba)
            })

        # 按可行性排序
        ideas = sorted(ideas, key=lambda x: x['hybrid_feasibility'], reverse=True)[:6]

        return ideas

    def _assess_transfer_feasibility(
        self,
        method: Dict,
        problem: Dict,
        similarity: float,
        is_cross_domain: bool
    ) -> float:
        """
        评估跨域迁移的可行性

        Args:
            method: 方法信息
            problem: 问题信息
            similarity: 语义相似度
            is_cross_domain: 是否跨域

        Returns:
            可行性分数（0-1）
        """
        feasibility = similarity  # 基础分

        # 成功方法的Extends数量越多，说明方法越成熟
        if method['extends_count'] >= 5:
            feasibility += 0.2
        elif method['extends_count'] >= 3:
            feasibility += 0.1

        # 跨域迁移更有创新性，但可行性略降
        if is_cross_domain:
            feasibility += 0.1  # 创新加分
            feasibility *= 0.9  # 风险折扣

        # 时间差不能太大（方法不能太陈旧）
        year_gap = problem.get('year', 2024) - method.get('year', 2024)
        if year_gap < 0 or year_gap > 10:
            feasibility *= 0.8

        return min(feasibility, 1.0)

    def _assess_hybrid_feasibility(
        self,
        paper_a: Dict,
        paper_b: Dict,
        comp_ab: float,
        comp_ba: float
    ) -> float:
        """
        评估混合方法的可行性

        Args:
            paper_a: 论文A信息
            paper_b: 论文B信息
            comp_ab: A的weakness与B的strength的互补性
            comp_ba: B的weakness与A的strength的互补性

        Returns:
            可行性分数（0-1）
        """
        # 基础分：取两个互补性的最大值
        feasibility = max(comp_ab, comp_ba)

        # 如果双向互补，加分
        if comp_ab > 0.3 and comp_ba > 0.3:
            feasibility += 0.2

        # 时间接近性（同时期的方法更容易结合）
        year_gap = abs(paper_a.get('year', 2024) - paper_b.get('year', 2024))
        if year_gap <= 2:
            feasibility += 0.15
        elif year_gap <= 5:
            feasibility += 0.05

        return min(feasibility, 1.0)

    def _generate_transfer_story(
        self,
        method: Dict,
        problem: Dict,
        similarity: float,
        is_cross_domain: bool
    ) -> str:
        """
        生成跨域迁移的创新故事

        Returns:
            描述文本
        """
        method_domain = ', '.join(method['domain'][:2]) if method['domain'] else '某领域'
        problem_domain = ', '.join(problem['domain'][:2]) if problem['domain'] else '该领域'

        domain_desc = f"从[{method_domain}]迁移到[{problem_domain}]" if is_cross_domain else f"在[{problem_domain}]内部应用"

        return (
            f"💡 创新点：{domain_desc}。"
            f"方法来源于{method['year']}年的成功经验（被{method['extends_count']}个工作Extends），"
            f"可用于解决{problem['year']}年论文中的未解决问题。"
            f"匹配度：{similarity:.0%}"
        )

    def _suggest_hybrid_strategy(
        self,
        paper_a: Dict,
        paper_b: Dict,
        comp_ab: float,
        comp_ba: float
    ) -> str:
        """
        建议混合方法的策略

        Returns:
            策略描述
        """
        if comp_ab > comp_ba:
            dominant = 'B'
            complementary = 'A'
            desc = f"以方法B为主，引入方法A来弥补B的不足（匹配度{comp_ab:.0%}）"
        else:
            dominant = 'A'
            complementary = 'B'
            desc = f"以方法A为主，引入方法B来弥补A的不足（匹配度{comp_ba:.0%}）"

        return f"🔧 混合策略：{desc}。建议在{dominant}的框架下，选择性地集成{complementary}的优势组件。"

    def _log_summary(self, report: Dict) -> None:
        """
        输出分析概要日志（增强版 - 双核心方向）

        Args:
            report: 分析报告
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ 主题发展脉络分析完成（双核心方向）")
        logger.info(f"{'='*80}")

        logger.info(f"\n📊 图谱概览:")
        overview = report.get('graph_overview', {})
        logger.info(f"  • 论文总数: {overview.get('total_papers', 0)}")
        logger.info(f"  • 引用关系: {overview.get('total_citations', 0)}")
        logger.info(f"  • 时间跨度: {overview.get('year_range', 'Unknown')}")

        # ======================== 核心方向1: 回溯脉络 ========================
        logger.info(f"\n{'='*80}")
        logger.info(f"🔙 【核心方向1】回溯脉络分析 (Retrospective Analysis)")
        logger.info(f"{'='*80}")

        retro = report.get('retrospective_analysis', {})

        # 1.1 进化主干 vs 旁支修补
        logger.info(f"\n📍 1.1 进化主干 vs 旁支修补:")
        backbone = retro.get('backbone_vs_incremental', {})
        summary = backbone.get('summary', {})
        logger.info(f"  • 主干路径（Overcomes/Realizes）: {summary.get('backbone_count', 0)} 条")
        logger.info(f"  • 渐进路径（Extends）: {summary.get('incremental_count', 0)} 条")
        logger.info(f"  • 主干/渐进比例: {summary.get('ratio', 0):.2f}")
        logger.info(f"  • 破局点数量: {summary.get('breakthrough_count', 0)} 个\n")

        # 展示破局点
        breakthrough = backbone.get('breakthrough_points', [])
        if breakthrough:
            logger.info(f"  🎯 Top 破局点（从Extends内卷跳出Overcomes）:")
            for i, bp in enumerate(breakthrough[:5], 1):
                logger.info(f"    [{i}] {bp['title']}")
                logger.info(f"        年份: {bp['year']}, 引用数: {bp['cited_by_count']}")
                logger.info(f"        突破模式: {bp['breakthrough_type']}")
                logger.info(f"        输入: {bp['in_extends']}个Extends → 输出: {bp['out_overcomes']}个Overcomes, {bp['out_realizes']}个Realizes")
                logger.info(f"        突破分数: {bp['breakthrough_score']:.1f}\n")

        # 展示主干连续链
        backbone_chains = backbone.get('backbone_chains', [])
        if backbone_chains:
            logger.info(f"  ⛓️  主干连续链（硬核攻坚演化线）:")
            for i, chain in enumerate(backbone_chains[:3], 1):
                logger.info(f"    [链{i}] 长度: {chain['length']}篇, 时间跨度: {chain['year_span']}年")
                for j, paper in enumerate(chain['chain'][:4], 1):  # 只显示前4篇
                    logger.info(f"      {j}. {paper['title']} ({paper['year']})")
                    if j < len(chain['edge_types']):
                        logger.info(f"         └─> [{chain['edge_types'][j-1]}]")
                logger.info("")

        # 展示渐进瓶颈
        bottlenecks = backbone.get('incremental_bottlenecks', [])
        if bottlenecks:
            logger.info(f"  ⚠️  渐进路径瓶颈（内卷终点）:")
            for i, bn in enumerate(bottlenecks[:3], 1):
                logger.info(f"    [{i}] {bn['title']} ({bn['year']})")
                logger.info(f"        {bn['reason']}\n")

        # 1.2 技术分叉口
        logger.info(f"\n🔀 1.2 技术分叉口（Technical Bifurcation）:")
        bifur = retro.get('technical_bifurcations', [])
        if bifur:
            logger.info(f"  发现 {len(bifur)} 个技术路线之争\n")
            for i, b in enumerate(bifur[:3], 1):
                logger.info(f"  [分叉点{i}] {b['parent']['title']} ({b['parent']['year']})")
                logger.info(f"    分歧评分: {b['divergence_score']:.2f}")
                logger.info(f"    竞争分支:")
                for j, branch in enumerate(b['branches'], 1):
                    logger.info(f"      {j}. [{branch['edge_type']}] {branch['title']} ({branch['year']})")
                    logger.info(f"         后续发展: {branch['subtree_size']}个工作, 深度{branch['subtree_depth']}, 状态: {branch['subtree_status']}")
                logger.info(f"    分析: {b['branch_comparison']}\n")
        else:
            logger.info(f"  未发现明显的技术分叉口\n")

        # 1.3 跨界入侵
        logger.info(f"\n🌐 1.3 跨界入侵（Cross-Domain Invasion）:")
        invasions = retro.get('cross_domain_invasions', [])
        if invasions:
            logger.info(f"  发现 {len(invasions)} 个跨域迁移案例（Adapts_to）\n")
            for i, inv in enumerate(invasions[:5], 1):
                impact = inv['impact_analysis']
                logger.info(f"  [入侵{i}] {inv['cross_domain_story']}")
                logger.info(f"    源: {inv['to']['title'][:60]}... ({inv['to']['year']})")
                logger.info(f"    目标: {inv['from']['title'][:60]}... ({inv['from']['year']})")
                logger.info(f"    影响: {impact['descendants_count']}个后续工作, 成功度: {impact['success_level']}\n")
        else:
            logger.info(f"  未发现跨域迁移案例\n")

        # ======================== 核心方向2: 预测未来 ========================
        logger.info(f"\n{'='*80}")
        logger.info(f"🔮 【核心方向2】预测未来 (Future Prediction)")
        logger.info(f"{'='*80}")

        future = report.get('future_prediction', {})

        # 2.1 等级一：捡漏型Idea
        logger.info(f"\n💡 2.1 等级一：捡漏型Idea（Low-Hanging Fruits）")
        logger.info(f"  寻找未被Realized的Future Work\n")
        level1 = future.get('level_1_low_hanging_fruits', [])
        if level1:
            logger.info(f"  发现 {len(level1)} 个现成的研究机会\n")
            for i, idea in enumerate(level1[:5], 1):
                paper = idea['paper']
                logger.info(f"  [Idea{i}] {paper['title']}")
                logger.info(f"    源论文: {paper['year']}年, 引用数: {paper['cited_by_count']}")
                logger.info(f"    难度: {idea['difficulty']}, 可行性: {idea['feasibility_score']:.2f}, 优先级: {idea['priority']:.1f}")
                logger.info(f"    Future Work: {idea['future_work'][:120]}...")
                logger.info(f"    建议: {idea['recommendation']}\n")
        else:
            logger.info(f"  未发现明显的捡漏机会\n")

        # 2.2 等级二：攻坚型Idea
        logger.info(f"\n🔨 2.2 等级二：攻坚型Idea（Hard Nuts）")
        logger.info(f"  寻找未被Overcomes的Limitation\n")
        level2 = future.get('level_2_hard_nuts', [])
        if level2:
            logger.info(f"  发现 {len(level2)} 个高价值攻坚方向\n")
            for i, idea in enumerate(level2[:5], 1):
                paper = idea['paper']
                logger.info(f"  [Idea{i}] {paper['title']}")
                logger.info(f"    源论文: {paper['year']}年, 引用数: {paper['cited_by_count']}")
                logger.info(f"    严重性: {idea['severity']:.2f}, 攻坚难度: {idea['attack_difficulty']}, 潜在影响: {idea['impact_potential']:.1f}")
                logger.info(f"    Limitation: {idea['limitation'][:120]}...")
                logger.info(f"    未解决原因: {idea['unsolved_reason']}")
                logger.info(f"    研究方向: {idea['research_direction']}\n")
        else:
            logger.info(f"  未发现明显的攻坚方向\n")

        # 2.3 等级三：创新型Idea
        logger.info(f"\n🚀 2.3 等级三：创新型Idea（Cross-Pollination & Hybrid Methods）")
        level3 = future.get('level_3_innovative_ideas', {})
        summary3 = level3.get('summary', {})
        logger.info(f"  跨域迁移: {summary3.get('cross_domain_count', 0)} 个")
        logger.info(f"  组合拳: {summary3.get('hybrid_count', 0)} 个")
        logger.info(f"  总计: {summary3.get('total_ideas', 0)} 个创新型Idea\n")

        # 模式A：跨域迁移
        cross_domain = level3.get('cross_domain_transfer', [])
        if cross_domain:
            logger.info(f"  🔄 模式A：跨域迁移（借尸还魂）\n")
            for i, idea in enumerate(cross_domain[:4], 1):
                method = idea['method_paper']
                target = idea['target_paper']
                logger.info(f"    [Idea{i}] {idea['innovation_story']}")
                logger.info(f"      方法来源: {method['title']}")
                logger.info(f"      目标问题: {target['title']}")
                logger.info(f"      可行性: {idea['feasibility']:.2f}, 跨域: {'是' if idea['is_cross_domain'] else '否'}\n")

        # 模式B：组合拳
        hybrid = level3.get('hybrid_methods', [])
        if hybrid:
            logger.info(f"  🥊 模式B：组合拳（混合方法）\n")
            for i, idea in enumerate(hybrid[:4], 1):
                paper_a = idea['paper_a']
                paper_b = idea['paper_b']
                scores = idea['complementarity_scores']
                logger.info(f"    [Idea{i}] 混合方法（A + B）")
                logger.info(f"      方法A: {paper_a['title']}")
                logger.info(f"      方法B: {paper_b['title']}")
                logger.info(f"      互补性: {scores['overall']:.2f}, 可行性: {idea['hybrid_feasibility']:.2f}")
                logger.info(f"      {idea['hybrid_strategy']}\n")

        logger.info(f"\n{'='*80}")
        logger.info(f"分析报告生成完毕")
        logger.info(f"{'='*80}\n")


    def _extract_evolutionary_paths(self, graph: nx.DiGraph, year_stats: Dict) -> List[Dict]:
        """
        提取关键进化路径（Critical Evolutionary Path Extraction）

        基于进化动量权重，找出推动领域进步的主干路径。

        算法思想：
        1. 为每条边赋予"进化动量"权重（基于引用类型）
        2. 在DAG中寻找权重和最大的路径
        3. 这条路径代表了创新的"脊梁"（Backbone of Innovation）

        Args:
            graph: 知识图谱
            year_stats: 年份统计信息

        Returns:
            关键进化路径列表
        """
        try:
            # 1. 验证图是否为DAG（有向无环图）
            if not nx.is_directed_acyclic_graph(graph):
                logger.warning("图中存在环，无法提取进化路径")
                return []

            # 2. 为每条边分配进化动量权重
            weighted_graph = self._create_weighted_graph(graph)

            # 3. 确定时间窗口
            if year_stats:
                years = sorted(year_stats.keys())
                start_year = years[0]
                end_year = years[-1]

                # 如果配置了时间窗口，缩小范围
                if self.evol_time_window_years:
                    start_year = max(start_year, end_year - self.evol_time_window_years)
            else:
                start_year = None
                end_year = None

            # 4. 找出早期和晚期节点
            early_nodes = []
            late_nodes = []

            for node_id in graph.nodes():
                node_year = graph.nodes[node_id].get('year')
                if not node_year:
                    continue

                # 早期节点（起始窗口的前20%）
                if start_year and node_year <= start_year + (end_year - start_year) * 0.2:
                    early_nodes.append(node_id)

                # 晚期节点（结束窗口的后20%）
                if end_year and node_year >= end_year - (end_year - start_year) * 0.2:
                    late_nodes.append(node_id)

            if not early_nodes or not late_nodes:
                logger.warning("时间跨度不足，无法提取进化路径")
                return []

            # 5. 从每个早期节点到每个晚期节点寻找最重路径
            all_paths = []

            for start_node in early_nodes:
                for end_node in late_nodes:
                    if start_node == end_node:
                        continue

                    # 检查是否存在路径
                    if not nx.has_path(weighted_graph, start_node, end_node):
                        continue

                    # 使用Bellman-Ford算法找最长路径（将权重取负）
                    try:
                        # NetworkX的最短路径算法，权重取负即为最长路径
                        path = nx.shortest_path(
                            weighted_graph,
                            start_node,
                            end_node,
                            weight='neg_weight'
                        )

                        # 计算路径的总权重
                        total_weight = 0
                        edges_info = []

                        for i in range(len(path) - 1):
                            u, v = path[i], path[i + 1]
                            edge_data = weighted_graph[u][v]
                            weight = edge_data['weight']
                            edge_type = edge_data.get('edge_type', 'Unknown')

                            total_weight += weight
                            edges_info.append({
                                'from': u,
                                'to': v,
                                'type': edge_type,
                                'weight': weight
                            })

                        # 过滤低权重路径
                        if total_weight < self.evol_min_weight:
                            continue

                        # 收集路径信息
                        path_papers = []
                        path_years = []

                        for node_id in path:
                            node_data = graph.nodes[node_id]
                            path_papers.append({
                                'id': node_id,
                                'title': node_data.get('title', ''),
                                'year': node_data.get('year')
                            })
                            if node_data.get('year'):
                                path_years.append(node_data.get('year'))

                        all_paths.append({
                            'path': path_papers,
                            'edges': edges_info,
                            'total_weight': total_weight,
                            'length': len(path),
                            'year_range': f"{min(path_years)}-{max(path_years)}" if path_years else "Unknown"
                        })

                    except nx.NetworkXNoPath:
                        continue
                    except Exception as e:
                        logger.debug(f"路径计算失败: {e}")
                        continue

            # 6. 按权重排序，取top N
            all_paths = sorted(all_paths, key=lambda x: x['total_weight'], reverse=True)[:self.evol_max_paths]

            # 7. 移除重复或高度重叠的路径
            unique_paths = self._remove_duplicate_paths(all_paths)

            logger.info(f"    发现 {len(unique_paths)} 条关键进化路径")

            return unique_paths

        except Exception as e:
            logger.warning(f"关键进化路径提取失败: {e}")
            return []

    def _create_weighted_graph(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        创建带权重的图

        为每条边分配进化动量权重

        Args:
            graph: 原始图

        Returns:
            带权重的图
        """
        weighted_graph = graph.copy()

        for u, v, data in weighted_graph.edges(data=True):
            edge_type = data.get('edge_type', 'Unknown')
            weight = self.evolutionary_weights.get(edge_type, 0.3)

            # 设置正权重和负权重（用于最长路径算法）
            weighted_graph[u][v]['weight'] = weight
            weighted_graph[u][v]['neg_weight'] = -weight

        return weighted_graph

    def _remove_duplicate_paths(self, paths: List[Dict], overlap_threshold: float = 0.7) -> List[Dict]:
        """
        移除重复或高度重叠的路径

        Args:
            paths: 路径列表
            overlap_threshold: 重叠阈值（节点重叠比例）

        Returns:
            去重后的路径列表
        """
        if len(paths) <= 1:
            return paths

        unique_paths = []

        for i, path1 in enumerate(paths):
            is_duplicate = False
            nodes1 = set([p['id'] for p in path1['path']])

            for path2 in unique_paths:
                nodes2 = set([p['id'] for p in path2['path']])

                # 计算Jaccard相似度
                intersection = len(nodes1 & nodes2)
                union = len(nodes1 | nodes2)

                if union > 0:
                    overlap = intersection / union
                    if overlap >= overlap_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique_paths.append(path1)

        return unique_paths

    def _detect_technical_bifurcations(self, graph: nx.DiGraph) -> List[Dict]:
        """
        检测技术分歧点（Technical Bifurcation Detection）- 增强版

        识别技术发展中的"岔路口"——同一个问题衍生出不同的技术流派

        核心思想：
        1. 寻找被多个后续工作通过Alternative引用的节点
        2. 或者：寻找Problem引发了多个不同Method的节点
        3. 分析这些分叉后的子树规模（哪条路走得更远？哪条路死掉了？）

        算法步骤：
        1. 寻找分叉结构：父节点P被多个子节点引用，且边类型为Alternative/Extends
        2. 验证子节点间无强引用关系（独立发展）
        3. 语义验证：子节点Method不同但Problem相同
        4. 追踪分支后续发展（子树大小分析）

        Args:
            graph: 知识图谱

        Returns:
            技术分歧点列表（包含分支后续发展分析）
        """
        try:
            bifurcations = []

            # 1. 遍历所有节点，寻找潜在的分叉父节点
            for parent_id in graph.nodes():
                # 获取父节点的所有后继节点（被引用的论文）
                successors = list(graph.successors(parent_id))

                if len(successors) < self.bifur_min_children:
                    continue

                # 2. 筛选符合fork edge类型的子节点
                fork_children = []
                for child_id in successors:
                    edge_data = graph[parent_id][child_id]
                    edge_type = edge_data.get('edge_type', 'Unknown')

                    if edge_type in self.bifur_fork_edge_types:
                        fork_children.append(child_id)

                if len(fork_children) < self.bifur_min_children:
                    continue

                # 3. 检测子节点两两之间是否独立（无强引用关系）
                independent_pairs = []

                for i in range(len(fork_children)):
                    for j in range(i + 1, len(fork_children)):
                        child_a = fork_children[i]
                        child_b = fork_children[j]

                        # 检查A→B或B→A是否存在强引用
                        has_strong_link = False

                        if graph.has_edge(child_a, child_b):
                            edge_type = graph[child_a][child_b].get('edge_type', 'Unknown')
                            if edge_type in ['Overcomes', 'Realizes', 'Extends']:
                                has_strong_link = True

                        if graph.has_edge(child_b, child_a):
                            edge_type = graph[child_b][child_a].get('edge_type', 'Unknown')
                            if edge_type in ['Overcomes', 'Realizes', 'Extends']:
                                has_strong_link = True

                        # 如果没有强引用，视为独立分支
                        if not has_strong_link:
                            independent_pairs.append((child_a, child_b))

                if not independent_pairs:
                    continue

                # 4. 语义验证：Method不同，但Problem相同
                parent_data = graph.nodes[parent_id]

                for child_a, child_b in independent_pairs:
                    child_a_data = graph.nodes[child_a]
                    child_b_data = graph.nodes[child_b]

                    # 从deep_analysis提取Method和Problem字段
                    child_a_deep = child_a_data.get('deep_analysis', {})
                    child_b_deep = child_b_data.get('deep_analysis', {})

                    method_a = child_a_deep.get('method', {}).get('content', '') or child_a_data.get('title', '')
                    method_b = child_b_deep.get('method', {}).get('content', '') or child_b_data.get('title', '')
                    problem_a = child_a_deep.get('problem', {}).get('content', '') or child_a_data.get('abstract', '')
                    problem_b = child_b_deep.get('problem', {}).get('content', '') or child_b_data.get('abstract', '')

                    # 计算相似度
                    method_similarity = self._calculate_text_similarity(method_a, method_b)
                    problem_similarity = self._calculate_text_similarity(problem_a, problem_b)

                    # 判定为技术分歧：Method不同但Problem相同
                    if (method_similarity < self.bifur_method_sim_threshold and
                        problem_similarity > self.bifur_problem_sim_threshold):

                        # 5. 追踪分支后续发展（子树分析）
                        branch_a_subtree = self._analyze_branch_subtree(graph, child_a)
                        branch_b_subtree = self._analyze_branch_subtree(graph, child_b)

                        bifurcations.append({
                            'parent': {
                                'id': parent_id,
                                'title': parent_data.get('title', '')[:80],
                                'year': parent_data.get('year'),
                                'cited_by_count': parent_data.get('cited_by_count', 0)
                            },
                            'branches': [
                                {
                                    'id': child_a,
                                    'title': child_a_data.get('title', '')[:80],
                                    'year': child_a_data.get('year'),
                                    'edge_type': graph[parent_id][child_a].get('edge_type', 'Unknown'),
                                    'subtree_size': branch_a_subtree['size'],
                                    'subtree_depth': branch_a_subtree['depth'],
                                    'subtree_status': branch_a_subtree['status']
                                },
                                {
                                    'id': child_b,
                                    'title': child_b_data.get('title', '')[:80],
                                    'year': child_b_data.get('year'),
                                    'edge_type': graph[parent_id][child_b].get('edge_type', 'Unknown'),
                                    'subtree_size': branch_b_subtree['size'],
                                    'subtree_depth': branch_b_subtree['depth'],
                                    'subtree_status': branch_b_subtree['status']
                                }
                            ],
                            'method_similarity': method_similarity,
                            'problem_similarity': problem_similarity,
                            'divergence_score': problem_similarity - method_similarity,  # 分歧评分
                            'branch_comparison': self._compare_branches(branch_a_subtree, branch_b_subtree)
                        })

            # 5. 按分歧评分排序，取top N
            bifurcations = sorted(
                bifurcations,
                key=lambda x: x['divergence_score'],
                reverse=True
            )[:self.bifur_max_bifurcations]

            logger.info(f"    发现 {len(bifurcations)} 个技术分歧点")

            return bifurcations

        except Exception as e:
            logger.warning(f"技术分歧点检测失败: {e}")
            return []

    def _analyze_branch_subtree(self, graph: nx.DiGraph, root_node: str) -> Dict:
        """
        分析分支的子树发展情况

        Args:
            graph: 知识图谱
            root_node: 分支根节点

        Returns:
            子树分析结果
        """
        try:
            # 使用BFS找到所有后代节点
            descendants = list(nx.descendants(graph, root_node))
            subtree_size = len(descendants)

            # 计算最大深度
            max_depth = 0
            if descendants:
                for desc in descendants:
                    if nx.has_path(graph, root_node, desc):
                        try:
                            path = nx.shortest_path(graph, root_node, desc)
                            depth = len(path) - 1
                            max_depth = max(max_depth, depth)
                        except:
                            continue

            # 判断分支状态
            status = 'unknown'
            if subtree_size == 0:
                status = 'dead'  # 死路
            elif subtree_size < 5:
                status = 'weak'  # 弱势发展
            elif subtree_size >= 5 and subtree_size < 15:
                status = 'moderate'  # 中等发展
            else:
                status = 'strong'  # 强势发展

            return {
                'size': subtree_size,
                'depth': max_depth,
                'status': status
            }

        except Exception as e:
            logger.debug(f"分支子树分析失败: {e}")
            return {
                'size': 0,
                'depth': 0,
                'status': 'unknown'
            }

    def _compare_branches(self, branch_a: Dict, branch_b: Dict) -> str:
        """
        比较两个分支的发展情况

        Args:
            branch_a: 分支A的子树分析
            branch_b: 分支B的子树分析

        Returns:
            比较结论
        """
        size_a = branch_a['size']
        size_b = branch_b['size']

        if size_a == 0 and size_b == 0:
            return "两条路线均未获得后续发展"
        elif size_a == 0:
            return f"分支A已死，分支B获得{size_b}个后续工作，成为主流路线"
        elif size_b == 0:
            return f"分支B已死，分支A获得{size_a}个后续工作，成为主流路线"
        elif size_a > size_b * 2:
            return f"分支A强势领先（{size_a} vs {size_b}），成为主流路线"
        elif size_b > size_a * 2:
            return f"分支B强势领先（{size_b} vs {size_a}），成为主流路线"
        else:
            return f"两条路线势均力敌（{size_a} vs {size_b}），技术路线之争仍在继续"

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算两段文本的相似度

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            相似度分数（0-1）
        """
        if not text1 or not text2:
            return 0.0

        try:
            if self.bifur_use_cosine:
                # 使用余弦相似度（需要向量化）
                # 简单实现：使用词袋模型
                words1 = set(re.findall(r'\b\w+\b', text1.lower()))
                words2 = set(re.findall(r'\b\w+\b', text2.lower()))

                # 计算余弦相似度（Jaccard相似度作为近似）
                if not words1 or not words2:
                    return 0.0

                intersection = len(words1 & words2)
                union = len(words1 | words2)

                return intersection / union if union > 0 else 0.0

            else:
                # 使用Jaccard相似度
                words1 = set(re.findall(r'\b\w+\b', text1.lower()))
                words2 = set(re.findall(r'\b\w+\b', text2.lower()))

                if not words1 or not words2:
                    return 0.0

                intersection = len(words1 & words2)
                union = len(words1 | words2)

                return intersection / union if union > 0 else 0.0

        except Exception as e:
            logger.debug(f"文本相似度计算失败: {e}")
            return 0.0

    def _detect_open_frontiers(self, graph: nx.DiGraph, year_stats: Dict) -> Dict:
        """
        检测未闭合前沿（Open Frontier Detection）

        识别未被解决的开放问题并生成跨域迁移研究Idea

        算法步骤：
        1. 筛选边缘节点：最近N年的论文
        2. 缺陷闭环检测：检查Limitation是否被后续工作解决
        3. 跨域匹配：匹配Limitation与其他论文的Contribution

        Args:
            graph: 知识图谱
            year_stats: 年份统计信息

        Returns:
            开放前沿字典，包含open_problems和research_ideas
        """
        try:
            # 1. 筛选边缘节点（最近N年）
            if not year_stats:
                logger.warning("无年份信息，无法筛选边缘节点")
                return {'open_problems': [], 'research_ideas': []}

            years = sorted(year_stats.keys())
            latest_year = years[-1]
            cutoff_year = latest_year - self.frontier_recent_years

            leaf_nodes = []
            for node_id in graph.nodes():
                node_data = graph.nodes[node_id]
                node_year = node_data.get('year')

                if node_year and node_year >= cutoff_year:
                    leaf_nodes.append(node_id)

            if not leaf_nodes:
                logger.warning(f"无最近{self.frontier_recent_years}年的论文")
                return {'open_problems': [], 'research_ideas': []}

            logger.info(f"    筛选出 {len(leaf_nodes)} 个边缘节点（{cutoff_year}-{latest_year}年）")

            # 2. 缺陷闭环检测：找出未被解决的Limitation
            open_problems = []

            for node_id in leaf_nodes:
                node_data = graph.nodes[node_id]

                # 从deep_analysis获取limitation或future_work
                deep_analysis = node_data.get('deep_analysis', {})
                limitation = deep_analysis.get('limitation', {}).get('content', '')
                if not limitation:
                    limitation = deep_analysis.get('future_work', {}).get('content', '')

                if not limitation:
                    continue  # 无Limitation，跳过

                # 检查是否有后续工作通过Overcomes或Realizes解决
                has_closure = False

                # 获取所有引用该节点的论文（后续工作）
                predecessors = list(graph.predecessors(node_id))

                for pred_id in predecessors:
                    edge_data = graph[pred_id][node_id]
                    edge_type = edge_data.get('edge_type', 'Unknown')

                    # 如果有Overcomes或Realizes类型的引用，说明问题已被解决
                    if edge_type in ['Overcomes', 'Realizes']:
                        has_closure = True
                        break

                # 如果未被解决，记录为开放问题
                if not has_closure:
                    open_problems.append({
                        'paper': {
                            'id': node_id,
                            'title': node_data.get('title', ''),
                            'year': node_data.get('year')
                        },
                        'limitation': limitation,
                        'out_degree': graph.out_degree(node_id)  # 有多少后续工作（但没解决）
                    })

            # 按out_degree降序排序（越多后续工作但未解决的越重要）
            open_problems = sorted(
                open_problems,
                key=lambda x: x['out_degree'],
                reverse=True
            )[:self.frontier_max_open_problems]

            logger.info(f"    发现 {len(open_problems)} 个未闭合的开放问题")

            # 3. 跨域匹配：为每个开放问题生成研究Idea
            research_ideas = self._generate_cross_domain_ideas(graph, open_problems)

            logger.info(f"    生成 {len(research_ideas)} 个跨域迁移Idea")

            return {
                'open_problems': open_problems,
                'research_ideas': research_ideas
            }

        except Exception as e:
            logger.warning(f"未闭合前沿探测失败: {e}")
            return {'open_problems': [], 'research_ideas': []}

    def _generate_cross_domain_ideas(self, graph: nx.DiGraph, open_problems: List[Dict]) -> List[Dict]:
        """
        生成跨域迁移研究Idea

        为每个未解决的Limitation找到可能的解决方案（其他论文的Contribution）

        Args:
            graph: 知识图谱
            open_problems: 未闭合问题列表

        Returns:
            研究Idea列表
        """
        research_ideas = []

        try:
            # 为每个开放问题寻找候选解决方案
            for problem in open_problems:
                target_node_id = problem['paper']['id']
                target_limitation = problem['limitation']

                # 候选解决方案列表
                candidate_solutions = []

                # 遍历图中所有其他节点
                for candidate_id in graph.nodes():
                    if candidate_id == target_node_id:
                        continue

                    candidate_data = graph.nodes[candidate_id]

                    # 从deep_analysis获取method
                    candidate_deep = candidate_data.get('deep_analysis', {})
                    method = candidate_deep.get('method', {}).get('content', '')

                    if not method:
                        continue

                    # 检查是否已经有引用关系（避免推荐已有的引用）
                    if graph.has_edge(target_node_id, candidate_id) or graph.has_edge(candidate_id, target_node_id):
                        continue

                    # 计算Limitation与Method的语义相似度
                    similarity = self._calculate_text_similarity(target_limitation, method)

                    # 过滤低相似度的候选
                    if similarity < self.frontier_lim_sim_threshold:
                        continue

                    candidate_solutions.append({
                        'paper': {
                            'id': candidate_id,
                            'title': candidate_data.get('title', ''),
                            'year': candidate_data.get('year')
                        },
                        'method': method,
                        'similarity': similarity
                    })

                # 按相似度排序
                candidate_solutions = sorted(
                    candidate_solutions,
                    key=lambda x: x['similarity'],
                    reverse=True
                )

                # 为该问题生成top N个Idea
                for solution in candidate_solutions[:2]:  # 每个问题最多2个Idea
                    research_ideas.append({
                        'target_paper': problem['paper'],
                        'target_limitation': target_limitation,
                        'solution_paper': solution['paper'],
                        'solution_method': solution['method'],
                        'similarity_score': solution['similarity'],
                        'idea_type': 'cross_domain_transfer'  # 跨域迁移
                    })

            # 按相似度排序，取top N
            research_ideas = sorted(
                research_ideas,
                key=lambda x: x['similarity_score'],
                reverse=True
            )[:self.frontier_max_ideas]

            return research_ideas

        except Exception as e:
            logger.warning(f"生成跨域Idea失败: {e}")
            return []


def create_analyzer(config: Dict = None) -> TopicEvolutionAnalyzer:
    """
    工厂函数：创建分析器实例

    Args:
        config: 配置字典

    Returns:
        TopicEvolutionAnalyzer实例
    """
    return TopicEvolutionAnalyzer(config)


if __name__ == "__main__":
    # 测试代码
    import networkx as nx

    # 创建测试图
    G = nx.DiGraph()

    # 添加测试节点
    papers = [
        {'id': 'p1', 'title': 'Attention Is All You Need', 'year': 2017, 'cited_by_count': 50000},
        {'id': 'p2', 'title': 'BERT: Pre-training of Deep Bidirectional Transformers', 'year': 2018, 'cited_by_count': 30000},
        {'id': 'p3', 'title': 'GPT-3: Language Models are Few-Shot Learners', 'year': 2020, 'cited_by_count': 20000},
        {'id': 'p4', 'title': 'Vision Transformer for Image Recognition', 'year': 2020, 'cited_by_count': 15000},
        {'id': 'p5', 'title': 'Switch Transformers: Scaling to Trillion Parameter Models', 'year': 2021, 'cited_by_count': 5000},
    ]

    for paper in papers:
        G.add_node(paper['id'], **paper)

    # 添加引用关系
    G.add_edge('p2', 'p1', edge_type='Extends')
    G.add_edge('p3', 'p2', edge_type='Overcomes')
    G.add_edge('p4', 'p1', edge_type='Adapts_to')
    G.add_edge('p5', 'p3', edge_type='Extends')

    # 创建分析器
    analyzer = create_analyzer()

    # 执行分析
    report = analyzer.analyze(G, 'Transformer Neural Networks')

    # 输出报告
    print("\n" + "="*60)
    print("分析报告:")
    print("="*60)
    print(f"主题: {report['topic']}")
    print(f"时间跨度: {report['graph_overview']['year_range']}")
    print(f"里程碑论文数: {len(report['milestone_papers'])}")
    print(f"研究分支数: {len(report['research_branches'])}")
