"""
深度综述生成模块
Deep Survey Analyzer

核心方法论：基于关系的图谱剪枝 + 关键演化路径识别

负责基于知识图谱生成深度学术综述，包括：
1. 基于关系的图谱剪枝 (Relation-Based Graph Pruning) - 解决"数据噪音"
   - 保留 Seed Papers
   - 只保留通过强逻辑关系（Overcomes, Realizes, Extends）与 Seed 连通的论文
   - 剔除仅由 Baselines 连接或孤立的论文
2. 关键演化路径识别 (Critical Evolutionary Paths) - 解决"碎片化"
   - 识别线性链条 (The Chain): A -> Overcomes -> B -> Extends -> C
   - 识别分化模式 (The Divergence): Seed -> [Multiple Routes]
   - 识别汇聚模式 (The Convergence): [Multiple Sources] -> Integration Point
   - 为每个演化路径生成叙事单元
3. 结构化 Deep Survey 报告 (Structured Survey Report)
   - Thread 形式展示各个演化故事
   - 配合可视化图和文字解读
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime

try:
    import networkx as nx
except ImportError:
    raise ImportError("需要安装networkx: pip install networkx")

logger = logging.getLogger(__name__)


class DeepSurveyAnalyzer:
    """
    深度综述分析器

    基于知识图谱生成深度学术综述
    采用关系剪枝 + 演化路径识别方法论
    """

    def __init__(self, config: Dict = None):
        """
        初始化深度综述分析器

        Args:
            config: 配置参数字典，支持:
                - llm_config_path: LLM配置文件路径
                - strong_relations: 强逻辑关系类型列表
                - weak_relations: 弱关系类型列表
                - min_chain_length: 最小链条长度
                - max_threads: 最大演化故事数量
                - pruning_mode: 剪枝模式 ('seed_centric' 或 'comprehensive')
                - min_component_size: 最小连通分量大小（用于过滤噪音簇）
        """
        self.config = config or {}

        # 定义强逻辑关系（用于剪枝保留）- 基于Socket Matching的6种关系类型
        self.strong_relations = self.config.get('strong_relations', [
            'Overcomes',   # 攻克/优化（纵向深化）- Match 1: Limitation→Problem
            'Realizes',    # 实现愿景（科研传承）- Match 2: Future_Work→Problem
            'Extends',     # 方法扩展（微创新）- Match 3: Method Extension
            'Alternative', # 另辟蹊径（颠覆创新）- Match 3: Alternative Route
            'Adapts_to'    # 技术迁移（横向扩散）- Match 4: Problem→Problem跨域
        ])

        # 定义弱关系（用于剪枝删除）
        self.weak_relations = self.config.get('weak_relations', [
            'Baselines'    # 基线对比（背景噪音）- 无匹配
        ])

        # 剪枝模式配置
        self.pruning_mode = self.config.get('pruning_mode', 'comprehensive')
        # 'seed_centric': 仅保留与Seed连通的强关系子图（原实现）
        # 'comprehensive': 保留所有强关系连通分量（新实现）

        # 最小连通分量大小（用于过滤噪音簇）
        self.min_component_size = self.config.get('min_component_size', 3)

        # 演化路径探索深度（用于分化和汇聚模式）
        self.exploration_depth = self.config.get('exploration_depth', 5)

        # 存储连通分量元数据（用于后续去重）
        self.strong_components = []

        # 初始化LLM客户端（用于生成叙事文本）
        self.llm_client = None
        llm_config_path = self.config.get('llm_config_path')
        if llm_config_path:
            try:
                from llm_config import LLMClient, LLMConfig
                llm_config = LLMConfig.from_file(llm_config_path)
                self.llm_client = LLMClient(llm_config)
                logger.info("✅ LLM客户端初始化成功")
            except Exception as e:
                logger.warning(f"LLM客户端初始化失败: {e}，将使用模板生成文本")

        logger.info("DeepSurveyAnalyzer 初始化完成")
        logger.info(f"  剪枝模式: {self.pruning_mode}")
        logger.info(f"  强逻辑关系: {self.strong_relations}")
        logger.info(f"  弱关系: {self.weak_relations}")
        logger.info(f"  最小连通分量大小: {self.min_component_size}")
        logger.info(f"  演化路径探索深度: {self.exploration_depth}")

    def analyze(self, graph: nx.DiGraph, topic: str) -> Dict:
        """
        执行深度综述分析

        Args:
            graph: 知识图谱（NetworkX有向图）
            topic: 研究主题

        Returns:
            深度综述分析结果
        """
        logger.info(f"开始深度综述分析: {topic}")
        logger.info(f"  原始图谱: {len(graph.nodes())} 个节点, {len(graph.edges())} 条边")

        if len(graph.nodes()) == 0:
            logger.warning("知识图谱为空，无法生成深度综述")
            return self._empty_result(topic)

        # ========== 第一步：基于关系的图谱剪枝 (Relation-Based Pruning) ==========
        logger.info("\n" + "="*60)
        logger.info("步骤1: 基于关系的图谱剪枝 (Relation-Based Pruning)")
        logger.info("="*60)
        pruned_graph, pruning_stats = self._prune_graph_by_relations(graph)
        logger.info(f"  剪枝后: {len(pruned_graph.nodes())} 个节点 (保留率: {len(pruned_graph.nodes())/len(graph.nodes())*100:.1f}%)")
        logger.info(f"  剪枝后: {len(pruned_graph.edges())} 条边 (保留率: {len(pruned_graph.edges())/len(graph.edges())*100:.1f}%)")

        if len(pruned_graph.nodes()) == 0:
            logger.warning("剪枝后图谱为空，无法生成综述")
            return self._empty_result(topic)

        # ========== 第二步：关键演化路径识别 (Critical Evolutionary Paths) ==========
        logger.info("\n" + "="*60)
        logger.info("步骤2: 关键演化路径识别 (Critical Evolutionary Paths)")
        logger.info("="*60)
        evolutionary_paths = self._identify_evolutionary_paths(pruned_graph)
        logger.info(f"  识别出 {len(evolutionary_paths)} 条演化路径")
        for i, path in enumerate(evolutionary_paths, 1):
            logger.info(f"    Thread {i}: {path['pattern_type']} - {len(path['papers'])} 篇论文")

        # ========== 第三步：生成结构化 Deep Survey 报告 ==========
        logger.info("\n" + "="*60)
        logger.info("步骤3: 生成结构化 Deep Survey 报告")
        logger.info("="*60)
        survey_report = self._generate_survey_report(
            topic=topic,
            pruned_graph=pruned_graph,
            evolutionary_paths=evolutionary_paths,
            pruning_stats=pruning_stats
        )

        result = {
            'topic': topic,
            'timestamp': datetime.now().isoformat(),
            'pruning_stats': pruning_stats,
            'evolutionary_paths': evolutionary_paths,
            'survey_report': survey_report,
            'summary': {
                'original_papers': len(graph.nodes()),
                'pruned_papers': len(pruned_graph.nodes()),
                'total_threads': len(evolutionary_paths),
            }
        }

        logger.info("\n深度综述分析完成 ✅")
        return result

    def _empty_result(self, topic: str) -> Dict:
        """返回空结果"""
        return {
            'topic': topic,
            'timestamp': datetime.now().isoformat(),
            'pruning_stats': {},
            'evolutionary_paths': [],
            'survey_report': {},
            'summary': {
                'original_papers': 0,
                'pruned_papers': 0,
                'total_threads': 0,
            }
        }

    def _prune_graph_by_relations(self, graph: nx.DiGraph) -> Tuple[nx.DiGraph, Dict]:
        """
        第一步：基于关系的图谱剪枝

        支持两种模式：
        - comprehensive: 保留所有强关系连通分量
        - seed_centric: 只保留与Seed连通的论文（原实现）

        Args:
            graph: 原始知识图谱

        Returns:
            (剪枝后的图谱, 统计信息)
        """
        logger.info("  正在执行图谱剪枝...")
        logger.info(f"    剪枝模式: {self.pruning_mode}")

        # 创建新图（保留原图结构）
        pruned_graph = nx.DiGraph()

        # Step 1: 识别所有 Seed Papers（两种模式都需要）
        seed_papers = set()
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            if node_data.get('is_seed', False):
                seed_papers.add(node_id)

        logger.info(f"    识别到 {len(seed_papers)} 个 Seed Papers")

        # Step 2: 根据剪枝模式确定要保留的论文
        if self.pruning_mode == 'comprehensive':
            # 新模式：保留所有强关系连通分量
            papers_to_keep = self._find_all_strong_components(graph)
        else:
            # 原模式：只保留与Seed连通的论文
            if len(seed_papers) == 0:
                logger.warning("    未找到 Seed Papers，将使用前5个论文节点作为种子")
                # 降级策略：选择前5个论文节点作为种子
                all_nodes = list(graph.nodes())
                seed_papers = set(all_nodes[:5])
                logger.info(f"    降级策略：选择了 {len(seed_papers)} 个论文节点作为种子")

            papers_to_keep = self._find_seed_connected_papers(graph, seed_papers)

        # Step 3: 构建剪枝后的子图
        for node_id in papers_to_keep:
            node_data = graph.nodes[node_id]
            pruned_graph.add_node(node_id, **node_data)

        # Step 4: 添加边（只保留强关系边）并统计关系类型分布
        strong_edges = 0
        weak_edges_removed = 0
        relation_type_count = {}  # 统计各种关系类型的数量

        for u, v, edge_data in graph.edges(data=True):
            if u in papers_to_keep and v in papers_to_keep:
                edge_type = edge_data.get('type') or edge_data.get('edge_type', '')

                # 统计关系类型
                if edge_type:
                    relation_type_count[edge_type] = relation_type_count.get(edge_type, 0) + 1

                if edge_type in self.strong_relations or edge_type == '':
                    # 保留强关系边（空类型默认保留）
                    pruned_graph.add_edge(u, v, **edge_data)
                    strong_edges += 1
                else:
                    weak_edges_removed += 1

        # 统计信息
        pruning_stats = {
            'original_papers': len(graph.nodes()),
            'pruned_papers': len(pruned_graph.nodes()),
            'removed_papers': len(graph.nodes()) - len(pruned_graph.nodes()),
            'pruning_mode': self.pruning_mode,

            # 新增：连通分量统计
            'strong_components_count': len(self.strong_components) if hasattr(self, 'strong_components') else 0,
            'components_with_seed': sum(1 for c in getattr(self, 'strong_components', []) if c['has_seed']),
            'largest_component_size': max((c['size'] for c in getattr(self, 'strong_components', [])), default=0),

            # 原有统计
            'seed_papers': len(seed_papers),
            'original_edges': len(graph.edges()),
            'strong_edges': strong_edges,
            'weak_edges_removed': weak_edges_removed,
            'retention_rate': len(pruned_graph.nodes()) / len(graph.nodes()) if len(graph.nodes()) > 0 else 0,
            'relation_type_distribution': relation_type_count
        }

        logger.info(f"    ✅ 剪枝完成:")
        logger.info(f"       - 保留论文: {pruning_stats['pruned_papers']} / {pruning_stats['original_papers']}")
        logger.info(f"       - 剔除论文: {pruning_stats['removed_papers']}")
        logger.info(f"       - 保留强关系边: {strong_edges}")
        logger.info(f"       - 剔除弱关系边: {weak_edges_removed}")

        # 输出连通分量统计（仅comprehensive模式）
        if self.pruning_mode == 'comprehensive' and hasattr(self, 'strong_components'):
            logger.info(f"    📊 强关系连通分量统计:")
            logger.info(f"       - 连通分量总数: {len(self.strong_components)}")
            logger.info(f"       - 包含种子的分量: {sum(1 for c in self.strong_components if c['has_seed'])}")
            logger.info(f"       - 最大分量大小: {max((c['size'] for c in self.strong_components), default=0)}")

            # 列出前5个最大的连通分量
            sorted_components = sorted(self.strong_components, key=lambda x: x['size'], reverse=True)
            for i, comp in enumerate(sorted_components[:5], 1):
                logger.info(f"       - 分量{i}: {comp['size']}篇论文, 总引用{comp['total_citations']}")

        # 输出关系类型分布
        if relation_type_count:
            logger.info(f"    📊 关系类型分布:")
            for rel_type, count in sorted(relation_type_count.items(), key=lambda x: x[1], reverse=True):
                percentage = count / sum(relation_type_count.values()) * 100
                logger.info(f"       - {rel_type}: {count} ({percentage:.1f}%)")

        return pruned_graph, pruning_stats

    def _bfs_strong_relations(
        self,
        graph: nx.DiGraph,
        start_node: str,
        direction: str = 'forward'
    ) -> Set[str]:
        """
        使用 BFS 从起始节点出发，沿强逻辑关系遍历图

        Args:
            graph: 图谱
            start_node: 起始节点
            direction: 'forward' (后继) 或 'backward' (前驱)

        Returns:
            可达节点集合
        """
        visited = set()
        queue = [start_node]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            # 获取邻居节点
            if direction == 'forward':
                neighbors = graph.successors(current)
            else:
                neighbors = graph.predecessors(current)

            for neighbor in neighbors:
                if neighbor in visited:
                    continue

                # 检查边的类型
                if direction == 'forward':
                    edge_data = graph.edges[current, neighbor]
                else:
                    edge_data = graph.edges[neighbor, current]

                edge_type = edge_data.get('type') or edge_data.get('edge_type', '')

                # 只沿强关系边遍历
                if edge_type in self.strong_relations or edge_type == '':
                    queue.append(neighbor)

        return visited

    def _bfs_strong_relations_bidirectional(
        self,
        graph: nx.DiGraph,
        start_node: str
    ) -> Set[str]:
        """
        从起始节点双向BFS，找到所有强关系连通的节点

        Args:
            graph: 图谱
            start_node: 起始节点

        Returns:
            可达节点集合（正向+反向的并集）
        """
        # 正向遍历：找到后继节点
        forward = self._bfs_strong_relations(graph, start_node, 'forward')
        # 反向遍历：找到前驱节点
        backward = self._bfs_strong_relations(graph, start_node, 'backward')

        # 返回两者的并集
        return forward | backward

    def _find_all_strong_components(self, graph: nx.DiGraph) -> Set[str]:
        """
        找到所有强关系连通分量

        策略：
        1. 遍历所有节点，使用双向BFS找到强关系连通分量
        2. 只保留大小 >= min_component_size 的连通分量
        3. 记录每个连通分量的元数据（用于后续去重）

        Args:
            graph: 原始图谱

        Returns:
            所有应保留的论文节点集合
        """
        visited = set()
        papers_to_keep = set()
        self.strong_components = []  # 清空并重新记录

        for node in graph.nodes():
            if node in visited:
                continue

            # 双向BFS找到该节点的强关系连通分量
            component = self._bfs_strong_relations_bidirectional(graph, node)
            visited.update(component)

            # 过滤小簇（可能是噪音）
            if len(component) >= self.min_component_size:
                papers_to_keep.update(component)

                # 记录连通分量元数据
                total_citations = sum(
                    graph.nodes[p].get('cited_by_count', 0)
                    for p in component
                )

                has_seed = any(
                    graph.nodes[p].get('is_seed', False)
                    for p in component
                )

                self.strong_components.append({
                    'papers': component,
                    'size': len(component),
                    'total_citations': total_citations,
                    'has_seed': has_seed
                })

        logger.info(f"    识别到 {len(self.strong_components)} 个强关系连通分量")
        logger.info(f"    保留 {len(papers_to_keep)} 篇论文（连通分量大小 >= {self.min_component_size}）")

        return papers_to_keep

    def _find_seed_connected_papers(
        self,
        graph: nx.DiGraph,
        seed_papers: Set[str]
    ) -> Set[str]:
        """
        原有逻辑：只保留与种子连通的论文（用于 seed_centric 模式）

        Args:
            graph: 图谱
            seed_papers: 种子论文集合

        Returns:
            与种子连通的论文集合
        """
        papers_to_keep = set(seed_papers)

        # 正向遍历：Seed -> 后续论文（通过强关系）
        for seed in seed_papers:
            reachable_forward = self._bfs_strong_relations(
                graph, seed, direction='forward'
            )
            papers_to_keep.update(reachable_forward)

        # 反向遍历：Seed <- 前驱论文（通过强关系）
        for seed in seed_papers:
            reachable_backward = self._bfs_strong_relations(
                graph, seed, direction='backward'
            )
            papers_to_keep.update(reachable_backward)

        logger.info(f"    通过强关系连通性分析，保留 {len(papers_to_keep)} 篇论文")

        return papers_to_keep

    def _lightweight_explore_chains(
        self,
        graph: nx.DiGraph,
        start_node: str,
        scope: Set[str],
        max_depth: int = 5
    ) -> Set[str]:
        """
        轻量级链条探索（用于预评估）

        沿强关系边进行BFS，限制深度

        Args:
            graph: 图谱
            start_node: 起始节点
            scope: 允许的节点范围
            max_depth: 最大探索深度

        Returns:
            可达节点集合
        """
        visited = set([start_node])
        current_layer = {start_node}

        for _ in range(max_depth):
            next_layer = set()
            for node in current_layer:
                # 探索后继节点
                for successor in graph.successors(node):
                    if successor not in scope or successor in visited:
                        continue
                    edge_data = graph.edges[node, successor]
                    edge_type = edge_data.get('type') or edge_data.get('edge_type', '')
                    if edge_type in self.strong_relations or edge_type == '':
                        next_layer.add(successor)
                        visited.add(successor)

            if not next_layer:
                break
            current_layer = next_layer

        return visited

    def _lightweight_explore_divergence(
        self,
        graph: nx.DiGraph,
        center_node: str,
        scope: Set[str],
        max_depth: int = 5
    ) -> Set[str]:
        """
        轻量级分化探索（用于预评估）

        从中心节点向predecessors方向探索

        Args:
            graph: 图谱
            center_node: 中心节点
            scope: 允许的节点范围
            max_depth: 最大探索深度

        Returns:
            可达节点集合
        """
        visited = set([center_node])

        # 获取所有通过强关系引用中心节点的前驱
        predecessors = [
            p for p in graph.predecessors(center_node)
            if p in scope
        ]

        for predecessor in predecessors:
            edge_data = graph.edges[predecessor, center_node]
            edge_type = edge_data.get('type') or edge_data.get('edge_type', '')

            if edge_type not in self.strong_relations:
                continue

            visited.add(predecessor)

            # 继续向后探索（限制深度）
            current = predecessor
            for _ in range(max_depth - 1):
                next_predecessors = [
                    np for np in graph.predecessors(current)
                    if np in scope and np not in visited
                ]

                if not next_predecessors:
                    break

                # 选择强关系前驱
                valid_predecessors = [
                    np for np in next_predecessors
                    if (graph.edges[np, current].get('type') or
                        graph.edges[np, current].get('edge_type', '')) in self.strong_relations
                ]

                if not valid_predecessors:
                    break

                # 选择引用数最高的
                next_node = max(
                    valid_predecessors,
                    key=lambda x: graph.nodes[x].get('cited_by_count', 0)
                )
                visited.add(next_node)
                current = next_node

        return visited

    def _lightweight_explore_convergence(
        self,
        graph: nx.DiGraph,
        center_node: str,
        scope: Set[str],
        max_depth: int = 5
    ) -> Set[str]:
        """
        轻量级汇聚探索（用于预评估）

        从中心节点向successors方向探索

        Args:
            graph: 图谱
            center_node: 中心节点
            scope: 允许的节点范围
            max_depth: 最大探索深度

        Returns:
            可达节点集合
        """
        visited = set([center_node])

        # 获取所有被中心节点通过强关系引用的后继
        successors = [
            s for s in graph.successors(center_node)
            if s in scope
        ]

        for successor in successors:
            edge_data = graph.edges[center_node, successor]
            edge_type = edge_data.get('type') or edge_data.get('edge_type', '')

            if edge_type not in self.strong_relations:
                continue

            visited.add(successor)

            # 继续向后探索（限制深度）
            current = successor
            for _ in range(max_depth - 1):
                next_successors = [
                    ns for ns in graph.successors(current)
                    if ns in scope and ns not in visited
                ]

                if not next_successors:
                    break

                # 选择强关系后继
                valid_successors = [
                    ns for ns in next_successors
                    if (graph.edges[current, ns].get('type') or
                        graph.edges[current, ns].get('edge_type', '')) in self.strong_relations
                ]

                if not valid_successors:
                    break

                # 选择引用数最高的
                next_node = max(
                    valid_successors,
                    key=lambda x: graph.nodes[x].get('cited_by_count', 0)
                )
                visited.add(next_node)
                current = next_node

        return visited

    def _pre_evaluate_node_coverage(
        self,
        graph: nx.DiGraph,
        node_id: str,
        scope: Set[str]
    ) -> int:
        """
        预评估节点能覆盖的论文数

        综合考虑链条、分化、汇聚三个方向

        Args:
            graph: 图谱
            node_id: 候选节点ID
            scope: 允许的节点范围（连通分量）

        Returns:
            预计能覆盖的论文总数
        """
        all_papers = set()

        # 1. 链条方向覆盖
        chain_papers = self._lightweight_explore_chains(
            graph, node_id, scope, self.exploration_depth
        )
        all_papers.update(chain_papers)

        # 2. 分化方向覆盖
        divergence_papers = self._lightweight_explore_divergence(
            graph, node_id, scope, self.exploration_depth
        )
        all_papers.update(divergence_papers)

        # 3. 汇聚方向覆盖
        convergence_papers = self._lightweight_explore_convergence(
            graph, node_id, scope, self.exploration_depth
        )
        all_papers.update(convergence_papers)

        return len(all_papers)

    def _find_key_nodes_in_component(
        self,
        graph: nx.DiGraph,
        component_papers: Set[str]
    ) -> List[str]:
        """
        找到连通分量的关键节点

        优先级：
        1. 种子节点（如果有）
        2. 预评估选择覆盖率最高的前3个节点

        Args:
            graph: 图谱
            component_papers: 连通分量的论文集合

        Returns:
            关键节点ID列表
        """
        key_nodes = []

        # 优先选择种子节点
        seed_nodes = [
            node_id for node_id in component_papers
            if graph.nodes[node_id].get('is_seed', False)
        ]
        key_nodes.extend(seed_nodes)

        # 如果没有种子，使用预评估机制
        if len(key_nodes) == 0:
            component_size = len(component_papers)

            # 性能优化：大连通分量（>50节点）使用降级策略
            if component_size > 50:
                logger.info(f"    连通分量过大({component_size}节点)，使用度中心性降级策略")
                subgraph = graph.subgraph(component_papers)
                centrality = nx.degree_centrality(subgraph)
                top_nodes = sorted(
                    centrality.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                key_nodes = [node_id for node_id, _ in top_nodes]
            else:
                # 预评估策略：计算度中心性前10个候选节点的覆盖率
                subgraph = graph.subgraph(component_papers)
                centrality = nx.degree_centrality(subgraph)
                top_candidates = sorted(
                    centrality.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]  # 候选池：前10个

                # 预评估每个候选节点的论文覆盖数
                candidate_scores = []
                for node_id, _ in top_candidates:
                    coverage = self._pre_evaluate_node_coverage(
                        graph, node_id, component_papers
                    )
                    candidate_scores.append((node_id, coverage))
                    logger.info(f"      预评估节点 {node_id[:20]}...: 预计覆盖 {coverage} 篇论文")

                # 按覆盖数排序，选择前3个
                candidate_scores.sort(key=lambda x: x[1], reverse=True)
                key_nodes = [node_id for node_id, _ in candidate_scores[:3]]

                logger.info(f"    预评估选择完成，选出3个最优节点")

        return key_nodes

    def _find_linear_chains_in_scope(
        self,
        graph: nx.DiGraph,
        start_node: str,
        scope: Set[str]
    ) -> List[List[str]]:
        """
        在指定范围内识别线性链条（避免跨连通分量识别）

        Args:
            graph: 图谱
            start_node: 起始节点
            scope: 允许的节点范围（连通分量内的节点）

        Returns:
            链条列表，每个链条是节点ID列表
        """
        chains = []
        min_chain_length = self.config.get('min_chain_length', 3)

        def dfs_chain(current_path: List[str]):
            """DFS 搜索链条（限制在scope内）"""
            current = current_path[-1]
            successors = list(graph.successors(current))

            if len(successors) == 0:
                # 到达终点
                if len(current_path) >= min_chain_length:
                    chains.append(current_path.copy())
                return

            # 优先沿着强关系继续
            for successor in successors:
                # 关键修改：只考虑scope内的节点
                if successor not in scope:
                    continue
                if successor in current_path:  # 避免环
                    continue

                edge_data = graph.edges[current, successor]
                edge_type = edge_data.get('type') or edge_data.get('edge_type', '')

                if edge_type in self.strong_relations or edge_type == '':
                    current_path.append(successor)
                    dfs_chain(current_path)
                    current_path.pop()

            # 如果没有找到强关系，记录当前链条
            if len(current_path) >= min_chain_length:
                chains.append(current_path.copy())

        dfs_chain([start_node])

        return chains

    def _find_divergence_pattern_in_scope(
        self,
        graph: nx.DiGraph,
        center_node: str,
        scope: Set[str]
    ) -> Optional[Dict]:
        """
        在指定范围内识别分化结构（避免跨连通分量识别）

        分化定义：一个中心节点被多篇后续论文引用/扩展
        方向：中心节点 -> 多条分支（向predecessors探索）

        Args:
            graph: 图谱
            center_node: 中心节点
            scope: 允许的节点范围（连通分量内的节点）

        Returns:
            分化结构字典，包含中心节点和各条路线
        """
        # 使用 predecessors 找引用中心节点的论文（Bug Fix #6）
        predecessors = list(graph.predecessors(center_node))

        # 关键修改：只考虑scope内的前驱节点
        predecessors = [p for p in predecessors if p in scope]

        if len(predecessors) < 2:
            return None

        # 对每个前驱节点，识别其演化路线
        routes = []
        for predecessor in predecessors:
            edge_data = graph.edges[predecessor, center_node]
            edge_type = edge_data.get('type') or edge_data.get('edge_type', '')

            # 只保留强关系路线
            if edge_type not in self.strong_relations:
                continue

            # 初始化路线
            route = {
                'relation_type': edge_type,
                'papers': [predecessor]
            }

            # 继续向后探索（最多exploration_depth层）- 找更新的论文
            current = predecessor
            for _ in range(self.exploration_depth):
                next_predecessors = list(graph.predecessors(current))
                # 关键修改：只考虑scope内的节点
                next_predecessors = [np for np in next_predecessors if np in scope]

                if len(next_predecessors) == 0:
                    break

                # 选择引用数最高的前驱（且是强关系）
                valid_predecessors = []
                for np in next_predecessors:
                    edge_data = graph.edges[np, current]
                    if (edge_data.get('type') or edge_data.get('edge_type', '')) in self.strong_relations:
                        valid_predecessors.append(np)

                if not valid_predecessors:
                    break

                next_node = max(
                    valid_predecessors,
                    key=lambda x: graph.nodes[x].get('cited_by_count', 0)
                )
                if next_node in route['papers']:
                    break
                route['papers'].append(next_node)
                current = next_node

            routes.append(route)

        # 至少有2条有效路线才算分化
        if len(routes) < 2:
            return None

        return {
            'center': center_node,
            'routes': routes
        }

    def _identify_paths_in_component(
        self,
        graph: nx.DiGraph,
        component: Dict
    ) -> List[Dict]:
        """
        为一个连通分量识别线性链条和分化结构

        策略：
        1. 找到连通分量的"关键节点"（高中心性或种子节点）
        2. 从关键节点识别链条和分化
        3. 标记连通分量ID，用于去重

        Args:
            graph: 图谱
            component: 连通分量字典（包含papers, size, total_citations等）

        Returns:
            演化路径列表
        """
        paths = []
        component_papers = component['papers']

        # 找到连通分量的关键节点
        key_nodes = self._find_key_nodes_in_component(graph, component_papers)

        for node_id in key_nodes:
            node_data = graph.nodes[node_id]

            # 识别线性链条（限制在当前连通分量内）
            chains = self._find_linear_chains_in_scope(
                graph,
                node_id,
                component_papers
            )
            for chain in chains:
                path = self._create_chain_narrative(graph, chain, node_data)
                path['component_id'] = id(component)  # 标记所属连通分量
                paths.append(path)

            # 识别分化结构（限制在当前连通分量内）
            divergence = self._find_divergence_pattern_in_scope(
                graph,
                node_id,
                component_papers
            )
            if divergence and len(divergence['routes']) > 1:
                path = self._create_divergence_narrative(graph, divergence, node_data)
                path['component_id'] = id(component)
                paths.append(path)

            # 识别汇聚结构（限制在当前连通分量内）
            convergence = self._find_convergence_pattern_in_scope(
                graph,
                node_id,
                component_papers
            )
            if convergence and len(convergence['routes']) > 1:
                path = self._create_convergence_narrative(graph, convergence, node_data)
                path['component_id'] = id(component)
                paths.append(path)
                logger.info(f"      识别到汇聚结构: {len(convergence['routes'])} 条路线汇聚")

        return paths

    def _calculate_path_priority(self, path: Dict) -> float:
        """
        计算路径优先级

        评分维度：
        1. 论文数量（权重：30%）
        2. 总引用数（权重：30%）
        3. 路径类型多样性（权重：20%）
        4. 关键关系类型（权重：20%）

        Args:
            path: 演化路径

        Returns:
            优先级分数
        """
        # 1. 论文数量（权重：30%）
        paper_count_score = len(path['papers']) * 10

        # 2. 总引用数（权重：30%）
        citation_score = path.get('total_citations', 0) / 100  # 归一化

        # 3. 路径类型多样性（权重：20%）
        diversity_score = 0
        if path['thread_type'] in ['divergence', 'convergence']:
            # 分化/汇聚结构：关系类型越多样，分数越高
            route_types = set(r['relation_type'] for r in path.get('routes', []))
            diversity_score = len(route_types) * 20
            # 特别加分：同时包含 Alternative 和 Extends
            if 'Alternative' in route_types and 'Extends' in route_types:
                diversity_score += 30
        else:
            # 链条结构：长度本身就代表演化深度
            diversity_score = len(path['papers']) * 5

        # 4. 是否包含关键关系（权重：20%）
        key_relation_score = 0
        if path['thread_type'] in ['divergence', 'convergence']:
            route_types = set(r['relation_type'] for r in path.get('routes', []))
            # Overcomes 和 Alternative 是最有价值的关系
            if 'Overcomes' in route_types:
                key_relation_score += 25
            if 'Alternative' in route_types:
                key_relation_score += 25
            if 'Realizes' in route_types:
                key_relation_score += 15

        return paper_count_score + citation_score + diversity_score + key_relation_score

    def _identify_evolutionary_paths(self, graph: nx.DiGraph) -> List[Dict]:
        """
        第二步：关键演化路径识别

        支持两种模式：
        - comprehensive: 为每个连通分量识别路径
        - seed_centric: 基于种子节点识别路径（原实现）

        识别两种核心演化模式：
        1. 线性链条 (The Chain): A -> Overcomes -> B -> Extends -> C
        2. 分化模式 (The Divergence): Seed -> [Multiple Routes]

        Args:
            graph: 剪枝后的图谱

        Returns:
            演化路径列表
        """
        logger.info("  正在识别演化路径...")

        evolutionary_paths = []

        if self.pruning_mode == 'comprehensive':
            # 新策略：为每个连通分量识别路径
            logger.info(f"    comprehensive模式：为 {len(self.strong_components)} 个连通分量识别路径")

            for component in self.strong_components:
                component_paths = self._identify_paths_in_component(
                    graph,
                    component
                )
                evolutionary_paths.extend(component_paths)

        else:
            # 原策略：基于种子节点识别
            # 识别 Seed Papers
            seed_papers = [node_id for node_id in graph.nodes()
                          if graph.nodes[node_id].get('is_seed', False)]

            if len(seed_papers) == 0:
                logger.warning("    未找到 Seed Papers，使用高中心性节点")
                # 降级：使用度中心性
                centrality = nx.degree_centrality(graph)
                top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                seed_papers = [node_id for node_id, _ in top_nodes]

            logger.info(f"    seed_centric模式：基于 {len(seed_papers)} 个种子节点识别演化路径")

            # 为每个 Seed 识别演化路径
            for seed_id in seed_papers:
                seed_data = graph.nodes[seed_id]

                # 模式1: 识别从该 Seed 出发的线性链条
                chains = self._find_linear_chains(graph, seed_id)
                for chain in chains:
                    path = self._create_chain_narrative(graph, chain, seed_data)
                    evolutionary_paths.append(path)

                # 模式2: 识别以该 Seed 为中心的分化结构
                divergence = self._find_divergence_pattern(graph, seed_id)
                if divergence and len(divergence['routes']) > 1:  # 至少有2条分支才算分化
                    path = self._create_divergence_narrative(graph, divergence, seed_data)
                    evolutionary_paths.append(path)

                # 模式3: 识别以该 Seed 为中心的汇聚结构
                convergence = self._find_convergence_pattern(graph, seed_id)
                if convergence and len(convergence['routes']) > 1:  # 至少有2条路线才算汇聚
                    path = self._create_convergence_narrative(graph, convergence, seed_data)
                    evolutionary_paths.append(path)
                    logger.info(f"    Seed {seed_id[:20]}... 形成汇聚结构")

        # 按重要性排序
        evolutionary_paths.sort(key=self._calculate_path_priority, reverse=True)

        # 增强的去重机制（使用新实现）
        evolutionary_paths = self._deduplicate_paths_enhanced(evolutionary_paths)

        # 只保留前N个最重要的故事
        max_threads = self.config.get('max_threads', 5)
        evolutionary_paths = evolutionary_paths[:max_threads]

        logger.info(f"    ✅ 识别完成: {len(evolutionary_paths)} 条演化路径")

        return evolutionary_paths

    def _calculate_dedup_threshold(self, path1: Dict, path2: Dict) -> float:
        """
        动态计算去重阈值

        规则：
        1. 同一连通分量内的路径：高阈值（0.8）
        2. 不同连通分量的路径：中等阈值（0.6）
        3. 不同类型的路径（chain vs star）：低阈值（0.5）

        Args:
            path1: 第一条路径
            path2: 第二条路径

        Returns:
            去重阈值
        """
        base_threshold = self.config.get('path_overlap_threshold', 0.8)

        # 如果来自不同连通分量，降低阈值
        if path1.get('component_id') != path2.get('component_id'):
            base_threshold = 0.6

        # 如果类型不同，进一步降低阈值
        if path1.get('thread_type') != path2.get('thread_type'):
            base_threshold = min(base_threshold, 0.5)

        return base_threshold

    def _are_semantically_different(self, path1: Dict, path2: Dict) -> bool:
        """
        判断两条路径是否语义不同（即使论文重叠）

        考虑因素：
        1. 关系类型差异：Overcomes vs Alternative
        2. 论文角色差异：同一论文在不同路径中的角色
        3. 演化方向差异：时间顺序不同

        Args:
            path1: 第一条路径
            path2: 第二条路径

        Returns:
            True表示语义不同，应保留两者
        """
        # 1. 检查主要关系类型是否不同（分化/汇聚结构）
        if path1.get('thread_type') in ['divergence', 'convergence'] and \
           path2.get('thread_type') in ['divergence', 'convergence']:
            routes1 = set(r['relation_type'] for r in path1.get('routes', []))
            routes2 = set(r['relation_type'] for r in path2.get('routes', []))

            # 如果分化/汇聚结构的关系类型完全不同，认为语义不同
            if len(routes1 & routes2) == 0:
                return True

        # 2. 检查中心节点是否不同（分化/汇聚结构）
        if path1.get('thread_type') in ['divergence', 'convergence'] and \
           path2.get('thread_type') in ['divergence', 'convergence']:
            papers1 = path1['papers']
            papers2 = path2['papers']

            center1 = next((p for p in papers1 if p.get('role') == 'center'), None)
            center2 = next((p for p in papers2 if p.get('role') == 'center'), None)

            if center1 and center2 and center1['paper_id'] != center2['paper_id']:
                return True

        # 3. 检查关系链的差异（链条结构）
        if path1.get('thread_type') == 'chain' and path2.get('thread_type') == 'chain':
            chain1 = path1.get('relation_chain', [])
            chain2 = path2.get('relation_chain', [])

            if len(chain1) > 0 and len(chain2) > 0:
                # 如果关系链的主要关系类型不同，认为语义不同
                types1 = set(r['relation_type'] for r in chain1)
                types2 = set(r['relation_type'] for r in chain2)

                max_len = max(len(types1), len(types2))
                if max_len > 0 and len(types1 & types2) / max_len < 0.5:
                    return True

        return False

    def _deduplicate_paths_enhanced(self, paths: List[Dict]) -> List[Dict]:
        """
        增强的路径去重机制

        多层去重策略：
        1. 基于论文集合的Jaccard相似度（现有机制）
        2. 考虑路径类型和角色差异（新增）
        3. 动态阈值调整（新增）

        Args:
            paths: 演化路径列表（已按优先级排序）

        Returns:
            去重后的路径列表
        """
        if len(paths) <= 1:
            return paths

        deduplicated = []
        removed_count = 0

        for i, path in enumerate(paths):
            path_papers = set(p['paper_id'] for p in path['papers'])

            is_duplicate = False
            for kept_path in deduplicated:
                kept_papers = set(p['paper_id'] for p in kept_path['papers'])

                # 计算Jaccard相似度
                intersection = len(path_papers & kept_papers)
                union = len(path_papers | kept_papers)
                similarity = intersection / union if union > 0 else 0

                # 动态阈值：根据路径特征调整
                threshold = self._calculate_dedup_threshold(path, kept_path)

                if similarity >= threshold:
                    # 进一步检查：是否语义不同但论文重叠
                    if self._are_semantically_different(path, kept_path):
                        # 语义不同，保留两者
                        continue

                    is_duplicate = True
                    removed_count += 1
                    logger.info(
                        f"    去重: Thread #{i+1} 与 Thread #{deduplicated.index(kept_path)+1} "
                        f"重叠度 {similarity:.2%}（阈值{threshold:.2%}），已移除"
                    )
                    break

            if not is_duplicate:
                deduplicated.append(path)

        if removed_count > 0:
            logger.info(f"    去重完成: 移除了 {removed_count} 条重复路径")

        return deduplicated

    def _deduplicate_paths(self, paths: List[Dict]) -> List[Dict]:
        """
        Bug Fix #2: 去重/合并重叠度高的演化路径

        如果两条路径的论文重合度超过阈值，保留更长/更重要的那条

        Args:
            paths: 演化路径列表（已按优先级排序）

        Returns:
            去重后的路径列表
        """
        if len(paths) <= 1:
            return paths

        overlap_threshold = self.config.get('path_overlap_threshold', 0.8)
        deduplicated = []
        removed_count = 0

        for i, path in enumerate(paths):
            # 提取当前路径的论文ID集合
            path_papers = set(p['paper_id'] for p in path['papers'])

            # 检查是否与已保留的路径重叠
            is_duplicate = False
            for kept_path in deduplicated:
                kept_papers = set(p['paper_id'] for p in kept_path['papers'])

                # 计算 Jaccard 相似度
                intersection = len(path_papers & kept_papers)
                union = len(path_papers | kept_papers)

                if union > 0:
                    similarity = intersection / union
                else:
                    similarity = 0

                # 如果重叠度超过阈值，标记为重复
                if similarity >= overlap_threshold:
                    is_duplicate = True
                    removed_count += 1
                    logger.info(f"    去重: Thread #{i+1} 与 Thread #{deduplicated.index(kept_path)+1} 重叠度 {similarity:.2%}，已移除")
                    break

            if not is_duplicate:
                deduplicated.append(path)

        if removed_count > 0:
            logger.info(f"    去重完成: 移除了 {removed_count} 条重复路径")

        return deduplicated

    def _find_linear_chains(self, graph: nx.DiGraph, start_node: str) -> List[List[str]]:
        """
        识别从起始节点出发的线性链条

        链条定义：A -> B -> C，每个节点最多只有一个主要后继

        Args:
            graph: 图谱
            start_node: 起始节点

        Returns:
            链条列表，每个链条是节点ID列表
        """
        chains = []
        min_chain_length = self.config.get('min_chain_length', 3)

        def dfs_chain(current_path: List[str]):
            """DFS 搜索链条"""
            current = current_path[-1]
            successors = list(graph.successors(current))

            if len(successors) == 0:
                # 到达终点
                if len(current_path) >= min_chain_length:
                    chains.append(current_path.copy())
                return

            # 优先沿着强关系继续
            for successor in successors:
                if successor in current_path:  # 避免环
                    continue

                edge_data = graph.edges[current, successor]
                edge_type = edge_data.get('type') or edge_data.get('edge_type', '')

                if edge_type in self.strong_relations or edge_type == '':
                    current_path.append(successor)
                    dfs_chain(current_path)
                    current_path.pop()

            # 如果没有找到强关系，记录当前链条
            if len(current_path) >= min_chain_length:
                chains.append(current_path.copy())

        dfs_chain([start_node])

        return chains

    def _find_divergence_pattern(self, graph: nx.DiGraph, center_node: str) -> Optional[Dict]:
        """
        识别以中心节点为核心的分化结构

        分化定义：一个基础性中心节点被多篇后续论文引用/扩展
        正确方向：寻找 predecessors（前驱），即引用中心节点的论文

        特别关注：Overcomes、Alternative、Extends 等分支

        Args:
            graph: 图谱
            center_node: 中心节点

        Returns:
            分化结构字典，包含中心节点和各条路线
        """
        # Bug Fix #6: 使用 predecessors 而不是 successors
        # predecessors = 引用了中心节点的论文（即后续研究）
        predecessors = list(graph.predecessors(center_node))

        if len(predecessors) < 2:
            return None

        # 对每个前驱节点（引用中心论文的论文），识别其演化路线
        routes = []
        for predecessor in predecessors:
            edge_data = graph.edges[predecessor, center_node]
            edge_type = edge_data.get('type') or edge_data.get('edge_type', '')

            # 只保留强关系路线（过滤掉 Baselines）
            if edge_type not in self.strong_relations:
                continue

            # 沿着这个前驱继续向后搜索，形成一条路线
            route = {
                'relation_type': edge_type,
                'papers': [predecessor]
            }

            # 继续向后探索（最多exploration_depth层）- 找更新的论文
            current = predecessor
            for _ in range(self.exploration_depth):
                # Bug Fix #6: 使用 predecessors 找更新的论文
                next_predecessors = list(graph.predecessors(current))
                if len(next_predecessors) == 0:
                    break
                # 选择引用数最高的前驱（且是强关系）
                valid_predecessors = []
                for np in next_predecessors:
                    edge_data = graph.edges[np, current]
                    if (edge_data.get('type') or edge_data.get('edge_type', '')) in self.strong_relations:
                        valid_predecessors.append(np)

                if not valid_predecessors:
                    break

                next_node = max(
                    valid_predecessors,
                    key=lambda x: graph.nodes[x].get('cited_by_count', 0)
                )
                if next_node in route['papers']:
                    break
                route['papers'].append(next_node)
                current = next_node

            routes.append(route)

        # 至少有2条有效路线才算分化
        if len(routes) < 2:
            return None

        return {
            'center': center_node,
            'routes': routes
        }

    def _find_convergence_pattern_in_scope(
        self,
        graph: nx.DiGraph,
        center_node: str,
        scope: Set[str]
    ) -> Optional[Dict]:
        """
        在指定范围内识别汇聚结构（避免跨连通分量识别）

        汇聚定义：一个综合性中心节点引用/整合了多篇前序论文
        方向：中心节点 <- 多条基础路线（向successors探索）

        Args:
            graph: 图谱
            center_node: 中心节点
            scope: 允许的节点范围（连通分量内的节点）

        Returns:
            汇聚结构字典，包含中心节点和各条路线
        """
        # 使用 successors 找被中心节点引用的论文（即基础工作）
        successors = list(graph.successors(center_node))

        # 关键修改：只考虑scope内的后继节点
        successors = [s for s in successors if s in scope]

        if len(successors) < 2:
            return None

        # 对每个后继节点，识别其基础路线
        routes = []
        for successor in successors:
            edge_data = graph.edges[center_node, successor]
            edge_type = edge_data.get('type') or edge_data.get('edge_type', '')

            # 只保留强关系路线
            if edge_type not in self.strong_relations:
                continue

            # 初始化路线
            route = {
                'relation_type': edge_type,
                'papers': [successor]
            }

            # 继续向后探索（最多exploration_depth层）- 找更早的基础论文
            current = successor
            for _ in range(self.exploration_depth):
                next_successors = list(graph.successors(current))
                # 关键修改：只考虑scope内的节点
                next_successors = [ns for ns in next_successors if ns in scope]

                if len(next_successors) == 0:
                    break

                # 选择引用数最高的后继（且是强关系）
                valid_successors = []
                for ns in next_successors:
                    edge_data = graph.edges[current, ns]
                    if (edge_data.get('type') or edge_data.get('edge_type', '')) in self.strong_relations:
                        valid_successors.append(ns)

                if not valid_successors:
                    break

                next_node = max(
                    valid_successors,
                    key=lambda x: graph.nodes[x].get('cited_by_count', 0)
                )
                if next_node in route['papers']:
                    break
                route['papers'].append(next_node)
                current = next_node

            routes.append(route)

        # 至少有2条有效路线才算汇聚
        if len(routes) < 2:
            return None

        return {
            'center': center_node,
            'routes': routes
        }

    def _find_convergence_pattern(self, graph: nx.DiGraph, center_node: str) -> Optional[Dict]:
        """
        识别以中心节点为核心的汇聚结构

        汇聚定义：一个综合性中心节点整合了多篇前序基础工作
        正确方向：寻找 successors（后继），即被中心节点引用的论文

        Args:
            graph: 图谱
            center_node: 中心节点

        Returns:
            汇聚结构字典，包含中心节点和各条路线
        """
        # 使用 successors 而非 predecessors
        # successors = 被中心节点引用的论文（即基础工作）
        successors = list(graph.successors(center_node))

        if len(successors) < 2:
            return None

        # 对每个后继节点（被中心引用的论文），识别其基础路线
        routes = []
        for successor in successors:
            edge_data = graph.edges[center_node, successor]
            edge_type = edge_data.get('type') or edge_data.get('edge_type', '')

            # 只保留强关系路线（过滤掉 Baselines）
            if edge_type not in self.strong_relations:
                continue

            # 沿着这个后继继续向后搜索，形成一条路线
            route = {
                'relation_type': edge_type,
                'papers': [successor]
            }

            # 继续向后探索（最多exploration_depth层）- 找更早的基础论文
            current = successor
            for _ in range(self.exploration_depth):
                # 使用 successors 找更早的基础论文
                next_successors = list(graph.successors(current))
                if len(next_successors) == 0:
                    break
                # 选择引用数最高的后继（且是强关系）
                valid_successors = []
                for ns in next_successors:
                    edge_data = graph.edges[current, ns]
                    if (edge_data.get('type') or edge_data.get('edge_type', '')) in self.strong_relations:
                        valid_successors.append(ns)

                if not valid_successors:
                    break

                next_node = max(
                    valid_successors,
                    key=lambda x: graph.nodes[x].get('cited_by_count', 0)
                )
                if next_node in route['papers']:
                    break
                route['papers'].append(next_node)
                current = next_node

            routes.append(route)

        # 至少有2条有效路线才算汇聚
        if len(routes) < 2:
            return None

        return {
            'center': center_node,
            'routes': routes
        }

    def _create_chain_narrative(
        self,
        graph: nx.DiGraph,
        chain: List[str],
        seed_data: Dict
    ) -> Dict:
        """
        为线性链条创建叙事单元

        生成模板：
        - 起因：论文 A 提出了 [Method A] 来解决 [Problem]，但在 [Limitation] 方面存在不足。
        - 转折：论文 B 针对这一不足，通过 [Method B] 成功克服了该问题。
        - 发展：随后，论文 C 在 B 的基础上，进一步扩展了其应用场景。

        Args:
            graph: 图谱
            chain: 链条节点列表
            seed_data: 种子节点数据

        Returns:
            叙事单元字典
        """
        # Bug Fix #1: 按时间排序链条（时间倒流问题）
        # 获取每篇论文的年份，按时间正序排序
        chain_with_years = []
        for paper_id in chain:
            node_data = graph.nodes[paper_id]
            year = node_data.get('year', 0)
            chain_with_years.append((paper_id, year))

        # 按年份升序排序
        chain_with_years.sort(key=lambda x: x[1])
        sorted_chain = [paper_id for paper_id, _ in chain_with_years]

        logger.info(f"    按时间排序链条: {[f'{pid}({y})' for pid, y in chain_with_years]}")

        papers_info = []
        total_citations = 0
        relation_chain = []  # 新增：详细的关系链

        for i, paper_id in enumerate(sorted_chain):
            node_data = graph.nodes[paper_id]
            papers_info.append({
                'paper_id': paper_id,
                'title': node_data.get('title', ''),
                'year': node_data.get('year', 0),
                'cited_by_count': node_data.get('cited_by_count', 0)
            })
            total_citations += node_data.get('cited_by_count', 0)

            # 构建关系链：提取每对论文之间的关系
            # Bug Fix #3: 修正关系方向 - 改为发展方向（早->晚）
            # Bug Fix #5: 修正关系语义 - 时间正序时需要反转关系含义
            if i < len(sorted_chain) - 1:
                next_paper_id = sorted_chain[i + 1]
                edge_type = 'Unknown'
                edge_description = 'Unknown'

                # 尝试两个方向查找边
                if graph.has_edge(paper_id, next_paper_id):
                    # 早论文 -> 晚论文（这种情况少见，可能是 Inspires 类型）
                    edge_data = graph.edges[paper_id, next_paper_id]
                    original_type = edge_data.get('type') or edge_data.get('edge_type', 'Temporal_Evolution')
                    edge_type = original_type
                    edge_description = original_type
                elif graph.has_edge(next_paper_id, paper_id):
                    # 晚论文 -> 早论文（常见情况：新论文 Overcomes 旧论文）
                    edge_data = graph.edges[next_paper_id, paper_id]
                    original_type = edge_data.get('type') or edge_data.get('edge_type', 'Unknown')

                    # 关键修复：反转关系语义以符合时间叙事
                    # 原始：新论文(2023) --Overcomes--> 旧论文(2021)
                    # 叙事：旧论文(2021) --Was_Overcome_By--> 新论文(2023)
                    edge_type = original_type
                    edge_description = self._reverse_relation_semantics(original_type)
                else:
                    # 没有直接边，标记为时间演进关系
                    edge_type = 'Temporal_Evolution'
                    edge_description = 'Temporal_Evolution'

                next_node_data = graph.nodes[next_paper_id]
                relation_chain.append({
                    'from_paper': {
                        'id': paper_id,
                        'title': node_data.get('title', ''),
                        'year': node_data.get('year', 0)
                    },
                    'to_paper': {
                        'id': next_paper_id,
                        'title': next_node_data.get('title', ''),
                        'year': next_node_data.get('year', 0)
                    },
                    'relation_type': edge_type,  # 原始关系类型
                    'narrative_relation': edge_description,  # 叙事用的关系描述
                    'direction': 'chronological'  # 明确标记为时间正序
                })

        # 提取关键信息（使用排序后的链条）
        first_paper = graph.nodes[sorted_chain[0]]
        last_paper = graph.nodes[sorted_chain[-1]]

        # 生成标题
        first_method = self._extract_key_method(first_paper)
        last_method = self._extract_key_method(last_paper)

        title = f"从 {first_method} 到 {last_method} 的演进之路"

        # 生成叙事文本（使用LLM或模板，使用排序后的链条）
        narrative = self._generate_chain_narrative_text(graph, sorted_chain)

        return {
            'thread_type': 'chain',
            'pattern_type': 'The Chain (线性链条)',
            'title': title,
            'narrative': narrative,
            'papers': papers_info,
            'total_citations': total_citations,
            'visual_structure': ' -> '.join([f"Paper_{i+1}" for i in range(len(sorted_chain))]),
            'relation_chain': relation_chain  # 新增：详细的关系链
        }

    def _create_divergence_narrative(
        self,
        graph: nx.DiGraph,
        divergence: Dict,
        seed_data: Dict
    ) -> Dict:
        """
        为分化结构创建叙事单元

        Bug Fix #6: 修正分化结构的时间逻辑
        正确叙事：中心论文（早期基础）-> 多条后续演进路线（晚期）

        生成模板：
        - 焦点：中心论文是该领域的基石，但它留下了 [Limitation] 的问题。
        - 分歧：学术界对此产生了不同的演进路线。
        - 对比：(插入各路线的对比)

        Args:
            graph: 图谱
            divergence: 分化结构
            seed_data: 种子节点数据

        Returns:
            叙事单元字典
        """
        center_id = divergence['center']
        center_data = graph.nodes[center_id]
        center_year = center_data.get('year', 0)

        papers_info = [{
            'paper_id': center_id,
            'title': center_data.get('title', ''),
            'year': center_year,
            'cited_by_count': center_data.get('cited_by_count', 0),
            'role': 'center'
        }]

        total_citations = center_data.get('cited_by_count', 0)

        # 收集所有路线的论文和关系链
        routes_info = []
        relation_chain = []  # 新增：详细的关系链

        for route_idx, route in enumerate(divergence['routes']):
            route_papers = []
            for paper_id in route['papers']:
                node_data = graph.nodes[paper_id]
                route_papers.append({
                    'paper_id': paper_id,
                    'title': node_data.get('title', ''),
                    'year': node_data.get('year', 0),
                    'cited_by_count': node_data.get('cited_by_count', 0)
                })
                total_citations += node_data.get('cited_by_count', 0)
                papers_info.append(route_papers[-1])

            routes_info.append({
                'relation_type': route['relation_type'],
                'papers': route_papers
            })

            # Bug Fix #6: 修正关系链方向
            # 现在 route['papers'] 中的论文引用了中心论文
            # 所以关系方向应该是：路线论文 -> 中心论文（图中）
            # 但叙事方向应该是：中心论文 -> 路线论文（时间）
            if route_papers:
                first_paper = route_papers[0]
                first_paper_year = first_paper['year']

                # 检查时间关系，确保叙事正确
                if center_year <= first_paper_year:
                    # 中心论文更早（正常情况）
                    relation_chain.append({
                        'from_paper': {
                            'id': center_id,
                            'title': center_data.get('title', ''),
                            'year': center_year
                        },
                        'to_paper': {
                            'id': first_paper['paper_id'],
                            'title': first_paper['title'],
                            'year': first_paper_year
                        },
                        'relation_type': route['relation_type'],
                        'narrative_relation': self._reverse_relation_semantics(route['relation_type']),
                        'route_id': route_idx + 1,
                        'direction': 'chronological'
                    })
                else:
                    # 异常情况：路线论文更早（记录但标注）
                    logger.warning(f"    分化结构异常: 路线论文 {first_paper['paper_id']} ({first_paper_year}) 早于中心论文 {center_id} ({center_year})")
                    relation_chain.append({
                        'from_paper': {
                            'id': first_paper['paper_id'],
                            'title': first_paper['title'],
                            'year': first_paper_year
                        },
                        'to_paper': {
                            'id': center_id,
                            'title': center_data.get('title', ''),
                            'year': center_year
                        },
                        'relation_type': route['relation_type'],
                        'narrative_relation': route['relation_type'],
                        'route_id': route_idx + 1,
                        'direction': 'reverse_chronological'
                    })

        # 提取核心问题
        seed_problem = self._extract_key_problem(center_data)

        # 生成标题
        title = f"针对 {seed_problem} 的多技术路线博弈"

        # 生成叙事文本
        narrative = self._generate_divergence_narrative_text(graph, divergence, center_data)

        return {
            'thread_type': 'divergence',
            'pattern_type': 'The Divergence (分化模式)',
            'title': title,
            'narrative': narrative,
            'center_paper': center_data.get('title', ''),
            'routes_count': len(routes_info),
            'routes': routes_info,
            'papers': papers_info,
            'total_citations': total_citations,
            'visual_structure': f"Center -> {len(routes_info)} Routes",
            'relation_chain': relation_chain  # 新增：详细的关系链
        }

    def _create_convergence_narrative(
        self,
        graph: nx.DiGraph,
        convergence: Dict,
        seed_data: Dict
    ) -> Dict:
        """
        为汇聚结构创建叙事单元

        汇聚模式：多条基础路线汇聚到一个综合性论文

        生成模板：
        - 背景：多个独立的研究方向分别探索了不同的技术路径
        - 汇聚：中心论文整合了这些方向，形成综合框架
        - 意义：标志着领域理论的系统化整合

        Args:
            graph: 图谱
            convergence: 汇聚结构
            seed_data: 种子节点数据

        Returns:
            叙事单元字典
        """
        center_id = convergence['center']
        center_data = graph.nodes[center_id]
        center_year = center_data.get('year', 0)

        papers_info = [{
            'paper_id': center_id,
            'title': center_data.get('title', ''),
            'year': center_year,
            'cited_by_count': center_data.get('cited_by_count', 0),
            'role': 'center'
        }]

        total_citations = center_data.get('cited_by_count', 0)

        # 收集所有路线的论文和关系链
        routes_info = []
        relation_chain = []

        for route_idx, route in enumerate(convergence['routes']):
            route_papers = []
            for paper_id in route['papers']:
                node_data = graph.nodes[paper_id]
                route_papers.append({
                    'paper_id': paper_id,
                    'title': node_data.get('title', ''),
                    'year': node_data.get('year', 0),
                    'cited_by_count': node_data.get('cited_by_count', 0)
                })
                total_citations += node_data.get('cited_by_count', 0)
                papers_info.append(route_papers[-1])

            routes_info.append({
                'relation_type': route['relation_type'],
                'papers': route_papers
            })

            # 构建关系链：中心论文引用基础路线
            if route_papers:
                first_paper = route_papers[0]
                first_paper_year = first_paper['year']

                # 汇聚结构：中心论文引用基础论文（时间上中心论文应该更晚）
                if center_year >= first_paper_year:
                    # 中心论文更晚（正常情况）
                    relation_chain.append({
                        'from_paper': {
                            'id': first_paper['paper_id'],
                            'title': first_paper['title'],
                            'year': first_paper_year
                        },
                        'to_paper': {
                            'id': center_id,
                            'title': center_data.get('title', ''),
                            'year': center_year
                        },
                        'relation_type': route['relation_type'],
                        'narrative_relation': f"被{center_data.get('title', '')}整合",
                        'route_id': route_idx + 1,
                        'direction': 'chronological'
                    })
                else:
                    # 异常情况：基础论文更晚
                    logger.warning(f"    汇聚结构异常: 基础论文 {first_paper['paper_id']} ({first_paper_year}) 晚于中心论文 {center_id} ({center_year})")
                    relation_chain.append({
                        'from_paper': {
                            'id': center_id,
                            'title': center_data.get('title', ''),
                            'year': center_year
                        },
                        'to_paper': {
                            'id': first_paper['paper_id'],
                            'title': first_paper['title'],
                            'year': first_paper_year
                        },
                        'relation_type': route['relation_type'],
                        'narrative_relation': route['relation_type'],
                        'route_id': route_idx + 1,
                        'direction': 'reverse_chronological'
                    })

        # 提取核心方法
        center_method = self._extract_key_method(center_data)

        # 生成标题
        title = f"多技术路线汇聚到 {center_method}"

        # 生成叙事文本
        narrative = self._generate_convergence_narrative_text(graph, convergence, center_data)

        return {
            'thread_type': 'convergence',
            'pattern_type': 'The Convergence (汇聚模式)',
            'title': title,
            'narrative': narrative,
            'center_paper': center_data.get('title', ''),
            'routes_count': len(routes_info),
            'routes': routes_info,
            'papers': papers_info,
            'total_citations': total_citations,
            'visual_structure': f"{len(routes_info)} Routes -> Center",
            'relation_chain': relation_chain
        }

    def _generate_chain_narrative_text(self, graph: nx.DiGraph, chain: List[str]) -> str:
        """
        生成线性链条的叙事文本

        Args:
            graph: 图谱
            chain: 链条节点列表

        Returns:
            叙事文本
        """
        if self.llm_client:
            return self._generate_chain_narrative_with_llm(graph, chain)
        else:
            return self._generate_chain_narrative_template(graph, chain)

    def _generate_chain_narrative_template(self, graph: nx.DiGraph, chain: List[str]) -> str:
        """
        使用模板生成链条叙事（降级方案）

        改进版本：引用类型感知叙事生成
        Bug Fix #4: 按时间正序叙述（早->晚）
        Bug Fix #5: 使用正确的关系语义
        """
        # 按年份排序（时间正序）
        chain_with_years = [(pid, graph.nodes[pid].get('year', 0)) for pid in chain]
        chain_with_years.sort(key=lambda x: x[1])
        sorted_chain = [pid for pid, _ in chain_with_years]

        narrative_parts = []

        for i, paper_id in enumerate(sorted_chain):
            node_data = graph.nodes[paper_id]
            title = node_data.get('title', 'Unknown')
            year = node_data.get('year', 'N/A')

            if i == 0:
                # 起源：最早的论文（保持原有逻辑）
                method = self._extract_key_method(node_data)
                problem = self._extract_key_problem(node_data)
                limitation = self._extract_key_limitation(node_data)
                narrative_parts.append(
                    f"**起源** ({year}年): 论文《{title}》首次提出了 {method} 来解决 {problem}，"
                    f"开创了这一研究方向。然而，该工作在 {limitation} 方面仍存在局限性。"
                )
            else:
                # 演进和最新进展：使用引用类型感知叙事
                prev_paper_id = sorted_chain[i-1]

                # 获取边的关系类型
                relation_type = self._get_relation_type(graph, prev_paper_id, paper_id)

                # 提取引用类型相关的信息
                info = self._extract_papers_info_for_relation(
                    graph, prev_paper_id, paper_id, relation_type
                )

                # 生成针对性叙事片段
                narrative_fragment = self._generate_relation_narrative_fragment(info)
                narrative_parts.append(narrative_fragment)

        return "\n\n".join(narrative_parts)

    def _generate_chain_narrative_with_llm(self, graph: nx.DiGraph, chain: List[str]) -> str:
        """
        使用LLM生成链条叙事

        改进版本：引用类型感知的Prompt增强
        Bug Fix #4: 按时间正序叙述
        """
        # 按年份排序（时间正序）
        chain_with_years = [(pid, graph.nodes[pid].get('year', 0)) for pid in chain]
        chain_with_years.sort(key=lambda x: x[1])
        sorted_chain = [pid for pid, _ in chain_with_years]

        # 准备上下文
        papers_context = []
        for i, paper_id in enumerate(sorted_chain, 1):
            node_data = graph.nodes[paper_id]
            title = node_data.get('title', '')
            year = node_data.get('year', '')

            # 提取基础信息
            method = self._extract_key_method(node_data)
            problem = self._extract_key_problem(node_data)
            limitation = self._extract_key_limitation(node_data)
            future_work = self._extract_key_future_work(node_data)

            paper_info = f"论文{i}: {title} ({year}年)\n" \
                         f"- 研究问题: {problem}\n" \
                         f"- 研究方法: {method}\n" \
                         f"- 局限性: {limitation}\n" \
                         f"- 未来工作: {future_work}"

            # 如果不是第一篇论文，添加与前一篇的关系信息
            if i > 1:
                prev_paper_id = sorted_chain[i-2]  # i从1开始，所以i-2是前一篇
                relation_type = self._get_relation_type(graph, prev_paper_id, paper_id)
                relation_focus = self._get_relation_focus_hint(relation_type)
                paper_info += f"\n- 与前一篇的关系: {relation_type} ({relation_focus})"

            papers_context.append(paper_info)

        context = "\n\n".join(papers_context)

        prompt = f"""你是一位学术综述专家。请基于以下技术演进链条，生成一段通顺的叙事文本。

**演进链条**（共{len(sorted_chain)}篇论文，按时间正序从早到晚排列）:
{context}

**任务要求**:
请按照以下结构生成叙事文本（3-5段，每段2-3句话）：
1. **起源**: 描述第一篇（最早）论文如何开创了这一方向，解决了什么问题，但留下了什么不足
2. **演进**: 描述中间论文如何针对前人的不足逐步改进（按时间顺序）
3. **最新进展**: 描述最后一篇（最新）论文如何在前人基础上实现了突破

**输出要求**:
- 使用连贯、学术化的中文表达
- 突出论文之间的因果关系和技术演进逻辑
- 按时间正序叙述（从早到晚），明确标注年份
- 每段以 **起源**、**演进**、**最新进展** 等标题开头
"""

        try:
            narrative = self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=600
            )
            return narrative.strip()
        except Exception as e:
            logger.warning(f"LLM生成叙事失败: {e}，使用模板方法")
            return self._generate_chain_narrative_template(graph, sorted_chain)

    def _generate_divergence_narrative_text(
        self,
        graph: nx.DiGraph,
        divergence: Dict,
        center_data: Dict
    ) -> str:
        """
        生成分化结构的叙事文本

        Args:
            graph: 图谱
            divergence: 分化结构
            center_data: 中心节点数据

        Returns:
            叙事文本
        """
        if self.llm_client:
            return self._generate_divergence_narrative_with_llm(graph, divergence, center_data)
        else:
            return self._generate_divergence_narrative_template(graph, divergence, center_data)

    def _generate_divergence_narrative_template(
        self,
        graph: nx.DiGraph,
        divergence: Dict,
        center_data: Dict
    ) -> str:
        """使用模板生成分化叙事（降级方案）

        改进版本：引用类型感知叙事生成
        """
        center_title = center_data.get('title', 'Unknown')
        center_year = center_data.get('year', 'N/A')
        problem = self._extract_key_problem(center_data)
        limitation = self._extract_key_limitation(center_data)

        narrative_parts = []

        # 焦点
        narrative_parts.append(
            f"**焦点**: 论文《{center_title}》({center_year}) 是该领域的基石工作，"
            f"它聚焦于 {problem}，但留下了 {limitation} 的问题。"
        )

        # 分歧 - 根据实际关系类型生成描述（增强版：引用类型感知）
        center_paper_id = divergence['center']
        routes_desc = []
        for i, route in enumerate(divergence['routes'], 1):
            route_papers = route['papers']
            relation = route['relation_type']

            # 获取路线中最新论文（通常是路线的终点）
            latest_paper_id = route_papers[-1] if route_papers else None
            if latest_paper_id:
                # 使用引用类型感知的信息提取
                info = self._extract_papers_info_for_relation(
                    graph, center_paper_id, latest_paper_id, relation
                )

                # 生成针对性的路线描述（简化版，适配分化场景）
                if relation == 'Overcomes':
                    method = info.get('curr_method', '新方法')
                    limitation = info.get('prev_limitation', '某些局限性')
                    desc = f"**路线{i}** (纵向深化): 针对「{limitation}」，通过 {method} 实现突破"
                elif relation == 'Realizes':
                    method = info.get('curr_method', '新方法')
                    future_work = info.get('prev_future_work', '前人设想')
                    desc = f"**路线{i}** (科研传承): 实现「{future_work}」的愿景，采用 {method}"
                elif relation == 'Extends':
                    method = info.get('curr_method', '改进方法')
                    desc = f"**路线{i}** (微创新): 保留核心架构并扩展，采用 {method}"
                elif relation == 'Alternative':
                    method = info.get('curr_method', '替代方法')
                    desc = f"**路线{i}** (颠覆创新): 提出截然不同的替代方案，使用 {method}"
                elif relation == 'Adapts_to':
                    method = info.get('curr_method', '迁移方法')
                    curr_domain = info.get('curr_domain', '新领域')
                    desc = f"**路线{i}** (横向扩散): 将技术迁移到「{curr_domain}」，采用 {method}"
                else:  # Baselines
                    method = info.get('curr_method', '新方法')
                    desc = f"**路线{i}**: 基于中心论文的工作，采用 {method}"

                routes_desc.append(desc)

        narrative_parts.append(
            f"**分歧**: 学术界对此产生了不同的演进路线。" +
            "；".join(routes_desc) + "。"
        )

        # 对比 - 突出不同路线的特点
        route_types = [r['relation_type'] for r in divergence['routes']]
        if 'Alternative' in route_types and 'Extends' in route_types:
            comparison = "这些路线体现了学术研究的多样性：一些选择颠覆式创新，另一些选择渐进式改进"
        elif 'Overcomes' in route_types:
            comparison = "这些路线共同推动了该领域痛点问题的解决，形成了多角度攻克的局面"
        else:
            comparison = "这些路线各有优势，共同推动了领域的多元化发展"

        narrative_parts.append(f"**对比**: {comparison}。（详见各路线论文的性能对比）")

        return "\n\n".join(narrative_parts)

    def _generate_divergence_narrative_with_llm(
        self,
        graph: nx.DiGraph,
        divergence: Dict,
        center_data: Dict
    ) -> str:
        """使用LLM生成分化叙事"""
        center_title = center_data.get('title', '')
        center_year = center_data.get('year', '')
        problem = self._extract_key_problem(center_data)
        limitation = self._extract_key_limitation(center_data)

        # 准备各路线的上下文
        routes_context = []
        for i, route in enumerate(divergence['routes'], 1):
            relation = route['relation_type']
            route_papers = []
            for paper_id in route['papers'][:2]:  # 每条路线最多2篇
                node_data = graph.nodes[paper_id]
                route_papers.append(
                    f"  - {node_data.get('title', '')} ({node_data.get('year', '')}): "
                    f"{self._extract_key_method(node_data)}"
                )

            routes_context.append(
                f"路线{i} ({relation}):\n" + "\n".join(route_papers)
            )

        routes_text = "\n\n".join(routes_context)

        prompt = f"""你是一位学术综述专家。请基于以下分化演进结构，生成一段通顺的叙事文本。

**中心论文**:
{center_title} ({center_year})
- 研究问题: {problem}
- 局限性: {limitation}

**演进路线** (共{len(divergence['routes'])}条路线):
{routes_text}

**任务要求**:
请按照以下结构生成叙事文本（3段，每段2-3句话）：
1. **焦点**: 描述中心论文的地位和遗留问题
2. **分歧**: 描述各条演进路线的不同技术方向
3. **对比**: 总结这些路线的异同和各自优势

**输出要求**:
- 使用连贯、学术化的中文表达
- 突出不同路线的技术差异和创新点
- 每段以 **焦点**、**分歧**、**对比** 等标题开头
"""

        try:
            narrative = self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=600
            )
            return narrative.strip()
        except Exception as e:
            logger.warning(f"LLM生成叙事失败: {e}，使用模板方法")
            return self._generate_divergence_narrative_template(graph, divergence, center_data)

    def _generate_convergence_narrative_text(
        self,
        graph: nx.DiGraph,
        convergence: Dict,
        center_data: Dict
    ) -> str:
        """
        生成汇聚结构的叙事文本

        Args:
            graph: 图谱
            convergence: 汇聚结构
            center_data: 中心节点数据

        Returns:
            叙事文本
        """
        if self.llm_client:
            return self._generate_convergence_narrative_with_llm(graph, convergence, center_data)
        else:
            return self._generate_convergence_narrative_template(graph, convergence, center_data)

    def _generate_convergence_narrative_template(
        self,
        graph: nx.DiGraph,
        convergence: Dict,
        center_data: Dict
    ) -> str:
        """使用模板生成汇聚叙事（降级方案）"""
        center_title = center_data.get('title', 'Unknown')
        center_year = center_data.get('year', 'N/A')
        center_method = self._extract_key_method(center_data)
        problem = self._extract_key_problem(center_data)

        narrative_parts = []

        # 背景：描述多个独立的基础路线
        center_paper_id = convergence['center']
        routes_desc = []
        for i, route in enumerate(convergence['routes'], 1):
            route_papers = route['papers']
            relation = route['relation_type']

            # 获取路线中最新论文（最接近汇聚中心的论文）
            latest_paper_id = route_papers[-1] if route_papers else None
            if latest_paper_id:
                # 使用引用类型感知的信息提取（路线→中心方向）
                info = self._extract_papers_info_for_relation(
                    graph, latest_paper_id, center_paper_id, relation
                )

                # 根据引用类型生成针对性的路线描述
                if relation == 'Overcomes':
                    limitation = info.get('prev_limitation', '某些局限性')
                    method = info.get('curr_method', '新方法')
                    desc = f"**路线{i}** (克服型): 识别了「{limitation}」，为中心论文提供了待解决的问题"
                elif relation == 'Realizes':
                    future_work = info.get('prev_future_work', '某些未来方向')
                    method = info.get('curr_method', '实现方法')
                    desc = f"**路线{i}** (实现型): 提出了「{future_work}」的研究方向，被中心论文实现"
                elif relation == 'Extends':
                    prev_method = info.get('prev_method', '基础方法')
                    curr_method = info.get('curr_method', '扩展方法')
                    desc = f"**路线{i}** (扩展型): 提供了 {prev_method} 的基础，被中心论文扩展为 {curr_method}"
                elif relation == 'Alternative':
                    prev_method = info.get('prev_method', '替代方法')
                    curr_method = info.get('curr_method', '统一方法')
                    desc = f"**路线{i}** (替代型): 提供了 {prev_method} 作为平行方案，被中心论文整合"
                elif relation == 'Adapts_to':
                    prev_domain = info.get('prev_domain', '原始领域')
                    curr_domain = info.get('curr_domain', '新领域')
                    desc = f"**路线{i}** (适配型): 在 {prev_domain} 中的工作，被中心论文适配到 {curr_domain}"
                elif relation == 'Baselines':
                    prev_method = info.get('prev_method', '基准方法')
                    desc = f"**路线{i}** (基准型): 提供了 {prev_method} 作为比较基准"
                else:
                    # 降级处理
                    latest_paper = graph.nodes[latest_paper_id]
                    method = self._extract_key_method(latest_paper)
                    year = latest_paper.get('year', 'N/A')
                    desc = f"路线{i} 在{year}年提出了 {method}"

                routes_desc.append(desc)

        narrative_parts.append(
            f"**背景**: 在{center_year}年之前，该领域存在多个独立的研究方向。" +
            "；".join(routes_desc) + "。这些方向各自为政，缺乏系统性整合。"
        )

        # 汇聚：描述中心论文如何整合这些方向
        narrative_parts.append(
            f"**汇聚**: 论文《{center_title}》({center_year}) 在此背景下应运而生，"
            f"它通过 {center_method} 将这 {len(convergence['routes'])} 条独立路线有机整合，"
            f"形成了统一的技术框架来解决 {problem}。"
        )

        # 意义：总结整合的价值
        integration_value = ""
        if len(convergence['routes']) >= 3:
            integration_value = "标志着该领域从探索阶段迈入系统化阶段"
        else:
            integration_value = "为该领域提供了理论整合的范例"

        narrative_parts.append(
            f"**意义**: 这种多方向汇聚{integration_value}，"
            f"使得原本分散的技术路线得以协同发挥作用，"
            f"推动了领域的理论统一和实践深化。"
        )

        return "\n\n".join(narrative_parts)

    def _generate_convergence_narrative_with_llm(
        self,
        graph: nx.DiGraph,
        convergence: Dict,
        center_data: Dict
    ) -> str:
        """使用LLM生成汇聚叙事"""
        center_title = center_data.get('title', '')
        center_year = center_data.get('year', '')
        center_method = self._extract_key_method(center_data)
        problem = self._extract_key_problem(center_data)

        # 准备各路线的上下文
        routes_context = []
        for i, route in enumerate(convergence['routes'], 1):
            relation = route['relation_type']
            route_papers = []
            for paper_id in route['papers'][:2]:  # 每条路线最多2篇
                node_data = graph.nodes[paper_id]
                route_papers.append(
                    f"  - {node_data.get('title', '')} ({node_data.get('year', '')}): "
                    f"{self._extract_key_method(node_data)}"
                )

            routes_context.append(
                f"路线{i} ({relation}):\n" + "\n".join(route_papers)
            )

        routes_text = "\n\n".join(routes_context)

        prompt = f"""你是一位学术综述专家。请基于以下汇聚演进结构，生成一段通顺的叙事文本。

**中心论文**:
{center_title} ({center_year})
- 研究问题: {problem}
- 整合方法: {center_method}

**基础路线** (共{len(convergence['routes'])}条独立路线被整合):
{routes_text}

**任务要求**:
请按照以下结构生成叙事文本（3段，每段2-3句话）：
1. **背景**: 描述中心论文之前多个独立方向的探索
2. **汇聚**: 描述中心论文如何整合这些方向形成统一框架
3. **意义**: 总结这种整合对领域发展的价值

**输出要求**:
- 使用连贯、学术化的中文表达
- 突出从分散到整合的演进特点
- 强调整合后的协同效应和理论价值
- 每段以 **背景**、**汇聚**、**意义** 等标题开头
"""

        try:
            narrative = self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=600
            )
            return narrative.strip()
        except Exception as e:
            logger.warning(f"LLM生成叙事失败: {e}，使用模板方法")
            return self._generate_convergence_narrative_template(graph, convergence, center_data)

    def _generate_survey_report(
        self,
        topic: str,
        pruned_graph: nx.DiGraph,
        evolutionary_paths: List[Dict],
        pruning_stats: Dict
    ) -> Dict:
        """
        第三步：生成结构化 Deep Survey 报告

        Args:
            topic: 研究主题
            pruned_graph: 剪枝后的图谱
            evolutionary_paths: 演化路径列表
            pruning_stats: 剪枝统计信息

        Returns:
            综述报告字典
        """
        logger.info("  正在生成 Deep Survey 报告...")

        report = {
            'title': f"Deep Survey: {topic}",
            'abstract': self._generate_abstract(topic, evolutionary_paths, pruning_stats),
            'threads': [],
            'metadata': {
                'total_papers_analyzed': pruning_stats['original_papers'],
                'papers_after_pruning': pruning_stats['pruned_papers'],
                'total_threads': len(evolutionary_paths),
                'generation_date': datetime.now().isoformat()
            }
        }

        # 为每个演化路径生成 Thread
        for i, path in enumerate(evolutionary_paths, 1):
            # 提取关系类型统计
            relation_stats = self._extract_relation_stats(path, pruned_graph)

            thread = {
                'thread_id': i,
                'thread_name': f"Thread {i}: {path['pattern_type']}",
                'title': path['title'],
                'pattern_type': path['pattern_type'],
                'thread_type': path.get('thread_type', 'unknown'),  # 新增：线程类型
                'narrative': path['narrative'],
                'papers': path['papers'],
                'total_citations': path.get('total_citations', 0),
                'visual_structure': path.get('visual_structure', ''),
                'relation_stats': relation_stats,  # 关系统计
                'relation_chain': path.get('relation_chain', []),  # 新增：详细关系链
                # 为可视化准备的数据
                'visualization_data': self._prepare_visualization_data(path, pruned_graph)
            }
            report['threads'].append(thread)

        logger.info(f"    ✅ 报告生成完成: {len(report['threads'])} 个 Threads")

        return report

    def _generate_abstract(
        self,
        topic: str,
        evolutionary_paths: List[Dict],
        pruning_stats: Dict
    ) -> str:
        """生成综述摘要"""
        total_papers = pruning_stats['pruned_papers']
        total_threads = len(evolutionary_paths)

        chains = sum(1 for p in evolutionary_paths if p['thread_type'] == 'chain')
        divergences = sum(1 for p in evolutionary_paths if p['thread_type'] == 'divergence')
        convergences = sum(1 for p in evolutionary_paths if p['thread_type'] == 'convergence')

        abstract = (
            f"本综述基于知识图谱分析了 {topic} 领域的演进历程。"
            f"通过关系剪枝，我们从原始图谱中筛选出 {total_papers} 篇高质量论文，"
            f"并识别出 {total_threads} 条关键演化路径。"
            f"其中包括 {chains} 条线性技术链条、{divergences} 个分化结构和 {convergences} 个汇聚结构，"
            f"完整呈现了该领域的技术演进脉络、分化趋势和整合模式。"
        )

        return abstract

    def _prepare_visualization_data(self, path: Dict, graph: nx.DiGraph) -> Dict:
        """
        准备可视化数据

        Bug Fix #3: 确保箭头方向表示"发展指向"（时间正序：早年份->晚年份）

        Args:
            path: 演化路径
            graph: 图谱

        Returns:
            可视化数据字典
        """
        papers = path['papers']

        # 提取节点和边
        nodes = []
        edges = []

        if path['thread_type'] == 'chain':
            # 线性链条 - 按时间排序后可视化
            # 按年份排序论文
            sorted_papers = sorted(papers, key=lambda p: p.get('year', 0))

            for i, paper_info in enumerate(sorted_papers):
                paper_id = paper_info['paper_id']
                node_data = graph.nodes.get(paper_id, {})

                nodes.append({
                    'id': paper_id,
                    'label': f"Paper {i+1}",
                    'title': paper_info.get('title', ''),
                    'year': paper_info.get('year', 0),
                    'citations': paper_info.get('cited_by_count', 0)
                })

                if i > 0:
                    prev_paper_id = sorted_papers[i-1]['paper_id']
                    # Bug Fix #3: 箭头方向 = 时间流向（早->晚）
                    edges.append({
                        'source': prev_paper_id,  # 早年份
                        'target': paper_id,        # 晚年份
                        'type': 'chronological_evolution',
                        'label': f"{sorted_papers[i-1].get('year', '')} → {paper_info.get('year', '')}"
                    })

        elif path['thread_type'] in ['divergence', 'convergence']:
            # 分化/汇聚结构
            # 中心节点
            center_paper = next((p for p in papers if p.get('role') == 'center'), papers[0])
            nodes.append({
                'id': center_paper['paper_id'],
                'label': 'Center',
                'title': center_paper.get('title', ''),
                'year': center_paper.get('year', 0),
                'citations': center_paper.get('cited_by_count', 0),
                'role': 'center'
            })

            # 路线节点
            for route in path.get('routes', []):
                for paper_info in route['papers']:
                    paper_id = paper_info['paper_id']
                    nodes.append({
                        'id': paper_id,
                        'label': paper_info.get('title', '')[:30] + '...',
                        'title': paper_info.get('title', ''),
                        'year': paper_info.get('year', 0),
                        'citations': paper_info.get('cited_by_count', 0)
                    })

                    # 处理第一个节点与中心的连接
                    if paper_info == route['papers'][0]:
                        center_year = center_paper.get('year', 0)
                        target_year = paper_info.get('year', 0)

                        # 根据模式类型和时间关系确定箭头方向
                        if path['thread_type'] == 'divergence':
                            # 分化：中心 -> 路线（中心论文更早）
                            if center_year <= target_year:
                                edges.append({
                                    'source': center_paper['paper_id'],
                                    'target': paper_id,
                                    'type': route.get('relation_type', 'related'),
                                    'direction': 'forward'
                                })
                            else:
                                # 异常情况：反向
                                edges.append({
                                    'source': paper_id,
                                    'target': center_paper['paper_id'],
                                    'type': f"Inspired_{route.get('relation_type', 'related')}",
                                    'direction': 'backward'
                                })
                        else:
                            # 汇聚：路线 -> 中心（中心论文更晚）
                            if center_year >= target_year:
                                edges.append({
                                    'source': paper_id,
                                    'target': center_paper['paper_id'],
                                    'type': route.get('relation_type', 'related'),
                                    'direction': 'forward'
                                })
                            else:
                                # 异常情况：反向
                                edges.append({
                                    'source': center_paper['paper_id'],
                                    'target': paper_id,
                                    'type': f"Extends_{route.get('relation_type', 'related')}",
                                    'direction': 'backward'
                                })

        return {
            'nodes': nodes,
            'edges': edges,
            'layout': 'hierarchical' if path['thread_type'] == 'chain' else 'radial',
            'direction_note': '箭头方向表示时间演进方向（早年份 → 晚年份）',
            'pattern_note': 'divergence=中心扩散, convergence=多源汇聚' if path['thread_type'] in ['divergence', 'convergence'] else ''
        }

    # ========== 辅助方法 ==========

    def _reverse_relation_semantics(self, relation_type: str) -> str:
        """
        反转关系语义以符合时间正序叙事

        在知识图谱中：新论文(2023) --Overcomes--> 旧论文(2021)
        在时间叙事中：旧论文(2021) --Was_Overcome_By--> 新论文(2023)

        Args:
            relation_type: 原始关系类型（从新论文指向旧论文）

        Returns:
            反转后的关系描述（从旧论文指向新论文）
        """
        relation_mapping = {
            'Overcomes': 'Was_Overcome_By',      # 被克服
            'Realizes': 'Inspired',              # 启发了
            'Extends': 'Was_Extended_By',        # 被扩展
            'Alternative': 'Led_To_Alternative', # 导致替代方案
            'Adapts_to': 'Was_Adapted_By',       # 被迁移
            'Baselines': 'Served_As_Baseline',   # 作为基线
        }

        return relation_mapping.get(relation_type, f'Led_To_{relation_type}')

    def _map_reversed_to_original_type(self, reversed_type: str) -> str:
        """
        将反转后的关系类型映射回原始类型（用于叙事生成）

        Args:
            reversed_type: 反转后的关系类型（如 'Was_Overcome_By'）

        Returns:
            原始关系类型（如 'Overcomes'）
        """
        mapping = {
            'Was_Overcome_By': 'Overcomes',
            'Inspired': 'Realizes',
            'Was_Extended_By': 'Extends',
            'Led_To_Alternative': 'Alternative',
            'Was_Adapted_By': 'Adapts_to',
            'Served_As_Baseline': 'Baselines'
        }
        return mapping.get(reversed_type, 'Baselines')

    def _get_relation_type(self, graph: nx.DiGraph, prev_paper_id: str, curr_paper_id: str) -> str:
        """
        获取两篇论文之间的关系类型（处理正向和反向边）

        Args:
            graph: 图谱
            prev_paper_id: 早期论文ID
            curr_paper_id: 晚期论文ID

        Returns:
            关系类型
        """
        if graph.has_edge(prev_paper_id, curr_paper_id):
            edge_data = graph.edges[prev_paper_id, curr_paper_id]
            return edge_data.get('type') or edge_data.get('edge_type', 'Baselines')
        elif graph.has_edge(curr_paper_id, prev_paper_id):
            edge_data = graph.edges[curr_paper_id, prev_paper_id]
            original_type = edge_data.get('type') or edge_data.get('edge_type', 'Baselines')
            reversed_type = self._reverse_relation_semantics(original_type)
            return self._map_reversed_to_original_type(reversed_type)
        else:
            return 'Baselines'

    def _get_relation_focus_hint(self, relation_type: str) -> str:
        """
        获取引用类型的叙事关注点提示（用于LLM Prompt）

        Args:
            relation_type: 引用关系类型

        Returns:
            关注点描述
        """
        hints = {
            'Overcomes': '关注前人的局限性如何被克服',
            'Realizes': '关注前人的愿景如何被实现',
            'Adapts_to': '关注方法如何跨领域迁移',
            'Extends': '关注方法如何被增量改进',
            'Alternative': '关注不同技术范式的对比',
            'Baselines': '作为背景铺垫'
        }
        return hints.get(relation_type, '继承前人工作')

    def _get_relation_description(self, graph: nx.DiGraph, old_paper_id: str, new_paper_id: str) -> str:
        """
        获取两篇论文之间的关系描述（用于叙事）

        Args:
            graph: 图谱
            old_paper_id: 早期论文ID
            new_paper_id: 晚期论文ID

        Returns:
            符合时间叙事的关系描述
        """
        # 检查是否存在边
        if graph.has_edge(old_paper_id, new_paper_id):
            # 早论文 -> 晚论文（正向）
            edge_data = graph.edges[old_paper_id, new_paper_id]
            relation_type = edge_data.get('type') or edge_data.get('edge_type', 'Unknown')
            return self._get_chinese_relation_desc(relation_type, is_reversed=False)

        elif graph.has_edge(new_paper_id, old_paper_id):
            # 晚论文 -> 早论文（需要反转语义）
            edge_data = graph.edges[new_paper_id, old_paper_id]
            relation_type = edge_data.get('type') or edge_data.get('edge_type', 'Unknown')
            return self._get_chinese_relation_desc(relation_type, is_reversed=True)

        else:
            return "在前人工作基础上"

    def _get_chinese_relation_desc(self, relation_type: str, is_reversed: bool) -> str:
        """
        获取关系的中文描述

        Args:
            relation_type: 关系类型
            is_reversed: 是否需要反转语义

        Returns:
            中文描述
        """
        if is_reversed:
            # 反向关系（晚论文 -> 早论文）需要反转语义
            descriptions = {
                'Overcomes': '针对前人的局限性进行了改进',
                'Realizes': '实现了前人提出的愿景',
                'Extends': '扩展了前人的方法',
                'Alternative': '提出了不同于前人的替代方案',
                'Adapts_to': '将前人的技术迁移到新领域',
                'Baselines': '在前人工作的基础上',
            }
        else:
            # 正向关系（早论文 -> 晚论文）
            descriptions = {
                'Inspires': '启发了后续研究',
                'Proposes': '提出了后续研究的方向',
                'Enables': '使后续研究成为可能',
            }

        return descriptions.get(relation_type, '在前人工作基础上')

    def _extract_relation_stats(self, path: Dict, graph: nx.DiGraph) -> Dict:
        """
        提取演化路径中的关系类型统计

        Args:
            path: 演化路径
            graph: 图谱

        Returns:
            关系统计字典
        """
        relation_count = {}

        if path['thread_type'] == 'chain':
            # 链条：统计相邻论文间的关系
            papers = path['papers']
            for i in range(len(papers) - 1):
                paper_id = papers[i]['paper_id']
                next_paper_id = papers[i + 1]['paper_id']

                if graph.has_edge(paper_id, next_paper_id):
                    edge_data = graph.edges[paper_id, next_paper_id]
                    edge_type = edge_data.get('type') or edge_data.get('edge_type', 'Unknown')
                    relation_count[edge_type] = relation_count.get(edge_type, 0) + 1

        elif path['thread_type'] in ['divergence', 'convergence']:
            # 分化/汇聚结构：统计各条路线的关系类型
            for route in path.get('routes', []):
                relation_type = route.get('relation_type', 'Unknown')
                relation_count[relation_type] = relation_count.get(relation_type, 0) + 1

        return {
            'total_relations': sum(relation_count.values()),
            'relation_distribution': relation_count,
            'dominant_relation': max(relation_count.items(), key=lambda x: x[1])[0] if relation_count else 'Unknown'
        }

    def _extract_key_method(self, node_data: Dict) -> str:
        """提取论文的关键方法"""
        # 优先从 deep_analysis 提取
        deep_analysis = node_data.get('deep_analysis', {})
        if isinstance(deep_analysis, dict):
            method = deep_analysis.get('method', {})
            # 兼容两种格式：字典 {content: ...} 或直接字符串
            if isinstance(method, dict):
                method_text = method.get('content', '')
            else:
                method_text = str(method) if method else ''

            if method_text:
                # 只取第一句话，避免太长
                first_sentence = method_text.split('\n')[0].strip()
                return first_sentence[:80] if len(first_sentence) > 80 else first_sentence

        # 从 rag_analysis 提取
        rag_analysis = node_data.get('rag_analysis', {})
        if isinstance(rag_analysis, dict):
            method = rag_analysis.get('method', '')
            if method:
                first_sentence = str(method).split('\n')[0].strip()
                return first_sentence[:80] if len(first_sentence) > 80 else first_sentence

        return '未提取到方法'

    def _extract_key_problem(self, node_data: Dict) -> str:
        """提取论文的关键问题"""
        deep_analysis = node_data.get('deep_analysis', {})
        if isinstance(deep_analysis, dict):
            problem = deep_analysis.get('problem', {})
            # 兼容两种格式：字典 {content: ...} 或直接字符串
            if isinstance(problem, dict):
                problem_text = problem.get('content', '')
            else:
                problem_text = str(problem) if problem else ''

            if problem_text:
                first_sentence = problem_text.split('\n')[0].strip()
                return first_sentence[:80] if len(first_sentence) > 80 else first_sentence

        rag_analysis = node_data.get('rag_analysis', {})
        if isinstance(rag_analysis, dict):
            problem = rag_analysis.get('problem', '')
            if problem:
                first_sentence = str(problem).split('\n')[0].strip()
                return first_sentence[:80] if len(first_sentence) > 80 else first_sentence

        return '未提取到问题'

    def _extract_key_limitation(self, node_data: Dict) -> str:
        """提取论文的关键局限性"""
        deep_analysis = node_data.get('deep_analysis', {})
        if isinstance(deep_analysis, dict):
            limitation = deep_analysis.get('limitation', {})
            # 兼容两种格式：字典 {content: ...} 或直接字符串
            if isinstance(limitation, dict):
                limitation_text = limitation.get('content', '')
            else:
                limitation_text = str(limitation) if limitation else ''

            if limitation_text:
                first_sentence = limitation_text.split('\n')[0].strip()
                return first_sentence[:80] if len(first_sentence) > 80 else first_sentence

        rag_analysis = node_data.get('rag_analysis', {})
        if isinstance(rag_analysis, dict):
            limitation = rag_analysis.get('limitation', '')
            if limitation:
                first_sentence = str(limitation).split('\n')[0].strip()
                return first_sentence[:80] if len(first_sentence) > 80 else first_sentence

        return '未提取到局限性'

    def _extract_key_future_work(self, node_data: Dict) -> str:
        """
        提取论文的未来工作方向（用于Realizes关系）

        Args:
            node_data: 论文节点数据

        Returns:
            未来工作描述（只取第一句，限制80字符）
        """
        # 优先从 deep_analysis 提取
        deep_analysis = node_data.get('deep_analysis', {})
        if isinstance(deep_analysis, dict):
            future_work = deep_analysis.get('future_work', {})
            # 兼容两种格式：字典 {content: ...} 或直接字符串
            if isinstance(future_work, dict):
                future_work_text = future_work.get('content', '')
            else:
                future_work_text = str(future_work) if future_work else ''

            # 过滤掉 "N/A" 或空值
            if future_work_text and future_work_text != "N/A":
                first_sentence = future_work_text.split('。')[0].strip()
                if not first_sentence:
                    first_sentence = future_work_text.split('\n')[0].strip()
                return first_sentence[:80] if len(first_sentence) > 80 else first_sentence

        # 备用：从 rag_analysis 提取
        rag_analysis = node_data.get('rag_analysis', {})
        if isinstance(rag_analysis, dict):
            future_work = rag_analysis.get('future_work', '')
            if future_work and future_work != "N/A":
                first_sentence = str(future_work).split('。')[0].strip()
                if not first_sentence:
                    first_sentence = str(future_work).split('\n')[0].strip()
                return first_sentence[:80] if len(first_sentence) > 80 else first_sentence

        return '未提取到未来工作'

    def _extract_papers_info_for_relation(
        self,
        graph: nx.DiGraph,
        prev_paper_id: str,
        curr_paper_id: str,
        relation_type: str
    ) -> Dict[str, str]:
        """
        根据引用类型提取论文的相关信息

        Args:
            graph: 图谱
            prev_paper_id: 前一篇（早期）论文ID
            curr_paper_id: 当前（晚期）论文ID
            relation_type: 引用关系类型

        Returns:
            包含叙事所需信息的字典
        """
        prev_node = graph.nodes[prev_paper_id]
        curr_node = graph.nodes[curr_paper_id]

        info = {
            'prev_title': prev_node.get('title', 'Unknown'),
            'prev_year': prev_node.get('year', 'N/A'),
            'curr_title': curr_node.get('title', 'Unknown'),
            'curr_year': curr_node.get('year', 'N/A'),
            'relation_type': relation_type
        }

        # 根据引用类型提取不同的要素
        if relation_type == 'Overcomes':
            # Overcomes: A.Limitation → B.Problem + Method
            info['prev_limitation'] = self._extract_key_limitation(prev_node)
            info['curr_problem'] = self._extract_key_problem(curr_node)
            info['curr_method'] = self._extract_key_method(curr_node)

        elif relation_type == 'Realizes':
            # Realizes: A.Future_Work → B.Problem + Method
            info['prev_future_work'] = self._extract_key_future_work(prev_node)
            info['curr_problem'] = self._extract_key_problem(curr_node)
            info['curr_method'] = self._extract_key_method(curr_node)

        elif relation_type == 'Adapts_to':
            # Adapts_to: A.Problem+Method → B.Problem+Method (跨域)
            info['prev_problem'] = self._extract_key_problem(prev_node)
            info['prev_method'] = self._extract_key_method(prev_node)
            info['curr_problem'] = self._extract_key_problem(curr_node)
            info['curr_method'] = self._extract_key_method(curr_node)
            # 尝试从deep_analysis提取领域信息
            prev_deep = prev_node.get('deep_analysis', {})
            curr_deep = curr_node.get('deep_analysis', {})
            info['prev_domain'] = prev_deep.get('domain', '原领域')
            info['curr_domain'] = curr_deep.get('domain', '新领域')

        elif relation_type == 'Extends':
            # Extends: A.Method → B.Method (扩展)
            info['prev_method'] = self._extract_key_method(prev_node)
            info['curr_method'] = self._extract_key_method(curr_node)
            info['curr_problem'] = self._extract_key_problem(curr_node)

        elif relation_type == 'Alternative':
            # Alternative: A.Method → B.Method (不同范式)
            info['prev_method'] = self._extract_key_method(prev_node)
            info['curr_method'] = self._extract_key_method(curr_node)
            info['prev_problem'] = self._extract_key_problem(prev_node)
            info['curr_problem'] = self._extract_key_problem(curr_node)

        else:  # Baselines 或其他
            # 默认提取通用信息
            info['prev_method'] = self._extract_key_method(prev_node)
            info['curr_method'] = self._extract_key_method(curr_node)
            info['curr_problem'] = self._extract_key_problem(curr_node)

        return info

    def _generate_relation_narrative_fragment(self, info: Dict[str, str]) -> str:
        """
        根据引用类型生成针对性的叙事片段

        Args:
            info: 通过 _extract_papers_info_for_relation 提取的信息字典

        Returns:
            叙事片段文本
        """
        relation_type = info.get('relation_type', 'Baselines')
        curr_year = info.get('curr_year', 'N/A')
        curr_title = info.get('curr_title', 'Unknown')

        if relation_type == 'Overcomes':
            # 关注：A的局限性 → B如何解决
            prev_limitation = info.get('prev_limitation', '某些局限性')
            curr_method = info.get('curr_method', '新方法')
            return (
                f"**克服局限** ({curr_year}年): 针对前人工作在「{prev_limitation}」方面的不足，"
                f"论文《{curr_title}》通过 {curr_method} 实现了突破性改进。"
            )

        elif relation_type == 'Realizes':
            # 关注：A的愿景 → B如何实现
            prev_future_work = info.get('prev_future_work', '前人的设想')
            curr_method = info.get('curr_method', '新方法')
            return (
                f"**实现愿景** ({curr_year}年): 呼应前人「{prev_future_work}」的展望，"
                f"论文《{curr_title}》通过 {curr_method} 将这一设想付诸实践。"
            )

        elif relation_type == 'Adapts_to':
            # 关注：A的领域 → B的领域
            prev_domain = info.get('prev_domain', '原领域')
            curr_domain = info.get('curr_domain', '新领域')
            curr_method = info.get('curr_method', '改进方法')
            return (
                f"**跨域迁移** ({curr_year}年): 论文《{curr_title}》将前人在「{prev_domain}」的技术"
                f"成功迁移到「{curr_domain}」，通过 {curr_method} 验证了方法的泛化能力。"
            )

        elif relation_type == 'Extends':
            # 关注：A的方法 → B如何增强
            prev_method = info.get('prev_method', '基础方法')
            curr_method = info.get('curr_method', '改进方法')
            return (
                f"**增量扩展** ({curr_year}年): 论文《{curr_title}》在「{prev_method}」的基础上，"
                f"通过 {curr_method} 实现了渐进式改进。"
            )

        elif relation_type == 'Alternative':
            # 关注：A的范式 → B的不同范式
            prev_method = info.get('prev_method', '原有方法')
            curr_method = info.get('curr_method', '替代方法')
            return (
                f"**另辟蹊径** ({curr_year}年): 不同于前人「{prev_method}」的思路，"
                f"论文《{curr_title}》提出了 {curr_method}，探索了解决问题的新范式。"
            )

        else:  # Baselines
            # 轻量级描述
            curr_method = info.get('curr_method', '新方法')
            return (
                f"**演进** ({curr_year}年): 论文《{curr_title}》在前人工作的基础上，"
                f"通过 {curr_method} 推进了该方向的发展。"
            )
