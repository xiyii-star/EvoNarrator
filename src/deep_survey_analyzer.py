"""
Deep Survey Generation Module
Deep Survey Analyzer

Core Methodology: Relation-Based Graph Pruning + Critical Evolutionary Path Identification

Responsible for generating deep academic surveys based on knowledge graphs, including:
1. Relation-Based Graph Pruning - Solving "Data Noise"
   - Retain Seed Papers
   - Only retain papers connected to Seeds through strong logical relations (Overcomes, Realizes, Extends)
   - Remove papers only connected by Baselines or isolated papers
2. Critical Evolutionary Path Identification - Solving "Fragmentation"
   - Identify linear chains (The Chain): A -> Overcomes -> B -> Extends -> C
   - Identify divergence patterns (The Divergence): Seed -> [Multiple Routes]
   - Identify convergence patterns (The Convergence): [Multiple Sources] -> Integration Point
   - Generate narrative units for each evolutionary path
3. Structured Deep Survey Report (Structured Survey Report)
   - Display evolutionary stories in Thread format
   - Combine visualization graphs with textual interpretation
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime

try:
    import networkx as nx
except ImportError:
    raise ImportError("Need to install networkx: pip install networkx")

logger = logging.getLogger(__name__)


class DeepSurveyAnalyzer:
    """
    Deep Survey Analyzer

    Generate deep academic surveys based on knowledge graphs
    Using relation pruning + evolutionary path identification methodology
    """

    def __init__(self, config: Dict = None):
        """
        Initialize Deep Survey Analyzer

        Args:
            config: Configuration parameter dictionary, supports:
                - llm_config_path: LLM configuration file path
                - strong_relations: List of strong logical relation types
                - weak_relations: List of weak relation types
                - min_chain_length: Minimum chain length
                - max_threads: Maximum number of evolutionary stories
                - pruning_mode: Pruning mode ('seed_centric' or 'comprehensive')
                - min_component_size: Minimum connected component size (for filtering noise clusters)
        """
        self.config = config or {}

        # Define strong logical relations (for pruning retention) - 6 relation types based on Socket Matching
        self.strong_relations = self.config.get('strong_relations', [
            'Overcomes',   # Overcome/Optimize (Vertical Deepening) - Match 1: Limitation→Problem
            'Realizes',    # Realize Vision (Research Inheritance) - Match 2: Future_Work→Problem
            'Extends',     # Method Extension (Incremental Innovation) - Match 3: Method Extension
            'Alternative', # Alternative Route (Disruptive Innovation) - Match 3: Alternative Route
            'Adapts_to'    # Technology Transfer (Horizontal Diffusion) - Match 4: Problem→Problem Cross-domain
        ])

        # Define weak relations (for pruning removal)
        self.weak_relations = self.config.get('weak_relations', [
            'Baselines'    # Baseline Comparison (Background Noise) - No Match
        ])

        # Pruning mode configuration
        self.pruning_mode = self.config.get('pruning_mode', 'comprehensive')
        # 'seed_centric': Only retain strong relation subgraph connected to Seeds (original implementation)
        # 'comprehensive': Retain all strong relation connected components (new implementation)

        # Minimum connected component size (for filtering noise clusters)
        self.min_component_size = self.config.get('min_component_size', 3)

        # Evolutionary path exploration depth (for divergence and convergence patterns)
        self.exploration_depth = self.config.get('exploration_depth', 5)

        # Store connected component metadata (for subsequent deduplication)
        self.strong_components = []

        # Initialize LLM client (for generating narrative text)
        self.llm_client = None
        llm_config_path = self.config.get('llm_config_path')
        if llm_config_path:
            try:
                from llm_config import LLMClient, LLMConfig
                llm_config = LLMConfig.from_file(llm_config_path)
                self.llm_client = LLMClient(llm_config)
                logger.info("LLM client initialized successfully")
            except Exception as e:
                logger.warning(f"LLM client initialization failed: {e}, will use template to generate text")

        logger.info("DeepSurveyAnalyzer initialization complete")
        logger.info(f"  Pruning mode: {self.pruning_mode}")
        logger.info(f"  Strong logical relations: {self.strong_relations}")
        logger.info(f"  Weak relations: {self.weak_relations}")
        logger.info(f"  Minimum connected component size: {self.min_component_size}")
        logger.info(f"  Evolutionary path exploration depth: {self.exploration_depth}")

    def analyze(self, graph: nx.DiGraph, topic: str) -> Dict:
        """
        Execute deep survey analysis

        Args:
            graph: Knowledge graph (NetworkX directed graph)
            topic: Research topic

        Returns:
            Deep survey analysis results
        """
        logger.info(f"Starting deep survey analysis: {topic}")
        logger.info(f"  Original graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")

        if len(graph.nodes()) == 0:
            logger.warning("Knowledge graph is empty, cannot generate deep survey")
            return self._empty_result(topic)

        # ========== Step 1: Relation-Based Graph Pruning ==========
        logger.info("\n" + "="*60)
        logger.info("Step 1: Relation-Based Graph Pruning")
        logger.info("="*60)
        pruned_graph, pruning_stats = self._prune_graph_by_relations(graph)
        logger.info(f"  After pruning: {len(pruned_graph.nodes())} nodes (retention rate: {len(pruned_graph.nodes())/len(graph.nodes())*100:.1f}%)")
        logger.info(f"  After pruning: {len(pruned_graph.edges())} edges (retention rate: {len(pruned_graph.edges())/len(graph.edges())*100:.1f}%)")

        if len(pruned_graph.nodes()) == 0:
            logger.warning("Graph is empty after pruning, cannot generate survey")
            return self._empty_result(topic)

        # ========== Step 2: Critical Evolutionary Path Identification ==========
        logger.info("\n" + "="*60)
        logger.info("Step 2: Critical Evolutionary Path Identification")
        logger.info("="*60)
        evolutionary_paths = self._identify_evolutionary_paths(pruned_graph)
        logger.info(f"  Identified {len(evolutionary_paths)} evolutionary paths")
        for i, path in enumerate(evolutionary_paths, 1):
            logger.info(f"    Thread {i}: {path['pattern_type']} - {len(path['papers'])} papers")

        # ========== Step 3: Generate Structured Deep Survey Report ==========
        logger.info("\n" + "="*60)
        logger.info("Step 3: Generate Structured Deep Survey Report")
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

        logger.info("\nDeep survey analysis complete")
        return result

    def _empty_result(self, topic: str) -> Dict:
        """Return empty result"""
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
        Step 1: Relation-Based Graph Pruning

        Supports two modes:
        - comprehensive: Retain all strong relation connected components
        - seed_centric: Only retain papers connected to Seeds (original implementation)

        Args:
            graph: Original knowledge graph

        Returns:
            (Pruned graph, statistics)
        """
        logger.info("  Executing graph pruning...")
        logger.info(f"    Pruning mode: {self.pruning_mode}")

        # Create new graph (preserve original graph structure)
        pruned_graph = nx.DiGraph()

        # Step 1: Identify all Seed Papers (needed for both modes)
        seed_papers = set()
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            if node_data.get('is_seed', False):
                seed_papers.add(node_id)

        logger.info(f"    Identified {len(seed_papers)} Seed Papers")

        # Step 2: Determine papers to retain based on pruning mode
        if self.pruning_mode == 'comprehensive':
            # New mode: Retain all strong relation connected components
            papers_to_keep = self._find_all_strong_components(graph)
        else:
            # Original mode: Only retain papers connected to Seeds
            if len(seed_papers) == 0:
                logger.warning("    No Seed Papers found, will use top 5 paper nodes as seeds")
                # Fallback strategy: Select top 5 paper nodes as seeds
                all_nodes = list(graph.nodes())
                seed_papers = set(all_nodes[:5])
                logger.info(f"    Fallback strategy: Selected {len(seed_papers)} paper nodes as seeds")

            papers_to_keep = self._find_seed_connected_papers(graph, seed_papers)

        # Step 3: Build pruned subgraph
        for node_id in papers_to_keep:
            node_data = graph.nodes[node_id]
            pruned_graph.add_node(node_id, **node_data)

        # Step 4: Add edges (only retain strong relation edges) and count relation type distribution
        strong_edges = 0
        weak_edges_removed = 0
        relation_type_count = {}  # Count the number of each relation type

        for u, v, edge_data in graph.edges(data=True):
            if u in papers_to_keep and v in papers_to_keep:
                edge_type = edge_data.get('type') or edge_data.get('edge_type', '')

                # Count relation types
                if edge_type:
                    relation_type_count[edge_type] = relation_type_count.get(edge_type, 0) + 1

                if edge_type in self.strong_relations or edge_type == '':
                    # Retain strong relation edges (empty type retained by default)
                    pruned_graph.add_edge(u, v, **edge_data)
                    strong_edges += 1
                else:
                    weak_edges_removed += 1

        # Statistics
        pruning_stats = {
            'original_papers': len(graph.nodes()),
            'pruned_papers': len(pruned_graph.nodes()),
            'removed_papers': len(graph.nodes()) - len(pruned_graph.nodes()),
            'pruning_mode': self.pruning_mode,

            # New: Connected component statistics
            'strong_components_count': len(self.strong_components) if hasattr(self, 'strong_components') else 0,
            'components_with_seed': sum(1 for c in getattr(self, 'strong_components', []) if c['has_seed']),
            'largest_component_size': max((c['size'] for c in getattr(self, 'strong_components', [])), default=0),

            # Original statistics
            'seed_papers': len(seed_papers),
            'original_edges': len(graph.edges()),
            'strong_edges': strong_edges,
            'weak_edges_removed': weak_edges_removed,
            'retention_rate': len(pruned_graph.nodes()) / len(graph.nodes()) if len(graph.nodes()) > 0 else 0,
            'relation_type_distribution': relation_type_count
        }

        logger.info(f"    Pruning complete:")
        logger.info(f"       - Retained papers: {pruning_stats['pruned_papers']} / {pruning_stats['original_papers']}")
        logger.info(f"       - Removed papers: {pruning_stats['removed_papers']}")
        logger.info(f"       - Retained strong relation edges: {strong_edges}")
        logger.info(f"       - Removed weak relation edges: {weak_edges_removed}")

        # Output connected component statistics (comprehensive mode only)
        if self.pruning_mode == 'comprehensive' and hasattr(self, 'strong_components'):
            logger.info(f"    Strong relation connected component statistics:")
            logger.info(f"       - Total connected components: {len(self.strong_components)}")
            logger.info(f"       - Components with seeds: {sum(1 for c in self.strong_components if c['has_seed'])}")
            logger.info(f"       - Largest component size: {max((c['size'] for c in self.strong_components), default=0)}")

            # List top 5 largest connected components
            sorted_components = sorted(self.strong_components, key=lambda x: x['size'], reverse=True)
            for i, comp in enumerate(sorted_components[:5], 1):
                logger.info(f"       - Component {i}: {comp['size']} papers, total citations {comp['total_citations']}")

        # Output relation type distribution
        if relation_type_count:
            logger.info(f"    Relation type distribution:")
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
        Use BFS to traverse graph from start node along strong logical relations

        Args:
            graph: Graph
            start_node: Start node
            direction: 'forward' (successors) or 'backward' (predecessors)

        Returns:
            Set of reachable nodes
        """
        visited = set()
        queue = [start_node]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            # Get neighbor nodes
            if direction == 'forward':
                neighbors = graph.successors(current)
            else:
                neighbors = graph.predecessors(current)

            for neighbor in neighbors:
                if neighbor in visited:
                    continue

                # Check edge type
                if direction == 'forward':
                    edge_data = graph.edges[current, neighbor]
                else:
                    edge_data = graph.edges[neighbor, current]

                edge_type = edge_data.get('type') or edge_data.get('edge_type', '')

                # Only traverse along strong relation edges
                if edge_type in self.strong_relations or edge_type == '':
                    queue.append(neighbor)

        return visited

    def _bfs_strong_relations_bidirectional(
        self,
        graph: nx.DiGraph,
        start_node: str
    ) -> Set[str]:
        """
        Bidirectional BFS from start node to find all nodes connected by strong relations

        Args:
            graph: Graph
            start_node: Start node

        Returns:
            Set of reachable nodes (union of forward + backward)
        """
        # Forward traversal: Find successor nodes
        forward = self._bfs_strong_relations(graph, start_node, 'forward')
        # Backward traversal: Find predecessor nodes
        backward = self._bfs_strong_relations(graph, start_node, 'backward')

        # Return union of both
        return forward | backward

    def _find_all_strong_components(self, graph: nx.DiGraph) -> Set[str]:
        """
        Find all strong relation connected components

        Strategy:
        1. Traverse all nodes, use bidirectional BFS to find strong relation connected components
        2. Only retain connected components with size >= min_component_size
        3. Record metadata for each connected component (for subsequent deduplication)

        Args:
            graph: Original graph

        Returns:
            Set of all paper nodes to retain
        """
        visited = set()
        papers_to_keep = set()
        self.strong_components = []  # Clear and re-record

        for node in graph.nodes():
            if node in visited:
                continue

            # Bidirectional BFS to find strong relation connected component of this node
            component = self._bfs_strong_relations_bidirectional(graph, node)
            visited.update(component)

            # Filter small clusters (may be noise)
            if len(component) >= self.min_component_size:
                papers_to_keep.update(component)

                # Record connected component metadata
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

        logger.info(f"    Identified {len(self.strong_components)} strong relation connected components")
        logger.info(f"    Retained {len(papers_to_keep)} papers (connected component size >= {self.min_component_size})")

        return papers_to_keep

    def _find_seed_connected_papers(
        self,
        graph: nx.DiGraph,
        seed_papers: Set[str]
    ) -> Set[str]:
        """
        Original logic: Only retain papers connected to seeds (for seed_centric mode)

        Args:
            graph: Graph
            seed_papers: Set of seed papers

        Returns:
            Set of papers connected to seeds
        """
        papers_to_keep = set(seed_papers)

        # Forward traversal: Seed -> subsequent papers (through strong relations)
        for seed in seed_papers:
            reachable_forward = self._bfs_strong_relations(
                graph, seed, direction='forward'
            )
            papers_to_keep.update(reachable_forward)

        # Backward traversal: Seed <- predecessor papers (through strong relations)
        for seed in seed_papers:
            reachable_backward = self._bfs_strong_relations(
                graph, seed, direction='backward'
            )
            papers_to_keep.update(reachable_backward)

        logger.info(f"    Through strong relation connectivity analysis, retained {len(papers_to_keep)} papers")

        return papers_to_keep

    def _lightweight_explore_chains(
        self,
        graph: nx.DiGraph,
        start_node: str,
        scope: Set[str],
        max_depth: int = 5
    ) -> Set[str]:
        """
        Lightweight chain exploration (for pre-evaluation)

        Performs BFS along strong relation edges with depth limit

        Args:
            graph: Graph
            start_node: Starting node
            scope: Allowed node scope
            max_depth: Maximum exploration depth

        Returns:
            Set of reachable nodes
        """
        visited = set([start_node])
        current_layer = {start_node}

        for _ in range(max_depth):
            next_layer = set()
            for node in current_layer:
                # Explore successor nodes
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
        Lightweight divergence exploration (for pre-evaluation)

        Explores from center node towards predecessors direction

        Args:
            graph: Graph
            center_node: Center node
            scope: Allowed node scope
            max_depth: Maximum exploration depth

        Returns:
            Set of reachable nodes
        """
        visited = set([center_node])

        # Get all predecessors that reference center node through strong relations
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

            # Continue exploring backwards (with depth limit)
            current = predecessor
            for _ in range(max_depth - 1):
                next_predecessors = [
                    np for np in graph.predecessors(current)
                    if np in scope and np not in visited
                ]

                if not next_predecessors:
                    break

                # Select strong relation predecessors
                valid_predecessors = [
                    np for np in next_predecessors
                    if (graph.edges[np, current].get('type') or
                        graph.edges[np, current].get('edge_type', '')) in self.strong_relations
                ]

                if not valid_predecessors:
                    break

                # Select the one with highest citation count
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
        Lightweight convergence exploration (for pre-evaluation)

        Explores from center node towards successors direction

        Args:
            graph: Graph
            center_node: Center node
            scope: Allowed node scope
            max_depth: Maximum exploration depth

        Returns:
            Set of reachable nodes
        """
        visited = set([center_node])

        # Get all successors referenced by center node through strong relations
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

            # Continue exploring forward (with depth limit)
            current = successor
            for _ in range(max_depth - 1):
                next_successors = [
                    ns for ns in graph.successors(current)
                    if ns in scope and ns not in visited
                ]

                if not next_successors:
                    break

                # Select strong relation successors
                valid_successors = [
                    ns for ns in next_successors
                    if (graph.edges[current, ns].get('type') or
                        graph.edges[current, ns].get('edge_type', '')) in self.strong_relations
                ]

                if not valid_successors:
                    break

                # Select the one with highest citation count
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
        Pre-evaluate the number of papers a node can cover

        Considers chain, divergence, and convergence directions comprehensively

        Args:
            graph: Graph
            node_id: Candidate node ID
            scope: Allowed node scope (connected component)

        Returns:
            Estimated total number of papers that can be covered
        """
        all_papers = set()

        # 1. Chain direction coverage
        chain_papers = self._lightweight_explore_chains(
            graph, node_id, scope, self.exploration_depth
        )
        all_papers.update(chain_papers)

        # 2. Divergence direction coverage
        divergence_papers = self._lightweight_explore_divergence(
            graph, node_id, scope, self.exploration_depth
        )
        all_papers.update(divergence_papers)

        # 3. Convergence direction coverage
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
        Find key nodes in connected component

        Priority:
        1. Seed nodes (if any)
        2. Pre-evaluation to select top 3 nodes with highest coverage

        Args:
            graph: Graph
            component_papers: Set of papers in connected component

        Returns:
            List of key node IDs
        """
        key_nodes = []

        # Prioritize seed nodes
        seed_nodes = [
            node_id for node_id in component_papers
            if graph.nodes[node_id].get('is_seed', False)
        ]
        key_nodes.extend(seed_nodes)

        # If no seeds, use pre-evaluation mechanism
        if len(key_nodes) == 0:
            component_size = len(component_papers)

            # Performance optimization: large components (>50 nodes) use fallback strategy
            if component_size > 50:
                logger.info(f"    Component too large ({component_size} nodes), using degree centrality fallback strategy")
                subgraph = graph.subgraph(component_papers)
                centrality = nx.degree_centrality(subgraph)
                top_nodes = sorted(
                    centrality.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                key_nodes = [node_id for node_id, _ in top_nodes]
            else:
                # Pre-evaluation strategy: calculate coverage for top 10 candidates by degree centrality
                subgraph = graph.subgraph(component_papers)
                centrality = nx.degree_centrality(subgraph)
                top_candidates = sorted(
                    centrality.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]  # Candidate pool: top 10

                # Pre-evaluate paper coverage for each candidate node
                candidate_scores = []
                for node_id, _ in top_candidates:
                    coverage = self._pre_evaluate_node_coverage(
                        graph, node_id, component_papers
                    )
                    candidate_scores.append((node_id, coverage))
                    logger.info(f"      Pre-evaluating node {node_id[:20]}...: estimated coverage {coverage} papers")

                # Sort by coverage count, select top 3
                candidate_scores.sort(key=lambda x: x[1], reverse=True)
                key_nodes = [node_id for node_id, _ in candidate_scores[:3]]

                logger.info(f"    Pre-evaluation complete, selected 3 optimal nodes")

        return key_nodes

    def _find_linear_chains_in_scope(
        self,
        graph: nx.DiGraph,
        start_node: str,
        scope: Set[str]
    ) -> List[List[str]]:
        """
        Identify linear chains within specified scope (avoid cross-component identification)

        Args:
            graph: Graph
            start_node: Starting node
            scope: Allowed node scope (nodes within connected component)

        Returns:
            List of chains, each chain is a list of node IDs
        """
        chains = []
        min_chain_length = self.config.get('min_chain_length', 3)

        def dfs_chain(current_path: List[str]):
            """DFS search for chains (limited to scope)"""
            current = current_path[-1]
            successors = list(graph.successors(current))

            if len(successors) == 0:
                # Reached endpoint
                if len(current_path) >= min_chain_length:
                    chains.append(current_path.copy())
                return

            # Prioritize continuing along strong relations
            for successor in successors:
                # Key modification: only consider nodes within scope
                if successor not in scope:
                    continue
                if successor in current_path:  # Avoid cycles
                    continue

                edge_data = graph.edges[current, successor]
                edge_type = edge_data.get('type') or edge_data.get('edge_type', '')

                if edge_type in self.strong_relations or edge_type == '':
                    current_path.append(successor)
                    dfs_chain(current_path)
                    current_path.pop()

            # If no strong relations found, record current chain
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
        Identify divergence structure within specified scope (avoid cross-component identification)

        Divergence definition: a center node is referenced/extended by multiple subsequent papers
        Direction: center node -> multiple branches (explore towards predecessors)

        Args:
            graph: Graph
            center_node: Center node
            scope: Allowed node scope (nodes within connected component)

        Returns:
            Divergence structure dictionary containing center node and routes
        """
        # Use predecessors to find papers that reference center node (Bug Fix #6)
        predecessors = list(graph.predecessors(center_node))

        # Key modification: only consider predecessor nodes within scope
        predecessors = [p for p in predecessors if p in scope]

        if len(predecessors) < 2:
            return None

        # For each predecessor node, identify its evolution route
        routes = []
        for predecessor in predecessors:
            edge_data = graph.edges[predecessor, center_node]
            edge_type = edge_data.get('type') or edge_data.get('edge_type', '')

            # Only keep strong relation routes
            if edge_type not in self.strong_relations:
                continue

            # Initialize route
            route = {
                'relation_type': edge_type,
                'papers': [predecessor]
            }

            # Continue exploring backwards (up to exploration_depth layers) - find newer papers
            current = predecessor
            for _ in range(self.exploration_depth):
                next_predecessors = list(graph.predecessors(current))
                # Key modification: only consider nodes within scope
                next_predecessors = [np for np in next_predecessors if np in scope]

                if len(next_predecessors) == 0:
                    break

                # Select predecessor with highest citation count (and strong relation)
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

        # At least 2 valid routes required for divergence
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
        Identify linear chains and divergence structures for a connected component

        Strategy:
        1. Find "key nodes" in connected component (high centrality or seed nodes)
        2. Identify chains and divergence from key nodes
        3. Mark component ID for deduplication

        Args:
            graph: Graph
            component: Connected component dictionary (contains papers, size, total_citations, etc.)

        Returns:
            List of evolution paths
        """
        paths = []
        component_papers = component['papers']

        # Find key nodes in connected component
        key_nodes = self._find_key_nodes_in_component(graph, component_papers)

        for node_id in key_nodes:
            node_data = graph.nodes[node_id]

            # Identify linear chains (limited to current connected component)
            chains = self._find_linear_chains_in_scope(
                graph,
                node_id,
                component_papers
            )
            for chain in chains:
                path = self._create_chain_narrative(graph, chain, node_data)
                path['component_id'] = id(component)  # Mark belonging component
                paths.append(path)

            # Identify divergence structure (limited to current connected component)
            divergence = self._find_divergence_pattern_in_scope(
                graph,
                node_id,
                component_papers
            )
            if divergence and len(divergence['routes']) > 1:
                path = self._create_divergence_narrative(graph, divergence, node_data)
                path['component_id'] = id(component)
                paths.append(path)

            # Identify convergence structure (limited to current connected component)
            convergence = self._find_convergence_pattern_in_scope(
                graph,
                node_id,
                component_papers
            )
            if convergence and len(convergence['routes']) > 1:
                path = self._create_convergence_narrative(graph, convergence, node_data)
                path['component_id'] = id(component)
                paths.append(path)
                logger.info(f"      Identified convergence structure: {len(convergence['routes'])} routes converging")

        return paths

    def _calculate_path_priority(self, path: Dict) -> float:
        """
        Calculate path priority

        Scoring dimensions:
        1. Paper count (weight: 30%)
        2. Total citations (weight: 30%)
        3. Path type diversity (weight: 20%)
        4. Key relation types (weight: 20%)

        Args:
            path: Evolution path

        Returns:
            Priority score
        """
        # 1. Paper count (weight: 30%)
        paper_count_score = len(path['papers']) * 10

        # 2. Total citations (weight: 30%)
        citation_score = path.get('total_citations', 0) / 100  # Normalize

        # 3. Path type diversity (weight: 20%)
        diversity_score = 0
        if path['thread_type'] in ['divergence', 'convergence']:
            # Divergence/convergence structure: more diverse relation types, higher score
            route_types = set(r['relation_type'] for r in path.get('routes', []))
            diversity_score = len(route_types) * 20
            # Special bonus: contains both Alternative and Extends
            if 'Alternative' in route_types and 'Extends' in route_types:
                diversity_score += 30
        else:
            # Chain structure: length itself represents evolution depth
            diversity_score = len(path['papers']) * 5

        # 4. Contains key relations (weight: 20%)
        key_relation_score = 0
        if path['thread_type'] in ['divergence', 'convergence']:
            route_types = set(r['relation_type'] for r in path.get('routes', []))
            # Overcomes and Alternative are the most valuable relations
            if 'Overcomes' in route_types:
                key_relation_score += 25
            if 'Alternative' in route_types:
                key_relation_score += 25
            if 'Realizes' in route_types:
                key_relation_score += 15

        return paper_count_score + citation_score + diversity_score + key_relation_score

    def _identify_evolutionary_paths(self, graph: nx.DiGraph) -> List[Dict]:
        """
        Step 2: Key evolutionary path identification

        Supports two modes:
        - comprehensive: Identify paths for each connected component
        - seed_centric: Identify paths based on seed nodes (original implementation)

        Identifies two core evolution patterns:
        1. Linear chain (The Chain): A -> Overcomes -> B -> Extends -> C
        2. Divergence pattern (The Divergence): Seed -> [Multiple Routes]

        Args:
            graph: Pruned graph

        Returns:
            List of evolutionary paths
        """
        logger.info("  Identifying evolutionary paths...")

        evolutionary_paths = []

        if self.pruning_mode == 'comprehensive':
            # New strategy: identify paths for each connected component
            logger.info(f"    comprehensive mode: identifying paths for {len(self.strong_components)} connected components")

            for component in self.strong_components:
                component_paths = self._identify_paths_in_component(
                    graph,
                    component
                )
                evolutionary_paths.extend(component_paths)

        else:
            # Original strategy: identify based on seed nodes
            # Identify Seed Papers
            seed_papers = [node_id for node_id in graph.nodes()
                          if graph.nodes[node_id].get('is_seed', False)]

            if len(seed_papers) == 0:
                logger.warning("    No Seed Papers found, using high centrality nodes")
                # Fallback: use degree centrality
                centrality = nx.degree_centrality(graph)
                top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                seed_papers = [node_id for node_id, _ in top_nodes]

            logger.info(f"    seed_centric mode: identifying evolutionary paths based on {len(seed_papers)} seed nodes")

            # Identify evolutionary paths for each Seed
            for seed_id in seed_papers:
                seed_data = graph.nodes[seed_id]

                # Pattern 1: Identify linear chains starting from this Seed
                chains = self._find_linear_chains(graph, seed_id)
                for chain in chains:
                    path = self._create_chain_narrative(graph, chain, seed_data)
                    evolutionary_paths.append(path)

                # Pattern 2: Identify divergence structure centered on this Seed
                divergence = self._find_divergence_pattern(graph, seed_id)
                if divergence and len(divergence['routes']) > 1:  # At least 2 branches for divergence
                    path = self._create_divergence_narrative(graph, divergence, seed_data)
                    evolutionary_paths.append(path)

                # Pattern 3: Identify convergence structure centered on this Seed
                convergence = self._find_convergence_pattern(graph, seed_id)
                if convergence and len(convergence['routes']) > 1:  # At least 2 routes for convergence
                    path = self._create_convergence_narrative(graph, convergence, seed_data)
                    evolutionary_paths.append(path)
                    logger.info(f"    Seed {seed_id[:20]}... forms convergence structure")

        # Sort by importance
        evolutionary_paths.sort(key=self._calculate_path_priority, reverse=True)

        # Enhanced deduplication mechanism (using new implementation)
        evolutionary_paths = self._deduplicate_paths_enhanced(evolutionary_paths)

        # Keep only top N most important stories
        max_threads = self.config.get('max_threads', 5)
        evolutionary_paths = evolutionary_paths[:max_threads]

        logger.info(f"    ✅ Identification complete: {len(evolutionary_paths)} evolutionary paths")

        return evolutionary_paths

    def _calculate_dedup_threshold(self, path1: Dict, path2: Dict) -> float:
        """
        Dynamically calculate deduplication threshold

        Rules:
        1. Paths within same connected component: high threshold (0.8)
        2. Paths from different connected components: medium threshold (0.6)
        3. Paths of different types (chain vs star): low threshold (0.5)

        Args:
            path1: First path
            path2: Second path

        Returns:
            Deduplication threshold
        """
        base_threshold = self.config.get('path_overlap_threshold', 0.8)

        # If from different connected components, lower threshold
        if path1.get('component_id') != path2.get('component_id'):
            base_threshold = 0.6

        # If different types, further lower threshold
        if path1.get('thread_type') != path2.get('thread_type'):
            base_threshold = min(base_threshold, 0.5)

        return base_threshold

    def _are_semantically_different(self, path1: Dict, path2: Dict) -> bool:
        """
        Determine if two paths are semantically different (even if papers overlap)

        Factors considered:
        1. Relation type differences: Overcomes vs Alternative
        2. Paper role differences: same paper in different roles in different paths
        3. Evolution direction differences: different temporal order

        Args:
            path1: First path
            path2: Second path

        Returns:
            True indicates semantically different, should keep both
        """
        # 1. Check if main relation types are different (divergence/convergence structures)
        if path1.get('thread_type') in ['divergence', 'convergence'] and \
           path2.get('thread_type') in ['divergence', 'convergence']:
            routes1 = set(r['relation_type'] for r in path1.get('routes', []))
            routes2 = set(r['relation_type'] for r in path2.get('routes', []))

            # If divergence/convergence structures have completely different relation types, consider semantically different
            if len(routes1 & routes2) == 0:
                return True

        # 2. Check if center nodes are different (divergence/convergence structures)
        if path1.get('thread_type') in ['divergence', 'convergence'] and \
           path2.get('thread_type') in ['divergence', 'convergence']:
            papers1 = path1['papers']
            papers2 = path2['papers']

            center1 = next((p for p in papers1 if p.get('role') == 'center'), None)
            center2 = next((p for p in papers2 if p.get('role') == 'center'), None)

            if center1 and center2 and center1['paper_id'] != center2['paper_id']:
                return True

        # 3. Check relation chain differences (chain structures)
        if path1.get('thread_type') == 'chain' and path2.get('thread_type') == 'chain':
            chain1 = path1.get('relation_chain', [])
            chain2 = path2.get('relation_chain', [])

            if len(chain1) > 0 and len(chain2) > 0:
                # If main relation types in chains are different, consider semantically different
                types1 = set(r['relation_type'] for r in chain1)
                types2 = set(r['relation_type'] for r in chain2)

                max_len = max(len(types1), len(types2))
                if max_len > 0 and len(types1 & types2) / max_len < 0.5:
                    return True

        return False

    def _deduplicate_paths_enhanced(self, paths: List[Dict]) -> List[Dict]:
        """
        Enhanced path deduplication mechanism

        Multi-layer deduplication strategy:
        1. Jaccard similarity based on paper sets (existing mechanism)
        2. Consider path type and role differences (new)
        3. Dynamic threshold adjustment (new)

        Args:
            paths: List of evolutionary paths (already sorted by priority)

        Returns:
            Deduplicated path list
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

                # Calculate Jaccard similarity
                intersection = len(path_papers & kept_papers)
                union = len(path_papers | kept_papers)
                similarity = intersection / union if union > 0 else 0

                # Dynamic threshold: adjust based on path characteristics
                threshold = self._calculate_dedup_threshold(path, kept_path)

                if similarity >= threshold:
                    # Further check: semantically different but papers overlap
                    if self._are_semantically_different(path, kept_path):
                        # Semantically different, keep both
                        continue

                    is_duplicate = True
                    removed_count += 1
                    logger.info(
                        f"    Deduplication: Thread #{i+1} overlaps with Thread #{deduplicated.index(kept_path)+1} "
                        f"by {similarity:.2%} (threshold {threshold:.2%}), removed"
                    )
                    break

            if not is_duplicate:
                deduplicated.append(path)

        if removed_count > 0:
            logger.info(f"    Deduplication complete: removed {removed_count} duplicate paths")

        return deduplicated

    def _deduplicate_paths(self, paths: List[Dict]) -> List[Dict]:
        """
        Bug Fix #2: Deduplicate/merge evolutionary paths with high overlap

        If two paths have paper overlap exceeding threshold, keep the longer/more important one

        Args:
            paths: List of evolutionary paths (already sorted by priority)

        Returns:
            Deduplicated path list
        """
        if len(paths) <= 1:
            return paths

        overlap_threshold = self.config.get('path_overlap_threshold', 0.8)
        deduplicated = []
        removed_count = 0

        for i, path in enumerate(paths):
            # Extract paper ID set for current path
            path_papers = set(p['paper_id'] for p in path['papers'])

            # Check if overlaps with already kept paths
            is_duplicate = False
            for kept_path in deduplicated:
                kept_papers = set(p['paper_id'] for p in kept_path['papers'])

                # Calculate Jaccard similarity
                intersection = len(path_papers & kept_papers)
                union = len(path_papers | kept_papers)

                if union > 0:
                    similarity = intersection / union
                else:
                    similarity = 0

                # If overlap exceeds threshold, mark as duplicate
                if similarity >= overlap_threshold:
                    is_duplicate = True
                    removed_count += 1
                    logger.info(f"    Deduplication: Thread #{i+1} overlaps with Thread #{deduplicated.index(kept_path)+1} by {similarity:.2%}, removed")
                    break

            if not is_duplicate:
                deduplicated.append(path)

        if removed_count > 0:
            logger.info(f"    Deduplication complete: removed {removed_count} duplicate paths")

        return deduplicated

    def _find_linear_chains(self, graph: nx.DiGraph, start_node: str) -> List[List[str]]:
        """
        Identify linear chains starting from a node

        Chain definition: A -> B -> C, each node has at most one main successor

        Args:
            graph: Graph
            start_node: Starting node

        Returns:
            List of chains, each chain is a list of node IDs
        """
        chains = []
        min_chain_length = self.config.get('min_chain_length', 3)

        def dfs_chain(current_path: List[str]):
            """DFS search for chains"""
            current = current_path[-1]
            successors = list(graph.successors(current))

            if len(successors) == 0:
                # Reached endpoint
                if len(current_path) >= min_chain_length:
                    chains.append(current_path.copy())
                return

            # Prioritize continuing along strong relations
            for successor in successors:
                if successor in current_path:  # Avoid cycles
                    continue

                edge_data = graph.edges[current, successor]
                edge_type = edge_data.get('type') or edge_data.get('edge_type', '')

                if edge_type in self.strong_relations or edge_type == '':
                    current_path.append(successor)
                    dfs_chain(current_path)
                    current_path.pop()

            # If no strong relations found, record current chain
            if len(current_path) >= min_chain_length:
                chains.append(current_path.copy())

        dfs_chain([start_node])

        return chains

    def _find_divergence_pattern(self, graph: nx.DiGraph, center_node: str) -> Optional[Dict]:
        """
        Identify divergence structure centered on a node

        Divergence definition: a foundational center node is referenced/extended by multiple subsequent papers
        Correct direction: find predecessors, i.e., papers that reference the center node

        Special focus: Overcomes, Alternative, Extends branches

        Args:
            graph: Graph
            center_node: Center node

        Returns:
            Divergence structure dictionary containing center node and routes
        """
        # Bug Fix #6: Use predecessors instead of successors
        # predecessors = papers that referenced the center node (i.e., subsequent research)
        predecessors = list(graph.predecessors(center_node))

        if len(predecessors) < 2:
            return None

        # For each predecessor node (papers citing the center paper), identify its evolution route
        routes = []
        for predecessor in predecessors:
            edge_data = graph.edges[predecessor, center_node]
            edge_type = edge_data.get('type') or edge_data.get('edge_type', '')

            # Only keep strong relation routes (filter out Baselines)
            if edge_type not in self.strong_relations:
                continue

            # Continue searching along this predecessor to form a route
            route = {
                'relation_type': edge_type,
                'papers': [predecessor]
            }

            # Continue exploring backwards (up to exploration_depth layers) - find newer papers
            current = predecessor
            for _ in range(self.exploration_depth):
                # Bug Fix #6: Use predecessors to find newer papers
                next_predecessors = list(graph.predecessors(current))
                if len(next_predecessors) == 0:
                    break
                # Select predecessor with highest citation count (and strong relation)
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

        # At least 2 valid routes required for divergence
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
        Identify convergence structure within specified scope (avoid cross-component identification)

        Convergence definition: a comprehensive center node references/integrates multiple prior papers
        Direction: center node <- multiple foundational routes (explore towards successors)

        Args:
            graph: Graph
            center_node: Center node
            scope: Allowed node scope (nodes within connected component)

        Returns:
            Convergence structure dictionary containing center node and routes
        """
        # Use successors to find papers referenced by center node (i.e., foundational work)
        successors = list(graph.successors(center_node))

        # Key modification: only consider successor nodes within scope
        successors = [s for s in successors if s in scope]

        if len(successors) < 2:
            return None

        # For each successor node, identify its foundational route
        routes = []
        for successor in successors:
            edge_data = graph.edges[center_node, successor]
            edge_type = edge_data.get('type') or edge_data.get('edge_type', '')

            # Only keep strong relation routes
            if edge_type not in self.strong_relations:
                continue

            # Initialize route
            route = {
                'relation_type': edge_type,
                'papers': [successor]
            }

            # Continue exploring forward (up to exploration_depth layers) - find earlier foundational papers
            current = successor
            for _ in range(self.exploration_depth):
                next_successors = list(graph.successors(current))
                # Key modification: only consider nodes within scope
                next_successors = [ns for ns in next_successors if ns in scope]

                if len(next_successors) == 0:
                    break

                # Select successor with highest citation count (and strong relation)
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

        # At least 2 valid routes required for convergence
        if len(routes) < 2:
            return None

        return {
            'center': center_node,
            'routes': routes
        }

    def _find_convergence_pattern(self, graph: nx.DiGraph, center_node: str) -> Optional[Dict]:
        """
        Identify convergence structure centered on a node

        Convergence definition: a comprehensive center node integrates multiple prior foundational works
        Correct direction: find successors, i.e., papers referenced by the center node

        Args:
            graph: Graph
            center_node: Center node

        Returns:
            Convergence structure dictionary containing center node and routes
        """
        # Use successors instead of predecessors
        # successors = papers referenced by center node (i.e., foundational work)
        successors = list(graph.successors(center_node))

        if len(successors) < 2:
            return None

        # For each successor node (papers referenced by center), identify its foundational route
        routes = []
        for successor in successors:
            edge_data = graph.edges[center_node, successor]
            edge_type = edge_data.get('type') or edge_data.get('edge_type', '')

            # Only keep strong relation routes (filter out Baselines)
            if edge_type not in self.strong_relations:
                continue

            # Continue searching along this successor to form a route
            route = {
                'relation_type': edge_type,
                'papers': [successor]
            }

            # Continue exploring forward (up to exploration_depth layers) - find earlier foundational papers
            current = successor
            for _ in range(self.exploration_depth):
                # Use successors to find earlier foundational papers
                next_successors = list(graph.successors(current))
                if len(next_successors) == 0:
                    break
                # Select successor with highest citation count (and strong relation)
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

        # At least 2 valid routes required for convergence
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
        Create narrative unit for linear chain

        Generation template:
        - Origin: Paper A proposed [Method A] to solve [Problem], but has limitations in [Limitation].
        - Turning point: Paper B addressed this limitation and successfully overcame the problem through [Method B].
        - Development: Subsequently, Paper C further extended its application scenarios based on B.

        Args:
            graph: Graph
            chain: List of chain nodes
            seed_data: Seed node data

        Returns:
            Narrative unit dictionary
        """
        # Bug Fix #1: Sort chain by time (time reversal issue)
        # Get year for each paper, sort in chronological order
        chain_with_years = []
        for paper_id in chain:
            node_data = graph.nodes[paper_id]
            year = node_data.get('year', 0)
            chain_with_years.append((paper_id, year))

        # Sort by year in ascending order
        chain_with_years.sort(key=lambda x: x[1])
        sorted_chain = [paper_id for paper_id, _ in chain_with_years]

        logger.info(f"    Sorted chain by time: {[f'{pid}({y})' for pid, y in chain_with_years]}")

        papers_info = []
        total_citations = 0
        relation_chain = []  # New: detailed relation chain

        for i, paper_id in enumerate(sorted_chain):
            node_data = graph.nodes[paper_id]
            papers_info.append({
                'paper_id': paper_id,
                'title': node_data.get('title', ''),
                'year': node_data.get('year', 0),
                'cited_by_count': node_data.get('cited_by_count', 0)
            })
            total_citations += node_data.get('cited_by_count', 0)

            # Build relation chain: extract relations between each pair of papers
            # Bug Fix #3: Correct relation direction - change to development direction (early->late)
            # Bug Fix #5: Correct relation semantics - need to reverse relation meaning in chronological order
            if i < len(sorted_chain) - 1:
                next_paper_id = sorted_chain[i + 1]
                edge_type = 'Unknown'
                edge_description = 'Unknown'

                # Try finding edge in both directions
                if graph.has_edge(paper_id, next_paper_id):
                    # Early paper -> Late paper (rare case, possibly Inspires type)
                    edge_data = graph.edges[paper_id, next_paper_id]
                    original_type = edge_data.get('type') or edge_data.get('edge_type', 'Temporal_Evolution')
                    edge_type = original_type
                    edge_description = original_type
                elif graph.has_edge(next_paper_id, paper_id):
                    # Late paper -> Early paper (common case: new paper Overcomes old paper)
                    edge_data = graph.edges[next_paper_id, paper_id]
                    original_type = edge_data.get('type') or edge_data.get('edge_type', 'Unknown')

                    # Key fix: reverse relation semantics to match temporal narrative
                    # Original: new paper(2023) --Overcomes--> old paper(2021)
                    # Narrative: old paper(2021) --Was_Overcome_By--> new paper(2023)
                    edge_type = original_type
                    edge_description = self._reverse_relation_semantics(original_type)
                else:
                    # No direct edge, mark as temporal evolution relation
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
                    'relation_type': edge_type,  # Original relation type
                    'narrative_relation': edge_description,  # Relation description for narrative
                    'direction': 'chronological'  # Explicitly mark as chronological order
                })

        # Extract key information (using sorted chain)
        first_paper = graph.nodes[sorted_chain[0]]
        last_paper = graph.nodes[sorted_chain[-1]]

        # Generate title
        first_method = self._extract_key_method(first_paper)
        last_method = self._extract_key_method(last_paper)

        title = f"Evolution from {first_method} to {last_method}"

        # Generate narrative text (using LLM or template, using sorted chain)
        narrative = self._generate_chain_narrative_text(graph, sorted_chain)

        return {
            'thread_type': 'chain',
            'pattern_type': 'The Chain (Linear Chain)',
            'title': title,
            'narrative': narrative,
            'papers': papers_info,
            'total_citations': total_citations,
            'visual_structure': ' -> '.join([f"Paper_{i+1}" for i in range(len(sorted_chain))]),
            'relation_chain': relation_chain  # New: detailed relation chain
        }

    def _create_divergence_narrative(
        self,
        graph: nx.DiGraph,
        divergence: Dict,
        seed_data: Dict
    ) -> Dict:
        """
        Create narrative unit for divergence structure

        Bug Fix #6: Correct temporal logic of divergence structure
        Correct narrative: center paper (early foundation) -> multiple subsequent evolution routes (later)

        Generation template:
        - Focus: The center paper is a cornerstone of the field, but it left [Limitation] problems.
        - Divergence: Academia developed different evolution routes in response.
        - Comparison: (insert comparison of routes)

        Args:
            graph: Graph
            divergence: Divergence structure
            seed_data: Seed node data

        Returns:
            Narrative unit dictionary
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

        # Collect papers and relation chains for all routes
        routes_info = []
        relation_chain = []  # New: detailed relation chain

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

            # Bug Fix #6: Correct relation chain direction
            # Now papers in route['papers'] reference the center paper
            # So relation direction should be: route paper -> center paper (in graph)
            # But narrative direction should be: center paper -> route paper (temporal)
            if route_papers:
                first_paper = route_papers[0]
                first_paper_year = first_paper['year']

                # Check temporal relation to ensure correct narrative
                if center_year <= first_paper_year:
                    # Center paper is earlier (normal case)
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
                    # Abnormal case: route paper is earlier (record but annotate)
                    logger.warning(f"    Divergence structure anomaly: route paper {first_paper['paper_id']} ({first_paper_year}) is earlier than center paper {center_id} ({center_year})")
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

        # Extract core problem
        seed_problem = self._extract_key_problem(center_data)

        # Generate title
        title = f"Multi-Technical Route Competition for {seed_problem}"

        # Generate narrative text
        narrative = self._generate_divergence_narrative_text(graph, divergence, center_data)

        return {
            'thread_type': 'divergence',
            'pattern_type': 'The Divergence (Divergence Pattern)',
            'title': title,
            'narrative': narrative,
            'center_paper': center_data.get('title', ''),
            'routes_count': len(routes_info),
            'routes': routes_info,
            'papers': papers_info,
            'total_citations': total_citations,
            'visual_structure': f"Center -> {len(routes_info)} Routes",
            'relation_chain': relation_chain  # New: detailed relation chain
        }

    def _create_convergence_narrative(
        self,
        graph: nx.DiGraph,
        convergence: Dict,
        seed_data: Dict
    ) -> Dict:
        """
        Create narrative unit for convergence structure

        Convergence pattern: multiple foundational routes converge to a comprehensive paper

        Generation template:
        - Background: Multiple independent research directions explored different technical paths
        - Convergence: Center paper integrated these directions to form comprehensive framework
        - Significance: Marks systematic integration of field theory

        Args:
            graph: Graph
            convergence: Convergence structure
            seed_data: Seed node data

        Returns:
            Narrative unit dictionary
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

        # Collect papers and relation chains for all routes
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

            # Build relation chain: center paper references foundational routes
            if route_papers:
                first_paper = route_papers[0]
                first_paper_year = first_paper['year']

                # Convergence structure: center paper references foundational papers (center paper should be later temporally)
                if center_year >= first_paper_year:
                    # Center paper is later (normal case)
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
                        'narrative_relation': f"integrated by {center_data.get('title', '')}",
                        'route_id': route_idx + 1,
                        'direction': 'chronological'
                    })
                else:
                    # Abnormal case: foundational paper is later
                    logger.warning(f"    Convergence structure anomaly: foundational paper {first_paper['paper_id']} ({first_paper_year}) is later than center paper {center_id} ({center_year})")
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

        # Extract core method
        center_method = self._extract_key_method(center_data)

        # Generate title
        title = f"Multiple Technical Routes Converging to {center_method}"

        # Generate narrative text
        narrative = self._generate_convergence_narrative_text(graph, convergence, center_data)

        return {
            'thread_type': 'convergence',
            'pattern_type': 'The Convergence (Convergence Pattern)',
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
        Generate narrative text for linear chain

        Args:
            graph: Graph
            chain: List of chain nodes

        Returns:
            Narrative text
        """
        if self.llm_client:
            return self._generate_chain_narrative_with_llm(graph, chain)
        else:
            return self._generate_chain_narrative_template(graph, chain)

    def _generate_chain_narrative_template(self, graph: nx.DiGraph, chain: List[str]) -> str:
        """
        Generate chain narrative using template (fallback method)

        Improved version: Citation type-aware narrative generation
        Bug Fix #4: Narrate in chronological order (early->late)
        Bug Fix #5: Use correct relation semantics
        """
        # Sort by year (chronological order)
        chain_with_years = [(pid, graph.nodes[pid].get('year', 0)) for pid in chain]
        chain_with_years.sort(key=lambda x: x[1])
        sorted_chain = [pid for pid, _ in chain_with_years]

        narrative_parts = []

        for i, paper_id in enumerate(sorted_chain):
            node_data = graph.nodes[paper_id]
            title = node_data.get('title', 'Unknown')
            year = node_data.get('year', 'N/A')

            if i == 0:
                # Origin: earliest paper (maintain original logic)
                method = self._extract_key_method(node_data)
                problem = self._extract_key_problem(node_data)
                limitation = self._extract_key_limitation(node_data)
                narrative_parts.append(
                    f"**Origin** ({year}): The paper '{title}' first proposed {method} to solve {problem}, "
                    f"pioneering this research direction. However, this work still has limitations in {limitation}."
                )
            else:
                # Evolution and latest progress: use citation type-aware narrative
                prev_paper_id = sorted_chain[i-1]

                # Get edge relation type
                relation_type = self._get_relation_type(graph, prev_paper_id, paper_id)

                # Extract citation type-related information
                info = self._extract_papers_info_for_relation(
                    graph, prev_paper_id, paper_id, relation_type
                )

                # Generate targeted narrative fragment
                narrative_fragment = self._generate_relation_narrative_fragment(info)
                narrative_parts.append(narrative_fragment)

        return "\n\n".join(narrative_parts)

    def _generate_chain_narrative_with_llm(self, graph: nx.DiGraph, chain: List[str]) -> str:
        """
        Generate chain narrative using LLM

        Improved version: Citation type-aware Prompt enhancement
        Bug Fix #4: Narrate in chronological order
        """
        # Sort by year (chronological order)
        chain_with_years = [(pid, graph.nodes[pid].get('year', 0)) for pid in chain]
        chain_with_years.sort(key=lambda x: x[1])
        sorted_chain = [pid for pid, _ in chain_with_years]

        # Prepare context
        papers_context = []
        for i, paper_id in enumerate(sorted_chain, 1):
            node_data = graph.nodes[paper_id]
            title = node_data.get('title', '')
            year = node_data.get('year', '')

            # Extract basic information
            method = self._extract_key_method(node_data)
            problem = self._extract_key_problem(node_data)
            limitation = self._extract_key_limitation(node_data)
            future_work = self._extract_key_future_work(node_data)

            paper_info = f"Paper {i}: {title} ({year})\n" \
                         f"- Research Problem: {problem}\n" \
                         f"- Research Method: {method}\n" \
                         f"- Limitations: {limitation}\n" \
                         f"- Future Work: {future_work}"

            # If not the first paper, add relationship information with previous paper
            if i > 1:
                prev_paper_id = sorted_chain[i-2]  # i starts from 1, so i-2 is the previous paper
                relation_type = self._get_relation_type(graph, prev_paper_id, paper_id)
                relation_focus = self._get_relation_focus_hint(relation_type)
                paper_info += f"\n- Relationship with previous paper: {relation_type} ({relation_focus})"

            papers_context.append(paper_info)

        context = "\n\n".join(papers_context)

        prompt = f"""You are an academic survey expert. Please generate a coherent narrative text based on the following technology evolution chain.

**Evolution Chain** ({len(sorted_chain)} papers in total, arranged chronologically from earliest to latest):
{context}

**Task Requirements**:
Please generate narrative text following this structure (3-5 paragraphs, 2-3 sentences each):
1. **Origin**: Describe how the first (earliest) paper pioneered this direction, what problem it solved, but what shortcomings it left
2. **Evolution**: Describe how intermediate papers gradually improved upon predecessors' shortcomings (in chronological order)
3. **Latest Progress**: Describe how the last (most recent) paper achieved breakthroughs building on previous work

**Output Requirements**:
- Use coherent, academic English expression
- Highlight causal relationships and technical evolution logic between papers
- Narrate in chronological order (earliest to latest), clearly marking years
- Begin each paragraph with headings like **Origin**, **Evolution**, **Latest Progress**
"""

        try:
            narrative = self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=600
            )
            return narrative.strip()
        except Exception as e:
            logger.warning(f"LLM narrative generation failed: {e}, using template method")
            return self._generate_chain_narrative_template(graph, sorted_chain)

    def _generate_divergence_narrative_text(
        self,
        graph: nx.DiGraph,
        divergence: Dict,
        center_data: Dict
    ) -> str:
        """
        Generate narrative text for divergence structure

        Args:
            graph: Graph
            divergence: Divergence structure
            center_data: Center node data

        Returns:
            Narrative text
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
        """Generate divergence narrative using template (fallback method)

        Improved version: Citation type-aware narrative generation
        """
        center_title = center_data.get('title', 'Unknown')
        center_year = center_data.get('year', 'N/A')
        problem = self._extract_key_problem(center_data)
        limitation = self._extract_key_limitation(center_data)

        narrative_parts = []

        # Focus
        narrative_parts.append(
            f"**Focus**: The paper '{center_title}' ({center_year}) is a foundational work in this field, "
            f"focusing on {problem}, but leaving the issue of {limitation}."
        )

        # Divergence - generate description based on actual relation type (enhanced: citation type-aware)
        center_paper_id = divergence['center']
        routes_desc = []
        for i, route in enumerate(divergence['routes'], 1):
            route_papers = route['papers']
            relation = route['relation_type']

            # Get the latest paper in the route (usually the endpoint of the route)
            latest_paper_id = route_papers[-1] if route_papers else None
            if latest_paper_id:
                # Use citation type-aware information extraction
                info = self._extract_papers_info_for_relation(
                    graph, center_paper_id, latest_paper_id, relation
                )

                # Generate targeted route description (simplified version, adapted for divergence scenario)
                if relation == 'Overcomes':
                    method = info.get('curr_method', 'new method')
                    limitation = info.get('prev_limitation', 'certain limitations')
                    desc = f"**Route {i}** (Vertical Deepening): Targeting '{limitation}', achieving breakthrough through {method}"
                elif relation == 'Realizes':
                    method = info.get('curr_method', 'new method')
                    future_work = info.get('prev_future_work', 'predecessors\' vision')
                    desc = f"**Route {i}** (Research Inheritance): Realizing the vision of '{future_work}', using {method}"
                elif relation == 'Extends':
                    method = info.get('curr_method', 'improved method')
                    desc = f"**Route {i}** (Incremental Innovation): Retaining core architecture and extending, using {method}"
                elif relation == 'Alternative':
                    method = info.get('curr_method', 'alternative method')
                    desc = f"**Route {i}** (Disruptive Innovation): Proposing a completely different alternative, using {method}"
                elif relation == 'Adapts_to':
                    method = info.get('curr_method', 'transfer method')
                    curr_domain = info.get('curr_domain', 'new domain')
                    desc = f"**Route {i}** (Horizontal Diffusion): Transferring technology to '{curr_domain}', using {method}"
                else:  # Baselines
                    method = info.get('curr_method', 'new method')
                    desc = f"**Route {i}**: Work based on center paper, using {method}"

                routes_desc.append(desc)

        narrative_parts.append(
            f"**Divergence**: The academic community has developed different evolution routes. " +
            "; ".join(routes_desc) + "."
        )

        # Comparison - highlight characteristics of different routes
        route_types = [r['relation_type'] for r in divergence['routes']]
        if 'Alternative' in route_types and 'Extends' in route_types:
            comparison = "These routes reflect the diversity of academic research: some choose disruptive innovation, others choose incremental improvement"
        elif 'Overcomes' in route_types:
            comparison = "These routes jointly promote the solution of pain points in this field, forming a multi-angle breakthrough situation"
        else:
            comparison = "These routes each have their advantages, jointly promoting the diversified development of the field"

        narrative_parts.append(f"**Comparison**: {comparison}. (See performance comparison of papers in each route)")

        return "\n\n".join(narrative_parts)

    def _generate_divergence_narrative_with_llm(
        self,
        graph: nx.DiGraph,
        divergence: Dict,
        center_data: Dict
    ) -> str:
        """Generate divergence narrative using LLM"""
        center_title = center_data.get('title', '')
        center_year = center_data.get('year', '')
        problem = self._extract_key_problem(center_data)
        limitation = self._extract_key_limitation(center_data)

        # Prepare context for each route
        routes_context = []
        for i, route in enumerate(divergence['routes'], 1):
            relation = route['relation_type']
            route_papers = []
            for paper_id in route['papers'][:2]:  # Maximum 2 papers per route
                node_data = graph.nodes[paper_id]
                route_papers.append(
                    f"  - {node_data.get('title', '')} ({node_data.get('year', '')}): "
                    f"{self._extract_key_method(node_data)}"
                )

            routes_context.append(
                f"Route {i} ({relation}):\n" + "\n".join(route_papers)
            )

        routes_text = "\n\n".join(routes_context)

        prompt = f"""You are an academic survey expert. Please generate a coherent narrative text based on the following divergence evolution structure.

**Center Paper**:
{center_title} ({center_year})
- Research Problem: {problem}
- Limitations: {limitation}

**Evolution Routes** ({len(divergence['routes'])} routes in total):
{routes_text}

**Task Requirements**:
Please generate narrative text following this structure (3 paragraphs, 2-3 sentences each):
1. **Focus**: Describe the status of the center paper and remaining issues
2. **Divergence**: Describe the different technical directions of each evolution route
3. **Comparison**: Summarize the similarities, differences, and respective advantages of these routes

**Output Requirements**:
- Use coherent, academic English expression
- Highlight technical differences and innovation points of different routes
- Begin each paragraph with headings like **Focus**, **Divergence**, **Comparison**
"""

        try:
            narrative = self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=600
            )
            return narrative.strip()
        except Exception as e:
            logger.warning(f"LLM narrative generation failed: {e}, using template method")
            return self._generate_divergence_narrative_template(graph, divergence, center_data)

    def _generate_convergence_narrative_text(
        self,
        graph: nx.DiGraph,
        convergence: Dict,
        center_data: Dict
    ) -> str:
        """
        Generate narrative text for convergence structure

        Args:
            graph: Graph
            convergence: Convergence structure
            center_data: Center node data

        Returns:
            Narrative text
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
        """Generate convergence narrative using template (fallback method)"""
        center_title = center_data.get('title', 'Unknown')
        center_year = center_data.get('year', 'N/A')
        center_method = self._extract_key_method(center_data)
        problem = self._extract_key_problem(center_data)

        narrative_parts = []

        # Background: describe multiple independent foundation routes
        center_paper_id = convergence['center']
        routes_desc = []
        for i, route in enumerate(convergence['routes'], 1):
            route_papers = route['papers']
            relation = route['relation_type']

            # Get the latest paper in the route (paper closest to convergence center)
            latest_paper_id = route_papers[-1] if route_papers else None
            if latest_paper_id:
                # Use citation type-aware information extraction (route → center direction)
                info = self._extract_papers_info_for_relation(
                    graph, latest_paper_id, center_paper_id, relation
                )

                # Generate targeted route description based on citation type
                if relation == 'Overcomes':
                    limitation = info.get('prev_limitation', 'certain limitations')
                    method = info.get('curr_method', 'new method')
                    desc = f"**Route {i}** (Overcoming): Identified '{limitation}', providing problems to be solved for the center paper"
                elif relation == 'Realizes':
                    future_work = info.get('prev_future_work', 'some future directions')
                    method = info.get('curr_method', 'implementation method')
                    desc = f"**Route {i}** (Realization): Proposed the research direction of '{future_work}', realized by the center paper"
                elif relation == 'Extends':
                    prev_method = info.get('prev_method', 'basic method')
                    curr_method = info.get('curr_method', 'extension method')
                    desc = f"**Route {i}** (Extension): Provided the foundation of {prev_method}, extended by the center paper to {curr_method}"
                elif relation == 'Alternative':
                    prev_method = info.get('prev_method', 'alternative method')
                    curr_method = info.get('curr_method', 'unified method')
                    desc = f"**Route {i}** (Alternative): Provided {prev_method} as a parallel approach, integrated by the center paper"
                elif relation == 'Adapts_to':
                    prev_domain = info.get('prev_domain', 'original domain')
                    curr_domain = info.get('curr_domain', 'new domain')
                    desc = f"**Route {i}** (Adaptation): Work in {prev_domain}, adapted by the center paper to {curr_domain}"
                elif relation == 'Baselines':
                    prev_method = info.get('prev_method', 'baseline method')
                    desc = f"**Route {i}** (Baseline): Provided {prev_method} as a comparison baseline"
                else:
                    # Fallback handling
                    latest_paper = graph.nodes[latest_paper_id]
                    method = self._extract_key_method(latest_paper)
                    year = latest_paper.get('year', 'N/A')
                    desc = f"Route {i} proposed {method} in {year}"

                routes_desc.append(desc)

        narrative_parts.append(
            f"**Background**: Before {center_year}, there were multiple independent research directions in this field. " +
            "; ".join(routes_desc) + ". These directions operated independently, lacking systematic integration."
        )

        # Convergence: describe how the center paper integrates these directions
        narrative_parts.append(
            f"**Convergence**: The paper '{center_title}' ({center_year}) emerged in this context, "
            f"organically integrating these {len(convergence['routes'])} independent routes through {center_method}, "
            f"forming a unified technical framework to solve {problem}."
        )

        # Significance: summarize the value of integration
        integration_value = ""
        if len(convergence['routes']) >= 3:
            integration_value = "marks the field's transition from exploration to systematization"
        else:
            integration_value = "provides a paradigm for theoretical integration in this field"

        narrative_parts.append(
            f"**Significance**: This multi-directional convergence {integration_value}, "
            f"enabling originally scattered technical routes to work synergistically, "
            f"promoting theoretical unification and practical deepening of the field."
        )

        return "\n\n".join(narrative_parts)

    def _generate_convergence_narrative_with_llm(
        self,
        graph: nx.DiGraph,
        convergence: Dict,
        center_data: Dict
    ) -> str:
        """Generate convergence narrative using LLM"""
        center_title = center_data.get('title', '')
        center_year = center_data.get('year', '')
        center_method = self._extract_key_method(center_data)
        problem = self._extract_key_problem(center_data)

        # Prepare context for each route
        routes_context = []
        for i, route in enumerate(convergence['routes'], 1):
            relation = route['relation_type']
            route_papers = []
            for paper_id in route['papers'][:2]:  # Maximum 2 papers per route
                node_data = graph.nodes[paper_id]
                route_papers.append(
                    f"  - {node_data.get('title', '')} ({node_data.get('year', '')}): "
                    f"{self._extract_key_method(node_data)}"
                )

            routes_context.append(
                f"Route {i} ({relation}):\n" + "\n".join(route_papers)
            )

        routes_text = "\n\n".join(routes_context)

        prompt = f"""You are an academic survey expert. Please generate a coherent narrative text based on the following convergence evolution structure.

**Center Paper**:
{center_title} ({center_year})
- Research Problem: {problem}
- Integration Method: {center_method}

**Foundation Routes** ({len(convergence['routes'])} independent routes integrated in total):
{routes_text}

**Task Requirements**:
Please generate narrative text following this structure (3 paragraphs, 2-3 sentences each):
1. **Background**: Describe the exploration of multiple independent directions before the center paper
2. **Convergence**: Describe how the center paper integrates these directions to form a unified framework
3. **Significance**: Summarize the value of this integration for field development

**Output Requirements**:
- Use coherent, academic English expression
- Highlight the evolution from scattered to integrated
- Emphasize synergistic effects and theoretical value after integration
- Begin each paragraph with headings like **Background**, **Convergence**, **Significance**
"""

        try:
            narrative = self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=600
            )
            return narrative.strip()
        except Exception as e:
            logger.warning(f"LLM narrative generation failed: {e}, using template method")
            return self._generate_convergence_narrative_template(graph, convergence, center_data)

    def _generate_survey_report(
        self,
        topic: str,
        pruned_graph: nx.DiGraph,
        evolutionary_paths: List[Dict],
        pruning_stats: Dict
    ) -> Dict:
        """
        Step 3: Generate structured Deep Survey report

        Args:
            topic: Research topic
            pruned_graph: Pruned graph
            evolutionary_paths: List of evolutionary paths
            pruning_stats: Pruning statistics

        Returns:
            Survey report dictionary
        """
        logger.info("  Generating Deep Survey report...")

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

        # Generate Thread for each evolutionary path
        for i, path in enumerate(evolutionary_paths, 1):
            # Extract relation type statistics
            relation_stats = self._extract_relation_stats(path, pruned_graph)

            thread = {
                'thread_id': i,
                'thread_name': f"Thread {i}: {path['pattern_type']}",
                'title': path['title'],
                'pattern_type': path['pattern_type'],
                'thread_type': path.get('thread_type', 'unknown'),  # New: thread type
                'narrative': path['narrative'],
                'papers': path['papers'],
                'total_citations': path.get('total_citations', 0),
                'visual_structure': path.get('visual_structure', ''),
                'relation_stats': relation_stats,  # Relation statistics
                'relation_chain': path.get('relation_chain', []),  # New: detailed relation chain
                # Data prepared for visualization
                'visualization_data': self._prepare_visualization_data(path, pruned_graph)
            }
            report['threads'].append(thread)

        logger.info(f"    ✅ Report generation complete: {len(report['threads'])} Threads")

        return report

    def _generate_abstract(
        self,
        topic: str,
        evolutionary_paths: List[Dict],
        pruning_stats: Dict
    ) -> str:
        """Generate survey abstract"""
        total_papers = pruning_stats['pruned_papers']
        total_threads = len(evolutionary_paths)

        chains = sum(1 for p in evolutionary_paths if p['thread_type'] == 'chain')
        divergences = sum(1 for p in evolutionary_paths if p['thread_type'] == 'divergence')
        convergences = sum(1 for p in evolutionary_paths if p['thread_type'] == 'convergence')

        abstract = (
            f"This survey analyzes the evolution of the {topic} field based on knowledge graph analysis. "
            f"Through relation pruning, we filtered {total_papers} high-quality papers from the original graph "
            f"and identified {total_threads} key evolutionary paths. "
            f"These include {chains} linear technology chains, {divergences} divergence structures, and {convergences} convergence structures, "
            f"fully presenting the field's technology evolution trajectory, divergence trends, and integration patterns."
        )

        return abstract

    def _prepare_visualization_data(self, path: Dict, graph: nx.DiGraph) -> Dict:
        """
        Prepare visualization data

        Bug Fix #3: Ensure arrow direction represents "development direction" (chronological order: early year -> late year)

        Args:
            path: Evolutionary path
            graph: Graph

        Returns:
            Visualization data dictionary
        """
        papers = path['papers']

        # Extract nodes and edges
        nodes = []
        edges = []

        if path['thread_type'] == 'chain':
            # Linear chain - visualize after sorting by time
            # Sort papers by year
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
                    # Bug Fix #3: Arrow direction = time flow (early->late)
                    edges.append({
                        'source': prev_paper_id,  # Early year
                        'target': paper_id,        # Late year
                        'type': 'chronological_evolution',
                        'label': f"{sorted_papers[i-1].get('year', '')} → {paper_info.get('year', '')}"
                    })

        elif path['thread_type'] in ['divergence', 'convergence']:
            # Divergence/Convergence structure
            # Center node
            center_paper = next((p for p in papers if p.get('role') == 'center'), papers[0])
            nodes.append({
                'id': center_paper['paper_id'],
                'label': 'Center',
                'title': center_paper.get('title', ''),
                'year': center_paper.get('year', 0),
                'citations': center_paper.get('cited_by_count', 0),
                'role': 'center'
            })

            # Route nodes
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

                    # Handle connection between first node and center
                    if paper_info == route['papers'][0]:
                        center_year = center_paper.get('year', 0)
                        target_year = paper_info.get('year', 0)

                        # Determine arrow direction based on pattern type and time relationship
                        if path['thread_type'] == 'divergence':
                            # Divergence: center -> route (center paper is earlier)
                            if center_year <= target_year:
                                edges.append({
                                    'source': center_paper['paper_id'],
                                    'target': paper_id,
                                    'type': route.get('relation_type', 'related'),
                                    'direction': 'forward'
                                })
                            else:
                                # Abnormal case: reverse direction
                                edges.append({
                                    'source': paper_id,
                                    'target': center_paper['paper_id'],
                                    'type': f"Inspired_{route.get('relation_type', 'related')}",
                                    'direction': 'backward'
                                })
                        else:
                            # Convergence: route -> center (center paper is later)
                            if center_year >= target_year:
                                edges.append({
                                    'source': paper_id,
                                    'target': center_paper['paper_id'],
                                    'type': route.get('relation_type', 'related'),
                                    'direction': 'forward'
                                })
                            else:
                                # Abnormal case: reverse direction
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
            'direction_note': 'Arrow direction represents time evolution direction (early year → late year)',
            'pattern_note': 'divergence=center diffusion, convergence=multi-source convergence' if path['thread_type'] in ['divergence', 'convergence'] else ''
        }

    # ========== Helper Methods ==========

    def _reverse_relation_semantics(self, relation_type: str) -> str:
        """
        Reverse relation semantics to conform to chronological narrative

        In knowledge graph: new paper(2023) --Overcomes--> old paper(2021)
        In chronological narrative: old paper(2021) --Was_Overcome_By--> new paper(2023)

        Args:
            relation_type: Original relation type (from new paper to old paper)

        Returns:
            Reversed relation description (from old paper to new paper)
        """
        relation_mapping = {
            'Overcomes': 'Was_Overcome_By',      # Was overcome
            'Realizes': 'Inspired',              # Inspired
            'Extends': 'Was_Extended_By',        # Was extended
            'Alternative': 'Led_To_Alternative', # Led to alternative
            'Adapts_to': 'Was_Adapted_By',       # Was adapted
            'Baselines': 'Served_As_Baseline',   # Served as baseline
        }

        return relation_mapping.get(relation_type, f'Led_To_{relation_type}')

    def _map_reversed_to_original_type(self, reversed_type: str) -> str:
        """
        Map reversed relation type back to original type (for narrative generation)

        Args:
            reversed_type: Reversed relation type (e.g., 'Was_Overcome_By')

        Returns:
            Original relation type (e.g., 'Overcomes')
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
        Get relation type between two papers (handle forward and reverse edges)

        Args:
            graph: Graph
            prev_paper_id: Early paper ID
            curr_paper_id: Late paper ID

        Returns:
            Relation type
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
        Get narrative focus hint for citation type (for LLM Prompt)

        Args:
            relation_type: Citation relation type

        Returns:
            Focus description
        """
        hints = {
            'Overcomes': 'Focus on how predecessors\' limitations were overcome',
            'Realizes': 'Focus on how predecessors\' vision was realized',
            'Adapts_to': 'Focus on how methods transfer across domains',
            'Extends': 'Focus on how methods are incrementally improved',
            'Alternative': 'Focus on comparison of different technical paradigms',
            'Baselines': 'Serves as background context'
        }
        return hints.get(relation_type, 'Inherits from previous work')

    def _get_relation_description(self, graph: nx.DiGraph, old_paper_id: str, new_paper_id: str) -> str:
        """
        Get relation description between two papers (for narrative)

        Args:
            graph: Graph
            old_paper_id: Early paper ID
            new_paper_id: Late paper ID

        Returns:
            Relation description conforming to temporal narrative
        """
        # Check if edge exists
        if graph.has_edge(old_paper_id, new_paper_id):
            # Early paper -> Late paper (forward)
            edge_data = graph.edges[old_paper_id, new_paper_id]
            relation_type = edge_data.get('type') or edge_data.get('edge_type', 'Unknown')
            return self._get_chinese_relation_desc(relation_type, is_reversed=False)

        elif graph.has_edge(new_paper_id, old_paper_id):
            # Late paper -> Early paper (need to reverse semantics)
            edge_data = graph.edges[new_paper_id, old_paper_id]
            relation_type = edge_data.get('type') or edge_data.get('edge_type', 'Unknown')
            return self._get_chinese_relation_desc(relation_type, is_reversed=True)

        else:
            return "building on previous work"

    def _get_chinese_relation_desc(self, relation_type: str, is_reversed: bool) -> str:
        """
        Get relation description

        Args:
            relation_type: Relation type
            is_reversed: Whether semantics need to be reversed

        Returns:
            Description
        """
        if is_reversed:
            # Reverse relation (late paper -> early paper) needs reversed semantics
            descriptions = {
                'Overcomes': 'improved upon predecessors\' limitations',
                'Realizes': 'realized predecessors\' vision',
                'Extends': 'extended predecessors\' methods',
                'Alternative': 'proposed alternative approach different from predecessors',
                'Adapts_to': 'transferred predecessors\' technology to new domain',
                'Baselines': 'building on previous work',
            }
        else:
            # Forward relation (early paper -> late paper)
            descriptions = {
                'Inspires': 'inspired subsequent research',
                'Proposes': 'proposed direction for subsequent research',
                'Enables': 'enabled subsequent research',
            }

        return descriptions.get(relation_type, 'building on previous work')

    def _extract_relation_stats(self, path: Dict, graph: nx.DiGraph) -> Dict:
        """
        Extract relation type statistics from evolutionary path

        Args:
            path: Evolutionary path
            graph: Graph

        Returns:
            Relation statistics dictionary
        """
        relation_count = {}

        if path['thread_type'] == 'chain':
            # Chain: count relations between adjacent papers
            papers = path['papers']
            for i in range(len(papers) - 1):
                paper_id = papers[i]['paper_id']
                next_paper_id = papers[i + 1]['paper_id']

                if graph.has_edge(paper_id, next_paper_id):
                    edge_data = graph.edges[paper_id, next_paper_id]
                    edge_type = edge_data.get('type') or edge_data.get('edge_type', 'Unknown')
                    relation_count[edge_type] = relation_count.get(edge_type, 0) + 1

        elif path['thread_type'] in ['divergence', 'convergence']:
            # Divergence/convergence structure: count relation types for each route
            for route in path.get('routes', []):
                relation_type = route.get('relation_type', 'Unknown')
                relation_count[relation_type] = relation_count.get(relation_type, 0) + 1

        return {
            'total_relations': sum(relation_count.values()),
            'relation_distribution': relation_count,
            'dominant_relation': max(relation_count.items(), key=lambda x: x[1])[0] if relation_count else 'Unknown'
        }

    def _extract_key_method(self, node_data: Dict) -> str:
        """Extract key method from paper"""
        # Prioritize extraction from deep_analysis
        deep_analysis = node_data.get('deep_analysis', {})
        if isinstance(deep_analysis, dict):
            method = deep_analysis.get('method', {})
            # Compatible with two formats: dictionary {content: ...} or direct string
            if isinstance(method, dict):
                method_text = method.get('content', '')
            else:
                method_text = str(method) if method else ''

            if method_text:
                # Only take first sentence to avoid being too long
                first_sentence = method_text.split('\n')[0].strip()
                return first_sentence[:80] if len(first_sentence) > 80 else first_sentence

        # Extract from rag_analysis
        rag_analysis = node_data.get('rag_analysis', {})
        if isinstance(rag_analysis, dict):
            method = rag_analysis.get('method', '')
            if method:
                first_sentence = str(method).split('\n')[0].strip()
                return first_sentence[:80] if len(first_sentence) > 80 else first_sentence

        return 'Method not extracted'

    def _extract_key_problem(self, node_data: Dict) -> str:
        """Extract key problem from paper"""
        deep_analysis = node_data.get('deep_analysis', {})
        if isinstance(deep_analysis, dict):
            problem = deep_analysis.get('problem', {})
            # Compatible with two formats: dictionary {content: ...} or direct string
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

        return 'Problem not extracted'

    def _extract_key_limitation(self, node_data: Dict) -> str:
        """Extract key limitation from paper"""
        deep_analysis = node_data.get('deep_analysis', {})
        if isinstance(deep_analysis, dict):
            limitation = deep_analysis.get('limitation', {})
            # Compatible with two formats: dictionary {content: ...} or direct string
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

        return 'Limitation not extracted'

    def _extract_key_future_work(self, node_data: Dict) -> str:
        """
        Extract future work direction from paper (for Realizes relation)

        Args:
            node_data: Paper node data

        Returns:
            Future work description (only first sentence, limited to 80 characters)
        """
        # Prioritize extraction from deep_analysis
        deep_analysis = node_data.get('deep_analysis', {})
        if isinstance(deep_analysis, dict):
            future_work = deep_analysis.get('future_work', {})
            # Compatible with two formats: dictionary {content: ...} or direct string
            if isinstance(future_work, dict):
                future_work_text = future_work.get('content', '')
            else:
                future_work_text = str(future_work) if future_work else ''

            # Filter out "N/A" or empty values
            if future_work_text and future_work_text != "N/A":
                first_sentence = future_work_text.split('.')[0].strip()
                if not first_sentence:
                    first_sentence = future_work_text.split('\n')[0].strip()
                return first_sentence[:80] if len(first_sentence) > 80 else first_sentence

        # Fallback: extract from rag_analysis
        rag_analysis = node_data.get('rag_analysis', {})
        if isinstance(rag_analysis, dict):
            future_work = rag_analysis.get('future_work', '')
            if future_work and future_work != "N/A":
                first_sentence = str(future_work).split('.')[0].strip()
                if not first_sentence:
                    first_sentence = str(future_work).split('\n')[0].strip()
                return first_sentence[:80] if len(first_sentence) > 80 else first_sentence

        return 'Future work not extracted'

    def _extract_papers_info_for_relation(
        self,
        graph: nx.DiGraph,
        prev_paper_id: str,
        curr_paper_id: str,
        relation_type: str
    ) -> Dict[str, str]:
        """
        Extract relevant information from papers based on citation type

        Args:
            graph: Graph
            prev_paper_id: Previous (early) paper ID
            curr_paper_id: Current (late) paper ID
            relation_type: Citation relation type

        Returns:
            Dictionary containing information needed for narrative
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

        # Extract different elements based on citation type
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
            # Adapts_to: A.Problem+Method → B.Problem+Method (cross-domain)
            info['prev_problem'] = self._extract_key_problem(prev_node)
            info['prev_method'] = self._extract_key_method(prev_node)
            info['curr_problem'] = self._extract_key_problem(curr_node)
            info['curr_method'] = self._extract_key_method(curr_node)
            # Try to extract domain information from deep_analysis
            prev_deep = prev_node.get('deep_analysis', {})
            curr_deep = curr_node.get('deep_analysis', {})
            info['prev_domain'] = prev_deep.get('domain', 'original domain')
            info['curr_domain'] = curr_deep.get('domain', 'new domain')

        elif relation_type == 'Extends':
            # Extends: A.Method → B.Method (extension)
            info['prev_method'] = self._extract_key_method(prev_node)
            info['curr_method'] = self._extract_key_method(curr_node)
            info['curr_problem'] = self._extract_key_problem(curr_node)

        elif relation_type == 'Alternative':
            # Alternative: A.Method → B.Method (different paradigm)
            info['prev_method'] = self._extract_key_method(prev_node)
            info['curr_method'] = self._extract_key_method(curr_node)
            info['prev_problem'] = self._extract_key_problem(prev_node)
            info['curr_problem'] = self._extract_key_problem(curr_node)

        else:  # Baselines or other
            # Extract general information by default
            info['prev_method'] = self._extract_key_method(prev_node)
            info['curr_method'] = self._extract_key_method(curr_node)
            info['curr_problem'] = self._extract_key_problem(curr_node)

        return info

    def _generate_relation_narrative_fragment(self, info: Dict[str, str]) -> str:
        """
        Generate targeted narrative fragment based on citation type

        Args:
            info: Information dictionary extracted via _extract_papers_info_for_relation

        Returns:
            Narrative fragment text
        """
        relation_type = info.get('relation_type', 'Baselines')
        curr_year = info.get('curr_year', 'N/A')
        curr_title = info.get('curr_title', 'Unknown')

        if relation_type == 'Overcomes':
            # Focus: A's limitation → How B solves it
            prev_limitation = info.get('prev_limitation', 'certain limitations')
            curr_method = info.get('curr_method', 'new method')
            return (
                f"**Overcoming Limitations** ({curr_year}): Addressing the shortcomings of previous work in '{prev_limitation}', "
                f"the paper '{curr_title}' achieved breakthrough improvements through {curr_method}."
            )

        elif relation_type == 'Realizes':
            # Focus: A's vision → How B realizes it
            prev_future_work = info.get('prev_future_work', 'predecessors\' vision')
            curr_method = info.get('curr_method', 'new method')
            return (
                f"**Realizing Vision** ({curr_year}): Responding to predecessors' outlook of '{prev_future_work}', "
                f"the paper '{curr_title}' put this vision into practice through {curr_method}."
            )

        elif relation_type == 'Adapts_to':
            # Focus: A's domain → B's domain
            prev_domain = info.get('prev_domain', 'original domain')
            curr_domain = info.get('curr_domain', 'new domain')
            curr_method = info.get('curr_method', 'improved method')
            return (
                f"**Cross-Domain Transfer** ({curr_year}): The paper '{curr_title}' successfully transferred "
                f"predecessors' technology in '{prev_domain}' to '{curr_domain}', validating the method's generalization ability through {curr_method}."
            )

        elif relation_type == 'Extends':
            # Focus: A's method → How B enhances it
            prev_method = info.get('prev_method', 'basic method')
            curr_method = info.get('curr_method', 'improved method')
            return (
                f"**Incremental Extension** ({curr_year}): Building on predecessors' '{prev_method}', "
                f"the paper '{curr_title}' achieved progressive improvement through {curr_method}."
            )

        elif relation_type == 'Alternative':
            # Focus: A's paradigm → B's different paradigm
            prev_method = info.get('prev_method', 'original method')
            curr_method = info.get('curr_method', 'alternative method')
            return (
                f"**Alternative Approach** ({curr_year}): Unlike predecessors' '{prev_method}' approach, "
                f"the paper '{curr_title}' proposed {curr_method}, exploring a new paradigm for solving the problem."
            )

        else:  # Baselines
            # Lightweight description
            curr_method = info.get('curr_method', 'new method')
            return (
                f"**Evolution** ({curr_year}): Building on previous work, "
                f"the paper '{curr_title}' advanced this direction through {curr_method}."
            )
