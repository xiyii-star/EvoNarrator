"""
Topic Evolution Analysis Module
Topic Evolution Analyzer

Responsible for analyzing the development trajectory of research topics in the knowledge graph, including:
1. Temporal evolution analysis
2. Key node identification (milestone papers)
3. Research branch analysis (community detection)
4. Citation chain analysis
5. Innovation pattern analysis
6. Critical Evolutionary Path Extraction
7. Technical Bifurcation Detection
8. Open Frontier Detection
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, Counter
from datetime import datetime
import re

try:
    import networkx as nx
except ImportError:
    raise ImportError("networkx installation required: pip install networkx")

logger = logging.getLogger(__name__)


# Evolutionary Momentum Weight Definition (Evolutionary Momentum Scores)
EVOLUTIONARY_WEIGHTS = {
    'Overcomes': 3.0,      # Qualitative change: Solved predecessors' defects
    'Realizes': 2.5,       # Filling gaps: Implemented predecessors' Future Work
    'Extends': 1.0,        # Quantitative change: Performance improvement
    'Alternative': 1.0,    # Side branch: Alternative approach
    'Adapts_to': 0.5,      # Transfer: Horizontal diffusion
    'Baselines': 0.1,      # Background noise: Almost ignored
    'Unknown': 0.3         # Unknown type: Low weight
}


class TopicEvolutionAnalyzer:
    """
    Topic Evolution Analyzer

    Analyzes the evolution patterns of research topics based on knowledge graph (NetworkX Graph)
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the analyzer

        Args:
            config: Configuration dictionary containing topic_evolution related settings
        """
        # Default configuration
        self.config = config or {}

        # Extract topic_evolution configuration
        evolution_config = self.config.get('topic_evolution', {})

        # Milestone paper configuration
        milestone_config = evolution_config.get('milestone', {})
        self.milestone_top_count = milestone_config.get('top_count', 10)
        self.milestone_citation_weight = milestone_config.get('citation_weight', 0.5)
        self.milestone_pagerank_weight = milestone_config.get('pagerank_weight', 1000)
        self.milestone_betweenness_weight = milestone_config.get('betweenness_weight', 500)
        self.milestone_display_count = milestone_config.get('display_count', 3)

        # Research branch configuration
        branch_config = evolution_config.get('branch', {})
        self.branch_min_size = branch_config.get('min_size', 3)
        self.branch_top_keywords = branch_config.get('top_keywords', 5)
        self.branch_display_count = branch_config.get('display_count', 3)
        self.branch_min_avg_citations = branch_config.get('min_avg_citations', 0)

        # Citation chain configuration
        chain_config = evolution_config.get('citation_chain', {})
        self.chain_max_chains = chain_config.get('max_chains', 5)
        self.chain_min_length = chain_config.get('min_length', 3)
        self.chain_start_from_top = chain_config.get('start_from_top_milestones', 5)

        # Time evolution configuration
        time_config = evolution_config.get('time_evolution', {})
        self.time_top_papers_per_year = time_config.get('top_papers_per_year', 3)
        self.time_include_citation_types = time_config.get('include_citation_types', True)

        # Innovation pattern configuration
        pattern_config = evolution_config.get('innovation_pattern', {})
        self.pattern_examples_per_type = pattern_config.get('examples_per_type', 3)
        self.pattern_sort_by_count = pattern_config.get('sort_by_count', True)

        # Keyword extraction configuration
        keyword_config = evolution_config.get('keyword_extraction', {})
        self.keyword_min_length = keyword_config.get('min_word_length', 3)
        self.keyword_remove_stopwords = keyword_config.get('remove_stopwords', True)
        self.keyword_case_sensitive = keyword_config.get('case_sensitive', False)

        # Critical evolutionary path configuration
        evolutionary_path_config = evolution_config.get('evolutionary_path', {})
        self.evol_enabled = evolutionary_path_config.get('enabled', True)
        self.evol_max_paths = evolutionary_path_config.get('max_paths', 3)
        self.evol_min_weight = evolutionary_path_config.get('min_total_weight', 3.0)
        self.evol_time_window_years = evolutionary_path_config.get('time_window_years', None)
        self.evol_custom_weights = evolutionary_path_config.get('custom_weights', {}) or {}

        # Merge custom weights
        self.evolutionary_weights = EVOLUTIONARY_WEIGHTS.copy()
        if self.evol_custom_weights:
            self.evolutionary_weights.update(self.evol_custom_weights)

        # Technical bifurcation detection configuration
        bifurcation_config = evolution_config.get('bifurcation', {})
        self.bifur_enabled = bifurcation_config.get('enabled', True)
        self.bifur_max_bifurcations = bifurcation_config.get('max_bifurcations', 5)
        self.bifur_fork_edge_types = bifurcation_config.get('fork_edge_types', ['Alternative', 'Extends'])
        self.bifur_min_children = bifurcation_config.get('min_children', 2)
        self.bifur_method_sim_threshold = bifurcation_config.get('method_similarity_threshold', 0.3)
        self.bifur_problem_sim_threshold = bifurcation_config.get('problem_similarity_threshold', 0.6)
        self.bifur_use_cosine = bifurcation_config.get('use_cosine_similarity', True)

        # Open frontier detection configuration
        frontier_config = evolution_config.get('open_frontier', {})
        self.frontier_enabled = frontier_config.get('enabled', True)
        self.frontier_recent_years = frontier_config.get('recent_years', 2)
        self.frontier_max_open_problems = frontier_config.get('max_open_problems', 10)
        self.frontier_max_ideas = frontier_config.get('max_cross_domain_ideas', 5)
        self.frontier_lim_sim_threshold = frontier_config.get('limitation_similarity_threshold', 0.5)
        self.frontier_min_contrib_score = frontier_config.get('min_contribution_score', 0.3)

        logger.info(f"TopicEvolutionAnalyzer initialization completed")
        logger.info(f"  Milestone papers: Top {self.milestone_top_count}, display {self.milestone_display_count}")
        logger.info(f"  Research branches: Min size {self.branch_min_size}, keywords {self.branch_top_keywords}")
        logger.info(f"  Citation chains: Max {self.chain_max_chains} chains, min length {self.chain_min_length}")
        logger.info(f"  Critical evolutionary paths: {'Enabled' if self.evol_enabled else 'Disabled'}, max {self.evol_max_paths} paths")
        logger.info(f"  Technical bifurcation detection: {'Enabled' if self.bifur_enabled else 'Disabled'}, max {self.bifur_max_bifurcations} bifurcations")
        logger.info(f"  Open frontier detection: {'Enabled' if self.frontier_enabled else 'Disabled'}, max {self.frontier_max_open_problems} problems")

    def analyze(self, graph: nx.DiGraph, topic: str) -> Dict:
        """
        Execute complete topic evolution analysis (dual-core directions)

        Core Direction 1: Retrospective Analysis
        Core Direction 2: Future Prediction

        Args:
            graph: NetworkX directed graph with paper information in nodes
            topic: Research topic name

        Returns:
            Analysis report dictionary
        """
        if len(graph.nodes()) == 0:
            logger.warning("Knowledge graph is empty, skipping topic evolution analysis")
            return {}

        logger.info(f"Starting topic evolution analysis: '{topic}'")
        logger.info(f"  Graph size: {len(graph.nodes())} nodes, {len(graph.edges())} edges")

        # Basic analysis
        year_stats = self._analyze_time_evolution(graph)
        milestone_papers = self._identify_milestone_papers(graph)

        # ========== Core Direction 1: Retrospective Analysis ==========
        logger.info("\n  🔙 Core Direction 1: Retrospective Analysis...")

        # 1.1 Identify evolutionary backbone vs incremental patches
        logger.info("    📍 Identifying evolutionary backbone vs incremental patches...")
        backbone_analysis = self._analyze_backbone_vs_incremental(graph)

        # 1.2 Identify technical bifurcations
        logger.info("    🔀 Identifying technical bifurcations...")
        bifurcations = self._detect_technical_bifurcations(graph) if self.bifur_enabled else []

        # 1.3 Identify cross-domain invasions
        logger.info("    🌐 Identifying cross-domain invasions...")
        cross_domain_invasions = self._detect_cross_domain_invasions(graph)

        # ========== Core Direction 2: Future Prediction ==========
        logger.info("\n  🔮 Core Direction 2: Future Prediction...")

        # 2.1 Level 1: Low-hanging fruit ideas (unrealized Future Work)
        logger.info("    💡 Level 1: Low-hanging fruit ideas...")
        low_hanging_fruits = self._detect_low_hanging_fruits(graph, year_stats)

        # 2.2 Level 2: Hard nut ideas (unsolved Limitations)
        logger.info("    🔨 Level 2: Hard nut ideas...")
        hard_nuts = self._detect_hard_nuts(graph, milestone_papers)

        # 2.3 Level 3: Innovative ideas (cross-domain transfer/hybrid methods)
        logger.info("    🚀 Level 3: Innovative ideas...")
        innovative_ideas = self._generate_innovative_ideas(graph)

        # Generate report
        report = {
            'topic': topic,
            'analysis_time': datetime.now().isoformat(),
            'graph_overview': {
                'total_papers': len(graph.nodes()),
                'total_citations': len(graph.edges()),
                'year_range': f"{min(year_stats.keys())}-{max(year_stats.keys())}" if year_stats else "Unknown"
            },

            # Core Direction 1: Retrospective Analysis
            'retrospective_analysis': {
                'backbone_vs_incremental': backbone_analysis,
                'technical_bifurcations': bifurcations,
                'cross_domain_invasions': cross_domain_invasions
            },

            # Core Direction 2: Future Prediction
            'future_prediction': {
                'level_1_low_hanging_fruits': low_hanging_fruits,
                'level_2_hard_nuts': hard_nuts,
                'level_3_innovative_ideas': innovative_ideas
            },

            # Retain original analysis (compatibility)
            'milestone_papers': milestone_papers,
            'time_evolution': dict(sorted(year_stats.items()))
        }

        # Output summary information
        self._log_summary(report)

        return report

    def _analyze_time_evolution(self, graph: nx.DiGraph) -> Dict:
        """
        Analyze temporal evolution

        Returns:
            Year statistics dictionary
        """
        year_stats = defaultdict(lambda: {
            'papers': [],
            'citation_types': defaultdict(int),
            'avg_citations': 0
        })

        # Collect papers for each year
        for node_id, node_data in graph.nodes(data=True):
            year = node_data.get('year')
            if year:
                year_stats[year]['papers'].append({
                    'id': node_id,
                    'title': node_data.get('title', ''),
                    'cited_by_count': node_data.get('cited_by_count', 0)
                })

        # Calculate statistics for each year
        for year, stats in year_stats.items():
            if stats['papers']:
                # Average citation count
                stats['avg_citations'] = sum(
                    p['cited_by_count'] for p in stats['papers']
                ) / len(stats['papers'])

                # Sort by citation count, keep only top N
                stats['papers'] = sorted(
                    stats['papers'],
                    key=lambda x: x['cited_by_count'],
                    reverse=True
                )[:self.time_top_papers_per_year]

        # Count citation type distribution for each year
        if self.time_include_citation_types:
            for source, target, edge_data in graph.edges(data=True):
                source_year = graph.nodes[source].get('year')
                edge_type = edge_data.get('edge_type', 'Unknown')
                if source_year:
                    year_stats[source_year]['citation_types'][edge_type] += 1

        return year_stats

    def _identify_milestone_papers(self, graph: nx.DiGraph) -> List[Dict]:
        """
        Identify milestone papers

        Using comprehensive scoring: citation count + PageRank + betweenness centrality

        Returns:
            List of milestone papers
        """
        # Calculate node importance metrics
        try:
            pagerank = nx.pagerank(graph, alpha=0.85)
            betweenness = nx.betweenness_centrality(graph)
        except Exception as e:
            logger.warning(f"Failed to calculate graph metrics: {e}, using default values")
            pagerank = {node: 0 for node in graph.nodes()}
            betweenness = {node: 0 for node in graph.nodes()}

        # Comprehensive scoring
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

        # Sort by comprehensive score
        milestone_papers = sorted(
            milestone_papers,
            key=lambda x: x['score'],
            reverse=True
        )[:self.milestone_top_count]

        return milestone_papers

    def _analyze_research_branches(self, graph: nx.DiGraph) -> List[Dict]:
        """
        Analyze research branches (community detection)

        Using Louvain algorithm for community detection

        Returns:
            List of research branches
        """
        try:
            # Use Louvain algorithm for community detection
            communities = nx.community.louvain_communities(graph.to_undirected())

            research_branches = []
            for i, community in enumerate(communities):
                if len(community) < self.branch_min_size:
                    continue

                # Analyze branch characteristics
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

                # Calculate average citation count
                avg_citations = sum(branch_citations) / len(branch_citations) if branch_citations else 0

                # Filter low-quality branches
                if avg_citations < self.branch_min_avg_citations:
                    continue

                # Identify branch keywords
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

            # Sort by size
            research_branches = sorted(
                research_branches,
                key=lambda x: x['size'],
                reverse=True
            )

            return research_branches

        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            return []

    def _analyze_citation_chains(
        self,
        graph: nx.DiGraph,
        milestone_papers: List[Dict]
    ) -> List[Dict]:
        """
        Analyze citation chains (citation inheritance paths)

        Starting from milestone papers, find the longest citation chains

        Args:
            graph: Knowledge graph
            milestone_papers: List of milestone papers

        Returns:
            List of citation chains
        """
        citation_chains = []

        try:
            # Track from top N milestone papers
            for start_node in milestone_papers[:self.chain_start_from_top]:
                start_id = start_node['id']
                if start_id not in graph:
                    continue

                # Find the longest path from this node
                lengths = nx.single_source_shortest_path_length(graph, start_id)
                if not lengths:
                    continue

                farthest_node = max(lengths.items(), key=lambda x: x[1])

                # Check path length
                if farthest_node[1] < self.chain_min_length - 1:  # -1 because length is edge count
                    continue

                # Get path
                path = nx.shortest_path(graph, start_id, farthest_node[0])
                if len(path) < self.chain_min_length:
                    continue

                # Build chain information
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
            logger.warning(f"Citation chain analysis failed: {e}")

        # Sort by length, take top N
        citation_chains = sorted(
            citation_chains,
            key=lambda x: x['length'],
            reverse=True
        )[:self.chain_max_chains]

        return citation_chains

    def _analyze_innovation_patterns(self, graph: nx.DiGraph) -> Dict:
        """
        Analyze innovation patterns (citation type statistics)

        Count distribution and examples of different citation types

        Returns:
            Innovation pattern dictionary
        """
        innovation_patterns = defaultdict(lambda: {
            'count': 0,
            'examples': []
        })

        for source, target, edge_data in graph.edges(data=True):
            edge_type = edge_data.get('edge_type', 'Unknown')
            innovation_patterns[edge_type]['count'] += 1

            # Save examples
            if len(innovation_patterns[edge_type]['examples']) < self.pattern_examples_per_type:
                source_data = graph.nodes[source]
                target_data = graph.nodes[target]
                innovation_patterns[edge_type]['examples'].append({
                    'from': source_data.get('title', '')[:50],
                    'to': target_data.get('title', '')[:50],
                    'from_year': source_data.get('year'),
                    'to_year': target_data.get('year')
                })

        # Convert to regular dictionary
        return {k: dict(v) for k, v in innovation_patterns.items()}

    def _extract_keywords(self, titles: List[str], top_k: int = 5) -> List[str]:
        """
        Extract keywords from title list

        Args:
            titles: List of titles
            top_k: Return top k keywords

        Returns:
            List of keywords
        """
        # Stopwords
        stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'using', 'based', 'via',
            'through', 'into', 'over', 'after', 'before', 'between', 'under'
        } if self.keyword_remove_stopwords else set()

        # Extract all words
        words = []
        for title in titles:
            # Convert to lowercase (if case-insensitive)
            title_processed = title if self.keyword_case_sensitive else title.lower()

            # Extract words (minimum length)
            pattern = rf'\b[a-zA-Z]{{{self.keyword_min_length},}}\b'
            if not self.keyword_case_sensitive:
                pattern = rf'\b[a-z]{{{self.keyword_min_length},}}\b'

            words_in_title = re.findall(pattern, title_processed)
            words.extend([w for w in words_in_title if w not in stopwords])

        # Count frequency
        word_counts = Counter(words)

        # Return top k keywords
        return [word for word, count in word_counts.most_common(top_k)]

    def _analyze_backbone_vs_incremental(self, graph: nx.DiGraph) -> Dict:
        """
        Identify evolutionary backbone vs incremental patches (enhanced version)

        Core insights:
        1. Backbone paths: Only keep Overcomes and Realizes type connections
           - Highest value evolutionary lines
           - Each node solves critical defects of predecessors

        2. Incremental paths: Only keep Extends type connections
           - Involution/benchmark-chasing evolutionary lines
           - Fine-tuning optimizations under the same methodology

        3. Breakthrough points: Suddenly jumping from Extends involution to Overcomes
           - These are the most valuable innovation points
           - Represent methodological breakthroughs

        Returns:
            Detailed backbone vs incremental analysis report
        """
        backbone_paths = []
        incremental_paths = []
        breakthrough_points = []

        # Count input/output edge types for each node
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

        # Build backbone paths and incremental paths
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

        # Identify breakthrough points: jumping from Extends involution to Overcomes
        # Criteria: multiple Extends inputs, Overcomes/Realizes outputs
        for node, stats in node_stats.items():
            # Determine if it's a breakthrough point
            is_breakthrough = False
            breakthrough_score = 0

            # Pattern 1: Jumping from Extends involution to Overcomes (most classic)
            if stats['in_extends'] >= 2 and stats['out_overcomes'] >= 1:
                is_breakthrough = True
                breakthrough_score = stats['in_extends'] * 1.0 + stats['out_overcomes'] * 3.0

            # Pattern 2: From pure Extends to Realizes (gap-filling breakthrough)
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
                    'breakthrough_type': 'Overcomes breakthrough' if stats['out_overcomes'] > 0 else 'Realizes gap-filling'
                })

        # Sort by breakthrough score
        breakthrough_points = sorted(
            breakthrough_points,
            key=lambda x: x['breakthrough_score'],
            reverse=True
        )[:10]

        # Analyze backbone path coherence
        backbone_chains = self._extract_backbone_chains(graph, backbone_paths)

        # Analyze incremental path bottlenecks
        incremental_bottlenecks = self._analyze_incremental_bottlenecks(graph, incremental_paths)

        return {
            'summary': {
                'backbone_count': len(backbone_paths),
                'incremental_count': len(incremental_paths),
                'breakthrough_count': len(breakthrough_points),
                'ratio': len(backbone_paths) / len(incremental_paths) if len(incremental_paths) > 0 else 0
            },
            'backbone_paths': backbone_paths[:20],  # Return top 20 backbone paths
            'incremental_paths': incremental_paths[:20],  # Return top 20 incremental paths
            'breakthrough_points': breakthrough_points,
            'backbone_chains': backbone_chains,  # Backbone continuous chains
            'incremental_bottlenecks': incremental_bottlenecks  # Incremental bottlenecks
        }

    def _extract_backbone_chains(self, graph: nx.DiGraph, backbone_paths: List[Dict]) -> List[Dict]:
        """
        Extract backbone continuous chains: consecutive Overcomes/Realizes paths

        These chains represent "hardcore breakthrough" evolutionary lines
        """
        # Build subgraph containing only backbone edges
        backbone_graph = nx.DiGraph()

        for path in backbone_paths:
            source = path['from']['id']
            target = path['to']['id']
            edge_type = path['type']
            backbone_graph.add_edge(source, target, edge_type=edge_type)

            # Add node attributes
            for node_id in [source, target]:
                if node_id in graph.nodes():
                    node_data = graph.nodes[node_id]
                    backbone_graph.add_node(node_id, **node_data)

        # Find all simple paths (length >= 3)
        chains = []

        # Find source nodes (nodes with in-degree 0)
        source_nodes = [n for n in backbone_graph.nodes() if backbone_graph.in_degree(n) == 0]
        # Find target nodes (nodes with out-degree 0)
        target_nodes = [n for n in backbone_graph.nodes() if backbone_graph.out_degree(n) == 0]

        for source in source_nodes:
            for target in target_nodes:
                if source == target:
                    continue

                if nx.has_path(backbone_graph, source, target):
                    # Find longest path
                    try:
                        path = nx.shortest_path(backbone_graph, source, target)

                        if len(path) >= 3:  # At least 3 nodes
                            chain_info = []
                            edge_types = []

                            for i, node in enumerate(path):
                                node_data = graph.nodes[node]
                                chain_info.append({
                                    'id': node,
                                    'title': node_data.get('title', '')[:50],
                                    'year': node_data.get('year')
                                })

                                # Get edge type
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

        # Sort by length
        chains = sorted(chains, key=lambda x: x['length'], reverse=True)[:5]

        return chains

    def _analyze_incremental_bottlenecks(self, graph: nx.DiGraph, incremental_paths: List[Dict]) -> List[Dict]:
        """
        Analyze incremental path bottlenecks: find papers heavily cited by Extends but without subsequent breakthroughs

        These papers represent "involution endpoints", possibly approaching bottlenecks
        """
        # Build subgraph containing only Extends edges
        extends_graph = nx.DiGraph()

        for path in incremental_paths:
            source = path['from']['id']
            target = path['to']['id']
            extends_graph.add_edge(source, target)

            # Add node attributes
            for node_id in [source, target]:
                if node_id in graph.nodes():
                    node_data = graph.nodes[node_id]
                    extends_graph.add_node(node_id, **node_data)

        bottlenecks = []

        for node in extends_graph.nodes():
            in_degree = extends_graph.in_degree(node)

            # Heavily cited by Extends (at least 3)
            if in_degree >= 3:
                # Check if there's a subsequent breakthrough (in original graph)
                has_breakthrough = False

                for pred in graph.predecessors(node):
                    edge_data = graph[pred][node]
                    edge_type = edge_data.get('edge_type', 'Unknown')

                    if edge_type in ['Overcomes', 'Realizes']:
                        has_breakthrough = True
                        break

                # Only bottlenecks without breakthroughs
                if not has_breakthrough:
                    node_data = graph.nodes[node]
                    bottlenecks.append({
                        'node': node,
                        'title': node_data.get('title', '')[:60],
                        'year': node_data.get('year'),
                        'cited_by_count': node_data.get('cited_by_count', 0),
                        'extends_in_count': in_degree,
                        'reason': f'Cited by {in_degree} Extends but no subsequent breakthrough, possibly reached bottleneck'
                    })

        # Sort by Extends citation count
        bottlenecks = sorted(bottlenecks, key=lambda x: x['extends_in_count'], reverse=True)[:5]

        return bottlenecks

    def _detect_cross_domain_invasions(self, graph: nx.DiGraph) -> List[Dict]:
        """
        Identify cross-domain invasions (Cross-Domain Invasion) - enhanced version

        Core idea:
        Track Adapts_to type connections to identify cross-domain methodology transfers.
        These nodes typically represent dimensional reduction attacks by methodologies, an important external infusion for this topic.

        Enhanced analysis:
        1. Track the origins of these nodes (original domain)
        2. Analyze transfer impact (number of subsequent works)
        3. Identify most successful cross-domain transfer cases

        Returns:
            List of cross-domain invasions (including impact analysis)
        """
        invasions = []

        for source, target, data in graph.edges(data=True):
            if data.get('edge_type') == 'Adapts_to':
                source_data = graph.nodes[source]
                target_data = graph.nodes[target]

                # Analyze transfer impact
                # 1. Count source's subsequent works (post-transfer impact)
                source_descendants = list(nx.descendants(graph, source)) if source in graph else []
                impact_count = len(source_descendants)

                # 2. Count target's original impact (foundation being transferred)
                target_citations = target_data.get('cited_by_count', 0)

                # 3. Determine transfer success level
                if impact_count > 10:
                    success_level = 'highly_successful'  # Highly successful
                elif impact_count > 5:
                    success_level = 'successful'  # Successful
                elif impact_count > 0:
                    success_level = 'moderate'  # Moderate
                else:
                    success_level = 'limited'  # Limited

                # 4. Try to identify original domain and target domain (based on keywords)
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

        # Sort by impact
        invasions = sorted(invasions, key=lambda x: x['impact_analysis']['descendants_count'], reverse=True)[:15]

        return invasions

    def _extract_domain_keywords(self, title: str) -> List[str]:
        """
        Extract domain keywords from title

        Args:
            title: Paper title

        Returns:
            List of keywords
        """
        # Common domain keywords
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
                    break  # One match is enough

        # If not found, return some generic keywords extracted from title
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
        Generate cross-domain transfer story description

        Args:
            source_keywords: Source domain keywords
            target_keywords: Target domain keywords
            year: Transfer year
            impact: Impact (number of subsequent works)
            success: Success level

        Returns:
            Description text
        """
        source_str = ', '.join(source_keywords[:2]) if source_keywords else 'unknown domain'
        target_str = ', '.join(target_keywords[:2]) if target_keywords else 'unknown domain'

        success_desc = {
            'highly_successful': f'generated {impact} subsequent works, becoming an important breakthrough in the field',
            'successful': f'generated {impact} subsequent works, achieving good development',
            'moderate': f'generated {impact} subsequent works, with some impact',
            'limited': 'limited subsequent development'
        }

        year_str = f"In {year}, " if year else ""

        return f"{year_str}transferred methods from [{target_str}] domain to [{source_str}] domain, {success_desc.get(success, 'impact unknown')}"

    def _detect_low_hanging_fruits(self, graph: nx.DiGraph, year_stats: Dict) -> List[Dict]:
        """
        Level 1: Low-hanging fruit ideas - find unrealized Future Work (enhanced version)

        Algorithm logic:
        1. Extract papers published in the last 3 years ("frontier")
        2. Extract Future_Work sections from their text
        3. Confirm whether subsequent papers have connected to it via Realizes
        4. If not, "implementing this paper's Future Work" is a ready-made idea

        Enhancements:
        - Sort by paper impact (citation count), prioritize high-impact papers' Future Work
        - Analyze Future Work feasibility (length, specificity)
        - Provide implementation difficulty assessment

        Returns:
            List of low-hanging fruit ideas
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

            # Get future_work from deep_analysis structure
            deep_analysis = node_data.get('deep_analysis', {})
            future_work = deep_analysis.get('future_work', {}).get('content', '')

            if not future_work or len(future_work) < 20:
                continue

            # Check if subsequent work has implemented it via Realizes
            has_realization = False
            for pred in graph.predecessors(node):
                edge_data = graph[pred][node]
                if edge_data.get('edge_type') == 'Realizes':
                    has_realization = True
                    break

            if not has_realization:
                # Assess feasibility (based on description length and specificity)
                feasibility_score = self._assess_idea_feasibility(future_work)

                # Assess difficulty
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
                    'priority': node_data.get('cited_by_count', 0) * feasibility_score,  # Comprehensive priority
                    'recommendation': self._generate_implementation_recommendation(future_work, difficulty)
                })

        # Sort by priority (citation count * feasibility)
        low_hanging = sorted(low_hanging, key=lambda x: x['priority'], reverse=True)[:15]

        return low_hanging

    def _assess_idea_feasibility(self, future_work: str) -> float:
        """
        Assess Future Work feasibility

        Args:
            future_work: Future Work description

        Returns:
            Feasibility score (0-1)
        """
        score = 0.5  # Base score

        # Longer length means more specific
        if len(future_work) > 100:
            score += 0.2
        elif len(future_work) > 200:
            score += 0.3

        # Contains specific keywords
        action_keywords = ['apply', 'extend', 'improve', 'combine', 'test', 'evaluate', 'implement']
        for kw in action_keywords:
            if kw in future_work.lower():
                score += 0.1
                break

        # Contains specific methods or datasets
        specific_keywords = ['dataset', 'benchmark', 'algorithm', 'model', 'framework']
        for kw in specific_keywords:
            if kw in future_work.lower():
                score += 0.1
                break

        return min(score, 1.0)

    def _assess_implementation_difficulty(self, future_work: str, paper_data: Dict) -> str:
        """
        Assess implementation difficulty

        Args:
            future_work: Future Work description
            paper_data: Paper data

        Returns:
            Difficulty level: 'easy', 'medium', 'hard'
        """
        # Determine difficulty based on keywords
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
        Generate implementation recommendation

        Args:
            future_work: Future Work description
            difficulty: Difficulty level

        Returns:
            Implementation recommendation text
        """
        difficulty_desc = {
            'easy': 'Low difficulty, can be quickly verified',
            'medium': 'Medium difficulty, requires some engineering implementation',
            'hard': 'High difficulty, requires in-depth research and innovation'
        }

        return f"{difficulty_desc.get(difficulty, 'difficulty unknown')}. It is recommended to read the original paper first, understand its core method, and then extend it."

    def _detect_hard_nuts(self, graph: nx.DiGraph, milestone_papers: List[Dict]) -> List[Dict]:
        """
        Level 2: Hard nut ideas - find Limitations not yet Overcome (enhanced version)

        Algorithm logic:
        1. Find several "cornerstone papers" with highest citations but relatively recent publication in this field
        2. Extract their Limitations
        3. Check if any new papers have connected to them via Overcomes
        4. If not yet, it means although everyone is citing it, its core defects still exist

        Enhancements:
        - Assess Limitation severity and impact
        - Analyze why it hasn't been solved yet (technical difficulty, resource requirements, etc.)
        - Provide breakthrough recommendations

        Your idea: Propose a new Method specifically targeting this unsolved Limitation

        Returns:
            List of hard nut ideas
        """
        hard_nuts = []

        for milestone in milestone_papers[:15]:  # Expand to top 15 papers
            node_id = milestone['id']
            if node_id not in graph:
                continue

            node_data = graph.nodes[node_id]

            # Get limitation from deep_analysis structure
            deep_analysis = node_data.get('deep_analysis', {})
            limitation = deep_analysis.get('limitation', {}).get('content', '')

            if not limitation or len(limitation) < 20:
                continue

            # Check if subsequent work has solved it via Overcomes
            has_overcome = False
            overcome_attempts = []  # Record attempts to solve

            for pred in graph.predecessors(node_id):
                edge_data = graph[pred][node_id]
                edge_type = edge_data.get('edge_type', 'Unknown')

                if edge_type == 'Overcomes':
                    has_overcome = True
                    break
                elif edge_type in ['Extends', 'Realizes']:
                    # Although not Overcomes, someone is trying to improve
                    overcome_attempts.append({
                        'id': pred,
                        'title': graph.nodes[pred].get('title', '')[:60],
                        'type': edge_type
                    })

            if not has_overcome:
                # Assess Limitation severity
                severity = self._assess_limitation_severity(limitation)

                # Analyze why it hasn't been solved
                unsolved_reason = self._analyze_unsolved_reason(
                    limitation,
                    len(overcome_attempts),
                    node_data
                )

                # Assess breakthrough difficulty
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
                    'overcome_attempts': overcome_attempts[:3],  # Show top 3 attempts
                    'impact_potential': node_data.get('cited_by_count', 0) * severity,  # Potential impact after solving
                    'research_direction': self._suggest_research_direction(limitation, attack_difficulty)
                })

        # Sort by potential impact
        hard_nuts = sorted(hard_nuts, key=lambda x: x['impact_potential'], reverse=True)[:12]

        return hard_nuts

    def _assess_limitation_severity(self, limitation: str) -> float:
        """
        Assess Limitation severity

        Args:
            limitation: Limitation description

        Returns:
            Severity score (0-1)
        """
        score = 0.5  # Base score

        # Severity keywords
        critical_keywords = ['critical', 'major', 'significant', 'fundamental', 'serious']
        moderate_keywords = ['important', 'notable', 'considerable']
        minor_keywords = ['minor', 'small', 'slight']

        limitation_lower = limitation.lower()

        # Check severity
        if any(kw in limitation_lower for kw in critical_keywords):
            score += 0.4
        elif any(kw in limitation_lower for kw in moderate_keywords):
            score += 0.2
        elif any(kw in limitation_lower for kw in minor_keywords):
            score -= 0.2

        # Check scope of impact
        scope_keywords = ['all', 'general', 'common', 'widespread', 'universal']
        if any(kw in limitation_lower for kw in scope_keywords):
            score += 0.2

        return min(max(score, 0.1), 1.0)

    def _analyze_unsolved_reason(self, limitation: str, attempt_count: int, paper_data: Dict) -> str:
        """
        Analyze why Limitation hasn't been solved yet

        Args:
            limitation: Limitation description
            attempt_count: Number of works attempting to solve it
            paper_data: Paper data

        Returns:
            Analysis conclusion
        """
        limitation_lower = limitation.lower()

        # High technical difficulty
        if any(kw in limitation_lower for kw in ['complex', 'difficult', 'challenging', 'non-trivial']):
            if attempt_count > 0:
                return f"Extremely high technical difficulty, {attempt_count} works have attempted improvements but failed to fundamentally solve it"
            else:
                return "Extremely high technical difficulty, no one has dared to challenge it yet"

        # High resource requirements
        if any(kw in limitation_lower for kw in ['expensive', 'costly', 'large-scale', 'computation']):
            return "Requires substantial computational resources or data, high requirements for researchers"

        # Theoretical foundation issues
        if any(kw in limitation_lower for kw in ['theoretical', 'fundamental', 'framework']):
            return "Involves theoretical foundation issues, requires methodological breakthroughs"

        # Other
        if attempt_count > 0:
            return f"{attempt_count} works have attempted improvements, but core issues remain unsolved"
        else:
            return "Has not received sufficient attention, or requires interdisciplinary knowledge"

    def _assess_attack_difficulty(self, limitation: str, citations: int, attempts: int) -> str:
        """
        Assess breakthrough difficulty

        Args:
            limitation: Limitation description
            citations: Paper citation count
            attempts: Number of works attempting to solve it

        Returns:
            Difficulty level: 'very_hard', 'hard', 'medium'
        """
        # High citations but no one solved = extremely hard
        if citations > 1000 and attempts == 0:
            return 'very_hard'

        # Someone tried but failed = hard
        if attempts > 0:
            return 'hard'

        # Other
        if citations > 500:
            return 'hard'
        else:
            return 'medium'

    def _suggest_research_direction(self, limitation: str, difficulty: str) -> str:
        """
        Suggest research direction

        Args:
            limitation: Limitation description
            difficulty: Difficulty level

        Returns:
            Research direction recommendation
        """
        limitation_lower = limitation.lower()

        # Provide recommendations based on Limitation type
        if 'scalability' in limitation_lower or 'scale' in limitation_lower:
            direction = "Consider distributed methods, approximation algorithms, or model compression techniques"
        elif 'generalization' in limitation_lower or 'generalize' in limitation_lower:
            direction = "Explore meta-learning, domain adaptation, or transfer learning methods"
        elif 'efficiency' in limitation_lower or 'computation' in limitation_lower:
            direction = "Research acceleration algorithms, model distillation, or hardware optimization"
        elif 'interpretability' in limitation_lower or 'explainability' in limitation_lower:
            direction = "Introduce interpretability methods, attention mechanisms, or causal reasoning"
        elif 'data' in limitation_lower and 'require' in limitation_lower:
            direction = "Explore few-shot learning, data augmentation, or unsupervised methods"
        else:
            direction = "Recommend starting from methodological innovation or cross-domain transfer perspectives"

        difficulty_prefix = {
            'very_hard': "⚠️ Extremely high difficulty project,",
            'hard': "🔥 High difficulty project,",
            'medium': "💪 Medium difficulty project,"
        }

        return f"{difficulty_prefix.get(difficulty, '')} {direction}"

    def _generate_innovative_ideas(self, graph: nx.DiGraph) -> Dict:
        """
        Level 3: Innovative ideas - cross-domain transfer and hybrid methods (complete reconstruction)

        Core idea:
        Leverage transitivity of Adapts_to or Alternative for reasoning

        Pattern A (Resurrection):
        1. Find a Method_X that is very successful in branch A (heavily Extends)
        2. Find a Problem_Y currently facing branch B (many unsolved Limitations)
        3. Predict: Calculate TextSimilarity(Method_X, Problem_Y's context)
        4. If logically feasible, transferring Method_X to solve Problem_Y is a typical Adapts_to innovation

        Pattern B (Combination punch):
        1. If papers A and B have Alternative relationship (solve same problem, different methods)
        2. Check if A's Limitation is exactly B's strength, and vice versa
        3. Predict: Propose a Hybrid Method combining A and B's advantages, which typically generates a powerful Overcomes paper

        Returns:
            Innovative idea dictionary, including cross_domain_transfer and hybrid_methods types
        """
        # Pattern A: Resurrection (cross-domain transfer)
        cross_domain_ideas = self._generate_cross_domain_transfer_ideas(graph)

        # Pattern B: Combination punch (hybrid methods)
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
        Pattern A: Cross-Domain Transfer - Generate cross-domain transfer ideas

        Algorithm steps:
        1. Find successful Methods with many Extends references (proving method effectiveness)
        2. Find Problems with unsolved Limitations
        3. Calculate semantic matching between Methods and Problems
        4. Recommend high-match cross-domain transfer solutions

        Returns:
            List of cross-domain transfer ideas
        """
        ideas = []

        # Step 1: Identify successful Methods (nodes with many Extends references)
        successful_methods = []

        for node in graph.nodes():
            # Count the number of Extends references
            in_extends = sum(
                1 for pred in graph.predecessors(node)
                if graph[pred][node].get('edge_type') == 'Extends'
            )

            if in_extends >= 3:  # Referenced by at least 3 Extends
                node_data = graph.nodes[node]

                # Get method from deep_analysis structure
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

        # Sort by Extends count
        successful_methods = sorted(successful_methods, key=lambda x: x['extends_count'], reverse=True)[:10]

        # Step 2: Identify Problems with unsolved Limitations
        unsolved_problems = []

        for node in graph.nodes():
            node_data = graph.nodes[node]

            # Get limitation from deep_analysis structure
            deep_analysis = node_data.get('deep_analysis', {})
            limitation = deep_analysis.get('limitation', {}).get('content', '')

            if not limitation or len(limitation) < 20:
                continue

            # Check if already solved by Overcomes
            has_overcome = any(
                graph[pred][node].get('edge_type') == 'Overcomes'
                for pred in graph.predecessors(node)
            )

            if not has_overcome:
                # Get problem field from deep_analysis
                deep_analysis = node_data.get('deep_analysis', {})
                problem = deep_analysis.get('problem', {}).get('content', '')
                if not problem:
                    problem = limitation  # If no problem, use limitation

                unsolved_problems.append({
                    'id': node,
                    'title': node_data.get('title', ''),
                    'year': node_data.get('year'),
                    'limitation': limitation,
                    'problem': problem,
                    'domain': self._extract_domain_keywords(node_data.get('title', ''))
                })

        # Step 3: Match Methods with Problems
        for method in successful_methods[:8]:  # Top 8 successful methods
            for problem in unsolved_problems[:15]:  # Top 15 unsolved problems
                # Avoid self-matching
                if method['id'] == problem['id']:
                    continue

                # Avoid existing citation relationships
                if graph.has_edge(method['id'], problem['id']) or graph.has_edge(problem['id'], method['id']):
                    continue

                # Calculate semantic similarity
                similarity = self._calculate_text_similarity(
                    method['method'],
                    problem['problem']
                )

                # Filter low similarity
                if similarity < 0.2:
                    continue

                # Check if cross-domain (increases innovation)
                is_cross_domain = len(set(method['domain']) & set(problem['domain'])) == 0

                # Assess transfer feasibility
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

        # Sort by feasibility and similarity
        ideas = sorted(ideas, key=lambda x: x['feasibility'] * x['similarity_score'], reverse=True)[:8]

        return ideas

    def _generate_hybrid_method_ideas(self, graph: nx.DiGraph) -> List[Dict]:
        """
        Pattern B: Hybrid Methods - Generate hybrid method ideas

        Algorithm steps:
        1. Find Alternative relationship pairs (same problem, different methods)
        2. Analyze if A's Limitation is B's strength
        3. Recommend Hybrid Method combining both advantages

        Returns:
            List of hybrid method ideas
        """
        ideas = []

        # Collect all Alternative relationships
        alternative_pairs = []

        for source, target, data in graph.edges(data=True):
            if data.get('edge_type') == 'Alternative':
                source_data = graph.nodes[source]
                target_data = graph.nodes[target]

                # Get fields from deep_analysis
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

        # Analyze each Alternative relationship pair
        for pair in alternative_pairs[:10]:  # Analyze top 10 pairs
            paper_a = pair['paper_a']
            paper_b = pair['paper_b']

            # Check if there is sufficient information
            if not paper_a['limitation'] or not paper_b['method']:
                continue

            # Check complementarity between A's Limitation and B's Method
            complementarity_ab = self._calculate_text_similarity(
                paper_a['limitation'],
                paper_b['method']
            )

            complementarity_ba = self._calculate_text_similarity(
                paper_b.get('limitation', ''),
                paper_a.get('method', '')
            ) if paper_b.get('limitation') and paper_a.get('method') else 0

            # At least one aspect must have complementarity
            if complementarity_ab < 0.3 and complementarity_ba < 0.3:
                continue

            # Assess hybrid method feasibility
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

        # Sort by feasibility
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
        Assess the feasibility of cross-domain transfer

        Args:
            method: Method information
            problem: Problem information
            similarity: Semantic similarity
            is_cross_domain: Whether cross-domain

        Returns:
            Feasibility score (0-1)
        """
        feasibility = similarity  # Base score

        # More Extends count indicates more mature method
        if method['extends_count'] >= 5:
            feasibility += 0.2
        elif method['extends_count'] >= 3:
            feasibility += 0.1

        # Cross-domain transfer is more innovative, but slightly less feasible
        if is_cross_domain:
            feasibility += 0.1  # Innovation bonus
            feasibility *= 0.9  # Risk discount

        # Time gap should not be too large (method should not be too outdated)
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
        Assess the feasibility of hybrid methods

        Args:
            paper_a: Paper A information
            paper_b: Paper B information
            comp_ab: Complementarity between A's weakness and B's strength
            comp_ba: Complementarity between B's weakness and A's strength

        Returns:
            Feasibility score (0-1)
        """
        # Base score: take the maximum of two complementarity scores
        feasibility = max(comp_ab, comp_ba)

        # Bonus if bidirectional complementarity
        if comp_ab > 0.3 and comp_ba > 0.3:
            feasibility += 0.2

        # Temporal proximity (methods from same period are easier to combine)
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
        Generate innovation story for cross-domain transfer

        Returns:
            Description text
        """
        method_domain = ', '.join(method['domain'][:2]) if method['domain'] else 'some domain'
        problem_domain = ', '.join(problem['domain'][:2]) if problem['domain'] else 'this domain'

        domain_desc = f"Transfer from [{method_domain}] to [{problem_domain}]" if is_cross_domain else f"Apply within [{problem_domain}]"

        return (
            f"Innovation: {domain_desc}. "
            f"Method comes from successful experience in {method['year']} (extended by {method['extends_count']} works), "
            f"can be used to solve unsolved problems in {problem['year']} paper. "
            f"Match score: {similarity:.0%}"
        )

    def _suggest_hybrid_strategy(
        self,
        paper_a: Dict,
        paper_b: Dict,
        comp_ab: float,
        comp_ba: float
    ) -> str:
        """
        Suggest hybrid method strategy

        Returns:
            Strategy description
        """
        if comp_ab > comp_ba:
            dominant = 'B'
            complementary = 'A'
            desc = f"Use method B as primary, introduce method A to compensate B's weaknesses (match score {comp_ab:.0%})"
        else:
            dominant = 'A'
            complementary = 'B'
            desc = f"Use method A as primary, introduce method B to compensate A's weaknesses (match score {comp_ba:.0%})"

        return f"Hybrid strategy: {desc}. Recommend selectively integrating {complementary}'s advantages within {dominant}'s framework."

    def _log_summary(self, report: Dict) -> None:
        """
        Output analysis summary log (enhanced version - dual core directions)

        Args:
            report: Analysis report
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Topic Evolution Analysis Complete (Dual Core Directions)")
        logger.info(f"{'='*80}")

        logger.info(f"\nGraph Overview:")
        overview = report.get('graph_overview', {})
        logger.info(f"  • Total papers: {overview.get('total_papers', 0)}")
        logger.info(f"  • Citation relationships: {overview.get('total_citations', 0)}")
        logger.info(f"  • Time span: {overview.get('year_range', 'Unknown')}")

        # ======================== Core Direction 1: Retrospective Analysis ========================
        logger.info(f"\n{'='*80}")
        logger.info(f"[Core Direction 1] Retrospective Analysis")
        logger.info(f"{'='*80}")

        retro = report.get('retrospective_analysis', {})

        # 1.1 Evolutionary backbone vs incremental branches
        logger.info(f"\n1.1 Evolutionary Backbone vs Incremental Branches:")
        backbone = retro.get('backbone_vs_incremental', {})
        summary = backbone.get('summary', {})
        logger.info(f"  • Backbone paths (Overcomes/Realizes): {summary.get('backbone_count', 0)} paths")
        logger.info(f"  • Incremental paths (Extends): {summary.get('incremental_count', 0)} paths")
        logger.info(f"  • Backbone/Incremental ratio: {summary.get('ratio', 0):.2f}")
        logger.info(f"  • Breakthrough points: {summary.get('breakthrough_count', 0)} points\n")

        # Display breakthrough points
        breakthrough = backbone.get('breakthrough_points', [])
        if breakthrough:
            logger.info(f"  Top Breakthrough Points (jumping from Extends to Overcomes):")
            for i, bp in enumerate(breakthrough[:5], 1):
                logger.info(f"    [{i}] {bp['title']}")
                logger.info(f"        Year: {bp['year']}, Citations: {bp['cited_by_count']}")
                logger.info(f"        Breakthrough type: {bp['breakthrough_type']}")
                logger.info(f"        Input: {bp['in_extends']} Extends → Output: {bp['out_overcomes']} Overcomes, {bp['out_realizes']} Realizes")
                logger.info(f"        Breakthrough score: {bp['breakthrough_score']:.1f}\n")

        # Display backbone continuous chains
        backbone_chains = backbone.get('backbone_chains', [])
        if backbone_chains:
            logger.info(f"  Backbone Continuous Chains (hardcore evolution lines):")
            for i, chain in enumerate(backbone_chains[:3], 1):
                logger.info(f"    [Chain {i}] Length: {chain['length']} papers, Time span: {chain['year_span']} years")
                for j, paper in enumerate(chain['chain'][:4], 1):  # Only show first 4 papers
                    logger.info(f"      {j}. {paper['title']} ({paper['year']})")
                    if j < len(chain['edge_types']):
                        logger.info(f"         └─> [{chain['edge_types'][j-1]}]")
                logger.info("")

        # Display incremental bottlenecks
        bottlenecks = backbone.get('incremental_bottlenecks', [])
        if bottlenecks:
            logger.info(f"  Incremental Path Bottlenecks (involution endpoints):")
            for i, bn in enumerate(bottlenecks[:3], 1):
                logger.info(f"    [{i}] {bn['title']} ({bn['year']})")
                logger.info(f"        {bn['reason']}\n")

        # 1.2 Technical bifurcations
        logger.info(f"\n1.2 Technical Bifurcations:")
        bifur = retro.get('technical_bifurcations', [])
        if bifur:
            logger.info(f"  Found {len(bifur)} technical route competitions\n")
            for i, b in enumerate(bifur[:3], 1):
                logger.info(f"  [Bifurcation {i}] {b['parent']['title']} ({b['parent']['year']})")
                logger.info(f"    Divergence score: {b['divergence_score']:.2f}")
                logger.info(f"    Competing branches:")
                for j, branch in enumerate(b['branches'], 1):
                    logger.info(f"      {j}. [{branch['edge_type']}] {branch['title']} ({branch['year']})")
                    logger.info(f"         Subsequent development: {branch['subtree_size']} works, depth {branch['subtree_depth']}, status: {branch['subtree_status']}")
                logger.info(f"    Analysis: {b['branch_comparison']}\n")
        else:
            logger.info(f"  No obvious technical bifurcations found\n")

        # 1.3 Cross-domain invasions
        logger.info(f"\n1.3 Cross-Domain Invasions:")
        invasions = retro.get('cross_domain_invasions', [])
        if invasions:
            logger.info(f"  Found {len(invasions)} cross-domain transfer cases (Adapts_to)\n")
            for i, inv in enumerate(invasions[:5], 1):
                impact = inv['impact_analysis']
                logger.info(f"  [Invasion {i}] {inv['cross_domain_story']}")
                logger.info(f"    Source: {inv['to']['title'][:60]}... ({inv['to']['year']})")
                logger.info(f"    Target: {inv['from']['title'][:60]}... ({inv['from']['year']})")
                logger.info(f"    Impact: {impact['descendants_count']} subsequent works, success level: {impact['success_level']}\n")
        else:
            logger.info(f"  No cross-domain transfer cases found\n")

        # ======================== Core Direction 2: Future Prediction ========================
        logger.info(f"\n{'='*80}")
        logger.info(f"[Core Direction 2] Future Prediction")
        logger.info(f"{'='*80}")

        future = report.get('future_prediction', {})

        # 2.1 Level 1: Low-Hanging Fruits
        logger.info(f"\n2.1 Level 1: Low-Hanging Fruits")
        logger.info(f"  Finding unrealized Future Work\n")
        level1 = future.get('level_1_low_hanging_fruits', [])
        if level1:
            logger.info(f"  Found {len(level1)} ready research opportunities\n")
            for i, idea in enumerate(level1[:5], 1):
                paper = idea['paper']
                logger.info(f"  [Idea{i}] {paper['title']}")
                logger.info(f"    Source paper: {paper['year']}, Citations: {paper['cited_by_count']}")
                logger.info(f"    Difficulty: {idea['difficulty']}, Feasibility: {idea['feasibility_score']:.2f}, Priority: {idea['priority']:.1f}")
                logger.info(f"    Future Work: {idea['future_work'][:120]}...")
                logger.info(f"    Recommendation: {idea['recommendation']}\n")
        else:
            logger.info(f"  No obvious low-hanging fruit opportunities found\n")

        # 2.2 Level 2: Hard Nuts
        logger.info(f"\n2.2 Level 2: Hard Nuts")
        logger.info(f"  Finding Limitations not yet Overcome\n")
        level2 = future.get('level_2_hard_nuts', [])
        if level2:
            logger.info(f"  Found {len(level2)} high-value research directions\n")
            for i, idea in enumerate(level2[:5], 1):
                paper = idea['paper']
                logger.info(f"  [Idea{i}] {paper['title']}")
                logger.info(f"    Source paper: {paper['year']}, Citations: {paper['cited_by_count']}")
                logger.info(f"    Severity: {idea['severity']:.2f}, Attack difficulty: {idea['attack_difficulty']}, Impact potential: {idea['impact_potential']:.1f}")
                logger.info(f"    Limitation: {idea['limitation'][:120]}...")
                logger.info(f"    Unsolved reason: {idea['unsolved_reason']}")
                logger.info(f"    Research direction: {idea['research_direction']}\n")
        else:
            logger.info(f"  No obvious hard nut directions found\n")

        # 2.3 Level 3: Innovative Ideas
        logger.info(f"\n2.3 Level 3: Innovative Ideas (Cross-Pollination & Hybrid Methods)")
        level3 = future.get('level_3_innovative_ideas', {})
        summary3 = level3.get('summary', {})
        logger.info(f"  Cross-domain transfer: {summary3.get('cross_domain_count', 0)} ideas")
        logger.info(f"  Hybrid methods: {summary3.get('hybrid_count', 0)} ideas")
        logger.info(f"  Total: {summary3.get('total_ideas', 0)} innovative ideas\n")

        # Pattern A: Cross-domain transfer
        cross_domain = level3.get('cross_domain_transfer', [])
        if cross_domain:
            logger.info(f"  Pattern A: Cross-Domain Transfer\n")
            for i, idea in enumerate(cross_domain[:4], 1):
                method = idea['method_paper']
                target = idea['target_paper']
                logger.info(f"    [Idea{i}] {idea['innovation_story']}")
                logger.info(f"      Method source: {method['title']}")
                logger.info(f"      Target problem: {target['title']}")
                logger.info(f"      Feasibility: {idea['feasibility']:.2f}, Cross-domain: {'Yes' if idea['is_cross_domain'] else 'No'}\n")

        # Pattern B: Hybrid methods
        hybrid = level3.get('hybrid_methods', [])
        if hybrid:
            logger.info(f"  Pattern B: Hybrid Methods\n")
            for i, idea in enumerate(hybrid[:4], 1):
                paper_a = idea['paper_a']
                paper_b = idea['paper_b']
                scores = idea['complementarity_scores']
                logger.info(f"    [Idea{i}] Hybrid method (A + B)")
                logger.info(f"      Method A: {paper_a['title']}")
                logger.info(f"      Method B: {paper_b['title']}")
                logger.info(f"      Complementarity: {scores['overall']:.2f}, Feasibility: {idea['hybrid_feasibility']:.2f}")
                logger.info(f"      {idea['hybrid_strategy']}\n")

        logger.info(f"\n{'='*80}")
        logger.info(f"Analysis report generation complete")
        logger.info(f"{'='*80}\n")


    def _extract_evolutionary_paths(self, graph: nx.DiGraph, year_stats: Dict) -> List[Dict]:
        """
        Extract critical evolutionary paths (Critical Evolutionary Path Extraction)

        Based on evolutionary momentum weights, find the backbone paths that drive field progress.

        Algorithm concept:
        1. Assign "evolutionary momentum" weights to each edge (based on citation type)
        2. Find paths with maximum weight sum in the DAG
        3. This path represents the "Backbone of Innovation"

        Args:
            graph: Knowledge graph
            year_stats: Year statistics information

        Returns:
            List of critical evolutionary paths
        """
        try:
            # 1. Verify if graph is a DAG (Directed Acyclic Graph)
            if not nx.is_directed_acyclic_graph(graph):
                logger.warning("Graph contains cycles, cannot extract evolutionary paths")
                return []

            # 2. Assign evolutionary momentum weights to each edge
            weighted_graph = self._create_weighted_graph(graph)

            # 3. Determine time window
            if year_stats:
                years = sorted(year_stats.keys())
                start_year = years[0]
                end_year = years[-1]

                # If time window is configured, narrow the range
                if self.evol_time_window_years:
                    start_year = max(start_year, end_year - self.evol_time_window_years)
            else:
                start_year = None
                end_year = None

            # 4. Find early and late nodes
            early_nodes = []
            late_nodes = []

            for node_id in graph.nodes():
                node_year = graph.nodes[node_id].get('year')
                if not node_year:
                    continue

                # Early nodes (first 20% of time window)
                if start_year and node_year <= start_year + (end_year - start_year) * 0.2:
                    early_nodes.append(node_id)

                # Late nodes (last 20% of time window)
                if end_year and node_year >= end_year - (end_year - start_year) * 0.2:
                    late_nodes.append(node_id)

            if not early_nodes or not late_nodes:
                logger.warning("Insufficient time span, cannot extract evolutionary paths")
                return []

            # 5. Find heaviest path from each early node to each late node
            all_paths = []

            for start_node in early_nodes:
                for end_node in late_nodes:
                    if start_node == end_node:
                        continue

                    # Check if path exists
                    if not nx.has_path(weighted_graph, start_node, end_node):
                        continue

                    # Use Bellman-Ford algorithm to find longest path (negate weights)
                    try:
                        # NetworkX's shortest path algorithm, negated weights give longest path
                        path = nx.shortest_path(
                            weighted_graph,
                            start_node,
                            end_node,
                            weight='neg_weight'
                        )

                        # Calculate total weight of the path
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

                        # Filter low-weight paths
                        if total_weight < self.evol_min_weight:
                            continue

                        # Collect path information
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
                        logger.debug(f"Path calculation failed: {e}")
                        continue

            # 6. Sort by weight, take top N
            all_paths = sorted(all_paths, key=lambda x: x['total_weight'], reverse=True)[:self.evol_max_paths]

            # 7. Remove duplicate or highly overlapping paths
            unique_paths = self._remove_duplicate_paths(all_paths)

            logger.info(f"    Found {len(unique_paths)} critical evolutionary paths")

            return unique_paths

        except Exception as e:
            logger.warning(f"Critical evolutionary path extraction failed: {e}")
            return []

    def _create_weighted_graph(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Create weighted graph

        Assign evolutionary momentum weights to each edge

        Args:
            graph: Original graph

        Returns:
            Weighted graph
        """
        weighted_graph = graph.copy()

        for u, v, data in weighted_graph.edges(data=True):
            edge_type = data.get('edge_type', 'Unknown')
            weight = self.evolutionary_weights.get(edge_type, 0.3)

            # Set positive weight and negative weight (for longest path algorithm)
            weighted_graph[u][v]['weight'] = weight
            weighted_graph[u][v]['neg_weight'] = -weight

        return weighted_graph

    def _remove_duplicate_paths(self, paths: List[Dict], overlap_threshold: float = 0.7) -> List[Dict]:
        """
        Remove duplicate or highly overlapping paths

        Args:
            paths: Path list
            overlap_threshold: Overlap threshold (node overlap ratio)

        Returns:
            Deduplicated path list
        """
        if len(paths) <= 1:
            return paths

        unique_paths = []

        for i, path1 in enumerate(paths):
            is_duplicate = False
            nodes1 = set([p['id'] for p in path1['path']])

            for path2 in unique_paths:
                nodes2 = set([p['id'] for p in path2['path']])

                # Calculate Jaccard similarity
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
        Detect technical bifurcations (Technical Bifurcation Detection) - Enhanced version

        Identify technical development "crossroads" - Same problem spawns different technical schools

        Core concept:
        1. Find nodes referenced by multiple subsequent works through Alternative cited nodes
        2. Or: Find Problem triggered multiple different Method nodes
        3. Analyze subtree size after these bifurcations (Which path goes further? Which path died out?)

        Algorithm steps:
        1. Find bifurcation structure: Parent node referenced by multiple child nodes, and edge type is Alternative/Extends
        2. Verify between child nodes no strong citation relationship (independent development)
        3. Semantic verification: child node Method different but Problem same
        4. Track branch subsequent development (subtree size analysis)

        Args:
            graph: Knowledge graph

        Returns:
            Technical divergence point list (contains branch subsequent development analysis)
        """
        try:
            bifurcations = []

            # 1. Traverse all nodes, find potential bifurcation parent nodes
            for parent_id in graph.nodes():
                # Get parent node all successor nodes (cited papers)
                successors = list(graph.successors(parent_id))

                if len(successors) < self.bifur_min_children:
                    continue

                # 2. Filter matching fork edge type child nodes
                fork_children = []
                for child_id in successors:
                    edge_data = graph[parent_id][child_id]
                    edge_type = edge_data.get('edge_type', 'Unknown')

                    if edge_type in self.bifur_fork_edge_types:
                        fork_children.append(child_id)

                if len(fork_children) < self.bifur_min_children:
                    continue

                # 3. Detect child nodes whether pairwise independent (no strong citation relationship)
                independent_pairs = []

                for i in range(len(fork_children)):
                    for j in range(i + 1, len(fork_children)):
                        child_a = fork_children[i]
                        child_b = fork_children[j]

                        # Check A→B or B→A whether strong citation exists
                        has_strong_link = False

                        if graph.has_edge(child_a, child_b):
                            edge_type = graph[child_a][child_b].get('edge_type', 'Unknown')
                            if edge_type in ['Overcomes', 'Realizes', 'Extends']:
                                has_strong_link = True

                        if graph.has_edge(child_b, child_a):
                            edge_type = graph[child_b][child_a].get('edge_type', 'Unknown')
                            if edge_type in ['Overcomes', 'Realizes', 'Extends']:
                                has_strong_link = True

                        # If no strong citation, considered independent branch
                        if not has_strong_link:
                            independent_pairs.append((child_a, child_b))

                if not independent_pairs:
                    continue

                # 4. Semantic verification: Method different, but Problem same
                parent_data = graph.nodes[parent_id]

                for child_a, child_b in independent_pairs:
                    child_a_data = graph.nodes[child_a]
                    child_b_data = graph.nodes[child_b]

                    # From deep_analysis extract Method and Problem fields
                    child_a_deep = child_a_data.get('deep_analysis', {})
                    child_b_deep = child_b_data.get('deep_analysis', {})

                    method_a = child_a_deep.get('method', {}).get('content', '') or child_a_data.get('title', '')
                    method_b = child_b_deep.get('method', {}).get('content', '') or child_b_data.get('title', '')
                    problem_a = child_a_deep.get('problem', {}).get('content', '') or child_a_data.get('abstract', '')
                    problem_b = child_b_deep.get('problem', {}).get('content', '') or child_b_data.get('abstract', '')

                    # Calculate similarity
                    method_similarity = self._calculate_text_similarity(method_a, method_b)
                    problem_similarity = self._calculate_text_similarity(problem_a, problem_b)

                    # Determined as technical divergence: Method different but Problem same
                    if (method_similarity < self.bifur_method_sim_threshold and
                        problem_similarity > self.bifur_problem_sim_threshold):

                        # 5. Track branch subsequent development (subtree analysis)
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
                            'divergence_score': problem_similarity - method_similarity,  # divergence_score
                            'branch_comparison': self._compare_branches(branch_a_subtree, branch_b_subtree)
                        })

            # 5. Sort by divergence score, take top N
            bifurcations = sorted(
                bifurcations,
                key=lambda x: x['divergence_score'],
                reverse=True
            )[:self.bifur_max_bifurcations]

            logger.info(f"    Found {len(bifurcations)} technical bifurcations")

            return bifurcations

        except Exception as e:
            logger.warning(f"Technical bifurcation detection failed: {e}")
            return []

    def _analyze_branch_subtree(self, graph: nx.DiGraph, root_node: str) -> Dict:
        """
        Analyze branch subtree development status

        Args:
            graph: Knowledge graph
            root_node: Branch root node

        Returns:
            Subtree analysis result
        """
        try:
            # Use BFS to find all descendant nodes
            descendants = list(nx.descendants(graph, root_node))
            subtree_size = len(descendants)

            # Calculate maximum depth
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

            # Determine branch status
            status = 'unknown'
            if subtree_size == 0:
                status = 'dead'  # Dead end
            elif subtree_size < 5:
                status = 'weak'  # Weak development
            elif subtree_size >= 5 and subtree_size < 15:
                status = 'moderate'  # Medium level development
            else:
                status = 'strong'  # Strong development

            return {
                'size': subtree_size,
                'depth': max_depth,
                'status': status
            }

        except Exception as e:
            logger.debug(f"Branch subtree analysis failed: {e}")
            return {
                'size': 0,
                'depth': 0,
                'status': 'unknown'
            }

    def _compare_branches(self, branch_a: Dict, branch_b: Dict) -> str:
        """
        Compare two branch development statuses

        Args:
            branch_a: Branch A subtree analysis
            branch_b: Branch B subtree analysis

        Returns:
            Comparison conclusion
        """
        size_a = branch_a['size']
        size_b = branch_b['size']

        if size_a == 0 and size_b == 0:
            return "Both routes have no subsequent development"
        elif size_a == 0:
            return f"Branch A dead, Branch B obtained {size_b} subsequent works, became mainstream route"
        elif size_b == 0:
            return f"Branch B dead, Branch A obtained {size_a} subsequent works, became mainstream route"
        elif size_a > size_b * 2:
            return f"Branch A strongly leading ({size_a} vs {size_b}), became mainstream route"
        elif size_b > size_a * 2:
            return f"Branch B strongly leading ({size_b} vs {size_a}), became mainstream route"
        else:
            return f"Two routes evenly matched ({size_a} vs {size_b}), technical route competition continues"

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate two segments text similarity

        Args:
            text1: Text 1
            text2: Text 2

        Returns:
            Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0

        try:
            if self.bifur_use_cosine:
                # Use cosine similarity (requires vectorization)
                # Simple implementation: Use bag-of-words model
                words1 = set(re.findall(r'\b\w+\b', text1.lower()))
                words2 = set(re.findall(r'\b\w+\b', text2.lower()))

                # Calculate cosine similarity (Jaccard similarity as approximation)
                if not words1 or not words2:
                    return 0.0

                intersection = len(words1 & words2)
                union = len(words1 | words2)

                return intersection / union if union > 0 else 0.0

            else:
                # Use Jaccard similarity
                words1 = set(re.findall(r'\b\w+\b', text1.lower()))
                words2 = set(re.findall(r'\b\w+\b', text2.lower()))

                if not words1 or not words2:
                    return 0.0

                intersection = len(words1 & words2)
                union = len(words1 | words2)

                return intersection / union if union > 0 else 0.0

        except Exception as e:
            logger.debug(f"Text similarity calculation failed: {e}")
            return 0.0

    def _detect_open_frontiers(self, graph: nx.DiGraph, year_stats: Dict) -> Dict:
        """
        Detect unclosed frontier (Open Frontier Detection)

        Identify unsolved open problems and generate cross-domain transfer research ideas

        Algorithm steps:
        1. Filter edge nodes: recent N year papers
        2. Defect closure detection: check Limitation whether subsequent work solved it
        3. Cross-domain match: match Limitation and Contribution from other papers

        Args:
            graph: Knowledge graph
            year_stats: Year citation count information

        Returns:
            Open frontier dict, contains open_problems and research_ideas
        """
        try:
            # 1. Filter edge nodes (recent N years)
            if not year_stats:
                logger.warning("No year citation information, cannot filter edge nodes")
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
                logger.warning(f"No recent {self.frontier_recent_years} year papers")
                return {'open_problems': [], 'research_ideas': []}

            logger.info(f"    Filtered out {len(leaf_nodes)} edge nodes ({cutoff_year}-{latest_year} years)")

            # 2. Defect closure detection: find unsolved Limitation
            open_problems = []

            for node_id in leaf_nodes:
                node_data = graph.nodes[node_id]

                # From deep_analysis get limitation or future_work
                deep_analysis = node_data.get('deep_analysis', {})
                limitation = deep_analysis.get('limitation', {}).get('content', '')
                if not limitation:
                    limitation = deep_analysis.get('future_work', {}).get('content', '')

                if not limitation:
                    continue  # No Limitation, skip

                # Check whether has subsequent work through Overcomes or Realizes solve
                has_closure = False

                # Get all citations to this node papers (subsequent work)
                predecessors = list(graph.predecessors(node_id))

                for pred_id in predecessors:
                    edge_data = graph[pred_id][node_id]
                    edge_type = edge_data.get('edge_type', 'Unknown')

                    # If exists Overcomes or Realizes type citation, indicates problem already solved
                    if edge_type in ['Overcomes', 'Realizes']:
                        has_closure = True
                        break

                # If unsolved, record as open problem
                if not has_closure:
                    open_problems.append({
                        'paper': {
                            'id': node_id,
                            'title': node_data.get('title', ''),
                            'year': node_data.get('year')
                        },
                        'limitation': limitation,
                        'out_degree': graph.out_degree(node_id)  # How many subsequent works (but not solved)
                    })

            # Sort by out_degree descending sort (more subsequent work but unresolved more important)
            open_problems = sorted(
                open_problems,
                key=lambda x: x['out_degree'],
                reverse=True
            )[:self.frontier_max_open_problems]

            logger.info(f"    Found {len(open_problems)} unclosed open problems")

            # 3. Cross-domain match: for each open problem generate research ideas
            research_ideas = self._generate_cross_domain_ideas(graph, open_problems)

            logger.info(f"    Generated {len(research_ideas)} cross-domain transfer ideas")

            return {
                'open_problems': open_problems,
                'research_ideas': research_ideas
            }

        except Exception as e:
            logger.warning(f"Unclosed frontier detection failed: {e}")
            return {'open_problems': [], 'research_ideas': []}

    def _generate_cross_domain_ideas(self, graph: nx.DiGraph, open_problems: List[Dict]) -> List[Dict]:
        """
        Generate cross-domain transfer research ideas

        For each unsolved Limitation find possible solutions (from other papers' Contribution)

        Args:
            graph: Knowledge graph
            open_problems: Unclosed problem list

        Returns:
            Research idea list
        """
        research_ideas = []

        try:
            # For each open problem find candidate solutions
            for problem in open_problems:
                target_node_id = problem['paper']['id']
                target_limitation = problem['limitation']

                # Candidate solution list
                candidate_solutions = []

                # Traverse graph medium all other nodes
                for candidate_id in graph.nodes():
                    if candidate_id == target_node_id:
                        continue

                    candidate_data = graph.nodes[candidate_id]

                    # From deep_analysis get method
                    candidate_deep = candidate_data.get('deep_analysis', {})
                    method = candidate_deep.get('method', {}).get('content', '')

                    if not method:
                        continue

                    # Check if citation relationship already exists (Avoid recommending existing citations)
                    if graph.has_edge(target_node_id, candidate_id) or graph.has_edge(candidate_id, target_node_id):
                        continue

                    # Calculate Limitation and Method semantic similarity
                    similarity = self._calculate_text_similarity(target_limitation, method)

                    # Filter low similarity candidates
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

                # Sort by similarity
                candidate_solutions = sorted(
                    candidate_solutions,
                    key=lambda x: x['similarity'],
                    reverse=True
                )

                # For this problem generate top N ideas
                for solution in candidate_solutions[:2]:  # Each problem at most 2 ideas
                    research_ideas.append({
                        'target_paper': problem['paper'],
                        'target_limitation': target_limitation,
                        'solution_paper': solution['paper'],
                        'solution_method': solution['method'],
                        'similarity_score': solution['similarity'],
                        'idea_type': 'cross_domain_transfer'  # Cross-domain transfer
                    })

            # Sort by similarity, take top N
            research_ideas = sorted(
                research_ideas,
                key=lambda x: x['similarity_score'],
                reverse=True
            )[:self.frontier_max_ideas]

            return research_ideas

        except Exception as e:
            logger.warning(f"Generate cross-domain idea failed: {e}")
            return []


def create_analyzer(config: Dict = None) -> TopicEvolutionAnalyzer:
    """
    Factory function: create analyzer instance

    Args:
        config: Config dict

    Returns:
        TopicEvolutionAnalyzer instance
    """
    return TopicEvolutionAnalyzer(config)


if __name__ == "__main__":
    # Test code
    import networkx as nx

    # Create test graph
    G = nx.DiGraph()

    # Add test nodes
    papers = [
        {'id': 'p1', 'title': 'Attention Is All You Need', 'year': 2017, 'cited_by_count': 50000},
        {'id': 'p2', 'title': 'BERT: Pre-training of Deep Bidirectional Transformers', 'year': 2018, 'cited_by_count': 30000},
        {'id': 'p3', 'title': 'GPT-3: Language Models are Few-Shot Learners', 'year': 2020, 'cited_by_count': 20000},
        {'id': 'p4', 'title': 'Vision Transformer for Image Recognition', 'year': 2020, 'cited_by_count': 15000},
        {'id': 'p5', 'title': 'Switch Transformers: Scaling to Trillion Parameter Models', 'year': 2021, 'cited_by_count': 5000},
    ]

    for paper in papers:
        G.add_node(paper['id'], **paper)

    # Add citation relationships
    G.add_edge('p2', 'p1', edge_type='Extends')
    G.add_edge('p3', 'p2', edge_type='Overcomes')
    G.add_edge('p4', 'p1', edge_type='Adapts_to')
    G.add_edge('p5', 'p3', edge_type='Extends')

    # Create analyzer
    analyzer = create_analyzer()

    # Execute analysis
    report = analyzer.analyze(G, 'Transformer Neural Networks')

    # Output report
    print("\n" + "="*60)
    print("Analysis report:")
    print("="*60)
    print(f"Topic: {report['topic']}")
    print(f"Time span: {report['graph_overview']['year_range']}")
    print(f"Milestone paper count: {len(report['milestone_papers'])}")
    print(f"Research branch count: {len(report['research_branches'])}")
