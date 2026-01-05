"""
Knowledge Graph Construction and Visualization Module
Build and visualize paper citation relationship graphs
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
    """Custom JSON encoder to handle datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class CitationGraph:
    """
    Paper Citation Knowledge Graph Builder
    """

    def __init__(self, topic: str = ""):
        self.graph = nx.DiGraph()
        self.papers = {}  # Store detailed paper information
        self.topic = topic  # Research topic
        self.deep_survey_report = {}  # Store deep survey report
        self.research_ideas = {}  # Store research ideas
        logger.info("Knowledge graph builder initialized")

    # Nodes
    def add_paper_node(self, paper: Dict) -> None:
        """
        Add paper node to the graph

        Args:
            paper: Paper information dictionary
        """
        paper_id = paper['id']

        # Store paper information
        self.papers[paper_id] = paper

        # Extract node attributes
        node_attrs = {
            'title': paper.get('title', 'Unknown'),
            'authors': paper.get('authors', []),
            'year': paper.get('year', 0),
            'cited_by_count': paper.get('cited_by_count', 0),
            'venue': paper.get('venue', ''),
            'is_open_access': paper.get('is_open_access', False),
            'is_seed': paper.get('is_seed', False)  # Add seed node marker
        }

        # If AI analysis results exist, add to node attributes (compatible with both ai_analysis and deep_analysis fields)
        analysis = paper.get('deep_analysis') or paper.get('ai_analysis')
        if analysis:
            # Compatible with two data structures
            # deep_analysis structure: {problem: {content: ...}, method: {content: ...}, ...}
            # ai_analysis structure: {research_problem: ..., solution: ..., ...}

            # Extract problem/research_problem
            problem = ''
            if isinstance(analysis.get('problem'), dict):
                problem = analysis['problem'].get('content', '')
            else:
                problem = analysis.get('research_problem', '')

            # Extract method/contributions
            method = ''
            contributions_list = []
            if isinstance(analysis.get('method'), dict):
                method = analysis['method'].get('content', '')
            else:
                contributions_list = analysis.get('contributions', [])

            # Extract limitation/limitations
            limitation = ''
            limitations_list = []
            if isinstance(analysis.get('limitation'), dict):
                limitation = analysis['limitation'].get('content', '')
            else:
                limitations_list = analysis.get('limitations', [])

            # Extract future_work
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
                'deep_analysis': analysis  # Save complete deep_analysis structure
            })

        # If rag_analysis results exist, add to node attributes (for fragment pooling)
        rag_analysis = paper.get('rag_analysis')
        if rag_analysis:
            node_attrs.update({
                'rag_limitation': rag_analysis.get('limitation', ''),
                'rag_method': rag_analysis.get('method', ''),
                'rag_problem': rag_analysis.get('problem', ''),
                'rag_future_work': rag_analysis.get('future_work', '')
            })

        self.graph.add_node(paper_id, **node_attrs)
        logger.debug(f"Added paper node: {paper_id}")

    def _calculate_node_size(self, cited_count: int, graph) -> float:
        """
        Calculate node size based on relative importance of citation count

        Args:
            cited_count: Citation count of the paper
            graph: Current graph (for calculating relative importance)

        Returns:
            Node size
        """
        # Get citation counts of all papers in the graph
        all_citations = []
        for node in graph.nodes():
            paper = self.papers.get(node, {})
            citations = paper.get('cited_by_count', 0)
            all_citations.append(citations)

        if not all_citations:
            return 20  # Default size

        # Calculate statistics
        min_citations = min(all_citations)
        max_citations = max(all_citations)
        avg_citations = sum(all_citations) / len(all_citations)

        # Define size range
        min_size = 8   # Minimum node size
        max_size = 60  # Maximum node size

        if max_citations == min_citations:
            # All papers have same citation count, return medium size
            return (min_size + max_size) / 2

        # Use logarithmic scaling to make differences more visible but not too extreme
        import math
        if cited_count <= 0:
            return min_size

        # Logarithmic scaling formula
        log_cited = math.log(cited_count + 1)
        log_max = math.log(max_citations + 1)
        log_min = math.log(min_citations + 1) if min_citations > 0 else 0

        if log_max == log_min:
            return (min_size + max_size) / 2

        # Linear mapping to target range
        normalized = (log_cited - log_min) / (log_max - log_min)
        size = min_size + normalized * (max_size - min_size)

        # Additional tiering logic
        if cited_count >= avg_citations * 3:
            # Very high citation papers, increase further
            size *= 1.2
        elif cited_count >= avg_citations * 2:
            # High citation papers, moderate increase
            size *= 1.1
        elif cited_count < avg_citations * 0.3:
            # Low citation papers, moderate decrease
            size *= 0.8

        return max(min_size, min(max_size, size))

    def _get_node_color(self, year):
        """Get node color based on year"""
        if not isinstance(year, int) or year < 1900:
            return '#808080'  # Gray for unknown year

        # Map year to color value (1990-2024)
        min_year, max_year = 1990, 2024
        normalized = (year - min_year) / (max_year - min_year)
        normalized = max(0, min(1, normalized))  # Ensure in 0-1 range

        # Use HSV color space: from blue to red
        hue = (1 - normalized) * 240  # 240 degrees is blue, 0 degrees is red
        return f'hsl({hue:.0f}, 70%, 60%)'

    # Edges
    def add_citation_edge(self, from_paper_id: str, to_paper_id: str, edge_type: str = "CITES") -> None:
        """
        Add citation relationship edge

        Args:
            from_paper_id: Citing paper ID
            to_paper_id: Cited paper ID
            edge_type: Citation relationship type
        """
        if from_paper_id in self.graph and to_paper_id in self.graph:
            self.graph.add_edge(
                from_paper_id,
                to_paper_id,
                edge_type=edge_type,
                weight=1
            )
            logger.debug(f"Added citation edge: {from_paper_id} -> {to_paper_id} ({edge_type})")

    def _infer_edge_type(self, citing_id: str, cited_id: str) -> str:
        """
        Execute during stage 4 for citation relationship inference

        Args:
            citing_id: Citing paper ID
            cited_id: Cited paper ID

        Returns:
            Citation relationship type
        """
        if citing_id not in self.papers or cited_id not in self.papers:
            return "CITES"

        citing_paper = self.papers[citing_id]
        cited_paper = self.papers[cited_id]

        # Get basic information
        citing_year = citing_paper.get('year', 0)
        cited_year = cited_paper.get('year', 0)
        citing_citations = citing_paper.get('cited_by_count', 0)
        cited_citations = cited_paper.get('cited_by_count', 0)

        # Get paper titles and key techniques (if AI analysis results exist)
        citing_title = citing_paper.get('title', '').lower()
        cited_title = cited_paper.get('title', '').lower()

        citing_techniques = []
        cited_techniques = []
        if 'ai_analysis' in citing_paper:
            citing_techniques = [tech.lower() for tech in citing_paper['ai_analysis'].get('key_techniques', [])]
        if 'ai_analysis' in cited_paper:
            cited_techniques = [tech.lower() for tech in cited_paper['ai_analysis'].get('key_techniques', [])]

        # 1. Time-based inference
        year_diff = citing_year - cited_year if citing_year > 0 and cited_year > 0 else 0

        # 2. Influence-based inference
        citation_ratio = citing_citations / max(1, cited_citations) if cited_citations > 0 else 1

        # 3. Technique similarity-based inference
        common_techniques = set(citing_techniques) & set(cited_techniques)
        technique_similarity = len(common_techniques) / max(1, len(set(citing_techniques) | set(cited_techniques)))

        # 4. Title keyword-based inference
        title_similarity = self._calculate_title_similarity(citing_title, cited_title)

        # Complex inference logic
        if year_diff >= 10:
            if cited_citations > 1000:
                return "CLASSIC_REFERENCE"  # Citing classic literature
            else:
                return "HISTORICAL_REFERENCE"  # Historical reference

        elif year_diff >= 5:
            if technique_similarity > 0.5:
                return "BUILDS_ON"  # Extends based on related work
            else:
                return "BACKGROUND_REFERENCE"  # Background knowledge citation

        elif abs(year_diff) <= 2:
            if technique_similarity > 0.7:
                return "DIRECT_COMPARISON"  # Direct comparison
            elif technique_similarity > 0.3:
                return "RELATED_WORK"  # Related work
            elif citation_ratio > 2:
                return "CHALLENGES"  # Challenges existing work
            else:
                return "CONTEMPORARY_REFERENCE"  # Contemporary reference

        elif year_diff < 0:  # Citing newer papers (rare but possible)
            return "FORWARD_REFERENCE"  # Forward reference

        else:
            # Comprehensive judgment
            if title_similarity > 0.5 or technique_similarity > 0.5:
                return "METHODOLOGICAL_REFERENCE"  # Methodological citation
            elif cited_citations > citing_citations * 5:
                return "AUTHORITATIVE_REFERENCE"  # Authoritative citation
            else:
                return "GENERAL_REFERENCE"  # General citation

    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """
        Calculate title similarity

        Args:
            title1: First title
            title2: Second title

        Returns:
            Similarity score (0-1)
        """
        if not title1 or not title2:
            return 0.0

        # Simple keyword overlap calculation
        words1 = set(title1.split())
        words2 = set(title2.split())

        # Filter common stop words
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
        Create edge traces for different types

        Args:
            graph: NetworkX graph object
            pos: Node position dictionary

        Returns:
            List of edge traces
        """
        # Define styles for different citation types
        edge_styles = {
            'CLASSIC_REFERENCE': {'color': '#FF6B6B', 'width': 2.5, 'dash': 'solid'},      # Red, thick line
            'BUILDS_ON': {'color': '#4ECDC4', 'width': 2.0, 'dash': 'solid'},             # Cyan, medium
            'DIRECT_COMPARISON': {'color': '#45B7D1', 'width': 2.0, 'dash': 'dash'},       # Blue, dashed
            'METHODOLOGICAL_REFERENCE': {'color': '#96CEB4', 'width': 1.5, 'dash': 'solid'}, # Green
            'AUTHORITATIVE_REFERENCE': {'color': '#FFEAA7', 'width': 2.0, 'dash': 'solid'},  # Yellow
            'RELATED_WORK': {'color': '#DDA0DD', 'width': 1.5, 'dash': 'dot'},            # Purple, dotted
            'CONTEMPORARY_REFERENCE': {'color': '#FFB6C1', 'width': 1.0, 'dash': 'solid'}, # Pink, thin line
            'GENERAL_REFERENCE': {'color': '#D3D3D3', 'width': 1.0, 'dash': 'solid'},      # Gray, default
            'HISTORICAL_REFERENCE': {'color': '#CD853F', 'width': 1.0, 'dash': 'dot'},    # Brown, dotted
            'BACKGROUND_REFERENCE': {'color': '#C0C0C0', 'width': 0.8, 'dash': 'solid'}   # Silver, thin line
        }

        # Group edges by type
        edges_by_type = {}
        for edge in graph.edges(data=True):
            from_node, to_node, attrs = edge
            edge_type = attrs.get('edge_type', 'GENERAL_REFERENCE')

            if edge_type not in edges_by_type:
                edges_by_type[edge_type] = []
            edges_by_type[edge_type].append((from_node, to_node))

        # Create trace for each type
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

            if edge_x:  # Only create trace when edges exist
                # Define Chinese names and descriptions for edge types
                edge_type_names = {
                    'CLASSIC_REFERENCE': 'Classic Citation (10+ years, high citations)',
                    'BUILDS_ON': 'Technical Extension (based on related work)',
                    'DIRECT_COMPARISON': 'Direct Comparison (high technical similarity)',
                    'METHODOLOGICAL_REFERENCE': 'Methodological Citation',
                    'AUTHORITATIVE_REFERENCE': 'Authoritative Citation',
                    'RELATED_WORK': 'Related Work',
                    'CONTEMPORARY_REFERENCE': 'Contemporary Reference',
                    'GENERAL_REFERENCE': 'General Citation',
                    'HISTORICAL_REFERENCE': 'Historical Reference',
                    'BACKGROUND_REFERENCE': 'Background Citation'
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
                    showlegend=True,  # Enable legend
                    name=display_name,
                    legendgroup=edge_type  # Group display
                ))

        return traces


    # Overall page layout
    def _generate_interactive_html_page(self, subgraph, pos) -> str:
        """
        Generate complete interactive HTML page

        Args:
            subgraph: Network graph subgraph
            pos: Node position dictionary

        Returns:
            Complete HTML page content
        """
        # Prepare node data
        nodes_data = []
        for node in subgraph.nodes():
            paper = self.papers.get(node, {})
            authors = paper.get('authors', [])
            first_author = authors[0] if authors else "Unknown"
            # Extract surname (assume it's the last word)
            first_author_surname = first_author.split()[-1] if first_author != "Unknown" else "Unknown"
            year = paper.get('year', 'Unknown')

            # Extract RAG analysis results
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
                # Add RAG analysis results
                'rag_problem': rag_analysis.get('problem', ''),
                'rag_method': rag_analysis.get('method', ''),
                'rag_limitation': rag_analysis.get('limitation', ''),
                'rag_future_work': rag_analysis.get('future_work', ''),
                'analysis_method': paper.get('analysis_method', ''),
                'sections_extracted': paper.get('sections_extracted', 0)
            })

        # Prepare edge data
        edges_data = []

        # Socket Matching relationship type styles (Tech Tree Schema - 6 core types)
        edge_styles = {
            # === Socket Matching core types (6 types) ===
            'Overcomes': {
                'color': '#E74C3C',      # Red - main path (overcome/optimize)
                'width': 3.0,
                'dash': 'solid',
                'description': 'Overcome/Optimize - B solves A\'s limitations (vertical deepening)'
            },
            'Realizes': {
                'color': '#9B59B6',      # Purple - forward-looking validation (realize vision)
                'width': 2.5,
                'dash': 'solid',
                'description': 'Realize Vision - B implements A\'s future work suggestions (research inheritance)'
            },
            'Extends': {
                'color': '#2ECC71',      # Green - method extension (micro-innovation)
                'width': 2.0,
                'dash': 'solid',
                'description': 'Method Extension - B makes incremental improvements on A\'s method (micro-innovation)'
            },
            'Alternative': {
                'color': '#E67E22',      # Orange - alternative approach (disruptive innovation)
                'width': 2.0,
                'dash': 'dot',
                'description': 'Alternative Approach - B solves problem with completely different paradigm (disruptive innovation)'
            },
            'Adapts_to': {
                'color': '#3498DB',      # Blue - branch diffusion (transfer/application)
                'width': 2.0,
                'dash': 'dash',
                'description': 'Transfer/Application - B applies A\'s method to new domain (horizontal diffusion)'
            },
            'Baselines': {
                'color': '#95A5A6',      # Gray - background noise (baseline comparison)
                'width': 1.0,
                'dash': 'solid',
                'description': 'Baseline Comparison - B only uses A as comparison object (no direct inheritance)'
            },

            # === Traditional types (backward compatibility) ===
            'CLASSIC_REFERENCE': {'color': '#FF6B6B', 'width': 2.5, 'dash': 'solid', 'description': 'Classic Citation'},
            'BUILDS_ON': {'color': '#4ECDC4', 'width': 2.0, 'dash': 'solid', 'description': 'Technical Extension'},
            'DIRECT_COMPARISON': {'color': '#45B7D1', 'width': 2.0, 'dash': 'dash', 'description': 'Direct Comparison'},
            'METHODOLOGICAL_REFERENCE': {'color': '#96CEB4', 'width': 1.5, 'dash': 'solid', 'description': 'Methodological Citation'},
            'AUTHORITATIVE_REFERENCE': {'color': '#FFEAA7', 'width': 2.0, 'dash': 'solid', 'description': 'Authoritative Citation'},
            'RELATED_WORK': {'color': '#DDA0DD', 'width': 1.5, 'dash': 'dot', 'description': 'Related Work'},
            'CONTEMPORARY_REFERENCE': {'color': '#FFB6C1', 'width': 1.0, 'dash': 'solid', 'description': 'Contemporary Citation'},
            'GENERAL_REFERENCE': {'color': '#CCCCCC', 'width': 1.0, 'dash': 'solid', 'description': 'General Citation'},
            'HISTORICAL_REFERENCE': {'color': '#CD853F', 'width': 1.0, 'dash': 'dot', 'description': 'Historical Citation'},
            'BACKGROUND_REFERENCE': {'color': '#C0C0C0', 'width': 0.8, 'dash': 'solid', 'description': 'Background Citation'}
        }

        for edge in subgraph.edges(data=True):
            from_node, to_node, attrs = edge
            edge_type = attrs.get('edge_type', 'Baselines')  # Default to Baselines
            style = edge_styles.get(edge_type, edge_styles.get('Baselines', {'color': '#CCCCCC', 'width': 1.0, 'dash': 'solid', 'description': 'Unknown Type'}))

            if from_node in pos and to_node in pos:
                edges_data.append({
                    'from': from_node,
                    'to': to_node,
                    'type': edge_type,
                    'color': style['color'],  # Directly use corresponding color
                    'original_color': style['color'],
                    'width': style['width'],
                    'dash': style['dash'],
                    'description': style.get('description', edge_type)
                })

        # Get year range
        years = [node['year'] for node in nodes_data if isinstance(node['year'], int)]
        min_year, max_year = (min(years), max(years)) if years else (2000, 2020)

        # Generate HTML content
        # Build title
        title = f"{self.topic} - Interactive Paper Citation Knowledge Graph" if self.topic else "Interactive Paper Citation Knowledge Graph"

        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
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
                <!-- Left graph section (70%) -->
                <div class="graph-section">
                    <div class="title">{title}</div>
                    <div class="graph-container">
                        <div id="graph" style="width:100%; height:100%;"></div>
                    </div>
                    <div class="legend-container">
                        <h4 style="margin-top:0; color:#2c3e50;">🔌 Socket Matching Citation Relationship Types (6 Core Types)</h4>
                        <div class="legend-grid">
                            <!-- Socket Matching core types (6 types) -->
                            <div class="legend-item" style="border-left: 3px solid #E74C3C;">
                                <div class="legend-color" style="background-color:#E74C3C; height:3px;"></div>
                                <span><strong>Overcomes</strong> - Overcome/Optimize (Vertical Deepening)</span>
                            </div>
                            <div class="legend-item" style="border-left: 3px solid #9B59B6;">
                                <div class="legend-color" style="background-color:#9B59B6; height:3px;"></div>
                                <span><strong>Realizes</strong> - Realize Vision (Research Inheritance)</span>
                            </div>
                            <div class="legend-item" style="border-left: 3px solid #2ECC71;">
                                <div class="legend-color" style="background-color:#2ECC71; height:2px;"></div>
                                <span><strong>Extends</strong> - Method Extension (Micro-innovation)</span>
                            </div>
                            <div class="legend-item" style="border-left: 3px solid #E67E22;">
                                <div class="legend-color" style="background-color:#E67E22; border: 2px dotted #E67E22; height:1px;"></div>
                                <span><strong>Alternative</strong> - Alternative Approach (Disruptive Innovation)</span>
                            </div>
                            <div class="legend-item" style="border-left: 3px solid #3498DB;">
                                <div class="legend-color" style="background-color:#3498DB; border: 2px dashed #3498DB; height:1px;"></div>
                                <span><strong>Adapts_to</strong> - Transfer/Application (Horizontal Diffusion)</span>
                            </div>
                            <div class="legend-item" style="border-left: 3px solid #95A5A6;">
                                <div class="legend-color" style="background-color:#95A5A6; height:1px;"></div>
                                <span><strong>Baselines</strong> - Baseline Comparison (Background Noise)</span>
                            </div>
                        </div>
                        <p style="margin-top:10px; font-size:11px; color:#666;">
                            💡 <strong>Logic Matching Matrix (4 Matches → 6 Types)</strong>: Match1(Limitation→Problem) → Overcomes | Match2(FutureWork→Problem) → Realizes | Match3(Method→Method) → Extends/Alternative | Match4(Cross-domain Problem) → Adapts_to | No Match → Baselines
                        </p>
                    </div>
                </div>

                <!-- Right details section (30%) -->
                <div class="details-section">
                    <!-- Tab navigation -->
                    <div class="tabs">
                        <div class="tab active" onclick="switchTab(event, 'paper-tab')">📄 Paper Details</div>
                        <div class="tab" onclick="switchTab(event, 'survey-tab')">📝 Deep Survey</div>
                        <div class="tab" onclick="switchTab(event, 'ideas-tab')">💡 Research Ideas</div>
                    </div>

                    <!-- Paper details tab -->
                    <div id="paper-tab" class="tab-content active">
                        <div class="stats">
                            <h4 style="margin-top:0;">Graph Statistics</h4>
                            <div class="stat-item">
                                <span>Total Papers:</span>
                                <span>{len(nodes_data)}</span>
                            </div>
                            <div class="stat-item">
                                <span>Citation Relationships:</span>
                                <span>{len(edges_data)}</span>
                            </div>
                            <div class="stat-item">
                                <span>Time Span:</span>
                                <span>{min_year} - {max_year}</span>
                            </div>
                        </div>
                        <div class="placeholder">
                            👆 Click on nodes in the graph to view detailed paper information
                        </div>
                    </div>

                    <!-- Deep survey tab -->
                    <div id="survey-tab" class="tab-content"></div>

                    <!-- Research ideas tab -->
                    <div id="ideas-tab" class="tab-content"></div>
                </div>
            </div>

            <script>
                // ========== Data Initialization ==========
                const nodesData = {json.dumps(nodes_data, ensure_ascii=False, indent=2)};
                const edgesData = {json.dumps(edges_data, ensure_ascii=False, indent=2)};

                // Deep Survey data
                const deepSurveyData = {json.dumps(self.deep_survey_report, ensure_ascii=False, indent=2, cls=DateTimeEncoder)};

                // Research Ideas data
                const researchIdeasData = {json.dumps(self.research_ideas, ensure_ascii=False, indent=2, cls=DateTimeEncoder)};

                // Group edge data by type
                const edgesByType = {{}};
                edgesData.forEach(edge => {{
                    if (!edgesByType[edge.type]) {{
                        edgesByType[edge.type] = [];
                    }}
                    edgesByType[edge.type].push(edge);
                }});

                // Create node trace
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
                    name: 'Paper Nodes'
                }};

                // Chart layout configuration
                const layout = {{
                    title: '',
                    showlegend: false,
                    hovermode: 'closest',
                    margin: {{ l: 0, r: 0, b: 40, t: 0 }},
                    xaxis: {{
                        title: 'Publication Year',
                        showgrid: true,
                        gridcolor: 'lightgray',
                        range: [{min_year - 1}, {max_year + 1}]
                    }},
                    yaxis: {{
                        title: 'Paper Distribution',
                        showgrid: true,
                        gridcolor: 'lightgray',
                        showticklabels: false
                    }},
                    plot_bgcolor: 'white',
                    paper_bgcolor: 'white'
                }};

                // ========== Utility Functions ==========
                // Create edge traces (generic function, eliminate duplicate logic)
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

                // Update graph (generic function)
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

                // ========== Initialize Chart ==========
                // Create initial edge style (using originally defined colors)
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

                // ========== Event Handlers ==========
                document.getElementById('graph').on('plotly_click', function(data) {{
                    if (data.points?.[0]?.customdata) {{
                        const node = data.points[0].customdata;
                        const nodeIndex = data.points[0].pointIndex;
                        showPaperDetails(node);
                        highlightClickedNodeAndEdges(nodeIndex, node);
                    }}
                }});


                // ========== Feature Functions ==========
                // Tab switch function
                function switchTab(event, tabId) {{
                    // Remove all active classes
                    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

                    // Add active class to current tab
                    event.target.classList.add('active');
                    document.getElementById(tabId).classList.add('active');

                    // Reset graph highlight when switching to "Paper Details" or "Research Ideas" tab
                    if (tabId === 'paper-tab' || tabId === 'ideas-tab') {{
                        resetGraphHighlight();
                        console.log(`Switched to ${{tabId}}, graph highlight reset`);
                    }}

                    // Load corresponding content based on tab ID
                    if (tabId === 'survey-tab') {{
                        renderDeepSurvey();
                    }} else if (tabId === 'ideas-tab') {{
                        renderResearchIdeas();
                    }}
                }}

                // Render deep survey (new data structure)
                function renderDeepSurvey() {{
                    const surveyTab = document.getElementById('survey-tab');

                    if (!deepSurveyData || Object.keys(deepSurveyData).length === 0) {{
                        surveyTab.innerHTML = '<div class="placeholder">No deep survey data available</div>';
                        return;
                    }}

                    let html = '<div style="padding:20px;">';

                    // Summary information (new structure)
                    if (deepSurveyData.summary) {{
                        html += `
                            <div class="stats">
                                <h4 style="margin-top:0;">📊 Survey Summary</h4>
                                <div class="stat-item"><span>Original Papers:</span><span>${{deepSurveyData.summary.original_papers || 0}} papers</span></div>
                                <div class="stat-item"><span>Pruned Papers:</span><span>${{deepSurveyData.summary.pruned_papers || 0}} papers</span></div>
                                <div class="stat-item"><span>Evolutionary Threads:</span><span>${{deepSurveyData.summary.total_threads || 0}} threads</span></div>
                            </div>
                        `;
                    }}

                    // Pruning statistics
                    if (deepSurveyData.pruning_stats) {{
                        const stats = deepSurveyData.pruning_stats;
                        const retentionRate = (stats.retention_rate * 100).toFixed(1);
                        html += `
                            <div class="stats" style="margin-top:15px; background:#fff3cd;">
                                <h4 style="margin-top:0;">✂️ Graph Pruning Statistics</h4>
                                <div class="stat-item"><span>Retention Rate:</span><span>${{retentionRate}}%</span></div>
                                <div class="stat-item"><span>Seed Papers:</span><span>${{stats.seed_papers || 0}} papers</span></div>
                                <div class="stat-item"><span>Strong Edges:</span><span>${{stats.strong_edges || 0}} edges</span></div>
                                <div class="stat-item"><span>Weak Edges Removed:</span><span>${{stats.weak_edges_removed || 0}} edges</span></div>
                            </div>
                        `;
                    }}

                    // Evolutionary paths (Threads)
                    const threads = deepSurveyData.survey_report?.threads || deepSurveyData.evolutionary_paths || [];
                    if (threads.length > 0) {{
                        html += `
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:20px;">
                                <h3 style="color:#2c3e50; margin:0;">🧵 Key Evolutionary Threads</h3>
                                <button id="resetGraphBtn" onclick="resetGraphHighlight()"
                                    style="padding:5px 12px; background:#6c757d; color:white; border:none; border-radius:4px; cursor:pointer; font-size:12px;">
                                    🔄 Reset Graph
                                </button>
                            </div>
                        `;
                        threads.forEach((thread, index) => {{
                            const threadTitle = thread.title || thread.thread_name || `Thread ${{index + 1}}`;
                            const patternType = thread.pattern_type || thread.thread_type || 'Unknown Type';
                            const paperCount = thread.papers ? thread.papers.length : 0;
                            const narrative = thread.narrative || 'No narrative text available';

                            // Define rich color palette (assigned by thread index)
                            const colorPalette = [
                                '#E74C3C',  // Red - Thread 0
                                '#3498DB',  // Blue - Thread 1
                                '#2ECC71',  // Green - Thread 2
                                '#F39C12',  // Orange - Thread 3
                                '#9B59B6',  // Purple - Thread 4
                                '#1ABC9C',  // Cyan - Thread 5
                                '#E67E22',  // Deep Orange - Thread 6
                                '#95A5A6',  // Gray - Thread 7
                                '#34495E',  // Deep Blue-Gray - Thread 8
                                '#16A085'   // Deep Cyan - Thread 9
                            ];

                            // Assign color based on thread index (ensure each thread has unique color)
                            let borderColor = colorPalette[index % colorPalette.length];
                            let highlightColor = borderColor;

                            // Collect all paper IDs for this thread
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
                                    <p><strong>📚 Paper Count:</strong> ${{paperCount}} papers</p>
                                    ${{thread.total_citations ? `<p><strong>📊 Total Citations:</strong> ${{thread.total_citations}}</p>` : ''}}
                                    <p style="font-size:11px; color:#666; margin-top:8px;">
                                        💡 <em>Click this card to highlight all papers in this thread on the left graph</em>
                                    </p>

                                    <details style="margin-top:10px;">
                                        <summary style="cursor:pointer; color:#3498DB; font-weight:bold;">📖 View Evolutionary Narrative</summary>
                                        <div style="margin-top:10px; padding:10px; background:#f8f9fa; border-radius:5px; line-height:1.6; white-space:pre-wrap;">
                                            ${{narrative}}
                                        </div>
                                    </details>

                                    ${{thread.papers && thread.papers.length > 0 ? `
                                        <details style="margin-top:10px;">
                                            <summary style="cursor:pointer; color:#2ECC71; font-weight:bold;">📄 View Paper List</summary>
                                            <ul style="margin-top:10px; padding-left:20px;">
                                                ${{thread.papers.map((p, pIndex) => `
                                                    <li style="margin:5px 0; cursor:pointer; color:#2980b9; transition:color 0.2s;"
                                                        onclick="event.stopPropagation(); showPaperFromThread('${{p.paper_id}}');"
                                                        onmouseover="this.style.color='#3498db'; this.style.textDecoration='underline';"
                                                        onmouseout="this.style.color='#2980b9'; this.style.textDecoration='none';"
                                                        title="Click to view paper details and highlight in graph">
                                                        <strong>${{p.title}}</strong>
                                                        (${{p.year || 'N/A'}}, Citations: ${{p.cited_by_count || 0}})
                                                    </li>
                                                `).join('')}}
                                            </ul>
                                        </details>
                                    ` : ''}}
                                </div>
                            `;
                        }});
                    }}

                    // Survey report abstract
                    if (deepSurveyData.survey_report?.abstract) {{
                        html += `
                            <div style="margin-top:20px; padding:15px; background:#e8f4f8; border-left:4px solid #3498DB; border-radius:5px;">
                                <h4 style="margin:0 0 10px 0; color:#2c3e50;">📝 Survey Abstract</h4>
                                <p style="line-height:1.6; color:#333; margin:0;">${{deepSurveyData.survey_report.abstract}}</p>
                            </div>
                        `;
                    }}

                    html += '</div>';
                    surveyTab.innerHTML = html;
                }}

                // Render research ideas
                function renderResearchIdeas() {{
                    const ideasTab = document.getElementById('ideas-tab');

                    if (!researchIdeasData || Object.keys(researchIdeasData).length === 0) {{
                        ideasTab.innerHTML = '<div class="placeholder">No research ideas data available</div>';
                        return;
                    }}

                    let html = '<div style="padding:20px;">';

                    // Statistics
                    html += `
                        <div class="stats">
                            <h4 style="margin-top:0;">💡 Idea Generation Statistics</h4>
                            <div class="stat-item"><span>Total Ideas:</span><span>${{researchIdeasData.total_ideas || 0}}</span></div>
                            <div class="stat-item"><span>Feasible Ideas:</span><span>${{researchIdeasData.successful_ideas || 0}}</span></div>
                            ${{researchIdeasData.pools ? `
                                <div class="stat-item"><span>Unsolved Limitations:</span><span>${{researchIdeasData.pools.unsolved_limitations || 0}}</span></div>
                                <div class="stat-item"><span>Candidate Methods:</span><span>${{researchIdeasData.pools.candidate_methods || 0}}</span></div>
                            ` : ''}}
                        </div>
                    `;

                    // Ideas list
                    if (researchIdeasData.ideas && researchIdeasData.ideas.length > 0) {{
                        html += '<h3 style="color:#2c3e50; margin-top:20px;">💡 Research Ideas List</h3>';
                        researchIdeasData.ideas.forEach((idea, index) => {{
                            const statusClass = idea.status === 'SUCCESS' ? 'status-success' : 'status-incompatible';
                            const statusText = idea.status === 'SUCCESS' ? '✓ Feasible' : '✗ Incompatible';

                            html += `
                                <div class="idea-card">
                                    <h4>
                                        Idea ${{index + 1}}: ${{idea.title || 'Untitled Idea'}}
                                        <span class="status-badge ${{statusClass}}">${{statusText}}</span>
                                    </h4>
                                    ${{idea.abstract ? `
                                        <p style="margin:10px 0; line-height:1.6; color:#444;">
                                            <strong>Abstract:</strong> ${{idea.abstract}}
                                        </p>
                                    ` : ''}}
                                    ${{idea.modification ? `
                                        <p style="margin:8px 0; padding:10px; background:#f8f9fa; border-radius:4px;">
                                            <strong>🔧 Key Innovation:</strong> ${{idea.modification}}
                                        </p>
                                    ` : ''}}
                                    ${{idea.reasoning ? `
                                        <details style="margin-top:10px;">
                                            <summary style="cursor:pointer; color:#3498DB;"><strong>View Reasoning Process</strong></summary>
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
                                      (node.authors.length > 5 ? ' et al.' : '');

                    // Build RAG analysis section HTML
                    let ragAnalysisHTML = '';
                    if (node.rag_problem || node.rag_method || node.rag_limitation || node.rag_future_work) {{
                        ragAnalysisHTML = `
                            <div class="paper-info" style="background:#e8f4f8; padding:15px; border-radius:8px; margin-top:15px;">
                                <h4 style="margin:0 0 10px 0; color:#1a73e8; font-size:15px;">🧠 Multi-Agent System Deep Analysis</h4>
                                ${{node.analysis_method ? `<p style="font-size:12px; color:#666; margin-bottom:10px;"><strong>Analysis Method:</strong> ${{node.analysis_method.toUpperCase()}}</p>` : ''}}
                                ${{node.sections_extracted ? `<p style="font-size:12px; color:#666; margin-bottom:10px;"><strong>Sections Extracted:</strong> ${{node.sections_extracted}} sections</p>` : ''}}
                            </div>
                            ${{node.rag_problem ? `
                            <div class="paper-info" style="border-left:3px solid #FF6B6B; padding-left:10px;">
                                <h4 style="margin:0 0 8px 0; color:#FF6B6B; font-size:14px;">📋 Research Problem</h4>
                                <p style="font-size:13px; line-height:1.6; color:#333;">${{node.rag_problem}}</p>
                            </div>
                            ` : ''}}
                            ${{node.rag_method ? `
                            <div class="paper-info" style="border-left:3px solid #4ECDC4; padding-left:10px;">
                                <h4 style="margin:0 0 8px 0; color:#4ECDC4; font-size:14px;">💡 Core Method</h4>
                                <p style="font-size:13px; line-height:1.6; color:#333;">${{node.rag_method}}</p>
                            </div>
                            ` : ''}}
                            ${{node.rag_limitation ? `
                            <div class="paper-info" style="border-left:3px solid #FFA500; padding-left:10px;">
                                <h4 style="margin:0 0 8px 0; color:#FFA500; font-size:14px;">⚠️ Limitation</h4>
                                <p style="font-size:13px; line-height:1.6; color:#333;">${{node.rag_limitation}}</p>
                            </div>
                            ` : ''}}
                            ${{node.rag_future_work ? `
                            <div class="paper-info" style="border-left:3px solid #9B59B6; padding-left:10px;">
                                <h4 style="margin:0 0 8px 0; color:#9B59B6; font-size:14px;">🔮 Future Work</h4>
                                <p style="font-size:13px; line-height:1.6; color:#333;">${{node.rag_future_work}}</p>
                            </div>
                            ` : ''}}
                        `;
                    }}

                    document.getElementById('paper-tab').innerHTML = `
                        <div class="stats">
                            <h4 style="margin-top:0;">Graph Statistics</h4>
                            <div class="stat-item"><span>Total Papers:</span><span>{len(nodes_data)}</span></div>
                            <div class="stat-item"><span>Citation Relationships:</span><span>{len(edges_data)}</span></div>
                            <div class="stat-item"><span>Time Span:</span><span>{min_year} - {max_year}</span></div>
                        </div>
                        <div class="paper-info">
                            <h3>${{node.title}}</h3>
                            <p><strong>Authors:</strong> ${{authorsText}}</p>
                            <p><strong>Year:</strong> ${{node.year}}</p>
                            <p><strong>Citations:</strong> ${{node.cited_by_count}}</p>
                            <p><strong>Venue:</strong> ${{node.venue || 'Unknown'}}</p>
                            <p><strong>Paper ID:</strong> ${{node.id}}</p>
                        </div>
                        ${{ragAnalysisHTML}}
                    `;
                }}

                function highlightClickedNodeAndEdges(nodeIndex, node) {{
                    // Only change node color - highlight clicked node
                    const nodeColors = nodesData.map((n, i) =>
                        i === nodeIndex ? '#FF4444' : n.color);

                    // Keep edges with original style unchanged
                    updateGraph(initialEdgeTraces, nodeColors);
                }}

                // ========== Evolutionary Thread Highlight Feature ==========
                function highlightThread(threadIndex, highlightColor) {{
                    // Get thread data
                    const threads = deepSurveyData.survey_report?.threads || deepSurveyData.evolutionary_paths || [];
                    if (threadIndex >= threads.length) return;

                    const thread = threads[threadIndex];
                    const threadPaperIds = new Set(thread.papers?.map(p => p.paper_id) || []);

                    console.log(`Highlighting Thread ${{threadIndex}}: ${{thread.title}}, containing ${{threadPaperIds.size}} papers`);

                    // Update node colors and sizes
                    const newColors = [];
                    const newSizes = [];
                    const newLineStyles = [];

                    nodesData.forEach((node, index) => {{
                        if (threadPaperIds.has(node.id)) {{
                            // Highlight: keep node's original color, enlarge by 1.5x
                            newColors.push(node.color);
                            newSizes.push(node.size * 1.5);
                            newLineStyles.push({{ width: 3, color: node.color }});
                        }} else {{
                            // Other nodes: gray out, shrink to 0.5x
                            newColors.push('#D3D3D3');
                            newSizes.push(node.size * 0.5);
                            newLineStyles.push({{ width: 1, color: '#CCCCCC' }});
                        }}
                    }});

                    // Update graph
                    const highlightedNodeTrace = {{
                        ...nodeTrace,
                        marker: {{
                            ...nodeTrace.marker,
                            size: newSizes,
                            color: newColors,
                            line: newLineStyles
                        }}
                    }};

                    // Also adjust edge transparency (highlight edges within thread)
                    const highlightedEdgeTraces = createEdgeTracesWithHighlight(threadPaperIds, highlightColor);

                    Plotly.react('graph', [...highlightedEdgeTraces, highlightedNodeTrace], layout);

                    // Scroll graph to highlighted area
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
                    console.log('Reset graph highlight');
                    // Restore original colors and sizes
                    updateGraph(initialEdgeTraces, nodesData.map(n => n.color));
                }}

                // Click paper from deep survey paper list, show details and highlight
                function showPaperFromThread(paperId) {{
                    console.log('Clicked paper from deep survey:', paperId);

                    // Find corresponding node index
                    const nodeIndex = nodesData.findIndex(n => n.id === paperId);

                    if (nodeIndex === -1) {{
                        console.warn('Paper not found in graph:', paperId);
                        alert('This paper is not in the currently displayed graph nodes');
                        return;
                    }}

                    const node = nodesData[nodeIndex];

                    // Switch to paper details tab
                    const paperTab = document.querySelector('.tab[onclick*="paper-tab"]');
                    if (paperTab) {{
                        paperTab.click();
                    }}

                    // Show paper details
                    showPaperDetails(node);

                    // Find the evolutionary thread this paper belongs to
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

                    // Highlight entire thread (nodes keep original color)
                    if (threadIndex !== -1) {{
                        console.log(`This paper belongs to thread ${{threadIndex}}, will highlight entire thread`);
                        highlightThread(threadIndex, null);  // Pass null because highlightColor is no longer needed
                    }} else {{
                        // If no thread found, fall back to single node highlight
                        console.log('This paper does not belong to any thread, only highlight single node');
                        highlightClickedNodeAndEdges(nodeIndex, node);
                    }}

                    console.log('Paper details displayed and node highlighted:', node.title);
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
        Create time-based layout with time axis as horizontal coordinate

        Args:
            graph: NetworkX graph object

        Returns:
            Node position dictionary {node_id: (x, y)}
        """
        pos = {}

        # Get year information for all papers
        papers_with_years = []
        for node in graph.nodes():
            paper = self.papers.get(node, {})
            year = paper.get('year', 2000)  # Default year 2000
            if not isinstance(year, int) or year < 1900 or year > 2030:
                year = 2000
            papers_with_years.append((node, year))

        # If no papers, return empty layout
        if not papers_with_years:
            return pos

        # Sort by year
        papers_with_years.sort(key=lambda x: x[1])

        # Get year range
        years = [year for _, year in papers_with_years]
        min_year = min(years)
        max_year = max(years)

        # Group by year, calculate paper count per year
        year_groups = {}
        for node, year in papers_with_years:
            if year not in year_groups:
                year_groups[year] = []
            year_groups[year].append(node)

        # Assign position for each node, ensure no overlap
        for year, nodes in year_groups.items():
            # X coordinate: directly use year as coordinate
            x = year

            # Calculate Y coordinate distribution, ensure nodes are well dispersed
            num_papers = len(nodes)
            if num_papers == 1:
                y_positions = [0]
            else:
                # Increase Y-axis distribution range to avoid overlap
                y_range = max(8, num_papers * 2.5)  # Increase minimum range and spacing
                y_positions = np.linspace(-y_range/2, y_range/2, num_papers)

            # Sort by citation count, influential papers in the middle
            nodes_with_citations = []
            for node in nodes:
                paper = self.papers.get(node, {})
                cited_count = paper.get('cited_by_count', 0)
                nodes_with_citations.append((node, cited_count))

            # Sort by citation count
            nodes_with_citations.sort(key=lambda x: x[1], reverse=True)

            # Assign Y coordinates: influential ones in middle, others dispersed to sides
            for i, (node, cited_count) in enumerate(nodes_with_citations):
                y = y_positions[i]

                # Increase random offset to avoid complete overlap
                y += np.random.uniform(-0.5, 0.5)

                # Increase X coordinate offset to avoid horizontal overlap
                x_offset = np.random.uniform(-0.6, 0.6)

                pos[node] = (x + x_offset, y)

        return pos


    # Main function
    def build_citation_network(self, papers: List[Dict], citation_data: List) -> None:
        """
        Build complete citation network

        Args:
            papers: Paper list
            citation_data: Citation relationship list
                - Supports tuples: [(citing_id, cited_id), ...]
                - Supports triples: [(citing_id, cited_id, edge_type), ...]
        """
        logger.info(f"Building citation network: {len(papers)} papers, {len(citation_data)} citation relationships")

        # Add all paper nodes
        for paper in papers:
            self.add_paper_node(paper)

        # Add citation relationships
        for edge_data in citation_data:
            if len(edge_data) == 3:
                # Triple: (citing_id, cited_id, edge_type)
                citing_id, cited_id, edge_type = edge_data
            elif len(edge_data) == 2:
                # Tuple: (citing_id, cited_id), need to infer type
                citing_id, cited_id = edge_data
                if citing_id in self.papers and cited_id in self.papers:
                    edge_type = self._infer_edge_type(citing_id, cited_id)
                else:
                    edge_type = "CITES"
            else:
                logger.warning(f"Invalid citation data format: {edge_data}")
                continue

            if citing_id in self.papers and cited_id in self.papers:
                self.add_citation_edge(citing_id, cited_id, edge_type)

        logger.info(f"Graph construction complete: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def visualize_graph(self, output_path: str = "./output/graph_visualization.html",
                       max_nodes: int = 50,
                       deep_survey_report: Dict = None,
                       research_ideas: Dict = None) -> str:
        """
        Visualize knowledge graph - new interactive layout design

        Args:
            output_path: Output file path
            max_nodes: Maximum number of nodes (avoid overly complex graph)
            deep_survey_report: Deep survey report (Stage 6 results)
            research_ideas: Research ideas report (Stage 7 results)

        Returns:
            Generated visualization file path
        """
        # Save reports to instance variables
        if deep_survey_report:
            self.deep_survey_report = deep_survey_report
        if research_ideas:
            self.research_ideas = research_ideas

        logger.info(f"Generating interactive knowledge graph visualization (max {max_nodes} nodes)...")

        # If too many nodes, select most important ones
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

        # Create time axis layout
        pos = self._create_time_based_layout(subgraph)

        # Generate HTML file with interactive features
        html_content = self._generate_interactive_html_page(subgraph, pos)

        # Save file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Interactive visualization file saved: {output_path}")
        return str(output_path)


    # Graph analysis
    def compute_metrics(self) -> Dict:
        """
        Calculate graph metrics

        Returns:
            Dictionary containing various graph metrics
        """
        logger.info("Calculating graph metrics...")

        metrics = {}

        if self.graph.number_of_nodes() > 0:
            # Basic statistics
            metrics['total_nodes'] = self.graph.number_of_nodes()
            metrics['total_edges'] = self.graph.number_of_edges()
            metrics['density'] = nx.density(self.graph)

            # PageRank (paper importance)
            try:
                pagerank = nx.pagerank(self.graph)
                metrics['top_papers_by_pagerank'] = sorted(
                    pagerank.items(), key=lambda x: x[1], reverse=True
                )[:10]
            except:
                metrics['top_papers_by_pagerank'] = []

            # Degree centrality
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

            # Connected components
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

        logger.info("Graph metrics calculation complete")
        return metrics

    def find_research_clusters(self, min_cluster_size: int = 3) -> List[List[str]]:
        """
        Discover research clusters

        Args:
            min_cluster_size: Minimum cluster size

        Returns:
            Cluster list
        """
        logger.info("Finding research clusters...")

        try:
            # Convert to undirected graph for clustering
            undirected_graph = self.graph.to_undirected()

            # Use Louvain algorithm for community detection
            # Here we use simple connected components as clusters
            clusters = list(nx.connected_components(undirected_graph))

            # Filter small clusters
            significant_clusters = [
                list(cluster) for cluster in clusters
                if len(cluster) >= min_cluster_size
            ]

            logger.info(f"Found {len(significant_clusters)} research clusters")
            return significant_clusters

        except Exception as e:
            logger.error(f"Cluster discovery failed: {e}")
            return []

    def export_graph_data(self, output_path: str) -> None:
        """
        Export graph data to JSON format

        Args:
            output_path: Output file path
        """
        logger.info(f"Exporting graph data to: {output_path}")

        # Calculate metrics
        logger.info("Calculating graph metrics...")
        metrics = self.compute_metrics()
        logger.info("Graph metrics calculation complete")

        # Collect seed node ID list
        seed_ids = [
            node_id for node_id, attrs in self.graph.nodes(data=True)
            if attrs.get('is_seed', False)
        ]

        # Prepare export data
        graph_data = {
            'nodes': [],
            'edges': [],
            'metrics': metrics,
            'metadata': {
                'total_papers': len(self.papers),
                'total_citations': self.graph.number_of_edges(),
                'seed_count': len(seed_ids),
                'seed_ids': seed_ids,  # Add seed node ID list
                'created_at': str(Path().resolve())
            }
        }

        # Export nodes
        for node_id, attrs in self.graph.nodes(data=True):
            node_data = {'id': node_id}
            node_data.update(attrs)
            graph_data['nodes'].append(node_data)

        # Export edges
        for from_node, to_node, attrs in self.graph.edges(data=True):
            edge_data = {
                'from': from_node,
                'to': to_node
            }
            edge_data.update(attrs)
            graph_data['edges'].append(edge_data)

        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)

        logger.info(f"Graph data export complete: {output_path}")
        if seed_ids:
            logger.info(f"  Contains {len(seed_ids)} seed nodes: {seed_ids[:3]}{'...' if len(seed_ids) > 3 else ''}")

    def _get_time_span(self) -> Dict:
        """Get paper time span"""
        years = [paper.get('year', 0) for paper in self.papers.values() if paper.get('year', 0) > 0]
        if years:
            return {
                'earliest': min(years),
                'latest': max(years),
                'span': max(years) - min(years)
            }
        return {'earliest': None, 'latest': None, 'span': 0}

    def _get_research_areas(self) -> List[str]:
        """Get research areas"""
        all_techniques = []
        for paper in self.papers.values():
            if 'ai_analysis' in paper:
                techniques = paper['ai_analysis'].get('key_techniques', [])
                all_techniques.extend(techniques)

        # Count frequency
        from collections import Counter
        technique_counts = Counter(all_techniques)
        return [tech for tech, count in technique_counts.most_common(10)]

    def _find_influential_papers(self, top_k: int = 5) -> List[Dict]:
        """Find most influential papers"""
        papers_with_scores = []

        for paper_id, paper in self.papers.items():
            score = 0
            score += paper.get('cited_by_count', 0) * 0.7  # Total citation count
            score += self.graph.in_degree(paper_id) * 10   # Citation count in graph

            papers_with_scores.append({
                'id': paper_id,
                'title': paper.get('title', 'Unknown'),
                'authors': paper.get('authors', []),
                'year': paper.get('year', 0),
                'influence_score': score,
                'cited_by_count': paper.get('cited_by_count', 0)
            })

        # Sort and return top K
        papers_with_scores.sort(key=lambda x: x['influence_score'], reverse=True)
        return papers_with_scores[:top_k]

    def _analyze_research_trends(self) -> Dict:
        """Analyze research trends"""
        # Count papers by year
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
    # Test code
    kg = CitationGraph()

    # Create test data
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

    test_citations = [('W2', 'W1')]  # W2 cites W1

    # Build graph
    kg.build_citation_network(test_papers, test_citations)

    # Generate visualization
    kg.visualize_graph("./test_graph.html")

    # Generate report
    report = kg.generate_analysis_report()
    print(f"Analysis report: {report['overview']}")