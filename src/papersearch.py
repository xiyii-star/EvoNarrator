"""
Complete 8-Step Literature Retrieval Pipeline with Citation Network Construction

Complete 8-step literature retrieval process and citation network construction

8-Step Pipeline:
---------------
Step 1+2: Seed Retrieval & OpenAlex Mapping (Combined)
    Strategy Optimization: Relaxed Retrieval + Ensure Mapping
    - Phase 1: Use arXiv API with relaxed retrieval conditions, get 3x target number of candidate papers (no year limit)
    - Phase 2: Batch map all candidate papers to OpenAlex
    - Phase 3: Keep only successfully mapped papers, sort by quality (citation count + relevance) to select best seeds
    - Advantage: Ensure all seeds can be used in OpenAlex citation network while maintaining high recall

Step 3: Forward Snowballing
    - Seed -> Who cited the Seed? -> Child nodes
    - Get detailed information of citing papers

Step 4: Backward Snowballing
    - Who was cited by Seed? <- Seed -> Parent nodes/Ancestors
    - Get detailed information of cited papers

Step 5: Lateral Supplementation/Co-citation Mining
    - Among child and parent nodes, who is repeatedly mentioned?
    - Filter high-value papers by co-citation threshold

Step 6 [Optional]: Second-Round Snowballing
    - Perform controlled expansion on first-round papers
    - Control expansion scale

Step 7: Recent Frontiers Supplementation
    - arXiv papers from recent 6-12 months
    - Similarity filtering

Step 8: Citation Closure Construction
    - Build complete network
    - Fill missing citation relationships and connect citations

Date: 2025-12-09
Version: 2.0 (Combined Step 1+2 for better seed quality)
"""

import logging
import time
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import yaml

# Import dependency modules
from openalex_client import OpenAlexClient
from arxiv_seed_retriever import ArxivSeedRetriever
from cross_database_mapper import CrossDatabaseMapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PaperSearchPipeline:
    """
    Complete 8-step literature retrieval process (optimized version: steps 1+2 combined)

    Strategy optimization:
    - Steps 1+2 combined: Relax arXiv retrieval → Batch OpenAlex mapping → Ensure all seeds are available
    - Integrate arXiv seed retrieval, OpenAlex citation expansion, co-citation mining
    - Ensure all seed papers have OpenAlex mapping for citation network expansion
    """

    def __init__(
        self,
        openalex_client: Optional[OpenAlexClient] = None,
        config_path: str = './config/config.yaml',
        llm_client = None
    ):
        """
        Initialize pipeline

        Args:
            openalex_client: OpenAlex client
            config_path: Configuration file path
            llm_client: LLM client (for query generation)
        """
        # Initialize clients
        self.openalex_client = openalex_client or OpenAlexClient()
        self.llm_client = llm_client

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize retrievers
        self._init_retrievers()

        # Data storage structures
        self.papers = {}  # paper_id -> paper_dict
        self.citation_edges = []  # [(source_id, target_id), ...]

        # Result cache for each step (for debugging and statistics)
        self.seed_papers = []  # Step 1: Seed papers
        self.mapped_seeds = []  # Step 2: Successfully mapped seeds
        self.unmapped_seeds = []  # Step 2: Failed to map seeds
        self.forward_papers = []  # Step 3: Forward citation papers
        self.backward_papers = []  # Step 4: Backward citation papers
        self.cocitation_papers = []  # Step 5: Co-citation papers
        self.second_round_papers = []  # Step 6: Second-round expansion papers (total)
        self.second_round_citing = []  # Step 6: Second-round forward citation papers
        self.second_round_ancestor = []  # Step 6: Second-round backward citation papers
        self.recent_papers = []  # Step 7: Recent papers

        # Statistics information
        self.statistics = {
            'seed_papers': 0,
            'arxiv_mapped': 0,
            'arxiv_unmapped': 0,
            'manual_citations_built': 0,
            'first_round_citing': 0,
            'first_round_ancestor': 0,
            'first_round_cocitation': 0,
            'second_round_enabled': False,
            'second_round_citing': 0,
            'second_round_ancestor': 0,
            'recent_papers': 0,
            'total_papers': 0,
            'total_edges': 0
        }

        logger.info("="*70)
        logger.info("Initializing paper search pipeline")
        logger.info("="*70)
        logger.info(f"Configuration parameters:")
        logger.info(f"  - Seed count: {self.config['seed_count']}")
        logger.info(f"  - Citations per seed: {self.config['citations_per_seed']}")
        logger.info(f"  - Co-citation threshold: {self.config['cocitation_threshold']}")
        logger.info(f"  - Second round expansion: {'Enabled' if self.config['enable_second_round'] else 'Disabled'}")
        logger.info(f"  - Recent papers count: {self.config['recent_count']}")
        logger.info("="*70)

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Configuration file does not exist: {config_path}, using default configuration")
                return self._default_config()

            with open(config_file, 'r', encoding='utf-8') as f:
                full_config = yaml.safe_load(f)
                snowball_config = full_config.get('snowball', {})

                config = {
                    # Seed retrieval
                    'seed_count': snowball_config.get('seed_count', 10),
                    'arxiv_years_back': snowball_config.get('arxiv_years_back', 5),

                    # Citation counts
                    'citations_per_seed': snowball_config.get('citations_per_seed', 15),
                    'references_per_seed': snowball_config.get('references_per_seed', 10),

                    # Co-citation
                    'cocitation_threshold': snowball_config.get('cocitation_threshold', 3),
                    'max_cocitation_papers': snowball_config.get('max_cocitation_papers', 20),

                    # Second round expansion
                    'enable_second_round': snowball_config.get('enable_second_round', True),
                    'second_round_limit': snowball_config.get('second_round_limit', 5),
                    'second_round_max_papers': snowball_config.get('second_round_max_papers', 50),

                    # Recent papers
                    'recent_months': snowball_config.get('recent_months', 12),
                    'recent_count': snowball_config.get('recent_count', 10),

                    # Other
                    'use_llm_query': snowball_config.get('use_llm_query', True),
                    'min_citation_count': snowball_config.get('min_citation_count', 5),

                    # LLM semantic expansion
                    'llm_semantic_expansion': snowball_config.get('llm_semantic_expansion', True),
                    'expansion_max_topics': snowball_config.get('expansion_max_topics', 4),
                    'expansion_max_keywords': snowball_config.get('expansion_max_keywords', 8),
                }

                logger.info(f"Successfully loaded configuration file: {config_path}")
                return config

        except Exception as e:
            logger.warning(f"Failed to load configuration: {e}, using default configuration")
            return self._default_config()

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'seed_count': 10,
            'arxiv_years_back': 5,
            'citations_per_seed': 15,
            'references_per_seed': 10,
            'cocitation_threshold': 3,
            'max_cocitation_papers': 20,
            'enable_second_round': True,
            'second_round_limit': 5,
            'second_round_max_papers': 50,
            'recent_months': 12,
            'recent_count': 10,
            'use_llm_query': True,
            'min_citation_count': 5,
            'llm_semantic_expansion': True,
            'expansion_max_topics': 4,
            'expansion_max_keywords': 8,
        }

    def _init_retrievers(self):
        """Initialize retrievers"""
        # arXiv seed retriever
        self.arxiv_retriever = ArxivSeedRetriever(
            max_results_per_query=self.config['seed_count'] * 2,
            years_back=self.config['arxiv_years_back'],
            min_relevance_score=0.5,
            llm_client=self.llm_client,
            use_llm_query_generation=self.config['use_llm_query'],
            enable_semantic_expansion=self.config.get('llm_semantic_expansion', True),
            expansion_max_topics=self.config.get('expansion_max_topics', 4),
            expansion_max_keywords=self.config.get('expansion_max_keywords', 8)
        )

        # Cross-database mapper
        self.cross_mapper = CrossDatabaseMapper(
            client=self.openalex_client,
            min_concept_score=0.3,
            required_concepts=["Computer Science"]
        )

        logger.info("Retrievers initialized successfully")

    def execute_full_pipeline(
        self,
        topic: str,
        keywords: Optional[List[str]] = None,
        categories: Optional[List[str]] = None
    ) -> Dict:
        """
        Execute complete 8-step retrieval pipeline

        Args:
            topic: Research topic
            keywords: Keyword list (optional)
            categories: arXiv category list (optional)

        Returns:
            {
                'papers': {paper_id: paper_dict},
                'citation_edges': [(source_id, target_id), ...],
                'statistics': {...}
            }
        """
        logger.info("\n" + "="*70)
        logger.info(f"Starting 8-step literature retrieval pipeline")
        logger.info(f"Topic: {topic}")
        logger.info("="*70 + "\n")

        start_time = time.time()

        # Step 1: High-quality seed retrieval
        self._step1_seed_retrieval(topic, keywords, categories)

        # Step 2: Cross-database ID mapping
        self._step2_cross_database_mapping()

        # Step 3: Forward snowballing
        self._step3_forward_snowballing()

        # Step 4: Backward snowballing
        self._step4_backward_snowballing()

        # Step 5: Co-citation mining
        self._step5_cocitation_mining()

        # Step 6: Second round expansion (optional)
        if self.config['enable_second_round']:
            self._step6_second_round_snowballing()

        # Step 7: Recent SOTA supplementation
        self._step7_recent_frontiers(topic, keywords, categories)

        # Step 8: Citation closure construction
        self._step8_citation_closure()

        # Update statistics
        self._finalize_statistics()

        elapsed_time = time.time() - start_time
        logger.info("\n" + "="*70)
        logger.info(f"8-step retrieval pipeline completed, time elapsed: {elapsed_time:.2f}s")
        logger.info("="*70)
        self._print_summary()

        return {
            'papers': self.papers,
            'citation_edges': self.citation_edges,
            'statistics': self.statistics
        }

    def _step1_seed_retrieval(
        self,
        topic: str,
        keywords: Optional[List[str]],
        categories: Optional[List[str]]
    ):
        """
        Step 1: High-quality seed retrieval (combined with Step 2: ensure OpenAlex mapping)

        Strategy:
        1. Relax arXiv retrieval conditions, get more candidate papers (expand initial retrieval scope, no year limit)
        2. Map all candidate papers to OpenAlex
        3. Keep only successfully mapped papers as final seeds
        4. Sort by quality (citation count + relevance) to select best seeds
        """
        logger.info("\n" + "="*70)
        logger.info("Step 1+2: Seed Retrieval & OpenAlex Mapping (Combined Process)")
        logger.info("="*70)
        logger.info("Strategy: Relax arXiv retrieval → Batch OpenAlex mapping → Keep successfully mapped papers")

        target_seed_count = self.config['seed_count']

        # Relax retrieval conditions: retrieve more candidate papers (3x target count)
        # Because considering mapping success rate, need more candidates
        candidate_count = target_seed_count * 3

        logger.info(f"\nPhase 1: arXiv Candidate Paper Retrieval (Relaxed Conditions)")
        logger.info(f"  - Target seed count: {target_seed_count}")
        logger.info(f"  - Candidate retrieval count: {candidate_count}")

        try:
            # Temporarily lower relevance threshold to get more candidates
            original_threshold = self.arxiv_retriever.min_relevance_score
            self.arxiv_retriever.min_relevance_score = 0.3  # Relax to 0.3

            # Use arXiv retriever to get candidate papers
            arxiv_candidates = self.arxiv_retriever.retrieve_seed_papers(
                topic=topic,
                keywords=keywords,
                categories=categories,
                max_seeds=candidate_count
            )

            # Restore original threshold
            self.arxiv_retriever.min_relevance_score = original_threshold

            # No longer filter by year, accept all candidate papers
            filtered_candidates = arxiv_candidates

            logger.info(f"  ✓ Retrieved {len(filtered_candidates)} candidate papers from arXiv (all years)")

            if not filtered_candidates:
                logger.warning("  ⚠️ No qualifying arXiv candidate papers found")
                self.seed_papers = []
                self.mapped_seeds = []
                self.unmapped_seeds = []
                return

            # Phase 2: Batch mapping to OpenAlex
            logger.info(f"\nPhase 2: Batch Mapping to OpenAlex")
            logger.info(f"  - Candidate paper count: {len(filtered_candidates)}")

            # Use mapper for ID mapping (disable concept verification, as arXiv stage already filtered)
            mapped_papers, mapping_stats = self.cross_mapper.map_arxiv_to_openalex(
                arxiv_papers=filtered_candidates,
                verify_concepts=False  # arXiv papers already filtered
            )

            logger.info(f"\nMapping results:")
            logger.info(f"  - Successfully mapped: {len(mapped_papers)} papers")
            logger.info(f"  - Failed to map: {mapping_stats['failed']} papers")
            logger.info(f"  - Success rate: {mapping_stats.get('success_rate', 0):.1%}")

            # Phase 3: Sort by quality, select best seeds
            logger.info(f"\nPhase 3: Select High-Quality Seeds")

            # Sort by combined citation count and relevance
            for paper in mapped_papers:
                # Combined score = normalized citation count * 0.6 + relevance score * 0.4
                cited_count = paper.get('cited_by_count', 0)
                relevance = paper.get('relevance_score', 0.5)

                # Simple normalization (log scale)
                normalized_citation = min(1.0, cited_count / 100.0) if cited_count > 0 else 0
                paper['quality_score'] = normalized_citation * 0.6 + relevance * 0.4

            # Sort
            mapped_papers.sort(key=lambda x: x.get('quality_score', 0), reverse=True)

            # Select top N as final seeds
            final_seeds = mapped_papers[:target_seed_count]

            logger.info(f"  - Selected top {len(final_seeds)} papers as final seeds")

            # Update seed lists
            self.seed_papers = filtered_candidates  # Keep original arXiv retrieval results for statistics
            self.mapped_seeds = final_seeds  # Successfully mapped seeds (for subsequent processes)
            self.unmapped_seeds = [
                p for p in filtered_candidates
                if not any(m.get('arxiv_id') == p.get('arxiv_id') for m in mapped_papers)
            ]

            # Add successfully mapped seed papers to papers dict and mark as seed nodes
            for paper in self.mapped_seeds:
                paper_id = paper['id']
                paper['is_seed'] = True  # Add seed node marker
                self.papers[paper_id] = paper

            logger.info(f"\nFinal seed statistics:")
            logger.info(f"  - arXiv candidate count: {len(filtered_candidates)} (all years)")
            logger.info(f"  - OpenAlex mapping successful: {len(mapped_papers)}")
            logger.info(f"  - Final seed count: {len(final_seeds)}")
            logger.info(f"  - Mapping failed (discarded): {len(self.unmapped_seeds)}")

            # Print seed paper information and IDs
            logger.info(f"\nFinal seed papers (Top {min(5, len(final_seeds))}):")
            for i, paper in enumerate(final_seeds[:5], 1):
                logger.info(
                    f"  [{i}] {paper['title'][:60]}... "
                    f"({paper.get('year', 'N/A')}, "
                    f"citations:{paper.get('cited_by_count', 0)}, "
                    f"quality:{paper.get('quality_score', 0):.2f})"
                )
                logger.info(f"      ID: {paper['id']}")
            if len(final_seeds) > 5:
                logger.info(f"  ... and {len(final_seeds) - 5} more")

            # Record all seed node IDs
            seed_ids = [p['id'] for p in final_seeds]
            logger.info(f"\nSeed node ID list: {seed_ids}")

            # If seed count is insufficient, issue warning
            if len(final_seeds) < target_seed_count:
                logger.warning(
                    f"\n⚠️ Warning: Final seed count ({len(final_seeds)}) "
                    f"is less than target count ({target_seed_count}), "
                    f"recommend relaxing retrieval conditions or adjusting year limits"
                )

        except Exception as e:
            logger.error(f"Seed retrieval and mapping failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.seed_papers = []
            self.mapped_seeds = []
            self.unmapped_seeds = []

    def _step2_cross_database_mapping(self):
        """
        Step 2: Cross-database ID mapping (already integrated into Step 1)

        Note: This step has been merged with Step 1, this method is kept as placeholder only
        Actual mapping logic has been completed in _step1_seed_retrieval
        """
        logger.info("\n" + "-"*70)
        logger.info("Step 2: Cross-database ID Mapping (integrated into Step 1)")
        logger.info("-"*70)
        logger.info("✓ Mapping completed in Step 1, skipping")

        # Statistics already updated in Step 1, no additional operations needed here
        pass

    def _handle_unmapped_seeds(self):
        """Handle seed papers that failed to map"""
        manual_count = 0

        for seed in self.unmapped_seeds:
            try:
                # Try to search in OpenAlex
                title = seed.get('title', '')
                if not title:
                    continue

                search_results = self.openalex_client.search_papers(
                    topic=title,
                    max_results=1,
                    sort_by="relevance"
                )

                if search_results:
                    paper = search_results[0]
                    paper_id = paper['id']
                    self.papers[paper_id] = paper
                    manual_count += 1
                    logger.info(f"  Manual search successful: {title[:50]}")

            except Exception as e:
                logger.debug(f"  Manual search failed: {seed.get('title', 'Unknown')[:50]} - {e}")

        self.statistics['manual_citations_built'] = manual_count
        logger.info(f"  - Manual search successful: {manual_count} papers")

    def _step3_forward_snowballing(self):
        """
        Step 3: Forward snowballing
        Find detailed information of papers that cited the seed papers
        """
        logger.info("\n" + "-"*70)
        logger.info("Step 3: Forward Snowballing")
        logger.info("-"*70)
        logger.info("Strategy: Seed -> Who cited the Seed? -> Child nodes")

        if not self.mapped_seeds:
            logger.warning("No available mapped seeds, skipping forward snowballing")
            return

        citing_papers = []
        citations_per_seed = self.config['citations_per_seed']

        logger.info(f"Starting to process {len(self.mapped_seeds)} seed papers, getting up to {citations_per_seed} citations each...")

        for i, seed in enumerate(self.mapped_seeds, 1):
            seed_id = seed['id']
            seed_title = seed.get('title', 'Unknown')

            try:
                # Get papers that cite this seed paper
                citations = self.openalex_client.get_citations(
                    paper_id=seed_id,
                    max_results=citations_per_seed
                )

                # Add citing papers
                new_papers_count = 0
                for citation in citations:
                    citation_id = citation['id']

                    # Add to paper collection
                    if citation_id not in self.papers:
                        self.papers[citation_id] = citation
                        citing_papers.append(citation)
                        new_papers_count += 1

                    # Add citation edge: citation -> seed
                    edge = (citation_id, seed_id)
                    if edge not in self.citation_edges:
                        self.citation_edges.append(edge)

                # Simplified output: only show key information
                logger.info(f"  [{i}/{len(self.mapped_seeds)}] {seed_title[:50]}... → +{new_papers_count} new papers (total {len(citations)} citations)")

            except Exception as e:
                logger.warning(f"  [{i}/{len(self.mapped_seeds)}] Failed to get citations: {seed_title[:40]}... - {e}")

        self.forward_papers = citing_papers
        logger.info(f"\n✅ Forward snowballing completed: added {len(citing_papers)} citing papers")

    def _step4_backward_snowballing(self):
        """
        Step 4: Backward snowballing
        Find which papers were cited by the seed papers
        """
        logger.info("\n" + "-"*70)
        logger.info("Step 4: Backward Snowballing")
        logger.info("-"*70)
        logger.info("Strategy: Who was cited by Seed? <- Seed -> Parent nodes/Ancestors")

        if not self.mapped_seeds:
            logger.warning("No available mapped seeds, skipping backward snowballing")
            return

        referenced_papers = []
        references_per_seed = self.config['references_per_seed']

        logger.info(f"Starting to process {len(self.mapped_seeds)} seed papers, getting up to {references_per_seed} references each...")

        for i, seed in enumerate(self.mapped_seeds, 1):
            seed_id = seed['id']
            seed_title = seed.get('title', 'Unknown')

            try:
                # Get papers cited by this seed paper
                references = self.openalex_client.get_references(
                    paper_id=seed_id,
                    max_results=references_per_seed
                )

                # Add cited papers
                new_papers_count = 0
                for reference in references:
                    reference_id = reference['id']

                    # Add to paper collection
                    if reference_id not in self.papers:
                        self.papers[reference_id] = reference
                        referenced_papers.append(reference)
                        new_papers_count += 1

                    # Add citation edge: seed -> reference
                    edge = (seed_id, reference_id)
                    if edge not in self.citation_edges:
                        self.citation_edges.append(edge)

                # Simplified output: only show key information
                logger.info(f"  [{i}/{len(self.mapped_seeds)}] {seed_title[:50]}... → +{new_papers_count} new papers (total {len(references)} references)")

            except Exception as e:
                logger.warning(f"  [{i}/{len(self.mapped_seeds)}] Failed to get references: {seed_title[:40]}... - {e}")

        self.backward_papers = referenced_papers
        logger.info(f"\n✅ Backward snowballing completed: added {len(referenced_papers)} ancestor papers")

    def _step5_cocitation_mining(self):
        """
        Step 5: Co-citation mining
        Find high-value papers that are repeatedly mentioned in child and parent nodes
        """
        logger.info("\n" + "-"*70)
        logger.info("Step 5: Co-citation Mining")
        logger.info("-"*70)
        logger.info("Strategy: Count co-cited papers")

        # Merge first-round seed papers + citation papers
        first_round_papers = self.forward_papers + self.backward_papers

        if not first_round_papers:
            logger.warning("No available first-round papers, skipping co-citation mining")
            return

        # Count how many times papers are cited
        cocitation_counter = Counter()

        for paper in first_round_papers:
            paper_id = paper['id']

            try:
                # Get references of this paper
                references = self.openalex_client.get_references(
                    paper_id=paper_id,
                    max_results=20
                )

                # Count citation frequency of references
                for ref in references:
                    ref_id = ref['id']
                    # Only count papers not in current paper collection
                    if ref_id not in self.papers:
                        cocitation_counter[ref_id] += 1

            except Exception as e:
                logger.debug(f"  Failed to get references: {paper['title'][:50]} - {e}")

        # Filter papers by co-citation count
        threshold = self.config['cocitation_threshold']
        max_papers = self.config['max_cocitation_papers']

        cocited_paper_ids = [
            paper_id for paper_id, count in cocitation_counter.most_common()
            if count >= threshold
        ][:max_papers]

        logger.info(f"Found {len(cocited_paper_ids)} co-cited papers (threshold≥{threshold})")

        # Get detailed information of co-cited papers
        cocitation_papers = []
        for paper_id in cocited_paper_ids:
            try:
                paper = self.openalex_client.get_paper_by_id(paper_id)
                if paper:
                    self.papers[paper_id] = paper
                    cocitation_papers.append(paper)

                    # Add citation edges (from papers citing co-cited papers to co-cited papers)
                    for citing_paper in first_round_papers:
                        citing_id = citing_paper['id']
                        try:
                            refs = self.openalex_client.get_references(citing_id, max_results=50)
                            if any(r['id'] == paper_id for r in refs):
                                edge = (citing_id, paper_id)
                                if edge not in self.citation_edges:
                                    self.citation_edges.append(edge)
                        except:
                            pass

            except Exception as e:
                logger.debug(f"  Failed to get co-cited paper: {paper_id} - {e}")

        self.cocitation_papers = cocitation_papers
        logger.info(f"Co-citation mining retrieved {len(cocitation_papers)} co-cited papers")

    def _step6_second_round_snowballing(self):
        """
        Step 6: Second-round snowballing (optional)
        Perform controlled expansion on first-round papers
        """
        logger.info("\n" + "-"*70)
        logger.info("Step 6: Second-Round Snowballing")
        logger.info("-"*70)
        logger.info("Strategy: Perform controlled expansion on first-round papers")

        self.statistics['second_round_enabled'] = True

        # Select high-quality first-round papers for expansion
        first_round_papers = self.forward_papers + self.backward_papers + self.cocitation_papers

        # Select top papers with highest citation counts
        sorted_papers = sorted(
            first_round_papers,
            key=lambda p: p.get('cited_by_count', 0),
            reverse=True
        )

        max_expand = self.config['second_round_max_papers']
        papers_to_expand = sorted_papers[:max_expand]

        logger.info(f"Selected {len(papers_to_expand)} highly-cited papers for second-round expansion")

        second_round_citing = []
        second_round_ancestor = []
        limit_per_paper = self.config['second_round_limit']

        for i, paper in enumerate(papers_to_expand, 1):
            paper_id = paper['id']
            paper_title = paper.get('title', 'Unknown')

            try:
                # Forward snowballing (get papers citing this paper)
                citations = self.openalex_client.get_citations(
                    paper_id=paper_id,
                    max_results=limit_per_paper
                )

                for citation in citations:
                    citation_id = citation['id']
                    if citation_id not in self.papers:
                        self.papers[citation_id] = citation
                        second_round_citing.append(citation)

                    edge = (citation_id, paper_id)
                    if edge not in self.citation_edges:
                        self.citation_edges.append(edge)

                # Backward snowballing (get papers cited by this paper)
                references = self.openalex_client.get_references(
                    paper_id=paper_id,
                    max_results=limit_per_paper
                )

                for reference in references:
                    reference_id = reference['id']
                    if reference_id not in self.papers:
                        self.papers[reference_id] = reference
                        second_round_ancestor.append(reference)

                    edge = (paper_id, reference_id)
                    if edge not in self.citation_edges:
                        self.citation_edges.append(edge)

                if i <= 5 or i % 10 == 0:
                    logger.info(f"  [{i}/{len(papers_to_expand)}] {paper_title[:50]}... -> {len(citations)} citations + {len(references)} references")

            except Exception as e:
                logger.debug(f"  Second-round expansion failed: {paper_title[:50]} - {e}")

        # Save second-round results
        self.second_round_citing = second_round_citing
        self.second_round_ancestor = second_round_ancestor
        self.second_round_papers = second_round_citing + second_round_ancestor

        logger.info(f"Second-round snowballing results:")
        logger.info(f"  - Forward child nodes: {len(second_round_citing)} papers")
        logger.info(f"  - Backward ancestors: {len(second_round_ancestor)} papers")

    def _step7_recent_frontiers(
        self,
        topic: str,
        keywords: Optional[List[str]],
        categories: Optional[List[str]]
    ):
        """
        Step 7: Recent SOTA supplementation
        Get recent papers from arXiv from the last 6-12 months
        """
        logger.info("\n" + "-"*70)
        logger.info("Step 7: Recent Frontiers Supplementation")
        logger.info("-"*70)
        logger.info(f"Strategy: arXiv papers from recent {self.config['recent_months']} months")

        try:
            # Use arXiv retriever to get recent papers
            try:
                import arxiv
            except ImportError:
                logger.warning("arxiv package not installed, skipping recent papers supplementation")
                self.recent_papers = []
                return

            # Calculate date range
            months_back = self.config['recent_months']
            start_date = datetime.now() - timedelta(days=30 * months_back)

            logger.info(f"  - Time range: >= {start_date.strftime('%Y-%m-%d')}")
            logger.info(f"  - Target count: {self.config['recent_count']} papers")

            # Temporarily lower relevance threshold to get more recent papers
            original_threshold = self.arxiv_retriever.min_relevance_score
            self.arxiv_retriever.min_relevance_score = 0.25  # Further relax to 0.25 (recent papers may have lower relevance scores)

            recent_papers_raw = self.arxiv_retriever.retrieve_seed_papers(
                topic=topic,
                keywords=keywords,
                categories=categories,
                max_seeds=self.config['recent_count'] * 3,  # Get more
                sort_by=arxiv.SortCriterion.SubmittedDate  # Sort by submission date
            )

            # Restore original threshold
            self.arxiv_retriever.min_relevance_score = original_threshold

            logger.info(f"  → Retrieved {len(recent_papers_raw)} candidate papers from arXiv")

            # Filter to keep only recent papers
            recent_filtered = []
            for paper in recent_papers_raw:
                # Use published_date field for precise time filtering
                published_date = paper.get('published_date')

                if published_date:
                    # If published_date is a datetime object
                    if isinstance(published_date, datetime):
                        pub_date = published_date
                    else:
                        # Try to parse string
                        try:
                            pub_date = datetime.fromisoformat(str(published_date).replace('Z', '+00:00'))
                        except:
                            # Fall back to year comparison
                            pub_year = paper.get('year', 0)
                            if pub_year >= start_date.year:
                                recent_filtered.append(paper)
                            continue

                    # Ensure pub_date is naive datetime (remove timezone info for comparison with start_date)
                    if pub_date.tzinfo is not None:
                        pub_date = pub_date.replace(tzinfo=None)

                    # Compare complete date
                    if pub_date >= start_date:
                        recent_filtered.append(paper)
                        logger.debug(f"  ✓ Kept: {paper['title'][:50]}... ({pub_date.strftime('%Y-%m-%d')})")
                    else:
                        logger.debug(f"  × Filtered: {paper['title'][:50]}... ({pub_date.strftime('%Y-%m-%d')}, earlier than {start_date.strftime('%Y-%m-%d')})")
                else:
                    # No date information, use year as fallback
                    pub_year = paper.get('year', 0)
                    if pub_year >= start_date.year:
                        recent_filtered.append(paper)
                        logger.debug(f"  ✓ Kept: {paper['title'][:50]}... (year:{pub_year})")

            recent_filtered = recent_filtered[:self.config['recent_count']]

            logger.info(f"  → After filtering, kept {len(recent_filtered)} papers from recent {months_back} months")

            # Map to OpenAlex
            if recent_filtered:
                logger.info(f"\nStarting mapping to OpenAlex...")
                mapped_recent, _ = self.cross_mapper.map_arxiv_to_openalex(
                    arxiv_papers=recent_filtered,
                    verify_concepts=False
                )

                logger.info(f"  → Successfully mapped {len(mapped_recent)} papers")

                # Add to paper collection
                for paper in mapped_recent:
                    paper_id = paper['id']
                    if paper_id not in self.papers:
                        self.papers[paper_id] = paper
                        self.recent_papers.append(paper)

                        # Try to connect citation relationships
                        self._connect_recent_paper(paper)

            logger.info(f"\n✅ Recent papers supplementation: added {len(self.recent_papers)} recent papers")

        except Exception as e:
            logger.error(f"Recent papers supplementation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.recent_papers = []

    def _connect_recent_paper(self, paper: Dict):
        """Connect recent papers to citation network"""
        paper_id = paper['id']

        try:
            # Get references of this paper
            references = self.openalex_client.get_references(
                paper_id=paper_id,
                max_results=20
            )

            # If references are in our paper collection, add citation edges
            for ref in references:
                ref_id = ref['id']
                if ref_id in self.papers:
                    edge = (paper_id, ref_id)
                    if edge not in self.citation_edges:
                        self.citation_edges.append(edge)

        except Exception as e:
            logger.debug(f"  Failed to connect recent paper: {paper['title'][:50]} - {e}")

    def _step8_citation_closure(self):
        """
        Step 8: Citation closure construction
        Fill missing citation relationships between papers to build complete network
        """
        logger.info("\n" + "-"*70)
        logger.info("Step 8: Citation Closure Construction")
        logger.info("-"*70)
        logger.info("Strategy: Complete citation relationships between papers")

        initial_edges = len(self.citation_edges)
        paper_ids = list(self.papers.keys())

        # Optimization: use set to speed up lookup
        paper_ids_set = set(paper_ids)
        citation_edges_set = set(self.citation_edges)

        logger.info(f"  Current paper count: {len(paper_ids)}")
        logger.info(f"  Current citation edge count: {initial_edges}")

        # Calculate actual check count
        max_check = min(50, len(paper_ids))
        logger.info(f"  Will check citation relationships of top {max_check} papers\n")

        # Check for missing citation relationships between papers
        new_edges = 0
        checked_papers = 0
        failed_papers = 0
        start_time = time.time()

        for i, source_id in enumerate(paper_ids[:max_check]):
            try:
                # Get references of this paper
                references = self.openalex_client.get_references(
                    paper_id=source_id,
                    max_results=50
                )

                # Batch check citation relationships
                for ref in references:
                    ref_id = ref['id']
                    if ref_id in paper_ids_set:
                        edge = (source_id, ref_id)
                        if edge not in citation_edges_set:
                            self.citation_edges.append(edge)
                            citation_edges_set.add(edge)
                            new_edges += 1

                checked_papers += 1

                # Optimize output frequency: output every 20% or every 10 papers
                if checked_papers % max(1, max_check // 5) == 0 or checked_papers % 10 == 0:
                    progress = (checked_papers / max_check) * 100
                    elapsed = time.time() - start_time
                    rate = checked_papers / elapsed if elapsed > 0 else 0
                    eta = (max_check - checked_papers) / rate if rate > 0 else 0

                    logger.info(
                        f"  Progress: [{checked_papers}/{max_check}] {progress:.0f}% | "
                        f"New edges: {new_edges} | "
                        f"Failed: {failed_papers} | "
                        f"Speed: {rate:.1f} papers/s | "
                        f"ETA: {eta:.0f}s"
                    )

            except Exception as e:
                failed_papers += 1
                logger.debug(f"  Skipped paper {source_id[:20]}... : {str(e)[:50]}")

        # Final statistics
        elapsed_total = time.time() - start_time
        logger.info(f"\n✅ Citation closure construction completed (time elapsed {elapsed_total:.1f}s):")
        logger.info(f"  Checked papers: {checked_papers}/{max_check}")
        logger.info(f"  Failed papers: {failed_papers}")
        logger.info(f"  Initial citation edges: {initial_edges}")
        logger.info(f"  New citation edges: {new_edges}")
        logger.info(f"  Final citation edges: {len(self.citation_edges)}")

        if new_edges > 0:
            logger.info(f"  Growth rate: +{(new_edges/initial_edges*100):.1f}%")

        # Calculate network density
        if len(paper_ids) > 1:
            max_possible_edges = len(paper_ids) * (len(paper_ids) - 1)
            density = len(self.citation_edges) / max_possible_edges * 100
            logger.info(f"  Network density: {density:.2f}%")

    def _finalize_statistics(self):
        """Update final statistics"""
        self.statistics.update({
            'seed_papers': len(self.seed_papers),
            'arxiv_mapped': len(self.mapped_seeds),
            'arxiv_unmapped': len(self.unmapped_seeds),
            'first_round_citing': len(self.forward_papers),
            'first_round_ancestor': len(self.backward_papers),
            'first_round_cocitation': len(self.cocitation_papers),
            # Fix: use separately cached lists
            'second_round_citing': len(self.second_round_citing),
            'second_round_ancestor': len(self.second_round_ancestor),
            'recent_papers': len(self.recent_papers),
            'total_papers': len(self.papers),
            'total_edges': len(self.citation_edges),
            # Add seed node ID list
            'seed_ids': [p['id'] for p in self.mapped_seeds]
        })

    def _print_summary(self):
        """Print final statistics summary"""
        stats = self.statistics

        logger.info("\n" + "="*70)
        logger.info("8-Step Retrieval Pipeline Statistics Summary")
        logger.info("="*70)
        logger.info(f"Seed Papers")
        logger.info(f"  - Total seeds: {stats['seed_papers']}")
        logger.info(f"  - arXiv mapping successful: {stats['arxiv_mapped']}")
        logger.info(f"  - arXiv mapping failed: {stats['arxiv_unmapped']}")
        if stats.get('seed_ids'):
            logger.info(f"  - Seed node IDs: {stats['seed_ids'][:3]}{'...' if len(stats['seed_ids']) > 3 else ''}")
        if stats['manual_citations_built'] > 0:
            logger.info(f"  - Manual search supplementation: {stats['manual_citations_built']}")

        logger.info(f"First-Round Snowballing")
        logger.info(f"  - Forward child nodes: {stats['first_round_citing']}")
        logger.info(f"  - Backward ancestors: {stats['first_round_ancestor']}")
        logger.info(f"  - Co-citation papers: {stats['first_round_cocitation']}")

        if stats['second_round_enabled']:
            logger.info(f"Second-Round Snowballing")
            logger.info(f"  - Forward child nodes: {stats['second_round_citing']}")
            logger.info(f"  - Backward ancestors: {stats['second_round_ancestor']}")

        logger.info(f"Recent SOTA")
        logger.info(f"  - Recent papers: {stats['recent_papers']}")

        logger.info(f"Final Results")
        logger.info(f"  - Total papers: {stats['total_papers']}")
        logger.info(f"  - Citation relationships: {stats['total_edges']}")

        if stats['total_papers'] > 0:
            avg_degree = stats['total_edges'] / stats['total_papers']
            logger.info(f"  - Average connectivity: {avg_degree:.2f}")

        logger.info("="*70)

    def get_statistics(self) -> Dict:
        """Return statistics information"""
        return self.statistics.copy()

    def get_papers(self) -> Dict[str, Dict]:
        """Return all papers"""
        return self.papers.copy()

    def get_citation_edges(self) -> List[Tuple[str, str]]:
        """Return all citation relationships"""
        return self.citation_edges.copy()


# ============================================================================
# Main Function
# ============================================================================

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = PaperSearchPipeline(
        config_path='./config/config.yaml'
    )

    # Execute complete 8-step retrieval
    result = pipeline.execute_full_pipeline(
        topic="Natural Language Processing",
        keywords=["transformer", "attention", "BERT", "GPT"],
        categories=["cs.CL", "cs.AI"]
    )

    # Output results
    print("\n" + "="*70)
    print("Final Retrieval Results:")
    print("="*70)
    print(f"Total papers: {len(result['papers'])}")
    print(f"Citation relationships: {len(result['citation_edges'])}")
    print(f"Average connectivity: {len(result['citation_edges']) / len(result['papers']):.2f}")
    print("="*70)
