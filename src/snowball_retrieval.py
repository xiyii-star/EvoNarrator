"""
Advanced Paper Retrieval Module - Multi-stage Retrieval Strategy Based on Snowball Method

Implements optimized six-step retrieval process (based on arXiv seeds + OpenAlex expansion):
1. High-Quality Seed Retrieval - Using arXiv API + Categories filtering
2. ID Mapping - arXiv -> OpenAlex, strict Concept verification
3. Forward Snowballing - Seed -> Who cited Seed? -> Get child nodes
4. Backward Snowballing - Who was cited by Seed? <- Seed -> Get parent nodes/ancestors
5. Lateral Supplement/Co-citation Mining - Among child and parent nodes, who is repeatedly mentioned but not yet in the library?
6. Add Recent Frontiers (Add Recent SOTA) - Latest arXiv papers (6-12 months) + similarity filtering
7. Closure Construction - Establish connections
"""

import logging
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import yaml
from openalex_client import OpenAlexClient
from arxiv_seed_retriever import ArxivSeedRetriever
from cross_database_mapper import CrossDatabaseMapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SnowballRetrieval:
    """
    Snowball Paper Retrieval System
    Multi-stage paper discovery and relationship construction based on citation relationships
    """

    def __init__(
        self,
        client: Optional[OpenAlexClient] = None,
        seed_count: Optional[int] = None,
        citations_per_seed: Optional[int] = None,
        recent_count: Optional[int] = None,
        seed_keywords: Optional[List[str]] = None,
        enable_second_round: Optional[bool] = None,
        second_round_limit: Optional[int] = None,
        use_arxiv_seeds: Optional[bool] = None,
        arxiv_years_back: Optional[int] = None,
        llm_client = None,
        config_path: str = './config/config.yaml'
    ):
        """
        Initialize snowball retrieval system

        Priority: passed parameters > config.yaml settings > default values

        Args:
            client: OpenAlex API client
            seed_count: Number of foundational seed papers (default: 5)
            citations_per_seed: Number of citation papers to select per seed paper (default: 8)
            recent_count: Number of recent papers (default: 10)
            seed_keywords: List of seed keywords for relevance filtering (default: [])
            enable_second_round: Whether to enable second round snowballing (default: True)
            second_round_limit: Expansion limit per paper in second round (default: 3)
            use_arxiv_seeds: Whether to use arXiv seed retrieval (default: True)
            arxiv_years_back: Number of years to look back for arXiv seeds (default: 5)
            llm_client: LLM client (for intelligent query generation)
            config_path: Configuration file path (default: './config/config.yaml')
        """
        # Load configuration file
        snowball_config = self._load_config(config_path)

        # Initialize clients
        self.client = client or OpenAlexClient()
        self.llm_client = llm_client

        # Parameter priority: passed parameters > config.yaml > default values
        self.seed_count = seed_count if seed_count is not None else snowball_config.get('seed_count', 5)
        self.citations_per_seed = citations_per_seed if citations_per_seed is not None else snowball_config.get('citations_per_seed', 8)
        self.recent_count = recent_count if recent_count is not None else snowball_config.get('recent_count', 10)
        self.seed_keywords = seed_keywords if seed_keywords is not None else snowball_config.get('seed_keywords', [])
        self.enable_second_round = enable_second_round if enable_second_round is not None else snowball_config.get('enable_second_round', True)
        self.second_round_limit = second_round_limit if second_round_limit is not None else snowball_config.get('second_round_limit', 3)

        # New: arXiv seed retrieval parameters
        self.use_arxiv_seeds = use_arxiv_seeds if use_arxiv_seeds is not None else snowball_config.get('use_arxiv_seeds', True)
        self.arxiv_years_back = arxiv_years_back if arxiv_years_back is not None else snowball_config.get('arxiv_years_back', 5)

        # Initialize arXiv retriever and cross-database mapper (if enabled)
        if self.use_arxiv_seeds:
            self.arxiv_retriever = ArxivSeedRetriever(
                max_results_per_query=self.seed_count * 2,  # Retrieve more, may decrease after mapping
                years_back=self.arxiv_years_back,
                min_relevance_score=0.6,
                llm_client=self.llm_client,  # Pass LLM client
                use_llm_query_generation=True  # Enable LLM query generation
            )
            self.cross_mapper = CrossDatabaseMapper(
                client=self.client,
                min_concept_score=0.3,
                required_concepts=["Computer Science"]
            )
        else:
            self.arxiv_retriever = None
            self.cross_mapper = None

        # Store retrieved papers
        self.seed_papers: List[Dict] = []
        self.citing_papers: List[Dict] = []  # Child nodes: papers citing seeds
        self.ancestor_papers: List[Dict] = []  # Parent nodes: papers cited by seeds
        self.cocitation_papers: List[Dict] = []  # Co-citation papers: repeatedly mentioned papers
        self.recent_papers: List[Dict] = []
        self.all_papers: Dict[str, Dict] = {}  # paper_id -> paper_data

        # Store citation relationships
        self.citation_edges: Set[Tuple[str, str]] = set()  # (citing_id, cited_id)

        # Second round statistics (for report generation)
        self.first_round_citing_count: int = 0
        self.first_round_ancestor_count: int = 0
        self.first_round_cocitation_count: int = 0
        self.second_round_citing_count: int = 0
        self.second_round_ancestor_count: int = 0
        self.second_round_cocitation_count: int = 0

        logger.info("Snowball retrieval system initialization complete")
        logger.info(f"  Configuration source: {config_path if Path(config_path).exists() else 'default values'}")
        logger.info(f"  Seed retrieval mode: {'arXiv priority' if self.use_arxiv_seeds else 'OpenAlex direct search'}")
        logger.info(f"  seed_count={self.seed_count}, citations_per_seed={self.citations_per_seed}")
        logger.info(f"  recent_count={self.recent_count}, enable_second_round={self.enable_second_round}")
        if self.enable_second_round:
            logger.info(f"  second_round_limit={self.second_round_limit}")
        if self.use_arxiv_seeds:
            logger.info(f"  arxiv_years_back={self.arxiv_years_back}")

    def _load_config(self, config_path: str) -> Dict:
        """
        Load snowball retrieval configuration from config file

        Args:
            config_path: Configuration file path

        Returns:
            Snowball configuration dictionary (snowball section)
        """
        config_file = Path(config_path)

        if not config_file.exists():
            logger.debug(f"Configuration file does not exist: {config_path}, will use default configuration")
            return {}

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                full_config = yaml.safe_load(f)

            snowball_config = full_config.get('snowball', {}) if full_config else {}
            logger.debug(f"Successfully loaded snowball configuration: {config_path}")
            return snowball_config

        except Exception as e:
            logger.warning(f"Failed to load configuration file: {e}, will use default configuration")
            return {}

    def execute_full_pipeline(
        self,
        topic: str,
        content_keyword: str,
        seed_year_threshold: int = 2023
    ) -> Dict:
        """
        Execute complete six-step retrieval pipeline

        Args:
            topic: Topic keyword
            content_keyword: Content keyword
            seed_year_threshold: Year threshold for seed papers (less than this year)

        Returns:
            Dictionary containing all papers and citation relationships
        """
        logger.info(f"Starting complete retrieval pipeline: topic='{topic}', content='{content_keyword}'")

        # Step 1: Foundational seeds
        logger.info("\n" + "=" * 60)
        logger.info("Step 1: Foundational Seeds (Seed Papers)")
        logger.info("=" * 60)
        self.seed_papers = self._select_seed_papers(
            topic=topic,
            content_keyword=content_keyword,
            year_threshold=seed_year_threshold
        )

        # Step 2: Forward snowballing - find child nodes
        logger.info("\n" + "=" * 60)
        logger.info("Step 2: Forward Snowballing")
        logger.info("Seed -> Who cited Seed? -> Get child nodes")
        logger.info("=" * 60)
        self.citing_papers = self._forward_snowballing(self.seed_papers)

        # Step 3: Backward snowballing - find parent nodes/ancestors
        logger.info("\n" + "=" * 60)
        logger.info("Step 3: Backward Snowballing")
        logger.info("Who was cited by Seed? <- Seed -> Get parent nodes/ancestors")
        logger.info("=" * 60)
        self.ancestor_papers = self._backward_snowballing(self.seed_papers)

        # Step 4: Lateral supplement/co-citation mining
        logger.info("\n" + "=" * 60)
        logger.info("Step 4: Lateral Supplement/Co-citation Mining")
        logger.info("Among child and parent nodes, who is repeatedly mentioned but not yet in the library?")
        logger.info("=" * 60)
        self.cocitation_papers = self._cocitation_mining(
            self.citing_papers,
            self.ancestor_papers
        )

        # Record first round statistics
        self.first_round_citing_count = len(self.citing_papers)
        self.first_round_ancestor_count = len(self.ancestor_papers)
        self.first_round_cocitation_count = len(self.cocitation_papers)

        # Second round snowballing (if enabled)
        if self.enable_second_round:
            logger.info("\n" + "=" * 80)
            logger.info("Starting second round snowballing expansion")
            logger.info("=" * 80)
            self._execute_second_round_snowballing()

        # Step 5: Add recent SOTA
        logger.info("\n" + "=" * 60)
        logger.info("Step 5: Add Recent SOTA (Recent Frontiers)")
        logger.info("=" * 60)
        self.recent_papers = self._add_recent_frontiers(
            topic=topic,
            content_keyword=content_keyword,
            year_threshold=seed_year_threshold
        )

        # Step 6: Build closure
        logger.info("\n" + "=" * 60)
        logger.info("Step 6: Build Citation Closure (Closure Construction)")
        logger.info("=" * 60)
        self._build_closure()

        # Generate statistics report
        result = self._generate_report()
        logger.info("\nRetrieval pipeline complete!")
        return result

    def _select_seed_papers(
        self,
        topic: str,
        content_keyword: str,
        year_threshold: int
    ) -> List[Dict]:
        """
        Step 1: Select foundational seed papers (optimized version)

        Strategy:
        - If arXiv enabled: Use arXiv Categories + keywords -> Map to OpenAlex -> Concept verification
        - If not enabled: Direct OpenAlex search (traditional mode)

        Args:
            topic: Topic keyword
            content_keyword: Content keyword
            year_threshold: Year threshold (will be overridden in arXiv mode)

        Returns:
            List of seed papers
        """
        if self.use_arxiv_seeds:
            logger.info("Using arXiv priority seed retrieval strategy")
            return self._select_seeds_from_arxiv(topic, content_keyword)
        else:
            logger.info("Using OpenAlex direct search strategy")
            return self._select_seeds_from_openalex(topic, content_keyword, year_threshold)

    def _select_seeds_from_arxiv(
        self,
        topic: str,
        content_keyword: str
    ) -> List[Dict]:
        """
        Retrieve seed papers from arXiv and map to OpenAlex

        Process:
        1. arXiv retrieval (using Categories + keywords)
        2. Cross-database mapping (arXiv -> OpenAlex)
        3. Concept filtering (ensure CS/AI domain)

        Args:
            topic: Topic
            content_keyword: Content keyword

        Returns:
            Mapped seed paper list (OpenAlex format)
        """
        logger.info("Step 1a: Retrieve high-quality seeds from arXiv")
        logger.info(f"  Topic: '{topic}', Keyword: '{content_keyword}'")
        logger.info(f"  Years back: {self.arxiv_years_back} years")

        # 1. Retrieve arXiv papers
        keywords = [content_keyword] + self.seed_keywords if self.seed_keywords else [content_keyword]
        arxiv_papers = self.arxiv_retriever.retrieve_seed_papers(
            topic=topic,
            keywords=keywords,
            max_seeds=self.seed_count * 2  # Retrieve more, will decrease after mapping
        )

        logger.info(f"  Retrieved {len(arxiv_papers)} candidate papers from arXiv")

        # 2. Map to OpenAlex
        logger.info("\nStep 1b: Cross-database mapping (arXiv -> OpenAlex)")
        mapped_papers, stats = self.cross_mapper.map_arxiv_to_openalex(
            arxiv_papers,
            verify_concepts=False  # Already filtered by category in arXiv stage, no need to verify concepts again
        )

        logger.info(f"  Successfully mapped {len(mapped_papers)} papers")
        if stats.get('filtered_by_concept', 0) > 0:
            logger.info(f"  Concept filtering: {stats.get('filtered_by_concept', 0)} papers (disabled)")

        # 3. Store to all_papers
        seeds = []
        for paper in mapped_papers[:self.seed_count]:
            self.all_papers[paper['id']] = paper
            seeds.append(paper)
            logger.info(
                f"  [{paper['year']}] {paper['title'][:60]}... "
                f"(citations: {paper['cited_by_count']}, arXiv: {paper.get('arxiv_id', 'N/A')})"
            )

        logger.info(f"\nSelected {len(seeds)} high-quality seed papers (arXiv verified)")
        return seeds

    def _select_seeds_from_openalex(
        self,
        topic: str,
        content_keyword: str,
        year_threshold: int
    ) -> List[Dict]:
        """
        Retrieve seed papers directly from OpenAlex (traditional mode)

        Args:
            topic: Topic keyword
            content_keyword: Content keyword
            year_threshold: Year threshold

        Returns:
            List of seed papers
        """
        query = f"{topic} {content_keyword}"
        logger.info(f"Search query: '{query}'")
        logger.info(f"Filter criteria: publication_year < {year_threshold}, sorted by citations")

        params = {
            'search': query,
            'per-page': self.seed_count,
            'sort': 'cited_by_count:desc',
            'filter': f'publication_year:<{year_threshold},cited_by_count:>50'
        }

        try:
            data = self.client._make_request('works', params)
            results = data.get('results', [])

            seeds = []
            for result in results[:self.seed_count]:
                paper = self.client._parse_paper(result)
                seeds.append(paper)
                self.all_papers[paper['id']] = paper
                logger.info(
                    f"  [{paper['year']}] {paper['title'][:60]}... "
                    f"(citations: {paper['cited_by_count']})"
                )

            logger.info(f"Found {len(seeds)} foundational papers")
            return seeds

        except Exception as e:
            logger.error(f"Failed to select foundational papers: {e}")
            return []

    def _deduplicate_and_log(
        self,
        new_papers: List[Dict],
        existing_dict: Dict[str, Dict],
        paper_type: str
    ) -> Tuple[List[Dict], int, int]:
        """
        Deduplicate and log statistics

        Args:
            new_papers: Newly retrieved paper list
            existing_dict: Existing paper dictionary {paper_id: paper_data}
            paper_type: Paper type description (for logging)

        Returns:
            (Deduplicated new paper list, original count, duplicate count)
        """
        original_count = len(new_papers)
        duplicates = 0
        deduplicated = []

        for paper in new_papers:
            paper_id = paper['id']
            if paper_id not in existing_dict:
                deduplicated.append(paper)
                existing_dict[paper_id] = paper
            else:
                duplicates += 1

        final_count = len(deduplicated)

        logger.info(f"{paper_type} deduplication statistics:")
        logger.info(f"   Original count: {original_count} papers")
        logger.info(f"   Duplicates detected: {duplicates} papers")
        logger.info(f"   After deduplication: {final_count} papers")

        return deduplicated, original_count, duplicates

    def _forward_snowballing(
        self,
        seed_papers: List[Dict],
        max_per_paper: Optional[int] = None
    ) -> List[Dict]:
        """
        Step 2: Forward snowballing - find successors
        Find out who cited these foundational papers

        Args:
            seed_papers: List of seed papers
            max_per_paper: Maximum expansion count per paper (None means use default value)

        Returns:
            List of citing papers
        """
        if max_per_paper is None:
            max_per_paper = self.citations_per_seed

        citing_papers_list = []  # Collect all papers

        for seed in seed_papers:
            seed_id = seed['id']
            seed_year = seed['year']

            logger.info(f"\nProcessing seed paper: {seed['title'][:50]}...")
            logger.info(f"  Seed ID: {seed_id}, Year: {seed_year}")

            # Get all papers citing this paper
            citing = self._get_filtered_citations(
                work_id=seed_id,
                min_year=seed_year,
                keywords=self.seed_keywords,
                max_results=max_per_paper
            )

            logger.info(f"  Found {len(citing)} related citing papers")

            for paper in citing:
                citing_papers_list.append(paper)
                # Record citation relationship
                self.citation_edges.add((paper['id'], seed_id))

        # Unified deduplication
        result, _, _ = self._deduplicate_and_log(
            citing_papers_list,
            self.all_papers,
            "Forward snowballing"
        )

        logger.info(f"\nForward snowballing complete, collected {len(result)} successor papers (after deduplication)")
        return result

    def _backward_snowballing(
        self,
        seed_papers: List[Dict],
        max_per_paper: Optional[int] = None
    ) -> List[Dict]:
        """
        Step 3: Backward snowballing - find parent nodes/ancestors
        Find out who these foundational papers cited (their references)

        Args:
            seed_papers: List of seed papers
            max_per_paper: Maximum expansion count per paper (None means use default value)

        Returns:
            List of ancestor papers
        """
        if max_per_paper is None:
            max_per_paper = self.citations_per_seed

        ancestor_papers_list = []  # Collect all papers

        for seed in seed_papers:
            seed_id = seed['id']
            logger.info(f"\nProcessing seed paper: {seed['title'][:50]}...")
            logger.info(f"  Seed ID: {seed_id}")

            # Get references cited by this paper
            references = self._get_references(
                work_id=seed_id,
                max_results=max_per_paper
            )

            logger.info(f"  Found {len(references)} references (parent nodes)")

            for ref in references:
                ancestor_papers_list.append(ref)
                # Record citation relationship: seed cited ancestor
                self.citation_edges.add((seed_id, ref['id']))

        # Unified deduplication
        result, _, _ = self._deduplicate_and_log(
            ancestor_papers_list,
            self.all_papers,
            "Backward snowballing"
        )

        logger.info(f"\nBackward snowballing complete, collected {len(result)} parent/ancestor papers (after deduplication)")
        return result

    def _cocitation_mining(
        self,
        citing_papers: List[Dict],
        ancestor_papers: List[Dict]
    ) -> List[Dict]:
        """
        Step 4: Lateral supplement/co-citation mining
        Among child and parent nodes, find papers repeatedly mentioned but not yet in the library

        Args:
            citing_papers: List of child node papers
            ancestor_papers: List of parent node papers

        Returns:
            List of co-citation papers
        """
        reference_counter = Counter()  # Count how many times each reference is cited
        all_references = []

        # Merge child and parent nodes (deduplicate)
        seen_ids = set()
        all_nodes = []
        for paper in citing_papers + ancestor_papers:
            if paper['id'] not in seen_ids:
                all_nodes.append(paper)
                seen_ids.add(paper['id'])

        logger.info(f"Analyzing co-citation patterns of {len(all_nodes)} papers (deduplicated)...")

        # Collect references from all papers
        analysis_limit = min(30, len(all_nodes))  # Limit count to control API calls
        for i, paper in enumerate(all_nodes[:analysis_limit], 1):
            logger.info(f"  [{i}/{analysis_limit}] Analyzing: {paper['title'][:40]}...")

            refs = self._get_references(paper['id'], max_results=10)
            for ref in refs:
                ref_id = ref['id']
                all_references.append(ref)
                reference_counter[ref_id] += 1

        # Find papers cited multiple times
        cocitation_papers_list = []
        threshold = 3  # Cited by at least 3 papers

        logger.info(f"\nCo-citation analysis: Find frequently mentioned papers (threshold: {threshold} times)")
        for ref_id, count in reference_counter.most_common(30):
            if count >= threshold:
                # Find corresponding paper details
                ref_paper = next((r for r in all_references if r['id'] == ref_id), None)
                if ref_paper:
                    cocitation_papers_list.append(ref_paper)
                    logger.info(
                        f"  Candidate co-citation paper: {ref_paper['title'][:50]}... "
                        f"(cited {count} times, total citations: {ref_paper['cited_by_count']})"
                    )

        # Unified deduplication
        result, _, _ = self._deduplicate_and_log(
            cocitation_papers_list,
            self.all_papers,
            "Co-citation mining"
        )

        logger.info(f"\nCo-citation mining complete, found {len(result)} repeatedly mentioned papers (after deduplication)")
        return result

    def _execute_second_round_snowballing(self):
        """
        Execute second round snowballing: perform another round of expansion on papers from first round
        Includes: citing_papers, ancestor_papers, cocitation_papers
        """
        # Merge all papers from first round (deduplicate)
        seen_ids = set()
        first_round_papers = []
        for paper in self.citing_papers + self.ancestor_papers + self.cocitation_papers:
            if paper['id'] not in seen_ids:
                first_round_papers.append(paper)
                seen_ids.add(paper['id'])

        logger.info(f"First round obtained {len(first_round_papers)} papers (deduplicated)")
        logger.info(f"Maximum {self.second_round_limit} citations per paper")

        # Second round forward snowballing
        logger.info("\n" + "-" * 60)
        logger.info("Second round forward snowballing: Find child nodes from first round papers")
        logger.info("-" * 60)

        # Use unified method with limit parameter
        second_citing = self._forward_snowballing(
            first_round_papers,
            max_per_paper=self.second_round_limit
        )
        self.second_round_citing_count = len(second_citing)

        # Merge with first round (deduplicate using dictionary)
        citing_dict = {p['id']: p for p in self.citing_papers}
        before_merge = len(citing_dict)
        citing_dict.update({p['id']: p for p in second_citing})
        after_merge = len(citing_dict)
        self.citing_papers = list(citing_dict.values())

        logger.info(f"Second round forward snowballing merge statistics:")
        logger.info(f"   First round child nodes: {len(self.citing_papers) - len(second_citing)} papers")
        logger.info(f"   Second round new: {self.second_round_citing_count} papers")
        logger.info(f"   Duplicates during merge: {before_merge + self.second_round_citing_count - after_merge} papers")
        logger.info(f"   Total after merge: {len(self.citing_papers)} papers")

        # Second round backward snowballing
        logger.info("\n" + "-" * 60)
        logger.info("Second round backward snowballing: Find parent nodes from first round papers")
        logger.info("-" * 60)

        second_ancestor = self._backward_snowballing(
            first_round_papers,
            max_per_paper=self.second_round_limit
        )
        self.second_round_ancestor_count = len(second_ancestor)

        # Merge with first round
        ancestor_dict = {p['id']: p for p in self.ancestor_papers}
        before_merge = len(ancestor_dict)
        ancestor_dict.update({p['id']: p for p in second_ancestor})
        after_merge = len(ancestor_dict)
        self.ancestor_papers = list(ancestor_dict.values())

        logger.info(f"Second round backward snowballing merge statistics:")
        logger.info(f"   First round parent nodes: {len(self.ancestor_papers) - len(second_ancestor)} papers")
        logger.info(f"   Second round new: {self.second_round_ancestor_count} papers")
        logger.info(f"   Duplicates during merge: {before_merge + self.second_round_ancestor_count - after_merge} papers")
        logger.info(f"   Total after merge: {len(self.ancestor_papers)} papers")

        # Second round co-citation mining
        logger.info("\n" + "-" * 60)
        logger.info("Second round co-citation mining: Analyze co-citation patterns of second round papers")
        logger.info("-" * 60)

        second_cocitation = self._cocitation_mining(
            second_citing,
            second_ancestor
        )
        self.second_round_cocitation_count = len(second_cocitation)

        # Merge with first round
        cocitation_dict = {p['id']: p for p in self.cocitation_papers}
        before_merge = len(cocitation_dict)
        cocitation_dict.update({p['id']: p for p in second_cocitation})
        after_merge = len(cocitation_dict)
        self.cocitation_papers = list(cocitation_dict.values())

        logger.info(f"Second round co-citation mining merge statistics:")
        logger.info(f"   First round co-citation: {len(self.cocitation_papers) - len(second_cocitation)} papers")
        logger.info(f"   Second round new: {self.second_round_cocitation_count} papers")
        logger.info(f"   Duplicates during merge: {before_merge + self.second_round_cocitation_count - after_merge} papers")
        logger.info(f"   Total after merge: {len(self.cocitation_papers)} papers")

        logger.info("\n" + "=" * 80)
        logger.info(f"Second round snowballing complete (deduplicated and merged with first round)")
        logger.info(f"   Final paper count (after deduplication):")
        logger.info(f"     - Child nodes: {len(self.citing_papers)} papers")
        logger.info(f"     - Parent nodes: {len(self.ancestor_papers)} papers")
        logger.info(f"     - Co-citation papers: {len(self.cocitation_papers)} papers")
        logger.info(f"     - Total: {len(self.citing_papers) + len(self.ancestor_papers) + len(self.cocitation_papers)} papers")
        logger.info("=" * 80)

    def _add_recent_frontiers(
        self,
        topic: str,
        content_keyword: str,
        year_threshold: int
    ) -> List[Dict]:
        """
        Step 5: Add recent SOTA papers (optimized version)

        Strategy:
        - If arXiv enabled: Use arXiv to retrieve recent papers (6-12 months) -> Map to OpenAlex
        - If not enabled: Use OpenAlex to search for recent papers

        Args:
            topic: Topic keyword
            content_keyword: Content keyword
            year_threshold: Year threshold (greater than or equal to this year)

        Returns:
            List of recent papers
        """
        if self.use_arxiv_seeds:
            logger.info("Using arXiv to retrieve recent SOTA papers")
            return self._add_recent_from_arxiv(topic, content_keyword)
        else:
            logger.info("Using OpenAlex to retrieve recent papers")
            return self._add_recent_from_openalex(topic, content_keyword, year_threshold)

    def _add_recent_from_arxiv(
        self,
        topic: str,
        content_keyword: str
    ) -> List[Dict]:
        """
        Retrieve recent papers from arXiv and map to OpenAlex

        Args:
            topic: Topic
            content_keyword: Content keyword

        Returns:
            List of recent papers (OpenAlex format)
        """
        logger.info("  Retrieving recent frontier papers from arXiv (6-12 months)")

        # 1. Retrieve recent arXiv papers
        keywords = [content_keyword] + self.seed_keywords if self.seed_keywords else [content_keyword]
        arxiv_recent = self.arxiv_retriever.retrieve_recent_papers(
            topic=topic,
            keywords=keywords,
            max_results=self.recent_count * 2,  # Retrieve more
            months_back=12
        )

        logger.info(f"  Retrieved {len(arxiv_recent)} recent papers from arXiv")

        # 2. Map to OpenAlex (no forced Concept verification, new papers may not be annotated yet)
        mapped_recent, stats = self.cross_mapper.map_arxiv_to_openalex(
            arxiv_recent,
            verify_concepts=False  # Relax verification for recent papers
        )

        logger.info(f"  Successfully mapped {len(mapped_recent)} recent papers")

        # 3. Deduplicate and store
        recent_papers_list = []
        for paper in mapped_recent[:self.recent_count]:
            recent_papers_list.append(paper)
            logger.info(
                f"  Recent: [{paper.get('published_date', paper['year'])}] "
                f"{paper['title'][:60]}... (arXiv: {paper.get('arxiv_id', 'N/A')})"
            )

        # Deduplicate
        result, _, _ = self._deduplicate_and_log(
            recent_papers_list,
            self.all_papers,
            "Recent SOTA papers (arXiv)"
        )

        logger.info(f"Added {len(result)} recent SOTA papers (arXiv)")
        return result

    def _add_recent_from_openalex(
        self,
        topic: str,
        content_keyword: str,
        year_threshold: int
    ) -> List[Dict]:
        """
        Retrieve recent papers from OpenAlex (traditional mode)

        Args:
            topic: Topic keyword
            content_keyword: Content keyword
            year_threshold: Year threshold (greater than or equal to this year)

        Returns:
            List of recent papers
        """
        query = f"{topic} {content_keyword}"
        logger.info(f"Searching for recent papers: '{query}'")
        logger.info(f"Filter criteria: publication_year >= {year_threshold}")

        params = {
            'search': query,
            'per-page': self.recent_count,
            'sort': 'cited_by_count:desc',  # Select highly cited papers among recent papers
            'filter': f'publication_year:>{year_threshold},cited_by_count:>5'
        }

        try:
            data = self.client._make_request('works', params)
            results = data.get('results', [])

            recent_papers_list = []
            for result in results[:self.recent_count]:
                paper = self.client._parse_paper(result)
                recent_papers_list.append(paper)
                logger.info(
                    f"  Candidate recent paper: [{paper['year']}] {paper['title'][:60]}... "
                    f"(citations: {paper['cited_by_count']})"
                )

            # Unified deduplication
            result, _, _ = self._deduplicate_and_log(
                recent_papers_list,
                self.all_papers,
                "Recent SOTA papers"
            )

            logger.info(f"Added {len(result)} recent SOTA papers (after deduplication)")
            return result

        except Exception as e:
            logger.error(f"Failed to add recent papers: {e}")
            return []

    def _build_closure(self):
        """
        Step 5: Build citation closure
        Build complete citation relationship network for all papers
        """
        paper_ids = list(self.all_papers.keys())
        total_papers = len(paper_ids)

        logger.info(f"Starting to build citation closure for {total_papers} papers...")

        # Get citation relationships for each paper
        for i, paper_id in enumerate(paper_ids, 1):
            paper = self.all_papers[paper_id]
            logger.info(
                f"  [{i}/{total_papers}] Processing: {paper['title'][:40]}..."
            )

            # Get other papers cited by this paper (within our collection)
            cited_papers = self._get_references(paper_id, max_results=20)

            for cited in cited_papers:
                cited_id = cited['id']
                # Only record citation relationships within the collection
                if cited_id in self.all_papers and cited_id != paper_id:
                    edge = (paper_id, cited_id)
                    if edge not in self.citation_edges:
                        self.citation_edges.add(edge)
                        logger.debug(f"    Add edge: {paper_id} -> {cited_id}")

        logger.info(f"Citation closure construction complete! Established {len(self.citation_edges)} citation relationships")

    def _get_filtered_citations(
        self,
        work_id: str,
        min_year: int,
        keywords: List[str],
        max_results: int
    ) -> List[Dict]:
        """
        Get filtered citing papers

        Args:
            work_id: Paper ID
            min_year: Minimum year
            keywords: Keyword list (for relevance filtering)
            max_results: Maximum number of results

        Returns:
            Filtered citing paper list
        """
        if not work_id.startswith('W'):
            work_id = f"W{work_id}"

        # Build filter conditions
        filters = [
            f'cites:{work_id}',
            f'publication_year:>{min_year}'
        ]

        params = {
            'filter': ','.join(filters),
            'per-page': max_results * 2,  # Retrieve more, filter later
            'sort': 'cited_by_count:desc'
        }

        try:
            data = self.client._make_request('works', params)
            results = data.get('results', [])

            # Parse and filter papers
            filtered = []
            for result in results:
                paper = self.client._parse_paper(result)

                # If keywords are required, perform relevance filtering
                if keywords and not self._is_relevant(paper, keywords):
                    continue

                filtered.append(paper)
                if len(filtered) >= max_results:
                    break

            return filtered

        except Exception as e:
            logger.error(f"Failed to get filtered citations: {e}")
            return []

    def _get_references(self, work_id: str, max_results: int = 10) -> List[Dict]:
        """Get references of a paper"""
        if not work_id.startswith('W'):
            work_id = f"W{work_id}"

        params = {
            'filter': f'cited_by:{work_id}',
            'per-page': max_results,
            'sort': 'cited_by_count:desc'
        }

        try:
            data = self.client._make_request('works', params)
            results = data.get('results', [])

            references = []
            for result in results:
                ref = self.client._parse_paper(result)
                references.append(ref)

            return references

        except Exception as e:
            logger.error(f"Failed to get references: {e}")
            return []

    def _is_relevant(self, paper: Dict, keywords: List[str]) -> bool:
        """
        Check if paper is relevant to keywords

        Args:
            paper: Paper data
            keywords: Keyword list

        Returns:
            Whether relevant
        """
        # Combine title and abstract for matching
        text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()

        # Match at least one keyword
        return any(keyword.lower() in text for keyword in keywords)

    def _generate_report(self) -> Dict:
        """
        Generate retrieval report

        Returns:
            Dictionary containing all data and statistics
        """
        report = {
            'statistics': {
                'total_papers': len(self.all_papers),
                'seed_papers': len(self.seed_papers),
                'citing_papers': len(self.citing_papers),
                'ancestor_papers': len(self.ancestor_papers),
                'cocitation_papers': len(self.cocitation_papers),
                'recent_papers': len(self.recent_papers),
                'total_edges': len(self.citation_edges),
                # First round statistics
                'first_round_citing': self.first_round_citing_count,
                'first_round_ancestor': self.first_round_ancestor_count,
                'first_round_cocitation': self.first_round_cocitation_count,
                # Second round statistics
                'second_round_citing': self.second_round_citing_count,
                'second_round_ancestor': self.second_round_ancestor_count,
                'second_round_cocitation': self.second_round_cocitation_count,
                # Whether second round is enabled
                'second_round_enabled': self.enable_second_round
            },
            'papers': self.all_papers,
            'citation_edges': list(self.citation_edges),
            'seed_ids': [p['id'] for p in self.seed_papers],
            'citing_ids': [p['id'] for p in self.citing_papers],
            'ancestor_ids': [p['id'] for p in self.ancestor_papers],
            'cocitation_ids': [p['id'] for p in self.cocitation_papers],
            'recent_ids': [p['id'] for p in self.recent_papers]
        }

        # Print statistics
        logger.info("\n" + "=" * 60)
        logger.info("Retrieval Statistics Report")
        logger.info("=" * 60)
        logger.info(f"Total papers: {report['statistics']['total_papers']}")
        logger.info(f"  1. Foundational seeds: {report['statistics']['seed_papers']}")
        logger.info(f"  2. Child nodes (citing seeds): {report['statistics']['citing_papers']}")
        logger.info(f"     - First round: {report['statistics']['first_round_citing']}")
        if self.enable_second_round:
            logger.info(f"     - Second round: {report['statistics']['second_round_citing']}")
        logger.info(f"  3. Parent nodes (cited by seeds): {report['statistics']['ancestor_papers']}")
        logger.info(f"     - First round: {report['statistics']['first_round_ancestor']}")
        if self.enable_second_round:
            logger.info(f"     - Second round: {report['statistics']['second_round_ancestor']}")
        logger.info(f"  4. Co-citation papers (lateral supplement): {report['statistics']['cocitation_papers']}")
        logger.info(f"     - First round: {report['statistics']['first_round_cocitation']}")
        if self.enable_second_round:
            logger.info(f"     - Second round: {report['statistics']['second_round_cocitation']}")
        logger.info(f"  5. Recent SOTA: {report['statistics']['recent_papers']}")
        logger.info(f"Total citation relationships: {report['statistics']['total_edges']}")
        logger.info(f"Average connections per paper: {report['statistics']['total_edges'] / max(report['statistics']['total_papers'], 1):.2f}")
        logger.info("=" * 60)

        return report

    def export_to_graph_format(self) -> Dict:
        """
        Export to graph data format (for visualization)

        Returns:
            Dictionary containing nodes and edges
        """
        nodes = []
        for paper_id, paper in self.all_papers.items():
            # Determine node type (priority order)
            if paper_id in [p['id'] for p in self.seed_papers]:
                node_type = 'seed'
            elif paper_id in [p['id'] for p in self.ancestor_papers]:
                node_type = 'ancestor'
            elif paper_id in [p['id'] for p in self.citing_papers]:
                node_type = 'citing'
            elif paper_id in [p['id'] for p in self.cocitation_papers]:
                node_type = 'cocitation'
            elif paper_id in [p['id'] for p in self.recent_papers]:
                node_type = 'recent'
            else:
                node_type = 'other'

            nodes.append({
                'id': paper_id,
                'label': paper['title'][:50],
                'type': node_type,
                'year': paper['year'],
                'citations': paper['cited_by_count'],
                'authors': paper['authors']
            })

        edges = [
            {'source': source, 'target': target}
            for source, target in self.citation_edges
        ]

        return {
            'nodes': nodes,
            'edges': edges
        }


if __name__ == "__main__":
    # Test code
    logger.info("Starting snowball retrieval system test...")

    # Create retrieval system
    retrieval = SnowballRetrieval(
        seed_count=5,
        citations_per_seed=6,
        recent_count=10,
        seed_keywords=["reasoning", "chain of thought", "prompting", "thinking"]
    )

    # Execute complete pipeline
    result = retrieval.execute_full_pipeline(
        topic="Large Language Models",
        content_keyword="Reasoning",
        seed_year_threshold=2023
    )

    # Export graph data
    graph_data = retrieval.export_to_graph_format()
    logger.info(f"\nGraph data export complete: {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")

    # Display partial results
    logger.info("\nExample papers (first 5):")
    for i, (paper_id, paper) in enumerate(list(result['papers'].items())[:5], 1):
        logger.info(f"{i}. [{paper['year']}] {paper['title']}")
        logger.info(f"   Citations: {paper['cited_by_count']}, Authors: {', '.join(paper['authors'])}")
