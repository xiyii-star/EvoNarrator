"""
Cross-Database ID Mapping Module
Map arXiv/Semantic Scholar papers to OpenAlex ID system

Core Features:
1. arXiv ID -> OpenAlex ID mapping
2. DOI -> OpenAlex ID mapping
3. Title search mapping
4. Concept filtering verification (ensure papers belong to target domain)
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from openalex_client import OpenAlexClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAlex Concept ID mapping
CONCEPT_IDS = {
    "Computer Science": "C41008148",
    "Artificial Intelligence": "C154945302",
    "Machine Learning": "C119857082",
    "Natural Language Processing": "C204321447",
    "Computer Vision": "C73124549",
    "Neural Networks": "C50644808",
    "Deep Learning": "C78519656",
    "Information Retrieval": "C46978859",
}


class CrossDatabaseMapper:
    """
    Cross-database paper mapper
    Map arXiv papers to OpenAlex with concept filtering
    """

    def __init__(
        self,
        client: Optional[OpenAlexClient] = None,
        min_concept_score: float = 0.3,
        required_concepts: Optional[List[str]] = None
    ):
        """
        Initialize mapper

        Args:
            client: OpenAlex client
            min_concept_score: Minimum concept confidence score (0-1)
            required_concepts: Required concept list (concept names, e.g. ["Computer Science", "Artificial Intelligence"])
        """
        self.client = client or OpenAlexClient()
        self.min_concept_score = min_concept_score
        self.required_concepts = required_concepts or ["Computer Science"]

        # Statistics
        self.stats = {
            'total_papers': 0,
            'mapped': 0,
            'failed': 0,
            'filtered_by_concept': 0,
            'mapping_methods': {
                'arxiv_id': 0,
                'doi': 0,
                'title_search': 0
            }
        }

        logger.info("Cross-database mapper initialization complete")
        logger.info(f"  min_concept_score={min_concept_score}")
        logger.info(f"  required_concepts={required_concepts}")

    def map_arxiv_to_openalex(
        self,
        arxiv_papers: List[Dict],
        verify_concepts: bool = True
    ) -> Tuple[List[Dict], Dict]:
        """
        Batch map arXiv papers to OpenAlex

        Args:
            arxiv_papers: List of arXiv papers
            verify_concepts: Whether to verify concepts (filter non-target domain papers)

        Returns:
            (List of successfully mapped papers, statistics)
        """
        logger.info(f"Starting to map {len(arxiv_papers)} arXiv papers to OpenAlex...")
        if not verify_concepts:
            logger.info("  Concept verification disabled (domain filtering already done in arXiv stage)")

        self.stats['total_papers'] = len(arxiv_papers)
        mapped_papers = []

        for i, arxiv_paper in enumerate(arxiv_papers, 1):
            logger.info(f"\n[{i}/{len(arxiv_papers)}] Mapping: {arxiv_paper['title'][:50]}...")

            # Attempt mapping
            openalex_paper = self._map_single_paper(arxiv_paper)

            if not openalex_paper:
                logger.warning(f"  Mapping failed")
                self.stats['failed'] += 1
                continue

            # Verify concepts (optional)
            if verify_concepts:
                is_valid, concept_info = self._verify_concepts(openalex_paper)
                if not is_valid:
                    logger.warning(
                        f"  Concept filtering: Does not belong to target domain "
                        f"(matched concepts: {concept_info})"
                    )
                    self.stats['filtered_by_concept'] += 1
                    continue

                logger.info(f"  Concept verification passed: {concept_info}")

            # Merge arXiv and OpenAlex data
            merged_paper = self._merge_paper_data(arxiv_paper, openalex_paper)
            mapped_papers.append(merged_paper)

            self.stats['mapped'] += 1
            logger.info(
                f"  Mapping successful: OpenAlex ID = {openalex_paper['id']}, "
                f"citations = {openalex_paper['cited_by_count']}"
            )

            # Avoid API rate limiting
            time.sleep(0.15)

        logger.info(f"\nMapping complete:")
        logger.info(f"  Total papers: {self.stats['total_papers']}")
        logger.info(f"  Mapping successful: {self.stats['mapped']}")
        logger.info(f"  Mapping failed: {self.stats['failed']}")
        if verify_concepts:
            logger.info(f"  Concept filtering: {self.stats['filtered_by_concept']}")

        return mapped_papers, self.stats

    def _map_single_paper(self, arxiv_paper: Dict) -> Optional[Dict]:
        """
        Map single paper to OpenAlex

        Attempt order:
        1. arXiv ID query
        2. DOI query
        3. Title exact search

        Args:
            arxiv_paper: arXiv paper data

        Returns:
            OpenAlex paper data, None if failed
        """
        # Method 1: Query by arXiv ID
        arxiv_id = arxiv_paper.get('arxiv_id')
        if arxiv_id:
            openalex_paper = self._query_by_arxiv_id(arxiv_id)
            if openalex_paper:
                self.stats['mapping_methods']['arxiv_id'] += 1
                logger.info(f"  → Method: arXiv ID")
                return openalex_paper

        # Method 2: Query by DOI
        doi = arxiv_paper.get('doi')
        if doi:
            openalex_paper = self._query_by_doi(doi)
            if openalex_paper:
                self.stats['mapping_methods']['doi'] += 1
                logger.info(f"  → Method: DOI")
                return openalex_paper

        # Method 3: Exact title search
        title = arxiv_paper.get('title')
        if title:
            openalex_paper = self._query_by_title(title)
            if openalex_paper:
                self.stats['mapping_methods']['title_search'] += 1
                logger.info(f"  → Method: Title search")
                return openalex_paper

        return None

    def _query_by_arxiv_id(self, arxiv_id: str) -> Optional[Dict]:
        """
        Query OpenAlex by arXiv ID

        Args:
            arxiv_id: arXiv ID (e.g. "2301.12345" or "2301.12345v2")

        Returns:
            OpenAlex paper data
        """
        # Clean arXiv ID (remove version number)
        clean_id = arxiv_id.split('v')[0]

        # OpenAlex supports filtering by external_ids.arxiv
        # Try different arXiv ID formats
        arxiv_formats = [
            clean_id,                              # 2301.12345
            f"arXiv:{clean_id}",                   # arXiv:2301.12345
        ]

        for arxiv_format in arxiv_formats:
            params = {
                'filter': f'ids.arxiv:{arxiv_format}',
                'per-page': 1
            }

            try:
                data = self.client._make_request('works', params)
                results = data.get('results', [])

                if results:
                    logger.debug(f"arXiv ID query successful, format: {arxiv_format}")
                    return self.client._parse_paper(results[0])

            except Exception as e:
                logger.debug(f"arXiv ID query failed (format: {arxiv_format}): {e}")
                continue

        return None

    def _query_by_doi(self, doi: str) -> Optional[Dict]:
        """
        Query OpenAlex by DOI

        Args:
            doi: DOI string

        Returns:
            OpenAlex paper data
        """
        # Clean DOI (remove prefix)
        clean_doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')

        params = {
            'filter': f'doi:{clean_doi}',
            'per-page': 1
        }

        try:
            data = self.client._make_request('works', params)
            results = data.get('results', [])

            if results:
                return self.client._parse_paper(results[0])

        except Exception as e:
            logger.debug(f"DOI query failed: {e}")

        return None

    def _query_by_title(self, title: str) -> Optional[Dict]:
        """
        Query OpenAlex by title search

        Args:
            title: Paper title

        Returns:
            OpenAlex paper data
        """
        params = {
            'filter': f'title.search:{title}',
            'per-page': 5  # Get top 5, select best match
        }

        try:
            data = self.client._make_request('works', params)
            results = data.get('results', [])

            if not results:
                return None

            # Find best matching paper by title
            best_match = None
            best_score = 0.0

            for result in results:
                result_title = result.get('title', '').lower()
                query_title = title.lower()

                # Simple similarity calculation (Jaccard similarity)
                similarity = self._compute_title_similarity(result_title, query_title)

                if similarity > best_score:
                    best_score = similarity
                    best_match = result

            # Only accept high similarity matches (threshold 0.7)
            if best_match and best_score >= 0.7:
                return self.client._parse_paper(best_match)

        except Exception as e:
            logger.debug(f"Title search failed: {e}")

        return None

    def _compute_title_similarity(self, title1: str, title2: str) -> float:
        """
        Calculate title similarity (Jaccard similarity)

        Args:
            title1: Title 1
            title2: Title 2

        Returns:
            Similarity score (0-1)
        """
        # Tokenize
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())

        # Remove common stopwords
        stopwords = {'a', 'an', 'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for'}
        words1 = words1 - stopwords
        words2 = words2 - stopwords

        # Jaccard similarity
        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _verify_concepts(self, openalex_paper: Dict) -> Tuple[bool, str]:
        """
        Verify paper concept labels (ensure it belongs to target domain)

        Args:
            openalex_paper: OpenAlex paper data

        Returns:
            (Whether verification passed, concept information description)
        """
        # Get paper concept labels (extract from raw API data)
        # Note: _parse_paper may not include concepts field, need to retrieve again
        paper_id = openalex_paper['id']

        try:
            # Ensure paper_id format is correct (remove possible existing W prefix)
            if paper_id.startswith('W'):
                clean_id = paper_id
            else:
                clean_id = f'W{paper_id}'

            # Retrieve detailed information again (including concepts)
            data = self.client._make_request(f'works/{clean_id}')

            if not data:
                return False, "Unable to retrieve paper details"

            concepts = data.get('concepts', [])

            if not concepts:
                return False, "No concept labels"

            # Check required concepts
            matched_concepts = []
            for concept in concepts:
                concept_name = concept.get('display_name', '')
                concept_score = concept.get('score', 0.0)

                # Check if matches required concepts
                for required in self.required_concepts:
                    if required.lower() in concept_name.lower():
                        if concept_score >= self.min_concept_score:
                            matched_concepts.append(f"{concept_name}({concept_score:.2f})")

            # At least match one required concept
            if matched_concepts:
                return True, ", ".join(matched_concepts)
            else:
                # List actual concepts
                actual_concepts = [
                    f"{c.get('display_name', 'Unknown')}({c.get('score', 0.0):.2f})"
                    for c in concepts[:3]
                ]
                return False, ", ".join(actual_concepts)

        except Exception as e:
            logger.debug(f"Concept verification failed: {e}")
            return False, "Verification failed"

    def _merge_paper_data(self, arxiv_paper: Dict, openalex_paper: Dict) -> Dict:
        """
        Merge arXiv and OpenAlex data

        Args:
            arxiv_paper: arXiv paper data
            openalex_paper: OpenAlex paper data

        Returns:
            Merged paper data
        """
        # Use OpenAlex data as base
        merged = openalex_paper.copy()

        # Add arXiv-specific fields
        merged['arxiv_id'] = arxiv_paper.get('arxiv_id')
        merged['arxiv_categories'] = arxiv_paper.get('categories', [])
        merged['arxiv_primary_category'] = arxiv_paper.get('primary_category')
        merged['arxiv_published_date'] = arxiv_paper.get('published_date')

        # If OpenAlex has no abstract, use arXiv's
        if not merged.get('abstract') and arxiv_paper.get('abstract'):
            merged['abstract'] = arxiv_paper['abstract']

        # If OpenAlex has no PDF link, use arXiv's
        if not merged.get('pdf_url') and arxiv_paper.get('pdf_url'):
            merged['pdf_url'] = arxiv_paper['pdf_url']

        # Mark source
        merged['source'] = 'arxiv+openalex'

        return merged

    def get_statistics(self) -> Dict:
        """
        Get mapping statistics

        Returns:
            Statistics dictionary
        """
        stats = self.stats.copy()
        if stats['total_papers'] > 0:
            stats['success_rate'] = stats['mapped'] / stats['total_papers']
            stats['filter_rate'] = stats['filtered_by_concept'] / stats['total_papers']
        else:
            stats['success_rate'] = 0.0
            stats['filter_rate'] = 0.0

        return stats


if __name__ == "__main__":
    # Test code
    from arxiv_seed_retriever import ArxivSeedRetriever

    # 1. Retrieve arXiv seed papers
    print("=" * 80)
    print("Step 1: Retrieve arXiv seed papers")
    print("=" * 80)

    arxiv_retriever = ArxivSeedRetriever(
        max_results_per_query=20,
        years_back=3,
        min_relevance_score=0.6
    )

    arxiv_papers = arxiv_retriever.retrieve_seed_papers(
        topic="Natural Language Processing",
        keywords=["transformer", "language model"],
        max_seeds=10
    )

    print(f"\nFound {len(arxiv_papers)} arXiv seed papers")

    # 2. Map to OpenAlex
    print("\n" + "=" * 80)
    print("Step 2: Map to OpenAlex")
    print("=" * 80)

    mapper = CrossDatabaseMapper(
        min_concept_score=0.3,
        required_concepts=["Computer Science", "Artificial Intelligence"]
    )

    mapped_papers, stats = mapper.map_arxiv_to_openalex(
        arxiv_papers,
        verify_concepts=True
    )

    # 3. Display results
    print("\n" + "=" * 80)
    print("Mapping Results")
    print("=" * 80)
    print(f"Total papers: {stats['total_papers']}")
    print(f"Mapping successful: {stats['mapped']} ({stats['success_rate']:.1%})")
    print(f"Mapping failed: {stats['failed']}")
    print(f"Concept filtering: {stats['filtered_by_concept']} ({stats['filter_rate']:.1%})")
    print(f"\nMapping method statistics:")
    for method, count in stats['mapping_methods'].items():
        print(f"  {method}: {count}")

    print(f"\nSuccessfully mapped paper examples (first 3):")
    for i, paper in enumerate(mapped_papers[:3], 1):
        print(f"\n[{i}] {paper['title']}")
        print(f"    arXiv ID: {paper.get('arxiv_id', 'N/A')}")
        print(f"    OpenAlex ID: {paper['id']}")
        print(f"    Year: {paper['year']}")
        print(f"    Citations: {paper['cited_by_count']}")
        print(f"    Source: {paper['source']}")
