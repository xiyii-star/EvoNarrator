"""
Simplified OpenAlex API client
For paper retrieval and citation relationship acquisition
"""

import requests
import time
from typing import List, Dict, Optional
import logging
from urllib.parse import urlencode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAlex concept ID constants
CONCEPT_COMPUTER_SCIENCE = "C41008148"  # Computer Science
CONCEPT_ARTIFICIAL_INTELLIGENCE = "C154945302"  # Artificial Intelligence


class OpenAlexClient:
    """
    OpenAlex API client - simplified version
    """

    def __init__(
        self,
        email: Optional[str] = "1743623557@qq.com",
        base_url: str = "https://api.openalex.org",
        rate_limit_delay: float = 0.1
    ):
        self.base_url = base_url.rstrip('/')
        self.email = email
        self.rate_limit_delay = rate_limit_delay

        self.session = requests.Session()
        if email:
            self.session.params = {'mailto': email}

        logger.info(f"OpenAlex client initialized (email: {email})")

    def _make_request(self, endpoint: str, params: Dict = None, silent_404: bool = False) -> Dict:
        """
        Send API request

        Args:
            endpoint: API endpoint
            params: Request parameters
            silent_404: Whether to silence 404 errors (default False)

        Returns:
            Response JSON data
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        params = params or {}

        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            # 404 errors can be silenced (many new papers not existing in OpenAlex is normal)
            if e.response.status_code == 404 and silent_404:
                logger.debug(f"Paper not found (404): {endpoint}")
                return {}
            else:
                logger.error(f"Request failed: {e.response.status_code} {e.response.reason} for url: {e.response.url}")
                return {}
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return {}

    def search_papers(
        self,
        topic: str,
        max_results: int = 10,
        sort_by: str = "cited_by_count",
        min_citations: int = 10,
        year_filter: Optional[str] = None,
        additional_filters: Optional[List[str]] = None,
        require_cs_ai: bool = False,
        concept_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Search papers (supports advanced filtering)

        Args:
            topic: Search topic
            max_results: Maximum number of results
            sort_by: Sort method
            min_citations: Minimum citation count
            year_filter: Year filter (e.g., "<2023", ">2022", "2020-2023")
            additional_filters: List of additional filter conditions
            require_cs_ai: Whether to restrict to CS/AI domain (default False, no restriction)
                          When set to True, filters CS OR AI, as long as one of the tags is present
            concept_filter: Custom concept filter (e.g., "C41008148" for Computer Science)
                          When specified, overrides the default CS OR AI filter
                          Supported concepts:
                          - C41008148: Computer Science
                          - C154945302: Artificial Intelligence

        Returns:
            List of papers
        """
        logger.info(f"Searching papers: '{topic}' (max {max_results} papers)")

        # Build filter conditions
        filters = [f'cited_by_count:>{min_citations}']

        # Enforce CS/AI domain restriction
        if require_cs_ai:
            if concept_filter:
                # Use custom concept filter
                filters.append(f'concepts.id:{concept_filter}')
                logger.info(f"Applying concept filter: {concept_filter}")
            else:
                # Default to Computer Science OR Artificial Intelligence concept
                # Papers need only one of CS or AI tags (using | for OR logic)
                filters.append('concepts.id:C41008148|C154945302')
                logger.info("Applying default filter: Computer Science (C41008148) OR Artificial Intelligence (C154945302)")

        if year_filter:
            filters.append(f'publication_year:{year_filter}')

        if additional_filters:
            filters.extend(additional_filters)

        params = {
            'search': topic,
            'per-page': min(max_results, 25),
            'sort': f'{sort_by}:desc',
            'filter': ','.join(filters)
        }

        try:
            data = self._make_request('works', params)
            results = data.get('results', [])

            papers = []
            for result in results[:max_results]:
                paper = self._parse_paper(result)
                papers.append(paper)

            logger.info(f"Found {len(papers)} papers")
            return papers

        except Exception as e:
            logger.error(f"Paper search failed: {e}")
            return []

    def get_citations(self, paper_id: str, max_results: int = 5) -> List[Dict]:
        """
        Get papers that cite this paper (forward snowballing)

        Args:
            paper_id: Paper ID (OpenAlex Work ID)
            max_results: Maximum number of results

        Returns:
            List of papers that cite this paper
        """
        logger.info(f"Getting papers that cite this paper: {paper_id}")

        # Ensure paper_id format is correct
        if not paper_id.startswith('W'):
            paper_id = f"W{paper_id}"

        # Use cites filter: find papers that cite this paper
        params = {
            'filter': f'cites:{paper_id}',
            'per-page': max_results,
            'sort': 'cited_by_count:desc'
        }

        try:
            data = self._make_request('works', params)
            results = data.get('results', [])

            citations = []
            for result in results:
                citation = self._parse_paper(result)
                citations.append(citation)

            logger.info(f"  → Found {len(citations)} papers that cite this paper")
            return citations

        except Exception as e:
            logger.error(f"Failed to get citing papers: {e}")
            return []

    def get_references(self, paper_id: str, max_results: int = 5) -> List[Dict]:
        """
        Get references cited by this paper (backward snowballing)

        Args:
            paper_id: Paper ID (OpenAlex Work ID)
            max_results: Maximum number of results

        Returns:
            List of references cited by this paper
        """
        logger.info(f"Getting references for this paper: {paper_id}")

        # Ensure paper_id format is correct
        if not paper_id.startswith('W'):
            paper_id = f"W{paper_id}"

        try:
            # Method 1: First get paper details, extract ID list from referenced_works field
            paper_data = self._make_request(f'works/{paper_id}', silent_404=True)

            if not paper_data:
                logger.debug(f"  Paper details not found or unavailable: {paper_id}")
                return []

            # Get referenced_works ID list
            referenced_work_ids = paper_data.get('referenced_works', [])

            if not referenced_work_ids:
                logger.debug(f"  This paper has no references")
                return []

            # Take first max_results items
            referenced_work_ids = referenced_work_ids[:max_results]

            logger.debug(f"  → Found {len(referenced_work_ids)} reference IDs, fetching details...")

            # Batch fetch detailed information for these papers
            references = []
            for ref_id in referenced_work_ids:
                # Extract clean ID (remove URL prefix)
                clean_ref_id = ref_id.split('/')[-1] if '/' in ref_id else ref_id

                try:
                    ref_data = self._make_request(f'works/{clean_ref_id}', silent_404=True)
                    if ref_data:
                        ref_paper = self._parse_paper(ref_data)
                        references.append(ref_paper)
                except Exception as e:
                    logger.debug(f"    Skipping reference {clean_ref_id}: {e}")
                    continue

            logger.info(f"  → Successfully fetched {len(references)} reference details")
            return references

        except Exception as e:
            logger.error(f"Failed to get references: {e}")
            return []

    def _parse_paper(self, raw_data: Dict) -> Dict:
        """Parse paper data to standard format"""
        # Extract author information
        authors = []
        for authorship in raw_data.get('authorships', [])[:3]:  # Only take first 3 authors
            author = authorship.get('author', {})
            authors.append(author.get('display_name', 'Unknown'))

        # Extract PDF link
        pdf_url = None
        open_access = raw_data.get('open_access', {})
        if open_access.get('is_oa') and open_access.get('oa_url'):
            pdf_url = open_access.get('oa_url')

        # Standardize paper data
        paper = {
            'id': raw_data.get('id', '').split('/')[-1] if raw_data.get('id') else '',
            'title': raw_data.get('title', 'Untitled'),
            'authors': authors,
            'year': raw_data.get('publication_year', 0),
            'cited_by_count': raw_data.get('cited_by_count', 0),
            'doi': raw_data.get('doi', ''),
            'pdf_url': pdf_url,
            'abstract': self._reconstruct_abstract(raw_data.get('abstract_inverted_index', {})),
            'venue': raw_data.get('host_venue', {}).get('display_name', ''),
            'is_open_access': open_access.get('is_oa', False)
        }

        return paper

    def get_work_with_concepts(self, work_id: str) -> Optional[Dict]:
        """
        Get detailed information for a single paper (including complete concept tags)

        Args:
            work_id: Paper ID

        Returns:
            Paper details dictionary containing concepts field
        """
        if not work_id.startswith('W'):
            work_id = f"W{work_id}"

        try:
            data = self._make_request(f'works/{work_id}')
            if data:
                # Return complete data (including concepts)
                paper = self._parse_paper(data)
                # Add concepts field
                paper['concepts'] = data.get('concepts', [])
                paper['primary_topic'] = data.get('primary_topic', {})
                return paper
            return None
        except Exception as e:
            logger.error(f"Failed to get paper details: {e}")
            return None

    def search_by_arxiv_id(self, arxiv_id: str) -> Optional[Dict]:
        """
        Search paper by arXiv ID

        Args:
            arxiv_id: arXiv ID (e.g., "2301.12345" or "2301.12345v2")

        Returns:
            Paper information dictionary
        """
        # Clean arXiv ID (remove version number)
        clean_id = arxiv_id.split('v')[0]

        params = {
            'filter': f'ids.arxiv:{clean_id}',
            'per-page': 1
        }

        try:
            data = self._make_request('works', params)
            results = data.get('results', [])

            if results:
                paper = self._parse_paper(results[0])
                paper['arxiv_id'] = arxiv_id
                return paper
            return None

        except Exception as e:
            logger.error(f"Search by arXiv ID failed: {e}")
            return None

    def search_by_doi(self, doi: str) -> Optional[Dict]:
        """
        Search paper by DOI

        Args:
            doi: DOI string

        Returns:
            Paper information dictionary
        """
        # Clean DOI (remove prefix)
        clean_doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')

        params = {
            'filter': f'doi:{clean_doi}',
            'per-page': 1
        }

        try:
            data = self._make_request('works', params)
            results = data.get('results', [])

            if results:
                return self._parse_paper(results[0])
            return None

        except Exception as e:
            logger.error(f"Search by DOI failed: {e}")
            return None

    def filter_by_concepts(
        self,
        papers: List[Dict],
        required_concepts: List[str],
        min_score: float = 0.3
    ) -> List[Dict]:
        """
        Filter papers by concept tags

        Args:
            papers: List of papers
            required_concepts: List of required concepts (concept names)
            min_score: Minimum confidence score

        Returns:
            Filtered list of papers
        """
        filtered = []

        for paper in papers:
            paper_id = paper['id']

            # Get complete concept information
            full_paper = self.get_work_with_concepts(paper_id)
            if not full_paper:
                continue

            concepts = full_paper.get('concepts', [])
            if not concepts:
                continue

            # Check if required concepts are matched
            matched = False
            for concept in concepts:
                concept_name = concept.get('display_name', '')
                concept_score = concept.get('score', 0.0)

                for required in required_concepts:
                    if required.lower() in concept_name.lower():
                        if concept_score >= min_score:
                            matched = True
                            break

                if matched:
                    break

            if matched:
                filtered.append(full_paper)

        logger.info(f"Concept filtering: {len(papers)} -> {len(filtered)} papers")
        return filtered

    def get_work_details(self, work_id: str) -> Optional[Dict]:
        """
        Get detailed information for a single paper (compatible with old code)

        Args:
            work_id: Paper ID

        Returns:
            Paper details dictionary
        """
        return self.get_work_with_concepts(work_id)

    def get_paper_by_id(self, paper_id: str) -> Optional[Dict]:
        """
        Get a single paper by ID (alias method)

        Args:
            paper_id: Paper ID

        Returns:
            Paper information dictionary
        """
        return self.get_work_with_concepts(paper_id)

    def _reconstruct_abstract(self, inverted_index: Dict) -> str:
        """Reconstruct abstract text"""
        if not inverted_index:
            return ""

        try:
            # Convert inverted index to text
            words_with_pos = []
            for word, positions in inverted_index.items():
                for pos in positions:
                    words_with_pos.append((pos, word))

            # Sort by position and join
            words_with_pos.sort(key=lambda x: x[0])
            abstract = ' '.join([word for pos, word in words_with_pos])

            # Truncate overly long abstracts
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."

            return abstract
        except:
            return ""


if __name__ == "__main__":
    # Test code
    client = OpenAlexClient()

    print("=" * 60)
    print("Example 1: Search 'Attention Mechanism' (default CS OR AI filter)")
    print("=" * 60)
    papers = client.search_papers("Attention Mechanism", max_results=3)

    for i, paper in enumerate(papers, 1):
        print(f"\n[{i}] {paper['title']}")
        print(f"    Authors: {', '.join(paper['authors'])}")
        print(f"    Year: {paper['year']}")
        print(f"    Citations: {paper['cited_by_count']}")
        if paper['pdf_url']:
            print(f"    PDF: {paper['pdf_url']}")

    print("\n" + "=" * 60)
    print("Example 2: Search 'Machine Learning' (AI domain only)")
    print("=" * 60)
    # Use only AI concept for filtering
    papers = client.search_papers(
        "Machine Learning",
        max_results=3,
        concept_filter="C154945302"  # Artificial Intelligence only
    )

    for i, paper in enumerate(papers, 1):
        print(f"\n[{i}] {paper['title']}")
        print(f"    Authors: {', '.join(paper['authors'])}")
        print(f"    Year: {paper['year']}")
        print(f"    Citations: {paper['cited_by_count']}")

    print("\n" + "=" * 60)
    print("Example 3: Search 'Neural Network' (no domain restriction)")
    print("=" * 60)
    # Disable CS/AI filtering
    papers = client.search_papers(
        "Neural Network",
        max_results=3,
        require_cs_ai=False
    )

    for i, paper in enumerate(papers, 1):
        print(f"\n[{i}] {paper['title']}")
        print(f"    Year: {paper['year']}")
        print(f"    Citations: {paper['cited_by_count']}")