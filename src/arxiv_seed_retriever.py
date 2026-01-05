"""
arXiv seed paper retrieval module
Retrieve high-quality seed papers based on arXiv API and classification system

Core strategies:
1. Use arXiv Categories for precise retrieval
2. Combine keywords (title, abstract) for filtering
3. Limit time range (recent 3-5 years)
4. Sort by relevance and citation count
"""

import arxiv
import logging
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from llm_config import LLMConfig, LLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# arXiv category mapping table (CS subfields)
ARXIV_CATEGORY_MAP = {
    "NLP": ["cs.CL"],  # Computation and Language
    "Machine Learning": ["cs.LG", "stat.ML"],  # Machine Learning
    "Computer Vision": ["cs.CV"],  # Computer Vision
    "Artificial Intelligence": ["cs.AI"],  # Artificial Intelligence
    "Robotics": ["cs.RO"],  # Robotics
    "Information Retrieval": ["cs.IR"],  # Information Retrieval
    "Neural Networks": ["cs.NE"],  # Neural and Evolutionary Computing
    "Cryptography": ["cs.CR"],  # Cryptography and Security
    "Software Engineering": ["cs.SE"],  # Software Engineering
    "Databases": ["cs.DB"],  # Databases
    "Distributed Computing": ["cs.DC"],  # Distributed Computing
    "Human-Computer Interaction": ["cs.HC"],  # Human-Computer Interaction
}


class ArxivSeedRetriever:
    """
    arXiv seed paper retriever
    Focused on retrieving high-quality domain seed papers
    """

    def __init__(
        self,
        max_results_per_query: int = 50,
        years_back: int = 5,
        min_relevance_score: float = 0.4,  # Lower threshold to improve recall
        llm_client: Optional[LLMClient] = None,
        use_llm_query_generation: bool = True,
        enable_semantic_expansion: bool = True,
        expansion_max_topics: int = 4,
        expansion_max_keywords: int = 8
    ):
        """
        Initialize arXiv seed retriever

        Args:
            max_results_per_query: Maximum number of results per query
            years_back: Number of years to look back (from current year)
            min_relevance_score: Minimum relevance score (0-1), default 0.4 to improve recall
            llm_client: LLM client (optional, for intelligent query generation)
            use_llm_query_generation: Whether to use LLM for query generation (default True)
            enable_semantic_expansion: Whether to enable semantic expansion (default True)
            expansion_max_topics: Maximum number of expanded topics (default 4)
            expansion_max_keywords: Maximum number of expanded keywords (default 8)
        """
        self.client = arxiv.Client()
        self.max_results_per_query = max_results_per_query
        self.years_back = years_back
        self.min_relevance_score = min_relevance_score
        self.llm_client = llm_client
        self.use_llm_query_generation = use_llm_query_generation
        self.enable_semantic_expansion = enable_semantic_expansion
        self.expansion_max_topics = expansion_max_topics
        self.expansion_max_keywords = expansion_max_keywords

        logger.info("arXiv seed retriever initialized")
        logger.info(f"  max_results_per_query={max_results_per_query}")
        logger.info(f"  years_back={years_back}")
        logger.info(f"  min_relevance_score={min_relevance_score}")
        logger.info(f"  use_llm_query_generation={use_llm_query_generation}")
        if use_llm_query_generation and llm_client:
            logger.info(f"  enable_semantic_expansion={enable_semantic_expansion}")
            if enable_semantic_expansion:
                logger.info(f"  expansion_max_topics={expansion_max_topics}")
                logger.info(f"  expansion_max_keywords={expansion_max_keywords}")

    def retrieve_seed_papers(
        self,
        topic: str,
        keywords: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        max_seeds: int = 100,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance
    ) -> List[Dict]:
        """
        Retrieve seed papers (high-quality core papers)

        Args:
            topic: Topic name (e.g., "Natural Language Processing")
            keywords: Keyword list (for title/abstract matching)
            categories: arXiv category list (e.g., ["cs.CL", "cs.AI"])
            max_seeds: Maximum number of seed papers
            sort_by: Sort method

        Returns:
            List of seed papers
        """
        logger.info(f"Starting seed paper retrieval: topic='{topic}'")

        # 1. Automatically infer categories
        if not categories:
            categories = self._infer_categories(topic)
            logger.info(f"Auto-inferred arXiv categories: {categories}")

        # 2. Build query
        query = self._build_arxiv_query(
            topic=topic,
            keywords=keywords,
            categories=categories
        )
        logger.info(f"arXiv query: {query}")

        # 3. Set time range
        # Set time range from 1995 to 2022
        start_date = datetime(1995, 1, 1, tzinfo=timezone.utc)
        end_date = datetime.now(timezone.utc) # Until now
        logger.info(f"Time range: >= {start_date.strftime('%Y-%m-%d')} to <= {end_date.strftime('%Y-%m-%d')}")

        # 4. Execute retrieval
        search = arxiv.Search(
            query=query,
            max_results=self.max_results_per_query,
            sort_by=sort_by,
            sort_order=arxiv.SortOrder.Descending
        )

        papers = []
        try:
            # Explicitly convert to list, catch network exceptions
            results = list(self.client.results(search))
            logger.info(f"  Successfully retrieved {len(results)} raw results")
        except Exception as e:
            logger.error(f"  ❌ arXiv API request failed: {e}")
            logger.warning("  Tip: Accessing arXiv from China may require a proxy, or try again later")
            return []

        for result in results:
            # Time filtering
            if result.published < start_date or result.published > end_date:
                continue

            # Convert to standard format
            paper = self._parse_arxiv_result(result)

            # Relevance filtering
            relevance_score = self._compute_relevance(paper, topic, keywords)
            paper['relevance_score'] = relevance_score

            if relevance_score >= self.min_relevance_score:
                papers.append(paper)
                logger.info(
                    f"  ✓ [{paper['year']}] {paper['title'][:60]}... "
                    f"(relevance: {relevance_score:.2f})"
                )

            if len(papers) >= max_seeds:
                break

        # 5. Sort by relevance
        papers.sort(key=lambda x: x['relevance_score'], reverse=True)

        logger.info(f"✅ Retrieved {len(papers)} high-quality seed papers")
        return papers[:max_seeds]

    def _infer_categories(self, topic: str) -> List[str]:
        """
        Infer arXiv categories based on topic

        Args:
            topic: Topic name

        Returns:
            Inferred category list
        """
        topic_lower = topic.lower()

        # Try to match predefined mapping
        for key, cats in ARXIV_CATEGORY_MAP.items():
            if key.lower() in topic_lower:
                return cats

        # Keyword matching
        if any(kw in topic_lower for kw in ["nlp", "language", "text", "translation"]):
            return ["cs.CL"]
        elif any(kw in topic_lower for kw in ["vision", "image", "video"]):
            return ["cs.CV"]
        elif any(kw in topic_lower for kw in ["learning", "neural", "deep"]):
            return ["cs.LG"]
        elif any(kw in topic_lower for kw in ["ai", "intelligence", "agent"]):
            return ["cs.AI"]

        # Default to general CS categories
        return ["cs.AI", "cs.LG"]

    def _build_arxiv_query(
        self,
        topic: str,
        keywords: Optional[List[str]],
        categories: List[str]
    ) -> str:
        """
        Build arXiv query string

        If LLM is enabled and client is available, use LLM to generate query
        Otherwise use traditional rule-based method

        Args:
            topic: Topic
            keywords: Keyword list
            categories: arXiv category list

        Returns:
            Query string
        """
        # Try to use LLM to generate query
        if self.use_llm_query_generation and self.llm_client:
            try:
                llm_query = self._generate_query_with_llm(topic, keywords, categories)
                if llm_query:
                    logger.info(f"✨ Using LLM-generated query: {llm_query}")
                    return llm_query
            except Exception as e:
                logger.warning(f"LLM query generation failed, falling back to traditional method: {e}")

        # Traditional rule-based method
        return self._build_arxiv_query_traditional(topic, keywords, categories)

    def _expand_semantic_concepts(
        self,
        topic: str,
        keywords: Optional[List[str]]
    ) -> Dict:
        """
        Phase 1: Semantic expansion
        Use LLM as domain expert to expand related concepts, synonyms, and subfields

        Args:
            topic: Research topic
            keywords: Keyword list (optional)

        Returns:
            Expanded concept dictionary, format:
            {
                'expanded_topics': [...],     # Related topics
                'expanded_keywords': [...],   # Expanded keywords
                'synonyms': [...],            # Synonyms
                'subfields': [...]            # Subfields
            }
        """
        logger.info(f"🔍 [Phase 1] Semantic expansion: topic='{topic}'")

        system_prompt = """You are a domain expert in computer science research.
Your task is to expand research topics and keywords by providing semantically
related concepts, synonyms, subfields, and alternative terminology.

Focus on:
- Computer Science, AI, and Machine Learning domains
- Academic and technical terminology
- Both broad and specific related concepts"""

        user_prompt = f"""Research Topic: {topic}
Current Keywords: {', '.join(keywords) if keywords else 'None'}

Please expand this research area by providing:
1. Related Topics: 2-{self.expansion_max_topics} semantically similar or overlapping research topics
2. Expanded Keywords: 5-{self.expansion_max_keywords} additional relevant technical terms, methods, or concepts
3. Synonyms: 2-4 alternative terms or abbreviations for the main topic
4. Subfields: 2-3 more specific subfields or applications within this area

Important:
- Focus on computer science and AI-related terms
- Use technical/academic terminology
- Keep each item concise (1-5 words)
- Avoid generic terms like "research", "study", "analysis"

Output ONLY valid JSON in this exact format (no markdown, no code blocks):
{{
  "expanded_topics": ["topic1", "topic2", ...],
  "expanded_keywords": ["keyword1", "keyword2", ...],
  "synonyms": ["synonym1", "synonym2", ...],
  "subfields": ["subfield1", "subfield2", ...]
}}"""

        try:
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=500
            )

            # Clean response (remove possible markdown code block markers)
            response = response.strip()
            if response.startswith('```'):
                # Remove ```json and ```
                lines = response.split('\n')
                response = '\n'.join(lines[1:-1]) if len(lines) > 2 else response
            response = response.strip()

            # Parse JSON
            expanded = json.loads(response)

            # Validate and limit quantities
            expanded_topics = expanded.get('expanded_topics', [])[:self.expansion_max_topics]
            expanded_keywords = expanded.get('expanded_keywords', [])[:self.expansion_max_keywords]
            synonyms = expanded.get('synonyms', [])[:4]
            subfields = expanded.get('subfields', [])[:3]

            result = {
                'expanded_topics': expanded_topics,
                'expanded_keywords': expanded_keywords,
                'synonyms': synonyms,
                'subfields': subfields
            }

            # Output expansion results
            logger.info(f"  ✅ Semantic expansion successful:")
            logger.info(f"    - Related topics({len(expanded_topics)}): {', '.join(expanded_topics[:3])}{'...' if len(expanded_topics) > 3 else ''}")
            logger.info(f"    - Expanded keywords({len(expanded_keywords)}): {', '.join(expanded_keywords[:5])}{'...' if len(expanded_keywords) > 5 else ''}")
            logger.info(f"    - Synonyms({len(synonyms)}): {', '.join(synonyms)}")
            logger.info(f"    - Subfields({len(subfields)}): {', '.join(subfields)}")

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"  ⚠️ JSON parsing failed: {e}")
            logger.warning(f"  LLM raw response: {response[:200]}...")
            return {}
        except Exception as e:
            logger.warning(f"  ⚠️ Semantic expansion failed: {e}")
            return {}

    def _generate_query_with_llm(
        self,
        topic: str,
        keywords: Optional[List[str]],
        categories: List[str]
    ) -> Optional[str]:
        """
        Use LLM to intelligently generate arXiv query string (two-phase method)

        If semantic expansion is enabled, execute:
          Phase 1: Semantic expansion - expand related concepts as domain expert
          Phase 2: Query construction - build precise query as librarian
        Otherwise, directly generate query (traditional single-phase method)

        Args:
            topic: Topic
            keywords: Keyword list
            categories: arXiv category list

        Returns:
            LLM-generated query string, None if failed
        """
        # Check if semantic expansion is enabled
        if self.enable_semantic_expansion:
            logger.info("\n" + "="*70)
            logger.info("🚀 Using two-phase LLM query generation (semantic expansion + query construction)")
            logger.info("="*70)

            # Phase 1: Semantic expansion
            try:
                expanded = self._expand_semantic_concepts(topic, keywords)
            except Exception as e:
                logger.warning(f"⚠️ Semantic expansion failed: {e}, using original input")
                expanded = {}

            # Merge original input and expansion results
            all_topics = [topic]
            if expanded:
                all_topics.extend(expanded.get('expanded_topics', []))
                all_topics.extend(expanded.get('synonyms', []))

            all_keywords = list(keywords) if keywords else []
            if expanded:
                all_keywords.extend(expanded.get('expanded_keywords', []))

            logger.info(f"\n📦 Merged results: {len(all_topics)} topics, {len(all_keywords)} keywords")

            # Phase 2: Query construction
            try:
                query = self._construct_arxiv_query_with_llm(
                    original_topic=topic,
                    all_topics=all_topics,
                    all_keywords=all_keywords,
                    categories=categories
                )
                if query:
                    logger.info("="*70 + "\n")
                    return query
            except Exception as e:
                logger.warning(f"⚠️ Query construction failed: {e}")

            logger.info("="*70 + "\n")
            return None

        else:
            # Traditional single-phase method (original logic)
            logger.info("💡 Using single-phase LLM query generation (traditional method)")

            # Build prompt
            system_prompt = """You are an expert at constructing arXiv API queries.
Your task is to generate effective search queries that will find relevant academic papers.

arXiv Query Syntax Rules:
1. Categories: Use "cat:cs.AI" or "cat:cs.LG" format
2. Title search: Use "ti:keyword" (without quotes for flexible matching)
3. Abstract search: Use "abs:keyword" (without quotes for flexible matching)
4. Boolean operators: AND, OR, ANDNOT
5. Parentheses for grouping: (cat:cs.AI OR cat:cs.LG)
6. For multi-word phrases: Break into key terms or use without quotes for flexible matching

Important Tips:
- Avoid overly strict exact phrase matching (don't use quotes for long phrases)
- Extract core keywords from long phrases
- Use OR to connect related terms
- Balance between precision and recall"""

            user_prompt = f"""Generate an arXiv search query for the following research topic:

Topic: {topic}
Additional Keywords: {', '.join(keywords) if keywords else 'None'}
Target Categories: {', '.join(categories)}

Requirements:
1. Include category filters: {' OR '.join([f'cat:{cat}' for cat in categories])}
2. Extract 3-5 core keywords from the topic (ignore stopwords like 'for', 'the', 'with')
3. Use flexible matching (no quotes for multi-word terms)
4. Connect keywords with OR for broader coverage
5. Use AND to combine category filters with keyword filters

Output ONLY the final query string, no explanation."""

            try:
                response = self.llm_client.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.3,
                    max_tokens=200
                )

                # Clean response (remove possible extra text)
                query = response.strip()
                # Remove possible markdown code block markers
                if query.startswith('```'):
                    query = '\n'.join(query.split('\n')[1:-1])
                query = query.strip()

                return query

            except Exception as e:
                logger.error(f"LLM query generation error: {e}")
                return None

    def _construct_arxiv_query_with_llm(
        self,
        original_topic: str,
        all_topics: List[str],
        all_keywords: List[str],
        categories: List[str]
    ) -> Optional[str]:
        """
        [Optimized version] Phase 2:
        1. LLM selects 3-5 most critical search terms (phrases)
        2. Python code automatically wraps them in (ti:"..." OR abs:"...") format
        This avoids LLM generating syntax errors or missing abs tags
        """

        # 1. Build Prompt, only require returning keyword list
        system_prompt = """You are an expert arXiv search optimizer.
Your task is to select the 3-5 MOST CRITICAL search terms from a list of candidates.
Select terms that will maximize the retrieval of high-quality papers.

Rules:
1. Include the full topic name (e.g., "Natural Language Processing").
2. Include the most common acronym (e.g., "NLP").
3. Include 1-2 core technical synonyms (e.g., "Computational Linguistics").
4. Output specific phrases, not generic words.
5. Return ONLY a Python-style list of strings."""

        user_prompt = f"""Task: Select search terms for arXiv.

Original Topic: {original_topic}
Candidates: {', '.join((all_topics + all_keywords)[:20])}

Output Format: ["term1", "term2", "term3"]
Output ONLY the list."""

        try:
            # 2. Call LLM
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.1, # Reduce randomness, want the most accurate
                max_tokens=100
            )

            # 3. Parse LLM returned list string
            import ast
            cleaned_response = response.strip()
            # Handle possible markdown markers
            if "```" in cleaned_response:
                cleaned_response = cleaned_response.split("```")[1].replace("json", "").replace("python", "").strip()

            try:
                # Safely convert string to list
                selected_terms = ast.literal_eval(cleaned_response)
                if not isinstance(selected_terms, list):
                    raise ValueError("Output is not a list")
            except:
                # Fallback: if parsing fails, simply split by comma, or directly use original Topic
                logger.warning(f"Failed to parse LLM list: {cleaned_response}, falling back to original Topic")
                selected_terms = [original_topic]

            logger.info(f"🧠 LLM selected core terms: {selected_terms}")

            # 4. [Key step] Python is responsible for strict syntax construction
            # Format: (cat:...) AND ((ti:"A" OR abs:"A") OR (ti:"B" OR abs:"B")...)

            # 4.1 Build category part
            cat_part = " OR ".join([f"cat:{cat}" for cat in categories])

            # 4.2 Build content part (automatically add quotes and dual-field retrieval for each term)
            content_parts = []
            for term in selected_terms:
                term = term.strip()
                if not term: continue
                # Force add quotes, handle special characters
                safe_term = f'"{term}"'
                # Generate (ti:"term" OR abs:"term")
                part = f'(ti:{safe_term} OR abs:{safe_term})'
                content_parts.append(part)

            if not content_parts:
                content_parts = [f'(ti:"{original_topic}" OR abs:"{original_topic}")']

            content_query = " OR ".join(content_parts)

            # 5. Combine final query
            final_query = f"({cat_part}) AND ({content_query})"

            logger.info(f"✅ Python assembled query successfully: {final_query}")
            return final_query

        except Exception as e:
            logger.error(f"❌ Query construction process error: {e}")
            return None

    def _build_arxiv_query_traditional(
        self,
        topic: str,
        keywords: Optional[List[str]],
        categories: List[str]
    ) -> str:
        """
        Traditional rule-based method to build arXiv query string

        Optimization strategy:
        1. Remove quotes, use loose matching (ti:keyword instead of ti:"keyword")
        2. Use OR to connect topic and keywords (improve recall)
        3. Rely on subsequent _compute_relevance for fine filtering

        Args:
            topic: Topic
            keywords: Keyword list
            categories: arXiv category list

        Returns:
            Query string
        """
        query_parts = []

        # Add category constraints
        if categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            query_parts.append(f"({cat_query})")

        # Build content query: topic OR keywords (loose matching)
        content_parts = []

        # Add topic (no quotes, loose matching)
        if topic:
            content_parts.append(f"ti:{topic}")
            content_parts.append(f"abs:{topic}")

        # Add keywords (no quotes, loose matching)
        if keywords:
            for kw in keywords:
                content_parts.append(f"ti:{kw}")
                content_parts.append(f"abs:{kw}")

        # Use OR to connect all content parts
        if content_parts:
            content_query = " OR ".join(content_parts)
            query_parts.append(f"({content_query})")

        # Use AND to connect categories and keywords
        return " AND ".join(query_parts)

    def _parse_arxiv_result(self, result: arxiv.Result) -> Dict:
        """
        Parse arXiv result to standard format

        Args:
            result: arxiv.Result object

        Returns:
            Standard paper dictionary
        """
        return {
            'arxiv_id': result.get_short_id(),
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'abstract': result.summary,
            'year': result.published.year,
            'published_date': result.published,
            'updated_date': result.updated,
            'categories': result.categories,
            'primary_category': result.primary_category,
            'pdf_url': result.pdf_url,
            'doi': result.doi,
            'comment': result.comment,
            'journal_ref': result.journal_ref,
            # For subsequent mapping
            'source': 'arxiv',
            'openalex_id': None  # To be mapped
        }

    def _compute_relevance(
        self,
        paper: Dict,
        topic: str,
        keywords: Optional[List[str]]
    ) -> float:
        """
        Calculate paper relevance score to topic

        Args:
            paper: Paper data
            topic: Topic
            keywords: Keyword list

        Returns:
            Relevance score (0-1)
        """
        text = (paper['title'] + ' ' + paper['abstract']).lower()
        topic_lower = topic.lower()

        score = 0.0

        # 1. Topic matching (weight: 0.4)
        # Improvement: split topic into words, calculate vocabulary coverage
        topic_words = [w for w in topic_lower.split() if len(w) > 2]  # Filter short words
        if topic_words:
            matched_topic_words = sum(1 for word in topic_words if word in text)
            topic_coverage = matched_topic_words / len(topic_words)
            score += 0.4 * topic_coverage
        else:
            # If topic is empty or too short, check complete match
            if topic_lower in text:
                score += 0.4

        # 2. Keyword matching (weight: 0.4)
        if keywords:
            matched_keywords = sum(1 for kw in keywords if kw.lower() in text)
            score += 0.4 * (matched_keywords / len(keywords))

        # 3. Category matching (weight: 0.2)
        primary_cat = paper.get('primary_category', '')
        if primary_cat.startswith('cs.'):
            score += 0.2

        return min(score, 1.0)

    def retrieve_recent_papers(
        self,
        topic: str,
        keywords: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        max_results: int = 20,
        months_back: int = 12
    ) -> List[Dict]:
        """
        Retrieve latest frontier papers (for step 4: supplement SOTA)

        Args:
            topic: Topic
            keywords: Keyword list
            categories: arXiv category list
            max_results: Maximum number of results
            months_back: Number of months to look back

        Returns:
            List of latest papers
        """
        logger.info(f"Retrieving latest papers: topic='{topic}', looking back {months_back} months")

        # Automatically infer categories
        if not categories:
            categories = self._infer_categories(topic)

        # Build query
        query = self._build_arxiv_query(topic, keywords, categories)

        # Set time range
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=30 * months_back)

        # Execute retrieval (sorted by submission date)
        search = arxiv.Search(
            query=query,
            max_results=max_results * 2,  # Retrieve more, filter later
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )

        papers = []
        try:
            # Explicitly convert to list, catch network exceptions
            results = list(self.client.results(search))
            logger.info(f"  Successfully retrieved {len(results)} raw latest paper results")
        except Exception as e:
            logger.error(f"  ❌ arXiv API request failed: {e}")
            logger.warning("  Tip: Accessing arXiv from China may require a proxy, or try again later")
            return []

        for result in results:
            # Only want the latest
            if result.published < cutoff_date:
                continue

            paper = self._parse_arxiv_result(result)
            relevance_score = self._compute_relevance(paper, topic, keywords)
            paper['relevance_score'] = relevance_score

            if relevance_score >= self.min_relevance_score:
                papers.append(paper)
                logger.info(
                    f"  ✓ [{paper['published_date'].strftime('%Y-%m')}] "
                    f"{paper['title'][:60]}... (relevance: {relevance_score:.2f})"
                )

            if len(papers) >= max_results:
                break

        logger.info(f"✅ Retrieved {len(papers)} latest papers")
        return papers


if __name__ == "__main__":
    # Test code
    retriever = ArxivSeedRetriever(
        max_results_per_query=50,
        years_back=5,
        min_relevance_score=0.6
    )

    # Example 1: Retrieve NLP seed papers
    print("=" * 80)
    print("Example 1: Retrieve NLP seed papers")
    print("=" * 80)
    seeds = retriever.retrieve_seed_papers(
        topic="Natural Language Processing",
        keywords=["transformer", "attention", "language model"],
        max_seeds=10
    )

    for i, paper in enumerate(seeds[:5], 1):
        print(f"\n[{i}] {paper['title']}")
        print(f"    arXiv ID: {paper['arxiv_id']}")
        print(f"    Year: {paper['year']}")
        print(f"    Category: {paper['primary_category']}")
        print(f"    Relevance: {paper['relevance_score']:.2f}")

    # Example 2: Retrieve latest papers
    print("\n" + "=" * 80)
    print("Example 2: Retrieve latest papers (last 6 months)")
    print("=" * 80)
    recent = retriever.retrieve_recent_papers(
        topic="Large Language Models",
        keywords=["reasoning", "chain of thought"],
        max_results=10,
        months_back=6
    )

    for i, paper in enumerate(recent[:5], 1):
        print(f"\n[{i}] {paper['title']}")
        print(f"    arXiv ID: {paper['arxiv_id']}")
        print(f"    Published date: {paper['published_date'].strftime('%Y-%m-%d')}")
        print(f"    Relevance: {paper['relevance_score']:.2f}")
