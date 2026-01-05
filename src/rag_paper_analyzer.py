"""
RAG-based Paper Analyzer
Uses Retrieval-Augmented Generation technology to extract structured information from PDF full text
"""

import re
import os
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    SentenceTransformer = None
    np = None
    cosine_similarity = None

try:
    from modelscope.hub.snapshot_download import snapshot_download
except ImportError:
    snapshot_download = None

logger = logging.getLogger(__name__)


@dataclass
class PaperSection:
    """Paper section data structure"""
    title: str
    content: str
    page_num: int
    section_type: str  # 'abstract', 'introduction', 'method', 'results', 'discussion', 'conclusion', 'references'


@dataclass
class ExtractionQuery:
    """Information extraction query"""
    query_text: str
    target_sections: List[str]  # Priority search section types
    keywords: List[str]  # Keywords
    max_results: int = 3


class RAGPaperAnalyzer:
    """
    RAG-based Paper Analyzer
    Intelligently extracts key information from papers through semantic retrieval and section recognition
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_modelscope: bool = True):
        """
        Initialize RAG paper analyzer

        Args:
            model_name: Embedding model name to use
            use_modelscope: Whether to use ModelScope to download model (default True)
        """
        self.model_name = model_name
        self.embedder = None
        self.use_modelscope = use_modelscope

        # Check dependencies
        if SentenceTransformer is None:
            logger.warning("sentence-transformers not installed, will use rule-based method")
            self.use_embeddings = False
        else:
            try:
                logger.info(f"Loading embedding model: {model_name}")

                # Check local model path first
                local_model_path = self._get_local_model_path(model_name)
                if local_model_path and os.path.exists(local_model_path):
                    logger.info(f"  Using local model: {local_model_path}")
                    self.embedder = SentenceTransformer(local_model_path)
                # Use ModelScope to download model
                elif self.use_modelscope and snapshot_download is not None:
                    try:
                        logger.info("  Using ModelScope to download model...")
                        model_dir = snapshot_download(
                            f'sentence-transformers/{model_name}',
                            cache_dir='./model',
                            revision='master'
                        )
                        logger.info(f"  Model downloaded to: {model_dir}")
                        self.embedder = SentenceTransformer(model_dir)
                    except Exception as e:
                        logger.warning(f"  ModelScope download failed: {e}, trying direct load...")
                        # Fallback to direct load
                        self.embedder = SentenceTransformer(model_name)
                else:
                    # Use HuggingFace directly (if ModelScope unavailable)
                    if self.use_modelscope and snapshot_download is None:
                        logger.warning("  modelscope not installed, using HuggingFace to download model")
                    self.embedder = SentenceTransformer(model_name)

                self.use_embeddings = True
                logger.info("RAG mode enabled")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}, will use rule-based method")
                self.use_embeddings = False

        # Initialize patterns and queries
        self._initialize_patterns()

        logger.info("RAG paper analyzer initialization complete")

    def _get_local_model_path(self, model_name: str) -> Optional[str]:
        """
        Check if local model path exists

        Args:
            model_name: Model name, e.g., 'all-MiniLM-L6-v2'

        Returns:
            Local model path, or None if it doesn't exist
        """
        # Try multiple possible local paths
        possible_paths = [
            # Path relative to current file
            Path(__file__).parent.parent / "model" / "sentence-transformers" / model_name,
            # Path relative to project root
            Path(__file__).parent.parent.parent / "KGdemo" / "model" / "sentence-transformers" / model_name,
            # Absolute path
            Path("/home/lexy/下载/CLwithRAG/KGdemo/model/sentence-transformers") / model_name,
        ]
        
        for path in possible_paths:
            if path.exists() and (path / "modules.json").exists():
                return str(path)
        
        return None

    def _initialize_patterns(self):
        """Initialize section patterns and extraction queries"""
        # Define section recognition patterns
        self.section_patterns = {
            'abstract': [
                r'^abstract\s*$',
                r'^summary\s*$',
            ],
            'introduction': [
                r'^1\.?\s*introduction',
                r'^introduction\s*$',
                r'^1\.?\s*background',
            ],
            'related_work': [
                r'^2\.?\s*related\s+work',
                r'^2\.?\s*background',
                r'^literature\s+review',
            ],
            'method': [
                r'^\d+\.?\s*method',
                r'^\d+\.?\s*approach',
                r'^\d+\.?\s*model',
                r'^\d+\.?\s*algorithm',
                r'^\d+\.?\s*framework',
                r'^\d+\.?\s*architecture',
            ],
            'experiment': [
                r'^\d+\.?\s*experiment',
                r'^\d+\.?\s*evaluation',
                r'^\d+\.?\s*results',
            ],
            'discussion': [
                r'^\d+\.?\s*discussion',
                r'^\d+\.?\s*analysis',
            ],
            'limitation': [
                r'^\d+\.?\s*limitation',
                r'^\d+\.?\s*weakness',
                r'^\d+\.?\s*threat',
            ],
            'conclusion': [
                r'^\d+\.?\s*conclusion',
                r'^\d+\.?\s*summary',
                r'^conclusion\s*$',
            ],
            'future_work': [
                r'^\d+\.?\s*future\s+work',
                r'^\d+\.?\s*future\s+direction',
                r'^\d+\.?\s*outlook',
            ],
            'references': [
                r'^references\s*$',
                r'^bibliography\s*$',
            ],
        }

        # Define extraction queries
        self.extraction_queries = {
            'problem': ExtractionQuery(
                query_text="What is the main problem or challenge this paper addresses?",
                target_sections=['abstract', 'introduction', 'related_work'],
                keywords=['problem', 'challenge', 'issue', 'gap', 'limitation', 'difficult'],
                max_results=3
            ),
            'method': ExtractionQuery(
                query_text="What are the main contributions and novel methods proposed?",
                target_sections=['abstract', 'introduction', 'method', 'conclusion'],
                keywords=['propose', 'contribution', 'novel', 'method', 'approach', 'introduce'],
                max_results=3
            ),
            'limitation': ExtractionQuery(
                query_text="What are the limitations or weaknesses discussed?",
                target_sections=['limitation', 'discussion', 'conclusion'],
                keywords=['limitation', 'weakness', 'drawback', 'constraint', 'future work'],
                max_results=3
            ),
            'future_work': ExtractionQuery(
                query_text="What future work or directions are suggested?",
                target_sections=['future_work', 'conclusion', 'discussion'],
                keywords=['future', 'next', 'further', 'extension', 'improve', 'enhance'],
                max_results=2
            ),
        }

        logger.info("RAG paper analyzer initialization complete")

    def analyze_paper(self, paper: Dict, pdf_path: Optional[str] = None) -> Dict:
        """
        Analyze paper and extract deep information

        Automatically extracts four fields: Problem, Contribution, Limitation, Future Work
        If section splitting fails or target sections are not found, automatically uses abstract

        Args:
            paper: Basic paper information
            pdf_path: PDF file path

        Returns:
            Paper dictionary containing analysis results
        """
        paper_id = paper.get('id', 'unknown')
        logger.info(f"Starting RAG analysis of paper: {paper_id}")

        # Extract PDF content and identify sections
        sections = []
        if pdf_path and Path(pdf_path).exists():
            sections = self._extract_sections_from_pdf(pdf_path)
            if sections:
                logger.info(f"Extracted {len(sections)} sections from PDF")
            else:
                logger.warning(f"PDF section extraction failed, falling back to abstract")
                sections = self._create_sections_from_abstract(paper)
        else:
            logger.info("PDF does not exist, building sections from abstract")
            sections = self._create_sections_from_abstract(paper)

        # If there's no abstract either, create a minimal section based on title
        if not sections:
            logger.warning("No abstract available, analyzing using title only")
            if paper.get('title'):
                sections = [PaperSection(
                    title='Title',
                    content=paper['title'],
                    page_num=0,
                    section_type='title'
                )]

        # If using embeddings, pre-compute section vectors
        section_embeddings = None
        if self.use_embeddings and sections and self.embedder:
            try:
                section_texts = [f"{s.title} {s.content}" for s in sections]
                section_embeddings = self.embedder.encode(section_texts)
                logger.info("Section vectorization complete")
            except Exception as e:
                logger.warning(f"Section vectorization failed: {e}, will use keyword retrieval")
                section_embeddings = None

        # Execute RAG retrieval and information extraction
        # Automatically extract all four fields
        analysis_result = {}
        extraction_fields = ['problem', 'method', 'limitation', 'future_work']

        for field in extraction_fields:
            if field in self.extraction_queries:
                logger.info(f"Extracting {field}...")
                try:
                    analysis_result[field] = self._extract_with_rag(
                        sections, section_embeddings, self.extraction_queries[field]
                    )
                except Exception as e:
                    logger.error(f"Failed to extract {field}: {e}")
                    analysis_result[field] = f"Extraction failed: {str(e)}"

        # Create enriched paper data
        enriched_paper = paper.copy()
        enriched_paper['rag_analysis'] = analysis_result
        enriched_paper['sections_extracted'] = len(sections)
        enriched_paper['analysis_method'] = 'rag' if self.use_embeddings else 'rule_based'

        logger.info(f"RAG paper analysis complete: {paper_id}")
        return enriched_paper

    def _extract_sections_from_pdf(self, pdf_path: str) -> List[PaperSection]:
        """Extract and identify sections from PDF"""
        if PyPDF2 is None:
            logger.error("PyPDF2 not installed, cannot extract PDF content")
            return []

        sections = []

        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)

                logger.info(f"PDF total pages: {total_pages}")

                # Extract text page by page
                full_text = ""
                page_texts = []

                for page_num in range(total_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        page_texts.append((page_num, page_text))
                        full_text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num}: {e}")
                        continue

                # Identify sections
                sections = self._identify_sections(full_text, page_texts)

                logger.info(f"Identified {len(sections)} sections")
                return sections

        except Exception as e:
            logger.error(f"PDF processing failed {pdf_path}: {e}")
            return []

    def _identify_sections(self, full_text: str, page_texts: List[Tuple[int, str]]) -> List[PaperSection]:
        """Identify sections in text"""
        sections = []
        lines = full_text.split('\n')

        current_section = None
        current_content = []
        current_page = 0

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            if not line_stripped:
                continue

            # Check if it's a section title
            section_type = self._match_section_type(line_stripped)

            if section_type:
                # Save previous section
                if current_section:
                    content = '\n'.join(current_content).strip()
                    if content:
                        sections.append(PaperSection(
                            title=current_section,
                            content=content,
                            page_num=current_page,
                            section_type=section_type
                        ))

                # Start new section
                current_section = line_stripped
                current_content = []
            else:
                # Add to current section content
                if current_section:
                    current_content.append(line_stripped)

        # Save last section
        if current_section and current_content:
            content = '\n'.join(current_content).strip()
            if content:
                sections.append(PaperSection(
                    title=current_section,
                    content=content,
                    page_num=current_page,
                    section_type='other'
                ))

        return sections

    def _match_section_type(self, line: str) -> Optional[str]:
        """Match section type"""
        line_lower = line.lower().strip()

        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                if re.match(pattern, line_lower, re.IGNORECASE):
                    return section_type

        return None

    def _create_sections_from_abstract(self, paper: Dict) -> List[PaperSection]:
        """Create simple sections from abstract"""
        sections = []

        if paper.get('title'):
            sections.append(PaperSection(
                title='Title',
                content=paper['title'],
                page_num=0,
                section_type='title'
            ))

        if paper.get('abstract'):
            sections.append(PaperSection(
                title='Abstract',
                content=paper['abstract'],
                page_num=0,
                section_type='abstract'
            ))

        return sections

    def _extract_with_rag(
        self,
        sections: List[PaperSection],
        section_embeddings: Optional[np.ndarray],
        query: ExtractionQuery
    ) -> str:
        """
        Extract information using RAG method

        If target sections or keyword-matched sentences are not found, automatically falls back to using abstract
        """

        if not sections:
            return "No content available"

        # Step 1: Filter target sections
        target_sections = [
            s for s in sections
            if s.section_type in query.target_sections
        ]

        # If target sections are empty, try using all sections
        if not target_sections:
            logger.warning(f"Target sections {query.target_sections} not found, using all available sections")
            target_sections = sections

        # Step 2: Filter sentences based on keywords
        relevant_sentences = []

        for section in target_sections:
            sentences = self._split_into_sentences(section.content)

            for sentence in sentences:
                # Check keyword matching
                keyword_count = sum(
                    1 for keyword in query.keywords
                    if keyword.lower() in sentence.lower()
                )

                if keyword_count > 0:
                    relevant_sentences.append({
                        'text': sentence,
                        'section': section.title,
                        'keyword_count': keyword_count
                    })

        # Step 3: If embeddings are enabled, use semantic retrieval
        if self.use_embeddings and relevant_sentences and self.embedder:
            # Vectorize candidate sentences
            sentence_texts = [s['text'] for s in relevant_sentences]
            sentence_embeddings = self.embedder.encode(sentence_texts)

            # Vectorize query
            query_embedding = self.embedder.encode([query.query_text])

            # Calculate similarity
            similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]

            # Add similarity scores
            for i, sent_dict in enumerate(relevant_sentences):
                sent_dict['similarity'] = similarities[i]

            # Comprehensive sorting: keyword count + semantic similarity
            relevant_sentences.sort(
                key=lambda x: x['keyword_count'] * 0.3 + x['similarity'] * 0.7,
                reverse=True
            )
        else:
            # Sort based on keywords only
            relevant_sentences.sort(key=lambda x: x['keyword_count'], reverse=True)

        # Step 4: Extract top-k results
        if not relevant_sentences:
            return "No relevant information found"

        top_sentences = relevant_sentences[:query.max_results]
        result_text = ' '.join([s['text'] for s in top_sentences])

        return result_text

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Use regular expression to split sentences
        sentences = re.split(r'[.!?]+\s+', text)

        # Filter out sentences that are too short
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        return sentences

    def batch_analyze_papers(
        self,
        papers: List[Dict],
        pdf_dir: Optional[str] = None
    ) -> List[Dict]:
        """Batch analyze papers"""
        logger.info(f"Starting batch RAG analysis of {len(papers)} papers")

        enriched_papers = []

        for i, paper in enumerate(papers):
            try:
                # Find corresponding PDF file
                pdf_path = None
                if pdf_dir:
                    paper_id = paper.get('id', '')
                    pdf_dir_path = Path(pdf_dir)

                    # Find matching PDF file
                    for pdf_file in pdf_dir_path.glob(f"{paper_id}*.pdf"):
                        pdf_path = str(pdf_file)
                        break

                enriched_paper = self.analyze_paper(paper, pdf_path)
                enriched_papers.append(enriched_paper)

                logger.info(f"Progress: {i+1}/{len(papers)}")

            except Exception as e:
                logger.error(f"Failed to analyze paper {paper.get('id', 'unknown')}: {e}")
                failed_paper = paper.copy()
                failed_paper['rag_analysis'] = {
                    'error': str(e)
                }
                enriched_papers.append(failed_paper)

        logger.info(f"Batch RAG analysis complete")
        return enriched_papers


if __name__ == "__main__":
    # Test code
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    analyzer = RAGPaperAnalyzer()

    # Test paper data
    test_paper = {
        'id': 'W2741809807',
        'title': 'Attention Is All You Need',
        'abstract': '''The dominant sequence transduction models are based on complex
        recurrent or convolutional neural networks. The problem is that these models
        are difficult to parallelize. We propose the Transformer, a model architecture
        eschewing recurrence and instead relying entirely on an attention mechanism
        to draw global dependencies between input and output. The main contribution
        is a novel attention-based architecture. However, the limitation is that it
        requires large amounts of training data. Future work includes applying this
        to other domains.''',
        'year': 2017,
    }

    # Test analysis
    result = analyzer.analyze_paper(test_paper)

    print("\n" + "="*80)
    print("RAG Analysis Results:")
    print("="*80)
    print(f"\nProblem:\n{result['rag_analysis']['problem']}\n")
    print(f"Contribution:\n{result['rag_analysis']['contribution']}\n")
    print(f"Limitation:\n{result['rag_analysis']['limitation']}\n")
    print(f"Future Work:\n{result['rag_analysis']['future_work']}\n")
    print(f"Analysis method: {result['analysis_method']}")
    print(f"Sections extracted: {result['sections_extracted']}")
    print("="*80)
