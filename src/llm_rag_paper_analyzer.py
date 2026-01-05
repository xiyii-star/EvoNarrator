"""
LLM-Enhanced RAG Paper Analyzer (Refactored Version)

Clear and easy-to-understand version:
- Uses independent LLM configuration manager
- Uses independent prompt manager
- Clear module separation
"""

import logging
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass

# PDF processing
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# Embedding model - sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# Numerical computation libraries - numpy and sklearn (independent import, not affected by sentence-transformers)
try:
    import numpy as np
except ImportError:
    np = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine_similarity = None

# For local model loading
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModel = None

# ModelScope (China mirror)
try:
    from modelscope.hub.snapshot_download import snapshot_download
except ImportError:
    snapshot_download = None

# Local modules
try:
    from llm_config import LLMClient, LLMConfig
    from prompt_manager import PromptManager
    from grobid_parser import GrobidPDFParser
except ImportError:
    # If direct import fails, try importing from src module
    from src.llm_config import LLMClient, LLMConfig
    from src.prompt_manager import PromptManager
    from src.grobid_parser import GrobidPDFParser

logger = logging.getLogger(__name__)


@dataclass
class PaperSection:
    """Paper section data structure"""
    title: str
    content: str
    page_num: int
    section_type: str


class LLMRAGPaperAnalyzer:
    """
    LLM-Enhanced RAG Paper Analyzer

    Main features:
    1. Extract paper sections from PDF or abstract
    2. Use RAG to retrieve relevant content
    3. Use LLM to generate high-quality analysis
    4. Automatically extract four key fields: Problem, Contribution, Limitation, Future Work
    """

    def __init__(
        self,
        llm_config_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_modelscope: bool = True,
        prompts_dir: str = "./prompts",
        max_context_length: int = 3000,
        grobid_url: Optional[str] = None,
        local_model_path: Optional[str] = None
    ):
        """
        Initialize analyzer

        Args:
            llm_config_path: LLM configuration file path
            embedding_model: Embedding model name
            use_modelscope: Whether to use ModelScope to download models
            prompts_dir: Prompts folder path
            max_context_length: LLM context maximum length
            grobid_url: GROBID service URL (optional, e.g. http://localhost:8070)
            local_model_path: Local model path (optional, e.g. ./model/sentence-transformers/all-MiniLM-L6-v2)
        """
        logger.info("="*60)
        logger.info("Initializing LLM RAG Paper Analyzer")
        logger.info("="*60)

        # Basic configuration
        self.embedding_model_name = embedding_model
        self.use_modelscope = use_modelscope
        self.max_context_length = max_context_length
        self.grobid_url = grobid_url
        self.local_model_path = local_model_path

        # Initialize GROBID parser (if URL provided)
        self.grobid_parser = None
        if grobid_url:
            self._init_grobid_parser()

        # Initialize embedding model
        self.embedder = None
        self.use_embeddings = False
        self._init_embedding_model()

        # Initialize LLM client
        self.llm_client = self._init_llm_client(llm_config_path)

        # Initialize prompt manager
        self.prompt_manager = PromptManager(prompts_dir)

        # Section recognition patterns
        self.section_patterns = self._get_section_patterns()

        # Fields to extract
        self.extraction_fields = ['problem', 'method', 'limitation', 'future_work']

        logger.info("="*60)
        logger.info("LLM RAG Paper Analyzer initialization completed")
        logger.info("="*60)

    def _init_embedding_model(self):
        """Initialize Embedding model"""
        # Check if sentence-transformers is installed
        if SentenceTransformer is None:
            logger.warning("sentence-transformers not installed, will use pure keyword retrieval")
            logger.warning("   Install command: pip install sentence-transformers")
            self.use_embeddings = False
            return

        # If no local model path, try auto-detection
        if not self.local_model_path:
            self.local_model_path = self._get_local_model_path(self.embedding_model_name)

        # Prioritize using local model path
        if self.local_model_path:
            try:
                import os
                if os.path.exists(self.local_model_path):
                    logger.info(f"  Detected local model: {self.local_model_path}")
                    logger.info(f"  Loading local Embedding model...")
                    self.embedder = SentenceTransformer(self.local_model_path)
                    self.use_embeddings = True
                    logger.info(f"  Local Embedding model loaded successfully!")
                    return
                else:
                    logger.warning(f"  Local model path does not exist: {self.local_model_path}")
            except Exception as e:
                logger.warning(f"  Local model loading failed: {e}, trying to download...")

        # If local model loading failed, try downloading

        try:
            logger.info(f"Loading Embedding model: {self.embedding_model_name}")

            # Use ModelScope mirror (faster in China)
            if self.use_modelscope and snapshot_download is not None:
                try:
                    logger.info("  Using ModelScope mirror...")
                    model_dir = snapshot_download(
                        f'sentence-transformers/{self.embedding_model_name}',
                        cache_dir='./model',
                        revision='master'
                    )
                    self.embedder = SentenceTransformer(model_dir)
                    logger.info(f"  Model downloaded from ModelScope: {model_dir}")
                except Exception as e:
                    logger.warning(f"  ModelScope download failed: {e}, trying HuggingFace...")
                    self.embedder = SentenceTransformer(self.embedding_model_name)
            else:
                self.embedder = SentenceTransformer(self.embedding_model_name)

            self.use_embeddings = True
            logger.info("  Embedding model loaded successfully")

        except Exception as e:
            logger.warning(f"  Embedding model loading failed: {e}, will use pure keyword retrieval")
            self.use_embeddings = False

    def _get_local_model_path(self, model_name: str) -> Optional[str]:
        """
        Check if local model path exists

        Args:
            model_name: Model name, e.g. 'all-MiniLM-L6-v2'

        Returns:
            Local model path, or None if not exists
        """
        import os
        from pathlib import Path

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

    def _init_grobid_parser(self):
        """Initialize GROBID parser"""
        try:
            logger.info(f"Initializing GROBID parser: {self.grobid_url}")
            self.grobid_parser = GrobidPDFParser(self.grobid_url)
            logger.info("GROBID parser enabled")
        except Exception as e:
            logger.warning(f"GROBID parser initialization failed: {e}, will use regex method")
            self.grobid_parser = None

    def _init_llm_client(self, config_path: Optional[str]) -> Optional[LLMClient]:
        """Initialize LLM client"""
        # If config path is None, don't use LLM
        if config_path is None:
            logger.info("LLM config path is None, skipping LLM initialization (will use basic analysis mode)")
            return None

        try:
            logger.info(f"Loading LLM config: {config_path}")
            config = LLMConfig.from_file(config_path)

            logger.info(f"  Provider: {config.provider}")
            logger.info(f"  Model: {config.model}")

            client = LLMClient(config)
            return client

        except FileNotFoundError:
            logger.warning(f"LLM config file not found: {config_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            return None

    def _get_section_patterns(self) -> Dict[str, List[str]]:
        """Define regex patterns for section recognition"""
        import re
        return {
            'abstract': [
                r'^abstract\s*$',
                r'^summary\s*$',
            ],
            'introduction': [
                r'^1\.?\s*introduction',
                r'^introduction\s*$',
            ],
            'related_work': [
                r'^2\.?\s*related\s+work',
                r'^2\.?\s*background',
            ],
            'method': [
                r'^\d+\.?\s*method',
                r'^\d+\.?\s*approach',
                r'^\d+\.?\s*model',
            ],
            'experiment': [
                r'^\d+\.?\s*experiment',
                r'^\d+\.?\s*evaluation',
            ],
            'discussion': [
                r'^\d+\.?\s*discussion',
                r'^\d+\.?\s*analysis',
            ],
            'limitation': [
                r'^\d+\.?\s*limitation',
            ],
            'conclusion': [
                r'^\d+\.?\s*conclusion',
                r'^conclusion\s*$',
            ],
            'future_work': [
                r'^\d+\.?\s*future\s+work',
            ],
            'references': [
                r'^references\s*$',
            ],
        }

    # ========== Core Analysis Methods ==========

    def analyze_paper(self, paper: Dict, pdf_path: Optional[str] = None) -> Dict:
        """
        Analyze paper and extract key information

        Automatically extract four fields: Problem, Contribution, Limitation, Future Work
        Support multi-level fallback strategy: PDF → Abstract → Title

        Args:
            paper: Paper basic information dictionary
            pdf_path: PDF file path (optional)

        Returns:
            Enhanced paper dictionary with analysis results
        """
        paper_id = paper.get('id', 'unknown')
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting paper analysis: {paper_id}")
        logger.info(f"{'='*60}")

        # Step 1: Extract sections and determine if PDF extraction succeeded
        sections, pdf_extracted = self._extract_paper_sections(paper, pdf_path)

        # Determine if sections were successfully extracted from PDF
        if pdf_extracted:
            logger.info("  PDF extraction successful, using RAG retrieval mode")
        else:
            logger.info("  PDF not extracted, using abstract direct generation mode")

        # Step 2: Calculate section embeddings (only when PDF extraction succeeded)
        section_embeddings = None
        if pdf_extracted:
            section_embeddings = self._compute_section_embeddings(sections)

        # Step 3: Extract all fields (pass pdf_extracted flag)
        analysis_result = self._extract_all_fields(sections, section_embeddings, pdf_extracted, paper)

        # Step 4: Build result
        enriched_paper = paper.copy()
        enriched_paper['rag_analysis'] = analysis_result
        enriched_paper['sections_extracted'] = len(sections)
        enriched_paper['section_types'] = [s.section_type for s in sections]
        enriched_paper['pdf_extracted'] = pdf_extracted
        enriched_paper['analysis_method'] = f'llm_rag_{self.llm_client.config.provider if self.llm_client else "none"}'

        logger.info(f"Paper analysis completed: {paper_id}")
        logger.info(f"   Extracted fields: {len(analysis_result)}")
        logger.info(f"   Sections: {len(sections)}")
        logger.info(f"   Analysis mode: {'RAG retrieval' if pdf_extracted else 'Abstract direct generation'}")
        logger.info(f"{'='*60}\n")

        return enriched_paper

    def _extract_paper_sections(self, paper: Dict, pdf_path: Optional[str]) -> tuple[List[PaperSection], bool]:
        """
        Extract paper sections (support multi-level fallback)

        Fallback strategy:
        1. Try to extract sections from PDF
        2. If failed, use abstract to build sections
        3. If no abstract, use title

        Args:
            paper: Paper information
            pdf_path: PDF path

        Returns:
            (Section list, flag indicating if PDF extraction succeeded)
        """
        sections = []

        # Level 1: Try PDF extraction
        if pdf_path and Path(pdf_path).exists():
            logger.info("  [1/3] Trying to extract sections from PDF...")
            sections = self._extract_sections_from_pdf(pdf_path)

            if sections:
                logger.info(f"  Extracted {len(sections)} sections from PDF")
                return sections, True  # PDF extraction succeeded
            else:
                logger.warning("  PDF section extraction failed")

        # Level 2: Fallback to abstract
        logger.info("  [2/3] Falling back to abstract...")
        sections = self._create_sections_from_abstract(paper)

        if sections:
            logger.info(f"  Built {len(sections)} sections from abstract")
            return sections, False  # Using abstract, PDF not extracted

        # Level 3: Fallback to title
        logger.info("  [3/3] Falling back to title...")
        if paper.get('title'):
            sections = [PaperSection(
                title='Title',
                content=paper['title'],
                page_num=0,
                section_type='title'
            )]
            logger.info("  Using title as minimal content")

        return sections, False  # Using title or empty, PDF not extracted

    def _encode_texts(self, texts):
        """
        Unified text encoding interface
        Supports sentence-transformers and local transformers models
        """
        if self.embedder:
            # Use sentence-transformers
            return self.embedder.encode(texts)
        elif hasattr(self, 'tokenizer') and hasattr(self, 'model'):
            # Use local transformers model
            import torch

            # Mean Pooling - take average pooling
            def mean_pooling(model_output, attention_mask):
                token_embeddings = model_output[0]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            # Encode text
            encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)

            with torch.no_grad():
                model_output = self.model(**encoded_input)

            # Perform pooling
            embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            return embeddings.numpy()
        else:
            return None

    def _compute_section_embeddings(self, sections: List[PaperSection]) -> Optional[any]:
        """Calculate section embeddings"""
        if not self.use_embeddings or not sections:
            return None

        try:
            logger.info("  Calculating section embeddings...")
            section_texts = [f"{s.title} {s.content}" for s in sections]
            embeddings = self._encode_texts(section_texts)

            if embeddings is not None:
                logger.info(f"  Section embeddings calculated ({len(embeddings)} vectors)")
                return embeddings
            else:
                logger.warning("  Encoder not properly initialized")
                return None

        except Exception as e:
            logger.warning(f"  Section embeddings calculation failed: {e}")
            return None

    def _extract_all_fields(
        self,
        sections: List[PaperSection],
        section_embeddings: Optional[any],
        pdf_extracted: bool,
        paper: Dict
    ) -> Dict[str, str]:
        """
        Automatically extract all four fields

        Args:
            sections: Paper section list
            section_embeddings: Section embeddings (optional)
            pdf_extracted: Whether PDF extraction succeeded
            paper: Original paper information (for getting abstract)

        Returns:
            {field: extracted_value}
        """
        logger.info("  Starting key field extraction...")

        results = {}

        for field in self.extraction_fields:
            logger.info(f"     • Extracting {field}...")

            try:
                value = self._extract_single_field(field, sections, section_embeddings, pdf_extracted, paper)
                results[field] = value

            except Exception as e:
                logger.error(f"     Failed to extract {field}: {e}")
                results[field] = f"Extraction failed: {str(e)}"

        logger.info(f"  Field extraction completed, successfully extracted {len(results)} fields")
        return results

    def _extract_single_field(
        self,
        field: str,
        sections: List[PaperSection],
        section_embeddings: Optional[any],
        pdf_extracted: bool,
        paper: Dict
    ) -> str:
        """
        Extract single field

        Process:
        - If PDF extraction succeeded: Use RAG to retrieve relevant context → LLM generation
        - If PDF not extracted: Directly use abstract as context → LLM generation

        Args:
            field: Field name
            sections: Section list
            section_embeddings: Section embeddings
            pdf_extracted: Whether PDF extraction succeeded
            paper: Original paper information

        Returns:
            Extracted content
        """
        if not sections:
            return "No content available"

        # Choose different strategy based on whether PDF extraction succeeded
        if pdf_extracted:
            # Strategy 1: PDF extraction succeeded -> Use RAG retrieval
            logger.info(f"       Using RAG retrieval mode to extract {field}")
            relevant_context = self._retrieve_relevant_content(
                field, sections, section_embeddings
            )

            if not relevant_context or relevant_context == "No relevant information found":
                # RAG retrieval failed, fallback to abstract
                logger.warning(f"       RAG retrieval found no relevant information, falling back to abstract")
                relevant_context = self._get_abstract_context(paper)
        else:
            # Strategy 2: PDF not extracted -> Directly use abstract
            logger.info(f"       Using abstract direct generation mode to extract {field}")
            relevant_context = self._get_abstract_context(paper)

        if not relevant_context or relevant_context == "No abstract available":
            return "No relevant information found"

        # Use LLM to generate
        if self.llm_client:
            return self._generate_with_llm(field, relevant_context)
        else:
            logger.warning("     LLM not configured, returning raw retrieval content")
            return relevant_context[:200]  # Return first 200 characters of retrieval content

    def _get_abstract_context(self, paper: Dict) -> str:
        """
        Get abstract as context

        Args:
            paper: Paper information

        Returns:
            Abstract text
        """
        abstract = paper.get('abstract', '')
        title = paper.get('title', '')

        if not abstract:
            return "No abstract available"

        # Build context
        context = f"Title: {title}\n\nAbstract: {abstract}" if title else f"Abstract: {abstract}"
        return context

    # ========== RAG Retrieval ==========

    def _retrieve_relevant_content(
        self,
        field: str,
        sections: List[PaperSection],
        section_embeddings: Optional[any]
    ) -> str:
        """
        Retrieve content relevant to field (RAG core)

        Supports:
        - Target section filtering
        - Keyword retrieval
        - Semantic similarity ranking (if embeddings available)
        - Fallback to abstract (if retrieval fails)

        Args:
            field: Field name
            sections: Section list
            section_embeddings: Section embeddings

        Returns:
            Relevant content text
        """
        # Define target sections and keywords
        target_sections_map = {
            'problem': ['abstract', 'introduction'],
            'method': ['abstract', 'introduction', 'method', 'conclusion'],
            'limitation': ['limitation', 'discussion', 'conclusion'],
            'future_work': ['future_work', 'conclusion', 'discussion']
        }

        keywords_map = {
            'problem': ['problem', 'challenge', 'issue', 'gap', 'limitation'],
            'method': ['propose', 'contribution', 'novel', 'method', 'introduce'],
            'limitation': ['limitation', 'weakness', 'drawback', 'shortcoming'],
            'future_work': ['future', 'next', 'further', 'improve', 'explore']
        }

        target_section_types = target_sections_map.get(field, [])
        keywords = keywords_map.get(field, [])

        # Step 1: Filter target sections
        filtered_sections = [
            s for s in sections
            if s.section_type in target_section_types
        ] if target_section_types else sections

        if not filtered_sections:
            logger.info(f"       Target sections {target_section_types} not found, using all sections")
            filtered_sections = sections

        # Step 2: Keyword retrieval
        relevant_chunks = []

        for section in filtered_sections:
            paragraphs = self._split_into_paragraphs(section.content)

            for paragraph in paragraphs:
                # Calculate keyword match count
                keyword_count = sum(
                    1 for kw in keywords
                    if kw.lower() in paragraph.lower()
                )

                if keyword_count > 0:
                    relevant_chunks.append({
                        'text': paragraph,
                        'section': section.title,
                        'keyword_count': keyword_count
                    })

        # Step 3: If not found, fallback to abstract
        if not relevant_chunks:
            logger.info(f"       Keyword retrieval found no matches, falling back to abstract")
            abstract_sections = [s for s in sections if s.section_type == 'abstract']

            if abstract_sections:
                abstract_text = abstract_sections[0].content
                return f"[Abstract (Fallback)]\n{abstract_text[:self.max_context_length]}"
            else:
                # Use first two sections
                all_content = "\n\n".join([f"[{s.title}]\n{s.content}" for s in sections[:2]])
                return all_content[:self.max_context_length] if all_content else "No relevant information found"

        # Step 4: Sort (keyword or semantic similarity)
        if self.use_embeddings and section_embeddings is not None:
            # Use semantic similarity ranking
            query_text = f"extract {field} from paper"
            chunk_texts = [c['text'] for c in relevant_chunks]

            try:
                chunk_embeddings = self._encode_texts(chunk_texts)
                query_embedding = self._encode_texts([query_text])

                if chunk_embeddings is not None and query_embedding is not None:
                    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

                    for i, chunk in enumerate(relevant_chunks):
                        chunk['similarity'] = similarities[i]

                    # Combined ranking (keyword 30% + semantic 70%)
                    relevant_chunks.sort(
                        key=lambda x: x['keyword_count'] * 0.3 + x['similarity'] * 0.7,
                        reverse=True
                    )
                else:
                    # Encoding failed, fallback to keyword ranking
                    relevant_chunks.sort(key=lambda x: x['keyword_count'], reverse=True)
            except Exception as e:
                # Fallback to keyword ranking
                logger.warning(f"      Semantic similarity calculation failed: {e}, using keyword ranking")
                relevant_chunks.sort(key=lambda x: x['keyword_count'], reverse=True)
        else:
            # Only keyword-based ranking
            relevant_chunks.sort(key=lambda x: x['keyword_count'], reverse=True)

        # Step 5: Build context
        context_parts = []
        current_length = 0

        for chunk in relevant_chunks[:5]:  # Take top 5
            chunk_text = f"[{chunk['section']}]\n{chunk['text']}"
            chunk_length = len(chunk_text)

            if current_length + chunk_length > self.max_context_length:
                break

            context_parts.append(chunk_text)
            current_length += chunk_length

        return "\n\n".join(context_parts) if context_parts else "No relevant information found"

    # ========== LLM Generation ==========

    def _generate_with_llm(self, field: str, context: str) -> str:
        """
        Use LLM to generate analysis result

        Args:
            field: Field name
            context: Retrieved context

        Returns:
            LLM-generated analysis
        """
        if not self.llm_client:
            return "LLM not configured"

        # Build full prompt
        full_prompt = self.prompt_manager.build_full_prompt(field, context)

        # Get system prompt
        system_prompt = self.prompt_manager.get_system_prompt()

        # Call LLM
        result = self.llm_client.generate(
            prompt=full_prompt,
            system_prompt=system_prompt
        )

        return result

    # ========== PDF Processing ==========

    def _extract_sections_from_pdf(self, pdf_path: str) -> List[PaperSection]:
        """
        Extract sections from PDF or TXT file (hybrid strategy)

        Strategy:
        1. If .txt file, read text directly
        2. If PDF: Prioritize GROBID (if available)
        3. Fallback to PyPDF2+regex
        """
        # Check file extension
        file_ext = Path(pdf_path).suffix.lower()

        # Strategy 0: If .txt file, read directly
        if file_ext == '.txt':
            logger.info("  Detected .txt file, reading text directly...")
            return self._extract_sections_from_txt(pdf_path)

        # Strategy 1: Try GROBID (for PDF only)
        if self.grobid_parser:
            try:
                logger.info("  Trying to parse PDF with GROBID...")
                sections = self.grobid_parser.extract_sections_from_pdf(pdf_path)

                if sections:
                    logger.info(f"  GROBID successfully extracted {len(sections)} sections")
                    return sections
                else:
                    logger.warning("  GROBID extracted no sections, falling back to regex method")
            except Exception as e:
                logger.warning(f"  GROBID parsing failed: {e}, falling back to regex method")

        # Strategy 2: Fallback to PyPDF2+regex
        logger.info("  Using PyPDF2+regex to parse PDF...")
        return self._extract_sections_with_pypdf2(pdf_path)

    def _extract_sections_with_pypdf2(self, pdf_path: str) -> List[PaperSection]:
        """Extract sections using PyPDF2 (regex method)"""
        if PyPDF2 is None:
            logger.error("  PyPDF2 not installed, cannot extract PDF")
            return []

        sections = []

        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)

                full_text = ""
                for page_num in range(total_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        full_text += page_text + "\n"
                    except:
                        continue

                sections = self._identify_sections(full_text)

        except Exception as e:
            logger.error(f"  PDF processing failed: {e}")

        return sections

    def _extract_sections_from_txt(self, txt_path: str) -> List[PaperSection]:
        """Read directly from .txt file and extract sections"""
        sections = []

        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                full_text = file.read()

            logger.info(f"  Successfully read .txt file, total {len(full_text)} characters")

            # Use same section recognition logic
            sections = self._identify_sections(full_text)

            if sections:
                logger.info(f"  Identified {len(sections)} sections from .txt file")
            else:
                logger.warning("  No clear sections identified, treating entire text as one section")
                # If no sections identified, treat entire text as one section
                sections = [PaperSection(
                    title='Full Text',
                    content=full_text[:10000],  # Limit length
                    page_num=0,
                    section_type='other'
                )]

        except Exception as e:
            logger.error(f"  .txt file processing failed: {e}")

        return sections

    def _identify_sections(self, full_text: str) -> List[PaperSection]:
        """Identify sections in text"""
        import re

        sections = []
        lines = full_text.split('\n')

        current_section = None
        current_content = []
        current_type = 'other'

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Check if it's a section title
            section_type = self._match_section_type(line_stripped)

            if section_type:
                # Save previous section
                if current_section and current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        sections.append(PaperSection(
                            title=current_section,
                            content=content,
                            page_num=0,
                            section_type=current_type
                        ))

                # Start new section
                current_section = line_stripped
                current_content = []
                current_type = section_type
            else:
                # Add to current section
                if current_section:
                    current_content.append(line_stripped)

        # Save last section
        if current_section and current_content:
            content = '\n'.join(current_content).strip()
            if content:
                sections.append(PaperSection(
                    title=current_section,
                    content=content,
                    page_num=0,
                    section_type=current_type
                ))

        return sections

    def _match_section_type(self, line: str) -> Optional[str]:
        """Match section type"""
        import re

        line_lower = line.lower().strip()

        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                if re.match(pattern, line_lower, re.IGNORECASE):
                    return section_type

        return None

    def _create_sections_from_abstract(self, paper: Dict) -> List[PaperSection]:
        """Create sections from abstract"""
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

    # ========== Utility Methods ==========

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        import re

        paragraphs = re.split(r'\n\s*\n|\n', text)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 30]

        return paragraphs

    # ========== Batch Analysis ==========

    def batch_analyze_papers(
        self,
        papers: List[Dict],
        pdf_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Batch analyze papers

        Args:
            papers: Paper list
            pdf_dir: PDF folder path

        Returns:
            Enhanced paper list
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Batch analyzing {len(papers)} papers")
        logger.info(f"{'='*60}\n")

        enriched_papers = []

        for i, paper in enumerate(papers):
            try:
                # Find PDF file
                pdf_path = None
                if pdf_dir:
                    paper_id = paper.get('id', '')
                    pdf_dir_path = Path(pdf_dir)

                    for pdf_file in pdf_dir_path.glob(f"{paper_id}*.pdf"):
                        pdf_path = str(pdf_file)
                        break

                # Analyze paper
                enriched_paper = self.analyze_paper(paper, pdf_path)
                enriched_papers.append(enriched_paper)

                logger.info(f"Progress: {i+1}/{len(papers)}\n")

            except Exception as e:
                logger.error(f"Paper analysis failed {paper.get('id', 'unknown')}: {e}")

                # Add failed paper
                failed_paper = paper.copy()
                failed_paper['rag_analysis'] = {
                    'problem': f'Analysis failed: {str(e)}',
                    'method': f'Analysis failed: {str(e)}',
                    'limitation': f'Analysis failed: {str(e)}',
                    'future_work': f'Analysis failed: {str(e)}'
                }
                enriched_papers.append(failed_paper)

        logger.info(f"{'='*60}")
        logger.info(f"Batch analysis completed")
        logger.info(f"{'='*60}\n")

        return enriched_papers


if __name__ == "__main__":
    # Test code
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*60)
    print("Testing LLM RAG Paper Analyzer (Refactored Version)")
    print("="*60)

    # Test paper data
    test_paper = {
        'id': 'W2741809807',
        'title': 'Attention Is All You Need',
        'abstract': '''The dominant sequence transduction models are based on complex
        recurrent or convolutional neural networks. The problem is that these models
        are difficult to parallelize. We propose the Transformer, a model architecture
        eschewing recurrence and instead relying entirely on an attention mechanism.''',
        'year': 2017,
    }

    try:
        # Create analyzer
        analyzer = LLMRAGPaperAnalyzer(
            llm_config_path='../llm_config_ollama.json',
            prompts_dir='../prompts'
        )

        # Analyze paper
        result = analyzer.analyze_paper(test_paper)

        # Display results
        print("\n" + "="*60)
        print("Analysis Results:")
        print("="*60)
        for field, value in result['rag_analysis'].items():
            print(f"\n{field.upper()}:")
            print(value)

        print("\n" + "="*60)
        print("Test completed")
        print("="*60)

    except Exception as e:
        print(f"Test failed: {e}")
