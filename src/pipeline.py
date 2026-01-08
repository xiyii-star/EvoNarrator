"""
Main Pipeline
Coordinates the entire paper knowledge graph construction process
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional
import time
from datetime import datetime
import yaml


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

from llm_rag_paper_analyzer import LLMRAGPaperAnalyzer
from openalex_client import OpenAlexClient
from pdf_downloader import PDFDownloader
from knowledge_graph import CitationGraph
from citation_type_inferencer import CitationTypeInferencer
from llm_config import create_llm_client
from topic_evolution_analyzer import TopicEvolutionAnalyzer
from snowball_retrieval import SnowballRetrieval
from papersearch import PaperSearchPipeline
from deep_survey_analyzer import DeepSurveyAnalyzer

# Import research_idea_generator with survey.py (supports evolutionary paths)
import importlib.util
_research_idea_spec = importlib.util.spec_from_file_location(
    "research_idea_generator_with_survey",
    str(Path(__file__).parent / "research_idea_generator with survey.py")
)
if _research_idea_spec and _research_idea_spec.loader:
    _research_idea_module = importlib.util.module_from_spec(_research_idea_spec)
    _research_idea_spec.loader.exec_module(_research_idea_module)
    ResearchIdeaGenerator = _research_idea_module.ResearchIdeaGenerator
else:
    # Fallback to standard version
    from research_idea_generator import ResearchIdeaGenerator

# Import DeepPaper Multi-Agent system
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from DeepPaper_Agent import DeepPaperOrchestrator
from DeepPaper_Agent.data_structures import PaperDocument, PaperSection

# Import DeepPaper 2.0 Multi-Agent system (Enhanced version - Integrated Citation Detective)
import importlib.util
_dp2_spec = importlib.util.spec_from_file_location(
    "DeepPaper_Agent2",
    str(Path(__file__).parent.parent / "DeepPaper_Agent2.0" / "orchestrator.py")
)
if _dp2_spec and _dp2_spec.loader:
    _dp2_module = importlib.util.module_from_spec(_dp2_spec)
    _dp2_spec.loader.exec_module(_dp2_module)
    DeepPaper2Orchestrator = _dp2_module.DeepPaper2Orchestrator
else:
    DeepPaper2Orchestrator = None



# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_full_config(config_path: str = './config/config.yaml') -> Dict:
    """
    Load full configuration from YAML file

    Args:
        config_path: Path to configuration file

    Returns:
        Full configuration dictionary
    """
    config_file = Path(config_path)

    if not config_file.exists():
        logger.warning(f"Config file does not exist: {config_path}, using default configuration")
        return {}

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded config file: {config_path}")
        return config if config else {}
    except Exception as e:
        logger.error(f"Failed to load config file: {e}, using default configuration")
        return {}


class PaperGraphPipeline:
    """
    Paper Knowledge Graph Construction Pipeline
    Integrates search, download, analysis, and graph construction in a complete workflow
    """

    def __init__(self, config: Dict = None):
        """
        Initialize pipeline

        Args:
            config: Configuration parameter dictionary
        """
        # Default configuration
        self.config = {
            'max_papers': 20,          # Maximum number of papers
            'max_citations': 3,        # Maximum citations per paper
            'max_references': 3,       # Maximum references per paper
            'max_total_papers': 100,   # Total paper limit (including extended citations)
            'min_citation_count': 10,  # Minimum citation count filter
            'download_pdfs': True,     # Whether to download PDFs
            'max_pdf_downloads': 5,    # Maximum PDF downloads
            'output_dir': './output',  # Output directory
            'data_dir': './data',      # Data directory
            'llm_config_file': './config/config.yaml',  # Unified config file path
            'grobid_url': None,        # GROBID service URL (e.g., http://localhost:8070)
            'save_stage_outputs': True,  # Whether to save outputs for each stage
        }

        if config:
            self.config.update(config)

        # Create output directories
        self.output_dir = Path(self.config['output_dir'])
        self.data_dir = Path(self.config['data_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Create stage output directory
        self.stage_output_dir = self.output_dir / 'stage_outputs'
        self.stage_output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize modules
        self.openalex_client = OpenAlexClient()
        self.pdf_downloader = PDFDownloader(
            download_dir=str(self.data_dir / 'papers')
        )

        # Initialize LLM client (globally shared)
        try:
            self.llm_client = create_llm_client(self.config.get('llm_config_file', './config/config.yaml'))
            logger.info("✅ LLM client initialized successfully (for query generation, etc.)")
        except Exception as e:
            logger.warning(f"LLM client initialization failed: {e}, some features may be limited")
            self.llm_client = None

        # Initialize paper analyzer (supports LLM enhancement)
        # Choose between DeepPaper or traditional RAG analyzer
        use_deep_paper = self.config.get('use_deep_paper', True)  # Default to DeepPaper

        if use_deep_paper:
            self.paper_analyzer = self._init_deep_paper_analyzer()
            self.use_deep_paper = True
        else:
            self.paper_analyzer = self._init_paper_analyzer()
            self.use_deep_paper = False

        # Initialize citation type inferencer (uses LLM for Socket Matching)
        self.citation_type_inferencer = self._init_citation_type_inferencer()

        # Initialize knowledge graph
        self.citation_graph = CitationGraph()

        # Initialize topic evolution analyzer (uses config file)
        full_config = load_full_config(self.config.get('llm_config_file', './config/config.yaml'))
        self.topic_evolution_analyzer = TopicEvolutionAnalyzer(config=full_config)

        # Initialize deep survey analyzer
        self.deep_survey_analyzer = DeepSurveyAnalyzer(config=full_config)

        # Initialize research idea generator
        self.research_idea_generator = ResearchIdeaGenerator(config=full_config)

        # Store intermediate results
        self.papers = []
        self.citation_edges = []
        self.enriched_papers = []
        self.typed_citation_edges = []  # New: citation edges with types
        self.deep_survey_report = {}  # New: deep survey report
        self.research_ideas = {}  # New: research ideas

        logger.info(f"Pipeline initialization complete, output directory: {self.output_dir}")

    def _init_deep_paper_analyzer(self):
        """
        Initialize DeepPaper Multi-Agent analyzer

        Supports two versions:
        - DeepPaper 1.0: Navigator → Extractor → Critic → Synthesizer
        - DeepPaper 2.0: LogicAnalyst + LimitationExtractor(with CitationDetective) + FutureWorkExtractor

        Returns:
            DeepPaperOrchestrator or DeepPaper2Orchestrator instance
        """
        from llm_config import LLMClient, LLMConfig

        # Get LLM config file path
        llm_config_path = Path('./config/config.yaml')

        if not llm_config_path.exists():
            logger.error(f"❌ LLM config file does not exist: {llm_config_path}")
            raise RuntimeError(f"DeepPaper requires LLM config file: {llm_config_path}")

        try:
            # Load LLM client
            config = LLMConfig.from_file(str(llm_config_path))
            llm_client = LLMClient(config)

            # Check if config specifies using version 2.0
            full_config = load_full_config(str(llm_config_path))
            use_version_2 = full_config.get('deep_paper', {}).get('use_version_2', False)
            use_citation_analysis = full_config.get('deep_paper', {}).get('use_citation_analysis', False)

            # Choose version based on configuration
            if use_version_2 and DeepPaper2Orchestrator is not None:
                logger.info(f"✅ Using DeepPaper 2.0 Multi-Agent analyzer")
                logger.info(f"   Config file: {llm_config_path}")
                logger.info(f"   Architecture: LogicAnalyst + LimitationExtractor + FutureWorkExtractor")
                if use_citation_analysis:
                    logger.info(f"   Citation analysis: Enabled (CitationDetective)")

                # Create DeepPaper 2.0 orchestrator
                orchestrator = DeepPaper2Orchestrator(
                    llm_client=llm_client,
                    use_citation_analysis=use_citation_analysis
                )

                logger.info(f"   Provider: {config.provider}")
                logger.info(f"   Model: {config.model}")

                return orchestrator
            else:
                # Use version 1.0
                if use_version_2 and DeepPaper2Orchestrator is None:
                    logger.warning("⚠️ DeepPaper 2.0 cannot be loaded, falling back to version 1.0")

                logger.info(f"✅ Using DeepPaper 1.0 Multi-Agent analyzer")
                logger.info(f"   Config file: {llm_config_path}")
                logger.info(f"   Architecture: Navigator → Extractor → Critic → Synthesizer")

                # Create DeepPaper 1.0 orchestrator
                orchestrator = DeepPaperOrchestrator(
                    llm_client=llm_client,
                    max_retries=self.config.get('deep_paper_max_retries', 2),
                    max_context_length=3000
                )

                logger.info(f"   Provider: {config.provider}")
                logger.info(f"   Model: {config.model}")
                logger.info(f"   Max Retries: {orchestrator.max_retries}")

                return orchestrator

        except Exception as e:
            logger.error(f"❌ Failed to initialize DeepPaper analyzer: {e}")
            raise RuntimeError(f"Unable to initialize DeepPaper analyzer: {e}")

    def _init_paper_analyzer(self):
        """
        Initialize paper analyzer

        Returns:
            LLMRAGPaperAnalyzer instance
        """

        # Get LLM config file path (defaults to unified config file)
        llm_config_path = Path('./config/config.yaml')

        if not llm_config_path.exists():
            logger.warning(f"LLM config file does not exist: {llm_config_path}, falling back to basic analysis")
            return LLMRAGPaperAnalyzer(
                llm_config_path=None,
                embedding_model='all-MiniLM-L6-v2',
                use_modelscope=True,
                prompts_dir='./prompts',
                max_context_length=3000,
                grobid_url=self.config.get('grobid_url')
            )

        try:
            logger.info(f"✅ Using LLM-enhanced RAG analyzer")
            logger.info(f"   Config file: {llm_config_path}")
            if self.config.get('grobid_url'):
                logger.info(f"   GROBID service: {self.config['grobid_url']}")

            return LLMRAGPaperAnalyzer(
                llm_config_path=str(llm_config_path),  # Convert to string
                embedding_model='all-MiniLM-L6-v2',
                use_modelscope=True,
                prompts_dir='./prompts',
                max_context_length=3000,
                grobid_url=self.config.get('grobid_url')
            )

        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM analyzer: {e}")
            raise RuntimeError(f"Unable to initialize paper analyzer: {e}")

    def _init_citation_type_inferencer(self):
        """
        Initialize citation type inferencer (uses Socket Matching)

        Returns:
            CitationTypeInferencer instance
        """
        llm_config_path = Path('./config/config.yaml')

        if not llm_config_path.exists():
            logger.warning(f"LLM config file does not exist: {llm_config_path}, will use rule-based method for citation type inference")
            return CitationTypeInferencer(llm_client=None, prompts_dir='./prompts')

        try:
            logger.info(f"✅ Using Socket Matching citation type inferencer (LLM mode)")
            logger.info(f"   Config file: {llm_config_path}")

            # Use config_path parameter directly, let CitationTypeInferencer load LLM internally
            return CitationTypeInferencer(
                config_path=str(llm_config_path),
                prompts_dir='./prompts'
            )

        except Exception as e:
            logger.error(f"❌ Failed to initialize Socket Matching inferencer: {e}")
            logger.warning("Falling back to rule-based method")
            return CitationTypeInferencer(llm_client=None, prompts_dir='./prompts')



    def run(self, topic: str) -> Dict:
        """
        Run the complete pipeline

        Args:
            topic: Research topic keywords

        Returns:
            Execution results dictionary
        """
        start_time = time.time()
        logger.info(f"Starting pipeline, research topic: '{topic}'")

        try:
            # Phase 1: Paper search and citation network construction
            logger.info("\n" + "="*60)
            logger.info("🔍 Phase 1: Paper Search and Citation Network Construction")
            logger.info("="*60)
            self._phase1_paper_search(topic)

            # Phase 2: PDF download
            if self.config['download_pdfs']:
                logger.info("\n" + "="*60)
                logger.info("📥 Phase 2: PDF Download")
                logger.info("="*60)
                self._phase2_pdf_download()

            # Phase 3: Paper RAG deep analysis
            logger.info("\n" + "="*60)
            logger.info("🧠 Phase 3: Paper RAG Deep Analysis")
            logger.info("="*60)
            self._phase3_paper_rag_analysis()

            # Phase 4: Citation relationship type inference
            logger.info("\n" + "="*60)
            logger.info("🔗 Phase 4: Citation Relationship Type Inference")
            logger.info("="*60)
            self._phase4_citation_type_inference()

            # Phase 5: Knowledge graph construction
            logger.info("\n" + "="*60)
            logger.info("📊 Phase 5: Knowledge Graph Construction and Visualization")
            logger.info("="*60)
            self._phase5_knowledge_graph()

            # Phase 6: Deep survey generation
            logger.info("\n" + "="*60)
            logger.info("📝 Phase 6: Deep Survey Generation")
            logger.info("="*60)
            self._phase6_deep_survey_generation(topic)

            # Phase 7: Research idea generation
            logger.info("\n" + "="*60)
            logger.info("💡 Phase 7: Research Idea Generation")
            logger.info("="*60)
            self._phase7_research_idea_generation(topic)

            # Phase 8: Results output
            logger.info("\n" + "="*60)
            logger.info("💾 Phase 8: Results Output and Report Generation")
            logger.info("="*60)
            results = self._phase8_output_results(topic)

            elapsed_time = time.time() - start_time
            logger.info(f"\n✅ Pipeline execution complete! Total time: {elapsed_time:.2f} seconds")

            return results

        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {e}")
            raise

    def _phase1_paper_search(self, topic: str) -> None:
        """
        Phase 1: Paper Search and Citation Network Construction

        Supports two modes:
        1. Traditional search: Simple search + citation expansion
        2. Enhanced snowball search: Complete 8-step method to build dense citation network
           Step 1: High-Quality Seed Retrieval
             - Use arXiv API + Categories for precise retrieval
             - Filter with keywords (title, abstract)
             - Limit time range (before 2022)
           Step 2: Cross-database ID Mapping
             - arXiv papers -> OpenAlex ID
             - If mapping fails, use manual search to build citation network
           Step 3: Forward Snowballing
             - Seed -> Who cited Seed? -> Child nodes
           Step 4: Backward Snowballing
             - Who was cited by Seed? <- Seed -> Parent nodes/ancestors
           Step 5: Horizontal Supplement/Co-citation Mining
             - Among child and parent nodes, who is repeatedly mentioned?
             - Co-citation threshold filtering
           Step 6 [Optional]: Second Round Snowballing
             - Perform controlled expansion on first-round papers
           Step 7: Supplement Latest SOTA (Recent Frontiers)
             - arXiv papers from last 6-12 months
             - Similarity filtering
           Step 8: Citation Closure Construction
             - Build complete network
        """
        logger.info(f"Search topic: '{topic}'")

        # Check if snowball search is enabled
        full_config = load_full_config(self.config.get('llm_config_file', './config/config.yaml'))
        use_snowball = full_config.get('snowball', {}).get('enabled', False)

        if use_snowball:
            logger.info("📊 Using enhanced snowball search mode (complete 8-step method)")
            # Create new 8-step paper search pipeline
            pipeline = PaperSearchPipeline(
                openalex_client=self.openalex_client,
                config_path=self.config.get('llm_config_file', './config/config.yaml'),
                llm_client=self.llm_client
            )

            # Execute complete 8-step search process
            # Read keywords and arXiv categories from config file
            snowball_config = full_config.get('snowball', {})
            keywords = snowball_config.get('search_keywords', None)
            categories = snowball_config.get('arxiv_categories', None)

            # Convert empty list to None
            if keywords == []:
                keywords = None
            if categories == []:
                categories = None

            logger.info(f"  Search parameters:")
            logger.info(f"    - Topic: {topic}")
            if keywords:
                logger.info(f"    - Keywords: {keywords}")
            if categories:
                logger.info(f"    - arXiv categories: {categories}")

            result = pipeline.execute_full_pipeline(
                topic=topic,
                keywords=keywords,
                categories=categories
            )

            # Convert result to paper list
            self.papers = list(result['papers'].values())
            self.citation_edges = result['citation_edges']

        else:
            logger.info("🔍 Using traditional search mode")
            self._traditional_paper_search(topic)

        # Check if Idea evaluation mode is enabled - filter by year
        idea_eval_config = full_config.get('research_idea', {}).get('idea_evaluation_mode', {})
        if idea_eval_config.get('enabled', False):
            filter_year = idea_eval_config.get('filter_year_after', 2022)
            self._filter_papers_by_year(filter_year)

    def _filter_papers_by_year(self, filter_year_after: int) -> None:
        """
        Filter papers: keep papers published before the specified year (excluding that year)

        Args:
            filter_year_after: Filter out papers from this year and later
                              For example, filter_year_after=2022 will keep papers from 2021 and earlier
        """
        original_count = len(self.papers)

        # Filter papers
        filtered_papers = []
        removed_papers = []

        for paper in self.papers:
            year = paper.get('publication_year') or paper.get('year')
            if year and year < filter_year_after:
                filtered_papers.append(paper)
            else:
                removed_papers.append(paper)

        self.papers = filtered_papers

        # Also filter citation edges: remove edges involving filtered papers
        if self.citation_edges:
            removed_ids = {p.get('id') for p in removed_papers}
            filtered_edges = []

            for edge in self.citation_edges:
                # citation_edges is tuple format: (source_id, target_id)
                if isinstance(edge, tuple):
                    source_id, target_id = edge
                else:
                    # Compatible with dict format
                    source_id = edge.get('source')
                    target_id = edge.get('target')

                # Only keep edges where both endpoints are not filtered
                if source_id not in removed_ids and target_id not in removed_ids:
                    filtered_edges.append(edge)

            self.citation_edges = filtered_edges

        # Log results
        logger.info("\n" + "="*60)
        logger.info(f"📅 Idea Evaluation Mode: Year Filtering")
        logger.info("="*60)
        logger.info(f"Filter rule: Keep papers before year {filter_year_after}")
        logger.info(f"Original paper count: {original_count}")
        logger.info(f"Kept papers: {len(self.papers)}")
        logger.info(f"Removed papers: {len(removed_papers)}")

        if self.citation_edges:
            logger.info(f"Citation edges: {len(self.citation_edges)}")

        # Show year distribution of kept papers
        if self.papers:
            years = [p.get('publication_year') or p.get('year') for p in self.papers if p.get('publication_year') or p.get('year')]
            if years:
                min_year = min(years)
                max_year = max(years)
                logger.info(f"Kept papers year range: {min_year} - {max_year}")

        logger.info("="*60 + "\n")

    def _traditional_paper_search(self, topic: str) -> None:
        """Traditional search mode"""
        self.papers = self.openalex_client.search_papers(
            topic=topic,
            max_results=self.config['max_papers'],
            sort_by="cited_by_count",
            min_citations=self.config['min_citation_count']
        )

        logger.info(f"✅ Found {len(self.papers)} papers")

        # Display found papers
        for i, paper in enumerate(self.papers[:5], 1):  # Only show first 5
            logger.info(f"  [{i}] {paper['title']} ({paper['year']}) - Citations: {paper['cited_by_count']}")

        if len(self.papers) > 5:
            logger.info(f"  ... and {len(self.papers) - 5} more papers")

    def _phase2_pdf_download(self) -> None:
        """Phase 2: PDF Download (Enhanced version, supports multiple sources and retries)"""
        max_downloads = self.config['max_pdf_downloads']
        logger.info(f"Starting PDF download (max {max_downloads} papers)...")
        logger.info("Using PDF URL download + arXiv title search download")

        download_results = self.pdf_downloader.batch_download(
            papers=self.papers,
            max_downloads=max_downloads
        )

        # Detailed statistics
        logger.info(f"✅ PDF download completed:")
        logger.info(f"  📥 Successfully downloaded: {download_results['downloaded']} papers")
        logger.info(f"  📁 Already exists: {download_results['exists']} papers")
        logger.info(f"  ❌ Download failed: {download_results['failed']} papers")
        logger.info(f"  📊 Total attempts: {download_results['attempted']} / {download_results['total_papers']} papers")

        # Provide suggestions if failure rate is high
        if download_results['failed'] > download_results['downloaded']:
            logger.warning("⚠️ High download failure rate, possible reasons:")
            logger.warning("  - Papers do not provide open access PDFs")
            logger.warning("  - Subscription or paid access required")
            logger.warning("  - Network connection issues or server limitations")
            logger.warning("  - Consider adding more PDF sources or using institutional access")

    def _phase3_paper_rag_analysis(self) -> None:
        """
        Phase 3: Deep Paper Analysis

        Selection based on configuration:
        - DeepPaper Multi-Agent: Iterative multi-agent system with Reflection Loop
        - Traditional RAG: Single retrieval + LLM generation

        Extracted fields: Problem, Method, Limitation, Future Work
        """
        if self.use_deep_paper:
            logger.info(f"🤖 Using DeepPaper Multi-Agent to analyze {len(self.papers)} papers")
            logger.info("   Architecture: Navigator → Extractor → Critic → Synthesizer")
            self._analyze_with_deep_paper()
        else:
            logger.info(f"Using traditional RAG analyzer to analyze {len(self.papers)} papers")
            self._analyze_with_traditional_rag()

    def _analyze_with_deep_paper(self) -> None:
        """
        Analyze papers using DeepPaper Multi-Agent system

        Supports versions 1.0 and 2.0, automatically adapts
        """
        from grobid_parser import GrobidPDFParser

        pdf_dir = self.data_dir / 'papers'
        grobid_url = self.config.get('grobid_url')

        # Check if using DeepPaper 2.0
        is_version_2 = isinstance(self.paper_analyzer, DeepPaper2Orchestrator) if DeepPaper2Orchestrator else False

        # Initialize GROBID parser (if available)
        grobid_parser = None
        if grobid_url:
            try:
                grobid_parser = GrobidPDFParser(grobid_url)
                logger.info(f"   GROBID service: {grobid_url}")
            except:
                logger.warning("   GROBID unavailable, will use PyPDF2")

        # Batch analysis
        deep_reports = []
        success_count = 0
        pdf_count = 0

        for i, paper in enumerate(self.papers):
            try:
                logger.info(f"\n   [{i+1}/{len(self.papers)}] {paper['title'][:50]}...")

                # Convert to PaperDocument
                paper_doc = self._convert_to_paper_document(
                    paper, pdf_dir, grobid_parser
                )

                # Analyze using DeepPaper
                # Optional: save individual reports for each paper to output/deep_paper/ directory
                deep_paper_output = self.output_dir / 'deep_paper' if self.config.get('save_deep_paper_reports', False) else None
                if deep_paper_output:
                    deep_paper_output.mkdir(parents=True, exist_ok=True)

                # Call different interfaces based on version
                if is_version_2:
                    # DeepPaper 2.0: requires paper_id for citation analysis
                    paper_id = paper.get('doi') or paper.get('id', '')
                    report = self.paper_analyzer.analyze_paper(
                        paper_document=paper_doc,
                        paper_id=paper_id,
                        output_dir=str(deep_paper_output) if deep_paper_output else None
                    )
                else:
                    # DeepPaper 1.0: only needs paper_document
                    report = self.paper_analyzer.analyze_paper(
                        paper_document=paper_doc,
                        output_dir=str(deep_paper_output) if deep_paper_output else None
                    )

                # Convert to pipeline format
                enriched_paper = paper.copy()
                enriched_paper['deep_analysis'] = report.to_dict()

                # Compatible with old format (for citation type inference)
                enriched_paper['rag_analysis'] = {
                    'problem': report.problem,
                    'method': report.method,
                    'limitation': report.limitation,
                    'future_work': report.future_work
                }

                # Mark version used
                version_tag = 'deep_paper_2.0' if is_version_2 else 'deep_paper_1.0'
                enriched_paper['analysis_method'] = version_tag

                # Extract quality information (if available)
                if hasattr(report, 'extraction_quality'):
                    enriched_paper['extraction_quality'] = report.extraction_quality
                enriched_paper['sections_extracted'] = len(paper_doc.sections)

                deep_reports.append(enriched_paper)
                success_count += 1

                if len(paper_doc.sections) > 1:
                    pdf_count += 1

            except Exception as e:
                logger.error(f"   ❌ Analysis failed: {e}")
                # Add failed paper
                failed_paper = paper.copy()
                failed_paper['rag_analysis'] = {
                    'problem': f'Analysis failed: {str(e)}',
                    'method': '',
                    'limitation': '',
                    'future_work': ''
                }
                failed_paper['analysis_method'] = 'failed'
                deep_reports.append(failed_paper)

        self.enriched_papers = deep_reports

        # Statistics
        version_name = "DeepPaper 2.0" if is_version_2 else "DeepPaper 1.0"
        logger.info(f"\n✅ {version_name} analysis completed:")
        logger.info(f"  Successfully analyzed: {success_count}/{len(self.papers)} papers")
        logger.info(f"  With PDF: {pdf_count} papers")
        logger.info(f"  Abstract only: {len(self.papers) - pdf_count} papers")

        # Display samples
        self._display_analysis_samples(sample_count=2)

    def _analyze_with_traditional_rag(self) -> None:
        """Analyze papers using traditional RAG analyzer"""
        logger.info("Extracting fields: Problem, Method, Limitation, Future Work")

        # Batch analysis using RAG/LLM analyzer
        pdf_dir = str(self.data_dir / 'papers')
        self.enriched_papers = self.paper_analyzer.batch_analyze_papers(
            self.papers,
            pdf_dir=pdf_dir
        )

        # Statistical analysis results
        success_count = 0
        with_pdf_count = 0

        for paper in self.enriched_papers:
            rag_analysis = paper.get('rag_analysis', {})
            if rag_analysis and 'error' not in rag_analysis:
                success_count += 1
            if paper.get('sections_extracted', 0) > 0:
                with_pdf_count += 1

        logger.info(f"✅ RAG analysis completed:")
        logger.info(f"  Successfully analyzed: {success_count}/{len(self.papers)} papers")
        logger.info(f"  With PDF: {with_pdf_count} papers")
        logger.info(f"  Abstract only: {len(self.papers) - with_pdf_count} papers")

        # Display samples
        self._display_analysis_samples(sample_count=2)

    def _convert_to_paper_document(self, paper: Dict, pdf_dir: Path, grobid_parser) -> PaperDocument:
        """Convert OpenAlex paper to PaperDocument format"""
        paper_id = paper.get('id', 'unknown')
        title = paper.get('title', 'Untitled')
        abstract = paper.get('abstract', '')
        authors = [
            author.get('author', {}).get('display_name', 'Unknown')
            for author in paper.get('authorships', [])
        ]
        year = paper.get('publication_year')

        # Extract sections
        sections = []

        # Try to extract from PDF
        pdf_path = self._find_pdf(paper_id, pdf_dir)
        if pdf_path and grobid_parser:
            try:
                sections = grobid_parser.extract_sections_from_pdf(pdf_path)
            except:
                pass

        # Fallback to abstract
        if not sections:
            if title:
                sections.append(PaperSection(
                    title='Title',
                    content=title,
                    page_num=0,
                    section_type='title'
                ))
            if abstract:
                sections.append(PaperSection(
                    title='Abstract',
                    content=abstract,
                    page_num=0,
                    section_type='abstract'
                ))

        return PaperDocument(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            sections=sections,
            metadata=paper
        )

    def _find_pdf(self, paper_id: str, pdf_dir: Path) -> Optional[str]:
        """Find paper PDF"""
        if not pdf_dir.exists():
            return None

        for pdf_file in pdf_dir.glob(f"{paper_id}*.pdf"):
            return str(pdf_file)

        return None

    def _display_analysis_samples(self, sample_count: int = 2):
        """Display analysis samples"""
        sample_count = min(sample_count, len(self.enriched_papers))
        if sample_count == 0:
            return

        logger.info(f"\n📋 Analysis result samples (first {sample_count} papers):")

        for i, paper in enumerate(self.enriched_papers[:sample_count], 1):
            rag_analysis = paper.get('rag_analysis', {})
            if rag_analysis and 'error' not in rag_analysis:
                logger.info(f"\n  [{i}] {paper['title'][:60]}...")
                logger.info(f"      Analysis method: {paper.get('analysis_method', 'N/A').upper()}")

                # Display quality score (DeepPaper specific)
                if 'extraction_quality' in paper:
                    quality = paper['extraction_quality']
                    avg_quality = sum(quality.values()) / len(quality) if quality else 0
                    logger.info(f"      Average quality: {avg_quality:.2f}")

                logger.info(f"      Number of sections: {paper.get('sections_extracted', 0)}")

                problem = rag_analysis.get('problem', '')[:100]
                if problem and problem not in ['No available content', 'No relevant information found', '']:
                    logger.info(f"      Problem: {problem}...")

                method = rag_analysis.get('method', '')[:100]
                if method and method not in ['No available content', 'No relevant information found', '']:
                    logger.info(f"      Method: {method}...")

    def _phase4_citation_type_inference(self) -> None:
        """
        Phase 4: Citation Relationship Type Inference (Socket Matching)

        Uses Socket Matching method to infer semantic types for each citation relationship:

        🔌 Socket Matching Core Concept:
        Use deep information from papers (Problem, Method, Limitation, Future_Work)
        as "sockets", and use LLM Agent to determine if these sockets can connect.

        📊 Supported Relationship Types (Socket Matching - 6 types):
        1. Overcomes - Overcome/Optimize (Vertical Deepening)
           B solves A's limitations
           Source: Match 1 (Limitation→Problem)

        2. Realizes - Realize Vision (Research Inheritance)
           B implements suggestions from A's Future Work
           Source: Match 2 (Future_Work→Problem)

        3. Extends - Method Extension (Minor Innovation)
           B makes incremental improvements on A's method
           Source: Match 3 Extension

        4. Alternative - Alternative Approach (Disruptive Innovation)
           B solves similar problems with completely different paradigms
           Source: Match 3 Alternative

        5. Adapts_to - Technology Transfer (Horizontal Diffusion)
           B applies A's method to new domain/scenario
           Source: Match 4 (Problem→Problem cross-domain)

        6. Baselines - Baseline Comparison (Background Noise)
           B only uses A as comparison object, no direct inheritance
           Source: No match

        🔗 Logic Connection Matrix (4 Matches → 6 types):
        - Match 1: A.Limitation ↔ B.Problem → Overcomes
        - Match 2: A.Future_Work ↔ B.Problem → Realizes
        - Match 3: A.Method ↔ B.Method → Extends / Alternative
        - Match 4: A.Problem ↔ B.Problem(cross-domain) → Adapts_to
        - No match → Baselines
        """
        logger.info("Starting citation relationship type inference (Socket Matching)...")

        # Check if using LLM mode
        if self.citation_type_inferencer.llm_client:
            logger.info("  🔌 Using LLM Socket Matching mode")
            logger.info("  ⏳ Estimated 12-20 seconds per edge (5 LLM calls)")
        else:
            logger.info("  📏 Using rule-based method mode (fallback)")

        # Batch infer citation types using inferencer
        self.typed_citation_edges, edge_type_statistics = \
            self.citation_type_inferencer.infer_edge_types(
                papers=self.enriched_papers,
                citation_edges=self.citation_edges
            )

        logger.info(f"✅ Citation relationship type inference completed:")
        logger.info(f"  Total citation relationships: {len(self.typed_citation_edges)}")
        logger.info(f"  Types labeled: {len(edge_type_statistics)}")

        # Display inference strategy description
        if self.citation_type_inferencer.llm_client:
            logger.info(f"\n📊 Socket Matching inference strategy:")
            logger.info(f"  • Deep semantic analysis: Based on Problem, Method, Limitation, Future_Work")
            logger.info(f"  • 4 Socket connections → 6 types:")
            logger.info(f"    - Match 1: Limitation↔Problem → Overcomes")
            logger.info(f"    - Match 2: FutureWork↔Problem → Realizes")
            logger.info(f"    - Match 3: Method↔Method → Extends / Alternative")
            logger.info(f"    - Match 4: Problem↔Problem(cross-domain) → Adapts_to")
            logger.info(f"  • LLM verification layer: Citation context evidence verification")
            logger.info(f"  • Comprehensive classification: Final classification based on all match results")
        else:
            logger.info(f"\n📊 Rule-based method inference strategy:")
            logger.info(f"  • Based on time difference: 10+ years → Classic/historical citation, 5+ years → Extension/background citation, within 2 years → Concurrent/comparison")
            logger.info(f"  • Based on citation count: High-citation papers → Authoritative citation")
            logger.info(f"  • Based on text similarity: Use simple vocabulary overlap calculation")
            logger.info(f"  • Comprehensive judgment: Infer most appropriate relationship type by combining multi-dimensional information")

    def _phase5_knowledge_graph(self) -> None:
        """
        Phase 5: Knowledge Graph Construction and Visualization

        Build knowledge graph from papers and citation relationships:
        1. Add paper nodes (containing RAG analysis results)
        2. Add citation edges (using edge types inferred in Phase 4)
        3. Compute graph metrics
        4. Generate interactive visualization
        """
        logger.info("Building knowledge graph...")

        # Build graph (using typed citation edges)
        self.citation_graph.build_citation_network(
            papers=self.enriched_papers,
            citation_data=self.typed_citation_edges  # Use typed edges inferred in Phase 4
        )

        # Compute graph metrics
        metrics = self.citation_graph.compute_metrics()
        logger.info(f"✅ Knowledge graph construction completed:")
        logger.info(f"    Nodes: {metrics.get('total_nodes', 0)}")
        logger.info(f"    Edges: {metrics.get('total_edges', 0)}")
        logger.info(f"    Graph density: {metrics.get('density', 0):.4f}")

    def _phase6_deep_survey_generation(self, topic: str) -> None:
        """
        Phase 6: Deep Survey Generation

        Core methodology: Relation-based graph pruning + Critical evolutionary path identification

        Executes three steps:
        1. Relation-Based Graph Pruning
           - Keep Seed Papers
           - Only keep papers connected to Seed through strong logical relationships (Overcomes, Realizes, Extends, Alternative, Adapts_to)
           - Remove papers only connected by weak relationships (Baselines) or isolated papers
           - Solves "data noise" problem

        2. Critical Evolutionary Path Identification
           - Identify linear chains (The Chain): A -> Overcomes -> B -> Extends -> C
           - Identify star bursts (The Star): Seed -> [Multiple Routes]
           - Generate narrative units for each evolutionary path
           - Solves "fragmentation" problem

        3. Structured Deep Survey Report
           - Display each evolutionary story in Thread format
           - Accompany with visualization and text interpretation
           - Include relationship chains, paper information, citation statistics, etc.

        Output:
        - self.deep_survey_report: Complete results including pruning statistics, evolutionary paths, survey report, etc.
        """
        # Check if enabled
        full_config = load_full_config(self.config.get('llm_config_file', './config/config.yaml'))
        if not full_config.get('deep_survey', {}).get('enabled', True):
            logger.info("Deep survey generation disabled, skipping")
            self.deep_survey_report = {}
            return

        logger.info("Starting deep survey generation...")

        # Get graph
        G = self.citation_graph.graph

        if len(G.nodes()) == 0:
            logger.warning("Knowledge graph is empty, skipping deep survey generation")
            self.deep_survey_report = {}
            return

        # Execute analysis using deep survey analyzer
        self.deep_survey_report = self.deep_survey_analyzer.analyze(G, topic)
        logger.info("✅ Deep survey generation completed")

    def _phase7_research_idea_generation(self, topic: str) -> None:
        """
        Phase 7: Research Idea Generation (Hypothesis Generator with Chain of Thought + Evolutionary Paths)

        Core method: Use Chain of Thought reasoning to generate feasible research ideas
        Enhanced feature: Integrate evolutionary paths from deep survey, learn evolutionary logic

        Three-step reasoning process:
        - Step 1: Analyze Compatibility
          Check if mathematical/algorithmic/theoretical properties of candidate methods are compatible with constraints
        - Step 2: Identify the Gap
          Determine what specific modifications are needed to bridge the gap, find "Bridging Variable"
        - Step 3: Draft the Idea
          Generate title, abstract (Background → Gap → Proposed Method → Expected Result)

        Evolutionary Path Learning (New):
        - Extract evolutionary paths from Phase 6 deep survey results
        - Learn evolutionary logic (Chain/Divergence/Convergence)
        - Reference evolutionary patterns from historical successful cases
        - More intelligently combine Limitation and Method

        Input sources:
        - Unsolved Limitations: Extract from rag_limitation and limitations fields of graph nodes
        - Candidate Methods: Extract from rag_method field of graph nodes
        - Evolutionary Paths: Extract from Phase 6 deep_survey_report

        Output states:
        - SUCCESS: Generated feasible innovative ideas
        - INCOMPATIBLE: Method and limitation are fundamentally incompatible
        - ERROR: Error occurred during generation
        """
        # Check if enabled
        full_config = load_full_config(self.config.get('llm_config_file', './config/config.yaml'))
        if not full_config.get('research_idea', {}).get('enabled', True):
            logger.info("Research idea generation disabled, skipping")
            self.research_ideas = {}
            return

        logger.info("Starting research idea generation (using Chain of Thought reasoning + evolutionary path learning)...")

        # Get knowledge graph
        G = self.citation_graph.graph

        if len(G.nodes()) == 0:
            logger.warning("Knowledge graph is empty, skipping research idea generation")
            self.research_ideas = {}
            return

        # Extract evolutionary paths from Phase 6 deep survey results
        evolutionary_paths = None
        if self.deep_survey_report and isinstance(self.deep_survey_report, dict):
            evolutionary_paths = self.deep_survey_report.get('evolutionary_paths', [])
            if evolutionary_paths:
                logger.info(f"  Extracted {len(evolutionary_paths)} evolutionary paths from deep survey")
                logger.info("  Will use evolutionary paths to learn evolutionary logic and generate smarter research ideas")
            else:
                logger.info("  No evolutionary paths found in deep survey, will use standard mode to generate ideas")
        else:
            logger.info("  Deep survey results unavailable, will use standard mode to generate ideas")

        # Use ResearchIdeaGenerator to generate ideas directly from knowledge graph
        # Internally will:
        # 1. Extract limitations and methods from graph nodes (fragmentation pooling)
        # 2. Perform Cartesian product matching of limitation × method
        # 3. Filter feasible solutions through Chain of Thought reasoning
        # 4. Integrate evolutionary path information, learn evolutionary logic (new)
        self.research_ideas = self.research_idea_generator.generate_from_knowledge_graph(
            graph=G,
            topic=topic,
            evolutionary_paths=evolutionary_paths,  # Pass evolutionary paths
            verbose=True
        )

        logger.info(f"✅ Research idea generation completed, successfully generated {self.research_ideas.get('successful_ideas', 0)} ideas")

    def _phase8_output_results(self, topic: str) -> Dict:
        """Phase 8: Output Results and Report Generation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_safe = topic.replace(" ", "_").replace("/", "_")

        # Get graph metrics
        graph_metrics = self.citation_graph.compute_metrics()

        # 1. Save paper data
        papers_file = self.output_dir / f"papers_{topic_safe}_{timestamp}.json"
        with open(papers_file, 'w', encoding='utf-8') as f:
            json.dump(self.enriched_papers, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)
        logger.info(f"📄 Paper data saved: {papers_file}")

        # 2. Save graph data
        graph_file = self.output_dir / f"graph_data_{topic_safe}_{timestamp}.json"
        self.citation_graph.export_graph_data(str(graph_file))
        logger.info(f"🔗 Graph data saved: {graph_file}")

        # 3. Generate visualization
        viz_file = self.output_dir / f"graph_viz_{topic_safe}_{timestamp}.html"
        max_nodes_in_viz = self.config.get('max_nodes_in_viz', 100)  # Read from config, default 100
        self.citation_graph.visualize_graph(
            str(viz_file),
            max_nodes=max_nodes_in_viz,
            deep_survey_report=self.deep_survey_report,
            research_ideas=self.research_ideas
        )
        logger.info(f"📊 Visualization file saved: {viz_file} (showing {max_nodes_in_viz} nodes, including deep survey and research ideas)")

        # 4. Save deep survey report
        deep_survey_file = self.output_dir / f"deep_survey_{topic_safe}_{timestamp}.json"
        with open(deep_survey_file, 'w', encoding='utf-8') as f:
            json.dump(self.deep_survey_report, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)
        logger.info(f"📝 Deep survey report saved: {deep_survey_file}")

        # 5. Save research ideas report
        research_ideas_file = self.output_dir / f"research_ideas_{topic_safe}_{timestamp}.json"
        with open(research_ideas_file, 'w', encoding='utf-8') as f:
            json.dump(self.research_ideas, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)
        logger.info(f"💡 Research ideas report saved: {research_ideas_file}")

        # Collect seed node IDs
        seed_ids = [p.get('id') for p in self.enriched_papers if p.get('is_seed', False)]

        # 7. Generate summary results
        results = {
            'topic': topic,
            'timestamp': timestamp,
            'summary': {
                'total_papers': len(self.enriched_papers),
                'successful_analysis': sum(
                    1 for p in self.enriched_papers
                    if p.get('rag_analysis') and 'error' not in p.get('rag_analysis', {})
                ),
                'citation_edges': len(self.citation_edges),
                'graph_nodes': graph_metrics.get('total_nodes', 0),
                'graph_edges': graph_metrics.get('total_edges', 0),
                'seed_count': len(seed_ids),
                'seed_ids': seed_ids,  # Add seed node ID list

                'analysis_method': 'multi-agent',  # Mark using RAG method
            },
            'files': {
                'papers_data': str(papers_file),
                'graph_data': str(graph_file),
                'visualization': str(viz_file),
                'deep_survey': str(deep_survey_file),
                'research_ideas': str(research_ideas_file),
            }
        }

        # Save summary results
        summary_file = self.output_dir / f"summary_{topic_safe}_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)

        logger.info(f"📊 Summary results saved: {summary_file}")
        logger.info("\n" + "="*60)
        logger.info("🎉 All result files:")
        for file_type, file_path in results['files'].items():
            logger.info(f"  {file_type}: {file_path}")
        logger.info(f"  summary: {summary_file}")
        logger.info("="*60)

        # Output seed node information
        if seed_ids:
            logger.info(f"\n🌱 Seed node information:")
            logger.info(f"  Total: {len(seed_ids)}")
            logger.info(f"  ID list: {seed_ids[:3]}{'...' if len(seed_ids) > 3 else ''}")
            logger.info("="*60)

        return results

    def get_stats(self) -> Dict:
        """Get current statistics"""
        return {
            'papers_count': len(self.papers),
            'citation_edges_count': len(self.citation_edges),
            'enriched_papers_count': len(self.enriched_papers),
            'pdf_stats': self.pdf_downloader.get_download_stats()
        }

    def load_from_cache(self, cache_file: str) -> bool:
        """Load data from cache file"""
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.papers = data.get('papers', [])
            self.citation_edges = data.get('citation_edges', [])
            self.enriched_papers = data.get('enriched_papers', [])

            logger.info(f"Successfully loaded data from cache: {cache_file}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load data from cache: {e}")
            return False

    def save_to_cache(self, cache_file: str) -> None:
        """Save data to cache file"""
        try:
            cache_data = {
                'papers': self.papers,
                'citation_edges': self.citation_edges,
                'enriched_papers': self.enriched_papers,
                'timestamp': datetime.now().isoformat(),
                'config': self.config
            }

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)

            logger.info(f"Data saved to cache: {cache_file}")

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")


if __name__ == "__main__":
    # Test code
    pipeline = PaperGraphPipeline()

    # Run pipeline
    results = pipeline.run("transformer neural networks")

    print(f"\n🎯 Pipeline execution results:")
    print(f"  Topic: {results['topic']}")
    print(f"  Total papers: {results['summary']['total_papers']}")
    print(f"  Successful analysis: {results['summary']['successful_analysis']}")
    print(f"  Graph nodes: {results['summary']['graph_nodes']}")
    print(f"  Graph edges: {results['summary']['graph_edges']}")
    print(f"  Visualization file: {results['files']['visualization']}")