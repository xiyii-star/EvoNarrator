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

# Import DeepPaper 2.0 Multi-Agent system (enhanced version - integrated Citation Detective)
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
    Load complete configuration from YAML file

    Args:
        config_path: Configuration file path

    Returns:
        Complete configuration dictionary
    """
    config_file = Path(config_path)

    if not config_file.exists():
        logger.warning(f"Configuration file does not exist: {config_path}, using default configuration")
        return {}

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration file: {config_path}")
        return config if config else {}
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}, using default configuration")
        return {}


class PaperGraphPipeline:
    """
    Paper Knowledge Graph Construction Pipeline
    Integrates the complete process of retrieval, download, analysis, and graph construction
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
            'max_total_papers': 100,   # Total paper limit (including expanded citations)
            'min_citation_count': 10,  # Minimum citation count filter
            'download_pdfs': True,     # Whether to download PDFs
            'max_pdf_downloads': 5,    # Maximum PDF downloads
            'output_dir': './output',  # Output directory
            'data_dir': './data',      # Data directory
            'llm_config_file': './config/config.yaml',  # Unified configuration file path
            'grobid_url': None,        # GROBID service URL (e.g. http://localhost:8070)
            'save_stage_outputs': True,  # Whether to save each stage's output
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
            logger.info("✅ LLM client initialized successfully (for query generation and other features)")
        except Exception as e:
            logger.warning(f"LLM client initialization failed: {e}, some features may be limited")
            self.llm_client = None

        # Initialize paper analyzer (supports LLM enhancement)
        # Choose between DeepPaper or traditional RAG analyzer
        use_deep_paper = self.config.get('use_deep_paper', True)  # Use DeepPaper by default

        if use_deep_paper:
            self.paper_analyzer = self._init_deep_paper_analyzer()
            self.use_deep_paper = True
        else:
            self.paper_analyzer = self._init_paper_analyzer()
            self.use_deep_paper = False

        # Initialize citation type inferencer (using LLM for Socket Matching)
        self.citation_type_inferencer = self._init_citation_type_inferencer()

        # Initialize knowledge graph
        self.citation_graph = CitationGraph()

        # Initialize topic evolution analyzer (using configuration file)
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
        self.typed_citation_edges = []  # New: typed citation edges
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

        # Get LLM configuration file path
        llm_config_path = Path('./config/config.yaml')

        if not llm_config_path.exists():
            logger.error(f"❌ LLM configuration file does not exist: {llm_config_path}")
            raise RuntimeError(f"DeepPaper requires LLM configuration file: {llm_config_path}")

        try:
            # Load LLM client
            config = LLMConfig.from_file(str(llm_config_path))
            llm_client = LLMClient(config)

            # Check if configuration specifies using version 2.0
            full_config = load_full_config(str(llm_config_path))
            use_version_2 = full_config.get('deep_paper', {}).get('use_version_2', False)
            use_citation_analysis = full_config.get('deep_paper', {}).get('use_citation_analysis', False)

            # Select version based on configuration
            if use_version_2 and DeepPaper2Orchestrator is not None:
                logger.info(f"✅ Using DeepPaper 2.0 Multi-Agent analyzer")
                logger.info(f"   Configuration file: {llm_config_path}")
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
                logger.info(f"   Configuration file: {llm_config_path}")
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
            raise RuntimeError(f"Cannot initialize DeepPaper analyzer: {e}")

    def _init_paper_analyzer(self):
        """
        Initialize paper analyzer

        Returns:
            LLMRAGPaperAnalyzer instance
        """

        # Get LLM configuration file path (use unified configuration file by default)
        llm_config_path = Path('./config/config.yaml')

        if not llm_config_path.exists():
            logger.warning(f"LLM config file does not exist: {llm_config_path}, downgrading to basic analysis")
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
                llm_config_path=str(llm_config_path),  # Convert to string for passing
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
        Initialize citation type inferencer (using Socket Matching)

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

            # Directly use config_path parameter, let CitationTypeInferencer load LLM internally
            return CitationTypeInferencer(
                config_path=str(llm_config_path),
                prompts_dir='./prompts'
            )

        except Exception as e:
            logger.error(f"❌ Failed to initialize Socket Matching inferencer: {e}")
            logger.warning("Downgrading to rule-based method")
            return CitationTypeInferencer(llm_client=None, prompts_dir='./prompts')



    def run(self, topic: str) -> Dict:
        """
        Run the complete pipeline

        Args:
            topic: Research topic keywords

        Returns:
            Run result dictionary
        """
        start_time = time.time()
        logger.info(f"Starting Pipeline run, research topic: '{topic}'")

        try:
            # Phase 1: Paper search and citation network construction
            logger.info("\n" + "="*60)
            logger.info("🔍 Phase 1: Paper search and citation network construction")
            logger.info("="*60)
            self._phase1_paper_search(topic)

            # Phase 2: PDF download
            if self.config['download_pdfs']:
                logger.info("\n" + "="*60)
                logger.info("📥 Phase 2: PDF download")
                logger.info("="*60)
                self._phase2_pdf_download()

            # Phase 3: Paper RAG deep analysis
            logger.info("\n" + "="*60)
            logger.info("🧠 Phase 3: Paper RAG deep analysis")
            logger.info("="*60)
            self._phase3_paper_rag_analysis()

            # Phase 4: Citation relationship type inference
            logger.info("\n" + "="*60)
            logger.info("🔗 Phase 4: Citation relationship type inference")
            logger.info("="*60)
            self._phase4_citation_type_inference()

            # Phase 5: Knowledge graph construction
            logger.info("\n" + "="*60)
            logger.info("📊 Phase 5: Knowledge graph construction and visualization")
            logger.info("="*60)
            self._phase5_knowledge_graph()

            # Phase 6: Deep survey generation
            logger.info("\n" + "="*60)
            logger.info("📝 Phase 6: Deep survey generation")
            logger.info("="*60)
            self._phase6_deep_survey_generation(topic)

            # Phase 7: Research idea generation
            logger.info("\n" + "="*60)
            logger.info("💡 Phase 7: Research idea generation")
            logger.info("="*60)
            self._phase7_research_idea_generation(topic)

            # Phase 8: Result output
            logger.info("\n" + "="*60)
            logger.info("💾 Phase 8: Result output and report generation")
            logger.info("="*60)
            results = self._phase8_output_results(topic)

            elapsed_time = time.time() - start_time
            logger.info(f"\n✅ Pipeline run completed! Total time: {elapsed_time:.2f} seconds")

            return results

        except Exception as e:
            logger.error(f"❌ Pipeline run failed: {e}")
            raise

    def _phase1_paper_search(self, topic: str) -> None:
        """
        Phase 1: Paper search and citation network construction

        Supports two modes:
        1. Traditional search: Simple search + citation expansion
        2. Enhanced snowballing search: Complete 8-step method to build dense citation network
           Step 1: High-Quality Seed Retrieval
              - Use arXiv API + Categories for precise search
              - Filter by keywords (title, abstract)
              - Limit time range (before 2022)
           Step 2: ID Mapping
              - arXiv 论文 -> OpenAlex ID
              - 如果映射失败,使用手动搜索构建引用网络
           步骤 3: 前向滚雪球
              - 种子 -> 谁引用了种子? -> 子节点
           步骤 4: 后向滚雪球
              - 谁被种子引用? <- 种子 -> 父节点/祖先
           步骤 5: 横向补充/共引挖掘
              - 在子节点和父节点中,谁被反复提及?
              - 共引阈值过滤
           步骤 6 [可选]: 第二轮滚雪球
              - 对第一轮论文进行另一次受控扩展
           步骤 7: 补充最新前沿 (Recent SOTA)
              - 最近 6-12 个月的 arXiv 论文
              - 相似度过滤
           步骤 8: 闭包构建
              - 构建完整网络
        """
        logger.info(f"搜索主题: '{topic}'")

        # 检查是否启用滚雪球搜索
        full_config = load_full_config(self.config.get('llm_config_file', './config/config.yaml'))
        use_snowball = full_config.get('snowball', {}).get('enabled', False)

        if use_snowball:
            logger.info("📊 使用增强滚雪球搜索模式 (完整 8 步方法)")
            # 创建新的 8 步论文搜索管道
            pipeline = PaperSearchPipeline(
                openalex_client=self.openalex_client,
                config_path=self.config.get('llm_config_file', './config/config.yaml'),
                llm_client=self.llm_client
            )

            # 执行完整的 8 步搜索流程
            # 从配置文件读取关键词和 arXiv 类别
            snowball_config = full_config.get('snowball', {})
            keywords = snowball_config.get('search_keywords', None)
            categories = snowball_config.get('arxiv_categories', None)

            # 如果配置为空列表,转换为 None
            if keywords == []:
                keywords = None
            if categories == []:
                categories = None

            logger.info(f"  搜索参数:")
            logger.info(f"    - 主题: {topic}")
            if keywords:
                logger.info(f"    - 关键词: {keywords}")
            if categories:
                logger.info(f"    - arXiv 类别: {categories}")

            result = pipeline.execute_full_pipeline(
                topic=topic,
                keywords=keywords,
                categories=categories
            )

            # 将结果转换为论文列表
            self.papers = list(result['papers'].values())
            self.citation_edges = result['citation_edges']

        else:
            logger.info("🔍 使用传统搜索模式")
            self._traditional_paper_search(topic)

        # 检查是否启用创意评估模式 - 按年份过滤
        idea_eval_config = full_config.get('research_idea', {}).get('idea_evaluation_mode', {})
        if idea_eval_config.get('enabled', False):
            filter_year = idea_eval_config.get('filter_year_after', 2022)
            self._filter_papers_by_year(filter_year)

    def _filter_papers_by_year(self, filter_year_after: int) -> None:
        """
        过滤论文: 保留指定年份之前的论文 (不包括该年份)

        Args:
            filter_year_after: 过滤掉此年份及之后的论文
                              例如, filter_year_after=2022 将保留 2021 年及更早的论文
        """
        original_count = len(self.papers)

        # 过滤论文
        filtered_papers = []
        removed_papers = []

        for paper in self.papers:
            year = paper.get('publication_year') or paper.get('year')
            if year and year < filter_year_after:
                filtered_papers.append(paper)
            else:
                removed_papers.append(paper)

        self.papers = filtered_papers

        # 同时过滤引用边: 移除涉及被过滤论文的边
        if self.citation_edges:
            removed_ids = {p.get('id') for p in removed_papers}
            filtered_edges = []

            for edge in self.citation_edges:
                # citation_edges 是元组格式: (source_id, target_id)
                if isinstance(edge, tuple):
                    source_id, target_id = edge
                else:
                    # 兼容字典格式
                    source_id = edge.get('source')
                    target_id = edge.get('target')

                # 只保留两端都未被过滤的边
                if source_id not in removed_ids and target_id not in removed_ids:
                    filtered_edges.append(edge)

            self.citation_edges = filtered_edges

        # 记录结果
        logger.info("\n" + "="*60)
        logger.info(f"📅 创意评估模式: 年份过滤")
        logger.info("="*60)
        logger.info(f"过滤规则: 保留 {filter_year_after} 年之前的论文")
        logger.info(f"原始论文数量: {original_count} 篇")
        logger.info(f"保留论文数量: {len(self.papers)} 篇")
        logger.info(f"移除论文数量: {len(removed_papers)} 篇")

        if self.citation_edges:
            logger.info(f"引用关系: {len(self.citation_edges)} 条边")

        # 显示保留论文的年份分布
        if self.papers:
            years = [p.get('publication_year') or p.get('year') for p in self.papers if p.get('publication_year') or p.get('year')]
            if years:
                min_year = min(years)
                max_year = max(years)
                logger.info(f"保留论文年份范围: {min_year} - {max_year}")

        logger.info("="*60 + "\n")

    def _traditional_paper_search(self, topic: str) -> None:
        """传统搜索模式"""
        self.papers = self.openalex_client.search_papers(
            topic=topic,
            max_results=self.config['max_papers'],
            sort_by="cited_by_count",
            min_citations=self.config['min_citation_count']
        )

        logger.info(f"✅ 找到 {len(self.papers)} 篇论文")

        # 显示找到的论文
        for i, paper in enumerate(self.papers[:5], 1):  # 只显示前 5 篇
            logger.info(f"  [{i}] {paper['title']} ({paper['year']}) - 引用数: {paper['cited_by_count']}")

        if len(self.papers) > 5:
            logger.info(f"  ... 以及其他 {len(self.papers) - 5} 篇论文")

    def _phase2_pdf_download(self) -> None:
        """阶段 2: PDF 下载 (增强版本, 支持多源和重试)"""
        max_downloads = self.config['max_pdf_downloads']
        logger.info(f"开始 PDF 下载 (最多 {max_downloads} 篇论文)...")
        logger.info("使用 PDF URL 下载 + arXiv 标题搜索下载")

        download_results = self.pdf_downloader.batch_download(
            papers=self.papers,
            max_downloads=max_downloads
        )

        # 详细统计
        logger.info(f"✅ PDF 下载完成:")
        logger.info(f"  📥 成功下载: {download_results['downloaded']} 篇")
        logger.info(f"  📁 已存在: {download_results['exists']} 篇")
        logger.info(f"  ❌ 下载失败: {download_results['failed']} 篇")
        logger.info(f"  📊 总尝试: {download_results['attempted']} / {download_results['total_papers']} 篇")

        # 如果失败率高,提供建议
        if download_results['failed'] > download_results['downloaded']:
            logger.warning("⚠️ 下载失败率较高, 可能原因:")
            logger.warning("  - 论文未提供开放访问 PDF")
            logger.warning("  - 需要订阅或付费访问")
            logger.warning("  - 网络连接问题或服务器限制")
            logger.warning("  - 考虑添加更多 PDF 源或使用机构访问")

    def _phase3_paper_rag_analysis(self) -> None:
        """
        阶段 3: 论文深度分析

        根据配置选择:
        - DeepPaper 多智能体: 带反思循环的迭代多智能体系统
        - 传统 RAG: 单次检索 + LLM 生成

        提取字段: Problem, Method, Limitation, Future Work
        """
        if self.use_deep_paper:
            logger.info(f"🤖 使用 DeepPaper 多智能体分析 {len(self.papers)} 篇论文")
            logger.info("   架构: Navigator → Extractor → Critic → Synthesizer")
            self._analyze_with_deep_paper()
        else:
            logger.info(f"使用传统 RAG 分析器分析 {len(self.papers)} 篇论文")
            self._analyze_with_traditional_rag()

    def _analyze_with_deep_paper(self) -> None:
        """
        使用 DeepPaper 多智能体系统分析论文

        支持 1.0 和 2.0 版本, 自动适配
        """
        from grobid_parser import GrobidPDFParser

        pdf_dir = self.data_dir / 'papers'
        grobid_url = self.config.get('grobid_url')

        # 检查是否使用 DeepPaper 2.0
        is_version_2 = isinstance(self.paper_analyzer, DeepPaper2Orchestrator) if DeepPaper2Orchestrator else False

        # 初始化 GROBID 解析器 (如果可用)
        grobid_parser = None
        if grobid_url:
            try:
                grobid_parser = GrobidPDFParser(grobid_url)
                logger.info(f"   GROBID 服务: {grobid_url}")
            except:
                logger.warning("   GROBID 不可用, 将使用 PyPDF2")

        # 批量分析
        deep_reports = []
        success_count = 0
        pdf_count = 0

        for i, paper in enumerate(self.papers):
            try:
                logger.info(f"\n   [{i+1}/{len(self.papers)}] {paper['title'][:50]}...")

                # 转换为 PaperDocument
                paper_doc = self._convert_to_paper_document(
                    paper, pdf_dir, grobid_parser
                )

                # 使用 DeepPaper 分析
                # 可选: 将单篇论文报告保存到 output/deep_paper/ 目录
                deep_paper_output = self.output_dir / 'deep_paper' if self.config.get('save_deep_paper_reports', False) else None
                if deep_paper_output:
                    deep_paper_output.mkdir(parents=True, exist_ok=True)

                # 根据版本调用不同接口
                if is_version_2:
                    # DeepPaper 2.0: 需要 paper_id 用于引用分析
                    paper_id = paper.get('doi') or paper.get('id', '')
                    report = self.paper_analyzer.analyze_paper(
                        paper_document=paper_doc,
                        paper_id=paper_id,
                        output_dir=str(deep_paper_output) if deep_paper_output else None
                    )
                else:
                    # DeepPaper 1.0: 只需要 paper_document
                    report = self.paper_analyzer.analyze_paper(
                        paper_document=paper_doc,
                        output_dir=str(deep_paper_output) if deep_paper_output else None
                    )

                # 转换为管道格式
                enriched_paper = paper.copy()
                enriched_paper['deep_analysis'] = report.to_dict()

                # 兼容旧格式 (用于引用类型推断)
                enriched_paper['rag_analysis'] = {
                    'problem': report.problem,
                    'method': report.method,
                    'limitation': report.limitation,
                    'future_work': report.future_work
                }

                # 标记使用的版本
                version_tag = 'deep_paper_2.0' if is_version_2 else 'deep_paper_1.0'
                enriched_paper['analysis_method'] = version_tag

                # 提取质量信息 (如果可用)
                if hasattr(report, 'extraction_quality'):
                    enriched_paper['extraction_quality'] = report.extraction_quality
                enriched_paper['sections_extracted'] = len(paper_doc.sections)

                deep_reports.append(enriched_paper)
                success_count += 1

                if len(paper_doc.sections) > 1:
                    pdf_count += 1

            except Exception as e:
                logger.error(f"   ❌ 分析失败: {e}")
                # 添加失败的论文
                failed_paper = paper.copy()
                failed_paper['rag_analysis'] = {
                    'problem': f'分析失败: {str(e)}',
                    'method': '',
                    'limitation': '',
                    'future_work': ''
                }
                failed_paper['analysis_method'] = 'failed'
                deep_reports.append(failed_paper)

        self.enriched_papers = deep_reports

        # 统计
        version_name = "DeepPaper 2.0" if is_version_2 else "DeepPaper 1.0"
        logger.info(f"\n✅ {version_name} 分析完成:")
        logger.info(f"  成功分析: {success_count}/{len(self.papers)} 篇")
        logger.info(f"  有 PDF: {pdf_count} 篇")
        logger.info(f"  仅摘要: {len(self.papers) - pdf_count} 篇")

        # 显示样本
        self._display_analysis_samples(sample_count=2)

    def _analyze_with_traditional_rag(self) -> None:
        """使用传统 RAG 分析器分析论文"""
        logger.info("提取字段: Problem, Method, Limitation, Future Work")

        # 使用 RAG/LLM 分析器进行批量分析
        pdf_dir = str(self.data_dir / 'papers')
        self.enriched_papers = self.paper_analyzer.batch_analyze_papers(
            self.papers,
            pdf_dir=pdf_dir
        )

        # 分析结果统计
        success_count = 0
        with_pdf_count = 0

        for paper in self.enriched_papers:
            rag_analysis = paper.get('rag_analysis', {})
            if rag_analysis and 'error' not in rag_analysis:
                success_count += 1
            if paper.get('sections_extracted', 0) > 0:
                with_pdf_count += 1

        logger.info(f"✅ RAG 分析完成:")
        logger.info(f"  成功分析: {success_count}/{len(self.papers)} 篇")
        logger.info(f"  有 PDF: {with_pdf_count} 篇")
        logger.info(f"  仅摘要: {len(self.papers) - with_pdf_count} 篇")

        # 显示样本
        self._display_analysis_samples(sample_count=2)

    def _convert_to_paper_document(self, paper: Dict, pdf_dir: Path, grobid_parser) -> PaperDocument:
        """将 OpenAlex 论文转换为 PaperDocument 格式"""
        paper_id = paper.get('id', 'unknown')
        title = paper.get('title', 'Untitled')
        abstract = paper.get('abstract', '')
        authors = [
            author.get('author', {}).get('display_name', 'Unknown')
            for author in paper.get('authorships', [])
        ]
        year = paper.get('publication_year')

        # 提取章节
        sections = []

        # 尝试从 PDF 提取
        pdf_path = self._find_pdf(paper_id, pdf_dir)
        if pdf_path and grobid_parser:
            try:
                sections = grobid_parser.extract_sections_from_pdf(pdf_path)
            except:
                pass

        # 回退到摘要
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
        """查找论文 PDF"""
        if not pdf_dir.exists():
            return None

        for pdf_file in pdf_dir.glob(f"{paper_id}*.pdf"):
            return str(pdf_file)

        return None

    def _display_analysis_samples(self, sample_count: int = 2):
        """显示分析样本"""
        sample_count = min(sample_count, len(self.enriched_papers))
        if sample_count == 0:
            return

        logger.info(f"\n📋 分析结果样本 (前 {sample_count} 篇论文):")

        for i, paper in enumerate(self.enriched_papers[:sample_count], 1):
            rag_analysis = paper.get('rag_analysis', {})
            if rag_analysis and 'error' not in rag_analysis:
                logger.info(f"\n  [{i}] {paper['title'][:60]}...")
                logger.info(f"      Analysis method: {paper.get('analysis_method', 'N/A').upper()}")

                # 显示质量分数 (DeepPaper 特有)
                if 'extraction_quality' in paper:
                    quality = paper['extraction_quality']
                    avg_quality = sum(quality.values()) / len(quality) if quality else 0
                    logger.info(f"      平均质量: {avg_quality:.2f}")

                logger.info(f"      章节数: {paper.get('sections_extracted', 0)}")

                problem = rag_analysis.get('problem', '')[:100]
                if problem and problem not in ['无可用内容', '未找到相关信息', '']:
                    logger.info(f"      Problem: {problem}...")

                method = rag_analysis.get('method', '')[:100]
                if method and method not in ['无可用内容', '未找到相关信息', '']:
                    logger.info(f"      Method: {method}...")

    def _phase4_citation_type_inference(self) -> None:
        """
        阶段 4: 引用关系类型推断 (Socket Matching)

        使用 Socket Matching 方法为每个引用关系推断语义类型:

        🔌 Socket Matching 核心思想:
        使用论文的深度信息 (Problem, Method, Limitation, Future_Work)
        作为 "插座", 并使用 LLM Agent 判断这些插座是否可以连接。

        📊 支持的关系类型 (Socket Matching - 6 种类型):
        1. Overcomes - 克服/优化 (垂直深化)
           B 解决了 A 的局限性
           来源: 匹配 1 (Limitation→Problem)

        2. Realizes - 实现愿景 (研究继承)
           B 实现了 A 在 Future Work 中的建议
           来源: 匹配 2 (Future_Work→Problem)

        3. Extends - 方法扩展 (渐进式创新)
           B 基于 A 的方法进行增量改进
           来源: 匹配 3 扩展

        4. Alternative - 替代方法 (颠覆式创新)
           B 使用完全不同的范式解决类似问题
           来源: 匹配 3 替代

        5. Adapts_to - 技术迁移 (横向扩散)
           B 将 A 的方法应用到新领域/场景
           来源: 匹配 4 (Problem→Problem 跨领域)

        6. Baselines - 基线比较 (背景噪声)
           B 仅将 A 作为比较对象, 无直接继承
           来源: 无匹配

        🔗 逻辑连接矩阵 (4 个匹配 → 6 种类型):
        - 匹配 1: A.Limitation ↔ B.Problem → Overcomes
        - 匹配 2: A.Future_Work ↔ B.Problem → Realizes
        - 匹配 3: A.Method ↔ B.Method → Extends / Alternative
        - 匹配 4: A.Problem ↔ B.Problem(跨领域) → Adapts_to
        - 无匹配 → Baselines
        """
        logger.info("开始引用关系类型推断 (Socket Matching)...")

        # 检查是否使用 LLM 模式
        if self.citation_type_inferencer.llm_client:
            logger.info("  🔌 使用 LLM Socket Matching 模式")
            logger.info("  ⏳ 预计每条边 12-20 秒 (5 次 LLM 调用)")
        else:
            logger.info("  📏 使用基于规则的方法模式 (降级)")

        # 使用推断器批量推断引用类型
        self.typed_citation_edges, edge_type_statistics = \
            self.citation_type_inferencer.infer_edge_types(
                papers=self.enriched_papers,
                citation_edges=self.citation_edges
            )

        logger.info(f"✅ 引用关系类型推断完成:")
        logger.info(f"  总引用关系: {len(self.typed_citation_edges)} 条边")
        logger.info(f"  标注类型: {len(edge_type_statistics)} 种")

        # 显示推断策略说明
        if self.citation_type_inferencer.llm_client:
            logger.info(f"\n📊 Socket Matching 推断策略:")
            logger.info(f"  • 深度语义分析: 基于 Problem, Method, Limitation, Future_Work")
            logger.info(f"  • 4 个插座连接 → 6 种类型:")
            logger.info(f"    - 匹配 1: Limitation↔Problem → Overcomes")
            logger.info(f"    - 匹配 2: FutureWork↔Problem → Realizes")
            logger.info(f"    - 匹配 3: Method↔Method → Extends / Alternative")
            logger.info(f"    - 匹配 4: Problem↔Problem(跨领域) → Adapts_to")
            logger.info(f"  • LLM 验证层: 引用上下文证据验证")
            logger.info(f"  • 综合分类: 基于所有匹配结果的最终分类")
        else:
            logger.info(f"\n📊 基于规则的推断策略:")
            logger.info(f"  • 基于时间差: 10+ 年 → 经典/历史引用, 5+ 年 → 扩展/背景引用, 2 年内 → 当代/比较")
            logger.info(f"  • 基于引用数: 高引用论文 → 权威引用")
            logger.info(f"  • 基于文本相似度: 使用简单词汇重叠计算")
            logger.info(f"  • 综合判断: 结合多维信息推断最合适的关系类型")

    def _phase5_knowledge_graph(self) -> None:
        """
        阶段 5: 知识图谱构建和可视化

        从论文和引用关系构建知识图谱, 并:
        1. 添加论文节点 (包括 RAG 分析结果)
        2. 添加引用边 (使用阶段 4 推断的边类型)
        3. 计算图指标
        4. 生成交互式可视化
        """
        logger.info("构建知识图谱...")

        # 构建图 (使用类型化的引用边)
        self.citation_graph.build_citation_network(
            papers=self.enriched_papers,
            citation_data=self.typed_citation_edges  # 使用阶段 4 推断的类型化边
        )

        # 计算图指标
        metrics = self.citation_graph.compute_metrics()
        logger.info(f"✅ 知识图谱构建完成:")
        logger.info(f"    节点数: {metrics.get('total_nodes', 0)}")
        logger.info(f"    边数: {metrics.get('total_edges', 0)}")
        logger.info(f"    图密度: {metrics.get('density', 0):.4f}")

    def _phase6_deep_survey_generation(self, topic: str) -> None:
        """
        阶段 6: 深度综述生成

        核心方法: 基于关系的图剪枝 + 关键演化路径识别

        执行三个步骤:
        1. 基于关系的图剪枝
           - 保留种子论文
           - 仅保留通过强逻辑关系连接到种子的论文
             (Overcomes, Realizes, Extends, Alternative, Adapts_to)
           - 移除仅通过弱关系 (Baselines) 连接或孤立的论文
           - 解决 "数据噪声" 问题

        2. 关键演化路径识别
           - 识别线性链 (The Chain): A -> Overcomes -> B -> Extends -> C
           - 识别星形爆发 (The Star): Seed -> [多条路线]
           - 为每条演化路径生成叙事单元
           - 解决 "碎片化" 问题

        3. 结构化深度综述报告
           - 以线程形式展示每个演化故事
           - 附带可视化图表和文本解释
           - 包括关系链、论文信息、引用统计等

        输出结果:
        - self.deep_survey_report: 包括剪枝统计、演化路径、综述报告等的完整结果
        """
        # 检查是否启用
        full_config = load_full_config(self.config.get('llm_config_file', './config/config.yaml'))
        if not full_config.get('deep_survey', {}).get('enabled', True):
            logger.info("深度综述生成已禁用, 跳过")
            self.deep_survey_report = {}
            return

        logger.info("开始深度综述生成...")

        # 获取图
        G = self.citation_graph.graph

        if len(G.nodes()) == 0:
            logger.warning("知识图谱为空, 跳过深度综述生成")
            self.deep_survey_report = {}
            return

        # 使用深度综述分析器执行分析
        self.deep_survey_report = self.deep_survey_analyzer.analyze(G, topic)
        logger.info("✅ 深度综述生成完成")

    def _phase7_research_idea_generation(self, topic: str) -> None:
        """
        阶段 7: 研究创意生成 (带思维链和演化路径的假设生成器)

        核心方法: 使用思维链推理生成可行的研究创意
        增强功能: 整合深度综述的演化路径, 学习演化逻辑

        三步推理过程:
        - 步骤 1: 分析兼容性
          检查候选方法的数学/算法/理论属性是否与局限性兼容
        - 步骤 2: 识别差距
          确定需要哪些具体修改来弥合差距, 找到 "桥接变量"
        - 步骤 3: 起草创意
          生成标题、摘要 (背景 → 差距 → 提出的方法 → 预期结果)

        演化路径学习 (新):
        - 从阶段 6 深度综述结果中提取演化路径
        - 学习演化逻辑 (链式/发散/收敛)
        - 参考历史成功案例的演化模式
        - 更智能地组合 Limitation 和 Method

        输入来源:
        - 未解决的局限性: 从图节点的 rag_limitation 和 limitations 字段提取
        - 候选方法: 从图节点的 rag_method 字段提取
        - 演化路径: 从阶段 6 的 deep_survey_report 提取

        输出状态:
        - SUCCESS: 生成可行的创新创意
        - INCOMPATIBLE: 方法和局限性根本不兼容
        - ERROR: 生成过程中发生错误
        """
        # 检查是否启用
        full_config = load_full_config(self.config.get('llm_config_file', './config/config.yaml'))
        if not full_config.get('research_idea', {}).get('enabled', True):
            logger.info("研究创意生成已禁用, 跳过")
            self.research_ideas = {}
            return

        logger.info("开始研究创意生成 (使用思维链推理 + 演化路径学习)...")

        # Get knowledge graph
        G = self.citation_graph.graph

        if len(G.nodes()) == 0:
            logger.warning("知识图谱为空, 跳过研究创意生成")
            self.research_ideas = {}
            return

        # 从阶段 6 深度综述结果中提取演化路径
        evolutionary_paths = None
        if self.deep_survey_report and isinstance(self.deep_survey_report, dict):
            evolutionary_paths = self.deep_survey_report.get('evolutionary_paths', [])
            if evolutionary_paths:
                logger.info(f"  从深度综述中提取了 {len(evolutionary_paths)} 条演化路径")
                logger.info("  将使用演化路径学习演化逻辑并生成更智能的研究创意")
            else:
                logger.info("  深度综述中未找到演化路径, 将使用标准模式生成创意")
        else:
            logger.info("  深度综述结果不可用, 将使用标准模式生成创意")

        # 使用 ResearchIdeaGenerator 直接从知识图谱生成创意
        # 内部将:
        # 1. 从图节点提取局限性和方法 (片段池化)
        # 2. 执行 limitation × method 的笛卡尔积匹配
        # 3. 通过思维链推理过滤可行解决方案
        # 4. 整合演化路径信息, 学习演化逻辑 (新)
        self.research_ideas = self.research_idea_generator.generate_from_knowledge_graph(
            graph=G,
            topic=topic,
            evolutionary_paths=evolutionary_paths,  # 传递演化路径
            verbose=True
        )

        logger.info(f"✅ 研究创意生成完成, 成功生成 {self.research_ideas.get('successful_ideas', 0)} 个创意")

    def _phase8_output_results(self, topic: str) -> Dict:
        """阶段 8: 输出结果和报告生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_safe = topic.replace(" ", "_").replace("/", "_")

        # 获取图指标
        graph_metrics = self.citation_graph.compute_metrics()

        # 1. 保存论文数据
        papers_file = self.output_dir / f"papers_{topic_safe}_{timestamp}.json"
        with open(papers_file, 'w', encoding='utf-8') as f:
            json.dump(self.enriched_papers, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)
        logger.info(f"📄 论文数据已保存: {papers_file}")

        # 2. 保存图数据
        graph_file = self.output_dir / f"graph_data_{topic_safe}_{timestamp}.json"
        self.citation_graph.export_graph_data(str(graph_file))
        logger.info(f"🔗 图数据已保存: {graph_file}")

        # 3. 生成可视化
        viz_file = self.output_dir / f"graph_viz_{topic_safe}_{timestamp}.html"
        max_nodes_in_viz = self.config.get('max_nodes_in_viz', 100)  # 从配置读取, 默认 100
        self.citation_graph.visualize_graph(
            str(viz_file),
            max_nodes=max_nodes_in_viz,
            deep_survey_report=self.deep_survey_report,
            research_ideas=self.research_ideas
        )
        logger.info(f"📊 可视化文件已保存: {viz_file} (显示 {max_nodes_in_viz} 个节点, 包括深度综述和研究创意)")

        # 4. 保存深度综述报告
        deep_survey_file = self.output_dir / f"deep_survey_{topic_safe}_{timestamp}.json"
        with open(deep_survey_file, 'w', encoding='utf-8') as f:
            json.dump(self.deep_survey_report, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)
        logger.info(f"📝 深度综述报告已保存: {deep_survey_file}")

        # 5. 保存研究创意报告
        research_ideas_file = self.output_dir / f"research_ideas_{topic_safe}_{timestamp}.json"
        with open(research_ideas_file, 'w', encoding='utf-8') as f:
            json.dump(self.research_ideas, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)
        logger.info(f"💡 研究创意报告已保存: {research_ideas_file}")

        # 收集种子节点 ID
        seed_ids = [p.get('id') for p in self.enriched_papers if p.get('is_seed', False)]

        # 7. 生成摘要结果
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

                'analysis_method': 'multi-agent',  # 标记使用 RAG 方法
            },
            'files': {
                'papers_data': str(papers_file),
                'graph_data': str(graph_file),
                'visualization': str(viz_file),
                'deep_survey': str(deep_survey_file),
                'research_ideas': str(research_ideas_file),
            }
        }

        # 保存摘要结果
        summary_file = self.output_dir / f"summary_{topic_safe}_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)

        logger.info(f"📊 摘要结果已保存: {summary_file}")
        logger.info("\n" + "="*60)
        logger.info("🎉 所有结果文件:")
        for file_type, file_path in results['files'].items():
            logger.info(f"  {file_type}: {file_path}")
        logger.info(f"  summary: {summary_file}")
        logger.info("="*60)

        # 输出种子节点信息
        if seed_ids:
            logger.info(f"\n🌱 种子节点信息:")
            logger.info(f"  总数: {len(seed_ids)}")
            logger.info(f"  ID 列表: {seed_ids[:3]}{'...' if len(seed_ids) > 3 else ''}")
            logger.info("="*60)

        return results

    def get_stats(self) -> Dict:
        """获取当前统计信息"""
        return {
            'papers_count': len(self.papers),
            'citation_edges_count': len(self.citation_edges),
            'enriched_papers_count': len(self.enriched_papers),
            'pdf_stats': self.pdf_downloader.get_download_stats()
        }

    def load_from_cache(self, cache_file: str) -> bool:
        """从缓存文件加载数据"""
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.papers = data.get('papers', [])
            self.citation_edges = data.get('citation_edges', [])
            self.enriched_papers = data.get('enriched_papers', [])

            logger.info(f"成功从缓存加载数据: {cache_file}")
            return True

        except Exception as e:
            logger.warning(f"从缓存加载数据失败: {e}")
            return False

    def save_to_cache(self, cache_file: str) -> None:
        """将数据保存到缓存文件"""
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

            logger.info(f"数据已保存到缓存: {cache_file}")

        except Exception as e:
            logger.error(f"保存缓存失败: {e}")


if __name__ == "__main__":
    # 测试代码
    pipeline = PaperGraphPipeline()

    # 运行管道
    results = pipeline.run("transformer neural networks")

    print(f"\n🎯 管道运行结果:")
    print(f"  主题: {results['topic']}")
    print(f"  总论文数: {results['summary']['total_papers']}")
    print(f"  成功分析: {results['summary']['successful_analysis']}")
    print(f"  图节点数: {results['summary']['graph_nodes']}")
    print(f"  图边数: {results['summary']['graph_edges']}")
    print(f"  可视化文件: {results['files']['visualization']}")