"""
基于LLM增强的RAG论文分析器（重构版）

结构清晰、易于理解的版本：
- 使用独立的LLM配置管理器
- 使用独立的提示词管理器
- 清晰的模块划分
"""

import logging
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass

# PDF处理
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# Embedding模型 - sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# 数值计算库 - numpy 和 sklearn（独立导入，不受 sentence-transformers 影响）
try:
    import numpy as np
except ImportError:
    np = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine_similarity = None

# 用于本地模型加载
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModel = None

# ModelScope（国内镜像）
try:
    from modelscope.hub.snapshot_download import snapshot_download
except ImportError:
    snapshot_download = None

# 本地模块
try:
    from llm_config import LLMClient, LLMConfig
    from prompt_manager import PromptManager
    from grobid_parser import GrobidPDFParser
except ImportError:
    # 如果直接导入失败，尝试从src模块导入
    from src.llm_config import LLMClient, LLMConfig
    from src.prompt_manager import PromptManager
    from src.grobid_parser import GrobidPDFParser

logger = logging.getLogger(__name__)


@dataclass
class PaperSection:
    """论文章节数据结构"""
    title: str
    content: str
    page_num: int
    section_type: str


class LLMRAGPaperAnalyzer:
    """
    LLM增强的RAG论文分析器

    主要功能：
    1. 从PDF或摘要中提取论文章节
    2. 使用RAG检索相关内容
    3. 使用LLM生成高质量分析
    4. 自动提取四个关键字段：Problem, Contribution, Limitation, Future Work
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
        初始化分析器

        Args:
            llm_config_path: LLM配置文件路径
            embedding_model: Embedding模型名称
            use_modelscope: 是否使用ModelScope下载模型
            prompts_dir: 提示词文件夹路径
            max_context_length: LLM上下文最大长度
            grobid_url: GROBID服务URL（可选，如 http://localhost:8070）
            local_model_path: 本地模型路径（可选，如 ./model/sentence-transformers/all-MiniLM-L6-v2）
        """
        logger.info("="*60)
        logger.info("初始化LLM RAG论文分析器")
        logger.info("="*60)

        # 基本配置
        self.embedding_model_name = embedding_model
        self.use_modelscope = use_modelscope
        self.max_context_length = max_context_length
        self.grobid_url = grobid_url
        self.local_model_path = local_model_path

        # 初始化GROBID解析器（如果提供了URL）
        self.grobid_parser = None
        if grobid_url:
            self._init_grobid_parser()

        # 初始化embedding模型
        self.embedder = None
        self.use_embeddings = False
        self._init_embedding_model()

        # 初始化LLM客户端
        self.llm_client = self._init_llm_client(llm_config_path)

        # 初始化提示词管理器
        self.prompt_manager = PromptManager(prompts_dir)

        # 章节识别模式
        self.section_patterns = self._get_section_patterns()

        # 要提取的字段
        self.extraction_fields = ['problem', 'method', 'limitation', 'future_work']

        logger.info("="*60)
        logger.info("✅ LLM RAG论文分析器初始化完成")
        logger.info("="*60)

    def _init_embedding_model(self):
        """初始化Embedding模型"""
        # 检查 sentence-transformers 是否安装
        if SentenceTransformer is None:
            logger.warning("⚠️ sentence-transformers未安装，将使用纯关键词检索")
            logger.warning("   安装命令: pip install sentence-transformers")
            self.use_embeddings = False
            return

        # 如果没有本地模型路径，尝试自动检测
        if not self.local_model_path:
            self.local_model_path = self._get_local_model_path(self.embedding_model_name)

        # 优先使用本地模型路径
        if self.local_model_path:
            try:
                import os
                if os.path.exists(self.local_model_path):
                    logger.info(f"  🔍 检测到本地模型: {self.local_model_path}")
                    logger.info(f"  📦 正在加载本地Embedding模型...")
                    self.embedder = SentenceTransformer(self.local_model_path)
                    self.use_embeddings = True
                    logger.info(f"  ✅ 本地Embedding模型加载成功!")
                    return
                else:
                    logger.warning(f"  ⚠️ 本地模型路径不存在: {self.local_model_path}")
            except Exception as e:
                logger.warning(f"  ❌ 本地模型加载失败: {e}，尝试下载...")

        # 如果本地模型加载失败，尝试下载

        try:
            logger.info(f"加载Embedding模型: {self.embedding_model_name}")

            # 使用ModelScope镜像（国内更快）
            if self.use_modelscope and snapshot_download is not None:
                try:
                    logger.info("  使用ModelScope镜像...")
                    model_dir = snapshot_download(
                        f'sentence-transformers/{self.embedding_model_name}',
                        cache_dir='./model',
                        revision='master'
                    )
                    self.embedder = SentenceTransformer(model_dir)
                    logger.info(f"  ✅ 模型已从ModelScope下载: {model_dir}")
                except Exception as e:
                    logger.warning(f"  ModelScope下载失败: {e}，尝试HuggingFace...")
                    self.embedder = SentenceTransformer(self.embedding_model_name)
            else:
                self.embedder = SentenceTransformer(self.embedding_model_name)

            self.use_embeddings = True
            logger.info("  ✅ Embedding模型加载成功")

        except Exception as e:
            logger.warning(f"  ❌ Embedding模型加载失败: {e}，将使用纯关键词检索")
            self.use_embeddings = False

    def _get_local_model_path(self, model_name: str) -> Optional[str]:
        """
        检查本地模型路径是否存在
        
        Args:
            model_name: 模型名称，如 'all-MiniLM-L6-v2'
            
        Returns:
            本地模型路径，如果不存在则返回 None
        """
        import os
        from pathlib import Path
        
        # 尝试多个可能的本地路径
        possible_paths = [
            # 相对于当前文件的路径
            Path(__file__).parent.parent / "model" / "sentence-transformers" / model_name,
            # 相对于项目根目录的路径
            Path(__file__).parent.parent.parent / "KGdemo" / "model" / "sentence-transformers" / model_name,
            # 绝对路径
            Path("/home/lexy/下载/CLwithRAG/KGdemo/model/sentence-transformers") / model_name,
        ]
        
        for path in possible_paths:
            if path.exists() and (path / "modules.json").exists():
                return str(path)
        
        return None

    def _init_grobid_parser(self):
        """初始化GROBID解析器"""
        try:
            logger.info(f"初始化GROBID解析器: {self.grobid_url}")
            self.grobid_parser = GrobidPDFParser(self.grobid_url)
            logger.info("✅ GROBID解析器已启用")
        except Exception as e:
            logger.warning(f"⚠️ GROBID解析器初始化失败: {e}，将使用正则表达式方法")
            self.grobid_parser = None

    def _init_llm_client(self, config_path: Optional[str]) -> Optional[LLMClient]:
        """初始化LLM客户端"""
        # 如果配置路径为None，则不使用LLM
        if config_path is None:
            logger.info("⚠️ LLM配置路径为None，跳过LLM初始化（将使用基础分析模式）")
            return None

        try:
            logger.info(f"加载LLM配置: {config_path}")
            config = LLMConfig.from_file(config_path)

            logger.info(f"  Provider: {config.provider}")
            logger.info(f"  Model: {config.model}")

            client = LLMClient(config)
            return client

        except FileNotFoundError:
            logger.warning(f"⚠️ LLM配置文件不存在: {config_path}")
            return None
        except Exception as e:
            logger.error(f"❌ 初始化LLM客户端失败: {e}")
            return None

    def _get_section_patterns(self) -> Dict[str, List[str]]:
        """定义章节识别的正则表达式模式"""
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

    # ========== 核心分析方法 ==========

    def analyze_paper(self, paper: Dict, pdf_path: Optional[str] = None) -> Dict:
        """
        分析论文并提取关键信息

        自动提取四个字段：Problem, Contribution, Limitation, Future Work
        支持多级降级策略：PDF → 摘要 → 标题

        Args:
            paper: 论文基础信息字典
            pdf_path: PDF文件路径（可选）

        Returns:
            包含分析结果的增强论文字典
        """
        paper_id = paper.get('id', 'unknown')
        logger.info(f"\n{'='*60}")
        logger.info(f"📄 开始分析论文: {paper_id}")
        logger.info(f"{'='*60}")

        # 步骤1: 提取章节内容并判断是否成功提取PDF
        sections, pdf_extracted = self._extract_paper_sections(paper, pdf_path)

        # 判断是否成功从PDF提取了章节
        if pdf_extracted:
            logger.info("  ✅ PDF提取成功，使用RAG检索模式")
        else:
            logger.info("  ⚠️ PDF未提取，使用摘要直接生成模式")

        # 步骤2: 计算章节向量（仅当PDF提取成功时）
        section_embeddings = None
        if pdf_extracted:
            section_embeddings = self._compute_section_embeddings(sections)

        # 步骤3: 提取所有字段（传入pdf_extracted标志）
        analysis_result = self._extract_all_fields(sections, section_embeddings, pdf_extracted, paper)

        # 步骤4: 构建结果
        enriched_paper = paper.copy()
        enriched_paper['rag_analysis'] = analysis_result
        enriched_paper['sections_extracted'] = len(sections)
        enriched_paper['section_types'] = [s.section_type for s in sections]
        enriched_paper['pdf_extracted'] = pdf_extracted
        enriched_paper['analysis_method'] = f'llm_rag_{self.llm_client.config.provider if self.llm_client else "none"}'

        logger.info(f"✅ 论文分析完成: {paper_id}")
        logger.info(f"   提取字段数: {len(analysis_result)}")
        logger.info(f"   章节数: {len(sections)}")
        logger.info(f"   分析模式: {'RAG检索' if pdf_extracted else '摘要直接生成'}")
        logger.info(f"{'='*60}\n")

        return enriched_paper

    def _extract_paper_sections(self, paper: Dict, pdf_path: Optional[str]) -> tuple[List[PaperSection], bool]:
        """
        提取论文章节（支持多级降级）

        降级策略:
        1. 尝试从PDF提取章节
        2. 如果失败，使用摘要构建章节
        3. 如果连摘要都没有，使用标题

        Args:
            paper: 论文信息
            pdf_path: PDF路径

        Returns:
            (章节列表, PDF是否成功提取的标志)
        """
        sections = []

        # Level 1: 尝试PDF提取
        if pdf_path and Path(pdf_path).exists():
            logger.info("  [1/3] 尝试从PDF提取章节...")
            sections = self._extract_sections_from_pdf(pdf_path)

            if sections:
                logger.info(f"  ✅ 从PDF提取了 {len(sections)} 个章节")
                return sections, True  # PDF提取成功
            else:
                logger.warning("  ❌ PDF章节提取失败")

        # Level 2: 降级到摘要
        logger.info("  [2/3] 降级使用摘要...")
        sections = self._create_sections_from_abstract(paper)

        if sections:
            logger.info(f"  ✅ 从摘要构建了 {len(sections)} 个章节")
            return sections, False  # 使用摘要,PDF未提取

        # Level 3: 降级到标题
        logger.info("  [3/3] 降级使用标题...")
        if paper.get('title'):
            sections = [PaperSection(
                title='Title',
                content=paper['title'],
                page_num=0,
                section_type='title'
            )]
            logger.info("  ✅ 使用标题作为最小内容")

        return sections, False  # 使用标题或空,PDF未提取

    def _encode_texts(self, texts):
        """
        统一的文本编码接口
        支持 sentence-transformers 和本地 transformers 模型
        """
        if self.embedder:
            # 使用 sentence-transformers
            return self.embedder.encode(texts)
        elif hasattr(self, 'tokenizer') and hasattr(self, 'model'):
            # 使用本地 transformers 模型
            import torch

            # Mean Pooling - 取平均池化
            def mean_pooling(model_output, attention_mask):
                token_embeddings = model_output[0]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            # 编码文本
            encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)

            with torch.no_grad():
                model_output = self.model(**encoded_input)

            # 执行池化
            embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

            # 归一化
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            return embeddings.numpy()
        else:
            return None

    def _compute_section_embeddings(self, sections: List[PaperSection]) -> Optional[any]:
        """计算章节向量"""
        if not self.use_embeddings or not sections:
            return None

        try:
            logger.info("  🔄 计算章节向量...")
            section_texts = [f"{s.title} {s.content}" for s in sections]
            embeddings = self._encode_texts(section_texts)

            if embeddings is not None:
                logger.info(f"  ✅ 章节向量计算完成 ({len(embeddings)} 个向量)")
                return embeddings
            else:
                logger.warning("  ❌ 编码器未正确初始化")
                return None

        except Exception as e:
            logger.warning(f"  ❌ 章节向量计算失败: {e}")
            return None

    def _extract_all_fields(
        self,
        sections: List[PaperSection],
        section_embeddings: Optional[any],
        pdf_extracted: bool,
        paper: Dict
    ) -> Dict[str, str]:
        """
        自动提取所有四个字段

        Args:
            sections: 论文章节列表
            section_embeddings: 章节向量（可选）
            pdf_extracted: PDF是否成功提取
            paper: 原始论文信息（用于获取摘要）

        Returns:
            {field: extracted_value}
        """
        logger.info("  🔍 开始提取关键字段...")

        results = {}

        for field in self.extraction_fields:
            logger.info(f"     • 提取 {field}...")

            try:
                value = self._extract_single_field(field, sections, section_embeddings, pdf_extracted, paper)
                results[field] = value

            except Exception as e:
                logger.error(f"     ❌ 提取 {field} 失败: {e}")
                results[field] = f"提取失败: {str(e)}"

        logger.info(f"  ✅ 字段提取完成，成功提取 {len(results)} 个字段")
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
        提取单个字段

        流程:
        - 如果PDF提取成功: 使用RAG检索相关上下文 → LLM生成
        - 如果PDF未提取: 直接使用摘要作为上下文 → LLM生成

        Args:
            field: 字段名
            sections: 章节列表
            section_embeddings: 章节向量
            pdf_extracted: PDF是否成功提取
            paper: 原始论文信息

        Returns:
            提取的内容
        """
        if not sections:
            return "无可用内容"

        # 根据PDF是否提取成功选择不同策略
        if pdf_extracted:
            # 策略1: PDF提取成功 -> 使用RAG检索
            logger.info(f"       使用RAG检索模式提取 {field}")
            relevant_context = self._retrieve_relevant_content(
                field, sections, section_embeddings
            )

            if not relevant_context or relevant_context == "未找到相关信息":
                # RAG检索失败,降级到摘要
                logger.warning(f"       RAG检索未找到相关信息,降级使用摘要")
                relevant_context = self._get_abstract_context(paper)
        else:
            # 策略2: PDF未提取 -> 直接使用摘要
            logger.info(f"       使用摘要直接生成模式提取 {field}")
            relevant_context = self._get_abstract_context(paper)

        if not relevant_context or relevant_context == "无摘要可用":
            return "未找到相关信息"

        # 使用LLM生成
        if self.llm_client:
            return self._generate_with_llm(field, relevant_context)
        else:
            logger.warning("     ⚠️ LLM未配置，返回原始检索内容")
            return relevant_context[:200]  # 返回检索内容的前200字符

    def _get_abstract_context(self, paper: Dict) -> str:
        """
        获取摘要作为上下文

        Args:
            paper: 论文信息

        Returns:
            摘要文本
        """
        abstract = paper.get('abstract', '')
        title = paper.get('title', '')

        if not abstract:
            return "无摘要可用"

        # 构建上下文
        context = f"Title: {title}\n\nAbstract: {abstract}" if title else f"Abstract: {abstract}"
        return context

    # ========== RAG检索 ==========

    def _retrieve_relevant_content(
        self,
        field: str,
        sections: List[PaperSection],
        section_embeddings: Optional[any]
    ) -> str:
        """
        检索与字段相关的内容（RAG核心）

        支持:
        - 目标章节过滤
        - 关键词检索
        - 语义相似度排序（如果有embeddings）
        - 降级到摘要（如果检索失败）

        Args:
            field: 字段名
            sections: 章节列表
            section_embeddings: 章节向量

        Returns:
            相关内容文本
        """
        # 定义目标章节和关键词
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

        # 步骤1: 过滤目标章节
        filtered_sections = [
            s for s in sections
            if s.section_type in target_section_types
        ] if target_section_types else sections

        if not filtered_sections:
            logger.info(f"       未找到目标章节 {target_section_types}，使用所有章节")
            filtered_sections = sections

        # 步骤2: 关键词检索
        relevant_chunks = []

        for section in filtered_sections:
            paragraphs = self._split_into_paragraphs(section.content)

            for paragraph in paragraphs:
                # 计算关键词匹配数
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

        # 步骤3: 如果没找到，降级到摘要
        if not relevant_chunks:
            logger.info(f"       关键词检索未找到匹配，降级使用摘要")
            abstract_sections = [s for s in sections if s.section_type == 'abstract']

            if abstract_sections:
                abstract_text = abstract_sections[0].content
                return f"[Abstract (Fallback)]\n{abstract_text[:self.max_context_length]}"
            else:
                # 使用前两个章节
                all_content = "\n\n".join([f"[{s.title}]\n{s.content}" for s in sections[:2]])
                return all_content[:self.max_context_length] if all_content else "未找到相关信息"

        # 步骤4: 排序（关键词 or 语义相似度）
        if self.use_embeddings and section_embeddings is not None:
            # 使用语义相似度排序
            query_text = f"extract {field} from paper"
            chunk_texts = [c['text'] for c in relevant_chunks]

            try:
                chunk_embeddings = self._encode_texts(chunk_texts)
                query_embedding = self._encode_texts([query_text])

                if chunk_embeddings is not None and query_embedding is not None:
                    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

                    for i, chunk in enumerate(relevant_chunks):
                        chunk['similarity'] = similarities[i]

                    # 综合排序（关键词30% + 语义70%）
                    relevant_chunks.sort(
                        key=lambda x: x['keyword_count'] * 0.3 + x['similarity'] * 0.7,
                        reverse=True
                    )
                else:
                    # 编码失败，降级到关键词排序
                    relevant_chunks.sort(key=lambda x: x['keyword_count'], reverse=True)
            except Exception as e:
                # 降级到关键词排序
                logger.warning(f"      语义相似度计算失败: {e}，使用关键词排序")
                relevant_chunks.sort(key=lambda x: x['keyword_count'], reverse=True)
        else:
            # 仅基于关键词排序
            relevant_chunks.sort(key=lambda x: x['keyword_count'], reverse=True)

        # 步骤5: 构建上下文
        context_parts = []
        current_length = 0

        for chunk in relevant_chunks[:5]:  # 取top 5
            chunk_text = f"[{chunk['section']}]\n{chunk['text']}"
            chunk_length = len(chunk_text)

            if current_length + chunk_length > self.max_context_length:
                break

            context_parts.append(chunk_text)
            current_length += chunk_length

        return "\n\n".join(context_parts) if context_parts else "未找到相关信息"

    # ========== LLM生成 ==========

    def _generate_with_llm(self, field: str, context: str) -> str:
        """
        使用LLM生成分析结果

        Args:
            field: 字段名
            context: 检索到的上下文

        Returns:
            LLM生成的分析
        """
        if not self.llm_client:
            return "LLM未配置"

        # 构建完整提示词
        full_prompt = self.prompt_manager.build_full_prompt(field, context)

        # 获取系统提示词
        system_prompt = self.prompt_manager.get_system_prompt()

        # 调用LLM
        result = self.llm_client.generate(
            prompt=full_prompt,
            system_prompt=system_prompt
        )

        return result

    # ========== PDF处理 ==========

    def _extract_sections_from_pdf(self, pdf_path: str) -> List[PaperSection]:
        """
        从PDF或TXT文件中提取章节（混合策略）

        策略:
        1. 如果是.txt文件，直接读取文本
        2. 如果是PDF：优先使用GROBID（如果可用）
        3. 降级到PyPDF2+正则表达式
        """
        # 检查文件扩展名
        file_ext = Path(pdf_path).suffix.lower()

        # 策略0: 如果是.txt文件，直接读取
        if file_ext == '.txt':
            logger.info("  检测到.txt文件，直接读取文本...")
            return self._extract_sections_from_txt(pdf_path)

        # 策略1: 尝试GROBID（仅对PDF）
        if self.grobid_parser:
            try:
                logger.info("  尝试使用GROBID解析PDF...")
                sections = self.grobid_parser.extract_sections_from_pdf(pdf_path)

                if sections:
                    logger.info(f"  ✅ GROBID成功提取 {len(sections)} 个章节")
                    return sections
                else:
                    logger.warning("  ⚠️ GROBID未提取到章节，降级到正则方法")
            except Exception as e:
                logger.warning(f"  ⚠️ GROBID解析失败: {e}，降级到正则方法")

        # 策略2: 降级到PyPDF2+正则表达式
        logger.info("  使用PyPDF2+正则表达式解析PDF...")
        return self._extract_sections_with_pypdf2(pdf_path)

    def _extract_sections_with_pypdf2(self, pdf_path: str) -> List[PaperSection]:
        """使用PyPDF2提取章节（正则表达式方法）"""
        if PyPDF2 is None:
            logger.error("  PyPDF2未安装，无法提取PDF")
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
            logger.error(f"  PDF处理失败: {e}")

        return sections

    def _extract_sections_from_txt(self, txt_path: str) -> List[PaperSection]:
        """从.txt文件直接读取并提取章节"""
        sections = []

        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                full_text = file.read()

            logger.info(f"  ✅ 成功读取.txt文件，共 {len(full_text)} 个字符")

            # 使用相同的章节识别逻辑
            sections = self._identify_sections(full_text)

            if sections:
                logger.info(f"  ✅ 从.txt文件识别出 {len(sections)} 个章节")
            else:
                logger.warning("  ⚠️ 未识别到明确章节，将整个文本作为一个章节")
                # 如果没有识别到章节，将整个文本作为一个章节
                sections = [PaperSection(
                    title='Full Text',
                    content=full_text[:10000],  # 限制长度
                    page_num=0,
                    section_type='other'
                )]

        except Exception as e:
            logger.error(f"  .txt文件处理失败: {e}")

        return sections

    def _identify_sections(self, full_text: str) -> List[PaperSection]:
        """识别文本中的章节"""
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

            # 检查是否是章节标题
            section_type = self._match_section_type(line_stripped)

            if section_type:
                # 保存前一个章节
                if current_section and current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        sections.append(PaperSection(
                            title=current_section,
                            content=content,
                            page_num=0,
                            section_type=current_type
                        ))

                # 开始新章节
                current_section = line_stripped
                current_content = []
                current_type = section_type
            else:
                # 添加到当前章节
                if current_section:
                    current_content.append(line_stripped)

        # 保存最后一个章节
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
        """匹配章节类型"""
        import re

        line_lower = line.lower().strip()

        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                if re.match(pattern, line_lower, re.IGNORECASE):
                    return section_type

        return None

    def _create_sections_from_abstract(self, paper: Dict) -> List[PaperSection]:
        """从摘要创建章节"""
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

    # ========== 工具方法 ==========

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """将文本分割为段落"""
        import re

        paragraphs = re.split(r'\n\s*\n|\n', text)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 30]

        return paragraphs

    # ========== 批量分析 ==========

    def batch_analyze_papers(
        self,
        papers: List[Dict],
        pdf_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        批量分析论文

        Args:
            papers: 论文列表
            pdf_dir: PDF文件夹路径

        Returns:
            增强的论文列表
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"📚 批量分析 {len(papers)} 篇论文")
        logger.info(f"{'='*60}\n")

        enriched_papers = []

        for i, paper in enumerate(papers):
            try:
                # 查找PDF文件
                pdf_path = None
                if pdf_dir:
                    paper_id = paper.get('id', '')
                    pdf_dir_path = Path(pdf_dir)

                    for pdf_file in pdf_dir_path.glob(f"{paper_id}*.pdf"):
                        pdf_path = str(pdf_file)
                        break

                # 分析论文
                enriched_paper = self.analyze_paper(paper, pdf_path)
                enriched_papers.append(enriched_paper)

                logger.info(f"进度: {i+1}/{len(papers)}\n")

            except Exception as e:
                logger.error(f"分析论文失败 {paper.get('id', 'unknown')}: {e}")

                # 添加失败的论文
                failed_paper = paper.copy()
                failed_paper['rag_analysis'] = {
                    'problem': f'分析失败: {str(e)}',
                    'method': f'分析失败: {str(e)}',
                    'limitation': f'分析失败: {str(e)}',
                    'future_work': f'分析失败: {str(e)}'
                }
                enriched_papers.append(failed_paper)

        logger.info(f"{'='*60}")
        logger.info(f"✅ 批量分析完成")
        logger.info(f"{'='*60}\n")

        return enriched_papers


if __name__ == "__main__":
    # 测试代码
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*60)
    print("测试LLM RAG论文分析器（重构版）")
    print("="*60)

    # 测试论文数据
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
        # 创建分析器
        analyzer = LLMRAGPaperAnalyzer(
            llm_config_path='../llm_config_ollama.json',
            prompts_dir='../prompts'
        )

        # 分析论文
        result = analyzer.analyze_paper(test_paper)

        # 显示结果
        print("\n" + "="*60)
        print("分析结果:")
        print("="*60)
        for field, value in result['rag_analysis'].items():
            print(f"\n{field.upper()}:")
            print(value)

        print("\n" + "="*60)
        print("✅ 测试完成")
        print("="*60)

    except Exception as e:
        print(f"❌ 测试失败: {e}")
