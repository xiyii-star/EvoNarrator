"""
基于RAG的论文分析器
使用检索增强生成技术从PDF全文中提取结构化信息
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
    """论文章节数据结构"""
    title: str
    content: str
    page_num: int
    section_type: str  # 'abstract', 'introduction', 'method', 'results', 'discussion', 'conclusion', 'references'


@dataclass
class ExtractionQuery:
    """信息提取查询"""
    query_text: str
    target_sections: List[str]  # 优先搜索的章节类型
    keywords: List[str]  # 关键词
    max_results: int = 3


class RAGPaperAnalyzer:
    """
    基于RAG的论文分析器
    通过语义检索和章节识别，智能提取论文中的关键信息
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_modelscope: bool = True):
        """
        初始化RAG论文分析器

        Args:
            model_name: 使用的embedding模型名称
            use_modelscope: 是否使用ModelScope下载模型（默认True）
        """
        self.model_name = model_name
        self.embedder = None
        self.use_modelscope = use_modelscope

        # 检查依赖
        if SentenceTransformer is None:
            logger.warning("sentence-transformers未安装，将使用基于规则的方法")
            self.use_embeddings = False
        else:
            try:
                logger.info(f"加载embedding模型: {model_name}")

                # 优先检查本地模型路径
                local_model_path = self._get_local_model_path(model_name)
                if local_model_path and os.path.exists(local_model_path):
                    logger.info(f"  使用本地模型: {local_model_path}")
                    self.embedder = SentenceTransformer(local_model_path)
                # 使用ModelScope下载模型
                elif self.use_modelscope and snapshot_download is not None:
                    try:
                        logger.info("  使用ModelScope下载模型...")
                        model_dir = snapshot_download(
                            f'sentence-transformers/{model_name}',
                            cache_dir='./model',
                            revision='master'
                        )
                        logger.info(f"  模型已下载到: {model_dir}")
                        self.embedder = SentenceTransformer(model_dir)
                    except Exception as e:
                        logger.warning(f"  ModelScope下载失败: {e}，尝试直接加载...")
                        # 降级到直接加载
                        self.embedder = SentenceTransformer(model_name)
                else:
                    # 直接使用HuggingFace（如果ModelScope不可用）
                    if self.use_modelscope and snapshot_download is None:
                        logger.warning("  modelscope未安装，使用HuggingFace下载模型")
                    self.embedder = SentenceTransformer(model_name)

                self.use_embeddings = True
                logger.info("✅ RAG模式已启用")
            except Exception as e:
                logger.warning(f"加载embedding模型失败: {e}，将使用基于规则的方法")
                self.use_embeddings = False

    def _get_local_model_path(self, model_name: str) -> Optional[str]:
        """
        检查本地模型路径是否存在
        
        Args:
            model_name: 模型名称，如 'all-MiniLM-L6-v2'
            
        Returns:
            本地模型路径，如果不存在则返回 None
        """
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

        # 定义章节识别模式
        self.section_patterns = {
            'abstract': [
                r'^abstract\s*$',
                r'^summary\s*$',
            ],
            'introduction': [
                r'^1\.?\s*introduction',
                r'^introduction\s*$',
                r'^1\.?\s*背景',
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

        # 定义提取查询
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

        logger.info("RAG论文分析器初始化完成")

    def analyze_paper(self, paper: Dict, pdf_path: Optional[str] = None) -> Dict:
        """
        分析论文并提取深层信息

        自动提取四个字段：Problem, Contribution, Limitation, Future Work
        如果章节切割失败或没有找到目标章节，自动使用摘要

        Args:
            paper: 基础论文信息
            pdf_path: PDF文件路径

        Returns:
            包含分析结果的论文字典
        """
        paper_id = paper.get('id', 'unknown')
        logger.info(f"开始RAG分析论文: {paper_id}")

        # 提取PDF内容和识别章节
        sections = []
        if pdf_path and Path(pdf_path).exists():
            sections = self._extract_sections_from_pdf(pdf_path)
            if sections:
                logger.info(f"从PDF提取了 {len(sections)} 个章节")
            else:
                logger.warning(f"PDF章节提取失败，降级使用摘要")
                sections = self._create_sections_from_abstract(paper)
        else:
            logger.info("PDF不存在，使用摘要构建章节")
            sections = self._create_sections_from_abstract(paper)

        # 如果连摘要都没有，创建一个基于标题的最小章节
        if not sections:
            logger.warning("无摘要，仅使用标题进行分析")
            if paper.get('title'):
                sections = [PaperSection(
                    title='Title',
                    content=paper['title'],
                    page_num=0,
                    section_type='title'
                )]

        # 如果使用embeddings，预计算章节向量
        section_embeddings = None
        if self.use_embeddings and sections and self.embedder:
            try:
                section_texts = [f"{s.title} {s.content}" for s in sections]
                section_embeddings = self.embedder.encode(section_texts)
                logger.info("章节向量化完成")
            except Exception as e:
                logger.warning(f"章节向量化失败: {e}，将使用关键词检索")
                section_embeddings = None

        # 执行RAG检索和信息提取
        # 自动提取所有四个字段
        analysis_result = {}
        extraction_fields = ['problem', 'method', 'limitation', 'future_work']

        for field in extraction_fields:
            if field in self.extraction_queries:
                logger.info(f"正在提取 {field}...")
                try:
                    analysis_result[field] = self._extract_with_rag(
                        sections, section_embeddings, self.extraction_queries[field]
                    )
                except Exception as e:
                    logger.error(f"提取 {field} 失败: {e}")
                    analysis_result[field] = f"提取失败: {str(e)}"

        # 创建增强的论文数据
        enriched_paper = paper.copy()
        enriched_paper['rag_analysis'] = analysis_result
        enriched_paper['sections_extracted'] = len(sections)
        enriched_paper['analysis_method'] = 'rag' if self.use_embeddings else 'rule_based'

        logger.info(f"RAG论文分析完成: {paper_id}")
        return enriched_paper

    def _extract_sections_from_pdf(self, pdf_path: str) -> List[PaperSection]:
        """从PDF中提取并识别章节"""
        if PyPDF2 is None:
            logger.error("PyPDF2未安装，无法提取PDF内容")
            return []

        sections = []

        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)

                logger.info(f"PDF总页数: {total_pages}")

                # 逐页提取文本
                full_text = ""
                page_texts = []

                for page_num in range(total_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        page_texts.append((page_num, page_text))
                        full_text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"提取第{page_num}页失败: {e}")
                        continue

                # 识别章节
                sections = self._identify_sections(full_text, page_texts)

                logger.info(f"识别到 {len(sections)} 个章节")
                return sections

        except Exception as e:
            logger.error(f"PDF处理失败 {pdf_path}: {e}")
            return []

    def _identify_sections(self, full_text: str, page_texts: List[Tuple[int, str]]) -> List[PaperSection]:
        """识别文本中的章节"""
        sections = []
        lines = full_text.split('\n')

        current_section = None
        current_content = []
        current_page = 0

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            if not line_stripped:
                continue

            # 检查是否是章节标题
            section_type = self._match_section_type(line_stripped)

            if section_type:
                # 保存前一个章节
                if current_section:
                    content = '\n'.join(current_content).strip()
                    if content:
                        sections.append(PaperSection(
                            title=current_section,
                            content=content,
                            page_num=current_page,
                            section_type=section_type
                        ))

                # 开始新章节
                current_section = line_stripped
                current_content = []
            else:
                # 添加到当前章节内容
                if current_section:
                    current_content.append(line_stripped)

        # 保存最后一个章节
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
        """匹配章节类型"""
        line_lower = line.lower().strip()

        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                if re.match(pattern, line_lower, re.IGNORECASE):
                    return section_type

        return None

    def _create_sections_from_abstract(self, paper: Dict) -> List[PaperSection]:
        """从摘要创建简单章节"""
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
        使用RAG方法提取信息

        如果没有找到目标章节或关键词匹配的句子，自动降级使用摘要
        """

        if not sections:
            return "无可用内容"

        # 第一步：过滤目标章节
        target_sections = [
            s for s in sections
            if s.section_type in query.target_sections
        ]

        # 如果目标章节为空，尝试使用所有章节
        if not target_sections:
            logger.warning(f"未找到目标章节 {query.target_sections}，使用所有可用章节")
            target_sections = sections

        # 第二步：基于关键词过滤句子
        relevant_sentences = []

        for section in target_sections:
            sentences = self._split_into_sentences(section.content)

            for sentence in sentences:
                # 检查关键词匹配
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

        # 第三步：如果启用了embeddings，使用语义检索
        if self.use_embeddings and relevant_sentences and self.embedder:
            # 对候选句子进行向量化
            sentence_texts = [s['text'] for s in relevant_sentences]
            sentence_embeddings = self.embedder.encode(sentence_texts)

            # 对查询进行向量化
            query_embedding = self.embedder.encode([query.query_text])

            # 计算相似度
            similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]

            # 添加相似度分数
            for i, sent_dict in enumerate(relevant_sentences):
                sent_dict['similarity'] = similarities[i]

            # 综合排序：关键词数量 + 语义相似度
            relevant_sentences.sort(
                key=lambda x: x['keyword_count'] * 0.3 + x['similarity'] * 0.7,
                reverse=True
            )
        else:
            # 仅基于关键词排序
            relevant_sentences.sort(key=lambda x: x['keyword_count'], reverse=True)

        # 第四步：提取top-k结果
        if not relevant_sentences:
            return "未找到相关信息"

        top_sentences = relevant_sentences[:query.max_results]
        result_text = ' '.join([s['text'] for s in top_sentences])

        return result_text

    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割为句子"""
        # 使用正则表达式分割句子
        sentences = re.split(r'[.!?]+\s+', text)

        # 过滤太短的句子
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        return sentences

    def batch_analyze_papers(
        self,
        papers: List[Dict],
        pdf_dir: Optional[str] = None
    ) -> List[Dict]:
        """批量分析论文"""
        logger.info(f"开始批量RAG分析 {len(papers)} 篇论文")

        enriched_papers = []

        for i, paper in enumerate(papers):
            try:
                # 查找对应的PDF文件
                pdf_path = None
                if pdf_dir:
                    paper_id = paper.get('id', '')
                    pdf_dir_path = Path(pdf_dir)

                    # 查找匹配的PDF文件
                    for pdf_file in pdf_dir_path.glob(f"{paper_id}*.pdf"):
                        pdf_path = str(pdf_file)
                        break

                enriched_paper = self.analyze_paper(paper, pdf_path)
                enriched_papers.append(enriched_paper)

                logger.info(f"进度: {i+1}/{len(papers)}")

            except Exception as e:
                logger.error(f"分析论文失败 {paper.get('id', 'unknown')}: {e}")
                failed_paper = paper.copy()
                failed_paper['rag_analysis'] = {
                    'error': str(e)
                }
                enriched_papers.append(failed_paper)

        logger.info(f"批量RAG分析完成")
        return enriched_papers


if __name__ == "__main__":
    # 测试代码
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    analyzer = RAGPaperAnalyzer()

    # 测试论文数据
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

    # 测试分析
    result = analyzer.analyze_paper(test_paper)

    print("\n" + "="*80)
    print("RAG分析结果:")
    print("="*80)
    print(f"\n问题 (Problem):\n{result['rag_analysis']['problem']}\n")
    print(f"贡献 (Contribution):\n{result['rag_analysis']['contribution']}\n")
    print(f"局限性 (Limitation):\n{result['rag_analysis']['limitation']}\n")
    print(f"未来工作 (Future Work):\n{result['rag_analysis']['future_work']}\n")
    print(f"分析方法: {result['analysis_method']}")
    print(f"提取章节数: {result['sections_extracted']}")
    print("="*80)
