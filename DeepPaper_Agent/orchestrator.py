"""
DeepPaper Orchestrator (协调器)
Multi-Agent系统的核心控制器

负责:
1. 管理四个Agent的交互流程
2. 实现Iterative Reflection Loop
3. 控制重试次数和终止条件
4. 收集和整合结果

工作流:
For each field (Problem, Method, Limitation, Future Work):
    1. Navigator → 定位章节
    2. Extractor → 提取内容
    3. Critic → 验证质量
    4. If not approved:
         → 根据Critic反馈重试Extractor (最多max_retries次)
    5. Synthesizer → 整合最终报告
"""

import logging
from typing import Dict, Optional, List
from pathlib import Path

from .navigator_agent import NavigatorAgent
from .extractor_agent import ExtractorAgent
from .critic_agent import CriticAgent
from .synthesizer_agent import SynthesizerAgent
from .fast_stream_extractor import FastStreamExtractor
from .dual_stream_synthesizer import DualStreamSynthesizer
from .data_structures import (
    PaperDocument,
    FieldType,
    ExtractionResult,
    FinalReport,
    PaperSection,
    SectionScope
)

logger = logging.getLogger(__name__)


class DeepPaperOrchestrator:
    """
    DeepPaper协调器
    管理Multi-Agent的迭代式论文解析流程
    """

    def __init__(
        self,
        llm_client,
        max_retries: int = 2,
        max_context_length: int = 3000
    ):
        """
        初始化协调器

        Args:
            llm_client: LLM客户端
            max_retries: 每个字段的最大重试次数
            max_context_length: LLM上下文最大长度
        """
        self.llm_client = llm_client
        self.max_retries = max_retries

        # 初始化所有Agent (包括Dual-Stream组件)
        logger.info("初始化DeepPaper Multi-Agent系统...")
        self.navigator = NavigatorAgent(llm_client)
        self.extractor = ExtractorAgent(llm_client, max_context_length)
        self.critic = CriticAgent(llm_client)
        self.synthesizer = SynthesizerAgent(llm_client)

        # 🆕 Dual-Stream组件
        self.fast_stream_extractor = FastStreamExtractor(llm_client)
        self.dual_stream_synthesizer = DualStreamSynthesizer(llm_client)

        logger.info("✅ Agent初始化完成 (含Dual-Stream):")
        logger.info("   - Navigator (导航员)")
        logger.info("   - Extractor (提取员)")
        logger.info("   - Critic (审查员)")
        logger.info("   - Synthesizer (总结员)")
        logger.info("   - FastStreamExtractor (快速流提取器) 🆕")
        logger.info("   - DualStreamSynthesizer (双流合成器) 🆕")

        # 要提取的字段
        self.fields_to_extract = [
            FieldType.PROBLEM,
            FieldType.METHOD,
            FieldType.LIMITATION,
            FieldType.FUTURE_WORK
        ]

    def analyze_paper(
        self,
        paper_document: PaperDocument,
        output_dir: Optional[str] = None
    ) -> FinalReport:
        """
        分析论文并返回深度解析报告

        Args:
            paper_document: 论文文档
            output_dir: 输出目录(可选)

        Returns:
            FinalReport: 最终报告
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 开始DeepPaper多Agent分析")
        logger.info(f"{'='*80}")
        logger.info(f"论文: {paper_document.title}")
        logger.info(f"章节数: {len(paper_document.sections)}")
        logger.info(f"{'='*80}\n")

        # 存储每个字段的提取结果
        extractions: Dict[FieldType, ExtractionResult] = {}
        iteration_counts: Dict[FieldType, int] = {}

        # 逐个字段处理
        for field in self.fields_to_extract:
            logger.info(f"\n{'─'*80}")
            logger.info(f"📋 处理字段: {field.value.upper()}")
            logger.info(f"{'─'*80}")

            extraction = self._extract_field_with_retry(
                paper_document,
                field
            )

            extractions[field] = extraction
            iteration_counts[field] = getattr(extraction, 'iterations', 1)

        # 使用Synthesizer生成最终报告
        logger.info(f"\n{'='*80}")
        logger.info(f"📝 生成最终报告")
        logger.info(f"{'='*80}\n")

        report = self.synthesizer.synthesize(
            paper=paper_document,
            extractions=extractions,
            iteration_counts=iteration_counts
        )

        # 保存报告
        if output_dir:
            self._save_report(report, output_dir)

        logger.info(f"\n{'='*80}")
        logger.info(f"✅ DeepPaper分析完成!")
        logger.info(f"{'='*80}\n")

        return report

    def _extract_field_with_retry(
        self,
        paper: PaperDocument,
        field: FieldType
    ) -> ExtractionResult:
        """
        带重试机制的字段提取

        🆕 Dual-Stream策略:
        - 对于PROBLEM和METHOD: 使用双流策略(Fast Stream + Slow Stream)
        - 对于LIMITATION和FUTURE_WORK: 使用传统流程(Navigator -> Extractor)

        实现Reflection Loop:
        Navigator → Extractor → Critic → (重试) → 最终结果

        Args:
            paper: 论文文档
            field: 字段类型

        Returns:
            ExtractionResult: 最终提取结果
        """
        # 🆕 判断是否使用Dual-Stream
        if field in [FieldType.PROBLEM, FieldType.METHOD]:
            logger.info(f"\n  🔀 使用Dual-Stream策略提取'{field.value}'")
            return self._extract_with_dual_stream(paper, field)
        else:
            logger.info(f"\n  📝 使用传统流程提取'{field.value}'")
            return self._extract_with_traditional_flow(paper, field)

    def _extract_with_dual_stream(
        self,
        paper: PaperDocument,
        field: FieldType
    ) -> ExtractionResult:
        """
        使用Dual-Stream策略提取字段

        流程:
        1. Fast Stream: 从Abstract/Introduction提取高层锚点
        2. Slow Stream: Navigator -> Extractor -> Critic (带重试)
           🆕 将Fast Stream的结果作为anchor guidance传递给Extractor
        3. Dual-Stream Synthesizer: 合并两个流的结果

        Args:
            paper: 论文文档
            field: 字段类型

        Returns:
            ExtractionResult: 合并后的最终结果
        """
        # 步骤1: Fast Stream - 提取锚点信息
        logger.info(f"\n  ⚡ [Fast Stream] 提取高层锚点...")
        fast_result = self.fast_stream_extractor.extract_anchor(paper, field)

        # 步骤2: Slow Stream - 传统流程(Navigator -> Extractor -> Critic)
        # 🆕 将Fast Stream的结果作为anchor guidance
        logger.info(f"\n  🐢 [Slow Stream] 提取详细证据...")
        logger.info(f"     → 使用Fast Stream锚点作为指导: {fast_result.content[:80]}...")

        slow_result = self._extract_with_traditional_flow(
            paper=paper,
            field=field,
            anchor_guidance=fast_result.content  # 🆕 传递锚点
        )

        # 步骤3: Dual-Stream Synthesizer - 合并结果
        logger.info(f"\n  🔀 [Dual-Stream Synthesizer] 合并两个流...")
        merged_result = self.dual_stream_synthesizer.merge(
            fast_result=fast_result,
            slow_result=slow_result,
            field=field
        )

        logger.info(f"  ✅ Dual-Stream提取完成")
        logger.info(f"     → 最终内容: {merged_result.content[:100]}...")
        logger.info(f"     → 证据数量: {len(merged_result.evidence)}")
        logger.info(f"     → 提取方法: {merged_result.extraction_method}")

        return merged_result

    def _extract_with_traditional_flow(
        self,
        paper: PaperDocument,
        field: FieldType,
        anchor_guidance: Optional[str] = None  # 🆕 接收锚点指导
    ) -> ExtractionResult:
        """
        使用传统流程提取字段 (原始逻辑)

        流程: Navigator -> Extractor -> Critic (带重试)

        Args:
            paper: 论文文档
            field: 字段类型
            anchor_guidance: 🆕 来自Fast Stream的锚点指导(可选)

        Returns:
            ExtractionResult: 提取结果
        """
        # 步骤1: Navigator定位章节
        scope = self.navigator.navigate(paper, field)

        # 初始化重试变量
        retry_count = 0
        retry_prompt = None
        current_extraction = None
        previous_content = None  # 🔧 记录上次提取结果,避免重复

        # Reflection Loop
        while retry_count <= self.max_retries:
            iteration_label = f"初次提取" if retry_count == 0 else f"第{retry_count}次重试"
            logger.info(f"\n  🔄 [{iteration_label}]")

            # 步骤2: Extractor提取内容
            # 🆕 传递anchor_guidance到Extractor
            current_extraction = self.extractor.extract(
                paper=paper,
                scope=scope,
                retry_prompt=retry_prompt,
                anchor_guidance=anchor_guidance  # 🆕 传递锚点指导
            )

            # 🔧 优化: 检测重复提取(内容完全相同)
            if previous_content and current_extraction.content == previous_content:
                logger.warning(f"  ⚠️ 检测到重复提取,重试无效,终止循环")
                current_extraction.iterations = retry_count + 1
                return current_extraction

            previous_content = current_extraction.content

            # 步骤3: Critic验证
            feedback = self.critic.critique(
                extraction=current_extraction,
                paper=paper,
                scope=scope
            )

            # 检查是否通过
            if feedback.approved:
                logger.info(f"  ✅ 字段'{field.value}'提取通过 (迭代{retry_count + 1}次)")
                current_extraction.iterations = retry_count + 1
                return current_extraction

            # 未通过,检查是否还能重试
            if retry_count >= self.max_retries:
                logger.warning(f"  ⚠️ 字段'{field.value}'达到最大重试次数,使用当前结果")
                current_extraction.iterations = retry_count + 1
                return current_extraction

            # 准备重试
            retry_count += 1
            logger.info(f"  ↻ Critic反馈: {feedback.feedback_message}")
            logger.info(f"  ↻ 准备重试...")

            # 根据Critic反馈调整
            retry_prompt = feedback.retry_prompt

            # 如果Critic建议了新的章节,更新scope
            if feedback.suggested_sections:
                # 🔧 优化: 避免重复的章节扩展
                new_sections = [s for s in feedback.suggested_sections if s not in scope.target_sections]
                if new_sections:
                    scope.target_sections = feedback.suggested_sections
                    scope.section_titles = [
                        paper.sections[i].title
                        for i in feedback.suggested_sections
                        if i < len(paper.sections)
                    ]
                    logger.info(f"     → 扩展搜索范围到: {scope.section_titles}")
                else:
                    logger.info(f"     → 搜索范围未变化,使用更强的retry_prompt")

        # 不应该走到这里,但作为保险
        logger.warning(f"  ⚠️ 字段'{field.value}'提取异常终止")
        current_extraction.iterations = retry_count + 1
        return current_extraction

    def _save_report(self, report: FinalReport, output_dir: str):
        """
        保存报告到文件

        生成两种格式:
        1. JSON格式(机器可读)
        2. Markdown格式(人类可读)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        safe_id = report.paper_id.replace('/', '_')
        json_file = output_path / f"deep_paper_{safe_id}.json"
        md_file = output_path / f"deep_paper_{safe_id}.md"

        # 保存JSON
        self.synthesizer.export_to_json(report, str(json_file))

        # 保存Markdown
        try:
            md_content = self.synthesizer.generate_human_readable_report(report)
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(md_content)
            logger.info(f"  ✅ Markdown报告已保存: {md_file}")
        except Exception as e:
            logger.warning(f"  ⚠️ Markdown保存失败: {e}")

    def batch_analyze_papers(
        self,
        papers: List[Dict],
        pdf_dir: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> List[FinalReport]:
        """
        批量分析论文

        Args:
            papers: 论文列表(OpenAlex格式)
            pdf_dir: PDF文件夹路径
            output_dir: 输出目录

        Returns:
            报告列表
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"📚 批量分析 {len(papers)} 篇论文")
        logger.info(f"{'='*80}\n")

        reports = []

        for i, paper in enumerate(papers):
            try:
                logger.info(f"\n{'▓'*80}")
                logger.info(f"进度: [{i+1}/{len(papers)}]")
                logger.info(f"{'▓'*80}")

                # 转换为PaperDocument
                paper_doc = self._convert_to_paper_document(paper, pdf_dir)

                # 分析
                report = self.analyze_paper(paper_doc, output_dir)
                reports.append(report)

                logger.info(f"✅ 论文 {i+1} 分析完成\n")

            except Exception as e:
                logger.error(f"❌ 论文 {i+1} 分析失败: {e}")
                continue

        logger.info(f"\n{'='*80}")
        logger.info(f"✅ 批量分析完成! 成功: {len(reports)}/{len(papers)}")
        logger.info(f"{'='*80}\n")

        return reports

    def _convert_to_paper_document(
        self,
        paper: Dict,
        pdf_dir: Optional[str] = None
    ) -> PaperDocument:
        """
        将OpenAlex格式的论文转换为PaperDocument

        尝试:
        1. 从PDF提取章节(如果pdf_dir提供且PDF存在)
        2. 降级到摘要
        """
        # 基础信息
        paper_id = paper.get('id', 'unknown')
        title = paper.get('title', 'Untitled')
        abstract = paper.get('abstract', '')
        authors = [author.get('author', {}).get('display_name', 'Unknown')
                   for author in paper.get('authorships', [])]
        year = paper.get('publication_year')

        # 提取章节
        sections = []

        # 尝试从PDF提取
        if pdf_dir:
            pdf_path = self._find_pdf(paper_id, pdf_dir)
            if pdf_path:
                sections = self._extract_sections_from_pdf(pdf_path)

        # 如果没有PDF或提取失败,使用摘要
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

    def _find_pdf(self, paper_id: str, pdf_dir: str) -> Optional[str]:
        """查找论文PDF"""
        pdf_dir_path = Path(pdf_dir)
        if not pdf_dir_path.exists():
            return None

        # 查找匹配的PDF
        for pdf_file in pdf_dir_path.glob(f"{paper_id}*.pdf"):
            return str(pdf_file)

        return None

    def _extract_sections_from_pdf(self, pdf_path: str) -> List[PaperSection]:
        """
        从PDF提取章节

        这里复用现有的grobid_parser或llm_rag_paper_analyzer中的方法
        """
        try:
            # 导入现有的解析器
            from grobid_parser import GrobidPDFParser

            # 尝试使用GROBID
            parser = GrobidPDFParser()
            sections = parser.extract_sections_from_pdf(pdf_path)

            if sections:
                logger.info(f"  ✅ 从PDF提取了 {len(sections)} 个章节")
                return sections

        except Exception as e:
            logger.warning(f"  ⚠️ PDF提取失败: {e}")

        return []