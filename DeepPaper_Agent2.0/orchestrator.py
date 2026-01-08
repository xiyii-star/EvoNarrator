"""
DeepPaper 2.0 Orchestrator (协调器)
整合所有组件，协调论文深度信息提取流程

工作流程:
1. Problem: 使用 LogicAnalystAgent 提取问题
2. Method: 使用 LogicAnalystAgent 提取方法
3. Limitation: 使用 LimitationExtractor (章节定位 + 引用分析)
4. Future Work: 使用 FutureWorkExtractor (章节定位)
5. 整合结果输出最终报告


"""

import json
import logging
from typing import Dict, Optional
from pathlib import Path
import sys

# 添加当前目录到sys.path以支持相对导入
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

# 导入数据结构
from data_structures import (
    PaperDocument,
    PaperSection,
    FinalReport,
    FieldType,
    ExtractionResult
)

# 导入各个Agent
from LogicAnalystAgent import LogicAnalystAgent
from LimitationExtractor import LimitationExtractor
from FutureWorkExtractor import FutureWorkExtractor

# 导入LLM配置
sys.path.append(str(Path(__file__).parent.parent))
from src.llm_config import LLMClient, LLMConfig

logger = logging.getLogger(__name__)


class DeepPaper2Orchestrator:
    """
    DeepPaper 2.0 协调器
    整合所有组件完成论文深度信息提取
    """

    def __init__(
        self,
        llm_client: LLMClient,
        use_citation_analysis: bool = False
    ):
        """
        初始化协调器

        Args:
            llm_client: LLM客户端
            use_citation_analysis: 是否对limitation使用引用分析
        """
        self.llm_client = llm_client
        self.use_citation_analysis = use_citation_analysis

        # 初始化所有Agent
        logger.info("初始化 DeepPaper 2.0 Multi-Agent 系统...")
        self.logic_analyst = LogicAnalystAgent(llm_client)
        self.limitation_extractor = LimitationExtractor(
            llm_client,
            use_citation_analysis=use_citation_analysis
        )
        self.future_work_extractor = FutureWorkExtractor(llm_client)

        logger.info("Agent初始化完成:")
        logger.info("   - LogicAnalystAgent (逻辑分析员)")
        logger.info("   - LimitationExtractor (局限性提取器)")
        logger.info("   - FutureWorkExtractor (未来工作提取器)")
        if use_citation_analysis:
            logger.info("   - CitationDetectiveAgent (引用侦探) [已启用]")

    def analyze_paper(
        self,
        paper_document: PaperDocument,
        paper_id: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> FinalReport:
        """
        分析论文并返回深度解析报告

        Args:
            paper_document: 论文文档
            paper_id: 论文ID（用于引用分析，可选）
            output_dir: 输出目录（可选）

        Returns:
            FinalReport: 最终报告
        """
        logger.info(f"\n{'=' * 80}")
        logger.info(f"🚀 开始 DeepPaper 2.0 分析")
        logger.info(f"{'=' * 80}")
        logger.info(f"论文: {paper_document.title}")
        logger.info(f"章节数: {len(paper_document.sections)}")
        logger.info(f"{'=' * 80}\n")

        # 构建用于LogicAnalystAgent的部分：标题、摘要、introduction
        paper_content = self._build_paper_content(paper_document)

        # 提取结果字典
        extractions: Dict[FieldType, ExtractionResult] = {}

        # 1. 提取 Problem 和 Method (使用 LogicAnalystAgent)
        logger.info(f"\n{'─' * 80}")
        logger.info(f"📋 [Step 1/2] 使用 LogicAnalystAgent 提取 Problem & Method")
        logger.info(f"{'─' * 80}")

        problem_result, method_result = self._extract_problem_and_method(
            paper_content,
            paper_document
        )

        extractions[FieldType.PROBLEM] = problem_result
        extractions[FieldType.METHOD] = method_result

        # 2. 提取 Limitation (使用 LimitationExtractor)
        logger.info(f"\n{'─' * 80}")
        logger.info(f"📋 [Step 2/2] 提取 Limitation & Future Work")
        logger.info(f"{'─' * 80}")

        limitation_result = self.limitation_extractor.extract(
            paper=paper_document,
            paper_id=paper_id
        )
        extractions[FieldType.LIMITATION] = limitation_result

        # 3. 提取 Future Work (使用 FutureWorkExtractor)
        future_work_result = self.future_work_extractor.extract(
            paper=paper_document
        )
        extractions[FieldType.FUTURE_WORK] = future_work_result

        # 4. 生成最终报告
        logger.info(f"\n{'=' * 80}")
        logger.info(f"📝 生成最终报告")
        logger.info(f"{'=' * 80}\n")

        report = self._build_final_report(paper_document, extractions)

        # 保存报告
        if output_dir:
            self._save_report(report, output_dir)

        logger.info(f"\n{'=' * 80}")
        logger.info(f"✅ DeepPaper 2.0 分析完成!")
        logger.info(f"{'=' * 80}\n")

        return report

    def _build_paper_content(self, paper: PaperDocument) -> str:
        """
        构建论文内容（用于LogicAnalystAgent）
        只包含标题、摘要和Introduction部分

        Args:
            paper: 论文文档

        Returns:
            论文内容字符串（标题 + 摘要 + Introduction）
        """
        content_parts = []

        # 添加标题
        if paper.title:
            content_parts.append(f"Title: {paper.title}\n")

        # 添加摘要
        if paper.abstract:
            content_parts.append(f"Abstract:\n{paper.abstract}\n")

        # 只添加Introduction章节
        for section in paper.sections:
            # 匹配Introduction章节（不区分大小写）
            if section.title.lower().strip() in ['introduction', '1. introduction', '1 introduction']:
                content_parts.append(f"\n{section.title}\n")
                content_parts.append(section.content)
                break  # 只取第一个匹配的Introduction

        return "\n".join(content_parts)

    def _extract_problem_and_method(
        self,
        paper_content: str,
        paper_document: PaperDocument
    ) -> tuple:
        """
        使用LogicAnalystAgent提取Problem和Method

        Args:
            paper_content: 论文全文
            paper_document: 论文文档对象

        Returns:
            (problem_result, method_result)
        """
        # 构建元数据
        metadata = {
            "title": paper_document.title,
            "authors": paper_document.authors,
            "year": paper_document.year
        }

        # 调用LogicAnalystAgent
        pairs = self.logic_analyst.analyze(
            paper_content=paper_content,
            paper_metadata=metadata
        )

        # 解析结果
        if pairs:
            # 取第一个最核心的Problem-Solution Pair
            main_pair = pairs[0]

            # 构建Problem结果
            problem_result = ExtractionResult(
                field=FieldType.PROBLEM,
                content=main_pair.problem,
                evidence=[{"text": main_pair.evidence}] if main_pair.evidence else [],
                extraction_method="logic_analyst",
                confidence=main_pair.confidence,
                iterations=1
            )

            # 构建Method结果
            method_content = f"{main_pair.solution}\n\n**Explanation:** {main_pair.explanation}"
            method_result = ExtractionResult(
                field=FieldType.METHOD,
                content=method_content,
                evidence=[{"text": main_pair.evidence}] if main_pair.evidence else [],
                extraction_method="logic_analyst",
                confidence=main_pair.confidence,
                iterations=1
            )

            logger.info(f"  ✅ Problem: {problem_result.content[:100]}...")
            logger.info(f"  ✅ Method: {method_result.content[:100]}...")

        else:
            # 降级：未找到
            logger.warning("  ⚠️ LogicAnalystAgent未找到Problem-Solution Pairs")
            problem_result = ExtractionResult(
                field=FieldType.PROBLEM,
                content="未找到明确的研究问题描述",
                evidence=[],
                extraction_method="logic_analyst",
                confidence=0.0,
                iterations=1
            )
            method_result = ExtractionResult(
                field=FieldType.METHOD,
                content="未找到明确的方法描述",
                evidence=[],
                extraction_method="logic_analyst",
                confidence=0.0,
                iterations=1
            )

        return problem_result, method_result

    def _build_final_report(
        self,
        paper: PaperDocument,
        extractions: Dict[FieldType, ExtractionResult]
    ) -> FinalReport:
        """
        构建最终报告

        Args:
            paper: 论文文档
            extractions: 提取结果字典

        Returns:
            FinalReport
        """
        # 提取各字段内容
        problem_ext = extractions.get(FieldType.PROBLEM)
        method_ext = extractions.get(FieldType.METHOD)
        limitation_ext = extractions.get(FieldType.LIMITATION)
        future_work_ext = extractions.get(FieldType.FUTURE_WORK)

        report = FinalReport(
            paper_id=paper.paper_id,
            title=paper.title,
            problem=problem_ext.content if problem_ext else "未提取",
            method=method_ext.content if method_ext else "未提取",
            limitation=limitation_ext.content if limitation_ext else "未提取",
            future_work=future_work_ext.content if future_work_ext else "未提取",
            problem_evidence=problem_ext.evidence if problem_ext else [],
            method_evidence=method_ext.evidence if method_ext else [],
            limitation_evidence=limitation_ext.evidence if limitation_ext else [],
            future_work_evidence=future_work_ext.evidence if future_work_ext else [],
            metadata={
                "authors": paper.authors,
                "year": paper.year,
                "extraction_methods": {
                    "problem": problem_ext.extraction_method if problem_ext else "unknown",
                    "method": method_ext.extraction_method if method_ext else "unknown",
                    "limitation": limitation_ext.extraction_method if limitation_ext else "unknown",
                    "future_work": future_work_ext.extraction_method if future_work_ext else "unknown"
                },
                "confidences": {
                    "problem": problem_ext.confidence if problem_ext else 0.0,
                    "method": method_ext.confidence if method_ext else 0.0,
                    "limitation": limitation_ext.confidence if limitation_ext else 0.0,
                    "future_work": future_work_ext.confidence if future_work_ext else 0.0
                }
            }
        )

        return report

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
        safe_id = report.paper_id.replace('/', '_').replace(':', '_')
        json_file = output_path / f"deeppaper2_{safe_id}.json"
        md_file = output_path / f"deeppaper2_{safe_id}.md"

        # 保存JSON
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"  ✅ JSON报告已保存: {json_file}")
        except Exception as e:
            logger.error(f"  ❌ JSON保存失败: {e}")

        # 保存Markdown
        try:
            md_content = self._generate_markdown_report(report)
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(md_content)
            logger.info(f"  ✅ Markdown报告已保存: {md_file}")
        except Exception as e:
            logger.error(f"  ❌ Markdown保存失败: {e}")

    def _generate_markdown_report(self, report: FinalReport) -> str:
        """生成Markdown格式的报告"""
        lines = [
            f"# {report.title}",
            "",
            "## Paper Information",
            f"- **Paper ID**: {report.paper_id}",
            f"- **Authors**: {', '.join(report.metadata.get('authors', []))}",
            f"- **Year**: {report.metadata.get('year', 'N/A')}",
            "",
            "---",
            "",
            "## Problem",
            "",
            report.problem,
            "",
            "---",
            "",
            "## Method",
            "",
            report.method,
            "",
            "---",
            "",
            "## Limitation",
            "",
            report.limitation,
            "",
            "---",
            "",
            "## Future Work",
            "",
            report.future_work,
            "",
            "---",
            "",
            "## Metadata",
            "",
            "### Extraction Methods",
            ""
        ]

        # 添加提取方法信息
        methods = report.metadata.get('extraction_methods', {})
        for field, method in methods.items():
            lines.append(f"- **{field}**: {method}")

        lines.append("")
        lines.append("### Confidences")
        lines.append("")

        # 添加置信度信息
        confidences = report.metadata.get('confidences', {})
        for field, confidence in confidences.items():
            lines.append(f"- **{field}**: {confidence:.2f}")

        return "\n".join(lines)


def main():
    """测试代码"""
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="DeepPaper 2.0 Orchestrator - 论文深度信息提取")
    parser.add_argument("--config", required=True, help="LLM配置文件路径")
    parser.add_argument("--paper", required=True, help="论文文本文件路径（JSON格式）")
    parser.add_argument("--paper-id", help="论文ID（用于引用分析，可选）")
    parser.add_argument("--use-citation", action="store_true", help="是否对limitation使用引用分析")
    parser.add_argument("--output", default="./output", help="输出目录")

    args = parser.parse_args()

    try:
        # 初始化LLM客户端
        config = LLMConfig.from_file(args.config)
        llm_client = LLMClient(config)

        # 读取论文文档
        with open(args.paper, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)

        # 转换为PaperDocument对象
        sections = [
            PaperSection(**section) for section in paper_data.get('sections', [])
        ]
        paper = PaperDocument(
            paper_id=paper_data.get('paper_id', 'unknown'),
            title=paper_data.get('title', ''),
            abstract=paper_data.get('abstract', ''),
            authors=paper_data.get('authors', []),
            year=paper_data.get('year'),
            sections=sections,
            metadata=paper_data.get('metadata')
        )

        # 创建协调器
        orchestrator = DeepPaper2Orchestrator(
            llm_client=llm_client,
            use_citation_analysis=args.use_citation
        )

        # 执行分析
        report = orchestrator.analyze_paper(
            paper_document=paper,
            paper_id=args.paper_id,
            output_dir=args.output
        )

        # 打印摘要
        print("\n" + "=" * 80)
        print("DeepPaper 2.0 Analysis Complete")
        print("=" * 80)
        print(f"\nPaper: {report.title}")
        print(f"\nProblem: {report.problem[:150]}...")
        print(f"\nMethod: {report.method[:150]}...")
        print(f"\nLimitation: {report.limitation[:150]}...")
        print(f"\nFuture Work: {report.future_work[:150]}...")
        print(f"\n✅ 完整报告已保存到: {args.output}")

    except Exception as e:
        logger.error(f"执行失败: {e}")
        raise


if __name__ == "__main__":
    main()
