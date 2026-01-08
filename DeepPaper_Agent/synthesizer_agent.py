"""
Synthesizer Agent (总结员)
负责整合所有Agent的结果,输出最终的结构化JSON

任务:
1. 收集所有经过Critic验证的提取结果
2. 整理Evidence(原文引用)
3. 生成结构化的JSON报告
4. 添加质量评分和元数据

核心价值:
- 提供可验证的输出(带Evidence)
- 生成规范的JSON格式
- 方便下游任务使用
"""

import logging
import json
from datetime import datetime
from typing import List, Dict
from .data_structures import (
    ExtractionResult,
    FinalReport,
    PaperDocument,
    FieldType
)

logger = logging.getLogger(__name__)


class SynthesizerAgent:
    """
    总结员Agent
    整合信息并生成最终报告
    """

    def __init__(self, llm_client=None):
        """
        初始化Synthesizer Agent

        Args:
            llm_client: LLM客户端(可选,用于生成摘要)
        """
        self.llm_client = llm_client

    def synthesize(
        self,
        paper: PaperDocument,
        extractions: Dict[FieldType, ExtractionResult],
        iteration_counts: Dict[FieldType, int]
    ) -> FinalReport:
        """
        合成最终报告

        Args:
            paper: 论文文档
            extractions: 提取结果字典 {FieldType: ExtractionResult}
            iteration_counts: 迭代次数统计

        Returns:
            FinalReport: 最终报告
        """
        logger.info(f"  📝 Synthesizer: 生成最终报告...")

        # 提取各字段内容
        problem_result = extractions.get(FieldType.PROBLEM)
        method_result = extractions.get(FieldType.METHOD)
        limitation_result = extractions.get(FieldType.LIMITATION)
        future_work_result = extractions.get(FieldType.FUTURE_WORK)

        # 构建报告
        report = FinalReport(
            paper_id=paper.paper_id,
            title=paper.title,
            problem=problem_result.content if problem_result else "未提取",
            problem_evidence=self._format_evidence(problem_result),
            method=method_result.content if method_result else "未提取",
            method_evidence=self._format_evidence(method_result),
            limitation=limitation_result.content if limitation_result else "未提取",
            limitation_evidence=self._format_evidence(limitation_result),
            future_work=future_work_result.content if future_work_result else "未提取",
            future_work_evidence=self._format_evidence(future_work_result),
            extraction_quality=self._compute_quality_scores(extractions),
            iteration_count=self._format_iteration_counts(iteration_counts),
            timestamp=datetime.now().isoformat()
        )

        logger.info(f"     ✅ 报告生成完成")
        self._log_report_summary(report)

        return report

    def _format_evidence(self, extraction: ExtractionResult = None) -> List[Dict]:
        """
        格式化Evidence列表

        Args:
            extraction: 提取结果

        Returns:
            格式化的Evidence列表
        """
        if not extraction or not extraction.evidence:
            return []

        # 限制每个字段最多保留5条evidence(避免过长)
        max_evidence = 5
        evidence_list = []

        for i, ev in enumerate(extraction.evidence[:max_evidence]):
            evidence_list.append({
                'id': i + 1,
                'section': ev.get('section', 'Unknown'),
                'text': ev.get('text', '')[:500],  # 限制长度
                'page': ev.get('page', 0)
            })

        return evidence_list

    def _compute_quality_scores(
        self,
        extractions: Dict[FieldType, ExtractionResult]
    ) -> Dict[str, float]:
        """
        计算各字段的质量评分

        评分维度:
        - 是否成功提取(非空)
        - Confidence分数
        - Evidence数量

        Returns:
            {field_name: score} (0-1)
        """
        scores = {}

        for field_type, extraction in extractions.items():
            if not extraction:
                scores[field_type.value] = 0.0
                continue

            # 基础分数:是否非空
            base_score = 0.0
            if extraction.content and extraction.content not in ["未找到相关信息", "未提取", "提取失败"]:
                base_score = 0.5

            # Confidence分数
            confidence_score = extraction.confidence * 0.3

            # Evidence分数
            evidence_score = min(len(extraction.evidence) / 3.0, 1.0) * 0.2

            # 总分
            total_score = base_score + confidence_score + evidence_score
            scores[field_type.value] = round(total_score, 2)

        return scores

    def _format_iteration_counts(
        self,
        iteration_counts: Dict[FieldType, int]
    ) -> Dict[str, int]:
        """格式化迭代次数"""
        return {
            field.value: count
            for field, count in iteration_counts.items()
        }

    def _log_report_summary(self, report: FinalReport):
        """输出报告摘要到日志"""
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 最终报告摘要:")
        logger.info(f"{'='*60}")
        logger.info(f"  论文ID: {report.paper_id}")
        logger.info(f"  标题: {report.title}")
        logger.info(f"\n  字段提取质量:")

        for field, score in report.extraction_quality.items():
            status = "✅" if score >= 0.6 else "⚠️" if score >= 0.3 else "❌"
            iterations = report.iteration_count.get(field, 0)
            logger.info(f"    {status} {field}: {score:.2f} (迭代{iterations}次)")

        logger.info(f"\n  内容预览:")
        logger.info(f"    Problem: {report.problem[:80]}...")
        logger.info(f"    Method: {report.method[:80]}...")
        logger.info(f"    Limitation: {report.limitation[:80]}...")
        logger.info(f"    Future Work: {report.future_work[:80]}...")
        logger.info(f"{'='*60}\n")

    def export_to_json(self, report: FinalReport, output_path: str) -> None:
        """
        导出报告为JSON文件

        Args:
            report: 最终报告
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

            logger.info(f"  ✅ 报告已导出: {output_path}")

        except Exception as e:
            logger.error(f"  ❌ 导出失败: {e}")

    def generate_human_readable_report(self, report: FinalReport) -> str:
        """
        生成人类可读的报告(Markdown格式)

        Args:
            report: 最终报告

        Returns:
            Markdown格式的报告文本
        """
        md_parts = [
            f"# 论文深度分析报告\n",
            f"**论文ID**: {report.paper_id}\n",
            f"**标题**: {report.title}\n",
            f"**生成时间**: {report.timestamp}\n",
            f"\n---\n",
            f"\n## 🎯 研究问题 (Problem)\n",
            f"{report.problem}\n",
            self._format_evidence_markdown("Problem", report.problem_evidence),
            f"\n## 💡 核心方法 (Method)\n",
            f"{report.method}\n",
            self._format_evidence_markdown("Method", report.method_evidence),
            f"\n## ⚠️ 局限性 (Limitation)\n",
            f"{report.limitation}\n",
            self._format_evidence_markdown("Limitation", report.limitation_evidence),
            f"\n## 🔮 未来工作 (Future Work)\n",
            f"{report.future_work}\n",
            self._format_evidence_markdown("Future Work", report.future_work_evidence),
            f"\n---\n",
            f"\n## 📊 提取质量评估\n",
        ]

        # 质量评分表格
        md_parts.append("\n| 字段 | 质量评分 | 迭代次数 |\n")
        md_parts.append("|------|---------|----------|\n")

        for field, score in report.extraction_quality.items():
            iterations = report.iteration_count.get(field, 0)
            md_parts.append(f"| {field} | {score:.2f} | {iterations} |\n")

        return "".join(md_parts)

    def _format_evidence_markdown(self, field_name: str, evidence_list: List[Dict]) -> str:
        """格式化Evidence为Markdown"""
        if not evidence_list:
            return "\n*无原文引用*\n"

        md_parts = [f"\n**原文引用 ({len(evidence_list)}条)**:\n"]

        for ev in evidence_list:
            section = ev.get('section', 'Unknown')
            text = ev.get('text', '')
            page = ev.get('page', 0)

            md_parts.append(f"\n> **[{section}]** (Page {page})  \n")
            md_parts.append(f"> {text}\n")

        return "".join(md_parts)
