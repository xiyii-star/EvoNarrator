"""
Section Locator Agent (章节定位员)
负责定位 limitation 和 future work 章节

参考 DeepPaper_Agent 的 NavigatorAgent 实现
使用 LLM 或规则定位目标章节
"""

import json
import logging
import re
from typing import List, Dict, Optional, Any
from pathlib import Path

# 导入数据结构
from data_structures import PaperDocument, PaperSection, SectionScope, FieldType

# 导入LLM配置
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.llm_config import LLMClient, LLMConfig

logger = logging.getLogger(__name__)


class SectionLocatorAgent:
    """
    章节定位 Agent
    用于定位 limitation 和 future work 所在的章节
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        初始化章节定位员

        Args:
            llm_client: LLM客户端（可选，用于智能定位）
        """
        self.llm_client = llm_client
        self.system_prompt = self._build_system_prompt() if llm_client else None

        # 预定义的章节映射（作为fallback）
        self.section_mapping = {
            FieldType.LIMITATION: ['limitation', 'discussion', 'conclusion'],
            FieldType.FUTURE_WORK: ['future_work', 'future work', 'conclusion', 'discussion']
        }

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """You are an expert in analyzing the structure of academic papers.

Your Task: Accurately locate specific information within the paper.

Input Description:

The paper consists of multiple sections.
Each section has a title and a section_type.
You need to determine the location of the target information based on the section title, type, and content preview.
Key Distinctions:

Limitation: The authors admit the shortcomings of their own method (not criticizing prior work).
Future Work: Future research directions proposed by the authors.
Notes:

Limitations and Future Work often appear at the end of the Discussion or Conclusion sections.
Look for transition words: However, Unfortunately, Future work, remains to be.
The output must be in valid JSON format.
Output Format:
{
    "target_sections": [list of section indices],
    "reasoning": "reasoning process",
    "confidence": 0.0-1.0
}
"""

    def locate(
        self,
        paper: PaperDocument,
        field: FieldType
    ) -> SectionScope:
        """
        定位指定字段所在的章节

        Args:
            paper: 论文文档
            field: 字段类型 (LIMITATION 或 FUTURE_WORK)

        Returns:
            SectionScope: 章节范围
        """
        logger.info(f"🔍 定位字段 '{field.value}' 的章节...")

        # 构建章节概览
        structure_overview = self._build_structure_overview(paper)

        # 使用LLM或规则定位
        if self.llm_client:
            scope = self._locate_with_llm(paper, field, structure_overview)
        else:
            scope = self._locate_with_rules(paper, field)

        logger.info(f"   ✅ 定位到 {len(scope.target_sections)} 个章节: {scope.section_titles}")
        return scope

    def _build_structure_overview(self, paper: PaperDocument) -> str:
        """
        构建论文结构概览

        包含:
        - 标题和摘要
        - 章节列表（标题、类型、首尾段落预览）
        """
        overview_parts = [
            f"Paper Title: {paper.title}",
            f"Total Sections: {len(paper.sections)}",
        ]

        # 添加摘要预览
        if paper.abstract:
            abstract_preview = paper.abstract[:300] if len(paper.abstract) > 300 else paper.abstract
            overview_parts.append(f"\nAbstract Preview:\n{abstract_preview}...")

        overview_parts.append("\n" + "=" * 60)
        overview_parts.append("Section Structure:")
        overview_parts.append("=" * 60)

        for i, section in enumerate(paper.sections):
            section_type = section.section_type if section.section_type != 'other' else 'unknown'
            content_length = len(section.content)

            # 提取首段和尾段
            paragraphs = [p.strip() for p in section.content.split('\n\n') if p.strip()]
            first_para = paragraphs[0][:150] if paragraphs else ""
            last_para = paragraphs[-1][:150] if len(paragraphs) > 1 else ""

            section_info = [
                f"\n[{i}] {section.title}",
                f"    Type: {section_type} | Length: {content_length} chars",
            ]

            if first_para:
                section_info.append(f"    First: {first_para}...")

            # 对于可能包含limitation/future work的章节，显示尾段
            if section_type in ['discussion', 'conclusion', 'limitation', 'future_work', 'unknown']:
                if last_para and last_para != first_para:
                    section_info.append(f"    Last: {last_para}...")

            overview_parts.append('\n'.join(section_info))

        return "\n".join(overview_parts)

    def _locate_with_llm(
        self,
        paper: PaperDocument,
        field: FieldType,
        structure_overview: str
    ) -> SectionScope:
        """
        使用LLM进行智能定位
        """
        prompt = self._build_locator_prompt(field, structure_overview)

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=0.2,
                max_tokens=1000
            )

            # 解析响应
            scope = self._parse_llm_response(response, paper, field)
            return scope

        except Exception as e:
            logger.warning(f"   ⚠️ LLM定位失败: {e}, 降级到规则匹配")
            return self._locate_with_rules(paper, field)

    def _build_locator_prompt(self, field: FieldType, structure_overview: str) -> str:
        """构建定位提示词"""
        field_descriptions = {
            FieldType.LIMITATION: "本文方法的局限性（不是前人工作的缺点）",
            FieldType.FUTURE_WORK: "未来工作方向/待改进的点"
        }

        field_hints = {
            FieldType.LIMITATION: """⚠️ 重要! 可能隐藏在Discussion/Conclusion末尾的转折词后。
            - 检查type为'discussion'/'conclusion'的章节的Last段落
            - 寻找: However, Unfortunately, Limitation, remains to be
            - 注意: 只要本文方法的局限性,不要baseline的缺点""",

            FieldType.FUTURE_WORK: """可能在独立的Future Work章节,或Conclusion/Discussion的结尾。
            - 检查type为'future_work'的章节(如果有)
            - 否则检查'conclusion'/'discussion'章节的Last段落
            - 寻找: future, next, further, explore, plan"""
        }

        prompt = f"""你是一个资深研究员,需要分析论文结构并定位关键信息。

目标字段: {field.value}
字段定义: {field_descriptions[field]}
搜索提示: {field_hints[field]}

论文结构:
{structure_overview}

任务:
1. 分析每个章节的Title, Type, 和内容预览
2. 找出最可能包含'{field.value}'的章节(返回章节索引)
3. 特别关注Discussion/Conclusion章节的尾段
4. 给出你的推理过程

输出格式(JSON):
{{
    "target_sections": [章节索引列表],
    "reasoning": "你的推理过程",
    "confidence": 0.0-1.0
}}

请输出:"""

        return prompt

    def _parse_llm_response(
        self,
        response: str,
        paper: PaperDocument,
        field: FieldType
    ) -> SectionScope:
        """解析LLM响应"""
        try:
            # 提取JSON
            json_str = self._extract_json(response)
            result = json.loads(json_str)

            target_sections = result.get('target_sections', [])
            reasoning = result.get('reasoning', 'No reasoning provided')
            confidence = result.get('confidence', 0.5)

            # 验证章节索引
            target_sections = [
                idx for idx in target_sections
                if 0 <= idx < len(paper.sections)
            ]

            if not target_sections:
                # 降级到规则
                return self._locate_with_rules(paper, field)

            section_titles = [paper.sections[idx].title for idx in target_sections]

            return SectionScope(
                field=field,
                target_sections=target_sections,
                section_titles=section_titles,
                reasoning=reasoning,
                confidence=confidence
            )

        except Exception as e:
            logger.warning(f"   ⚠️ 解析LLM响应失败: {e}")
            return self._locate_with_rules(paper, field)

    def _extract_json(self, text: str) -> str:
        """从文本中提取JSON内容"""
        # 尝试找到JSON代码块
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()

        # 尝试找到花括号包围的内容
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return text[start:end]

        return text.strip()

    def _locate_with_rules(self, paper: PaperDocument, field: FieldType) -> SectionScope:
        """
        使用规则匹配进行定位（fallback）

        策略:
        1. 首先按章节类型匹配
        2. 如果失败，按标题关键词匹配
        3. 如果还是失败，返回最后几个章节
        """
        target_types = self.section_mapping.get(field, [])

        target_sections = []
        section_titles = []

        # 步骤1: 按章节类型匹配
        for i, section in enumerate(paper.sections):
            if section.section_type in target_types:
                target_sections.append(i)
                section_titles.append(section.title)

        # 步骤2: 如果失败，使用标题匹配
        if not target_sections:
            logger.info(f"   → 类型匹配失败,尝试标题匹配...")
            target_sections, section_titles = self._match_by_title(paper, field)

        # 步骤3: 如果还是失败，返回最后几个章节（limitation/future work通常在结尾）
        if not target_sections:
            logger.warning(f"   → 标题匹配也失败,使用最后3个章节作为默认范围")
            target_sections = list(range(max(0, len(paper.sections) - 3), len(paper.sections)))
            section_titles = [paper.sections[i].title for i in target_sections]

        return SectionScope(
            field=field,
            target_sections=target_sections,
            section_titles=section_titles,
            reasoning=f"Rule-based matching for {field.value} using section types: {target_types}",
            confidence=0.6
        )

    def _match_by_title(self, paper: PaperDocument, field: FieldType) -> tuple:
        """根据章节标题匹配"""
        title_keywords_map = {
            FieldType.LIMITATION: [
                'limitation', 'discussion', 'conclusion',
                'summary', 'result', 'drawback'
            ],
            FieldType.FUTURE_WORK: [
                'future', 'conclusion', 'discussion',
                'outlook', 'direction', 'next step', 'ongoing'
            ]
        }

        keywords = title_keywords_map.get(field, [])

        target_sections = []
        section_titles = []

        for i, section in enumerate(paper.sections):
            title_lower = section.title.lower()
            # 检查标题是否包含任何关键词
            if any(kw in title_lower for kw in keywords):
                target_sections.append(i)
                section_titles.append(section.title)

        logger.info(f"   → 标题匹配找到 {len(target_sections)} 个章节")

        return target_sections, section_titles


def main():
    """测试代码"""
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="Section Locator Agent - 章节定位")
    parser.add_argument("--config", help="LLM配置文件路径（可选）")
    parser.add_argument("--paper", required=True, help="论文文本文件路径（JSON格式）")
    parser.add_argument("--field", required=True, choices=["limitation", "future_work"],
                        help="要定位的字段类型")

    args = parser.parse_args()

    try:
        # 初始化LLM客户端（如果提供配置）
        llm_client = None
        if args.config:
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

        # 创建定位agent
        field_type = FieldType.LIMITATION if args.field == "limitation" else FieldType.FUTURE_WORK
        agent = SectionLocatorAgent(llm_client)

        # 执行定位
        scope = agent.locate(paper, field_type)

        # 打印结果
        print("\n" + "=" * 80)
        print(f"Section Locator Results for '{field_type.value}'")
        print("=" * 80)
        print(f"\nTarget Sections: {scope.target_sections}")
        print(f"Section Titles: {scope.section_titles}")
        print(f"Confidence: {scope.confidence:.2f}")
        print(f"\nReasoning:\n{scope.reasoning}")

    except Exception as e:
        logger.error(f"执行失败: {e}")
        raise


if __name__ == "__main__":
    main()
