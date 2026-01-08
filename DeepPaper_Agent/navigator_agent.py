"""
Navigator Agent (导航员)
负责分析论文结构,为每个提取字段确定搜索范围

任务:
1. 只读论文目录(章节标题)和首尾段
2. 为Problem/Method/Limitation/Future Work划定搜索范围
3. 不提取具体内容,只输出章节索引和推理过程

核心价值:
- 解决"找不到位置"的问题
- 比向量检索更智能:能推测隐含位置
- 例如:Limitation可能隐藏在Discussion或Conclusion结尾
"""

import logging
from typing import List, Dict
from .data_structures import PaperDocument, SectionScope, FieldType

logger = logging.getLogger(__name__)


class NavigatorAgent:
    """
    导航员Agent
    分析论文结构并定位关键信息位置
    """

    def __init__(self, llm_client):
        """
        初始化Navigator Agent

        Args:
            llm_client: LLM客户端(用于推理)
        """
        self.llm_client = llm_client

        # 预定义的章节映射(作为fallback)
        self.section_mapping = {
            FieldType.PROBLEM: ['abstract', 'introduction', 'conclusion'],
            FieldType.METHOD: ['abstract', 'introduction', 'method', 'conclusion'],
            FieldType.LIMITATION: ['limitation', 'discussion', 'conclusion'],
            FieldType.FUTURE_WORK: ['future_work', 'conclusion', 'discussion']
        }

    def navigate(self, paper: PaperDocument, field: FieldType) -> SectionScope:
        """
        为指定字段导航到相关章节

        Args:
            paper: 论文文档
            field: 要提取的字段类型

        Returns:
            SectionScope: 章节范围和推理
        """
        logger.info(f"  🧭 Navigator: 为字段'{field.value}'定位章节...")

        # 构建结构化的章节概览
        structure_overview = self._build_structure_overview(paper)

        # 使用LLM推理最佳章节
        if self.llm_client:
            scope = self._navigate_with_llm(paper, field, structure_overview)
        else:
            # 降级:使用规则匹配
            scope = self._navigate_with_rules(paper, field)

        logger.info(f"     → 定位到 {len(scope.target_sections)} 个章节: {scope.section_titles}")
        logger.info(f"     → 推理: {scope.reasoning[:100]}...")

        return scope

    def _build_structure_overview(self, paper: PaperDocument) -> str:
        """
        构建论文结构概览(用于LLM输入)

        利用GROBID已解析的结构化信息:
        - section_type: GROBID推断的章节类型
        - 章节标题和内容已被GROBID精确提取

        包含:
        - 标题和摘要
        - 章节列表(标题 + 类型 + 首段 + 尾段)
        - 内容长度信息(帮助LLM判断章节重要性)
        """
        overview_parts = [
            f"Paper Title: {paper.title}",
            f"Total Sections: {len(paper.sections)}",
        ]

        # 添加摘要(如果有)
        if paper.abstract:
            abstract_preview = paper.abstract[:400] if len(paper.abstract) > 400 else paper.abstract
            overview_parts.append(f"\nAbstract Preview:\n{abstract_preview}...")

        overview_parts.append("\n" + "="*60)
        overview_parts.append("Section Structure (parsed by GROBID):")
        overview_parts.append("="*60)

        for i, section in enumerate(paper.sections):
            # 获取GROBID解析的章节类型
            section_type = section.section_type if section.section_type != 'other' else 'unknown'

            # 计算内容统计
            content_length = len(section.content)
            paragraph_count = len([p for p in section.content.split('\n\n') if p.strip()])

            # 提取首段和尾段(GROBID已分段好的内容)
            paragraphs = [p.strip() for p in section.content.split('\n\n') if p.strip()]
            first_para = paragraphs[0][:200] if paragraphs else ""
            last_para = paragraphs[-1][:200] if len(paragraphs) > 1 else ""

            # 构建章节信息
            section_info = [
                f"\n[{i}] {section.title}",
                f"    Type: {section_type} | Length: {content_length} chars | Paragraphs: {paragraph_count}",
            ]

            if first_para:
                section_info.append(f"    First: {first_para}...")

            # 对于可能包含limitation/future work的章节,额外显示尾段
            if section_type in ['discussion', 'conclusion', 'limitation', 'future_work', 'unknown']:
                if last_para and last_para != first_para:
                    section_info.append(f"    Last: {last_para}...")

            overview_parts.append('\n'.join(section_info))

        return "\n".join(overview_parts)

    def _navigate_with_llm(
        self,
        paper: PaperDocument,
        field: FieldType,
        structure_overview: str
    ) -> SectionScope:
        """
        使用LLM进行智能导航

        LLM的优势:
        - 理解隐含位置(如Limitation隐藏在Conclusion的"However"后)
        - 推理章节相关性
        - 处理非标准章节名称
        """
        prompt = self._build_navigator_prompt(field, structure_overview)

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self._get_navigator_system_prompt()
            )

            # 解析LLM响应
            scope = self._parse_llm_response(response, paper, field)
            return scope

        except Exception as e:
            logger.warning(f"     ⚠️ LLM导航失败: {e}, 降级到规则匹配")
            return self._navigate_with_rules(paper, field)

    def _build_navigator_prompt(self, field: FieldType, structure_overview: str) -> str:
        """构建Navigator的Prompt"""

        field_descriptions = {
            FieldType.PROBLEM: "研究问题/挑战/要解决的核心问题",
            FieldType.METHOD: "本文提出的方法/模型/技术方案",
            FieldType.LIMITATION: "本文方法的局限性(不是前人工作的缺点!)",
            FieldType.FUTURE_WORK: "未来工作方向/待改进的点"
        }

        field_hints = {
            FieldType.PROBLEM: """通常在Abstract或Introduction中明确说明。
            - 标准论文: 寻找 "problem", "challenge", "gap" 等词
            - Review/工具类论文: 寻找 "lack", "incompatible", "need for" 等词""",

            FieldType.METHOD: """可能在Abstract, Introduction, Method或Conclusion中。
            - 关键词: "propose", "method", "approach", "model", "algorithm"
            - 工具类论文: "provide", "enable", "API", "framework", "system\"""",

            FieldType.LIMITATION: """⚠️ 重要! 可能隐藏在Discussion/Conclusion末尾的转折词后。
            - 专门检查type为'discussion'/'conclusion'的章节的Last段落
            - 寻找: However, Unfortunately, Future work, remains to be
            - 注意: 只要本文方法的局限性,不要baseline的缺点""",

            FieldType.FUTURE_WORK: """可能在独立的Future Work章节,或Conclusion/Discussion的结尾。
            - 检查type为'future_work'的章节(如果有)
            - 否则检查'conclusion'/'discussion'章节的Last段落"""
        }

        prompt = f"""你是一个资深研究员,需要分析论文结构并定位关键信息。

目标字段: {field.value}
字段定义: {field_descriptions[field]}
搜索提示: {field_hints[field]}

论文结构 (由GROBID解析):
{structure_overview}

任务:
1. 分析每个章节的Title, Type, 和内容预览
2. 利用GROBID提供的section_type信息(introduction, discussion, conclusion等)
3. 找出最可能包含'{field.value}'的章节(返回章节索引)
4. 不要只看Type! 要根据内容预览推测隐含位置
5. 给出你的推理过程

输出格式(JSON):
{{
    "target_sections": [章节索引列表],
    "reasoning": "你的推理过程",
    "confidence": 0.0-1.0
}}

示例(Limitation):
如果章节Type是"conclusion",但Last段落有"However, our method still faces...",
那么应该包含该章节。

请输出:"""

        return prompt

    def _get_navigator_system_prompt(self) -> str:
        """Navigator的系统Prompt"""
        return """你是一个论文结构分析专家。

你的任务是准确定位信息位置,而不是提取内容。

输入说明:
- 论文结构由GROBID(专业PDF解析工具)解析
- 每个章节有Type字段(如introduction, discussion, conclusion)
- First/Last是章节首尾段落的预览

关键区分:
- 作者批评'前人工作'的缺点 -> 不是Limitation
- 作者承认'自己方法'的不足 -> 才是Limitation

注意事项:
1. 优先使用GROBID提供的section_type进行定位
2. 如果Type是'unknown',则根据标题和内容推断
3. 对于Limitation/Future Work,重点查看Last段落

重要:输出必须是有效的JSON格式。"""

    def _parse_llm_response(
        self,
        response: str,
        paper: PaperDocument,
        field: FieldType
    ) -> SectionScope:
        """解析LLM的导航响应"""
        import json
        import re

        try:
            # 尝试提取JSON
            json_match = re.search(r'\{[^\}]+\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)

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
                return self._navigate_with_rules(paper, field)

            section_titles = [paper.sections[idx].title for idx in target_sections]

            return SectionScope(
                field=field,
                target_sections=target_sections,
                section_titles=section_titles,
                reasoning=reasoning,
                confidence=confidence
            )

        except Exception as e:
            logger.warning(f"     ⚠️ 解析LLM响应失败: {e}")
            return self._navigate_with_rules(paper, field)

    def _navigate_with_rules(self, paper: PaperDocument, field: FieldType) -> SectionScope:
        """
        使用规则匹配进行导航(fallback)

        基于预定义的section_mapping
        """
        target_types = self.section_mapping.get(field, [])

        target_sections = []
        section_titles = []

        for i, section in enumerate(paper.sections):
            if section.section_type in target_types:
                target_sections.append(i)
                section_titles.append(section.title)

        # 🔧 优化: 如果section_type是'unknown'或'other',使用标题匹配
        if not target_sections:
            logger.info(f"     → section_type匹配失败,尝试标题匹配...")
            target_sections, section_titles = self._match_by_title(paper, field)

        # 如果还是没找到,使用前3个章节
        if not target_sections:
            logger.warning(f"     → 标题匹配也失败,使用前3个章节作为默认范围")
            target_sections = list(range(min(3, len(paper.sections))))
            section_titles = [paper.sections[i].title for i in target_sections]

        return SectionScope(
            field=field,
            target_sections=target_sections,
            section_titles=section_titles,
            reasoning=f"Rule-based matching for {field.value} using section types: {target_types}",
            confidence=0.6
        )

    def _match_by_title(self, paper: PaperDocument, field: FieldType) -> tuple:
        """
        根据章节标题匹配目标章节

        用于处理section_type识别失败的情况(如'Unknown Section')
        """
        # 定义标题关键词映射
        title_keywords_map = {
            FieldType.PROBLEM: [
                'abstract', 'introduction', 'intro', 'background',
                'motivation', 'problem', 'challenge'
            ],
            FieldType.METHOD: [
                'abstract', 'introduction', 'method', 'approach',
                'contribution', 'overview', 'conclusion', 'summary'
            ],
            FieldType.LIMITATION: [
                'limitation', 'discussion', 'conclusion',
                'future work', 'summary', 'result'
            ],
            FieldType.FUTURE_WORK: [
                'future', 'conclusion', 'discussion',
                'outlook', 'direction', 'next step'
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

        logger.info(f"     → 标题匹配找到 {len(target_sections)} 个章节")

        return target_sections, section_titles
