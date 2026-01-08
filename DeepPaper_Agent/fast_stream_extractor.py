"""
Fast Stream Extractor (快速流提取器)
专门负责从Abstract和Introduction提取高层概念

核心价值:
- 不依赖Navigator的章节定位
- 直接从摘要和开头段落快速提取核心思想
- 为Problem和Method提供"主干"信息
- 速度快,精度高(摘要本身就是高质量的总结)

使用场景:
- Problem字段: 从Abstract/Introduction提取核心问题陈述
- Method字段: 从Abstract/Introduction提取高层方法描述
"""

import logging
from typing import Optional, List
from .data_structures import (
    PaperDocument,
    ExtractionResult,
    FieldType
)

logger = logging.getLogger(__name__)


class FastStreamExtractor:
    """
    快速流提取器
    专注于从Abstract和Introduction的前几段提取高层信息
    """

    def __init__(self, llm_client):
        """
        初始化Fast Stream Extractor

        Args:
            llm_client: LLM客户端
        """
        self.llm_client = llm_client

    def extract_anchor(
        self,
        paper: PaperDocument,
        field: FieldType
    ) -> ExtractionResult:
        """
        提取锚点信息(Anchor)

        从Abstract + Introduction的前1-2段提取高层概念
        这些内容通常包含最浓缩和精准的Problem/Method陈述

        Args:
            paper: 论文文档
            field: 字段类型(只支持PROBLEM和METHOD)

        Returns:
            ExtractionResult: 快速流提取结果
        """
        if field not in [FieldType.PROBLEM, FieldType.METHOD]:
            raise ValueError(f"Fast Stream只支持PROBLEM和METHOD,不支持{field.value}")

        logger.info(f"  ⚡ Fast Stream: 快速提取'{field.value}'的锚点信息...")

        # 构建快速流输入:Abstract + Introduction前2段
        fast_context = self._build_fast_context(paper)

        if not fast_context:
            logger.warning(f"     ⚠️ 无法构建快速流上下文(缺少Abstract/Introduction)")
            return ExtractionResult(
                field=field,
                content="",
                evidence=[],
                confidence=0.0,
                extraction_method="fast_stream_failed"
            )

        # 使用LLM提取高层信息
        extraction = self._extract_with_llm(
            field=field,
            context=fast_context,
            paper_title=paper.title
        )

        logger.info(f"     → Fast Stream提取完成")
        logger.info(f"     → 锚点信息预览: {extraction.content[:100]}...")

        return extraction

    def _build_fast_context(self, paper: PaperDocument) -> str:
        """
        构建快速流上下文

        策略:
        1. 优先使用Abstract(如果存在)
        2. 提取Introduction的前2-3段
        3. 限制总长度在1500字符左右(保证快速和精准)

        Returns:
            快速流上下文文本
        """
        context_parts = []

        # 1. Abstract(如果有)
        if paper.abstract and len(paper.abstract.strip()) > 50:
            context_parts.append(f"=== Abstract ===\n{paper.abstract}\n")

        # 2. Introduction的前2-3段
        intro_section = None
        for section in paper.sections:
            if section.section_type in ['introduction', 'intro']:
                intro_section = section
                break

        if intro_section:
            paragraphs = self._split_into_paragraphs(intro_section.content)
            # 取前2-3段(通常包含问题陈述和方法概述)
            top_paragraphs = paragraphs[:3]

            if top_paragraphs:
                intro_text = "\n\n".join(top_paragraphs)
                context_parts.append(f"=== Introduction (First Paragraphs) ===\n{intro_text}\n")

        # 合并上下文
        full_context = "\n".join(context_parts)

        # 限制长度(避免过长)
        max_length = 2000
        if len(full_context) > max_length:
            full_context = full_context[:max_length] + "..."
            logger.info(f"     → 快速流上下文过长,截断至{max_length}字符")

        logger.info(f"     → 快速流上下文长度: {len(full_context)}字符")

        return full_context

    def _extract_with_llm(
        self,
        field: FieldType,
        context: str,
        paper_title: str
    ) -> ExtractionResult:
        """
        使用LLM从快速流上下文中提取信息

        快速流的Prompt特点:
        - 强调"高层概念"和"核心思想"
        - 要求简洁(1-2句话)
        - 不需要技术细节
        """
        prompt = self._build_fast_stream_prompt(field, context, paper_title)

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self._get_fast_stream_system_prompt(field)
            )

            # 解析响应
            content = self._parse_response(response)

            # 构建evidence(来自Abstract/Introduction)
            evidence = [{
                'section': 'Abstract/Introduction',
                'text': context[:500],  # 保留前500字符作为evidence
                'page': 0,
                'source': 'fast_stream'
            }]

            return ExtractionResult(
                field=field,
                content=content,
                evidence=evidence,
                extraction_method="fast_stream",
                confidence=0.9  # 快速流通常精度很高
            )

        except Exception as e:
            logger.error(f"     ❌ Fast Stream LLM提取失败: {e}")
            return ExtractionResult(
                field=field,
                content="",
                evidence=[],
                confidence=0.0,
                extraction_method="fast_stream_error"
            )

    def _build_fast_stream_prompt(
        self,
        field: FieldType,
        context: str,
        paper_title: str
    ) -> str:
        """构建快速流Prompt"""

        if field == FieldType.PROBLEM:
            instruction = """提取**本文**要解决的核心研究问题/挑战。

⚠️ 关键区分:
- ✅ 提取: 本文(this paper/we/our work)要解决的问题
- ❌ 忽略: Related Work中提到的其他论文的问题
- ❌ 忽略: 对baseline/prior methods的批评

识别技巧:
1. 寻找"we address", "this paper tackles", "our goal is"等第一人称陈述
2. 关注Abstract中明确说明的gap/challenge/issue
3. 如果提到"existing methods lack X", X就是本文要解决的问题

要求:
- 用1-2句话概括核心问题
- 关注"what problem"而不是"how to solve"
- 避免过于泛化(如"需要更好的模型")
- 直接说明问题本身,不要加"本文解决了..."这种元话

输出格式: 直接输出问题陈述(不要前缀,不要元话)"""

        else:  # METHOD
            instruction = """提取本文提出的核心方法/技术方案的高层描述。

要求:
- 用2-3个bullet points概括主要方法
- 关注"what"和"why",不要过多技术细节
- 突出创新点和关键思想
- 区分"本文方法" vs "前人baseline"

输出格式:
- 方法1: 简短描述(1句话)
- 方法2: 简短描述(1句话)
- ..."""

        prompt = f"""论文标题: {paper_title}

任务: {instruction}

上下文 (Abstract + Introduction首段):
{context}

请仔细阅读上述内容,提取核心的高层信息。
注意:这是快速流提取,只需要高层概念,不需要技术细节。

输出:"""

        return prompt

    def _get_fast_stream_system_prompt(self, field: FieldType) -> str:
        """快速流的系统Prompt"""

        base = """你是一个论文速读专家,擅长从摘要和开头快速抓住核心思想。

关键原则:
1. 只提取高层概念,不要技术细节
2. 简洁明了,避免冗长
3. 直接输出内容,不要元话(如"根据摘要...")
4. Abstract和Introduction的前几段通常包含最精准的陈述"""

        if field == FieldType.PROBLEM:
            base += """

特别提示(Problem):
- 寻找明确的问题陈述(challenge, gap, issue, lack)
- Review类论文可能用"现有工具不兼容/零散"等表述
- 工具类论文可能用"缺少统一框架/接口"等表述"""

        else:  # METHOD
            base += """

特别提示(Method):
- 寻找"propose", "present", "introduce", "provide"等动词
- 关注创新点和核心贡献
- 工具类论文的方法可能是"提供统一API/框架/接口"等"""

        return base

    def _parse_response(self, response: str) -> str:
        """解析LLM响应"""
        response = response.strip()

        # 移除常见的元话前缀
        prefixes_to_remove = [
            "根据摘要",
            "根据上述内容",
            "本文的",
            "本文提出的",
        ]

        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
                if response.startswith(',') or response.startswith(','):
                    response = response[1:].strip()

        return response

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """分割段落"""
        import re
        paragraphs = re.split(r'\n\s*\n|\n', text)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]
        return paragraphs
