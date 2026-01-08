"""
Dual Stream Synthesizer (双流合成器)
负责将Fast Stream和Slow Stream的结果合并

策略:
- Fast Stream (Anchor): 提供主干信息(高层概念)
- Slow Stream (Evidence): 提供支撑证据(技术细节)
- Synthesizer: 将两者合并成完整的提取结果

核心价值:
- 确保Problem和Method始终有高层描述(来自Abstract/Introduction)
- 用Slow Stream的详细信息补充和验证Fast Stream的结果
- 避免"找不到"或"过于笼统"的问题
"""

import logging
from typing import Optional
from .data_structures import (
    ExtractionResult,
    FieldType
)

logger = logging.getLogger(__name__)


class DualStreamSynthesizer:
    """
    双流合成器
    将Fast Stream和Slow Stream的结果智能合并
    """

    def __init__(self, llm_client):
        """
        初始化Dual Stream Synthesizer

        Args:
            llm_client: LLM客户端
        """
        self.llm_client = llm_client

    def merge(
        self,
        fast_result: ExtractionResult,
        slow_result: ExtractionResult,
        field: FieldType
    ) -> ExtractionResult:
        """
        合并Fast Stream和Slow Stream的结果

        合并策略:
        1. 如果Fast Stream成功且Slow Stream失败 -> 使用Fast Stream
        2. 如果Fast Stream失败且Slow Stream成功 -> 使用Slow Stream
        3. 如果两者都成功 -> 使用LLM智能合并
        4. 如果两者都失败 -> 返回空结果

        Args:
            fast_result: Fast Stream提取结果
            slow_result: Slow Stream提取结果
            field: 字段类型

        Returns:
            ExtractionResult: 合并后的结果
        """
        logger.info(f"  🔀 Dual Stream Synthesizer: 合并'{field.value}'的双流结果...")

        # 判断提取是否成功
        fast_success = self._is_successful(fast_result)
        slow_success = self._is_successful(slow_result)

        logger.info(f"     → Fast Stream: {'✅ 成功' if fast_success else '❌ 失败'}")
        logger.info(f"     → Slow Stream: {'✅ 成功' if slow_success else '❌ 失败'}")

        # 策略1: Fast成功, Slow失败 -> 用Fast
        if fast_success and not slow_success:
            logger.info(f"     → 策略: 使用Fast Stream结果(Slow Stream失败)")
            return self._use_fast_stream(fast_result)

        # 策略2: Fast失败, Slow成功 -> 用Slow
        if not fast_success and slow_success:
            logger.info(f"     → 策略: 使用Slow Stream结果(Fast Stream失败)")
            return self._use_slow_stream(slow_result)

        # 策略3: 两者都成功 -> 智能合并
        if fast_success and slow_success:
            logger.info(f"     → 策略: 智能合并两个流的结果")
            return self._merge_both_streams(fast_result, slow_result, field)

        # 策略4: 两者都失败 -> 返回空结果
        logger.warning(f"     ⚠️ 两个流都失败,返回空结果")
        return ExtractionResult(
            field=field,
            content="未找到相关信息",
            evidence=[],
            confidence=0.0,
            extraction_method="dual_stream_both_failed"
        )

    def _is_successful(self, result: ExtractionResult) -> bool:
        """
        判断提取是否成功

        判断标准:
        - content非空
        - content不是"未找到"类的消息
        - confidence > 0.3
        """
        if not result or not result.content:
            return False

        empty_indicators = [
            "未找到相关信息",
            "未找到",
            "未提取",
            "提取失败",
            "not found",
            "no relevant"
        ]

        content_lower = result.content.strip().lower()
        if any(indicator in content_lower for indicator in empty_indicators):
            return False

        # 内容太短(<20字符)也认为失败
        if len(result.content.strip()) < 20:
            return False

        # confidence太低也认为失败
        if result.confidence < 0.3:
            return False

        return True

    def _use_fast_stream(self, fast_result: ExtractionResult) -> ExtractionResult:
        """
        使用Fast Stream结果

        适用场景: Slow Stream失败,但Fast Stream成功
        """
        # 直接返回Fast Stream结果,标记方法为dual_stream_fast_only
        fast_result.extraction_method = "dual_stream_fast_only"
        logger.info(f"     → 使用Fast Stream结果: {fast_result.content[:100]}...")
        return fast_result

    def _use_slow_stream(self, slow_result: ExtractionResult) -> ExtractionResult:
        """
        使用Slow Stream结果

        适用场景: Fast Stream失败,但Slow Stream成功
        """
        # 直接返回Slow Stream结果,标记方法为dual_stream_slow_only
        slow_result.extraction_method = "dual_stream_slow_only"
        logger.info(f"     → 使用Slow Stream结果: {slow_result.content[:100]}...")
        return slow_result

    def _merge_both_streams(
        self,
        fast_result: ExtractionResult,
        slow_result: ExtractionResult,
        field: FieldType
    ) -> ExtractionResult:
        """
        智能合并两个流的结果

        合并原则:
        - Fast Stream提供主干(高层概念)
        - Slow Stream提供细节(技术证据)
        - 使用LLM进行智能合并,避免冗余

        Args:
            fast_result: Fast Stream结果
            slow_result: Slow Stream结果
            field: 字段类型

        Returns:
            ExtractionResult: 合并后的结果
        """
        # 检查内容是否高度相似(避免重复合并)
        if self._are_contents_similar(fast_result.content, slow_result.content):
            logger.info(f"     → 两个流的内容高度相似,使用Fast Stream结果")
            fast_result.extraction_method = "dual_stream_similar_content"
            # 合并evidence
            fast_result.evidence = fast_result.evidence + slow_result.evidence
            return fast_result

        # 使用LLM进行智能合并
        try:
            merged_content = self._merge_with_llm(
                fast_content=fast_result.content,
                slow_content=slow_result.content,
                field=field
            )

            # 合并evidence(Fast Stream的evidence优先)
            merged_evidence = fast_result.evidence + slow_result.evidence

            # 计算合并后的confidence(取较高值,因为有两个流的支持)
            merged_confidence = max(fast_result.confidence, slow_result.confidence)
            merged_confidence = min(merged_confidence + 0.1, 1.0)  # 略微提高,但不超过1.0

            logger.info(f"     → 合并完成: {merged_content[:100]}...")

            return ExtractionResult(
                field=field,
                content=merged_content,
                evidence=merged_evidence,
                extraction_method="dual_stream_merged",
                confidence=merged_confidence
            )

        except Exception as e:
            logger.error(f"     ❌ LLM合并失败: {e}, 降级使用Fast Stream")
            fast_result.extraction_method = "dual_stream_merge_failed_use_fast"
            return fast_result

    def _are_contents_similar(self, content1: str, content2: str) -> bool:
        """
        判断两个内容是否高度相似

        使用简单的Jaccard相似度
        """
        # 转为小写并分词
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        # 计算Jaccard相似度
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        if union == 0:
            return False

        similarity = intersection / union

        # 相似度 > 0.6认为高度相似
        is_similar = similarity > 0.6
        logger.info(f"     → 内容相似度: {similarity:.2f} ({'相似' if is_similar else '不同'})")

        return is_similar

    def _merge_with_llm(
        self,
        fast_content: str,
        slow_content: str,
        field: FieldType
    ) -> str:
        """
        使用LLM智能合并两个流的内容

        合并原则:
        - Fast Stream作为主干(高层描述)
        - Slow Stream作为补充(技术细节)
        - 避免重复和冗余
        - 保持简洁和结构化
        """
        prompt = self._build_merge_prompt(fast_content, slow_content, field)

        response = self.llm_client.generate(
            prompt=prompt,
            system_prompt=self._get_merge_system_prompt()
        )

        return response.strip()

    def _build_merge_prompt(
        self,
        fast_content: str,
        slow_content: str,
        field: FieldType
    ) -> str:
        """构建合并Prompt"""

        field_instructions = {
            FieldType.PROBLEM: """合并两个Problem描述。

要求:
- 保留Fast Stream的高层问题陈述作为主干
- 如果Slow Stream提供了更多具体细节,适当补充
- 避免重复相同的信息
- 最终输出1-2句话的精炼问题陈述""",

            FieldType.METHOD: """合并两个Method描述。

要求:
- 保留Fast Stream的高层方法概述作为开头
- 补充Slow Stream中的关键技术细节
- 使用bullet points组织
- 每个要点1-2句话,突出创新点
- 避免冗长和重复"""
        }

        instruction = field_instructions.get(
            field,
            "合并两个提取结果,保持简洁和准确"
        )

        prompt = f"""任务: {instruction}

Fast Stream (来自Abstract/Introduction - 高层概念):
{fast_content}

Slow Stream (来自正文章节 - 技术细节):
{slow_content}

请智能合并上述两个提取结果:
1. 以Fast Stream的高层描述为主干
2. 补充Slow Stream中的关键细节
3. 去除重复信息
4. 保持简洁和结构化

合并结果:"""

        return prompt

    def _get_merge_system_prompt(self) -> str:
        """合并系统Prompt"""
        return """你是一个信息整合专家,擅长合并多个来源的信息。

原则:
1. 保留高层概念(来自Fast Stream)
2. 补充关键细节(来自Slow Stream)
3. 去除冗余和重复
4. 保持简洁明了
5. 直接输出合并结果,不要元话

输出风格:
- 对于Problem: 1-2句话的精炼陈述
- 对于Method: 简洁的bullet points(2-4个要点)"""
