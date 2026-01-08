"""
Critic Agent (审查员/反思者)
这是Multi-Agent系统的核心创新点!

任务:
1. 验证Extractor提取的内容是否准确
2. 检查三种典型问题:
   - 提取为空(Recall问题)
   - 提取错误(Precision问题,如提取了baseline的limitation)
   - 内容太泛(Quality问题)
3. 给出具体的改进指令,指导Extractor重试
4. 触发重新提取并迭代改进

核心价值:
- ACL级别的创新:Reflection Loop
- 自动质量控制,避免人工核查
- 提升提取的Precision, Recall和Quality
- 持续迭代直到达到质量标准
"""

import logging
import re
from typing import Optional, List, Callable
from data_structures import (
    ExtractionResult,
    CriticFeedback,
    FieldType,
    PaperDocument,
    SectionScope
)

logger = logging.getLogger(__name__)


class CriticAgent:
    """
    审查员Agent
    验证提取质量并提供反馈
    """

    def __init__(self, llm_client, max_iterations: int = 3, strict_mode: bool = False):
        """
        初始化Critic Agent

        Args:
            llm_client: LLM客户端(用于复杂验证)
            max_iterations: 最大重试次数(默认3次)
            strict_mode: 严格模式(False时更宽容,适合困难论文)
        """
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.strict_mode = strict_mode

    def critique_and_retry(
        self,
        extraction: ExtractionResult,
        paper: PaperDocument,
        extractor_func: Callable,
        scope: Optional[SectionScope] = None,
        evaluation_level: str = "both"
    ) -> ExtractionResult:
        """
        审查并重试提取(主入口)

        🆕 自动重试机制:
        1. 审查当前提取结果
        2. 如果不通过,使用反馈指导重新提取
        3. 重复直到通过或达到最大迭代次数

        Args:
            extraction: 初始提取结果
            paper: 论文文档
            extractor_func: 提取函数,签名为 func(paper, feedback) -> ExtractionResult
            scope: 章节范围(可选)
            evaluation_level: 评估级别

        Returns:
            ExtractionResult: 最终通过审查的结果
        """
        current_extraction = extraction
        iteration = 0

        logger.info(f"  🔄 开始 Critic-Retry Loop (最多{self.max_iterations}次)")

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"\n  📍 Iteration {iteration}/{self.max_iterations}")

            # 审查当前结果
            feedback = self.critique(
                extraction=current_extraction,
                paper=paper,
                scope=scope,
                evaluation_level=evaluation_level
            )

            # 如果通过审查,直接返回
            if feedback.approved:
                logger.info(f"  ✅ 审查通过! (迭代{iteration}次)")
                current_extraction.iterations = iteration
                return current_extraction

            # 如果未通过且已达最大迭代次数
            if iteration >= self.max_iterations:
                logger.warning(f"  ⚠️ 达到最大迭代次数({self.max_iterations}),返回当前结果")
                current_extraction.iterations = iteration
                return current_extraction

            # 否则,使用反馈重新提取
            logger.info(f"  🔧 审查未通过,触发重新提取...")
            logger.info(f"     反馈: {feedback.feedback_message}")

            try:
                # 调用extractor重新提取
                new_extraction = extractor_func(paper, feedback)

                # 检查新结果是否有改进
                if self._has_improvement(current_extraction, new_extraction):
                    logger.info(f"  📈 提取结果已改进")
                    current_extraction = new_extraction
                else:
                    logger.warning(f"  ⚠️ 提取结果无明显改进,保留当前结果")
                    # 仍然更新为新结果,因为可能是不同角度的提取
                    current_extraction = new_extraction

            except Exception as e:
                logger.error(f"  ❌ 重新提取失败: {e}")
                current_extraction.iterations = iteration
                return current_extraction

        current_extraction.iterations = iteration
        return current_extraction

    def critique(
        self,
        extraction: ExtractionResult,
        paper: PaperDocument,
        scope: Optional[SectionScope] = None,
        evaluation_level: str = "both"
    ) -> CriticFeedback:
        """
        审查提取结果

        🆕 Layered Critique Strategy:
        - Level 1 (Concept): 必须清晰简单,小白能懂(主要来自Abstract)
        - Level 2 (Technical): 必须包含关键技术名词(主要来自Body)

        Args:
            extraction: Extractor的提取结果
            paper: 论文文档
            scope: 章节范围
            evaluation_level: 评估级别 ("concept", "technical", "both")

        Returns:
            CriticFeedback: 反馈和改进建议
        """
        field = extraction.field
        logger.info(f"  🔍 Critic: 审查字段'{field.value}'的提取结果...")
        logger.info(f"     → 评估级别: {evaluation_level}")

        # 场景A: 提取为空 (Recall提升)
        if self._is_empty_extraction(extraction):
            return self._handle_empty_extraction(extraction, paper, scope)

        # 场景B: 提取错误 (Precision提升)
        if field == FieldType.LIMITATION:
            wrong_target_feedback = self._check_wrong_target(extraction, paper)
            if wrong_target_feedback:
                return wrong_target_feedback

        # 场景C: 内容太泛 (Quality提升)
        if self._is_too_generic(extraction):
            return self._handle_too_generic(extraction, paper)

        # 通过审查
        logger.info(f"     ✅ 审查通过")
        return CriticFeedback(
            field=field,
            approved=True,
            feedback_type="approved",
            feedback_message="提取质量良好,通过审查"
        )

    def _handle_empty_extraction(
        self,
        extraction: ExtractionResult,
        paper: PaperDocument,
        scope: Optional[SectionScope]
    ) -> CriticFeedback:
        """
        处理空提取(Recall提升策略)

        策略:
        - 建议Extractor检查转折词(However, Future work, remains)
        - 建议扩展搜索范围
        - 给出具体的retry prompt
        """
        field = extraction.field
        logger.info(f"     ⚠️ 提取为空,生成Recall提升反馈...")

        # 根据字段类型给出针对性建议
        retry_strategies = {
            FieldType.PROBLEM: {
                'message': 'Problem通常在Abstract或Introduction开头明确说明。请重新检查这些章节的前几段。',
                'prompt': """请仔细检查Abstract和Introduction的开头段落,寻找描述"要解决的问题"或"研究动机"的句子。

对于Review类论文,问题可能表述为:
- "存在多种不兼容的实现"
- "缺乏统一的接口"
- "需要更好的互操作性"

请从段落中直接提取,不要说"未找到"。"""
            },
            FieldType.METHOD: {
                'message': 'Method可能用"we propose", "our method", "approach"等词表述。',
                'prompt': """请寻找包含以下关键词的段落:
- "propose", "method", "approach", "model", "algorithm"
- "present", "develop", "design", "implement"

对于工具/框架类论文,方法可能描述为:
- 设计了某个系统/框架
- 实现了某种技术方案
- 提供了某个工具/接口

请描述2-3个核心技术点,每个用1-2句话说明。"""
            },
            FieldType.LIMITATION: {
                'message': 'Limitation可能隐藏在Discussion或Conclusion末尾的转折词后。',
                'prompt': """大多数论文会在Discussion或Conclusion的结尾用转折词暗示局限性。
请重新检查这些章节的最后几段,专门寻找:
- However, ...
- Unfortunately, ...
- Future work could address...
- One limitation is...
- It remains challenging to...
- Still faces challenges with...

⚠️ 如果实在没有明确的limitation,可以从Future Work推断。
注意:只提取本文方法的局限性,不要提取baseline的缺点。"""
            },
            FieldType.FUTURE_WORK: {
                'message': 'Future Work通常在Conclusion或独立的Future Work章节中。',
                'prompt': """请检查Conclusion章节或标题包含"Future"的章节,寻找未来工作方向。

关键词:
- "future work", "future research", "next step"
- "plan to", "will explore", "could be extended"
- "remains to be", "would benefit from"

如果没有明确的Future Work章节,可以从Limitation或Discussion末尾推断改进方向。"""
            }
        }

        strategy = retry_strategies.get(field, {
            'message': f'未找到{field.value},请扩展搜索范围。',
            'prompt': f'请重新阅读论文,寻找与{field.value}相关的内容。'
        })

        # 建议新的章节(扩展范围)
        suggested_sections = self._suggest_fallback_sections(field, paper, scope)

        return CriticFeedback(
            field=field,
            approved=False,
            feedback_type="empty_retry",
            feedback_message=strategy['message'],
            suggested_sections=suggested_sections,
            retry_prompt=strategy['prompt']
        )

    def _check_wrong_target(
        self,
        extraction: ExtractionResult,
        paper: PaperDocument
    ) -> Optional[CriticFeedback]:
        """
        检查是否提取错误(特别针对Limitation字段)

        判断逻辑:
        - 如果主语是"LSTM", "RNN", "CNN", "previous work"等 -> 错误
        - 如果主语是"our method", "we", "the proposed"等 -> 正确
        """
        content = extraction.content

        # 使用规则检查
        wrong_indicators = [
            r'\bLSTM\b',
            r'\bRNN\b',
            r'\bCNN\b',
            r'\bprevious (work|method|approach)',
            r'\bprior (work|method|approach)',
            r'\bexisting (method|approach)',
            r'\bbaseline',
            r'\btraditional (method|approach)'
        ]

        for pattern in wrong_indicators:
            if re.search(pattern, content, re.IGNORECASE):
                # 可能是错误提取,进一步确认
                # 检查是否有否定词(如"unlike LSTM, our method...")
                if not re.search(r'\b(unlike|different from|compared to)\b', content, re.IGNORECASE):
                    logger.info(f"     ⚠️ 检测到可能的错误提取(主语为前人工作)")

                    # 如果有LLM,使用LLM验证
                    if self.llm_client:
                        is_wrong = self._verify_wrong_target_with_llm(content, paper.title)
                        if is_wrong:
                            return self._create_wrong_target_feedback(extraction)
                    else:
                        # 无LLM,直接判定为错误
                        return self._create_wrong_target_feedback(extraction)

        return None

    def _verify_wrong_target_with_llm(self, content: str, paper_title: str) -> bool:
        """使用LLM验证是否提取了错误的对象"""
        prompt = f"""论文标题: {paper_title}

提取的Limitation内容:
{content}

问题: 这段内容是在说"本文方法的局限性"还是"前人工作/baseline的缺点"?

判断规则:
- 如果主语是"LSTM", "CNN", "previous methods"等 -> 这是在说前人工作的缺点
- 如果主语是"our method", "the proposed approach", "we"等 -> 这是在说本文的局限性

请回答: "本文方法" 或 "前人工作"

回答:"""

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt="你是一个论文分析专家,擅长区分本文方法和前人工作。"
            )

            if "前人" in response or "baseline" in response.lower() or "prior" in response.lower():
                return True  # 错误提取

        except Exception as e:
            logger.warning(f"     ⚠️ LLM验证失败: {e}")

        return False

    def _create_wrong_target_feedback(self, extraction: ExtractionResult) -> CriticFeedback:
        """创建错误目标的反馈"""
        return CriticFeedback(
            field=extraction.field,
            approved=False,
            feedback_type="wrong_target",
            feedback_message="检测到可能提取了baseline或前人工作的缺点,而非本文方法的局限性。",
            retry_prompt="""请重新检查原文。

⚠️ 重要区分:
- 如果句子主语是"LSTM", "RNN", "previous methods" -> 这是在批评前人工作,不要提取
- 如果句子主语是"our method", "the proposed", "we" -> 这才是本文的局限性,必须提取

请只提取本文方法(our/proposed)的局限性。"""
        )

    def _is_too_generic(self, extraction: ExtractionResult) -> bool:
        """
        检查内容是否太泛化

        例如:
        - "Our method needs more data" (太泛,应该说明需要什么类型的数据)
        - "The model is slow" (太泛,应该说明在什么场景下慢)
        """
        content = extraction.content.strip()

        # 太短可能太泛
        if len(content) < 50:
            return True

        # 检查泛化指标词
        generic_patterns = [
            r'\bmore data\b',
            r'\bmore training\b',
            r'\bimprove\b.*\bperformance\b',
            r'\bfurther study\b',
            r'\blarge-scale\b.*\bexperiment',
        ]

        # 如果只是简单提到这些而没有具体说明
        for pattern in generic_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                # 检查是否有具体描述
                if len(content.split()) < 30:  # 词数太少
                    return True

        return False

    def _handle_too_generic(
        self,
        extraction: ExtractionResult,
        paper: PaperDocument
    ) -> CriticFeedback:
        """处理内容太泛的情况(Quality提升)"""
        logger.info(f"     ⚠️ 内容太泛化,生成Quality提升反馈...")

        field = extraction.field

        retry_prompt_map = {
            FieldType.LIMITATION: """提取的局限性太笼统。请回到原文,查找:
- 具体需要什么类型的数据?
- 在哪种具体场景下表现不佳?
- 具体有什么技术瓶颈?

请提供更详细的描述。""",

            FieldType.FUTURE_WORK: """提取的未来工作太笼统。请查找:
- 具体要改进什么?
- 具体要探索什么方向?
- 具体要做什么实验?

请提供更具体的未来工作描述。""",

            FieldType.PROBLEM: """提取的研究问题太泛。请明确:
- 具体是什么问题?
- 在什么场景下出现?
- 为什么这个问题重要?""",

            FieldType.METHOD: """提取的方法描述太笼统。请具体说明:
- 提出了什么具体的方法/模型/算法?
- 方法的核心技术是什么?
- 具体是如何实现的?"""
        }

        retry_prompt = retry_prompt_map.get(
            field,
            "提取内容太笼统,请提供更具体的描述。"
        )

        return CriticFeedback(
            field=field,
            approved=False,
            feedback_type="too_generic",
            feedback_message="提取内容太泛化,需要更具体的描述",
            retry_prompt=retry_prompt
        )

    def _suggest_fallback_sections(
        self,
        field: FieldType,
        paper: PaperDocument,
        current_scope: Optional[SectionScope]
    ) -> List[int]:
        """
        建议fallback章节(当当前范围未找到时)

        策略:扩展到更多可能的章节
        """
        # 扩展映射
        fallback_mapping = {
            FieldType.PROBLEM: ['abstract', 'introduction', 'related_work'],
            FieldType.METHOD: ['abstract', 'introduction', 'method', 'conclusion', 'experiment'],
            FieldType.LIMITATION: ['discussion', 'conclusion', 'experiment', 'method'],
            FieldType.FUTURE_WORK: ['conclusion', 'discussion', 'limitation']
        }

        target_types = fallback_mapping.get(field, [])

        # 找到新的章节(不在current_scope中的)
        new_sections = []
        current_sections = current_scope.target_sections if current_scope else []

        for i, section in enumerate(paper.sections):
            if section.section_type in target_types and i not in current_sections:
                new_sections.append(i)

        # 如果还是没有,返回所有章节
        if not new_sections:
            new_sections = list(range(len(paper.sections)))

        return new_sections

    def _has_improvement(
        self,
        old_extraction: ExtractionResult,
        new_extraction: ExtractionResult
    ) -> bool:
        """
        判断新提��结果是��比旧结果有改进

        改进指标:
        1. 内容长度显著增加(从空到非空,或长度增加>30%)
        2. 证据数量增加
        3. 置信度提升
        4. 不再包含"未找到"等空指标

        Args:
            old_extraction: 旧的提取结果
            new_extraction: 新的提取结果

        Returns:
            bool: 是否有改进
        """
        old_content = old_extraction.content.strip()
        new_content = new_extraction.content.strip()

        # 指标1: 从空到非空
        old_is_empty = self._is_empty_extraction(old_extraction)
        new_is_empty = self._is_empty_extraction(new_extraction)

        if old_is_empty and not new_is_empty:
            logger.info(f"     ✓ 改进: 从空提取变为有内容")
            return True

        if not old_is_empty and new_is_empty:
            logger.warning(f"     ✗ 退步: 从有内容变为空提取")
            return False

        # 指标2: 内容长度显著增加
        old_length = len(old_content)
        new_length = len(new_content)

        if new_length > old_length * 1.3:  # 增加>30%
            logger.info(f"     ✓ 改进: 内容长度增加 {old_length} -> {new_length}")
            return True

        # 指标3: 证据数量增加
        if len(new_extraction.evidence) > len(old_extraction.evidence):
            logger.info(f"     ✓ 改进: 证据数量增加 {len(old_extraction.evidence)} -> {len(new_extraction.evidence)}")
            return True

        # 指标4: 置信度提升
        if new_extraction.confidence > old_extraction.confidence + 0.1:
            logger.info(f"     ✓ 改进: 置信度提升 {old_extraction.confidence:.2f} -> {new_extraction.confidence:.2f}")
            return True

        # 指标5: 移除了"未找到"等空指标
        empty_indicators = [
            "未找到相关信息", "未找到", "没有找到",
            "无相关内容", "提取失败", "not found", "no relevant"
        ]

        old_has_empty_indicator = any(ind in old_content.lower() for ind in empty_indicators)
        new_has_empty_indicator = any(ind in new_content.lower() for ind in empty_indicators)

        if old_has_empty_indicator and not new_has_empty_indicator:
            logger.info(f"     ✓ 改进: 移除了空指标词")
            return True

        # 指标6: 结构化程度提升(有bullet points)
        old_has_structure = any(marker in old_content for marker in ['-', '•', '*', '1.', '2.'])
        new_has_structure = any(marker in new_content for marker in ['-', '•', '*', '1.', '2.'])

        if not old_has_structure and new_has_structure:
            logger.info(f"     ✓ 改进: 内容变得更结构化(有bullet points)")
            return True

        logger.info(f"     → 无明显改进")
        return False

    def _is_empty_extraction(self, extraction: ExtractionResult) -> bool:
        """
        增强的空提取检测

        检测策略:
        1. 内容为空或极短(<10字符)
        2. 包含"未找到"等空指标词
        3. 内容太短(<30字符)且无结构化标记
        4. 特殊情况: 说"未找到"但有证据(不算��,是质量问题)

        Args:
            extraction: 提取结果

        Returns:
            bool: 是否为空提取
        """
        content = extraction.content.strip().lower()

        # 场景1: 内容确实为空或极短
        if not content or len(content) < 10:
            logger.info(f"     → 空提取检测: 内容为空或极短(长度={len(content)})")
            return True

        # 场景2: 内容包含"未找到"等字样
        empty_indicators = [
            "未找到相关信息", "未找到", "没有找到", "无相关内容",
            "提取失败", "not found", "no relevant", "no information",
            "cannot find", "not available", "未提取", "无法提取",
            "未明确", "不明确", "unclear", "not specified"
        ]

        has_empty_indicator = any(indicator in content for indicator in empty_indicators)

        if has_empty_indicator:
            # 🔧 关键优化: 如果有证据,说明不是真正的空,而是提取质量问题
            if len(extraction.evidence) >= 2:
                logger.info(f"     → 空提取检测: content说'未找到'但有{len(extraction.evidence)}条证据 -> 非空(质量问题)")
                return False
            logger.info(f"     → 空提取检测: 包含空指标词且证据不足")
            return True

        # 场景3: 内容太短(<30字符)且没有结构化标记
        if len(content) < 30:
            has_structure = any(marker in extraction.content for marker in ['-', '•', '*', '1.', '2.'])
            if not has_structure:
                logger.info(f"     → 空提取检测: 内容太短({len(content)}字符)且无结构")
                return True

        # 场景4: 只有泛泛的陈述,没有具体信息
        generic_only_patterns = [
            r'^(本文|the paper|this paper|our method).*$',  # 只有主语没有具体内容
            r'^(we|they|it).*但.*$',  # 只有转折没有细节
        ]

        if len(content) < 50:  # 只检查很短的内容
            for pattern in generic_only_patterns:
                if re.match(pattern, content, re.IGNORECASE):
                    logger.info(f"     → 空提取检测: 只有泛泛陈述,无具体信息")
                    return True

        return False
