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

核心价值:
- ACL级别的创新:Reflection Loop
- 自动质量控制,避免人工核查
- 提升提取的Precision, Recall和Quality
"""

import logging
import re
from typing import Optional, List
from .data_structures import (
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

    def __init__(self, llm_client):
        """
        初始化Critic Agent

        Args:
            llm_client: LLM客户端(用于复杂验证)
        """
        self.llm_client = llm_client

    def critique(
        self,
        extraction: ExtractionResult,
        paper: PaperDocument,
        scope: SectionScope,
        evaluation_level: str = "both"  # 🆕 新增: "concept", "technical", "both"
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

        # 🆕 场景C: Layered Critique (针对Problem和Method)
        if field in [FieldType.PROBLEM, FieldType.METHOD]:
            layered_feedback = self._layered_critique(
                extraction,
                paper,
                evaluation_level,
                extraction.extraction_method  # 🆕 使用extraction_method判断来源
            )
            if layered_feedback and not layered_feedback.approved:
                return layered_feedback

        # 场景D: 内容太泛 (Quality提升)
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

    def _layered_critique(
        self,
        extraction: ExtractionResult,
        paper: PaperDocument,
        evaluation_level: str,
        extraction_method: str
    ) -> Optional[CriticFeedback]:
        """
        🆕 分层审查 (Layered Critique)

        针对Problem和Method字段,检查两个层次:
        1. Concept Level: 是否有清晰的高层概念(小白能懂)
        2. Technical Level: 是否有关键技术细节(专业人士能用)

        判断逻辑:
        - Fast Stream结果: 主要检查Concept Level
        - Slow Stream结果: 主要检查Technical Level
        - Merged结果: 检查Both

        Args:
            extraction: 提取结果
            paper: 论文文档
            evaluation_level: "concept", "technical", "both"
            extraction_method: 提取方法(用于判断来源)

        Returns:
            CriticFeedback: 如果不通过,返回反馈;如果通过,返回None
        """
        field = extraction.field
        content = extraction.content.strip()

        # 根据extraction_method自动判断应该检查哪个层次
        if "fast_stream" in extraction_method:
            # Fast Stream结果:主要检查Concept Level
            check_level = "concept"
        elif extraction_method in ["llm", "llm_deep_read"]:
            # Slow Stream结果:主要检查Technical Level
            check_level = "technical"
        else:
            # 其他情况:检查Both
            check_level = evaluation_level

        logger.info(f"     → 分层审查级别: {check_level}")

        # Level 1: Concept检查
        if check_level in ["concept", "both"]:
            concept_issue = self._check_concept_level(content, field)
            if concept_issue:
                logger.info(f"     ⚠️ Concept Level检查失败: {concept_issue}")
                return self._create_concept_level_feedback(field, concept_issue)

        # Level 2: Technical检查
        if check_level in ["technical", "both"]:
            technical_issue = self._check_technical_level(content, field, extraction)
            if technical_issue:
                logger.info(f"     ⚠️ Technical Level检查失败: {technical_issue}")
                return self._create_technical_level_feedback(field, technical_issue)

        # 两个层次都通过
        return None

    def _check_concept_level(self, content: str, field: FieldType) -> Optional[str]:
        """
        检查Concept Level (高层概念)

        要求:
        - 清晰简单,小白能懂
        - 不应该全是技术术语(jargon)
        - 应该有明确的主语和动词

        Returns:
            如果有问题,返回问题描述;否则返回None
        """
        # 检查1: 内容太短,可能缺少高层概念
        if len(content) < 30:
            return "内容太短,缺少高层概念描述"

        # 检查2: 全是技术术语,缺少清晰表述
        # 统计技术术语密度(大写字母开头的词,如"BERT", "ResNet")
        words = content.split()
        technical_words = [w for w in words if len(w) > 2 and w[0].isupper() and w.isupper()]
        technical_ratio = len(technical_words) / max(len(words), 1)

        if technical_ratio > 0.4:  # 超过40%是技术术语
            return "内容过于技术化,缺少清晰的高层概念描述(小白难懂)"

        # 检查3: 缺少明确的核心陈述
        if field == FieldType.PROBLEM:
            # Problem应该有明确的问题陈述
            problem_indicators = [
                'problem', 'challenge', 'issue', 'difficulty',
                'gap', 'lack', 'absence', 'limitation of existing',
                'incompatible', 'fragmented', '问题', '挑战', '缺乏'
            ]
            if not any(ind in content.lower() for ind in problem_indicators):
                return "缺少明确的问题陈述(应包含problem/challenge/gap等关键词)"

        elif field == FieldType.METHOD:
            # Method应该有明确的方法陈述
            method_indicators = [
                'propose', 'present', 'introduce', 'develop', 'design',
                'method', 'approach', 'model', 'framework', 'system',
                'provide', 'enable', '提出', '方法', '框架'
            ]
            if not any(ind in content.lower() for ind in method_indicators):
                return "缺少明确的方法陈述(应包含propose/method/framework等关键词)"

        return None  # 通过Concept Level检查

    def _check_technical_level(
        self,
        content: str,
        field: FieldType,
        extraction: ExtractionResult
    ) -> Optional[str]:
        """
        检查Technical Level (技术细节)

        要求:
        - 必须包含关键技术名词
        - 应该有具体的技术组件或机制
        - 不应该只有概念没有细节

        Returns:
            如果有问题,返回问题描述;否则返回None
        """
        if field == FieldType.METHOD:
            # Method需要技术细节
            # 检查1: 是否只有高层描述,没有技术组件
            has_bullet_points = any(marker in content for marker in ['-', '•', '*', '1.', '2.'])

            if not has_bullet_points and len(content) < 100:
                return "只有高层方法描述,缺少具体技术组件(应用bullet points列出)"

            # 检查2: 是否包含超参数/标准设置(应该忽略这些)
            hyperparameter_indicators = [
                r'\bbatch.?size',
                r'\blearning.?rate',
                r'\bepoch',
                r'\boptimizer.*adam',
                r'\bdataset.*split',
                r'\btrain.*test.*split'
            ]

            for pattern in hyperparameter_indicators:
                if re.search(pattern, content, re.IGNORECASE):
                    return "包含标准设置/超参数(如batch size, learning rate),应该忽略这些,只提取核心创新机制"

            # 检查3: 是否包含创新机制的关键词
            innovation_indicators = [
                'loss', 'attention', 'module', 'layer', 'architecture',
                'mechanism', 'component', 'algorithm', 'strategy',
                'encoder', 'decoder', 'embedding', 'fusion',
                '损失', '机制', '模块', '架构'
            ]

            if not any(ind in content.lower() for ind in innovation_indicators):
                return "缺少核心创新机制描述(如new loss function, new module, new architecture)"

        elif field == FieldType.PROBLEM:
            # Problem需要具体的细节或场景
            # 检查: 是否有具体描述
            if len(content) < 40:
                return "Problem描述太简短,缺少具体细节或场景说明"

        return None  # 通过Technical Level检查

    def _create_concept_level_feedback(
        self,
        field: FieldType,
        issue: str
    ) -> CriticFeedback:
        """
        创建Concept Level的反馈

        指导Extractor回到Abstract/Introduction补充高层概念
        """
        retry_prompts = {
            FieldType.PROBLEM: """⚠️ Missing High-Level Concept

当前提取内容{issue}。

请回到**Abstract或Introduction的前2-3段**,寻找:
1. 清晰简单的问题陈述(小白能懂)
2. 明确的gap/challenge/issue描述
3. 为什么这个问题重要

输出要求:
- 用1-2句话清晰陈述核心问题
- 避免纯技术术语
- 让非专业人士也能理解

示例:
✅ 好: "Existing NLP tools are fragmented and incompatible, making it difficult to compare models"
❌ 差: "BERT, GPT-2, RoBERTa使用不同的API"
""",

            FieldType.METHOD: """⚠️ Missing High-Level Concept

当前提取内容{issue}。

请回到**Abstract或Introduction**,寻找:
1. 本文提出的核心方法/框架的高层描述
2. 方法的核心思想(what + why)
3. 主要创新点概述

输出要求:
- 先用1句话概括核心方法
- 再用2-3个bullet points列出主要组件
- 突出创新点,不要列举超参数

示例:
✅ 好: "Unified API framework that provides:
      - Standardized interface for model loading
      - Built-in caching mechanism"
❌ 差: "使用ResNet50, batch size=32, learning rate=0.001"
"""
        }

        retry_prompt = retry_prompts.get(field, "请补充高层概念描述")
        retry_prompt = retry_prompt.replace("{issue}", issue)

        return CriticFeedback(
            field=field,
            approved=False,
            feedback_type="missing_context",
            feedback_message=f"Concept Level检查失败: {issue}",
            retry_prompt=retry_prompt
        )

    def _create_technical_level_feedback(
        self,
        field: FieldType,
        issue: str
    ) -> CriticFeedback:
        """
        创建Technical Level的反馈

        指导Extractor深入Methodology章节补充技术细节
        """
        retry_prompts = {
            FieldType.METHOD: """⚠️ Missing Technical Details

当前提取内容{issue}。

请深入**Methodology/Method章节**,寻找:
1. 实现核心方法的关键机制
2. 创新的技术组件(e.g., 新的loss function, 新的module, 新的architecture)
3. 区分"本文创新" vs "baseline方法" vs "标准设置"

⚠️ 重要:
- ✅ 提取: 新的loss function, 新的attention mechanism, 新的module
- ❌ 忽略: batch size, learning rate, optimizer, dataset split
- ❌ 忽略: baseline方法(如"我们对比了BERT")

输出要求:
- 用bullet points列出2-4个核心技术组件
- 每个组件1-2句话,突出创新点
""",

            FieldType.PROBLEM: """⚠️ Missing Problem Details

当前提取内容{issue}。

请在**Introduction或Related Work章节**找到:
1. 具体的统计数据或场景来解释为什么这是一个问题
2. 问题的具体表现或影响
3. 本文作者明确陈述的motivation

输出要求:
- 结合高层问题陈述和具体细节
- 用1-2句话说明问题的重要性
"""
        }

        retry_prompt = retry_prompts.get(field, "请补充技术细节")
        retry_prompt = retry_prompt.replace("{issue}", issue)

        return CriticFeedback(
            field=field,
            approved=False,
            feedback_type="too_generic",
            feedback_message=f"Technical Level检查失败: {issue}",
            retry_prompt=retry_prompt
        )

    def _is_empty_extraction(self, extraction: ExtractionResult) -> bool:
        """
        检查是否为空提取

        优化逻辑:
        1. 不仅检查content,还要检查evidence数量
        2. 区分"真正的空" vs "LLM说未找到但有证据"
        """
        empty_indicators = [
            "未找到相关信息",
            "未找到",
            "没有找到",
            "无相关内容",
            "提取失败",
            "not found",
            "no relevant"
        ]

        content = extraction.content.strip().lower()

        # 场景1: content确实为空或极短
        if not content or len(content) < 10:
            return True

        # 场景2: content包含"未找到"等字样
        has_empty_indicator = any(indicator in content for indicator in empty_indicators)

        if has_empty_indicator:
            # 🔧 关键优化: 如果有证据,说明不是真正的空,而是提取质量问题
            if len(extraction.evidence) >= 3:
                logger.info(f"     → 检测到: content说'未找到'但有{len(extraction.evidence)}条证据")
                return False  # 不算空提取,交给quality检查
            return True

        # 场景3: content太短(<30字符)且没有bullet points
        if len(content) < 30 and '-' not in content and '•' not in content:
            return True

        return False

    def _handle_empty_extraction(
        self,
        extraction: ExtractionResult,
        paper: PaperDocument,
        scope: SectionScope
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
        current_scope: SectionScope
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
        for i, section in enumerate(paper.sections):
            if section.section_type in target_types and i not in current_scope.target_sections:
                new_sections.append(i)

        # 如果还是没有,返回所有章节
        if not new_sections:
            new_sections = list(range(len(paper.sections)))

        return new_sections
