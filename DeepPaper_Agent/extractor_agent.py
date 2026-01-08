"""
Extractor Agent (提取员)
负责深入阅读指定章节,提取具体内容

任务:
1. 根据Navigator的指引,读取指定章节
2. 使用Sliding Window + LLM进行语义级精读
3. 提取关键信息并保留原文引用(Evidence)
4. 区分"本文方法"vs"前人工作"

核心价值:
- 解决"关键词匹配不准"的问题
- 语义理解而非简单检索
- 提供可验证的Evidence
"""

import logging
from typing import List, Dict, Optional
from .data_structures import (
    PaperDocument,
    SectionScope,
    ExtractionResult,
    FieldType
)

logger = logging.getLogger(__name__)


class ExtractorAgent:
    """
    提取员Agent
    深度阅读章节并提取关键信息
    """

    def __init__(self, llm_client, max_context_length: int = 3000):
        """
        初始化Extractor Agent

        Args:
            llm_client: LLM客户端
            max_context_length: 最大上下文长度
        """
        self.llm_client = llm_client
        self.max_context_length = max_context_length

    def extract(
        self,
        paper: PaperDocument,
        scope: SectionScope,
        retry_prompt: Optional[str] = None,
        anchor_guidance: Optional[str] = None  # 🆕 来自Fast Stream的锚点指导
    ) -> ExtractionResult:
        """
        提取指定字段的内容

        Args:
            paper: 论文文档
            scope: Navigator指定的章节范围
            retry_prompt: Critic提供的重试指令(可选)
            anchor_guidance: 🆕 来自Fast Stream的高层锚点(可选)

        Returns:
            ExtractionResult: 提取结果
        """
        field = scope.field
        logger.info(f"  📖 Extractor: 提取字段'{field.value}'...")
        logger.info(f"     → 目标章节: {scope.section_titles}")

        # 🆕 如果有anchor guidance,记录日志
        if anchor_guidance:
            logger.info(f"     → 锚点指导: {anchor_guidance[:80]}...")

        # 获取目标章节的内容(从 Navigator 的 scope 中提取)
        # scope.target_sections 是 Navigator 智能定位的章节索引列表
        target_sections = [paper.sections[i] for i in scope.target_sections]

        # 使用Sliding Window从目标章节中提取相关段落
        # 注意: 这里只处理 Navigator 指定的章节,不是所有章节
        relevant_chunks = self._extract_relevant_chunks(
            target_sections,
            field,
            retry_prompt
        )

        if not relevant_chunks:
            logger.warning(f"     ⚠️ 未找到相关内容")
            return ExtractionResult(
                field=field,
                content="未找到相关信息",
                evidence=[],
                confidence=0.0
            )

        # 使用LLM生成提取结果
        # 🆕 传递anchor_guidance到LLM
        extraction = self._extract_with_llm(
            field,
            relevant_chunks,
            paper.title,
            retry_prompt,
            anchor_guidance=anchor_guidance  # 🆕 传递锚点
        )

        logger.info(f"     → 提取完成,找到 {len(extraction.evidence)} 条证据")
        logger.info(f"     → 内容预览: {extraction.content[:100]}...")

        return extraction

    def _extract_relevant_chunks(
        self,
        sections: List,
        field: FieldType,
        retry_prompt: Optional[str] = None
    ) -> List[Dict]:
        """
        从章节中提取相关段落(Sliding Window)
        
        注意: sections 参数已经是 Navigator 指定的目标章节范围,
        不是所有章节。这些章节由 Navigator 根据字段类型智能定位。

        优化策略:
        1. 使用更丰富的关键词(包括review类论文的特殊词汇)
        2. 考虑段落位置(首段/尾段权重更高)
        3. 结合转折词识别
        """
        # 定义关键词(根据字段类型) - 扩展版本
        keywords_map = {
            FieldType.PROBLEM: [
                # 标准问题词
                'problem', 'challenge', 'issue', 'difficulty',
                'gap', 'question', 'task', 'address',
                # Review/工具类论文的问题表述
                'lack', 'absence', 'limitation of existing',
                'incompatible', 'fragmented', 'diverse',
                'need for', 'require', 'motivate', 'why'
            ],
            FieldType.METHOD: [
                # 标准方法词
                'propose', 'method', 'approach', 'model', 'algorithm',
                'technique', 'introduce', 'present', 'develop', 'design',
                # 工具/框架类论文的方法表述
                'provide', 'enable', 'facilitate', 'implement',
                'system', 'framework', 'API', 'interface', 'architecture',
                'improve', 'enhance', 'optimize', 'solution'
            ],
            FieldType.LIMITATION: [
                'limitation', 'drawback', 'weakness', 'shortcoming',
                'however', 'unfortunately', 'lack', 'cannot',
                'difficult', 'challenge', 'remains', 'future work',
                'still', 'yet to', 'not yet', 'constrained',
                'trade-off', 'sacrifice'
            ],
            FieldType.FUTURE_WORK: [
                'future', 'next', 'further', 'improve', 'extend',
                'explore', 'investigate', 'plan', 'ongoing',
                'will', 'could', 'would', 'intend',
                'remain to be', 'open question'
            ]
        }

        keywords = keywords_map.get(field, [])

        # 特别处理:如果是retry且有specific keywords
        if retry_prompt and "look for" in retry_prompt.lower():
            # 从retry_prompt中提取额外关键词
            import re
            extra_keywords = re.findall(r"'([^']+)'|\"([^\"]+)\"", retry_prompt)
            keywords.extend([kw[0] or kw[1] for kw in extra_keywords])

        relevant_chunks = []

        # 遍历 Navigator 指定的目标章节
        for section_idx, section in enumerate(sections):
            # 分割段落
            paragraphs = self._split_into_paragraphs(section.content)

            for para_idx, para in enumerate(paragraphs):
                # 计算关键词匹配度
                keyword_count = sum(
                    1 for kw in keywords
                    if kw.lower() in para.lower()
                )

                if keyword_count > 0:
                    # 🔧 优化: 计算段落得分(考虑位置和转折词)
                    score = keyword_count

                    # 位置权重: 首段和尾段权重高
                    if para_idx == 0:  # 首段
                        score += 1.5
                    elif para_idx == len(paragraphs) - 1:  # 尾段
                        score += 1.2

                    # 转折词权重(对Limitation和Future Work重要)
                    if field in [FieldType.LIMITATION, FieldType.FUTURE_WORK]:
                        transition_words = ['however', 'unfortunately', 'future', 'remains', 'still']
                        if any(tw in para.lower() for tw in transition_words):
                            score += 2.0

                    relevant_chunks.append({
                        'section': section.title,
                        'text': para,
                        'page': section.page_num,
                        'keyword_count': keyword_count,
                        'score': score,  # 综合得分
                        'position': para_idx
                    })

        # 🔧 优化: 按综合得分排序(而非仅关键词数量)
        relevant_chunks.sort(key=lambda x: x['score'], reverse=True)

        # 限制数量(避免上下文过长)
        max_chunks = 8

        # 🔧 如果chunks很少(<3),放宽条件,提取目标章节的所有段落
        if len(relevant_chunks) < 3:
            logger.info(f"     → 关键词匹配chunks较少({len(relevant_chunks)}),扩展到目标章节的所有段落")
            all_chunks = []
            # 注意: sections 仍然是 Navigator 指定的目标章节,不是所有章节
            for section in sections:
                paragraphs = self._split_into_paragraphs(section.content)
                for para in paragraphs[:3]:  # 每个章节取前3段
                    all_chunks.append({
                        'section': section.title,
                        'text': para,
                        'page': section.page_num,
                        'keyword_count': 0,
                        'score': 0.5
                    })
            return all_chunks[:max_chunks]

        return relevant_chunks[:max_chunks]

    def _extract_with_llm(
        self,
        field: FieldType,
        chunks: List[Dict],
        paper_title: str,
        retry_prompt: Optional[str] = None,
        anchor_guidance: Optional[str] = None  # 🆕 来自Fast Stream的锚点
    ) -> ExtractionResult:
        """
        使用LLM从chunks中提取并总结信息

        🆕 如果提供了anchor_guidance,则使用它来指导提取
        """
        # 构建上下文
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(
                f"[Evidence {i+1}] From section '{chunk['section']}':\n{chunk['text']}"
            )
        context = "\n\n".join(context_parts)

        # 限制长度 - 优化: 智能截断,保留关键词最多的chunks
        if len(context) > self.max_context_length:
            # 按关键词密度重新排序chunks,保留最相关的
            top_chunks = chunks[:5]  # 保留前5个最相关的
            context_parts = []
            for i, chunk in enumerate(top_chunks):
                context_parts.append(
                    f"[Evidence {i+1}] From section '{chunk['section']}':\n{chunk['text']}"
                )
            context = "\n\n".join(context_parts)
            logger.info(f"     → 上下文过长,保留前{len(top_chunks)}个最相关段落")

        # 构建提取prompt
        # 🆕 传递anchor_guidance
        prompt = self._build_extraction_prompt(
            field,
            context,
            paper_title,
            retry_prompt,
            anchor_guidance=anchor_guidance  # 🆕 传递锚点
        )

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self._get_extractor_system_prompt(field)
            )

            # 记录原始响应,便于调试
            logger.debug(f"     → LLM原始响应: {response[:200]}...")

            # 解析响应
            content = self._parse_extraction_response(response)

            # 🔧 关键优化: 检测LLM是否过度谨慎返回"未找到"
            if self._is_llm_being_too_cautious(content, chunks):
                logger.warning(f"     ⚠️ LLM返回'未找到',但有{len(chunks)}条证据 - 使用降级策略")
                content = self._fallback_extraction(chunks, field)
                confidence = 0.5
            else:
                confidence = 0.8

            # 构建evidence列表
            evidence = [
                {
                    'section': chunk['section'],
                    'text': chunk['text'],
                    'page': chunk['page']
                }
                for chunk in chunks
            ]

            return ExtractionResult(
                field=field,
                content=content,
                evidence=evidence,
                extraction_method="llm_deep_read",
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"     ❌ LLM提取失败: {e}")
            # 降级:使用规则提取
            fallback_content = self._fallback_extraction(chunks, field)
            return ExtractionResult(
                field=field,
                content=fallback_content,
                evidence=[{'section': chunks[0]['section'], 'text': chunks[0]['text'], 'page': chunks[0]['page']}] if chunks else [],
                confidence=0.4
            )

    def _build_extraction_prompt(
        self,
        field: FieldType,
        context: str,
        paper_title: str,
        retry_prompt: Optional[str] = None,
        anchor_guidance: Optional[str] = None  # 🆕 来自Fast Stream的锚点
    ) -> str:
        """构建提取Prompt

        🆕 使用anchor_guidance来指导Slow Stream的提取
        """

        field_instructions = {
            FieldType.PROBLEM: """提取**本文**要解决的核心研究问题。

⚠️ 关键区分:
- ✅ 提取: 本文(this paper/we/our work)要解决的问题
- ❌ 忽略: Related Work中提到的其他论文的问题
- ❌ 忽略: 对baseline/prior methods的批评

识别技巧:
1. 寻找"we address", "this paper tackles", "our goal is"等第一人称陈述
2. 关注明确说明的gap/challenge/issue
3. 如果提到"existing methods lack X", X就是本文要解决的问题

要求:
- 用1-2句话概括核心问题
- 关注论文开头明确说明的问题
- 避免过于笼统(如"我们需要更好的模型")""",

            FieldType.METHOD: """提取**本文**提出的方法/模型/技术方案。

⚠️ 关键区分:
- ✅ 提取: 本文(this paper/we/our)提出/设计/实现的方法
- ❌ 忽略: Related Work中提到的其他方法
- ❌ 忽略: Baseline/prior methods的描述
- ❌ 忽略: 标准设置(dataset split, batch size, optimizer等)

识别技巧:
1. 寻找"we propose", "we present", "we introduce"等陈述
2. 关注核心创新机制(e.g., 新的loss function, 新的module)
3. 注意"in this work"或"our approach"等明确的自指

要求:
- 描述方法的核心思想和关键技术
- 用简洁的bullet points列出主要技术点
- 突出创新点,不要列举超参数""",

            FieldType.LIMITATION: """提取本文方法的局限性。
⚠️ 重要区分:
- 如果说"LSTM无法处理长序列" -> 这是baseline的问题,不要提取
- 如果说"Our method still struggles with..." -> 这是本文的局限性,必须提取

注意转折词:However, Unfortunately, Future work, remains to be explored

输出格式:
- 列举2-3个具体的局限性
- 每个局限性用1句话说明""",

            FieldType.FUTURE_WORK: """提取未来工作方向。
- 作者明确提到的future work
- 或者从limitation推断出的改进方向
- 用2-3个bullet points列出"""
        }

        base_prompt = f"""论文标题: {paper_title}

任务: {field_instructions[field]}

相关段落:
{context}

"""

        # 🆕 如果有anchor_guidance,添加锚点指导部分
        if anchor_guidance and field in [FieldType.PROBLEM, FieldType.METHOD]:
            if field == FieldType.PROBLEM:
                anchor_instruction = f"""
🎯 锚点指导 (来自Abstract):
{anchor_guidance}

任务: 基于上述锚点,在相关段落中找到:
1. 具体的细节、统计数据或场景来解释为什么这是一个问题
2. 本文作者明确陈述的问题(寻找第一人称: we/our)
3. ❌ 忽略Related Work中其他论文的问题

输出: 结合锚点和段落细节,用1-2句话阐述本文的核心问题
"""
            else:  # METHOD
                anchor_instruction = f"""
🎯 锚点指导 (来自Abstract):
{anchor_guidance}

任务: 基于上述核心方法,在Methodology章节中找到:
1. 实现这个方法的关键机制(e.g., 新的loss function, 新的module)
2. 创新点和核心技术组件
3. ❌ 忽略标准设置(dataset split, batch size, optimizer等)
4. ❌ 忽略baseline方法的描述

输出: 结合锚点和技术细节,用bullet points列出主要方法组件
"""
            base_prompt += anchor_instruction

        # 如果有retry_prompt,加入特殊指令
        if retry_prompt:
            base_prompt += f"\n⚠️ 重试指令:\n{retry_prompt}\n"

        base_prompt += """
输出要求:
1. 直接输出提取的内容(不要输出"根据段落..."这种元话)
2. 内容具体,避免太泛化
3. ⚠️ 重要: 即使信息不完整,也要尽量从段落中提取相关内容
4. 只有在段落完全不相关时才输出"未找到相关信息"
5. 使用bullet points列出要点,每个要点1-2句话

输出:"""

        return base_prompt

    def _get_extractor_system_prompt(self, field: FieldType) -> str:
        """Extractor的系统Prompt"""
        base_prompt = "你是一个资深研究员,擅长精读论文并提取关键信息。"

        if field == FieldType.LIMITATION:
            base_prompt += """
特别注意区分:
- 作者批评前人工作(Prior Work)的缺点 -> 不要提取
- 作者承认自己方法(Ours)的不足 -> 必须提取

主语判断很重要!"""

        return base_prompt

    def _parse_extraction_response(self, response: str) -> str:
        """解析LLM的提取响应"""
        # 移除可能的前缀
        response = response.strip()

        # 移除常见的元话
        prefixes_to_remove = [
            "根据上述段落",
            "根据提供的段落",
            "从段落中可以看出",
            "本文的",
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
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 30]
        return paragraphs

    def _is_llm_being_too_cautious(self, content: str, chunks: List[Dict]) -> bool:
        """
        检测LLM是否过度谨慎返回"未找到"

        判断逻辑:
        - 如果content是"未找到"类的回答
        - 但chunks数量 >= 3 (说明有相关证据)
        - 则认为LLM过度谨慎
        """
        empty_indicators = [
            "未找到相关信息",
            "未找到",
            "没有找到",
            "无相关内容",
            "not found",
            "no relevant",
            "no information"
        ]

        content_lower = content.strip().lower()

        # 检查是否是空回答
        is_empty_response = any(indicator in content_lower for indicator in empty_indicators)

        # 如果是空回答但有多个chunks,说明LLM过度谨慎
        if is_empty_response and len(chunks) >= 3:
            return True

        # 如果回答太短(<20字符)但有chunks
        if len(content.strip()) < 20 and len(chunks) >= 2:
            return True

        return False

    def _fallback_extraction(self, chunks: List[Dict], field: FieldType) -> str:
        """
        降级提取策略: 当LLM失败时,使用规则提取

        策略:
        1. 选择关键词密度最高的1-2个chunks
        2. 简单拼接并清理
        3. 添加前缀说明来源
        """
        if not chunks:
            return "未找到相关信息"

        # 取前2个最相关的chunks
        top_chunks = chunks[:2]

        # 提取核心句子
        extracted_parts = []
        for chunk in top_chunks:
            text = chunk['text'].strip()
            # 简单清理:去除过长的句子(可能是无关内容)
            sentences = text.split('. ')
            relevant_sentences = [s for s in sentences if len(s) < 300][:3]
            extracted_parts.extend(relevant_sentences)

        # 拼接
        content = '. '.join(extracted_parts)

        # 限制长度
        if len(content) > 500:
            content = content[:500] + "..."

        logger.info(f"     → 使用降级策略提取: {len(extracted_parts)}个句子")

        return content if content else chunks[0]['text'][:300]
