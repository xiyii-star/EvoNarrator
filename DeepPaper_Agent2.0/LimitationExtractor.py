"""
Limitation Extractor (局限性提取器)
结合章节定位和引用分析提取论文的局限性

工作流程:
1. 使用 SectionLocatorAgent 定位 limitation 章节
2. 从定位的章节中提取 limitation
3. 使用 CitationDetectiveAgent 从引用中提取额外的 limitation
4. 合并两部分结果
5. 🆕 使用 CriticAgent 审查并自动重试提取
"""

import json
import logging
import re
from typing import List, Dict, Optional, Any
from pathlib import Path

# 导入数据结构
from data_structures import PaperDocument, ExtractionResult, FieldType, SectionScope, CriticFeedback

# 导入其他Agent
from SectionLocatorAgent import SectionLocatorAgent
from CitationDetectiveAgent import CitationDetectiveAgent
from critic_agent import CriticAgent

# 导入LLM配置
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.llm_config import LLMClient, LLMConfig

logger = logging.getLogger(__name__)


class LimitationExtractor:
    """
    Limitation 提取器
    结合章节定位和引用分析
    """

    def __init__(
        self,
        llm_client: LLMClient,
        use_citation_analysis: bool = False,
        use_critic: bool = True,
        max_context_length: int = 3000,
        max_iterations: int = 3
    ):
        """
        初始化 Limitation 提取器

        Args:
            llm_client: LLM客户端
            use_citation_analysis: 是否使用引用分析
            use_critic: 是否使用CriticAgent进行质量检查和重试
            max_context_length: 最大上下文长度
            max_iterations: 最大重试次数(当use_critic=True时)
        """
        self.llm_client = llm_client
        self.use_citation_analysis = use_citation_analysis
        self.use_critic = use_critic
        self.max_context_length = max_context_length

        # 初始化子组件
        self.locator = SectionLocatorAgent(llm_client)
        if use_citation_analysis:
            self.citation_detective = CitationDetectiveAgent(llm_client)
        else:
            self.citation_detective = None

        # 🆕 初始化 CriticAgent
        if use_critic:
            self.critic = CriticAgent(llm_client, max_iterations=max_iterations)
        else:
            self.critic = None

        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """You are a professional expert in scientific paper analysis.

Your Task: Extract the limitations of the method proposed in this paper.

⚠️ Important Distinction:

✅ Extract: Limitations of "this paper" / "our method" / "we".
❌ Ignore: Criticisms of "prior work" / "baseline methods" made by the authors.
❌ Ignore: Weaknesses of other papers mentioned in the "Related Work" section.
Identification Techniques:

Look for transition words: However, Unfortunately, Limitation, still, yet.
Focus on self-references: "our method", "our approach", "the proposed framework".
Pay attention to honest descriptions usually found at the end of the Discussion or Conclusion sections.
Output Requirements:

List 2-4 specific limitations.
Use 1-2 sentences to explain each limitation.
Use bullet points format.
Output the content directly, avoiding meta-talk (e.g., do not say "According to the paragraph...").

"""

    def extract(
        self,
        paper: PaperDocument,
        paper_id: Optional[str] = None,
        feedback: Optional[CriticFeedback] = None
    ) -> ExtractionResult:
        """
        提取论文的局限性

        Args:
            paper: 论文文档
            paper_id: 论文ID（用于引用分析，可选）
            feedback: CriticAgent反馈（用于重试，可选）

        Returns:
            ExtractionResult: 提取结果
        """
        logger.info("📋 开始提取 Limitation...")

        # 第一部分: 从论文章节提取
        logger.info("  📖 [Part 1] 从论文章节提取...")
        section_limitations, section_evidence, scope = self._extract_from_sections(paper, feedback)

        # 第二部分: 从引用分析提取（如果启用）
        citation_limitations = []
        if self.use_citation_analysis and paper_id:
            logger.info("  🔍 [Part 2] 从引用分析提取...")
            citation_limitations = self._extract_from_citations(paper_id)

        # 合并结果
        merged_content, merged_evidence = self._merge_results(
            section_limitations,
            section_evidence,
            citation_limitations
        )

        logger.info(f"  ✅ Limitation初步提取完成")
        logger.info(f"     → 章节提取: {len(section_limitations)} 条")
        logger.info(f"     → 引用提取: {len(citation_limitations)} 条")
        logger.info(f"     → 最终内容: {merged_content[:100]}...")

        initial_result = ExtractionResult(
            field=FieldType.LIMITATION,
            content=merged_content,
            evidence=merged_evidence,
            extraction_method="section_locator + citation_detective" if self.use_citation_analysis else "section_locator",
            confidence=0.8 if section_limitations else 0.3,
            iterations=1
        )

        # 🆕 使用 CriticAgent 进行质量检查和自动重试
        if self.use_critic and self.critic and not feedback:  # 只在初次提取时使用critic
            logger.info("\n  🔍 启动 CriticAgent 质量检查...")

            # 定义重新提取函数
            def retry_extract(paper_doc, critic_feedback):
                return self.extract(paper_doc, paper_id, critic_feedback)

            # 调用 critique_and_retry
            final_result = self.critic.critique_and_retry(
                extraction=initial_result,
                paper=paper,
                extractor_func=retry_extract,
                scope=scope,
                evaluation_level="both"
            )

            return final_result

        return initial_result

    def _extract_from_sections(self, paper: PaperDocument, feedback: Optional[CriticFeedback] = None) -> tuple:
        """
        从论文章节提取局限性

        Args:
            paper: 论文文档
            feedback: Critic反馈(用于重试)

        Returns:
            (limitations列表, evidence列表, scope)
        """
        # 步骤1: 定位章节 (如果有反馈,使用建议的章节)
        if feedback and feedback.suggested_sections:
            logger.info(f"  → 使用Critic建议的章节: {feedback.suggested_sections}")
            # 创建一个临时scope
            scope = SectionScope(
                field=FieldType.LIMITATION,
                target_sections=feedback.suggested_sections,
                section_titles=[paper.sections[i].title for i in feedback.suggested_sections if i < len(paper.sections)],
                reasoning="Based on Critic feedback"
            )
        else:
            scope = self.locator.locate(paper, FieldType.LIMITATION)

        if not scope.target_sections:
            logger.warning("  ⚠️ 未找到相关章节")
            return [], [], scope

        # 步骤2: 提取相关段落
        relevant_chunks = self._extract_relevant_chunks(paper, scope)

        if not relevant_chunks:
            logger.warning("  ⚠️ 未找到相关段落")
            return [], [], scope

        # 步骤3: 使用LLM提取 (带上feedback以增强提示)
        limitations, evidence = self._extract_with_llm(relevant_chunks, paper.title, feedback)

        return limitations, evidence, scope

    def _extract_relevant_chunks(
        self,
        paper: PaperDocument,
        scope: SectionScope
    ) -> List[Dict]:
        """
        提取相关段落（使用关键词匹配）

        🆕 优化策略:
        1. 扩展关键词列表（包括review类论文的特殊词汇）
        2. 考虑段落位置（首段/尾段权重更高）
        3. 结合转折词识别
        4. 智能降级：当匹配少时自动扩展到所有段落
        """
        # 🆕 扩展关键词列表（参考DeepPaper_Agent）
        keywords = [
            # 基础limitation词
            'limitation', 'drawback', 'weakness', 'shortcoming',
            'however', 'unfortunately', 'lack', 'cannot',
            'difficult', 'challenge', 'remains', 'future work',
            'still', 'yet to', 'not yet', 'constrained',
            'trade-off', 'sacrifice',
            # 🆕 review/工具类论文的limitation表述
            'limited to', 'restricted', 'incomplete', 'partial',
            'not cover', 'exclude', 'beyond scope', 'out of scope',
            'fail to', 'unable to', 'not address', 'not handle',
            # 🆕 更多转折和承认不足的表述
            'although', 'nevertheless', 'despite', 'admits',
            'acknowledge', 'recognize', 'confess', 'concede'
        ]

        relevant_chunks = []

        for section_idx in scope.target_sections:
            section = paper.sections[section_idx]
            paragraphs = self._split_into_paragraphs(section.content)

            for para_idx, para in enumerate(paragraphs):
                # 计算关键词匹配度
                keyword_count = sum(
                    1 for kw in keywords
                    if kw.lower() in para.lower()
                )

                if keyword_count > 0:
                    # 🆕 计算综合得分（考虑位置和转折词）
                    score = keyword_count

                    # 🆕 位置权重：首段和尾段权重高
                    if para_idx == 0:  # 首段
                        score += 1.5
                    elif para_idx >= len(paragraphs) - 2:  # 末尾两段
                        score += 2.5  # limitation更常出现在末尾

                    # 🆕 转折词权重（对Limitation很重要）
                    transition_words = [
                        'however', 'unfortunately', 'limitation',
                        'remains', 'still', 'despite', 'although',
                        'nevertheless', 'yet to'
                    ]
                    if any(tw in para.lower() for tw in transition_words):
                        score += 2.0

                    # 🆕 "本文方法"指示词权重（确保提取的是本文的limitation）
                    self_reference_words = [
                        'our method', 'our approach', 'our model',
                        'our system', 'our work', 'we', 'this paper'
                    ]
                    if any(sw in para.lower() for sw in self_reference_words):
                        score += 1.0

                    relevant_chunks.append({
                        'section': section.title,
                        'text': para,
                        'page': section.page_num,
                        'keyword_count': keyword_count,
                        'score': score,
                        'position': para_idx
                    })

        # 按综合得分排序
        relevant_chunks.sort(key=lambda x: x['score'], reverse=True)

        # 🆕 智能降级：如果chunks很少(<3)，扩展到目标章节的所有段落
        max_chunks = 8  # 增加到8个
        if len(relevant_chunks) < 3:
            logger.info(f"     → 关键词匹配chunks较少({len(relevant_chunks)})，扩展到目标章节的所有段落")
            all_chunks = []
            for section_idx in scope.target_sections:
                section = paper.sections[section_idx]
                paragraphs = self._split_into_paragraphs(section.content)
                # 🆕 重点提取每个章节的末尾段落（limitation通常在这里）
                for para_idx, para in enumerate(paragraphs):
                    # 优先选择末尾段落
                    if para_idx >= len(paragraphs) - 3 or para_idx < 2:
                        all_chunks.append({
                            'section': section.title,
                            'text': para,
                            'page': section.page_num,
                            'keyword_count': 0,
                            'score': 0.5 if para_idx >= len(paragraphs) - 3 else 0.3,
                            'position': para_idx
                        })
            # 按位置得分排序
            all_chunks.sort(key=lambda x: x['score'], reverse=True)
            return all_chunks[:max_chunks]

        return relevant_chunks[:max_chunks]

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """分割段落"""
        paragraphs = re.split(r'\n\s*\n|\n', text)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 30]
        return paragraphs

    def _extract_with_llm(
        self,
        chunks: List[Dict],
        paper_title: str,
        feedback: Optional[CriticFeedback] = None
    ) -> tuple:
        """
        使用LLM从chunks中提取局限性

        Args:
            chunks: 相关段落列表
            paper_title: 论文标题
            feedback: Critic反馈(用于增强提示)

        Returns:
            (limitations列表, evidence列表)
        """
        # 构建上下文
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(
                f"[Evidence {i+1}] From section '{chunk['section']}':\n{chunk['text']}"
            )
        context = "\n\n".join(context_parts)

        # 🆕 智能截断：优先保留得分最高的chunks
        if len(context) > self.max_context_length:
            # 按得分重新排序，保留最相关的
            sorted_chunks = sorted(chunks, key=lambda x: x.get('score', 0), reverse=True)
            top_chunks = sorted_chunks[:5]  # 保留前5个
            context_parts = []
            for i, chunk in enumerate(top_chunks):
                context_parts.append(
                    f"[Evidence {i+1}] From section '{chunk['section']}':\n{chunk['text']}"
                )
            context = "\n\n".join(context_parts)
            logger.info(f"     → 上下文过长,保留前{len(top_chunks)}个最相关段落")
            chunks = top_chunks  # 更新chunks用于后续evidence构建

        # 构建提示词
        prompt = f"""论文标题: {paper_title}

任务: 提取本文方法的局限性

⚠️ 关键区分:
- 如果说"LSTM无法处理长序列" -> 这是baseline的问题,不要提取
- 如果说"Our method still struggles with..." -> 这是本文的局限性,必须提取

相关段落:
{context}

输出要求:
1. 列举2-4个具体的局限性
2. 每个局限性用1-2句话说明
3. 使用bullet points格式 (以 "- " 开头)
4. 只提取本文方法的局限性
5. ⚠️ 重要: 即使信息不完整,也要尽量从段落中提取相关内容
6. 只有在段落完全不相关时才输出"未找到相关信息"
"""

        # 🆕 如果有Critic反馈,添加到提示中
        if feedback and feedback.retry_prompt:
            prompt += f"\n⚠️ Critic反馈:\n{feedback.retry_prompt}\n"

        prompt += "\n输出:"

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=0.3,
                max_tokens=1000
            )

            # 记录原始响应,便于调试
            logger.debug(f"     → LLM原始响应: {response[:200]}...")

            # 解析响应
            limitations = self._parse_llm_response(response)

            # 🆕 检测LLM是否过度谨慎返回"未找到"
            if self._is_llm_being_too_cautious(limitations, chunks):
                logger.warning(f"     ⚠️ LLM返回为空或过短,但有{len(chunks)}条证据 - 使用降级策略")
                limitations = self._fallback_extraction(chunks)

            # 构建evidence
            evidence = [
                {
                    'section': chunk['section'],
                    'text': chunk['text'],
                    'page': chunk['page']
                }
                for chunk in chunks
            ]

            return limitations, evidence

        except Exception as e:
            logger.error(f"     ❌ LLM提取失败: {e}")
            # 降级：使用规则提取
            fallback_limitations = self._fallback_extraction(chunks)
            evidence = [{'section': chunks[0]['section'], 'text': chunks[0]['text'], 'page': chunks[0]['page']}] if chunks else []
            return fallback_limitations, evidence

    def _parse_llm_response(self, response: str) -> List[str]:
        """
        解析LLM响应，提取局限性列表

        返回: limitations列表
        """
        limitations = []
        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            # 匹配bullet points
            if line.startswith('- ') or line.startswith('• ') or line.startswith('* '):
                limitation = line[2:].strip()
                if limitation and len(limitation) > 10:
                    limitations.append(limitation)
            # 匹配数字列表
            elif re.match(r'^\d+[\.\)]\s+', line):
                limitation = re.sub(r'^\d+[\.\)]\s+', '', line).strip()
                if limitation and len(limitation) > 10:
                    limitations.append(limitation)

        return limitations

    def _fallback_extraction(self, chunks: List[Dict]) -> List[str]:
        """
        降级提取策略

        🆕 改进版本：
        1. 提取包含关键词的完整句子
        2. 过滤过长/过短的句子
        3. 限制返回数量
        """
        if not chunks:
            return []

        limitations = []
        # 限定关键词：确保提取的是limitation相关内容
        key_indicators = [
            'limitation', 'however', 'cannot', 'still',
            'unfortunately', 'lack', 'difficult', 'challenge',
            'remains', 'constrained', 'trade-off'
        ]

        for chunk in chunks[:4]:  # 扩展到前4个chunks
            text = chunk['text'].strip()
            # 简单提取包含关键词的句子
            sentences = text.split('. ')
            for sentence in sentences:
                if any(kw in sentence.lower() for kw in key_indicators):
                    # 过滤长度
                    if 20 < len(sentence) < 300:
                        # 清理句子
                        cleaned = sentence.strip()
                        if cleaned and cleaned not in limitations:
                            limitations.append(cleaned)

        return limitations[:4]  # 最多返回4条

    def _is_llm_being_too_cautious(self, limitations: List[str], chunks: List[Dict]) -> bool:
        """
        🆕 检测LLM是否过度谨慎返回"未找到"

        判断逻辑（参考DeepPaper_Agent）:
        - 如果limitations为空或只有"未找到"类的回答
        - 但chunks数量 >= 3 (说明有相关证据)
        - 则认为LLM过度谨慎

        Args:
            limitations: LLM提取的limitations列表
            chunks: 相关段落列表

        Returns:
            bool: True表示LLM过度谨慎
        """
        # 检查是否为空
        if not limitations or len(limitations) == 0:
            return len(chunks) >= 3

        # 检查是否所有条目都是"未找到"类的回答
        empty_indicators = [
            "未找到相关信息",
            "未找到",
            "没有找到",
            "无相关内容",
            "not found",
            "no relevant",
            "no information",
            "no limitation",
            "no clear limitation"
        ]

        all_empty = all(
            any(indicator in lim.lower() for indicator in empty_indicators)
            for lim in limitations
        )

        if all_empty and len(chunks) >= 3:
            return True

        # 检查是否所有条目都太短（<20字符）
        all_too_short = all(len(lim.strip()) < 20 for lim in limitations)
        if all_too_short and len(chunks) >= 2:
            return True

        return False

    def _extract_from_citations(self, paper_id: str) -> List[str]:
        """
        从引用分析中提取局限性

        Args:
            paper_id: 论文ID（ArXiv ID或DOI）

        Returns:
            limitations列表
        """
        if not self.citation_detective:
            return []

        try:
            # 使用引用侦探
            result = self.citation_detective.analyze(
                paper_id=paper_id,
                top_k=10,
                min_citation_count=0,
                use_llm_analysis=False  # 使用规则提取即可
            )

            return result.extracted_limitations

        except Exception as e:
            logger.warning(f"  ⚠️ 引用分析失败: {e}")
            return []

    def _merge_results(
        self,
        section_limitations: List[str],
        section_evidence: List[Dict],
        citation_limitations: List[str]
    ) -> tuple:
        """
        合并章节提取和引用提取的结果

        Returns:
            (merged_content, merged_evidence)
        """
        # 构建最终内容
        content_parts = []

        # 第一部分：章节提取
        if section_limitations:
            for limitation in section_limitations:
                content_parts.append(f"- {limitation}")

        # 第二部分：引用提取
        if citation_limitations:
            if section_limitations:  # 如果前面有内容，添加空行分隔
                content_parts.append("")
            for limitation in citation_limitations[:3]:  # 最多保留3条
                content_parts.append(f"- {limitation}")

        # 如果都为空
        if not content_parts:
            return "未找到明确的局限性描述", []

        merged_content = "\n".join(content_parts)

        # 合并evidence（只保留章节evidence）
        merged_evidence = section_evidence

        return merged_content, merged_evidence


def main():
    """测试代码"""
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="Limitation Extractor - 局限性提取")
    parser.add_argument("--config", required=True, help="LLM配置文件路径")
    parser.add_argument("--paper", required=True, help="论文文本文件路径（JSON格式）")
    parser.add_argument("--paper-id", help="论文ID（用于引用分析，可选）")
    parser.add_argument("--use-citation", action="store_true", help="是否使用引用分析")
    parser.add_argument("--output", default="limitation_results.json", help="输出文件路径")

    args = parser.parse_args()

    try:
        # 初始化LLM客户端
        config = LLMConfig.from_file(args.config)
        llm_client = LLMClient(config)

        # 读取论文文档
        with open(args.paper, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)

        # 转换为PaperDocument对象
        from data_structures import PaperSection
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

        # 创建提取器
        extractor = LimitationExtractor(
            llm_client=llm_client,
            use_citation_analysis=args.use_citation
        )

        # 执行提取
        result = extractor.extract(paper, paper_id=args.paper_id)

        # 保存结果
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        # 打印结果
        print("\n" + "=" * 80)
        print("Limitation Extraction Results")
        print("=" * 80)
        print(f"\nContent:\n{result.content}")
        print(f"\nEvidence Count: {len(result.evidence)}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"\n✅ 结果已保存到: {args.output}")

    except Exception as e:
        logger.error(f"执行失败: {e}")
        raise


if __name__ == "__main__":
    main()
