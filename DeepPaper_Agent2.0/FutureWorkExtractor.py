"""
Future Work Extractor (未来工作提取器)
使用章节定位方法提取论文的未来工作方向

工作流程:
1. 使用 SectionLocatorAgent 定位 future work 章节
2. 从定位的章节中提取 future work
3. 🆕 使用 CriticAgent 审查并自动重试提取
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
from critic_agent import CriticAgent

# 导入LLM配置
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.llm_config import LLMClient, LLMConfig

logger = logging.getLogger(__name__)


class FutureWorkExtractor:
    """
    Future Work 提取器
    使用章节定位方法提取未来工作方向
    """

    def __init__(
        self,
        llm_client: LLMClient,
        use_critic: bool = True,
        max_context_length: int = 3000,
        max_iterations: int = 3
    ):
        """
        初始化 Future Work 提取器

        Args:
            llm_client: LLM客户端
            use_critic: 是否使用CriticAgent进行质量检查和重试
            max_context_length: 最大上下文长度
            max_iterations: 最大重试次数(当use_critic=True时)
        """
        self.llm_client = llm_client
        self.use_critic = use_critic
        self.max_context_length = max_context_length

        # 初始化子组件
        self.locator = SectionLocatorAgent(llm_client)

        # 🆕 初始化 CriticAgent
        if use_critic:
            self.critic = CriticAgent(llm_client, max_iterations=max_iterations)
        else:
            self.critic = None

        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """You are a professional expert in scientific paper analysis.

Your Task: Extract the future work directions proposed in this paper.

Identification Techniques:

Look for keywords: future, next, further, explore, plan, will, could, would.
Focus on explicit mentions of "future work".
Look for improvement directions inferred from acknowledged limitations.
Pay attention to the outlook at the end of the Conclusion or Discussion sections.
Output Requirements:

List 2-4 specific future work directions.
Use 1-2 sentences to explain each direction.
Use bullet points format.
Output the content directly, avoiding meta-talk (e.g., do not say "According to the paragraph...").

"""

    def extract(self, paper: PaperDocument, feedback: Optional[CriticFeedback] = None) -> ExtractionResult:
        """
        提取论文的未来工作方向

        Args:
            paper: 论文文档
            feedback: CriticAgent反馈（用于重试，可选）

        Returns:
            ExtractionResult: 提取结果
        """
        logger.info("📋 开始提取 Future Work...")

        # 步骤1: 定位章节 (如果有反馈,使用建议的章节)
        if feedback and feedback.suggested_sections:
            logger.info(f"  → 使用Critic建议的章节: {feedback.suggested_sections}")
            scope = SectionScope(
                field=FieldType.FUTURE_WORK,
                target_sections=feedback.suggested_sections,
                section_titles=[paper.sections[i].title for i in feedback.suggested_sections if i < len(paper.sections)],
                reasoning="Based on Critic feedback"
            )
        else:
            logger.info("  🔍 定位相关章节...")
            scope = self.locator.locate(paper, FieldType.FUTURE_WORK)

        if not scope.target_sections:
            logger.warning("  ⚠️ 未找到相关章节")
            initial_result = ExtractionResult(
                field=FieldType.FUTURE_WORK,
                content="未找到明确的未来工作描述",
                evidence=[],
                extraction_method="section_locator",
                confidence=0.0,
                iterations=1
            )

            # 如果启用critic且是首次提取，仍然尝试自动重试
            if self.use_critic and self.critic and not feedback:
                return self._apply_critic(initial_result, paper, scope)
            return initial_result

        # 步骤2: 提取相关段落
        logger.info("  📖 提取相关段落...")
        relevant_chunks = self._extract_relevant_chunks(paper, scope)

        if not relevant_chunks:
            logger.warning("  ⚠️ 未找到相关段落")
            initial_result = ExtractionResult(
                field=FieldType.FUTURE_WORK,
                content="未找到明确的未来工作描述",
                evidence=[],
                extraction_method="section_locator",
                confidence=0.0,
                iterations=1
            )

            if self.use_critic and self.critic and not feedback:
                return self._apply_critic(initial_result, paper, scope)
            return initial_result

        # 步骤3: 使用LLM提取
        logger.info("  🤖 使用LLM提取...")
        content, evidence = self._extract_with_llm(relevant_chunks, paper.title, feedback)

        logger.info(f"  ✅ Future Work初步提取完成")
        logger.info(f"     → 最终内容: {content[:100]}...")

        initial_result = ExtractionResult(
            field=FieldType.FUTURE_WORK,
            content=content,
            evidence=evidence,
            extraction_method="section_locator",
            confidence=0.8 if content and "未找到" not in content else 0.3,
            iterations=1
        )

        # 🆕 使用 CriticAgent 进行质量检查和自动重试
        if self.use_critic and self.critic and not feedback:
            return self._apply_critic(initial_result, paper, scope)

        return initial_result

    def _apply_critic(self, initial_result: ExtractionResult, paper: PaperDocument, scope: SectionScope) -> ExtractionResult:
        """应用CriticAgent进行质量检查和重试"""
        logger.info("\n  🔍 启动 CriticAgent 质量检查...")

        # 定义重新提取函数
        def retry_extract(paper_doc, critic_feedback):
            return self.extract(paper_doc, critic_feedback)

        # 调用 critique_and_retry
        final_result = self.critic.critique_and_retry(
            extraction=initial_result,
            paper=paper,
            extractor_func=retry_extract,
            scope=scope,
            evaluation_level="both"
        )

        return final_result

    def _extract_relevant_chunks(
        self,
        paper: PaperDocument,
        scope: SectionScope
    ) -> List[Dict]:
        """
        提取相关段落（使用关键词匹配）

        🆕 优化策略:
        1. 扩展关键词列表（包括更多future work的表述）
        2. 考虑段落位置（首段/尾段权重更高）
        3. 结合关键词权重
        4. 智能降级：当匹配少时自动扩展到所有段落
        """
        # 🆕 扩展关键词列表
        keywords = [
            # 基础future work词
            'future', 'next', 'further', 'improve', 'extend',
            'explore', 'investigate', 'plan', 'ongoing',
            'will', 'could', 'would', 'intend',
            'remain to be', 'open question', 'direction',
            'outlook', 'prospect',
            # 🆕 更多future work的表述
            'future work', 'future direction', 'future research',
            'future study', 'future effort', 'next step',
            'in the future', 'going forward', 'moving forward',
            'to be explored', 'to be investigated', 'to be addressed',
            'worth exploring', 'worth investigating',
            'potential direction', 'potential improvement',
            'promising direction', 'open problem'
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
                    # 🆕 计算综合得分（考虑位置和关键词权重）
                    score = keyword_count

                    # 🆕 位置权重：首段和尾段权重高
                    if para_idx == 0:  # 首段
                        score += 1.5
                    elif para_idx >= len(paragraphs) - 2:  # 末尾两段
                        score += 2.5  # future work更常出现在末尾

                    # 🆕 关键"future"词权重
                    future_indicators = ['future work', 'future direction', 'future research']
                    if any(fi in para.lower() for fi in future_indicators):
                        score += 2.0
                    elif 'future' in para.lower():
                        score += 1.5

                    # 🆕 "展望"类词权重
                    outlook_words = ['outlook', 'prospect', 'promising', 'potential']
                    if any(ow in para.lower() for ow in outlook_words):
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
                # 🆕 重点提取每个章节的末尾段落（future work通常在这里）
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
        使用LLM从chunks中提取未来工��

        Args:
            chunks: 相关段落列表
            paper_title: 论文标题
            feedback: Critic反馈(用于增强提示)

        Returns:
            (content, evidence)
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

任务: 提取论文中提出的未来工作方向

相关段落:
{context}

输出要求:
1. 列举2-4个具体的未来工作方向
2. 每个方向用1-2句话说明
3. 使用bullet points格式 (以 "- " 开头)
4. 关注作者明确提到的future work
5. 也可以从limitation推断出改进方向
6. ⚠️ 重要: 即使信息不完整,也要尽量从段落中提取相关内容
7. 只有在段落完全不相关时才输出"未找到明确的未来工作描述"
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
            future_works = self._parse_llm_response(response)

            # 🆕 检测LLM是否过度谨慎返回"未找到"
            if self._is_llm_being_too_cautious(future_works, chunks):
                logger.warning(f"     ⚠️ LLM返回为空或过短,但有{len(chunks)}条证据 - 使用降级策略")
                fallback_content = self._fallback_extraction(chunks)
                content = fallback_content
            elif future_works:
                content = "\n".join([f"- {fw}" for fw in future_works])
            else:
                content = "未找到明确的未来工作描述"

            # 构建evidence
            evidence = [
                {
                    'section': chunk['section'],
                    'text': chunk['text'],
                    'page': chunk['page']
                }
                for chunk in chunks
            ]

            return content, evidence

        except Exception as e:
            logger.error(f"     ❌ LLM提取失败: {e}")
            # 降级：使用规则提取
            fallback_content = self._fallback_extraction(chunks)
            evidence = [{'section': chunks[0]['section'], 'text': chunks[0]['text'], 'page': chunks[0]['page']}] if chunks else []
            return fallback_content, evidence

    def _parse_llm_response(self, response: str) -> List[str]:
        """
        解析LLM响应，提取未来工作列表

        返回: future_works列表
        """
        future_works = []
        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            # 匹配bullet points
            if line.startswith('- ') or line.startswith('• ') or line.startswith('* '):
                future_work = line[2:].strip()
                if future_work and len(future_work) > 10:
                    future_works.append(future_work)
            # 匹配数字列表
            elif re.match(r'^\d+[\.\)]\s+', line):
                future_work = re.sub(r'^\d+[\.\)]\s+', '', line).strip()
                if future_work and len(future_work) > 10:
                    future_works.append(future_work)

        return future_works

    def _fallback_extraction(self, chunks: List[Dict]) -> str:
        """
        降级提取策略

        🆕 改进版本：
        1. 提取包含关键词的完整句子
        2. 过滤过长/过短的句子
        3. 限制返回数量
        """
        if not chunks:
            return "未找到明确的未来工作描述"

        future_works = []
        # 限定关键词：确保提取的是future work相关内容
        key_indicators = [
            'future', 'next', 'explore', 'plan', 'will',
            'could', 'would', 'further', 'improve', 'extend',
            'investigate', 'direction', 'outlook'
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
                        if cleaned and cleaned not in future_works:
                            future_works.append(cleaned)

        if future_works:
            return "\n".join([f"- {fw}" for fw in future_works[:4]])  # 最多返回4条
        else:
            return "未找到明确的未来工作描述"

    def _is_llm_being_too_cautious(self, future_works: List[str], chunks: List[Dict]) -> bool:
        """
        🆕 检测LLM是否过度谨慎返回"未找到"

        判断逻辑（参考DeepPaper_Agent）:
        - 如果future_works为空或只有"未找到"类的回答
        - 但chunks数量 >= 3 (说明有相关证据)
        - 则认为LLM过度谨慎

        Args:
            future_works: LLM提取的future works列表
            chunks: 相关段落列表

        Returns:
            bool: True表示LLM过度谨慎
        """
        # 检查是否为空
        if not future_works or len(future_works) == 0:
            return len(chunks) >= 3

        # 检查是否所有条目都是"未找到"类的回答
        empty_indicators = [
            "未找到明确的未来工作描述",
            "未找到",
            "没有找到",
            "无相关内容",
            "not found",
            "no relevant",
            "no information",
            "no future work",
            "no clear future"
        ]

        all_empty = all(
            any(indicator in fw.lower() for indicator in empty_indicators)
            for fw in future_works
        )

        if all_empty and len(chunks) >= 3:
            return True

        # 检查是否所有条目都太短（<20字符）
        all_too_short = all(len(fw.strip()) < 20 for fw in future_works)
        if all_too_short and len(chunks) >= 2:
            return True

        return False


def main():
    """测试代码"""
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="Future Work Extractor - 未来工作提取")
    parser.add_argument("--config", required=True, help="LLM配置文件路径")
    parser.add_argument("--paper", required=True, help="论文文本文件路径（JSON格式）")
    parser.add_argument("--output", default="future_work_results.json", help="输出文件路径")

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
        extractor = FutureWorkExtractor(llm_client=llm_client)

        # 执行提取
        result = extractor.extract(paper)

        # 保存结果
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        # 打印结果
        print("\n" + "=" * 80)
        print("Future Work Extraction Results")
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
