"""
Citation Detective Agent (引用侦探)
负责外部验证，找出同行对论文的真实评价

职责：获取引用该论文的其他论文，分析引用上下文，提取真实的limitations
输入：Paper ID (DOI, ArXiv ID, Semantic Scholar ID等)
输出：来自后续研究的真实Limitation列表

主要API：Semantic Scholar Academic Graph API
- 直接提供引用上下文（citation contexts）
- 覆盖率高（2亿+论文，75%+包含上下文）
- 数据质量高（真实的引用文本，非摘要近似）
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

# 导入LLM配置
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.llm_config import LLMClient, LLMConfig

# 第三方库
try:
    import requests
    from xml.etree import ElementTree as ET
except ImportError:
    raise ImportError("请安装依赖: pip install requests")

logger = logging.getLogger(__name__)


@dataclass
class CitationContext:
    """引用上下文数据结构"""
    citing_paper_id: str  # 引用论文的ID
    citing_paper_title: str  # 引用论文的标题
    citing_paper_authors: List[str]  # 引用论文的作者
    citing_paper_year: Optional[int]  # 引用论文的年份
    citation_count: int  # 引用论文的引用数（影响力指标）
    context_text: str  # 引用上下文的文本
    has_critical_keyword: bool  # 是否包含批判性关键词
    critical_keywords: List[str]  # 匹配到的批判性关键词
    extracted_limitation: Optional[str] = None  # 提取的limitation描述
    confidence: float = 0.0  # 置信度 (0-1)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class CitationAnalysisResult:
    """引用分析结果"""
    target_paper_id: str  # 目标论文ID
    total_citations: int  # 总引用数
    analyzed_citations: int  # 分析的引用数
    critical_citations: int  # 包含批判性评价的引用数
    citation_contexts: List[CitationContext]  # 引用上下文列表
    extracted_limitations: List[str]  # 提取的limitation列表

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "target_paper_id": self.target_paper_id,
            "total_citations": self.total_citations,
            "analyzed_citations": self.analyzed_citations,
            "critical_citations": self.critical_citations,
            "citation_contexts": [ctx.to_dict() for ctx in self.citation_contexts],
            "extracted_limitations": self.extracted_limitations
        }


class CitationDetectiveAgent:
    """
    Citation Detective Agent
    负责获取引用信息，分析引用上下文，提取真实的peer评价
    """

    # 批判性关键词（转折词、负面评价）
    CRITICAL_KEYWORDS = [
        "however", "although", "but", "nevertheless", "nonetheless",
        "limitation", "limitations", "limited", "limits",
        "fails to", "fail to", "failed", "failure",
        "weakness", "weaknesses", "drawback", "drawbacks",
        "problem", "problems", "issue", "issues",
        "challenge", "challenges", "difficulty", "difficulties",
        "cannot", "can not", "unable to",
        "does not", "doesn't", "do not", "don't",
        "insufficient", "inadequate", "lack", "lacks", "lacking",
        "poor", "worse", "inferior",
        "unfortunately", "regrettably",
        "contradict", "contradicts", "inconsistent"
    ]

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        初始化引用侦探

        Args:
            llm_client: LLM客户端（可选，用于深度语义分析）
        """
        self.llm_client = llm_client
        self.system_prompt = self._build_system_prompt() if llm_client else None

    def _build_system_prompt(self) -> str:
        """构建系统提示词（用于LLM深度分析）"""
        return """You are a professional expert in scientific literature analysis.

Your Task:
Extract genuine, specific limitations from citation contexts.

Input: A citation context (a text segment where a citing paper references the target paper).

Output Requirements:

Determine if the citation contains critical evaluation or points out limitations of the target paper.
If yes, extract the specific limitation description (avoid generalizations).
Summarize this limitation using clear, objective language.
Output Format (JSON):

<JSON>
{
    "has_limitation": true/false,
    "limitation": "Description of the specific limitation",
    "confidence": 0.9,
    "reasoning": "Basis for judgment"
}
Notes:

Extract only actual criticisms and limitations; do not extract positive evaluations.
Descriptions must be specific; avoid vague phrases like "has certain limitations".
If the context is a neutral citation (no criticism), return has_limitation=false.

"""

    def analyze(
        self,
        paper_id: str,
        top_k: int = 10,
        min_citation_count: int = 0,
        use_llm_analysis: bool = False
    ) -> CitationAnalysisResult:
        """
        分析论文的引用情况，提取真实的limitations

        Args:
            paper_id: 论文ID（ArXiv ID, DOI等）
            top_k: 分析Top K篇引用论文
            min_citation_count: 最小引用数过滤（影响力过滤）
            use_llm_analysis: 是否使用LLM进行深度语义分析

        Returns:
            引用分析结果
        """
        logger.info(f"开始引用侦探分析: {paper_id}")

        # Step 1: 获取引用该论文的列表
        citing_papers = self._fetch_citing_papers(paper_id)
        logger.info(f"✅ 找到 {len(citing_papers)} 篇引用论文")

        if not citing_papers:
            logger.warning("未找到引用信息")
            return CitationAnalysisResult(
                target_paper_id=paper_id,
                total_citations=0,
                analyzed_citations=0,
                critical_citations=0,
                citation_contexts=[],
                extracted_limitations=[]
            )

        # Step 2: 按影响力排序，取Top K
        citing_papers = self._rank_by_impact(citing_papers, top_k, min_citation_count)
        logger.info(f"✅ 筛选出 Top {len(citing_papers)} 篇高影响力论文")

        # Step 3: 获取引用上下文
        citation_contexts = self._extract_citation_contexts(
            paper_id, citing_papers
        )
        logger.info(f"✅ 提取到 {len(citation_contexts)} 条引用上下文")

        # Step 4: 过滤批判性上下文
        critical_contexts = self._filter_critical_contexts(citation_contexts)
        logger.info(f"✅ 筛选出 {len(critical_contexts)} 条批判性引用")

        # Step 5: 提取limitations
        if use_llm_analysis and self.llm_client:
            limitations = self._extract_limitations_with_llm(critical_contexts)
        else:
            limitations = self._extract_limitations_rule_based(critical_contexts)

        logger.info(f"✅ 提取到 {len(limitations)} 条真实limitations")

        # 构建结果
        result = CitationAnalysisResult(
            target_paper_id=paper_id,
            total_citations=len(citing_papers),
            analyzed_citations=len(citation_contexts),
            critical_citations=len(critical_contexts),
            citation_contexts=critical_contexts,
            extracted_limitations=limitations
        )

        return result

    def _fetch_citing_papers(self, paper_id: str) -> List[Dict[str, Any]]:
        """
        获取引用该论文的列表（使用 Semantic Scholar API）

        Args:
            paper_id: 论文ID (支持 DOI, ArXiv ID, Semantic Scholar ID等)

        Returns:
            引用论文列表（包含上下文信息）
        """
        return self._fetch_from_semantic_scholar(paper_id)

    def _fetch_from_semantic_scholar(self, paper_id: str) -> List[Dict[str, Any]]:
        """
        从 Semantic Scholar API 获取引用信息（包含引用上下文）

        Args:
            paper_id: 论文ID (支持 DOI, ArXiv ID, Semantic Scholar ID等)

        Returns:
            引用论文列表（包含contexts字段）
        """
        logger.info(f"从 Semantic Scholar 获取引用信息: {paper_id}")

        # 标准化 paper_id 格式
        if not paper_id.startswith(('DOI:', 'arXiv:', 'CorpusId:', 'PMID:')):
            # 自动识别格式
            if paper_id.startswith('10.'):
                paper_id = f'DOI:{paper_id}'
            elif paper_id.replace('.', '').replace('v', '').isdigit():
                paper_id = f'arXiv:{paper_id}'
            elif paper_id.isdigit():
                paper_id = f'CorpusId:{paper_id}'
            # 否则保持原样，让 API 自动处理

        try:
            # Semantic Scholar API - 获取引用列表
            base_url = "https://api.semanticscholar.org/graph/v1"
            url = f"{base_url}/paper/{paper_id}/citations"

            params = {
                'fields': 'contexts,intents,isInfluential,'
                         'citingPaper.paperId,citingPaper.title,'
                         'citingPaper.year,citingPaper.citationCount,'
                         'citingPaper.authors',
                'limit': 1000  # 获取尽可能多的引用
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            citations = data.get('data', [])

            citing_papers = []
            for cite in citations:
                citing_paper = cite.get('citingPaper', {})

                if not citing_paper:
                    continue

                # 构建统一的数据格式
                paper_info = {
                    "id": citing_paper.get('paperId', ''),
                    "title": citing_paper.get('title', ''),
                    "authors": [
                        author.get('name', '')
                        for author in citing_paper.get('authors', [])
                    ],
                    "year": citing_paper.get('year'),
                    "citation_count": citing_paper.get('citationCount', 0),
                    # 关键：引用上下文（Semantic Scholar 直接提供）
                    "contexts": cite.get('contexts', []),
                    "intents": cite.get('intents', []),
                    "is_influential": cite.get('isInfluential', False)
                }

                citing_papers.append(paper_info)

            logger.info(f"✅ 从 Semantic Scholar 获取到 {len(citing_papers)} 篇引用论文")
            logger.info(f"✅ 其中 {sum(1 for p in citing_papers if p.get('contexts'))} 篇包含引用上下文")

            return citing_papers

        except requests.exceptions.RequestException as e:
            logger.error(f"Semantic Scholar API 调用失败: {e}")
            logger.info("💡 提示：如果是 404 错误，请检查 paper_id 格式是否正确")
            return []
        except Exception as e:
            logger.error(f"处理 Semantic Scholar 响应失败: {e}")
            return []

    def _rank_by_impact(
        self,
        papers: List[Dict[str, Any]],
        top_k: int,
        min_citation_count: int
    ) -> List[Dict[str, Any]]:
        """
        按影响力排序并过滤

        Args:
            papers: 论文列表
            top_k: 取Top K
            min_citation_count: 最小引用数

        Returns:
            排序后的论文列表
        """
        # 过滤低引用论文
        filtered = [p for p in papers if p.get("citation_count", 0) >= min_citation_count]

        # 按引用数排序
        sorted_papers = sorted(
            filtered,
            key=lambda x: x.get("citation_count", 0),
            reverse=True
        )

        return sorted_papers[:top_k]

    def _extract_citation_contexts(
        self,
        target_paper_id: str,
        citing_papers: List[Dict[str, Any]]
    ) -> List[CitationContext]:
        """
        提取引用上下文（使用 Semantic Scholar 提供的 contexts）

        Args:
            target_paper_id: 目标论文ID（保留参数以兼容现有接口）
            citing_papers: 引用论文列表（已包含 contexts 字段）

        Returns:
            引用上下文列表
        """
        contexts = []

        for paper in citing_papers:
            # Semantic Scholar 直接提供引用上下文
            raw_contexts = paper.get('contexts', [])

            # 如果没有上下文，跳过该引用
            if not raw_contexts:
                logger.debug(f"跳过无上下文的引用: {paper.get('title', 'Unknown')[:50]}...")
                continue

            # 合并所有上下文
            context_text = " ".join(raw_contexts)

            # 检查是否包含批判性关键词
            has_critical, keywords = self._check_critical_keywords(context_text)

            context = CitationContext(
                citing_paper_id=paper.get("id", ""),
                citing_paper_title=paper.get("title", ""),
                citing_paper_authors=paper.get("authors", []),
                citing_paper_year=paper.get("year"),
                citation_count=paper.get("citation_count", 0),
                context_text=context_text,
                has_critical_keyword=has_critical,
                critical_keywords=keywords
            )

            contexts.append(context)

        logger.info(f"✅ 成功提取 {len(contexts)} 条引用上下文")
        logger.info(f"✅ 其中 {sum(1 for c in contexts if c.has_critical_keyword)} 条包含批判性关键词")

        return contexts

    def _check_critical_keywords(self, text: str) -> Tuple[bool, List[str]]:
        """
        检查文本中是否包含批判性关键词

        Args:
            text: 文本

        Returns:
            (是否包含, 匹配的关键词列表)
        """
        text_lower = text.lower()
        found_keywords = []

        for keyword in self.CRITICAL_KEYWORDS:
            if keyword in text_lower:
                found_keywords.append(keyword)

        return len(found_keywords) > 0, found_keywords

    def _filter_critical_contexts(
        self,
        contexts: List[CitationContext]
    ) -> List[CitationContext]:
        """
        过滤出包含批判性关键词的上下文

        Args:
            contexts: 引用上下文列表

        Returns:
            批判性上下文列表
        """
        return [ctx for ctx in contexts if ctx.has_critical_keyword]

    def _extract_limitations_rule_based(
        self,
        contexts: List[CitationContext]
    ) -> List[str]:
        """
        基于规则提取limitations

        Args:
            contexts: 批判性上下文列表

        Returns:
            limitation列表
        """
        limitations = []

        for ctx in contexts:
            # 简单规则：提取包含批判性关键词的句子
            sentences = self._split_sentences(ctx.context_text)

            for sentence in sentences:
                # 检查句子是否包含批判性关键词
                has_critical, keywords = self._check_critical_keywords(sentence)

                if has_critical:
                    # 构造limitation描述
                    limitation = (
                        f"[{ctx.citing_paper_title[:50]}... "
                        f"({ctx.citing_paper_year})] "
                        f"{sentence.strip()}"
                    )
                    limitations.append(limitation)
                    ctx.extracted_limitation = sentence.strip()

        return limitations

    def _extract_limitations_with_llm(
        self,
        contexts: List[CitationContext]
    ) -> List[str]:
        """
        使用LLM深度提取limitations

        Args:
            contexts: 批判性上下文列表

        Returns:
            limitation列表
        """
        limitations = []

        for ctx in contexts:
            try:
                # 构建提示词
                prompt = f"""【引用论文】
标题: {ctx.citing_paper_title}
作者: {', '.join(ctx.citing_paper_authors)}
年份: {ctx.citing_paper_year}

【引用上下文】
{ctx.context_text}

请分析以上引用上下文，判断是否包含对目标论文的批判性评价或局限性指出。
如果包含，请提取具体的limitation描述。
"""

                # 调用LLM
                response = self.llm_client.generate(
                    prompt=prompt,
                    system_prompt=self.system_prompt,
                    temperature=0.2,
                    max_tokens=500
                )

                # 解析响应
                result = self._parse_llm_response(response)

                if result.get("has_limitation"):
                    limitation = (
                        f"[{ctx.citing_paper_title[:50]}... "
                        f"({ctx.citing_paper_year})] "
                        f"{result.get('limitation', '')}"
                    )
                    limitations.append(limitation)
                    ctx.extracted_limitation = result.get("limitation")
                    ctx.confidence = result.get("confidence", 0.0)

            except Exception as e:
                logger.warning(f"LLM分析失败: {e}")
                continue

        return limitations

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应"""
        try:
            # 提取JSON
            json_str = self._extract_json(response)
            return json.loads(json_str)
        except Exception as e:
            logger.warning(f"LLM响应解析失败: {e}")
            return {"has_limitation": False}

    def _extract_json(self, text: str) -> str:
        """从文本中提取JSON"""
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()

        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return text[start:end]

        return text.strip()

    def _split_sentences(self, text: str) -> List[str]:
        """简单的句子分割"""
        # 按句号、问号、感叹号分割
        sentences = re.split(r'[.!?]\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def export_results(
        self,
        result: CitationAnalysisResult,
        output_path: str,
        format: str = "json"
    ):
        """
        导出分析结果

        Args:
            result: 分析结果
            output_path: 输出文件路径
            format: 输出格式 (json, txt, md)
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            if format == "json":
                self._export_json(result, output_file)
            elif format == "txt":
                self._export_txt(result, output_file)
            elif format == "md":
                self._export_markdown(result, output_file)
            else:
                raise ValueError(f"不支持的导出格式: {format}")

            logger.info(f"✅ 结果已导出到: {output_path}")

        except Exception as e:
            logger.error(f"导出失败: {e}")

    def _export_json(self, result: CitationAnalysisResult, output_file: Path):
        """导出为JSON格式"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

    def _export_txt(self, result: CitationAnalysisResult, output_file: Path):
        """导出为文本格式"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Citation Detective Report: Peer Review & Limitations\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Target Paper ID: {result.target_paper_id}\n")
            f.write(f"Total Citations: {result.total_citations}\n")
            f.write(f"Analyzed Citations: {result.analyzed_citations}\n")
            f.write(f"Critical Citations: {result.critical_citations}\n\n")

            f.write("-" * 80 + "\n")
            f.write("Extracted Limitations:\n")
            f.write("-" * 80 + "\n\n")

            for i, limitation in enumerate(result.extracted_limitations, 1):
                f.write(f"{i}. {limitation}\n\n")

            if result.citation_contexts:
                f.write("-" * 80 + "\n")
                f.write("Critical Citation Contexts:\n")
                f.write("-" * 80 + "\n\n")

                for i, ctx in enumerate(result.citation_contexts, 1):
                    f.write(f"【Context {i}】\n")
                    f.write(f"Citing Paper: {ctx.citing_paper_title}\n")
                    f.write(f"Authors: {', '.join(ctx.citing_paper_authors)}\n")
                    f.write(f"Year: {ctx.citing_paper_year}\n")
                    f.write(f"Citation Count: {ctx.citation_count}\n")
                    f.write(f"Keywords: {', '.join(ctx.critical_keywords)}\n\n")
                    f.write(f"Context:\n{ctx.context_text}\n\n")
                    f.write("-" * 80 + "\n\n")

    def _export_markdown(self, result: CitationAnalysisResult, output_file: Path):
        """导出为Markdown格式"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Citation Detective Report: Peer Review & Limitations\n\n")

            f.write("## Summary\n\n")
            f.write(f"- **Target Paper ID**: {result.target_paper_id}\n")
            f.write(f"- **Total Citations**: {result.total_citations}\n")
            f.write(f"- **Analyzed Citations**: {result.analyzed_citations}\n")
            f.write(f"- **Critical Citations**: {result.critical_citations}\n\n")

            f.write("---\n\n")
            f.write("## Extracted Limitations\n\n")

            for i, limitation in enumerate(result.extracted_limitations, 1):
                f.write(f"{i}. {limitation}\n\n")

            if result.citation_contexts:
                f.write("---\n\n")
                f.write("## Critical Citation Contexts\n\n")

                for i, ctx in enumerate(result.citation_contexts, 1):
                    f.write(f"### Context {i}\n\n")
                    f.write(f"**Citing Paper**: {ctx.citing_paper_title}\n\n")
                    f.write(f"**Authors**: {', '.join(ctx.citing_paper_authors)}\n\n")
                    f.write(f"**Year**: {ctx.citing_paper_year}\n\n")
                    f.write(f"**Citation Count**: {ctx.citation_count}\n\n")
                    f.write(f"**Critical Keywords**: {', '.join(ctx.critical_keywords)}\n\n")
                    f.write(f"**Context**:\n\n")
                    f.write(f"> {ctx.context_text}\n\n")
                    f.write("---\n\n")


def main():
    """测试代码"""
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="Citation Detective Agent - 引用侦探")
    parser.add_argument("--paper-id", required=True, help="论文ID (ArXiv ID或DOI)")
    parser.add_argument("--top-k", type=int, default=10, help="分析Top K篇引用论文")
    parser.add_argument("--min-citations", type=int, default=0, help="最小引用数过滤")
    parser.add_argument("--use-llm", action="store_true", help="使用LLM深度分析")
    parser.add_argument("--config", help="LLM配置文件路径（使用LLM时必需）")
    parser.add_argument("--output", default="citation_analysis_results.json", help="输出文件路径")
    parser.add_argument("--format", choices=["json", "txt", "md"], default="json", help="输出格式")

    args = parser.parse_args()

    try:
        # 初始化agent
        llm_client = None
        if args.use_llm:
            if not args.config:
                raise ValueError("使用LLM分析时必须提供配置文件 (--config)")
            config = LLMConfig.from_file(args.config)
            llm_client = LLMClient(config)

        agent = CitationDetectiveAgent(llm_client)

        # 执行分析
        result = agent.analyze(
            paper_id=args.paper_id,
            top_k=args.top_k,
            min_citation_count=args.min_citations,
            use_llm_analysis=args.use_llm
        )

        # 导出结果
        agent.export_results(result, args.output, format=args.format)

        # 打印摘要
        print("\n" + "=" * 80)
        print(f"Citation Analysis Complete")
        print("=" * 80)
        print(f"Total Citations: {result.total_citations}")
        print(f"Analyzed: {result.analyzed_citations}")
        print(f"Critical: {result.critical_citations}")
        print(f"Extracted Limitations: {len(result.extracted_limitations)}")
        print("\n" + "-" * 80)
        print("Top Limitations:")
        print("-" * 80)

        for i, limitation in enumerate(result.extracted_limitations[:5], 1):
            print(f"\n{i}. {limitation}")

    except Exception as e:
        logger.error(f"执行失败: {e}")
        raise


if __name__ == "__main__":
    main()
