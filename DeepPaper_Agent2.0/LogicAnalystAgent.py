"""
Logic Analyst Agent (逻辑分析员)
负责内部逻辑，寻找"问题与解法"的映射

职责：分析论文核心逻辑，提取 Problem-Solution (P-S) Pairs
输入：论文全文（或 Abstract + Intro + Method 核心段落）
输出：结构化的 Problem-Solution Pairs
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# 导入LLM配置
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.llm_config import LLMClient, LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class ProblemSolutionPair:
    """问题-解法对数据结构"""
    problem: str  # 核心痛点 (The Lock)
    solution: str  # 核心机制 (The Key)
    explanation: str  # 机制如何解决痛点的解释
    confidence: float = 0.0  # 置信度 (0-1)
    evidence: Optional[str] = None  # 支持证据（论文中的原文引用）

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


class LogicAnalystAgent:
    """
    Logic Analyst Agent
    负责分析论文的核心逻辑，提取问题-解法映射关系
    """

    def __init__(self, llm_client: LLMClient):
        """
        初始化逻辑分析员

        Args:
            llm_client: LLM客户端
        """
        self.llm_client = llm_client
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """You are a professional expert in scientific paper logic analysis.

Your Task:

Read the paper content and identify the core pain point (The Lock/Problem) the authors aim to solve.
Identify the core mechanism/solution (The Key/Solution) designed by the authors.
Explain specifically how this mechanism addresses the pain point.
Output Requirements:

Output in the form of "Problem-Solution Pairs".
Problem descriptions must be precise and specific, avoiding generalizations.
Solution descriptions must focus on the core mechanism, avoiding the listing of irrelevant technical details.
You must clearly explain the causal relationship of "how the solution solves the problem".
If there are multiple pairs, list them separately (focusing on the core 1-3 pairs).
Output Format (JSON):
{
    "pairs": [
        {
            "problem": "Description of the problem",
            "solution": "Description of the solution/mechanism",
            "explanation": "Detailed explanation of how the solution solves the problem",
            "confidence": 0.9,
            "evidence": "Citation from original text (optional)"
        }
    ]
}
Note:

Do not generate problems irrelevant to the core innovation.
Do not simply repeat the abstract; distill the logical relationships.
Focus on "why this design solves this problem".
"""

    def analyze(
        self,
        paper_content: str,
        paper_metadata: Optional[Dict[str, Any]] = None
    ) -> List[ProblemSolutionPair]:
        """
        分析论文，提取问题-解法对

        Args:
            paper_content: 论文内容（全文或关键段落）
            paper_metadata: 论文元数据（可选，如标题、作者等）

        Returns:
            问题-解法对列表
        """
        logger.info("开始逻辑分析：提取 Problem-Solution Pairs")

        # 构建分析提示词
        user_prompt = self._build_analysis_prompt(paper_content, paper_metadata)

        # 调用LLM分析
        try:
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=self.system_prompt,
                temperature=0.3,  # 较低温度保证逻辑一致性
                max_tokens=2000
            )

            # 解析响应
            pairs = self._parse_response(response)

            logger.info(f"✅ 提取到 {len(pairs)} 个 Problem-Solution Pairs")
            return pairs

        except Exception as e:
            logger.error(f"逻辑分析失败: {e}")
            return []

    def _build_analysis_prompt(
        self,
        paper_content: str,
        paper_metadata: Optional[Dict[str, Any]]
    ) -> str:
        """构建分析提示词"""
        prompt_parts = []

        # 添加元数据信息
        if paper_metadata:
            prompt_parts.append("【论文信息】")
            if "title" in paper_metadata:
                prompt_parts.append(f"标题: {paper_metadata['title']}")
            if "authors" in paper_metadata:
                prompt_parts.append(f"作者: {', '.join(paper_metadata['authors'])}")
            if "year" in paper_metadata:
                prompt_parts.append(f"年份: {paper_metadata['year']}")
            prompt_parts.append("")

        # 添加论文内容
        prompt_parts.append("【论文内容】")
        prompt_parts.append(paper_content)
        prompt_parts.append("")

        # 添加分析指令
        prompt_parts.append("【分析任务】")
        prompt_parts.append("请分析以上论文，找出核心的 Problem-Solution Pairs。")
        prompt_parts.append("请以JSON格式输出，包含 pairs 数组，每个元素包含：")
        prompt_parts.append("- problem: 核心痛点描述")
        prompt_parts.append("- solution: 核心机制描述")
        prompt_parts.append("- explanation: 机制如何解决痛点的详细解释")
        prompt_parts.append("- confidence: 置信度 (0-1)")
        prompt_parts.append("- evidence: 论文原文引用（可选）")

        return "\n".join(prompt_parts)

    def _parse_response(self, response: str) -> List[ProblemSolutionPair]:
        """
        解析LLM响应，提取问题-解法对

        Args:
            response: LLM原始响应

        Returns:
            问题-解法对列表
        """
        pairs = []

        try:
            # 尝试提取JSON
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            # 解析pairs数组
            if "pairs" in data and isinstance(data["pairs"], list):
                for pair_data in data["pairs"]:
                    pair = ProblemSolutionPair(
                        problem=pair_data.get("problem", ""),
                        solution=pair_data.get("solution", ""),
                        explanation=pair_data.get("explanation", ""),
                        confidence=float(pair_data.get("confidence", 0.0)),
                        evidence=pair_data.get("evidence")
                    )
                    pairs.append(pair)

            logger.info(f"成功解析 {len(pairs)} 个 Problem-Solution Pairs")

        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}")
            # 尝试从文本中提取
            pairs = self._fallback_parse(response)

        except Exception as e:
            logger.error(f"响应解析失败: {e}")

        return pairs

    def _extract_json(self, text: str) -> str:
        """
        从文本中提取JSON内容

        Args:
            text: 包含JSON的文本

        Returns:
            JSON字符串
        """
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

    def _fallback_parse(self, response: str) -> List[ProblemSolutionPair]:
        """
        后备解析方法：从文本中提取问题-解法对

        Args:
            response: LLM响应文本

        Returns:
            问题-解法对列表
        """
        logger.info("使用后备解析方法提取 Problem-Solution Pairs")

        pairs = []

        # 简单的文本解析逻辑（可以根据实际响应格式优化）
        lines = response.split("\n")
        current_problem = None
        current_solution = None
        current_explanation = None

        for line in lines:
            line = line.strip()

            if line.lower().startswith("problem:") or line.lower().startswith("问题:"):
                current_problem = line.split(":", 1)[1].strip()

            elif line.lower().startswith("solution:") or line.lower().startswith("解法:"):
                current_solution = line.split(":", 1)[1].strip()

            elif line.lower().startswith("explanation:") or line.lower().startswith("解释:"):
                current_explanation = line.split(":", 1)[1].strip()

                # 当收集到完整的三元组时，创建pair
                if current_problem and current_solution and current_explanation:
                    pair = ProblemSolutionPair(
                        problem=current_problem,
                        solution=current_solution,
                        explanation=current_explanation,
                        confidence=0.5  # 后备解析置信度较低
                    )
                    pairs.append(pair)

                    # 重置
                    current_problem = None
                    current_solution = None
                    current_explanation = None

        return pairs

    def export_results(
        self,
        pairs: List[ProblemSolutionPair],
        output_path: str,
        format: str = "json"
    ):
        """
        导出分析结果

        Args:
            pairs: 问题-解法对列表
            output_path: 输出文件路径
            format: 输出格式 (json, txt, md)
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            if format == "json":
                self._export_json(pairs, output_file)
            elif format == "txt":
                self._export_txt(pairs, output_file)
            elif format == "md":
                self._export_markdown(pairs, output_file)
            else:
                raise ValueError(f"不支持的导出格式: {format}")

            logger.info(f"✅ 结果已导出到: {output_path}")

        except Exception as e:
            logger.error(f"导出失败: {e}")

    def _export_json(self, pairs: List[ProblemSolutionPair], output_file: Path):
        """导出为JSON格式"""
        data = {
            "pairs": [pair.to_dict() for pair in pairs],
            "total_count": len(pairs)
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _export_txt(self, pairs: List[ProblemSolutionPair], output_file: Path):
        """导出为文本格式"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Logic Analysis Report: Problem-Solution Pairs\n")
            f.write("=" * 80 + "\n\n")

            for i, pair in enumerate(pairs, 1):
                f.write(f"【Pair {i}】\n")
                f.write(f"Confidence: {pair.confidence:.2f}\n\n")
                f.write(f"Problem (The Lock):\n{pair.problem}\n\n")
                f.write(f"Solution (The Key):\n{pair.solution}\n\n")
                f.write(f"Explanation:\n{pair.explanation}\n\n")

                if pair.evidence:
                    f.write(f"Evidence:\n{pair.evidence}\n\n")

                f.write("-" * 80 + "\n\n")

    def _export_markdown(self, pairs: List[ProblemSolutionPair], output_file: Path):
        """导出为Markdown格式"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Logic Analysis Report: Problem-Solution Pairs\n\n")

            for i, pair in enumerate(pairs, 1):
                f.write(f"## Pair {i}\n\n")
                f.write(f"**Confidence:** {pair.confidence:.2f}\n\n")

                f.write(f"### 🔒 Problem (The Lock)\n\n")
                f.write(f"{pair.problem}\n\n")

                f.write(f"### 🔑 Solution (The Key)\n\n")
                f.write(f"{pair.solution}\n\n")

                f.write(f"### 💡 Explanation\n\n")
                f.write(f"{pair.explanation}\n\n")

                if pair.evidence:
                    f.write(f"### 📝 Evidence\n\n")
                    f.write(f"> {pair.evidence}\n\n")

                f.write("---\n\n")


def main():
    """测试代码"""
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="Logic Analyst Agent - 论文逻辑分析")
    parser.add_argument("--config", required=True, help="LLM配置文件路径")
    parser.add_argument("--paper", required=True, help="论文文本文件路径")
    parser.add_argument("--output", default="logic_analysis_results.json", help="输出文件路径")
    parser.add_argument("--format", choices=["json", "txt", "md"], default="json", help="输出格式")

    args = parser.parse_args()

    try:
        # 初始化LLM客户端
        config = LLMConfig.from_file(args.config)
        llm_client = LLMClient(config)

        # 读取论文内容
        with open(args.paper, 'r', encoding='utf-8') as f:
            paper_content = f.read()

        # 创建分析agent
        agent = LogicAnalystAgent(llm_client)

        # 执行分析
        pairs = agent.analyze(paper_content)

        # 导出结果
        agent.export_results(pairs, args.output, format=args.format)

        # 打印摘要
        print("\n" + "=" * 80)
        print(f"Logic Analysis Complete: Found {len(pairs)} Problem-Solution Pairs")
        print("=" * 80)

        for i, pair in enumerate(pairs, 1):
            print(f"\n【Pair {i}】(Confidence: {pair.confidence:.2f})")
            print(f"Problem: {pair.problem[:100]}...")
            print(f"Solution: {pair.solution[:100]}...")

    except Exception as e:
        logger.error(f"执行失败: {e}")
        raise


if __name__ == "__main__":
    main()
