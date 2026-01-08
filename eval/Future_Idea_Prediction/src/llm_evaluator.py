"""
LLM评估模块：
使用LLM评估生成的idea与真实论文的相似度
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm


class LLMEvaluator:
    """使用LLM评估idea与论文的相似度"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        config_dict: Optional[Dict] = None
    ):
        """
        初始化LLM评估器

        Args:
            model_name: 模型名称（如果提供则覆盖config_dict）
            api_key: API密钥（如果提供则覆盖config_dict）
            base_url: API基础URL（如果提供则覆盖config_dict）
            temperature: 温度参数
            max_tokens: 最大token数
            config_dict: 配置字典（从ConfigLoader获取）
        """
        # 优先使用传入的参数，否则使用config_dict
        if config_dict:
            self.model_name = model_name or config_dict.get('model', 'gpt-4')
            self.api_key = api_key or config_dict.get('api_key')
            self.base_url = base_url or config_dict.get('base_url')
            self.temperature = config_dict.get('temperature', temperature)
            self.max_tokens = config_dict.get('max_tokens', max_tokens)
        else:
            self.model_name = model_name or "gpt-4"
            self.api_key = api_key
            self.base_url = base_url
            self.temperature = temperature
            self.max_tokens = max_tokens

        # 初始化OpenAI客户端（兼容本地模型）
        try:
            from openai import OpenAI
            if self.base_url:
                # 本地模型或自定义API
                self.client = OpenAI(api_key=self.api_key or "dummy-key", base_url=self.base_url)
            else:
                # OpenAI官方API
                self.client = OpenAI(api_key=self.api_key)

            print(f"LLM Evaluator initialized:")
            print(f"  Model: {self.model_name}")
            print(f"  Base URL: {self.base_url or 'Default (OpenAI)'}")
            print(f"  Temperature: {self.temperature}")
            print(f"  Max Tokens: {self.max_tokens}")
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def create_evaluation_prompt(self, idea_text: str, paper_abstract: str) -> str:
        """
        创建评估prompt

        Args:
            idea_text: 系统生成的idea
            paper_abstract: 真实论文的abstract

        Returns:
            评估prompt
        """
        prompt = f"""Role: You are a Senior Research Reviewer.

Task: I will provide you with two research ideas: one is a system prediction, and the other is a real published paper. Please determine if their core innovations are essentially the same.

Idea A (System Prediction):
{idea_text}

Idea B (Real Paper):
{paper_abstract}

Criteria:
1. Is the problem addressed consistent?
2. Are the proposed core methods/technical routes highly similar?
3. Are the application scenarios and goals the same?

Output Format (Please strictly follow this JSON format):
{{
    "match_score": <Integer between 0-10, where 10 means almost identical>,
    "problem_consistency": <Integer 0-10, score for problem consistency>,
    "method_similarity": <Integer 0-10, score for method similarity>,
    "application_similarity": <Integer 0-10, score for application similarity>,
    "reason": "<Brief explanation for the given scores>"
}}

Please output ONLY the JSON, with no additional text.
"""

        return prompt

    def evaluate_single_pair(
        self,
        idea_text: str,
        paper_abstract: str,
        temperature: Optional[float] = None
    ) -> Dict:
        """
        评估单个idea-paper对

        Args:
            idea_text: 系统生成的idea
            paper_abstract: 真实论文的abstract
            temperature: 生成温度（如果为None则使用初始化时的值）

        Returns:
            评估结果字典
        """
        prompt = self.create_evaluation_prompt(idea_text, paper_abstract)

        # 使用传入的temperature或默认值
        temp = temperature if temperature is not None else self.temperature

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert research reviewer who evaluates research ideas objectively."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temp,
                max_tokens=self.max_tokens
            )

            result_text = response.choices[0].message.content.strip()

            # 解析JSON结果
            # 尝试提取JSON部分
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_json = json.loads(json_match.group(0))
            else:
                result_json = json.loads(result_text)

            # 验证必要字段
            required_fields = ["match_score", "reason"]
            for field in required_fields:
                if field not in result_json:
                    raise ValueError(f"Missing required field: {field}")

            return result_json

        except Exception as e:
            print(f"Error evaluating pair: {e}")
            return {
                "match_score": 0,
                "problem_consistency": 0,
                "method_similarity": 0,
                "application_similarity": 0,
                "reason": f"Evaluation failed: {str(e)}",
                "error": True
            }

    def evaluate_idea_against_papers(
        self,
        idea_text: str,
        candidate_papers: List[Dict],
        threshold: float = 7.0
    ) -> Dict:
        """
        评估一个idea与多篇候选论文的相似度

        Args:
            idea_text: 系统生成的idea
            candidate_papers: 候选论文列表，每个包含paper_id, title, abstract, year等
            threshold: 认为是匹配的最低分数阈值

        Returns:
            评估结果字典
        """
        results = []

        for paper in tqdm(candidate_papers, desc="Evaluating papers"):
            eval_result = self.evaluate_single_pair(
                idea_text,
                paper["abstract"]
            )

            result_entry = {
                "paper_id": paper["paper_id"],
                "paper_title": paper["title"],
                "paper_year": paper["year"],
                "match_score": eval_result["match_score"],
                "problem_consistency": eval_result.get("problem_consistency", 0),
                "method_similarity": eval_result.get("method_similarity", 0),
                "application_similarity": eval_result.get("application_similarity", 0),
                "reason": eval_result["reason"],
                "is_match": eval_result["match_score"] >= threshold
            }

            results.append(result_entry)

        # 找到最高分
        best_match = max(results, key=lambda x: x["match_score"])

        return {
            "idea_text": idea_text,
            "all_results": results,
            "best_match": best_match,
            "has_match": best_match["match_score"] >= threshold,
            "max_score": best_match["match_score"]
        }

    def batch_evaluate_ideas(
        self,
        ideas: List[Dict],
        retrieval_results: Dict[str, List[Tuple[Dict, float]]],
        threshold: float = 7.0,
        save_path: str = None
    ) -> List[Dict]:
        """
        批量评估多个ideas

        Args:
            ideas: idea列表，每个包含idea_id和idea_text
            retrieval_results: 检索结果，key为idea_id，value为(paper, score)列表
            threshold: 匹配阈值
            save_path: 保存结果的路径

        Returns:
            评估结果列表
        """
        all_results = []

        for idea in tqdm(ideas, desc="Evaluating ideas"):
            idea_id = idea["idea_id"]
            idea_text = idea["idea_text"]

            # 获取候选论文
            candidates = retrieval_results.get(idea_id, [])
            candidate_papers = [paper for paper, _ in candidates]

            if not candidate_papers:
                print(f"Warning: No candidates found for idea {idea_id}")
                continue

            # 评估
            eval_result = self.evaluate_idea_against_papers(
                idea_text,
                candidate_papers,
                threshold
            )

            eval_result["idea_id"] = idea_id
            all_results.append(eval_result)

        # 保存结果
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

            print(f"Evaluation results saved to {save_path}")

        return all_results


def main():
    """示例：LLM评估"""

    # 初始化评估器（这里使用本地部署的模型作为示例）
    evaluator = LLMEvaluator(
        model_name="gpt-4",  # 或本地模型名称
        base_url="http://localhost:8000/v1"  # 本地模型API地址，如果使用OpenAI则设为None
    )

    # 示例数据
    idea_text = """
    We propose a novel attention mechanism for transformer models that reduces computational
    complexity from O(n^2) to O(n log n) while maintaining performance. The key innovation is
    a hierarchical attention pattern that groups similar tokens together.
    """

    paper_abstract = """
    This paper introduces an efficient attention mechanism for transformers that achieves
    linear complexity. We use locality-sensitive hashing to group similar tokens and compute
    attention only within groups. Experiments show comparable performance to standard attention
    with significantly reduced computation.
    """

    # 评估单个pair
    result = evaluator.evaluate_single_pair(idea_text, paper_abstract)

    print("Evaluation Result:")
    print(f"Match Score: {result['match_score']}/10")
    print(f"Reason: {result['reason']}")


if __name__ == "__main__":
    main()
