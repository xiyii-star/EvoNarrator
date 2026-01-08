"""
Pairwise Win-Rate 评估
用于对比两个方法生成的创意，计算胜率

使用场景:
- MyMethod vs Naive LLM
- MyMethod vs Standard RAG
- MyMethod vs Ablation variants
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import random
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_config import LLMClient, LLMConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PairwiseComparator:
    """
    成对比较评估器

    模拟导师决策场景："如果您是导师，您会建议学生尝试哪个Idea？"
    """

    def __init__(self, llm_config: LLMConfig):
        """
        初始化比较器

        Args:
            llm_config: LLM配置
        """
        self.llm_client = LLMClient(llm_config)
        logger.info(f"✅ Pairwise比较器初始化成功，模型: {llm_config.model}")

    def compare_pair(
        self,
        idea_a: Dict,
        idea_b: Dict,
        method_a_name: str = "Method A",
        method_b_name: str = "Method B",
        randomize_order: bool = True
    ) -> Dict:
        """
        比较两个创意，返回胜者

        Args:
            idea_a: 方法A的创意
            idea_b: 方法B的创意
            method_a_name: 方法A名称（用于日志）
            method_b_name: 方法B名称（用于日志）
            randomize_order: 是否随机化展示顺序（避免位置偏差）

        Returns:
            比较结果字典，包含胜者和理由
        """
        # 随机化顺序避免位置偏差
        if randomize_order and random.random() < 0.5:
            presented_first = 'B'
            presented_second = 'A'
            first_idea = idea_b
            second_idea = idea_a
        else:
            presented_first = 'A'
            presented_second = 'B'
            first_idea = idea_a
            second_idea = idea_b

        # 构建比较提示词
        prompt = self._build_comparison_prompt(
            first_idea, second_idea,
            presented_first, presented_second
        )

        # 调用LLM进行比较
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self._get_system_prompt(),
                temperature=0.3,  # 低温度保证一致性
                max_tokens=800
            )

            # 解析响应
            result = self._parse_comparison_response(
                response,
                presented_first,
                presented_second
            )

            # 转换回原始标签
            if result['winner'] == 'A':
                actual_winner = 'A' if presented_first == 'A' else 'B'
            elif result['winner'] == 'B':
                actual_winner = 'B' if presented_second == 'B' else 'A'
            else:
                actual_winner = 'TIE'

            result['winner'] = actual_winner
            result['method_a_name'] = method_a_name
            result['method_b_name'] = method_b_name

            return result

        except Exception as e:
            logger.error(f"比较创意时出错: {e}")
            return {
                'winner': 'ERROR',
                'reasoning': f"比较失败: {str(e)}",
                'confidence': None,
                'method_a_name': method_a_name,
                'method_b_name': method_b_name
            }

    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """You are an experienced research advisor evaluating research ideas.

Your role: Imagine you are a Ph.D. advisor. Two students present their research ideas to you.
Your task: Decide which idea you would recommend the student to pursue.

Evaluation criteria (in order of importance):
1. **Impact Potential**: Which idea could make a more significant contribution to the field?
2. **Feasibility**: Which idea is more likely to succeed given current resources and technology?
3. **Novelty**: Which idea is more innovative and less incremental?
4. **Clarity**: Which idea has a clearer research plan and stronger logical reasoning?

Decision guidelines:
- Be decisive: Choose one idea as the winner (or declare a tie only if truly equivalent)
- Think holistically: Consider all aspects, not just one dimension
- Be practical: Favor ideas that balance ambition with feasibility
- Value substance over style: Look for concrete technical details, not vague claims

Provide your decision in JSON format:
{
  "winner": "A" | "B" | "TIE",
  "reasoning": "<2-3 sentences explaining your decision>",
  "confidence": "high" | "medium" | "low"
}"""

    def _build_comparison_prompt(
        self,
        idea_1: Dict,
        idea_2: Dict,
        label_1: str,
        label_2: str
    ) -> str:
        """构建比较提示词"""
        prompt_parts = [
            "Two students present their research ideas. Please decide which one you would recommend.",
            "",
            f"**Idea {label_1}:**",
            "",
            f"Title: {idea_1.get('title', 'N/A')}",
            "",
            f"Abstract: {idea_1.get('abstract', 'N/A')}",
            "",
            f"Key Modification: {idea_1.get('modification', 'N/A')}",
            "",
            f"Reasoning: {idea_1.get('reasoning', 'N/A')}",
            "",
            "---",
            "",
            f"**Idea {label_2}:**",
            "",
            f"Title: {idea_2.get('title', 'N/A')}",
            "",
            f"Abstract: {idea_2.get('abstract', 'N/A')}",
            "",
            f"Key Modification: {idea_2.get('modification', 'N/A')}",
            "",
            f"Reasoning: {idea_2.get('reasoning', 'N/A')}",
            "",
            "---",
            "",
            "**Your Decision:**",
            f"Which idea (A or B) would you recommend? Or are they equivalent (TIE)?",
            "",
            "Provide your answer in JSON format:",
            "{",
            '  "winner": "A" | "B" | "TIE",',
            '  "reasoning": "<explanation>",',
            '  "confidence": "high" | "medium" | "low"',
            "}"
        ]

        return "\n".join(prompt_parts)

    def _parse_comparison_response(
        self,
        response: str,
        label_1: str,
        label_2: str
    ) -> Dict:
        """解析比较响应"""
        import re

        try:
            # 提取JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)

                # 验证winner字段
                winner = result.get('winner', '').upper()
                if winner not in ['A', 'B', 'TIE']:
                    # 尝试从reasoning推断
                    reasoning = result.get('reasoning', '').lower()
                    if 'idea a' in reasoning or label_1.lower() in reasoning:
                        winner = 'A'
                    elif 'idea b' in reasoning or label_2.lower() in reasoning:
                        winner = 'B'
                    else:
                        winner = 'TIE'

                result['winner'] = winner
                result['reasoning'] = result.get('reasoning', '')
                result['confidence'] = result.get('confidence', 'medium')

                return result
            else:
                raise ValueError("未找到JSON格式的比较结果")

        except Exception as e:
            logger.error(f"解析比较响应失败: {e}")
            logger.error(f"响应内容: {response[:300]}")

            # 降级：尝试从文本推断
            response_lower = response.lower()
            if 'idea a' in response_lower or 'choose a' in response_lower:
                winner = 'A'
            elif 'idea b' in response_lower or 'choose b' in response_lower:
                winner = 'B'
            else:
                winner = 'TIE'

            return {
                'winner': winner,
                'reasoning': f"解析失败，从文本推断: {response[:200]}",
                'confidence': 'low'
            }

    def batch_compare(
        self,
        ideas_a: List[Dict],
        ideas_b: List[Dict],
        method_a_name: str = "Method A",
        method_b_name: str = "Method B",
        max_pairs: int = None,
        num_rounds: int = 1
    ) -> Dict:
        """
        批量比较两组创意

        Args:
            ideas_a: 方法A的创意列表
            ideas_b: 方法B的创意列表
            method_a_name: 方法A名称
            method_b_name: 方法B名称
            max_pairs: 最大比较对数（None表示全部）
            num_rounds: 重复比较轮数（用于计算稳定胜率）

        Returns:
            批量比较结果，包含胜率统计
        """
        logger.info("=" * 80)
        logger.info(f"开始Pairwise比较: {method_a_name} vs {method_b_name}")
        logger.info(f"重复轮数: {num_rounds}")
        logger.info("=" * 80)

        # 配对创意
        n_pairs = min(len(ideas_a), len(ideas_b))
        if max_pairs:
            n_pairs = min(n_pairs, max_pairs)

        logger.info(f"比较对数: {n_pairs}")

        all_round_results = []
        total_wins_a = 0
        total_wins_b = 0
        total_ties = 0
        total_errors = 0

        for round_num in range(num_rounds):
            logger.info(f"\n{'='*80}")
            logger.info(f"第 {round_num+1}/{num_rounds} 轮比较")
            logger.info(f"{'='*80}")

            results = []
            wins_a = 0
            wins_b = 0
            ties = 0
            errors = 0

            for i in range(n_pairs):
                logger.info(f"\n比较 {i+1}/{n_pairs}:")
                logger.info(f"  {method_a_name}: {ideas_a[i].get('title', 'N/A')[:60]}...")
                logger.info(f"  {method_b_name}: {ideas_b[i].get('title', 'N/A')[:60]}...")

                comparison = self.compare_pair(
                    idea_a=ideas_a[i],
                    idea_b=ideas_b[i],
                    method_a_name=method_a_name,
                    method_b_name=method_b_name,
                    randomize_order=True
                )

                winner = comparison['winner']

                if winner == 'A':
                    wins_a += 1
                    logger.info(f"  ✅ 胜者: {method_a_name}")
                elif winner == 'B':
                    wins_b += 1
                    logger.info(f"  ✅ 胜者: {method_b_name}")
                elif winner == 'TIE':
                    ties += 1
                    logger.info(f"  🤝 平局")
                else:
                    errors += 1
                    logger.warning(f"  ❌ 比较失败")

                logger.info(f"  信心: {comparison.get('confidence', 'N/A')}")
                logger.info(f"  理由: {comparison.get('reasoning', 'N/A')[:100]}...")

                results.append(comparison)

            # 汇总本轮结果
            total_wins_a += wins_a
            total_wins_b += wins_b
            total_ties += ties
            total_errors += errors

            logger.info(f"\n--- 第 {round_num+1} 轮结果 ---")
            logger.info(f"{method_a_name} 胜: {wins_a}/{n_pairs}")
            logger.info(f"{method_b_name} 胜: {wins_b}/{n_pairs}")
            logger.info(f"平局: {ties}/{n_pairs}")

            all_round_results.append({
                'round': round_num + 1,
                'wins_a': wins_a,
                'wins_b': wins_b,
                'ties': ties,
                'errors': errors,
                'comparisons': results
            })

        # 计算总体统计数据
        total_comparisons = n_pairs * num_rounds
        total_valid = total_comparisons - total_errors
        win_rate_a = (total_wins_a / total_valid * 100) if total_valid > 0 else 0
        win_rate_b = (total_wins_b / total_valid * 100) if total_valid > 0 else 0
        tie_rate = (total_ties / total_valid * 100) if total_valid > 0 else 0

        summary = {
            'method_a_name': method_a_name,
            'method_b_name': method_b_name,
            'num_rounds': num_rounds,
            'pairs_per_round': n_pairs,
            'total_comparisons': total_comparisons,
            'wins_a': total_wins_a,
            'wins_b': total_wins_b,
            'ties': total_ties,
            'errors': total_errors,
            'win_rate_a': round(win_rate_a, 1),
            'win_rate_b': round(win_rate_b, 1),
            'tie_rate': round(tie_rate, 1),
            'round_results': all_round_results
        }

        logger.info("\n" + "=" * 80)
        logger.info("最终比较结果汇总")
        logger.info("=" * 80)
        logger.info(f"总比较次数: {total_comparisons} ({num_rounds} 轮 × {n_pairs} 对)")
        logger.info(f"{method_a_name} 胜: {total_wins_a} ({win_rate_a:.1f}%)")
        logger.info(f"{method_b_name} 胜: {total_wins_b} ({win_rate_b:.1f}%)")
        logger.info(f"平局: {total_ties} ({tie_rate:.1f}%)")
        if total_errors > 0:
            logger.warning(f"错误: {total_errors}")

        return summary


def load_ideas_from_file(file_path: str) -> List[Dict]:
    """
    从JSON文件加载创意列表

    Args:
        file_path: 文件路径

    Returns:
        创意列表
    """
    logger.info(f"加载创意文件: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取ideas字段
    if isinstance(data, dict) and 'ideas' in data:
        ideas = data['ideas']
    elif isinstance(data, list):
        ideas = data
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")

    logger.info(f"  加载了 {len(ideas)} 个创意")

    return ideas


def save_comparison_results(results: Dict, output_file: str):
    """保存比较结果"""
    logger.info(f"保存比较结果到: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("✅ 结果已保存")


def generate_comparison_report(results: Dict, output_file: str):
    """生成Markdown格式的对比报告"""
    logger.info(f"生成对比报告: {output_file}")

    method_a = results['method_a_name']
    method_b = results['method_b_name']
    win_rate_a = results['win_rate_a']
    win_rate_b = results['win_rate_b']
    num_rounds = results.get('num_rounds', 1)

    report_lines = [
        f"# Pairwise Win-Rate 对比报告",
        "",
        f"**对比方法**: {method_a} vs {method_b}",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**比较轮数**: {num_rounds}",
        f"**每轮对数**: {results.get('pairs_per_round', 'N/A')}",
        "",
        "## 胜率统计",
        "",
        "| 方法 | 胜场 | 胜率 |",
        "|------|------|------|",
        f"| **{method_a}** | {results['wins_a']}/{results['total_comparisons']} | **{win_rate_a:.1f}%** |",
        f"| {method_b} | {results['wins_b']}/{results['total_comparisons']} | {win_rate_b:.1f}% |",
        f"| 平局 | {results['ties']}/{results['total_comparisons']} | {results['tie_rate']:.1f}% |",
        "",
        "## 可视化",
        "",
        f"```",
        f"{method_a}: {'█' * int(win_rate_a / 5)} {win_rate_a:.1f}%",
        f"{method_b}: {'█' * int(win_rate_b / 5)} {win_rate_b:.1f}%",
        f"平局:    {'█' * int(results['tie_rate'] / 5)} {results['tie_rate']:.1f}%",
        f"```",
        ""
    ]

    # 如果有多轮比较，显示每轮结果
    if num_rounds > 1:
        report_lines.extend([
            "## 各轮比较结果",
            "",
            "| 轮次 | 方法A胜 | 方法B胜 | 平局 | 错误 |",
            "|------|---------|---------|------|------|"
        ])

        for round_data in results.get('round_results', []):
            round_num = round_data['round']
            wins_a = round_data['wins_a']
            wins_b = round_data['wins_b']
            ties = round_data['ties']
            errors = round_data['errors']
            report_lines.append(
                f"| 第{round_num}轮 | {wins_a} | {wins_b} | {ties} | {errors} |"
            )

        report_lines.append("")

    # 详细比较结果（只显示第一轮的详细信息，避免报告过长）
    report_lines.extend([
        "## 详细比较结果（第1轮）",
        ""
    ])

    round_results = results.get('round_results', [])
    if round_results:
        first_round = round_results[0]
        for i, comp in enumerate(first_round['comparisons'], 1):
            winner = comp['winner']
            if winner == 'A':
                winner_name = method_a
            elif winner == 'B':
                winner_name = method_b
            else:
                winner_name = "平局"

            report_lines.extend([
                f"### 比较 {i}",
                "",
                f"**胜者**: {winner_name} (信心: {comp.get('confidence', 'N/A')})",
                "",
                f"**理由**: {comp.get('reasoning', 'N/A')}",
                ""
            ])

    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    logger.info("✅ 报告已生成")


def main():
    """命令行入口"""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description='Pairwise Win-Rate 评估',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # MyMethod vs Naive LLM
  python idea_eval/pairwise_compare.py \\
    --method-a results/mymethod_ideas.json \\
    --method-b results/naive_llm_ideas.json \\
    --name-a "MyMethod" \\
    --name-b "Naive LLM" \\
    --output results/pairwise_mymethod_vs_naive.json

  # MyMethod vs Standard RAG
  python idea_eval/pairwise_compare.py \\
    --method-a results/mymethod_ideas.json \\
    --method-b results/standardrag_ideas.json \\
    --name-a "MyMethod" \\
    --name-b "Standard RAG" \\
    --output results/pairwise_mymethod_vs_rag.json
        """
    )

    parser.add_argument('--method-a', type=str, required=True, help='方法A的创意文件')
    parser.add_argument('--method-b', type=str, required=True, help='方法B的创意文件')
    parser.add_argument('--name-a', type=str, default='Method A', help='方法A名称')
    parser.add_argument('--name-b', type=str, default='Method B', help='方法B名称')
    parser.add_argument('--output', type=str, required=True, help='输出结果JSON文件')
    parser.add_argument('--report', type=str, help='输出Markdown报告文件（可选）')
    parser.add_argument('--config', type=str, default='/home/lexy/下载/CLwithRAG/KGdemo/config/config.yaml', help='LLM配置文件')
    parser.add_argument('--max-pairs', type=int, help='最大比较对数（默认全部）')
    parser.add_argument('--num-rounds', type=int, default=1, help='重复比较轮数（默认1轮，建议3-5轮获得稳定胜率）')

    args = parser.parse_args()

    # 检查文件
    for file_path in [args.method_a, args.method_b]:
        if not Path(file_path).exists():
            logger.error(f"❌ 文件不存在: {file_path}")
            sys.exit(1)

    try:
        # 加载配置
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        llm_config_dict = config.get('llm', {})
        llm_config = LLMConfig(
            provider=llm_config_dict.get('provider', 'openai'),
            model=llm_config_dict.get('model', 'gpt-4o'),
            api_key=llm_config_dict.get('api_key'),
            base_url=llm_config_dict.get('base_url'),
            temperature=0.3,
            max_tokens=800
        )

        # 初始化比较器
        comparator = PairwiseComparator(llm_config)

        # 加载创意
        ideas_a = load_ideas_from_file(args.method_a)
        ideas_b = load_ideas_from_file(args.method_b)

        # 批量比较
        results = comparator.batch_compare(
            ideas_a=ideas_a,
            ideas_b=ideas_b,
            method_a_name=args.name_a,
            method_b_name=args.name_b,
            max_pairs=args.max_pairs,
            num_rounds=args.num_rounds
        )

        # 保存结果
        save_comparison_results(results, args.output)

        # 生成报告
        if args.report:
            generate_comparison_report(results, args.report)
        else:
            # 自动生成报告文件名
            report_file = args.output.replace('.json', '_report.md')
            generate_comparison_report(results, report_file)

        logger.info(f"\n🎉 Pairwise比较完成！")
        logger.info(f"结果文件: {args.output}")
        logger.info(f"胜率: {args.name_a} {results['win_rate_a']:.1f}% vs {args.name_b} {results['win_rate_b']:.1f}%")

    except Exception as e:
        logger.error(f"❌ 比较失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
