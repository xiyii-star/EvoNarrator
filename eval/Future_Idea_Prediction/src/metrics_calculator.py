"""
命中率计算模块：
计算Hit Rate@K等评估指标
"""

import json
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from collections import defaultdict


class MetricsCalculator:
    """评估指标计算器"""

    def __init__(self):
        """初始化指标计算器"""
        pass

    def calculate_hit_rate_at_k(
        self,
        evaluation_results: List[Dict],
        k_values: List[int] = [1, 3, 5, 10],
        threshold: float = 7.0
    ) -> Dict[int, float]:
        """
        计算Hit Rate@K

        Hit Rate@K: 在前K个生成的idea中，有多少个成功匹配到至少一篇真实论文

        Args:
            evaluation_results: 评估结果列表
            k_values: K值列表
            threshold: 认为匹配成功的最低分数

        Returns:
            Dict[K, hit_rate]
        """
        total_ideas = len(evaluation_results)
        if total_ideas == 0:
            return {k: 0.0 for k in k_values}

        # 统计每个K值的命中数
        hit_counts = {k: 0 for k in k_values}

        for result in evaluation_results:
            # 检查是否有匹配
            has_match = result.get("has_match", False)
            max_score = result.get("max_score", 0)

            # 如果超过阈值，则认为命中
            if has_match or max_score >= threshold:
                for k in k_values:
                    hit_counts[k] += 1

        # 计算命中率，转换为Python原生float
        hit_rates = {k: float(hit_counts[k] / total_ideas) for k in k_values}

        return hit_rates

    def calculate_precision_at_k(
        self,
        evaluation_results: List[Dict],
        k: int = 5,
        threshold: float = 7.0
    ) -> float:
        """
        计算Precision@K

        Precision@K: 在返回的前K个论文中，匹配的论文占比

        Args:
            evaluation_results: 评估结果列表
            k: K值
            threshold: 匹配阈值

        Returns:
            Precision@K
        """
        total_retrieved = 0
        total_relevant = 0

        for result in evaluation_results:
            all_results = result.get("all_results", [])[:k]
            total_retrieved += len(all_results)
            total_relevant += sum(1 for r in all_results if r["match_score"] >= threshold)

        if total_retrieved == 0:
            return 0.0

        return float(total_relevant / total_retrieved)

    def calculate_mrr(
        self,
        evaluation_results: List[Dict],
        threshold: float = 7.0
    ) -> float:
        """
        计算Mean Reciprocal Rank (MRR)

        MRR: 第一个相关结果的排名的倒数的平均值

        Args:
            evaluation_results: 评估结果列表
            threshold: 匹配阈值

        Returns:
            MRR值
        """
        reciprocal_ranks = []

        for result in evaluation_results:
            all_results = result.get("all_results", [])

            # 找到第一个匹配的位置
            first_match_rank = None
            for rank, r in enumerate(all_results, 1):
                if r["match_score"] >= threshold:
                    first_match_rank = rank
                    break

            if first_match_rank:
                reciprocal_ranks.append(1.0 / first_match_rank)
            else:
                reciprocal_ranks.append(0.0)

        if not reciprocal_ranks:
            return 0.0

        return float(np.mean(reciprocal_ranks))

    def calculate_average_max_score(
        self,
        evaluation_results: List[Dict]
    ) -> float:
        """
        计算平均最高分

        Args:
            evaluation_results: 评估结果列表

        Returns:
            平均最高分
        """
        if not evaluation_results:
            return 0.0

        max_scores = [result.get("max_score", 0) for result in evaluation_results]
        return float(np.mean(max_scores))

    def calculate_score_distribution(
        self,
        evaluation_results: List[Dict]
    ) -> Dict[str, int]:
        """
        计算分数分布

        Args:
            evaluation_results: 评估结果列表

        Returns:
            分数区间的分布统计
        """
        distribution = {
            "0-2": 0,
            "3-4": 0,
            "5-6": 0,
            "7-8": 0,
            "9-10": 0
        }

        for result in evaluation_results:
            max_score = result.get("max_score", 0)

            if max_score <= 2:
                distribution["0-2"] += 1
            elif max_score <= 4:
                distribution["3-4"] += 1
            elif max_score <= 6:
                distribution["5-6"] += 1
            elif max_score <= 8:
                distribution["7-8"] += 1
            else:
                distribution["9-10"] += 1

        return distribution

    def calculate_year_wise_metrics(
        self,
        evaluation_results: List[Dict],
        threshold: float = 7.0
    ) -> Dict[str, Dict]:
        """
        按年份统计指标

        Args:
            evaluation_results: 评估结果列表
            threshold: 匹配阈值

        Returns:
            每年的统计指标
        """
        year_stats = defaultdict(lambda: {"total": 0, "matches": 0, "scores": []})

        for result in evaluation_results:
            best_match = result.get("best_match", {})
            year = best_match.get("paper_year", "unknown")
            score = best_match.get("match_score", 0)

            year_stats[year]["total"] += 1
            year_stats[year]["scores"].append(score)

            if score >= threshold:
                year_stats[year]["matches"] += 1

        # 计算每年的命中率和平均分
        year_metrics = {}
        for year, stats in year_stats.items():
            year_metrics[year] = {
                "total_ideas": int(stats["total"]),
                "matched_ideas": int(stats["matches"]),
                "hit_rate": float(stats["matches"] / stats["total"]) if stats["total"] > 0 else 0.0,
                "average_score": float(np.mean(stats["scores"])) if stats["scores"] else 0.0,
                "max_score": float(np.max(stats["scores"])) if stats["scores"] else 0.0,
                "min_score": float(np.min(stats["scores"])) if stats["scores"] else 0.0
            }

        return year_metrics

    def get_top_matches(
        self,
        evaluation_results: List[Dict],
        top_n: int = 10
    ) -> List[Dict]:
        """
        获取得分最高的Top N匹配

        Args:
            evaluation_results: 评估结果列表
            top_n: 返回的top N数量

        Returns:
            Top N匹配列表
        """
        # 按max_score排序
        sorted_results = sorted(
            evaluation_results,
            key=lambda x: x.get('max_score', 0),
            reverse=True
        )

        top_matches = []
        for result in sorted_results[:top_n]:
            if 'best_match' in result:
                best_match = result['best_match']
                top_matches.append({
                    'idea_id': result['idea_id'],
                    'score': float(result['max_score']),
                    'paper_title': best_match.get('paper_title', 'N/A'),
                    'paper_year': best_match.get('paper_year', 'N/A'),
                    'reason': best_match.get('reason', 'N/A')
                })

        return top_matches

    def generate_report(
        self,
        evaluation_results: List[Dict],
        k_values: List[int] = [1, 3, 5, 10],
        threshold: float = 7.0,
        save_path: str = None,
        top_n: int = 10
    ) -> Dict:
        """
        生成完整的评估报告

        Args:
            evaluation_results: 评估结果列表
            k_values: K值列表
            threshold: 匹配阈值
            save_path: 保存报告的路径
            top_n: 显示Top N最佳匹配（默认10）

        Returns:
            完整的评估报告字典
        """
        print("Generating evaluation report...")

        report = {
            "total_ideas": int(len(evaluation_results)),
            "threshold": float(threshold),
            "hit_rate_at_k": self.calculate_hit_rate_at_k(evaluation_results, k_values, threshold),
            "precision_at_5": self.calculate_precision_at_k(evaluation_results, k=5, threshold=threshold),
            "mrr": self.calculate_mrr(evaluation_results, threshold),
            "average_max_score": self.calculate_average_max_score(evaluation_results),
            "score_distribution": self.calculate_score_distribution(evaluation_results),
            "year_wise_metrics": self.calculate_year_wise_metrics(evaluation_results, threshold),
            "top_matches": self.get_top_matches(evaluation_results, top_n=top_n)
        }

        # 打印报告
        self._print_report(report)

        # 保存报告
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            print(f"\nReport saved to {save_path}")

        return report

    def _print_report(self, report: Dict):
        """打印报告到控制台"""
        print("\n" + "="*80)
        print("EVALUATION REPORT".center(80))
        print("="*80)

        print(f"\nTotal Ideas Evaluated: {report['total_ideas']}")
        print(f"Match Threshold: {report['threshold']}")

        print("\n" + "-"*80)
        print("Hit Rate@K:")
        for k, rate in sorted(report['hit_rate_at_k'].items()):
            print(f"  Hit Rate@{k:2d}: {rate:.2%}")

        print("\n" + "-"*80)
        print("Other Metrics:")
        print(f"  Precision@5: {report['precision_at_5']:.2%}")
        print(f"  MRR:         {report['mrr']:.4f}")
        print(f"  Avg Max Score: {report['average_max_score']:.2f}/10")

        print("\n" + "-"*80)
        print("Score Distribution:")
        for range_str, count in report['score_distribution'].items():
            percentage = count / report['total_ideas'] * 100 if report['total_ideas'] > 0 else 0
            print(f"  {range_str}: {count:4d} ({percentage:5.1f}%)")

        print("\n" + "-"*80)
        print("Year-wise Metrics:")
        for year, metrics in sorted(report['year_wise_metrics'].items()):
            print(f"\n  {year}:")
            print(f"    Total Ideas:   {metrics['total_ideas']}")
            print(f"    Matched:       {metrics['matched_ideas']}")
            print(f"    Hit Rate:      {metrics['hit_rate']:.2%}")
            print(f"    Avg Score:     {metrics['average_score']:.2f}/10")
            print(f"    Score Range:   {metrics['min_score']:.1f} - {metrics['max_score']:.1f}")

        # 打印Top N最佳匹配
        if 'top_matches' in report and report['top_matches']:
            print("\n" + "-"*80)
            print(f"Top {len(report['top_matches'])} Best Matches:")
            for i, match in enumerate(report['top_matches'], 1):
                print(f"\n  #{i} - Idea: {match['idea_id']} (Score: {match['score']:.1f}/10)")
                print(f"      Paper: {match['paper_title']}")
                print(f"      Year:  {match['paper_year']}")
                # 截断reason到合理长度
                reason = match['reason']
                if len(reason) > 150:
                    reason = reason[:147] + "..."
                print(f"      Reason: {reason}")

        print("\n" + "="*80)


def main():
    """示例：计算指标"""

    # 加载评估结果
    results_file = "/home/lexy/下载/CLwithRAG/KGdemo/eval/Future_Idea_Prediction/results/evaluation_results.json"

    if not Path(results_file).exists():
        print(f"Results file not found: {results_file}")
        print("Please run the evaluation first.")
        return

    with open(results_file, "r", encoding="utf-8") as f:
        evaluation_results = json.load(f)

    # 计算指标
    calculator = MetricsCalculator()
    report = calculator.generate_report(
        evaluation_results,
        k_values=[1, 3, 5, 10, 20],
        threshold=7.0,
        save_path="/home/lexy/下载/CLwithRAG/KGdemo/eval/Future_Idea_Prediction/results/metrics_report.json"
    )


if __name__ == "__main__":
    main()
