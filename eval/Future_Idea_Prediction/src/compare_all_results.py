"""
批量评估结果汇总分析脚本
比较所有评估结果，生成对比报告
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import sys


def load_metrics_report(result_dir: Path) -> Dict:
    """加载单个结果目录的metrics报告"""
    metrics_file = result_dir / "metrics_report.json"

    if not metrics_file.exists():
        return None

    with open(metrics_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def collect_all_results(results_base_dir: Path) -> List[Dict]:
    """收集所有结果目录的metrics"""
    all_results = []

    # 遍历results目录下的所有子目录
    for result_dir in results_base_dir.iterdir():
        if not result_dir.is_dir():
            continue

        # 跳过非结果目录
        if not result_dir.name.endswith('_results'):
            continue

        metrics = load_metrics_report(result_dir)
        if metrics:
            all_results.append({
                'name': result_dir.name,
                'path': str(result_dir),
                'metrics': metrics
            })

    return all_results


def print_comparison_report(all_results: List[Dict]):
    """打印对比报告"""

    if not all_results:
        print("No evaluation results found!")
        return

    print("\n" + "="*100)
    print("BATCH EVALUATION COMPARISON REPORT".center(100))
    print("="*100 + "\n")

    print(f"Total evaluated idea sets: {len(all_results)}\n")

    # 表头
    print("-"*100)
    print(f"{'Idea Set Name':<60} {'Ideas':>8} {'Avg Score':>10} {'Hit@5':>10}")
    print("-"*100)

    # 每个结果的摘要
    results_summary = []
    for result in all_results:
        name = result['name'].replace('_results', '')
        if len(name) > 57:
            name = name[:54] + "..."

        metrics = result['metrics']
        total_ideas = metrics.get('total_ideas', 0)
        avg_score = metrics.get('average_max_score', 0.0)
        hit_rate_5 = metrics.get('hit_rate_at_k', {}).get('5', 0.0) * 100

        print(f"{name:<60} {total_ideas:>8} {avg_score:>9.2f} {hit_rate_5:>9.1f}%")

        results_summary.append({
            'name': result['name'],
            'total_ideas': total_ideas,
            'avg_score': avg_score,
            'hit_rate_5': hit_rate_5,
            'metrics': metrics
        })

    print("-"*100)

    # 统计汇总
    total_ideas_all = sum(r['total_ideas'] for r in results_summary)
    avg_score_all = sum(r['avg_score'] * r['total_ideas'] for r in results_summary) / total_ideas_all if total_ideas_all > 0 else 0
    avg_hit_rate_5 = sum(r['hit_rate_5'] for r in results_summary) / len(results_summary) if results_summary else 0

    print(f"\n{'OVERALL':<60} {total_ideas_all:>8} {avg_score_all:>9.2f} {avg_hit_rate_5:>9.1f}%")
    print("-"*100)

    # 详细统计
    print("\n" + "="*100)
    print("DETAILED STATISTICS".center(100))
    print("="*100 + "\n")

    # 找出最佳和最差
    best_by_score = max(results_summary, key=lambda x: x['avg_score'])
    worst_by_score = min(results_summary, key=lambda x: x['avg_score'])
    best_by_hit_rate = max(results_summary, key=lambda x: x['hit_rate_5'])

    print(f"Best Average Score:  {best_by_score['name'].replace('_results', '')}")
    print(f"  Average Score: {best_by_score['avg_score']:.2f}/10")
    print(f"  Total Ideas:   {best_by_score['total_ideas']}")
    print()

    print(f"Worst Average Score: {worst_by_score['name'].replace('_results', '')}")
    print(f"  Average Score: {worst_by_score['avg_score']:.2f}/10")
    print(f"  Total Ideas:   {worst_by_score['total_ideas']}")
    print()

    print(f"Best Hit Rate@5:     {best_by_hit_rate['name'].replace('_results', '')}")
    print(f"  Hit Rate@5:    {best_by_hit_rate['hit_rate_5']:.1f}%")
    print(f"  Average Score: {best_by_hit_rate['avg_score']:.2f}/10")
    print()

    # 分数分布汇总
    print("-"*100)
    print("SCORE DISTRIBUTION SUMMARY")
    print("-"*100)

    score_ranges = ["0-2", "3-4", "5-6", "7-8", "9-10"]
    total_distribution = {r: 0 for r in score_ranges}

    for result in results_summary:
        dist = result['metrics'].get('score_distribution', {})
        for range_str, count in dist.items():
            if range_str in total_distribution:
                total_distribution[range_str] += count

    total_count = sum(total_distribution.values())

    print(f"\n{'Score Range':<15} {'Count':>10} {'Percentage':>12}")
    print("-"*40)
    for range_str in score_ranges:
        count = total_distribution[range_str]
        percentage = (count / total_count * 100) if total_count > 0 else 0
        print(f"{range_str:<15} {count:>10} {percentage:>11.1f}%")
    print("-"*40)

    # Top matches汇总
    print("\n" + "="*100)
    print("TOP MATCHES ACROSS ALL EVALUATIONS".center(100))
    print("="*100 + "\n")

    # 收集所有top matches
    all_top_matches = []
    for result in all_results:
        result_name = result['name'].replace('_results', '')
        top_matches = result['metrics'].get('top_matches', [])

        for match in top_matches[:3]:  # 只取每个结果的前3名
            all_top_matches.append({
                'source': result_name,
                'idea_id': match['idea_id'],
                'score': match['score'],
                'paper_title': match['paper_title'],
                'paper_year': match['paper_year']
            })

    # 按分数排序
    all_top_matches.sort(key=lambda x: x['score'], reverse=True)

    print("Top 10 Best Matches Overall:\n")
    for i, match in enumerate(all_top_matches[:10], 1):
        source = match['source']
        if len(source) > 50:
            source = source[:47] + "..."

        print(f"{i:2d}. Score: {match['score']:.1f}/10 - {match['idea_id']}")
        print(f"    From:  {source}")
        print(f"    Paper: {match['paper_title'][:70]}...")
        print(f"    Year:  {match['paper_year']}")
        print()

    print("="*100)


def save_comparison_json(all_results: List[Dict], output_file: Path):
    """保存对比结果为JSON"""
    comparison_data = {
        'total_evaluations': len(all_results),
        'evaluations': []
    }

    for result in all_results:
        comparison_data['evaluations'].append({
            'name': result['name'],
            'path': result['path'],
            'total_ideas': result['metrics'].get('total_ideas', 0),
            'average_max_score': result['metrics'].get('average_max_score', 0.0),
            'hit_rate_at_k': result['metrics'].get('hit_rate_at_k', {}),
            'score_distribution': result['metrics'].get('score_distribution', {}),
            'top_3_matches': result['metrics'].get('top_matches', [])[:3]
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, ensure_ascii=False, indent=2)

    print(f"\nComparison data saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="汇总分析所有评估结果"
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="../results",
        help="结果目录（默认：../results）"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="../results/comparison_report.json",
        help="保存对比报告的JSON文件"
    )

    args = parser.parse_args()

    results_dir = Path(args.output).parent.parent / Path(args.results_dir).name
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    # 收集所有结果
    print("Collecting evaluation results...")
    all_results = collect_all_results(results_dir)

    if not all_results:
        print(f"No evaluation results found in {results_dir}")
        print("Make sure you have run the batch evaluation first.")
        sys.exit(1)

    print(f"Found {len(all_results)} evaluation results\n")

    # 打印对比报告
    print_comparison_report(all_results)

    # 保存JSON
    output_file = Path(args.output)
    save_comparison_json(all_results, output_file)


if __name__ == "__main__":
    main()
