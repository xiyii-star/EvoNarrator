#!/usr/bin/env python3
"""
统一分析所有评估结果中的ideas
将所有ideas汇聚在一起，进行平等的统计分析
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_all_evaluation_results(results_dir):
    """加载所有评估结果目录中的evaluation_results.json"""
    all_ideas = []
    results_base = Path(results_dir)
    
    # 遍历所有结果目录
    for result_dir in results_base.iterdir():
        if not result_dir.is_dir() or not result_dir.name.endswith('_results'):
            continue
        
        eval_file = result_dir / "evaluation_results.json"
        if not eval_file.exists():
            print(f"Warning: {eval_file} not found, skipping...")
            continue
        
        print(f"Loading: {result_dir.name}")
        with open(eval_file, 'r', encoding='utf-8') as f:
            eval_results = json.load(f)
        
        # 提取每个idea的信息
        for result in eval_results:
            idea_data = {
                'idea_id': result.get('idea_id', ''),
                'idea_text': result.get('idea_text', ''),
                'max_score': result.get('max_score', 0),
                'has_match': result.get('has_match', False),
                'best_match': result.get('best_match', {}),
                'source': result_dir.name  # 记录来源
            }
            all_ideas.append(idea_data)
    
    print(f"\nTotal ideas loaded: {len(all_ideas)}")
    return all_ideas

def calculate_unified_metrics(all_ideas, threshold=7.0):
    """计算统一的指标"""
    total_ideas = len(all_ideas)
    scores = [idea['max_score'] for idea in all_ideas]
    
    # 基本统计
    avg_score = np.mean(scores)
    median_score = np.median(scores)
    std_score = np.std(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    # 匹配统计
    matched_ideas = [idea for idea in all_ideas if idea['has_match']]
    match_count = len(matched_ideas)
    match_rate = match_count / total_ideas if total_ideas > 0 else 0
    
    # 分数分布
    score_distribution = {
        '0-2': sum(1 for s in scores if 0 <= s < 3),
        '3-4': sum(1 for s in scores if 3 <= s < 5),
        '5-6': sum(1 for s in scores if 5 <= s < 7),
        '7-8': sum(1 for s in scores if 7 <= s < 9),
        '9-10': sum(1 for s in scores if 9 <= s <= 10)
    }
    
    # 按年份统计
    year_stats = defaultdict(lambda: {'count': 0, 'scores': [], 'matched': 0})
    for idea in all_ideas:
        year = idea['best_match'].get('paper_year', 'unknown')
        year_stats[year]['count'] += 1
        year_stats[year]['scores'].append(idea['max_score'])
        if idea['has_match']:
            year_stats[year]['matched'] += 1
    
    year_metrics = {}
    for year, stats in year_stats.items():
        year_metrics[year] = {
            'total': stats['count'],
            'matched': stats['matched'],
            'hit_rate': stats['matched'] / stats['count'] if stats['count'] > 0 else 0,
            'avg_score': np.mean(stats['scores']) if stats['scores'] else 0,
            'min_score': np.min(stats['scores']) if stats['scores'] else 0,
            'max_score': np.max(stats['scores']) if stats['scores'] else 0
        }
    
    return {
        'total_ideas': total_ideas,
        'average_max_score': avg_score,
        'median_max_score': median_score,
        'std_max_score': std_score,
        'min_score': min_score,
        'max_score': max_score,
        'match_count': match_count,
        'match_rate': match_rate,
        'score_distribution': score_distribution,
        'year_metrics': year_metrics,
        'all_scores': scores
    }

def plot_unified_score_distribution(metrics, save_path=None):
    """图表1: 统一的分数分布（带KDE和高亮）"""
    scores = metrics['all_scores']
    
    # 增大图表尺寸以容纳更多内容
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 绘制直方图
    n, bins, patches = ax.hist(scores, bins=20, range=(0, 10), 
                              alpha=0.7, color='steelblue', 
                              edgecolor='black', linewidth=1.2,
                              label='Score Distribution')
    
    # 计算并绘制KDE
    if len(scores) > 1:
        from scipy import stats
        kde = stats.gaussian_kde(scores)
        x_kde = np.linspace(0, 10, 200)
        y_kde = kde(x_kde)
        y_kde_normalized = y_kde * len(scores) * (bins[1] - bins[0])
        ax.plot(x_kde, y_kde_normalized, 'r-', linewidth=2.5, 
                label='Kernel Density Estimate (KDE)', alpha=0.8)
    
    # 高亮最高分案例
    max_score = metrics['max_score']
    if max_score >= 7.0:
        bin_idx = np.digitize([max_score], bins) - 1
        bin_idx = min(bin_idx[0], len(patches) - 1)
        bar_height = n[bin_idx] if bin_idx < len(n) else 0
        
        arrow_y = bar_height + max(n) * 0.15
        ax.annotate('', xy=(max_score, bar_height), 
                   xytext=(max_score, arrow_y),
                   arrowprops=dict(arrowstyle='->', lw=3, color='red', 
                                  connectionstyle='arc3,rad=0'))
        
        annotation_text = f"Max Score: {max_score:.1f}"
        ax.annotate(annotation_text,
                   xy=(max_score, arrow_y),
                   xytext=(max_score + 1.5, arrow_y + max(n) * 0.1),
                   fontsize=13,
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow', 
                            edgecolor='red', linewidth=2.5, alpha=0.95),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color='red',
                                  connectionstyle='arc3,rad=0.2'))
    
    # 添加统计信息 - 移到左上角，避免与右上角重叠
    stats_text = f'Total Ideas: {metrics["total_ideas"]}\n'
    stats_text += f'Mean: {metrics["average_max_score"]:.2f}\n'
    stats_text += f'Median: {metrics["median_max_score"]:.2f}\n'
    stats_text += f'Std: {metrics["std_max_score"]:.2f}\n'
    stats_text += f'Match Rate: {metrics["match_rate"]*100:.1f}%'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=13, verticalalignment='top', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.9, 
                    edgecolor='black', linewidth=1.5))
    
    # 添加分数区间统计 - 移到右上角下方，避免与图例重叠
    dist = metrics['score_distribution']
    range_text = "Score Ranges:\n"
    for range_name in ['0-2', '3-4', '5-6', '7-8', '9-10']:
        count = dist[range_name]
        pct = count / metrics['total_ideas'] * 100 if metrics['total_ideas'] > 0 else 0
        range_text += f"{range_name}: {count:3d} ({pct:5.1f}%)\n"
    
    # 调整位置到右上角下方，避免与图例重叠
    ax.text(0.98, 0.75, range_text, transform=ax.transAxes,
           fontsize=12, verticalalignment='top', horizontalalignment='right',
           fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.9,
                    edgecolor='black', linewidth=1.5))
    
    # 增大所有标签和标题字体
    ax.set_xlabel('Match Score (0-10)', fontsize=15, fontweight='bold')
    ax.set_ylabel('Number of Ideas', fontsize=15, fontweight='bold')
    ax.set_title('Unified Score Distribution Across All Ideas\n' + 
                f'Total: {metrics["total_ideas"]} ideas from all evaluations', 
                fontsize=16, fontweight='bold', pad=25)
    ax.set_xlim(0, 10)
    ax.set_xticks(np.arange(0, 11, 1))
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 将图例移到右上角，但调整位置避免重叠
    legend = ax.legend(loc='upper right', fontsize=13, framealpha=0.95, 
                      bbox_to_anchor=(0.98, 0.98), frameon=True, fancybox=True)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.2)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig

def plot_score_distribution_pie(metrics, save_path=None):
    """图表2: 分数分布饼图"""
    dist = metrics['score_distribution']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    labels = list(dist.keys())
    sizes = list(dist.values())
    colors_map = {
        '0-2': '#d62728',
        '3-4': '#ff7f0e',
        '5-6': '#2ca02c',
        '7-8': '#1f77b4',
        '9-10': '#9467bd'
    }
    colors = [colors_map[label] for label in labels]
    
    # 只显示非零的部分
    non_zero_data = [(l, s) for l, s in zip(labels, sizes) if s > 0]
    if non_zero_data:
        labels, sizes = zip(*non_zero_data)
        colors = [colors_map[label] for label in labels]
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                      autopct='%1.1f%%', startangle=90,
                                      textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title(f'Score Distribution (Total: {metrics["total_ideas"]} ideas)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig

def plot_year_comparison(metrics, save_path=None):
    """图表3: 按年份的统计对比"""
    year_metrics = metrics['year_metrics']
    
    if not year_metrics:
        print("No year data available")
        return None
    
    years = sorted([y for y in year_metrics.keys() if y != 'unknown'])
    if not years:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Metrics by Paper Year', fontsize=16, fontweight='bold')
    
    # 子图1: 平均分数
    ax1 = axes[0, 0]
    avg_scores = [year_metrics[y]['avg_score'] for y in years]
    bars1 = ax1.bar(years, avg_scores, color='steelblue', edgecolor='black')
    ax1.set_ylabel('Average Score', fontweight='bold')
    ax1.set_title('Average Score by Year', fontweight='bold')
    ax1.set_ylim(0, 10)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, score in zip(bars1, avg_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 子图2: 匹配率
    ax2 = axes[0, 1]
    hit_rates = [year_metrics[y]['hit_rate'] * 100 for y in years]
    bars2 = ax2.bar(years, hit_rates, color='green', edgecolor='black')
    ax2.set_ylabel('Hit Rate (%)', fontweight='bold')
    ax2.set_title('Match Rate by Year', fontweight='bold')
    ax2.set_ylim(0, max(hit_rates) * 1.2 if hit_rates else 10)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, rate in zip(bars2, hit_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 子图3: Ideas数量
    ax3 = axes[1, 0]
    counts = [year_metrics[y]['total'] for y in years]
    bars3 = ax3.bar(years, counts, color='coral', edgecolor='black')
    ax3.set_ylabel('Number of Ideas', fontweight='bold')
    ax3.set_title('Ideas Count by Year', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, count in zip(bars3, counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 子图4: 分数范围
    ax4 = axes[1, 1]
    min_scores = [year_metrics[y]['min_score'] for y in years]
    max_scores = [year_metrics[y]['max_score'] for y in years]
    x_pos = np.arange(len(years))
    width = 0.35
    ax4.bar(x_pos - width/2, min_scores, width, label='Min Score', 
           color='lightcoral', edgecolor='black')
    ax4.bar(x_pos + width/2, max_scores, width, label='Max Score', 
           color='lightblue', edgecolor='black')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(years)
    ax4.set_ylabel('Score', fontweight='bold')
    ax4.set_title('Score Range by Year', fontweight='bold')
    ax4.set_ylim(0, 10)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig

def plot_comprehensive_summary(metrics, save_path=None):
    """图表4: 综合汇总信息"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # 创建文本摘要
    summary_text = f"""
UNIFIED EVALUATION SUMMARY
{'='*60}

Total Ideas Evaluated: {metrics['total_ideas']}

SCORE STATISTICS:
  Average Score: {metrics['average_max_score']:.2f}
  Median Score:  {metrics['median_max_score']:.2f}
  Std Deviation: {metrics['std_max_score']:.2f}
  Min Score:     {metrics['min_score']:.2f}
  Max Score:     {metrics['max_score']:.2f}

MATCH STATISTICS:
  Matched Ideas: {metrics['match_count']} / {metrics['total_ideas']}
  Match Rate:    {metrics['match_rate']*100:.2f}%

SCORE DISTRIBUTION:
"""
    
    dist = metrics['score_distribution']
    for range_name in ['0-2', '3-4', '5-6', '7-8', '9-10']:
        count = dist[range_name]
        pct = count / metrics['total_ideas'] * 100 if metrics['total_ideas'] > 0 else 0
        summary_text += f"  {range_name}: {count:3d} ({pct:5.1f}%)\n"
    
    if metrics['year_metrics']:
        summary_text += "\nYEAR-WISE METRICS:\n"
        for year in sorted([y for y in metrics['year_metrics'].keys() if y != 'unknown']):
            ym = metrics['year_metrics'][year]
            summary_text += f"  {year}: {ym['total']} ideas, "
            summary_text += f"Avg={ym['avg_score']:.2f}, "
            summary_text += f"Hit Rate={ym['hit_rate']*100:.1f}%\n"
    
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
           fontsize=11, family='monospace',
           verticalalignment='center', horizontalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=20))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig

def main():
    """主函数"""
    # 结果目录
    results_dir = Path("/home/lexy/下载/CLwithRAG/KGdemo/eval/Future_Idea_Prediction/results")
    
    # 输出目录
    output_dir = Path("/home/lexy/下载/CLwithRAG/KGdemo/eval/Future_Idea_Prediction/draw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("UNIFIED IDEAS ANALYSIS")
    print("="*60)
    print("\nLoading all evaluation results...")
    
    all_ideas = load_all_evaluation_results(results_dir)
    
    if not all_ideas:
        print("Error: No ideas found!")
        return
    
    print("\nCalculating unified metrics...")
    metrics = calculate_unified_metrics(all_ideas)
    
    print("\n" + "="*60)
    print("UNIFIED METRICS SUMMARY")
    print("="*60)
    print(f"Total Ideas: {metrics['total_ideas']}")
    print(f"Average Score: {metrics['average_max_score']:.2f}")
    print(f"Median Score: {metrics['median_max_score']:.2f}")
    print(f"Match Rate: {metrics['match_rate']*100:.2f}%")
    print(f"Score Range: {metrics['min_score']:.2f} - {metrics['max_score']:.2f}")
    print("\nScore Distribution:")
    for range_name, count in metrics['score_distribution'].items():
        pct = count / metrics['total_ideas'] * 100 if metrics['total_ideas'] > 0 else 0
        print(f"  {range_name}: {count} ({pct:.1f}%)")
    
    print("\nGenerating visualizations...")
    
    # 生成所有图表
    plot_unified_score_distribution(
        metrics,
        save_path=output_dir / "unified_score_distribution.png"
    )
    
    plot_score_distribution_pie(
        metrics,
        save_path=output_dir / "unified_score_pie.png"
    )
    
    if metrics['year_metrics']:
        plot_year_comparison(
            metrics,
            save_path=output_dir / "unified_year_comparison.png"
        )
    
    plot_comprehensive_summary(
        metrics,
        save_path=output_dir / "unified_summary.png"
    )
    
    print("\n" + "="*60)
    print("All visualizations generated successfully!")
    print(f"Output directory: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()

