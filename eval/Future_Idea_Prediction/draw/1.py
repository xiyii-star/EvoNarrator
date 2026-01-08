#!/usr/bin/env python3
"""
匹配质量分布与核心案例高亮 (Score Distribution with Case Highlight)

目的：向审稿人展示系统的稳定性（大部分都在 5-6 分，不是瞎猜）和上限能力（能达到 8 分）。
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_evaluation_results(results_dir):
    """加载评估结果"""
    results_path = Path(results_dir) / "evaluation_results.json"
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def extract_scores(evaluation_results):
    """提取所有匹配分数"""
    scores = []
    highlight_case = None
    
    for result in evaluation_results:
        # 获取每个idea的最佳匹配分数
        max_score = result.get("max_score", 0)
        scores.append(max_score)
        
        # 查找 score 8.0 的案例用于高亮
        if max_score == 8.0 and highlight_case is None:
            best_match = result.get("best_match", {})
            highlight_case = {
                "idea_id": result.get("idea_id", ""),
                "score": max_score,
                "paper_title": best_match.get("paper_title", ""),
                "paper_year": best_match.get("paper_year", ""),
                "reason": best_match.get("reason", "")
            }
    
    return np.array(scores), highlight_case

def plot_score_distribution(scores, highlight_case=None, save_path=None):
    """绘制分数分布图，带有KDE和案例高亮"""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 绘制直方图
    n, bins, patches = ax.hist(scores, bins=20, range=(0, 10), 
                              alpha=0.7, color='steelblue', 
                              edgecolor='black', linewidth=1.2,
                              label='Score Distribution')
    
    # 计算并绘制KDE（核密度估计）
    if len(scores) > 1:
        # 使用高斯核密度估计
        kde = stats.gaussian_kde(scores)
        x_kde = np.linspace(0, 10, 200)
        y_kde = kde(x_kde)
        
        # 归一化KDE以匹配直方图的高度
        y_kde_normalized = y_kde * len(scores) * (bins[1] - bins[0])
        
        ax.plot(x_kde, y_kde_normalized, 'r-', linewidth=2.5, 
                label='Kernel Density Estimate (KDE)', alpha=0.8)
    
    # 高亮 score 8.0 的案例
    if highlight_case:
        score_8_x = highlight_case["score"]
        
        # 找到对应的直方图高度
        bin_idx = np.digitize([score_8_x], bins) - 1
        bin_idx = min(bin_idx[0], len(patches) - 1)
        bar_height = n[bin_idx] if bin_idx < len(n) else 0
        
        # 绘制箭头指向 score 8.0
        arrow_y = bar_height + max(n) * 0.15
        ax.annotate('', xy=(score_8_x, bar_height), 
                   xytext=(score_8_x, arrow_y),
                   arrowprops=dict(arrowstyle='->', lw=3, color='red', 
                                  connectionstyle='arc3,rad=0'))
        
        # 添加标注文本 - 使用用户指定的标注
        annotation_text = "Google DeepMind 2025 Match\nScore 8.0"
        ax.annotate(annotation_text,
                   xy=(score_8_x, bar_height),
                   xytext=(score_8_x + 1.5, arrow_y + max(n) * 0.1),
                   fontsize=11,
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow', 
                            edgecolor='red', linewidth=2.5, alpha=0.95),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color='red',
                                  connectionstyle='arc3,rad=0.2'))
    
    # 添加统计信息文本框
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    std_score = np.std(scores)
    
    stats_text = f'Mean: {mean_score:.2f}\nMedian: {median_score:.2f}\nStd: {std_score:.2f}\nN: {len(scores)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 设置标签和标题
    ax.set_xlabel('Match Score (0-10)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Ideas', fontsize=13, fontweight='bold')
    ax.set_title('Match Quality Distribution with Case Highlight\n' + 
                'System Stability (Most scores 5-6) & Upper Bound Capability (Score 8.0)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # 设置x轴范围
    ax.set_xlim(0, 10)
    ax.set_xticks(np.arange(0, 11, 1))
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # 添加分数区间标注
    # 计算各分数区间的数量
    score_ranges = {
        '0-2': np.sum((scores >= 0) & (scores < 3)),
        '3-4': np.sum((scores >= 3) & (scores < 5)),
        '5-6': np.sum((scores >= 5) & (scores < 7)),
        '7-8': np.sum((scores >= 7) & (scores < 9)),
        '9-10': np.sum(scores >= 9)
    }
    
    # 在图上添加分数区间统计
    range_text = "Score Ranges:\n"
    for range_name, count in score_ranges.items():
        range_text += f"{range_name}: {count}\n"
    
    ax.text(0.98, 0.98, range_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig, ax

def main():
    """主函数"""
    # 结果目录
    results_dir = Path("/home/lexy/下载/CLwithRAG/KGdemo/eval/Future_Idea_Prediction/results/research_ideas_Using_LLM_agent_to_generate_novel_and_original_research_ideas_without_human_participation_20251210_103343_results")
    
    # 输出目录
    output_dir = Path("/home/lexy/下载/CLwithRAG/KGdemo/eval/Future_Idea_Prediction/draw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading evaluation results...")
    evaluation_results = load_evaluation_results(results_dir)
    
    print("Extracting scores...")
    scores, highlight_case = extract_scores(evaluation_results)
    
    print(f"Total ideas: {len(scores)}")
    print(f"Score statistics:")
    print(f"  Mean: {np.mean(scores):.2f}")
    print(f"  Median: {np.median(scores):.2f}")
    print(f"  Std: {np.std(scores):.2f}")
    print(f"  Min: {np.min(scores):.2f}")
    print(f"  Max: {np.max(scores):.2f}")
    
    if highlight_case:
        print(f"\nHighlight case found:")
        print(f"  Idea ID: {highlight_case['idea_id']}")
        print(f"  Score: {highlight_case['score']}")
        print(f"  Paper: {highlight_case['paper_title']}")
        print(f"  Year: {highlight_case['paper_year']}")
    
    print("\nGenerating plot...")
    save_path = output_dir / "score_distribution_with_highlight.png"
    fig, ax = plot_score_distribution(scores, highlight_case, save_path)
    
    # 显示图片
    plt.show()
    
    print("\nDone!")

if __name__ == "__main__":
    main()

