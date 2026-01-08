#!/usr/bin/env python3
"""
细粒度匹配能力雷达图 (Fine-Grained Capability Radar)

根据 comparison_report.json 绘制雷达图，展示不同评估结果或最佳评估结果的细粒度匹配能力。
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_comparison_report(comparison_file):
    """加载比较报告"""
    with open(comparison_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_evaluation_results(results_path):
    """加载单个评估结果"""
    results_path = Path(results_path)
    eval_path = results_path / "evaluation_results.json"
    
    if not eval_path.exists():
        print(f"Warning: Evaluation results not found: {eval_path}")
        return None
    
    with open(eval_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def extract_top_ideas(evaluation_results, top_n=3):
    """提取前N个高分Idea的细粒度匹配数据"""
    if not evaluation_results:
        return []
    
    # 按max_score排序，选择前N个
    sorted_ideas = sorted(evaluation_results, 
                         key=lambda x: x.get("max_score", 0), 
                         reverse=True)
    
    top_ideas = []
    for idea in sorted_ideas[:top_n]:
        best_match = idea.get("best_match", {})
        idea_data = {
            "idea_id": idea.get("idea_id", ""),
            "problem_consistency": best_match.get("problem_consistency", 0),
            "method_similarity": best_match.get("method_similarity", 0),
            "application_similarity": best_match.get("application_similarity", 0),
            "overall_score": best_match.get("match_score", 0),
            "paper_title": best_match.get("paper_title", ""),
            "paper_year": best_match.get("paper_year", "")
        }
        top_ideas.append(idea_data)
    
    return top_ideas

def plot_fine_grained_radar(top_ideas, save_path=None, title_suffix=""):
    """绘制细粒度匹配能力雷达图"""
    
    if not top_ideas:
        print("No ideas to plot!")
        return None, None
    
    # 定义类别
    categories = ['Problem\nConsistency', 'Method\nSimilarity', 
                   'Application\nSimilarity', 'Overall\nScore']
    N = len(categories)
    
    # 计算角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合曲线
    
    # 创建图形（更大的尺寸以容纳更多Idea）
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    
    # 定义颜色和样式（支持至少5个Idea）
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22', '#16A085']
    linestyles = ['solid', 'dashed', 'dotted', 'dashdot', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (1, 1)), (0, (3, 5, 1, 5))]
    
    # 为每个Idea绘制雷达图
    for idx, idea in enumerate(top_ideas):
        # 提取数据
        values = [
            idea['problem_consistency'],
            idea['method_similarity'],
            idea['application_similarity'],
            idea['overall_score']
        ]
        values += values[:1]  # 闭合曲线
        
        # 生成标签
        idea_label = f"Idea #{idea['idea_id'].split('_')[1]}"
        # 简化论文标题作为副标签
        paper_title = idea['paper_title']
        if len(paper_title) > 35:
            paper_title = paper_title[:32] + "..."
        label = f"{idea_label}"
        
        # 绘制线条
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        ax.plot(angles, values, linewidth=2.5, linestyle=linestyle, 
               label=label, color=color)
        ax.fill(angles, values, color, alpha=0.15)
    
    # 设置标签和刻度（放大四个方向的标签）
    plt.xticks(angles[:-1], categories, color='black', size=18, weight='bold')
    ax.set_rlabel_position(0)
    plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], 
               color="grey", size=10)
    plt.ylim(0, 10)
    
    # 设置标题
    title = "Fine-grained Alignment Analysis of Top Generated Ideas"
    if title_suffix:
        title += f"\n{title_suffix}"
    plt.title(title, size=16, weight='bold', y=1.15, pad=20)
    
    # 设置图例（放大右上角的标注）
    plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), 
              fontsize=19, framealpha=0.9, ncol=1)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig, ax

def main():
    """主函数"""
    # 比较报告文件
    comparison_file = Path("/home/lexy/下载/CLwithRAG/KGdemo/eval/Future_Idea_Prediction/results/comparison_report.json")
    
    # 输出目录
    output_dir = Path("/home/lexy/下载/CLwithRAG/KGdemo/eval/Future_Idea_Prediction/draw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading comparison report...")
    comparison_data = load_comparison_report(comparison_file)
    
    # 找到平均分最高的评估结果
    best_eval = max(comparison_data['evaluations'], 
                   key=lambda x: x.get('average_max_score', 0))
    
    print(f"\nBest evaluation: {best_eval['name']}")
    print(f"Average max score: {best_eval['average_max_score']}")
    
    # 加载最佳评估结果的详细数据
    # comparison_report.json 在 results 目录下，所以需要找到 results 目录
    results_base = comparison_file.parent  # results 目录
    eval_path_str = best_eval['path']
    
    # 处理相对路径 ../results/xxx -> xxx
    if eval_path_str.startswith('../results/'):
        eval_dir_name = eval_path_str.replace('../results/', '')
    else:
        eval_dir_name = eval_path_str
    
    results_path = results_base / eval_dir_name
    
    print(f"\nLoading evaluation results from: {results_path}")
    evaluation_results = load_evaluation_results(results_path)
    
    if not evaluation_results:
        print("Error: Could not load evaluation results!")
        return
    
    print("Extracting top ideas...")
    top_ideas = extract_top_ideas(evaluation_results, top_n=5)
    
    print(f"\nTop {len(top_ideas)} ideas selected:")
    for idea in top_ideas:
        print(f"\n{idea['idea_id']}:")
        print(f"  Problem Consistency: {idea['problem_consistency']}")
        print(f"  Method Similarity: {idea['method_similarity']}")
        print(f"  Application Similarity: {idea['application_similarity']}")
        print(f"  Overall Score: {idea['overall_score']}")
        print(f"  Paper: {idea['paper_title']} ({idea['paper_year']})")
    
    print("\nGenerating radar chart...")
    save_path = output_dir / "fine_grained_radar.png"
    title_suffix = f"{best_eval['name']} (Avg Score: {best_eval['average_max_score']:.2f})"
    fig, ax = plot_fine_grained_radar(top_ideas, save_path, title_suffix)
    
    if fig:
        # 关闭图形以释放内存
        plt.close(fig)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
