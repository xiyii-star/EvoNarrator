"""
绘制 Pairwise Win-Rate 对比结果的可视化图表
Style: High Contrast, Publication Ready, Big Fonts, Zoomed Y-Axis
"""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
import argparse
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# 配置字体：优先尝试无衬线字体，兼容中文
import matplotlib
try:
    # 尝试设置 Arial 或 Helvetica (学术常用)，如果失败则回退
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'SimHei', 'DejaVu Sans']
except:
    pass
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_round_progression(results: dict, output_file: str):
    """
    绘制累积胜率趋势图 (紧凑视图，大字体)
    """
    # --- 1. 数据处理 ---
    all_comparisons = []
    if 'round_results' in results:
        for round_data in results['round_results']:
            if 'comparisons' in round_data:
                all_comparisons.extend(round_data['comparisons'])
    
    if not all_comparisons:
        logger.warning("数据缺失，无法绘图")
        return

    x_indices = []
    y_win_rates = []
    cumulative_wins = 0
    
    for i, comp in enumerate(all_comparisons):
        count = i + 1
        if comp.get('winner') == 'A':
            cumulative_wins += 1
        win_rate = (cumulative_wins / count) * 100
        x_indices.append(count)
        y_win_rates.append(win_rate)

    # --- 2. 绘图初始化 ---
    # 使用较宽的比例
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    
    # 颜色定义
    COLOR_MAIN = '#1f77b4'  # 经典的Matplotlib蓝，深沉且清晰
    COLOR_BASELINE = '#555555' # 深灰基准线

    # --- 3. 绘制主曲线 ---
    # 标记点密度控制：避免点挤在一起
    marker_interval = max(1, len(x_indices) // 15)
    
    ax.plot(x_indices, y_win_rates, 
            color=COLOR_MAIN, 
            linewidth=4,          # 线条加粗
            marker='o', 
            markersize=9,         # 标记变大
            markevery=marker_interval,
            label='Ours')

    # --- 4. 坐标轴与范围设置 (核心修改) ---
    
    # Y轴范围：只显示 40% 到 102% (切除底部空白)
    # 如果数据中有低于40的情况，自动调整下限
    min_val = min(y_win_rates)
    lower_bound = 40 if min_val > 42 else max(0, min_val - 5)
    ax.set_ylim(lower_bound, 103)
    
    # X轴范围
    ax.set_xlim(0, len(x_indices) * 1.02)

    # 刻度字体暴力加大
    ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=6)
    
    # 设置Y轴格式为百分比 (可选，如果不想显示%符号可注释掉这行)
    # ax.yaxis.set_major_formatter(ticker.PercentFormatter())

    # --- 5. 辅助元素 ---
    
    # 50% 基准线
    ax.axhline(y=50, color=COLOR_BASELINE, linestyle='--', alpha=0.6, linewidth=2)
    # 基准线文字
    ax.text(0.5, 50 + 1.5, 'Baseline (50%)', color=COLOR_BASELINE, fontsize=14, alpha=0.8)

    # 尾部数值标注 (加大加粗)
    final_rate = y_win_rates[-1]
    ax.text(x_indices[-1] + (len(x_indices) * 0.01), final_rate, 
            f'{final_rate:.1f}%', 
            color=COLOR_MAIN, 
            fontweight='bold', 
            fontsize=22,         # 字体加大到22
            va='center')

    # --- 6. 标签与标题 (加大) ---
    
    method_b_name = results.get('method_b_name', 'Baseline')
    ax.set_title(f"Cumulative Win-Rate: Ours vs {method_b_name}", 
                 fontsize=24, fontweight='bold', pad=25)
    
    ax.set_xlabel("Number of Comparisons", fontsize=20, fontweight='bold', labelpad=10)
    ax.set_ylabel("Win Rate (%)", fontsize=20, fontweight='bold', labelpad=10)
    
    # --- 7. 美化边框 ---
    
    # 去除上方和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 加粗左侧和下方边框
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # 图例
    ax.legend(loc='lower right', frameon=True, fontsize=16, framealpha=0.9, edgecolor='gray')

    # --- 8. 保存 ---
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    logger.info(f"✅ 优化版趋势图已保存: {output_file}")
    plt.close()


# -------------------------------------------------------------------------
# 下面是其他辅助绘图函数 (保持简单兼容)
# -------------------------------------------------------------------------

def plot_win_rate_bar(results: dict, output_file: str):
    method_a = "Ours"
    method_b = results['method_b_name']
    win_rates = [results['win_rate_a'], results['win_rate_b'], results['tie_rate']]
    labels = [method_a, method_b, 'Tie']
    colors = ['#1f77b4', '#aaaaaa', '#d3d3d3']

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, win_rates, color=colors, width=0.6)
    
    ax.set_ylim(0, 100)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylabel("Win Rate (%)", fontsize=16, fontweight='bold')
    ax.set_title(f"{method_a} vs {method_b}", fontsize=18, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, rate in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{rate:.1f}%', ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Input JSON file')
    parser.add_argument('-t', '--type', type=str, default='trend', help='Chart type: trend/bar')
    args = parser.parse_args()

    if not Path(args.input).exists():
        print("File not found.")
        return

    with open(args.input, 'r', encoding='utf-8') as f:
        results = json.load(f)

    output_dir = Path(args.input).parent
    stem = Path(args.input).stem
    
    # 默认绘制趋势图
    if args.type == 'trend' or args.type == 'all':
        plot_round_progression(results, str(output_dir / f"{stem}_trend.png"))
    
    if args.type == 'bar' or args.type == 'all':
        plot_win_rate_bar(results, str(output_dir / f"{stem}_bar.png"))

if __name__ == "__main__":
    main()
