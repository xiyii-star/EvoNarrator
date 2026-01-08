import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 数据准备
# ==========================================
labels = ['Novelty', 'Feasibility', 'Theoretical', 'Logical', 'Average']
num_vars = len(labels)

data = {
    'Eval: GPT-4o': {
        'CoI':       [4.73, 4.63, 4.35, 4.38, 4.52],
        'MOOSEChem': [4.90, 4.70, 4.40, 4.40, 4.60],
        'NaiveLLM':  [4.80, 4.60, 4.43, 4.40, 4.56],
        'w/o Path':  [4.77, 4.70, 4.80, 4.76, 4.75],
        'Ours':      [4.81, 4.66, 4.86, 4.86, 4.80]
    },
    'Eval: Deepseek-v3.2': {
        'CoI':       [4.77, 4.60, 4.40, 4.40, 4.54],
        'MOOSEChem': [4.80, 4.70, 4.40, 4.40, 4.58],
        'NaiveLLM':  [4.76, 4.57, 4.37, 4.37, 4.52],
        'w/o Path':  [4.66, 4.64, 4.70, 4.78, 4.69],
        'Ours':      [4.81, 4.59, 4.84, 4.86, 4.78]
    },
    'Eval: GPT-3.5-turbo': {
        'CoI':       [4.40, 4.43, 4.40, 4.40, 4.41],
        'MOOSEChem': [4.60, 4.50, 4.40, 4.40, 4.48],
        'NaiveLLM':  [4.66, 4.54, 4.40, 4.40, 4.50],
        'w/o Path':  [4.64, 4.60, 4.70, 4.66, 4.65],
        'Ours':      [4.75, 4.67, 4.88, 4.88, 4.80]
    },
    'Eval: Human': {
        'CoI':       [4.25, 3.75, 3.25, 3.75, 3.81],
        'MOOSEChem': [4.50, 4.00, 4.00, 4.00, 4.13],
        'NaiveLLM':  [4.27, 3.81, 3.27, 3.55, 3.70],
        'w/o Path':  [4.60, 4.00, 4.10, 4.10, 4.20],
        'Ours':      [4.61, 4.06, 4.70, 4.80, 4.54]
    }
}

# ==========================================
# 2. 样式升级：颜色 + 线型 + 标记点 (Marker)
# ==========================================
# 颜色选取了高对比度的 Tableau 调色盘
# 格式: 'Name': {color, linewidth, linestyle, marker, markersize, zorder}
styles = {
    'Ours': {
        'c': '#D62728', 'lw': 2.5, 'ls': '-', 'm': 'o', 'ms': 6, 'z': 10, 'label': 'Ours (Proposed)'
    }, # 红色，实心圆点，最粗，最上层
    
    'w/o Path': {
        'c': '#1F77B4', 'lw': 1.8, 'ls': '--', 'm': 's', 'ms': 5, 'z': 5, 'label': 'w/o Path'
    }, # 蓝色，虚线，方块点
    
    'MOOSEChem': {
        'c': '#2CA02C', 'lw': 1.8, 'ls': '-.', 'm': '^', 'ms': 5, 'z': 4, 'label': 'MOOSEChem'
    }, # 绿色，点划线，三角点
    
    'NaiveLLM': {
        'c': '#9467BD', 'lw': 1.8, 'ls': ':', 'm': 'D', 'ms': 4, 'z': 3, 'label': 'NaiveLLM'
    }, # 紫色，点线，菱形点
    
    'CoI': {
        'c': '#7F7F7F', 'lw': 1.5, 'ls': '-', 'm': 'x', 'ms': 5, 'z': 2, 'label': 'CoI'
    }, # 灰色，细实线，叉号
}

# 角度设置
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1] # 闭合

# ==========================================
# 3. 绘图主逻辑
# ==========================================
# 调整 figsize，留出右侧空间给图例
fig = plt.figure(figsize=(22, 5)) 

# 使用 GridSpec 布局：左边放4个图，右边留一小块给图例
# width_ratios=[1, 1, 1, 1, 0.4] 意味着前4列等宽，第5列窄一点专门放图例
gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.2])
axs = [fig.add_subplot(gs[0, i], polar=True) for i in range(4)]

evaluators = list(data.keys())

for idx, ax in enumerate(axs):
    eval_name = evaluators[idx]
    eval_data = data[eval_name]
    
    # --- 绘制线条 ---
    for method, style in styles.items():
        if method not in eval_data: continue
        
        scores = eval_data[method]
        values = scores + scores[:1] # 闭合
        
        # 绘图核心
        ax.plot(angles, values, 
                color=style['c'], 
                linewidth=style['lw'], 
                linestyle=style['ls'],
                marker=style['m'],      # 【关键改进】添加标记点
                markersize=style['ms'], # 标记点大小
                label=style['label'], 
                zorder=style['z'],
                alpha=0.9)              # 线条稍微不透明一点
        
        # 仅给 Ours 填充颜色，强调面积优势
        if method == 'Ours':
            ax.fill(angles, values, color=style['c'], alpha=0.15, zorder=style['z']-1)

    # --- 坐标轴美化 ---
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # 标签字体加黑加粗
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10.5, weight='bold', color='#333333')
    
    # Y轴范围设置 (3.0 - 5.0)
    ax.set_ylim(3.0, 5.0)
    
    # 自定义Y轴网格线
    ax.set_rlabel_position(0)
    # 只显示几个关键刻度，减少杂乱
    plt.setp(ax.get_yticklabels(), visible=False) # 隐藏默认刻度字
    ax.set_yticks([3.5, 4.0, 4.5, 5.0])           # 设置刻度线位置
    ax.grid(color='#AAAAAA', linestyle='--', linewidth=0.5, alpha=0.4) # 淡化网格
    
    # 仅在第一个图显示Y轴刻度值，避免重复
    if idx == 0:
        ax.text(0, 3.5, '3.5', color='gray', fontsize=8, ha='center', va='center', zorder=20)
        ax.text(0, 4.0, '4.0', color='gray', fontsize=8, ha='center', va='center', zorder=20)
        ax.text(0, 4.5, '4.5', color='gray', fontsize=8, ha='center', va='center', zorder=20)

    # 标题
    ax.set_title(eval_name, size=15, weight='bold', pad=25, y=1.05)
    
    # 边框颜色
    ax.spines['polar'].set_color('#888888')

# ==========================================
# 4. 独立的侧边图例 (Right Side Legend)
# ==========================================
# 在最右侧创建一个空白的 axes 专门放图例
ax_legend = fig.add_subplot(gs[0, 4])
ax_legend.axis('off') # 隐藏坐标轴

# 获取第一个图的句柄用于生成图例
handles, lbls = axs[0].get_legend_handles_labels()

# 绘制图例
legend = ax_legend.legend(handles, lbls, 
                          loc='center left', 
                          title="Methods",
                          title_fontsize=12,
                          fontsize=11, 
                          frameon=False, # 去掉图例边框，显得更干净
                          labelspacing=1.2) # 增加行间距

# 将图例标题设为粗体
plt.setp(legend.get_title(), weight='bold')

plt.tight_layout()

# 保存
plt.savefig('radar_chart_improved.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('radar_chart_improved.png', format='png', bbox_inches='tight', dpi=300)

plt.show()
