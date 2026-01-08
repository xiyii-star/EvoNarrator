# 引用关系分类评估实验

本评估框架用于比较两种引用关系分类方法的性能：

1. **Baseline**: Zero-shot LLM 分类（仅使用 Abstract）
2. **SocketMatch**: LLM + DeepPaper 深度信息 + SocketMatch 逻辑

## 📋 目录结构

```
eval/citation_eval/
├── data/
│   ├── golden_citation_dataset.xlsx    # 标注数据集（230条引用关系）
│   └── paper_pdf/                      # 论文PDF文件（可选）
├── src/
│   ├── baseline_evaluator.py          # Baseline 评估器
│   ├── socketmatch_evaluator.py       # SocketMatch 评估器
│   └── metrics_calculator.py          # 评估指标计算与美观输出
├── prompts/
│   ├── baseline_classification.txt     # Baseline 提示词模板
│   ├── match_limitation_problem.txt    # Match 1: Limitation→Problem
│   ├── match_future_work_problem.txt   # Match 2: Future_Work→Problem
│   ├── match_method_extension.txt      # Match 3: Method Extension
│   └── match_problem_adaptation.txt    # Match 4: Problem Adaptation
├── results/                             # 评估结果输出目录
│   └── YYYYMMDD_HHMMSS/                # 每次运行的时间戳目录
│       ├── baseline_predictions.json
│       ├── baseline_report.json
│       ├── socketmatch_predictions.json
│       ├── socketmatch_report.json
│       ├── comparison.json
│       └── tables/                     # 论文表格（可选生成）
├── run_evaluation.py                    # 主评估脚本
├── regenerate_report.py                 # 重新生成美观报告
├── generate_tables.py                   # 生成论文表格
├── README.md                            # 本文档
└── TABLE_GENERATION.md                  # 表格生成指南
```

## 📊 数据集格式

数据集文件：`data/golden_citation_dataset.xlsx`（**230 条标注的引用关系**）

### Excel 列说明

**引用论文（Citing Paper）字段：**

- `citing_paper_id`: 论文 ID
- `citing_paper_title`: 论文标题
- `citing_paper_abstract`: 论文摘要
- `citing_paper_rag_problem`: 问题陈述（DeepPaper 提取）
- `citing_paper_rag_contribution`: 贡献/方法（DeepPaper 提取）
- `citing_paper_rag_limitation`: 局限性（DeepPaper 提取）
- `citing_paper_rag_future_work`: 未来工作（DeepPaper 提取）

**被引用论文（Cited Paper）字段：**

- `cited_paper_*`: 与引用论文相同的字段结构

**标签：**

- `edge_type`: 真实标签（Ground Truth）

### 支持的引用关系类型（6种）

| 类型                  | 中文名称  | 描述                           | Socket 检测方式                     |
| --------------------- | --------- | ------------------------------ | ----------------------------------- |
| **Overcomes**   | 攻克/优化 | B 解决了 A 的局限性            | Match 1: A.Limitation ↔ B.Problem  |
| **Realizes**    | 实现愿景  | B 实现了 A 的未来工作建议      | Match 2: A.Future_Work ↔ B.Problem |
| **Extends**     | 方法扩展  | B 扩展/改进了 A 的方法         | Match 3: Method Extension           |
| **Alternative** | 另辟蹊径  | B 提供了解决相同问题的替代方法 | Match 3: Method Alternative         |
| **Adapts_to**   | 技术迁移  | B 将 A 的方法迁移到新领域      | Match 4: Problem 跨域               |
| **Baselines**   | 基线对比  | B 将 A 作为基线/背景           | 无匹配时的默认类型                  |

## 环境配置

### 1. 安装依赖

```bash
pip install openpyxl numpy pyyaml openai
```

### 2. 配置 LLM

编辑 `config/config.yaml` 文件：

```yaml
llm:
  provider: "openai"  # 或 "anthropic", "local"
  model: "gpt-4o-mini"
  api_key: "your-api-key"
  base_url: "https://api.openai.com/v1"  # 可选
  temperature: 0.3
  max_tokens: 500
```

## 运行评估

### 完整评估（两种方法）

```bash
cd /home/lexy/下载/CLwithRAG/KGdemo
python eval/citation_eval/run_evaluation.py
```

### 仅运行 Baseline

```bash
python eval/citation_eval/run_evaluation.py --method baseline
```

### 仅运行 SocketMatch

```bash
python eval/citation_eval/run_evaluation.py --method socketmatch
```

### 快速测试（采样 10 条数据）

```bash
python eval/citation_eval/run_evaluation.py --sample-size 10
```

### 自定义参数

```bash
python eval/citation_eval/run_evaluation.py \
    --data eval/citation_eval/data/golden_citation_dataset.xlsx \
    --config config/config.yaml \
    --output eval/citation_eval/results \
    --method both \
    --sample-size 50
```

## 输出结果

评估结果保存在 `results/YYYYMMDD_HHMMSS/` 目录下：

### 1. Baseline 结果

- `baseline_predictions.json`: 预测结果
- `baseline_report.json`: 评估报告

### 2. SocketMatch 结果

- `socketmatch_predictions.json`: 预测结果
- `socketmatch_report.json`: 评估报告

### 3. 比较结果

- `comparison.json`: 方法性能比较
- `improvement.json`: 性能提升统计

### 4. 日志

- `evaluation.log`: 详细运行日志

### 📊 输出格式说明

评估结果采用美观的表格格式输出，包含：

- ✅ 整体性能指标（Accuracy, Macro F1, Weighted F1）
- 📋 各类别详细指标（Precision, Recall, F1, Support）
- 📊 类别分布对比
- 🔬 方法性能对比分析
- 💡 指标含义说明

📄 生成论文表格

评估完成后，可以生成论文所需的表格文件（LaTeX, CSV, Markdown）：

```bash
python eval/citation_eval/generate_tables.py <结果目录>
```

这将生成：

- **LaTeX 表格**: 直接用于论文排版（table1_main_results.tex, table2_classwise_f1.tex）
- **CSV 表格**: 便于 Excel 查看和编辑
- **Markdown 表格**: 适合 GitHub README

**详细的表格生成指南，请参考 [TABLE_GENERATION.md](TABLE_GENERATION.md)**

## 评估指标

### 整体指标

- **Accuracy**: 准确率（整体预测正确的比例）
- **Macro F1**: 宏平均 F1 分数（各类别 F1 的平均值）
- **Weighted F1**: 加权平均 F1 分数（按类别样本量加权）

### 每类别指标

- **Precision**: 精确率（预测为该类别中真正属于该类别的比例）
- **Recall**: 召回率（真实为该类别中被正确预测的比例）
- **F1-Score**: F1 分数（Precision 和 Recall 的调和平均）
- **Support**: 支持数（该类别在真实标签中的样本数）

## 方法说明

### Baseline 方法

- **输入**: 仅使用论文的 Abstract（摘要）
- **方法**: Zero-shot LLM 分类
- **优点**: 简单直接，无需额外信息提取
- **缺点**: 信息有限，可能无法捕捉深层关系

### SocketMatch 方法

- **输入**: DeepPaper 提取的深度信息（Problem, Contribution, Limitation, Future Work）
- **方法**:
  1. Socket Matching: 检查 4 个匹配模式
     - Match 1: A.Limitation ↔ B.Problem
     - Match 2: A.Future_Work ↔ B.Problem
     - Match 3: A.Method ↔ B.Method
     - Match 4: A.Problem ↔ B.Problem（跨域）
  2. 基于优先级的决策树分类
- **优点**:
  - 深度语义理解
  - 逻辑清晰的分类规则
  - 考虑论文间的多维度关系
- **缺点**: 依赖深度信息提取质量

## 示例输出

```
================================================================================
评估报告: SocketMatch (Deep Info + Logic)
================================================================================

【整体指标】 (样本数: 230)
  Accuracy:    0.7826
  Macro F1:    0.7234
  Weighted F1: 0.7689

【每类别指标】
类别             Precision    Recall       F1-Score     Support
--------------------------------------------------------------------------------
Overcomes        0.8500       0.8095       0.8293       42
Realizes         0.7647       0.7222       0.7429       36
Extends          0.7200       0.7826       0.7500       46
Alternative      0.6923       0.6429       0.6667       28
Adapts_to        0.7778       0.7000       0.7368       30
Baselines        0.8571       0.9000       0.8780       48

【类别分布】
真实标签分布:
  Baselines      :   48 ( 20.9%)
  Extends        :   46 ( 20.0%)
  Overcomes      :   42 ( 18.3%)
  ...
```

## 故障排除

### 1. 导入错误

确保已安装所有依赖包：

```bash
pip install openpyxl numpy pyyaml openai
```

### 2. LLM 连接失败

- 检查 `config/config.yaml` 中的 API Key 是否正确
- 检查网络连接
- 尝试使用代理或更换 API endpoint

### 3. SocketMatch 全部预测为 Baselines ⚠️

**症状**：

- SocketMatch 方法的所有预测都是 `Baselines`
- 其他 5 种类别的 F1 分数都为 0
- Macro F1 异常低（~0.14）

**原因**：提示词文件路径不正确，导致所有 Socket Match 检测被跳过

**诊断方法**：
查看日志中是否有以下警告：

```
WARNING - 提示词文件不存在: match_limitation_problem.txt
WARNING - 缺少 match_limitation_problem 提示词，跳过
INFO - 无匹配结果 -> Baselines
```

**解决方法**：

✅ **已在代码中修复**（2025-12-10，[run_evaluation.py:162](run_evaluation.py#L162)）

修复后，重新运行评估：

```bash
python eval/citation_eval/run_evaluation.py --method socketmatch --sample-size 10
```

你应该看到正确的日志：

```
INFO - 加载 5 个提示词模板
INFO - CitationTypeInferencer 初始化完成
INFO -   模式: LLM Socket Matching
INFO -     → Match 1: 不匹配 (is_match=False)
INFO -     → Match 2 具体性检查: ✓ 高具体性 (specificity=high, conf=0.90)
INFO -   ✓ Match 2 (Future_Work→Problem) 匹配成功 -> Realizes (置信度: 0.80)
```

**如果仍然遇到此问题**：

```bash
# 检查提示词文件是否存在
ls -la eval/citation_eval/prompts/match_*.txt

# 如果不存在，从项目根目录复制
cp prompts/match_*.txt eval/citation_eval/prompts/
```

详细的问题诊断和修复过程请参考：[BUGFIX_REPORT.md](BUGFIX_REPORT.md)（如果存在）

### 4. 内存不足

使用 `--sample-size` 参数减少数据量：

```bash
python eval/citation_eval/run_evaluation.py --sample-size 50
```

### 5. 查看美观的评估报告

如果运行时没看到美观的表格输出，可以从已保存的结果重新生成：

```bash
python eval/citation_eval/regenerate_report.py eval/citation_eval/results/20251210_114707
```

## 扩展建议

1. **添加更多 Baseline 方法**

   - 传统机器学习方法（SVM, Random Forest）
   - BERT-based 分类器
2. **优化 SocketMatch 逻辑**

   - 调整匹配阈值
   - 添加更多匹配规则
3. **分析错误案例**

   - 查看预测结果 JSON 文件
   - 分析常见错误模式
4. **可视化结果**

   - 混淆矩阵热力图
   - 每类别 F1 分数柱状图

## 📚 相关文档

- **[TABLE_GENERATION.md](TABLE_GENERATION.md)**: 论文表格生成完整指南
- **[BUGFIX_REPORT.md](BUGFIX_REPORT.md)**: 已知问题和修复报告（如果存在）
- **项目根目录**: DeepPaper 系统整体说明

## 🔧 核心文件说明

### 评估脚本

- **[run_evaluation.py](run_evaluation.py)**: 主评估脚本，协调两种方法的评估流程
- **[regenerate_report.py](regenerate_report.py)**: 从已保存的 JSON 文件重新生成美观的终端报告
- **[generate_tables.py](generate_tables.py)**: 生成论文所需的表格文件（LaTeX/CSV/Markdown）

### 评估器

- **[src/baseline_evaluator.py](src/baseline_evaluator.py)**: Baseline 方法实现，使用 Zero-shot LLM
- **[src/socketmatch_evaluator.py](src/socketmatch_evaluator.py)**: SocketMatch 方法实现，封装 CitationTypeInferencer
- **[src/metrics_calculator.py](src/metrics_calculator.py)**: 评估指标计算和美观的表格输出

### 提示词模板

- **[prompts/baseline_classification.txt](prompts/baseline_classification.txt)**: Baseline 分类提示词
- **[prompts/match_limitation_problem.txt](prompts/match_limitation_problem.txt)**: Match 1 检测提示词
- **[prompts/match_future_work_problem.txt](prompts/match_future_work_problem.txt)**: Match 2 检测提示词
- **[prompts/match_method_extension.txt](prompts/match_method_extension.txt)**: Match 3 检测提示词
- **[prompts/match_problem_adaptation.txt](prompts/match_problem_adaptation.txt)**: Match 4 检测提示词

## 📊 完整工作流程示例

```bash
# 1. 运行完整评估
python eval/citation_eval/run_evaluation.py --method both

# 输出：
# ✅ Baseline 评估完成
# ✅ SocketMatch 评估完成
# ✅ 性能对比完成
# 📁 结果保存到: eval/citation_eval/results/20251210_150000/

# 2. 查看美观的报告（如果需要）
python eval/citation_eval/regenerate_report.py eval/citation_eval/results/20251210_150000

# 3. 生成论文表格
python eval/citation_eval/generate_tables.py eval/citation_eval/results/20251210_150000

# 输出：
# ✅ 所有表格已生成到: eval/citation_eval/results/20251210_150000/tables/
#    • LaTeX 表格: .../latex/
#    • CSV 表格: .../csv/
#    • Markdown 表格: .../tables.md

# 4. 在 LaTeX 论文中使用
# 复制 tables/latex/*.tex 到论文目录，然后引用：
# \input{table1_main_results.tex}
```

## 联系方式

如有问题，请联系项目维护者。

---

**最后更新**: 2025-12-10
**版本**: v1.1
**状态**: ✅ 已测试并验证（路径问题已修复）
