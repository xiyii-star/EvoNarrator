# 消融实验 (Ablation Studies)

## 概述

本目录包含 DeepPaper 2.0 系统的完整消融实验代码，用于验证架构中各个关键组件的有效性。

## 三个消融实验

### 1. Ablation 1: Reflection Loop (CriticAgent) 的有效性

**目的**: 证明迭代优化机制对提取质量的提升作用

**实验设置**:
- **Variant A**: DeepPaper 2.0 (Full) - 使用CriticAgent进行迭代优化
- **Variant B**: DeepPaper 2.0 w/o CriticAgent - 单次提取，不进行迭代

**关注指标**:
- Precision (精确度)
- Format Compliance (格式符合度)

**预期结果**:
- 去掉Critic后，幻觉率上升，提取内容变得宽泛（Generic），不再具体
- 随着迭代次数增加，质量呈阶梯式上升

**图表类型**: 折线图
- X轴: 迭代轮数 (0, 1, 2, 3)
- Y轴: 提取质量得分

---

### 2. Ablation 2: LogicAnalystAgent vs. 简单提取

**目的**: 证明因果推理对Problem-Method配对准确性的提升

**实验设置**:
- **Variant A**: 使用LogicAnalystAgent (寻找因果链)
- **Variant B**: 使用普通的Summarization Prompt

**关注指标**:
- Pairing Accuracy (问题和解法是否对应)

**预期结果**:
- 普通提取容易把Problem A和Solution B混在一起
- LogicAnalyst能精准对齐

**图表类型**: 柱状图
- 对比两种方法的Pairing Accuracy

---

### 3. Ablation 3: Citation Detective (双流合并) 的增益

**目的**: 证明引用分析能发现作者未承认的隐式局限性

**实验设置**:
- **Variant A**: Full System (Section + Citation)
- **Variant B**: Only Section Analysis

**关注指标**:
- Limitation Recall (特别是Implicit Limitation)

**预期结果**:
- 对于高引用论文，Variant A能发现作者自己没承认但同行指出的缺点

**图表类型**: 堆叠柱状图
- 柱子分为"Self-Reported" (作者说的) 和 "Peer-Review" (引用挖掘的)
- DeepPaper 2.0拥有额外的"Peer-Review"增量块

---

## 文件说明

### 核心文件

1. **ablation_studies.py** - 主消融实验脚本
   - 包含所有三个消融实验的完整实现
   - 可以选择运行单个或全部实验

2. **ablation_visualization.py** - 结果可视化脚本
   - 生成各个消融实验的图表
   - 生成综合对比图

3. **ablation_no_critic.py** - 单独的无Critic实验（已有）
   - 专门用于对比有无CriticAgent的差异

### 配置文件

- `config.yaml` - LLM配置文件（在项目根目录）
- Golden set Excel文件（可选，主要用于标注数据）

---

## 使用方法

### 环境准备

```bash
# 安装依赖
pip install matplotlib numpy pandas openpyxl PyPDF2

# 启动GROBID服务（可选，推荐使用以获得更好的PDF解析效果）
docker run -d -p 8070:8070 lfoppiano/grobid:0.8.0
```

### 运行消融实验

#### 1. 运行所有消融实验

```bash
cd /home/lexy/下载/CLwithRAG/KGdemo/eval/deeppaper_eval/src

python ablation_studies.py \
    --papers_dir /path/to/papers_pdf \
    --output_dir ./results/ablation \
    --grobid_url http://localhost:8070
```

#### 2. 只运行特定实验

```bash
# 只运行实验1 (CriticAgent)
python ablation_studies.py \
    --papers_dir /path/to/papers_pdf \
    --output_dir ./results/ablation \
    --ablation 1

# 只运行实验2 (LogicAnalyst)
python ablation_studies.py \
    --papers_dir /path/to/papers_pdf \
    --output_dir ./results/ablation \
    --ablation 2

# 只运行实验3 (Citation Detective)
python ablation_studies.py \
    --papers_dir /path/to/papers_pdf \
    --output_dir ./results/ablation \
    --ablation 3
```

#### 3. 测试模式（处理前3篇论文）

```bash
python ablation_studies.py \
    --papers_dir /path/to/papers_pdf \
    --output_dir ./results/ablation \
    --limit 3
```

### 可视化结果

```bash
# 生成所有图表
python ablation_visualization.py \
    --results_dir ./results/ablation \
    --output_dir ./results/ablation/figures
```

---

## 输出文件

### 实验结果（JSON格式）

运行实验后，会在`output_dir`生成以下JSON文件：

1. **ablation_1_critic_results.json**
   ```json
   {
     "ablation_1_with_critic": [...],
     "ablation_1_without_critic": [...],
     "summary": {
       "total_papers": 10,
       "description": "Ablation Study 1: Impact of CriticAgent",
       "variant_a": "Full system with iterative refinement",
       "variant_b": "Single-pass extraction without Critic"
     }
   }
   ```

2. **ablation_2_logic_analyst_results.json**
   ```json
   {
     "ablation_2_logic_analyst": [...],
     "ablation_2_simple_summarization": [...],
     "summary": {
       "total_papers": 10,
       "description": "Ablation Study 2: LogicAnalystAgent vs. Simple",
       "variant_a": "LogicAnalystAgent with causal reasoning",
       "variant_b": "Simple summarization without reasoning"
     }
   }
   ```

3. **ablation_3_citation_results.json**
   ```json
   {
     "ablation_3_with_citation": [...],
     "ablation_3_section_only": [...],
     "summary": {
       "total_papers": 10,
       "description": "Ablation Study 3: Impact of Citation Detective",
       "variant_a": "Dual-stream (Section + Citation)",
       "variant_b": "Section-only (no citation)"
     }
   }
   ```

### 可视化图表（PNG格式）

在`output_dir/figures`会生成以下图表：

1. **ablation_1_critic_impact.png**
   - 折线图：展示迭代次数对质量的影响

2. **ablation_2_logic_analyst_pairing.png**
   - 柱状图：对比LogicAnalyst和简单方法的Pairing Accuracy

3. **ablation_3_citation_detective_impact.png**
   - 堆叠柱状图：展示Self-Reported vs. Peer-Review的局限性来源

4. **ablation_comprehensive_comparison.png**
   - 综合对比图：三个实验的整体效果对比

---

## 实验数据格式

### 每个实验结果的数据结构

```python
{
    'paper_id': 'W1234567890',
    'problem': '...',           # 提取的问题
    'method': '...',            # 提取的方法
    'limitation': '...',        # 提取的局限性
    'future_work': '...',       # 提取的未来工作
    'extraction_time': 15.2,    # 提取耗时（秒）
    'metadata': {
        'use_critic': True,     # 是否使用Critic
        'iterations': {         # 各字段的迭代次数
            'problem': 2,
            'method': 3,
            'limitation': 2,
            'future_work': 1
        },
        'confidences': {        # 各字段的置信度
            'problem': 0.92,
            'method': 0.89,
            'limitation': 0.85,
            'future_work': 0.88
        },
        'extraction_methods': { # 各字段使用的提取方法
            'problem': 'logic_analyst',
            'method': 'logic_analyst',
            'limitation': 'section_locator + citation_detective',
            'future_work': 'section_locator'
        }
    }
}
```

---

## 评估指标

### 1. Precision (精确度)
- 提取内容的准确性
- 计算方法：与Golden Set对比，使用BLEU或ROUGE

### 2. Recall (召回率)
- 提取内容的完整性
- 特别关注隐式局限性的召回

### 3. Format Compliance (格式符合度)
- 提取内容是否符合预定义格式
- 是否包含必要的结构化信息

### 4. Pairing Accuracy (配对准确性)
- Problem和Method是否正确对应
- 人工标注 + 自动验证

---

## 预期论文图表

### 图1: CriticAgent的迭代优化效果
```
Quality Score
    1.0 |                    ●━━━● (With Critic)
        |               ●━━━●
    0.9 |          ●━━━●
        |     ●━━━●
    0.8 |━━━●
        |
    0.7 | ●━━━━━━━━━━━━━━━━━━━━━━━ (Without Critic)
        |
    0.6 +--------------------------------
        0    1    2    3    4
             Iteration Round
```

### 图2: LogicAnalyst vs. Simple的Pairing Accuracy
```
Pairing Accuracy
    1.0 |
        |        ████████████
    0.9 |        ████████████ 0.91
        |        ████████████
        |        ████████████
    0.8 |        ████████████
        |        ████████████
        |        ████████████
    0.7 | ██████ ████████████ 0.68
        | ██████ ████████████
    0.6 | ██████ ████████████
        +----------------------
          Simple  LogicAnalyst
```

### 图3: Citation Detective的双流增益
```
Limitation Count
    8 |                 ████
      |                 ████ Peer-Review
    6 |                 ████
      |                 ████
    4 |        ████     ████
      |        ████     ████ Self-Reported
    2 |        ████     ████
      |        ████     ████
    0 +------------------------
        Section  Section+Citation
          Only      (Full)
```

---

## 故障排查

### 问题1: GROBID服务连接失败
```bash
# 检查GROBID是否运行
curl http://localhost:8070/api/isalive

# 如果没有运行，启动GROBID
docker run -d -p 8070:8070 lfoppiano/grobid:0.8.0
```

### 问题2: 导入错误
```bash
# 确保在正确的目录运行
cd /home/lexy/下载/CLwithRAG/KGdemo/eval/deeppaper_eval/src

# 检查Python路径
python -c "import sys; print('\n'.join(sys.path))"
```

### 问题3: 内存不足
```bash
# 减少批处理大小
python ablation_studies.py --limit 5  # 只处理5篇论文
```

---

## 注意事项

1. **运行时间**: 完整实验可能需要较长时间（每篇论文约30-60秒）
2. **API限制**: 注意LLM API的调用限制，建议添加适当的延迟
3. **结果保存**: 建议在每个实验之间保存中间结果
4. **随机性**: 由于LLM的随机性，建议运行多次取平均值

---

## 引用

如果使用本消融实验代码，请引用：

```
@article{deeppaper2024,
  title={DeepPaper 2.0: Multi-Agent Architecture for Deep Academic Paper Analysis},
  author={...},
  journal={...},
  year={2024}
}
```

---

## 更新日志

- 2024-12-18: 初始版本，包含三个完整消融实验
- 2024-12-18: 添加可视化脚本
- 2024-12-18: 添加综合对比图

---

## 联系方式

如有问题，请联系项目维护者或提交Issue。
