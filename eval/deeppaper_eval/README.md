# DeepPaper: Multi-Agent System for Academic Paper Analysis

**论文深度信息提取的多智能体系统评估框架**

![Status](https://img.shields.io/badge/Status-Completed-success) ![Papers](https://img.shields.io/badge/Papers-79-blue) ![Methods](https://img.shields.io/badge/Methods-6-orange) ![Samples](https://img.shields.io/badge/Samples-1896-red)

## 📋 项目概述

本项目是一个**完整的多智能体系统评估框架**,专注于学术论文的深度信息提取任务。系统通过对比6种不同方法(包括完整多智能体架构、消融实验和基线方法),评估它们在提取论文核心信息(Problem、Contribution、Limitation、Future Work)时的性能表现。

### 核心特性

- 🤖 **多智能体协作架构**: Navigator→Extractor→Critic→Synthesizer 四阶段协同工作
- 📊 **大规模评估**: 79篇学术论文 × 6种方法 × 4个关键字段 = **1,896个评估样本**
- 📈 **多维度指标**: ROUGE、BLEU、BERTScore、LLM-based评估等多层次评估体系
- 🔬 **消融实验**: 系统化验证Navigator和Critic组件的有效性
- 📑 **完整可复现**: 提供数据集、代码、配置和详细文档

### 实验结果概览

| 方法                               | ROUGE-1 F1       | 相比基线提升     | 架构特点       |
| ---------------------------------- | ---------------- | ---------------- | -------------- |
| **DeepPaper Multi-Agent** ✨ | **0.4263** | **+18.1%** | 完整四阶段协作 |
| Naive LLM (GPT-4o)                 | 0.3609           | -                | 直接提取基线   |
| Ablation: No Critic                | 0.3586           | -15.9%           | 移除迭代优化   |
| Ablation: No Navigator             | 0.3227           | -24.3%           | 移除章节导航   |
| Pure RAG                           | 0.2503           | -41.3%           | 仅检索拼接     |

**最新评估报告**: [result/evaluation_report.md](result/evaluation_report.md) | **可复现报告**: [REPRODUCIBILITY_REPORT.md](REPRODUCIBILITY_REPORT.md)

---

## 🏗️ 多智能体系统架构

### DeepPaper Multi-Agent 工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input: Academic Paper                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  🧭 Navigator Agent   │  ← 章节定位与导航
              │  - 识别相关章节       │
              │  - 构建论文结构图     │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  📝 Extractor Agent   │  ← 信息提取
              │  - 从定位章节提取     │
              │  - 4个字段并行提取    │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  🔍 Critic Agent      │  ← 质量控制与迭代
              │  - 评估提取质量       │
              │  - 提出改进建议       │
              │  - 触发重提取(可选)   │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  🎯 Synthesizer Agent │  ← 结果整合
              │  - 整合所有提取结果   │
              │  - 生成结构化输出     │
              └──────────┬───────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │  Output: Structured Analysis   │
         │  • Problem                     │
         │  • Contribution                │
         │  • Limitation                  │
         │  • Future Work                 │
         └───────────────────────────────┘
```

### 6种对比方法详解

| # | 方法名称 | 实现文件 | 核心特点 | 评估目的 |
|---|---------|---------|---------|---------|
| 1️⃣ | **DeepPaper Multi-Agent** | [mymethod.py](src/mymethod.py) | 完整四阶段协作架构 | 主方法 - 验证整体性能 |
| 2️⃣ | **Naive LLM Baseline** | [baseline_naive_llm.py](src/baseline_naive_llm.py) | 直接输入全文,一次性提取 | 基线对比 - 验证改进幅度 |
| 3️⃣ | **Ablation: No Critic** | [ablation_no_critic.py](src/ablation_no_critic.py) | Navigator + Extractor + Synthesizer | 消融实验 - 验证迭代优化价值 |
| 4️⃣ | **Ablation: No Navigator** | [ablation_no_navigator.py](src/ablation_no_navigator.py) | Extractor + Critic + Synthesizer | 消融实验 - 验证章节导航价值 |
| 5️⃣ | **Pure RAG** | [rag_paper.py](src/rag_paper.py) | 向量检索 + 段落拼接 | 对比检索方法 |
| 6️⃣ | **LLM + RAG** | [llm_rag_paper.py](src/llm_rag_paper.py) | RAG检索 + LLM生成 | 混合方法对比 |

### 关键设计决策

#### ✅ 为什么需要Navigator?
- **问题**: 学术论文长度通常超过LLM上下文限制(8-32页)
- **解决方案**: Navigator智能定位相关章节,减少噪音,提高准确性
- **验证**: 消融实验显示Navigator带来**24.3%性能提升** (0.4263 vs 0.3227)

#### ✅ 为什么需要Critic?
- **问题**: 首次提取可能不完整或不准确
- **解决方案**: Critic评估质量并触发迭代优化
- **验证**: 消融实验显示Critic带来**15.9%性能提升** (0.4263 vs 0.3586)

#### ✅ 为什么需要Synthesizer?
- **问题**: 多次迭代的结果需要智能整合
- **解决方案**: Synthesizer基于质量评分选择最佳版本或融合多个版本
- **效果**: 确保输出的一致性和完整性

---

## 📁 项目结构

```
deeppaper_eval/
├── 📊 data/                           # 数据集 (79篇论文)
│   ├── golden_set_79papers.xlsx      # 人工标注的黄金标准集
│   ├── papers_txt/                   # 解析后的论文纯文本 (P001-P079.txt)
│   ├── papers_pdf/                   # 原始PDF文件
│   └── pdf.py                        # PDF解析工具
│
├── 🤖 src/                            # 6种方法实现
│   ├── mymethod.py                   # ① DeepPaper Multi-Agent (主方法)
│   ├── baseline_naive_llm.py        # ② Naive LLM (基线)
│   ├── ablation_no_critic.py        # ③ 消融实验: 无Critic
│   ├── ablation_no_navigator.py     # ④ 消融实验: 无Navigator
│   ├── rag_paper.py                 # ⑤ Pure RAG
│   └── llm_rag_paper.py             # ⑥ LLM + RAG
│
├── 📈 evaluator/                      # 评估系统
│   ├── metrics.py                    # 多维评估指标 (ROUGE/BLEU/BERTScore/LLM-Eval)
│   └── evaluator.py                  # 主评估器 (批量对比、报告生成)
│
├── 📋 result/                         # 评估结果
│   ├── *_results.json                # 各方法的提取结果
│   ├── evaluation_report.json        # 详细评估数据
│   ├── evaluation_report.csv         # Excel兼容对比表
│   └── evaluation_report.md          # 人类可读报告
│
├── 🚀 运行脚本
│   ├── quick_start.sh                # ⭐ 交互式快速开始菜单
│   ├── run_all_experiments.py        # 自动运行所有方法
│   └── run_enhanced_evaluation.py    # 增强评估 (BERTScore + LLM-Eval)
│
├── 📖 文档
│   ├── README.md                     # 本文档 (项目概述与使用指南)
│   ├── REPRODUCIBILITY_REPORT.md     # ⭐ 可复现性报告
│   ├── PROJECT_SUMMARY.md            # 项目技术总结
│   └── EVALUATION_GUIDE.md           # 评估指标详细说明
│
└── requirements.txt                  # Python依赖包
```

### 核心模块说明

| 模块 | 文件 | 功能 | 关键类/函数 |
|-----|------|------|------------|
| 🤖 多智能体系统 | [mymethod.py](src/mymethod.py) | 完整的四阶段协作架构 | `NavigatorAgent`, `ExtractorAgent`, `CriticAgent`, `SynthesizerAgent` |
| 📈 评估指标 | [metrics.py](evaluator/metrics.py) | 多维度评估计算 | `compute_rouge()`, `compute_bleu()`, `compute_bertscore()`, `llm_evaluate()` |
| 🎯 主评估器 | [evaluator.py](evaluator/evaluator.py) | 批量评估与报告生成 | `PaperEvaluator`, `evaluate_all_methods()` |
| 🚀 自动化运行 | [run_all_experiments.py](run_all_experiments.py) | 一键运行所有实验 | `run_all_methods()` |

---

## 📊 数据集说明

### 输入数据

**1. Golden Set**: [golden_set_79papers.xlsx](data/golden_set_79papers.xlsx)
- 79篇学术论文的人工标注黄金标准集
- 字段: `Paper_ID`, `Paper_Title`, `Human_Problem`, `Human_Contribution`, `Human_Limitation`, `Human_Future_Work`
- 用于: 评估所有方法的提取质量

**2. Paper Texts**: [papers_txt/](data/papers_txt/)
- 79个纯文本文件 (P001.txt ~ P079.txt)
- 从原始PDF解析而来,去除图表和公式
- 平均长度: 8,000-15,000 tokens

### 输出格式

每个方法生成标准化JSON结果 (示例: [mymethod_results.json](result/mymethod_results.json)):

```json
{
  "method_name": [
    {
      "paper_id": "P001",
      "problem": "本文解决的核心问题...",
      "contribution": "主要贡献包括...",
      "limitation": "当前方法的局限性...",
      "future_work": "未来可以改进的方向...",
      "extraction_time": 12.5,
      "metadata": {...}
    }
  ]
}
```

### 评估报告格式

评估器生成三种格式报告:
- 📄 **JSON** ([evaluation_report.json](result/evaluation_report.json)): 完整数据,适合程序化分析
- 📊 **CSV** ([evaluation_report.csv](result/evaluation_report.csv)): 对比表格,可导入Excel
- 📝 **Markdown** ([evaluation_report.md](result/evaluation_report.md)): 可读报告,适合文档展示

---

## 🚀 快速开始

### 方式一: 交互式菜单 (推荐)

最简单的使用方式,提供友好的交互式界面:

```bash
cd /home/lexy/下载/CLwithRAG/KGdemo/eval/deeppaper_eval
bash quick_start.sh
```

**菜单选项:**
1. 📊 查看当前评估结果
2. 🔄 重新运行标准评估 (ROUGE + BLEU)
3. 🎯 运行增强评估 (+ BERTScore)
4. 🧪 测试LLM评估连接
5. 🤖 运行完整LLM评估

### 方式二: 环境配置 + 命令行

#### 步骤1: 安装依赖

```bash
# 基础依赖
pip install pandas openpyxl numpy openai

# 评估指标
pip install rouge-score nltk

# 可选: BERTScore (语义相似度评估)
pip install bert-score

# 可选: 如果运行mymethod,需要安装项目根目录依赖
cd ../../..
pip install -r requirements.txt
cd eval/deeppaper_eval
```

#### 步骤2: 配置API密钥

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 可选,使用代理时设置
```

#### 步骤3: 运行实验

**选项A: 运行所有方法并自动评估**
```bash
python run_all_experiments.py \
  --golden_set data/golden_set_79papers.xlsx \
  --papers_dir data/papers_txt \
  --results_dir result
```

**选项B: 运行特定方法**
```bash
# 运行DeepPaper Multi-Agent
python src/mymethod.py \
  --golden_set data/golden_set_79papers.xlsx \
  --papers_dir data/papers_txt \
  --output result/mymethod_results.json

# 运行基线方法
python src/baseline_naive_llm.py \
  --golden_set data/golden_set_79papers.xlsx \
  --papers_dir data/papers_txt \
  --output result/naive_baseline_results.json \
  --model gpt-4o-2024-11-20
```

**选项C: 仅评估已有结果**
```bash
# 标准评估
python evaluator/evaluator.py \
  --golden_set data/golden_set_79papers.xlsx \
  --results_dir result \
  --output result/evaluation_report

# 增强评估 (包含BERTScore)
python run_enhanced_evaluation.py \
  --golden_set data/golden_set_79papers.xlsx \
  --results_dir result \
  --output result/enhanced_evaluation_report \
  --use_bertscore
```

### 预期运行时间与成本

| 任务 | 估计时间 | API调用次数 | 估算成本 (GPT-4o) |
|------|---------|------------|------------------|
| 单个方法 (79篇) | 30-90分钟 | ~79-158次 | $2-5 |
| 所有6种方法 | 3-6小时 | ~474-948次 | $12-30 |
| 标准评估 (ROUGE/BLEU) | <5分钟 | 0 | $0 |
| LLM评估 (可选) | 1-2小时 | ~316次 | $8-15 |

💡 **节省成本技巧**: 先在小数据集测试 (修改代码只处理前5篇论文)

---

## 📈 评估指标体系

本系统采用多层次评估体系,从词汇匹配到语义理解全方位评估提取质量:

### 核心指标 (默认启用)

| 指标 | 类型 | 说明 | 优点 | 适用场景 |
|------|------|------|------|---------|
| **ROUGE-1/2/L** | 词汇重叠 | 单词/双词/最长公共子序列匹配 | 标准化,可复现 | 内容完整性评估 |
| **BLEU** | N-gram匹配 | 机器翻译领域经典指标 | 评估流畅度 | 生成质量评估 |

### 增强指标 (可选)

| 指标 | 安装 | 说明 | 优点 | 成本 |
|------|------|------|------|------|
| **BERTScore** | `pip install bert-score` | 基于BERT语义相似度 | 捕捉语义信息 | GPU推荐 |
| **LLM-Eval** | 需OpenAI API | GPT-4人类偏好评估 | 最接近人类判断 | $8-15/79篇 |

详细指标说明和使用指南: [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)

---

## 🏆 实验结果与分析

### 整体性能对比

基于79篇论文的完整评估,各方法性能排名如下:

| 排名 | 方法                         | ROUGE-1 F1 ⬆️ | ROUGE-2 F1 | ROUGE-L F1 | BLEU   | 关键特点 |
|------|------------------------------|---------------|------------|------------|--------|---------|
| 🥇 1 | **DeepPaper Multi-Agent**    | **0.4263**    | **0.1956** | **0.2771** | **0.1121** | 完整四阶段协作 |
| 🥈 2 | Naive LLM (GPT-4o)           | 0.3609        | 0.1105     | 0.2213     | 0.0357     | 直接提取基线 |
| 🥉 3 | Ablation: No Critic          | 0.3586        | 0.1398     | 0.2218     | 0.0739     | 无迭代优化 |
| 4    | Ablation: No Navigator       | 0.3227        | 0.1070     | 0.1918     | 0.0548     | 无章节导航 |
| 5    | Pure RAG                     | 0.2503        | 0.0410     | 0.1448     | 0.0112     | 仅检索拼接 |
| 6    | LLM + RAG                    | 0.0233        | 0.0037     | 0.0214     | 0.0000     | ⚠️ 实现异常 |

📊 **完整评估报告**: [result/evaluation_report.md](result/evaluation_report.md)

### 核心发现

#### ✅ 成功验证

1. **Multi-Agent架构有效性**
   - 相比最佳基线(Naive LLM)提升 **+18.1%** (ROUGE-1: 0.4263 vs 0.3609)
   - 在所有评估指标上均取得最佳性能
   - 在Limitation和Future Work字段上尤为突出 (ROUGE-1达到0.51和0.50)

2. **组件贡献度分析** (消融实验)
   - **Navigator组件**: 带来 **+24.3%** 性能提升 (0.4263 vs 0.3227)
   - **Critic组件**: 带来 **+15.9%** 性能提升 (0.4263 vs 0.3586)
   - 结论: Navigator(章节导航)比Critic(迭代优化)对性能影响更大

3. **各字段性能对比**
   ```
   Problem:       ROUGE-1 = 0.42  (中等)
   Contribution:  ROUGE-1 = 0.38  (较低,论文描述差异大)
   Limitation:    ROUGE-1 = 0.51  (最高,表述相对标准)
   Future Work:   ROUGE-1 = 0.50  (最高,模式清晰)
   ```

#### ⚠️ 意外发现

1. **Naive LLM基线表现超预期**
   - 排名第2,得益于GPT-4o的强大理解能力
   - 说明: 现代LLM在单次提取任务上已非常强大
   - 启示: Multi-Agent的优势在于处理更复杂/更长的论文时更加稳定

2. **LLM+RAG方法异常**
   - 性能远低于预期,需要进一步调试
   - 可能原因: 检索策略不当、提示词设计问题、实现bug

### 实验假设验证

| 假设 | 预期排名 | 实际排名 | 验证结果 |
|------|---------|---------|---------|
| Multi-Agent最优 | 1 | 1 | ✅ 成立 |
| 移除Critic性能下降 | 2-3 | 3 | ✅ 成立 |
| 移除Navigator性能下降更多 | 3-4 | 4 | ✅ 成立 |
| Naive LLM基线水平 | 4-5 | 2 | ⚠️ 超预期 |
| LLM+RAG中等性能 | 3-4 | 6 | ❌ 异常 |
| Pure RAG最差 | 6 | 5 | ✅ 基本成立 |

**结论**: 核心假设(Multi-Agent架构优越性)得到充分验证,消融实验有效证明了各组件的价值。

---

## ⚙️ 扩展与定制

### 添加新的提取方法

1. 在 [src/](src/) 创建新的Python文件 (如 `my_new_method.py`)
2. 实现标准接口:
   ```python
   def extract_paper_info(paper_text: str, paper_id: str) -> dict:
       return {
           "paper_id": paper_id,
           "problem": "...",
           "contribution": "...",
           "limitation": "...",
           "future_work": "..."
       }
   ```
3. 在 [run_all_experiments.py](run_all_experiments.py) 中注册方法

### 添加新的评估指标

1. 在 [evaluator/metrics.py](evaluator/metrics.py) 实现新指标:
   ```python
   def compute_my_metric(prediction: str, reference: str) -> float:
       # 实现你的指标逻辑
       return score
   ```
2. 在 [evaluator/evaluator.py](evaluator/evaluator.py) 中集成并更新报告生成逻辑

### 扩展到更多论文

1. 准备新论文文本文件 (放入 `data/papers_txt/`)
2. 更新 [golden_set_79papers.xlsx](data/golden_set_79papers.xlsx) 添加人工标注
3. 重新运行实验和评估

---

## 💡 最佳实践与注意事项

### ✅ 推荐做法

- 🎯 **先用交互式菜单**: 运行 `bash quick_start.sh` 最简单
- 💰 **小数据集测试**: 修改代码限制论文数量 (如只处理前5篇) 以节省成本
- 📊 **优先标准评估**: ROUGE/BLEU免费且快速,BERTScore和LLM评估可选
- 🔄 **增量运行**: 单独运行失败的方法,不需要全部重跑
- 📖 **查阅文档**: 遇到问题先查看 [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) 和 [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

### ⚠️ 常见陷阱

- ❌ **API密钥未设置**: 记得 `export OPENAI_API_KEY="..."`
- ❌ **依赖未安装**: BERTScore需单独安装 `pip install bert-score`
- ❌ **路径错误**: 确保在 `deeppaper_eval/` 目录下运行命令
- ❌ **成本失控**: 完整LLM评估约$8-15,先小规模测试

### 🔍 故障排查

| 问题 | 解决方案 |
|------|---------|
| BERTScore显示为0 | 安装: `pip install bert-score` |
| llm_rag方法性能异常低 | 该方法实现可能有bug,需调试检索和生成逻辑 |
| API调用失败 | 检查 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL` 配置 |
| 论文文本缺失 | 确保 `data/papers_txt/` 有完整的79个txt文件 |
| 只想运行部分论文 | 修改方法脚本: `papers = papers[:5]` |

---

## 📚 相关文档

| 文档 | 用途 | 推荐阅读 |
|------|------|---------|
| **[README.md](README.md)** | 项目概述、使用指南 (本文档) | ⭐⭐⭐ 必读 |
| **[REPRODUCIBILITY_REPORT.md](REPRODUCIBILITY_REPORT.md)** | 可复现性完整报告 | ⭐⭐⭐ 强烈推荐 |
| **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** | 项目技术总结 | ⭐⭐ 技术细节 |
| **[EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)** | 评估指标详细说明 | ⭐⭐ 指标深入理解 |
| **[result/evaluation_report.md](result/evaluation_report.md)** | 最新实验结果 | ⭐⭐⭐ 结果分析 |

---

## ❓ 常见问题 (FAQ)

<details>
<summary><b>Q1: 如何只测试5篇论文以节省成本?</b></summary>

修改任意方法脚本,在加载论文后添加切片:
```python
papers_list = load_papers(...)
papers_list = papers_list[:5]  # 只处理前5篇
```
</details>

<details>
<summary><b>Q2: 为什么llm_rag方法表现这么差?</b></summary>

该方法可能存在实现问题,建议检查:
- RAG检索是否返回相关文本段落
- LLM生成提示词是否合理
- 查看日志确认是否有运行时错误
</details>

<details>
<summary><b>Q3: 如何使用自己的OpenAI代理?</b></summary>

设置环境变量:
```bash
export OPENAI_BASE_URL="https://your-proxy.com/v1"
export OPENAI_API_KEY="your-key"
```
</details>

<details>
<summary><b>Q4: BERTScore一直显示0怎么办?</b></summary>

需要单独安装:
```bash
pip install bert-score
python run_enhanced_evaluation.py --use_bertscore ...
```
</details>

<details>
<summary><b>Q5: papers_txt 和 papers_pdf 有什么区别?</b></summary>

- `papers_txt/`: 已解析的纯文本,**直接用于实验**
- `papers_pdf/`: 原始PDF文件,需用 `data/pdf.py` 解析才能使用
</details>

---

## 📋 项目状态

| 类别 | 状态 | 说明 |
|------|------|------|
| ✅ **核心功能** | 已完成 | 6种方法实现、标准评估、79篇论文评估 |
| ✅ **文档** | 已完成 | README、可复现报告、技术总结、评估指南 |
| ✅ **评估报告** | 已完成 | JSON/CSV/Markdown多格式报告 |
| 🔄 **增强功能** | 进行中 | BERTScore优化、LLM评估测试 |
| ⚠️ **待修复** | 已知问题 | llm_rag方法性能异常需调试 |
| 🔮 **未来计划** | 规划中 | 可视化分析工具、更多评估指标 |

**最后更新**: 2025-12-16
**评估完成日期**: 2025-12-05
**数据规模**: 79篇论文 × 6种方法 × 4个字段 = **1,896个评估样本**

---

## 📖 引用

如果本评估系统对您的研究有帮助,请引用:

```bibtex
@software{deeppaper_eval_2025,
  title     = {DeepPaper: Multi-Agent System for Academic Paper Analysis},
  author    = {KGdemo Project},
  year      = {2025},
  url       = {https://github.com/yourusername/CLwithRAG/tree/main/eval/deeppaper_eval},
  note      = {Comprehensive evaluation framework with 79 papers, 6 methods, and multi-dimensional metrics}
}
```

---

## 📄 许可证

MIT License - 详见项目根目录LICENSE文件

---

<div align="center">

**🌟 如果本项目对您有帮助,请给个Star! 🌟**

[📊 查看评估结果](result/evaluation_report.md) | [📖 可复现报告](REPRODUCIBILITY_REPORT.md) | [🚀 快速开始](quick_start.sh)

</div>
