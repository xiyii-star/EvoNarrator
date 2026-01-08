# Future Idea Prediction Evaluation System

这是一个用于评估科研idea生成系统的完整评估框架，基于向量检索和LLM评估的方法。该系统通过将生成的研究想法与真实发表的学术论文进行匹配和评分，来衡量idea生成系统的质量和创新性。

## 🎯 系统概述

本系统采用**两阶段评估方法**：

1. **向量检索阶段**：使用语义相似度快速筛选候选论文（Top-K）
2. **LLM评估阶段**：深度分析idea与论文的匹配程度，给出0-10分的详细评分

**核心优势**：
- ✅ 自动化评估，无需人工标注
- ✅ 多维度评分（问题一致性、方法相似度、应用场景等）
- ✅ 完整的指标体系（Hit Rate、Precision、MRR等）
- ✅ 支持批量评估和结果对比分析

## 📁 项目结构

```
Future_Idea_Prediction/
├── src/                              # 源代码目录
│   ├── evaluation_pipeline.py        # 主评估流程
│   ├── vector_retrieval.py           # 向量检索模块
│   ├── llm_evaluator.py              # LLM评估模块
│   ├── metrics_calculator.py         # 指标计算模块
│   ├── config_loader.py              # 配置加载模块
│   ├── compare_all_results.py        # 批量结果对比工具
│   └── show_top_matches.py           # Top-N匹配展示工具
├── data/                             # 数据目录
│   ├── iclr/                         # ICLR论文数据（JSONL格式）
│   ├── ideas/                        # 生成的ideas文件（JSON格式）
│   ├── vector_db/                    # 向量数据库缓存
│   └── sample_ideas.json             # 示例ideas文件
├── results/                          # 评估结果目录
│   └── <idea_filename>_results/      # 每个ideas文件的独立结果
│       ├── retrieval_results.json    # 向量检索结果
│       ├── evaluation_results.json   # LLM评估结果
│       └── metrics_report.json       # 指标报告
├── run_evaluation.sh                 # 单文件评估脚本
├── batch_evaluate_all.sh             # 批量评估脚本
├── requirements.txt                  # Python依赖
├── README.md                         # 本文档
└── SUCCESSFUL_MATCHES_REPORT.md      # 成功匹配案例分析报告
```

## 🏗️ 系统架构

评估系统包含四个核心模块：

### 1. 配置加载模块 ([config_loader.py](src/config_loader.py))
- 从主项目的 `config/config.yaml` 读取LLM配置
- 自动读取API密钥、模型名称、base_url等
- 支持命令行参数覆盖配置文件

### 2. 向量检索模块 ([vector_retrieval.py](src/vector_retrieval.py))
- 使用本地Embedding模型将论文摘要转换为向量
- 构建内存向量数据库（支持磁盘缓存）
- 对生成的idea进行Top-K相似论文检索

### 3. LLM评估模块 ([llm_evaluator.py](src/llm_evaluator.py))
- 使用LLM对idea与真实论文进行深度对比评估
- 从问题一致性、方法相似度、应用场景等多维度打分
- 输出0-10分的匹配分数和详细评估理由

### 4. 指标计算模块 ([metrics_calculator.py](src/metrics_calculator.py))
- 计算Hit Rate@K（命中率）
- 计算Precision@K、MRR等检索指标
- 生成按年份统计的评估报告
- 提供Top-N最佳匹配排序

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装：
```bash
pip install sentence-transformers numpy openai tqdm pyyaml
```

### 准备数据

#### 1. Ideas文件格式

将你生成的ideas保存为JSON或JSONL格式：

**JSON格式（推荐）：**
```json
[
  {
    "idea_id": "idea_001",
    "idea_text": "We propose a novel attention mechanism..."
  },
  {
    "idea_id": "idea_002",
    "idea_text": "This paper introduces a new training method..."
  }
]
```

**JSONL格式：**
```jsonl
{"idea_id": "idea_001", "idea_text": "We propose a novel attention mechanism..."}
{"idea_id": "idea_002", "idea_text": "This paper introduces a new training method..."}
```

**字段说明：**
- `idea_id`: 唯一标识符（必需）
- `idea_text`: idea的完整文本描述（必需，建议包含背景、方法、创新点等）

#### 2. 配置LLM

**方法一：使用主项目配置文件（推荐）**

系统会自动从 `../../config/config.yaml` 读取LLM配置：

```yaml
llm:
  enabled: true
  provider: openai
  model: gpt-4o-2024-11-20
  api_key: your-api-key-here
  base_url: https://api.openai.com/v1
  temperature: 0.3
  max_tokens: 4096
  embedding_model: all-MiniLM-L6-v2
```

**方法二：命令行参数覆盖**

```bash
python src/evaluation_pipeline.py \
  --ideas_file your_ideas.json \
  --llm_model gpt-4 \
  --llm_api_key your-api-key \
  --llm_base_url http://localhost:8000/v1
```

### 运行评估

#### 单文件评估

**使用快速启动脚本（推荐）：**

```bash
# 使用示例数据
./run_evaluation.sh data/sample_ideas.json

# 使用你自己的ideas文件
./run_evaluation.sh path/to/your_ideas.json

# 传递额外参数
./run_evaluation.sh your_ideas.json --threshold 5.0 --top_k 10
```

**或使用Python命令：**

```bash
cd src
python evaluation_pipeline.py \
  --ideas_file ../data/sample_ideas.json \
  --top_k 5 \
  --threshold 7.0
```

#### 批量评估多个Ideas文件

如果你在 `data/ideas/` 目录下有多个ideas文件需要评估：

```bash
# 顺序批量评估（推荐新手）
./batch_evaluate_all.sh

# 传递额外参数
./batch_evaluate_all.sh --threshold 5.0 --top_k 10
```

**特点：**
- 自动评估 `data/ideas/` 目录下的所有JSON文件
- 每个文件生成独立的结果目录
- 显示详细进度和最终统计
- 支持失败重试

### 查看评估结果

#### 1. 查看单个文件的Top-N最佳匹配

```bash
cd src

# 查看Top 10最佳匹配（默认）
python show_top_matches.py

# 查看Top 5最佳匹配
python show_top_matches.py --top_n 5

# 显示完整的评估理由
python show_top_matches.py --top_n 10 --show_full_reason

# 指定具体的结果文件
python show_top_matches.py \
  --evaluation_results ../results/your_results/evaluation_results.json \
  --ideas_file ../data/your_ideas.json \
  --top_n 5
```

**输出示例：**
```
====================================================================================================
                                TOP 5 BEST MATCHING IDEAS AND PAPERS
====================================================================================================

====================================================================================================
RANK #1 - Idea: idea_003 - Match Score: 8.0/10
====================================================================================================

📝 IDEA TITLE:
   Selective Agreement Synthesized: Enhancing Opinion Dynamics Modeling...

📝 IDEA ABSTRACT:
   Background: Understanding opinion dynamics in LLM interactions...

📄 MATCHED PAPER:
   Title: Simple synthetic data reduces sycophancy in large language models
   Year:  2025

💡 EVALUATION REASON:
   Both ideas address the issue of sycophancy in large language model interactions...
```

#### 2. 对比所有批量评估结果

```bash
cd src

# 对比所有结果文件
python compare_all_results.py

# 指定结果目录和输出文件
python compare_all_results.py \
  --results_dir ../results \
  --output ../results/comparison.json
```

**输出示例：**
```
====================================================================================================
                              BATCH EVALUATION COMPARISON REPORT
====================================================================================================

Total evaluated idea sets: 8

----------------------------------------------------------------------------------------------------
Idea Set Name                                                Ideas  Avg Score     Hit@5
----------------------------------------------------------------------------------------------------
research_ideas_Using_LLM_agent_103343                          10       5.6      10.0%
research_ideas_NLP_213731                                      10       4.9       0.0%
research_ideas_NLP_212636                                      10       4.6       0.0%
----------------------------------------------------------------------------------------------------
OVERALL                                                        90       4.3       1.1%
----------------------------------------------------------------------------------------------------

Best Average Score: research_ideas_Using_LLM_agent_103343
  Average Score: 5.6/10
  Total Ideas:   10

Top 10 Best Matches Overall:
 1. Score: 8.0/10 - idea_003
    From:  research_ideas_Using_LLM_agent_103343
    Paper: Simple synthetic data reduces sycophancy in large language models
    Year:  2025
```

## 📊 评估指标

系统计算以下核心指标：

### 1. Hit Rate@K（命中率）
在生成的前K个idea中，有多少个成功匹配到至少一篇真实论文（分数≥阈值）。

- **Hit Rate@1**: 第1个idea是否命中
- **Hit Rate@3**: 前3个idea中是否有命中
- **Hit Rate@5**: 前5个idea中是否有命中
- **Hit Rate@10**: 前10个idea中是否有命中

### 2. Precision@K（精确率）
在返回的前K个候选论文中，匹配论文的占比。

### 3. Mean Reciprocal Rank (MRR)
第一个相关结果的排名的倒数的平均值，衡量检索质量。

公式：`MRR = (1/N) * Σ(1/rank_i)`

### 4. 分数分布
统计匹配分数在各个区间的分布：

| 分数区间 | 相关性评价   | 说明                   |
| -------- | ------------ | ---------------------- |
| 9.0-10.0 | 几乎完全一致 | idea与论文高度吻合     |
| 7.0-8.9  | 高度相关     | 核心问题和方法相似     |
| 5.0-6.9  | 中等相关     | 部分问题或方法有重叠   |
| 3.0-4.9  | 略有相关     | 仅在主题领域上有关联   |
| 0-2.9    | 基本不相关   | 几乎没有相似之处       |

### 5. 按年份统计
分别统计与2023、2024、2025年论文的匹配情况，帮助分析idea的时效性。

## 📈 输出结果详解

评估完成后会在 `results/<idea_filename>_results/` 目录下生成以下文件：

### 1. retrieval_results.json
向量检索的Top-K结果，包含：
- 每个idea检索到的候选论文
- 向量相似度分数
- 论文基本信息（标题、年份、摘要等）

### 2. evaluation_results.json
LLM详细评估结果，包含：
- 每个idea与每篇候选论文的匹配分数（0-10分）
- 详细的评估理由
- 多维度评分解释

### 3. metrics_report.json
完整的指标报告，包含：
- Hit Rate@K统计
- Precision、MRR等指标
- 分数分布统计
- 按年份的匹配情况
- Top-N最佳匹配详情

**示例报告输出：**

```
================================================================================
                           EVALUATION REPORT
================================================================================

Total Ideas Evaluated: 100
Match Threshold: 7.0

--------------------------------------------------------------------------------
Hit Rate@K:
  Hit Rate@ 1: 15.00%
  Hit Rate@ 3: 32.00%
  Hit Rate@ 5: 45.00%
  Hit Rate@10: 58.00%

--------------------------------------------------------------------------------
Other Metrics:
  Precision@5: 25.60%
  MRR:         0.2345
  Avg Max Score: 5.67/10

--------------------------------------------------------------------------------
Score Distribution:
  0-2:   20 ( 20.0%)
  3-4:   25 ( 25.0%)
  5-6:   30 ( 30.0%)
  7-8:   20 ( 20.0%)
  9-10:   5 (  5.0%)

--------------------------------------------------------------------------------
Year-wise Metrics:

  2023:
    Total Ideas:   35
    Matched:       10
    Hit Rate:      28.57%
    Avg Score:     5.8/10
    Score Range:   2.0 - 9.0

  2024:
    Total Ideas:   40
    Matched:       15
    Hit Rate:      37.50%
    Avg Score:     6.2/10
    Score Range:   1.5 - 8.5

  2025:
    Total Ideas:   25
    Matched:       5
    Hit Rate:      20.00%
    Avg Score:     5.1/10
    Score Range:   2.5 - 7.5

--------------------------------------------------------------------------------
Top 10 Best Matches:

  #1 - Idea: idea_042 (Score: 9.2/10)
      Paper: Attention Is All You Need
      Year:  2023
      Reason: Both ideas propose transformer-based architectures...

  #2 - Idea: idea_018 (Score: 8.5/10)
      Paper: BERT: Pre-training of Deep Bidirectional Transformers
      Year:  2024
      Reason: Similar approaches to pre-training language models...

  ... (showing top 10)

================================================================================
```

## 🔧 高级用法

### 命令行参数完整列表

```bash
python src/evaluation_pipeline.py \
  --config_path PATH              # 配置文件路径（默认：../../config/config.yaml）
  --ideas_file PATH               # Ideas文件路径（必需）
  --data_dir PATH                 # ICLR论文数据目录（默认：data/iclr）
  --vector_db_dir PATH            # 向量数据库保存目录（默认：data/vector_db）
  --results_dir PATH              # 结果保存目录（默认：results）
  --embedding_model NAME          # Embedding模型名称（覆盖配置文件）
  --llm_model NAME                # LLM模型名称（覆盖配置文件）
  --llm_api_key KEY               # LLM API密钥（覆盖配置文件）
  --llm_base_url URL              # LLM API基础URL（覆盖配置文件）
  --top_k N                       # 检索候选论文数量（默认：5）
  --threshold SCORE               # 匹配分数阈值（默认：7.0）
  --force_rebuild_db              # 强制重建向量数据库
```

### 分步运行（高级用户）

如果需要分步运行或自定义流程，可以单独使用各个模块：

#### 步骤1：构建向量数据库

```python
from vector_retrieval import EmbeddingModel, PaperVectorRetrieval

# 初始化
embedding_model = EmbeddingModel()
retrieval_system = PaperVectorRetrieval(embedding_model)

# 加载论文并构建向量数据库
papers = retrieval_system.load_papers_from_jsonl([
    "data/iclr/iclr_2023_submitted.jsonl",
    "data/iclr/iclr2024_submissions.jsonl",
    "data/iclr/iclr2025_submissions.jsonl"
])
retrieval_system.build_vector_database(papers)

# 保存
retrieval_system.save_database("data/vector_db")
```

#### 步骤2：检索候选论文

```python
# 检索
idea_text = "Your generated idea text..."
top_papers = retrieval_system.retrieve_similar_papers(idea_text, top_k=5)

for paper, score in top_papers:
    print(f"Similarity: {score:.4f}")
    print(f"Title: {paper['title']}")
    print(f"Year: {paper['year']}")
    print()
```

#### 步骤3：LLM评估

```python
from llm_evaluator import LLMEvaluator

# 初始化评估器
evaluator = LLMEvaluator(
    model_name="gpt-4",
    api_key="your-api-key"
)

# 评估单个pair
result = evaluator.evaluate_single_pair(
    idea_text="Your idea...",
    paper_abstract="Paper abstract..."
)

print(f"Match Score: {result['match_score']}/10")
print(f"Reason: {result['reason']}")
```

#### 步骤4：计算指标

```python
from metrics_calculator import MetricsCalculator
import json

# 加载评估结果
with open("results/evaluation_results.json") as f:
    evaluation_results = json.load(f)

# 计算指标
calculator = MetricsCalculator()
report = calculator.generate_report(
    evaluation_results,
    k_values=[1, 3, 5, 10],
    threshold=7.0
)
```

### 优化建议

#### 1. 调整评估阈值

根据你的需求调整匹配分数阈值：

```bash
# 严格模式（只接受高度相关）
./run_evaluation.sh your_ideas.json --threshold 7.5

# 宽松模式（发现更多潜在匹配）
./run_evaluation.sh your_ideas.json --threshold 5.0

# 探索模式（查看所有相关性）
./run_evaluation.sh your_ideas.json --threshold 3.0
```

#### 2. 增加检索候选数

```bash
# 增加候选数以提高召回率
./run_evaluation.sh your_ideas.json --top_k 15

# 对于交叉领域的ideas，建议使用更大的K值
./run_evaluation.sh your_ideas.json --top_k 20
```

#### 3. 使用专业Embedding模型

当前默认使用 `all-MiniLM-L6-v2`（通用模型），可以升级为学术领域专用模型：

```yaml
# 修改 config.yaml
llm:
  embedding_model: "allenai/specter"  # 学术论文专用
  # 或
  embedding_model: "allenai/scibert_scivocab_uncased"  # 科学文献专用
```

**推荐模型：**
- **SPECTER**: 专门为学术论文设计，效果最佳
- **SciBERT**: 科学文献预训练，性能优秀
- **all-MiniLM-L6-v2**: 通用模型，速度快但精度略低

#### 4. 扩大论文数据库

当前数据库包含ICLR 2023-2025论文。你可以添加更多会议/期刊：

```python
# 在 evaluation_pipeline.py 中添加更多数据源
papers = retrieval_system.load_papers_from_jsonl([
    "data/iclr/iclr_2023_submitted.jsonl",
    "data/iclr/iclr2024_submissions.jsonl",
    "data/iclr/iclr2025_submissions.jsonl",
    "data/acl/acl_2024.jsonl",           # 添加ACL
    "data/emnlp/emnlp_2024.jsonl",       # 添加EMNLP
])
```

## 💡 成功案例分析

查看 [SUCCESSFUL_MATCHES_REPORT.md](SUCCESSFUL_MATCHES_REPORT.md) 了解详细的成功匹配案例分析。

**关键发现：**
1. ✅ **聚焦具体问题**：避免过于宽泛，选择明确的研究挑战
2. ✅ **采用热门方法**：使用当前学术界认可的技术手段
3. ✅ **适度扩展范围**：在核心思想基础上合理创新
4. ✅ **关注时效性**：紧跟最新研究趋势和未解决问题

## ❓ 常见问题

### 问题1：ImportError: No module named 'sentence_transformers'

**解决方法：**
```bash
pip install sentence-transformers
```

### 问题2：OpenAI API错误

**可能原因：**
- API key不正确
- API base_url不可访问
- 配额不足

**解决方法：**
```bash
# 检查配置文件
cat ../../config/config.yaml

# 测试API连接
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.openai.com/v1/models
```

### 问题3：内存不足

**解决方法：**
- 减少 `top_k` 值
- 分批处理论文
- 使用更小的Embedding模型

### 问题4：向量数据库加载失败

**解决方法：**
```bash
# 强制重建向量数据库
./run_evaluation.sh your_ideas.json --force_rebuild_db
```

### 问题5：评估速度太慢

**优化方法：**
1. 使用向量数据库缓存（第二次运行会快很多）
2. 减少 `top_k` 值（减少LLM调用次数）
3. 使用更快的LLM模型（如gpt-3.5-turbo）
4. 使用本地LLM（避免网络延迟）

### 问题6：匹配率太低

**调整建议：**
```bash
# 1. 降低阈值
./run_evaluation.sh your_ideas.json --threshold 5.0

# 2. 增加候选数
./run_evaluation.sh your_ideas.json --top_k 15

# 3. 使用学术专用Embedding模型
# 修改 config.yaml 中的 embedding_model

# 4. 扩大论文数据库范围
# 添加更多会议/期刊的论文数据
```

## 🛠️ 系统扩展

系统设计为模块化，可以轻松扩展：

### 1. 更换Embedding模型

```bash
# 方法1：修改配置文件
# config.yaml:
llm:
  embedding_model: "allenai/specter"

# 方法2：命令行参数
python src/evaluation_pipeline.py \
  --embedding_model "allenai/specter" \
  --ideas_file your_ideas.json
```

### 2. 使用不同的LLM

```bash
# 使用本地模型
python src/evaluation_pipeline.py \
  --llm_model "local-model" \
  --llm_base_url "http://localhost:8000/v1" \
  --ideas_file your_ideas.json

# 使用其他API服务
python src/evaluation_pipeline.py \
  --llm_model "gpt-4" \
  --llm_base_url "https://api.custom-service.com/v1" \
  --llm_api_key "your-key" \
  --ideas_file your_ideas.json
```

### 3. 自定义评估Prompt

修改 [llm_evaluator.py](src/llm_evaluator.py:80) 中的 `create_evaluation_prompt` 方法：

```python
def create_evaluation_prompt(self, idea_text: str, paper_abstract: str) -> str:
    """自定义你的评估prompt"""
    return f"""
    你的自定义评估提示...

    Idea: {idea_text}
    Paper: {paper_abstract}

    请从以下维度评估...
    """
```

### 4. 添加新的指标

在 [metrics_calculator.py](src/metrics_calculator.py) 中添加新的计算方法：

```python
def calculate_custom_metric(self, evaluation_results: List[Dict]) -> float:
    """实现你的自定义指标"""
    # 你的计算逻辑
    pass
```

## 📚 相关资源

- **主项目配置**：`../../config/config.yaml`
- **成功案例分析**：[SUCCESSFUL_MATCHES_REPORT.md](SUCCESSFUL_MATCHES_REPORT.md)
- **ICLR论文数据**：`data/iclr/`
- **示例Ideas**：`data/sample_ideas.json`

## 📝 开发日志

### v1.0 - 初始版本
- ✅ 完整的向量检索+LLM评估pipeline
- ✅ 支持单文件和批量评估
- ✅ 丰富的评估指标体系
- ✅ 结果对比和可视化工具

### 未来计划
- [ ] 支持更多论文数据源（ACL、EMNLP、NeurIPS等）
- [ ] 添加交互式Web界面
- [ ] 支持多语言论文评估
- [ ] 集成更多评估维度（创新性、可行性等）
- [ ] 提供预训练的idea质量评分模型

## 🤝 贡献指南

欢迎提Issue和PR！如有问题或建议，请在GitHub上提交。

## 📄 许可证

本项目遵循MIT许可证。

---

**最后更新时间**: 2025-12-10
**系统版本**: v1.0
**维护者**: Future Idea Prediction Team
