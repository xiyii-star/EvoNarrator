# DeepPaper 2.0 - 论文深度信息提取系统

DeepPaper 2.0 是一个基于 Multi-Agent 架构的论文深度信息提取系统，能够自动提取论文的四个核心维度：**Problem（研究问题）、Method（方法）、Limitation（局限性）、Future Work（未来工作）**。

## 目录

- [核心创新](#核心创新-)
- [系统架构](#系统架构)
- [项目结构](#项目结构)
- [工作流程](#工作流程)
- [安装与配置](#安装与配置)
- [使用方法](#使用方法)
- [输出示例](#输出示例)
- [核心组件详解](#核心组件详解)
- [特性](#特性)
- [与 DeepPaper 1.0 的区别](#与-deeppaper-10-的区别)
- [注意事项](#注意事项)

---

## 核心创新 ⭐

### 1. Reflection Loop（反思循环）质量验证机制

- 引入 **CriticAgent（审查员）** 实现自动质量控制
- 自动检测提取问题：空结果、错误提取、内容过于宽泛
- 提供具体改进指令，触发迭代重新提取
- 持续优化直到达到质量标准或最大重试次数
- **ACL 级别创新**：自动化提升 Precision、Recall 和 Quality，无需人工干预

### 2. Problem-Solution Pair 逻辑分析

- 使用 **LogicAnalystAgent** 提取问题-解法对（P-S Pairs）
- 聚焦论文核心逻辑，找出"痛点"与"机制"的因果关系
- 自动生成解释：机制如何具体解决问题

### 3. 双流合并：章节分析 + 引用分析

- **章节分析**：从论文自身的 Discussion/Conclusion 提取
- **引用分析**（可选）：通过 CitationDetectiveAgent 从引用论文中挖掘真实评价
- 提供更全面、客观的 Limitation 信息

---

## 系统架构

### 核心组件

```
DeepPaper_Agent2.0/
├── orchestrator.py              # 协调器（Orchestrator）
├── data_structures.py           # 数据结构定义
├── LogicAnalystAgent.py         # 逻辑分析员
├── SectionLocatorAgent.py       # 章节定位员
├── LimitationExtractor.py       # 局限性提取器
├── FutureWorkExtractor.py       # 未来工作提取器
├── CitationDetectiveAgent.py    # 引用侦探
├── critic_agent.py              # 审查员（质量验证与反思）
└── __init__.py                  # 包初始化
```

### Agent 职责分工

| Agent | 职责 | 输入 | 输出 |
|-------|------|------|------|
| **DeepPaper2Orchestrator** | 协调整个提取流程，整合所有组件 | 论文文档 | 最终报告 (JSON + Markdown) |
| **LogicAnalystAgent** | 分析论文核心逻辑，提取 Problem-Solution Pairs | 论文全文 | Problem & Method |
| **SectionLocatorAgent** | 智能定位 Limitation/Future Work 章节 | 论文文档 + 目标字段 | 章节范围 (SectionScope) |
| **LimitationExtractor** | 提取局限性（章节 + 引用） | 论文文档 + Paper ID | Limitation 内容 |
| **FutureWorkExtractor** | 提取未来工作方向 | 论文文档 | Future Work 内容 |
| **CitationDetectiveAgent** | 从引用论文中挖掘真实评价 | Paper ID | 引用分析结果 |
| **CriticAgent** | 质量验证与反思，触发重试 | 提取结果 | 反馈 + 改进指令 |

---

## 项目结构

### 1. 数据结构 (`data_structures.py`)

定义了系统中使用的所有数据类型：

- **PaperSection**: 论文章节（标题、内容、页码、类型）
- **PaperDocument**: 论文文档（ID、标题、摘要、作者、章节列表）
- **FieldType**: 枚举类型（PROBLEM, METHOD, LIMITATION, FUTURE_WORK）
- **SectionScope**: 章节范围（目标章节索引、推理过程、置信度）
- **ExtractionResult**: 提取结果（字段类型、内容、证据、方法、置信度、迭代次数）
- **CriticFeedback**: Critic 反馈（是否通过、反馈类型、改进指令）
- **FinalReport**: 最终报告（四个字段的内容 + 证据 + 元数据）
- **ProblemSolutionPair**: 问题-解法对（问题、解法、解释、置信度、证据）
- **CitationContext**: 引用上下文（引用论文信息、上下文文本、批判性关键词）
- **CitationAnalysisResult**: 引用分析结果（总引用数、分析数、批判性引用数、提取的 limitations）

### 2. 核心 Agent

#### LogicAnalystAgent (`LogicAnalystAgent.py`)

**职责**：分析论文核心逻辑，提取 Problem-Solution Pairs

**核心方法**：
- `analyze(paper_content, paper_metadata)`: 分析论文，返回 Problem-Solution Pairs 列表

**特点**：
- 聚焦"痛点"与"机制"的因果关系
- 自动生成解释：机制如何解决问题
- 支持多个 P-S Pairs（聚焦最核心的 1-3 个）

#### SectionLocatorAgent (`SectionLocatorAgent.py`)

**职责**：智能定位 Limitation 和 Future Work 所在的章节

**核心方法**：
- `locate(paper, field)`: 定位指定字段的章节，返回 SectionScope

**定位策略**：
1. **LLM 智能定位**（优先）：分析章节标题、类型、内容预览
2. **规则匹配**（fallback）：基于章节类型和关键词匹配

**关键技巧**：
- 关注 Discussion/Conclusion 末尾的转折词
- 识别自指（our method, we, this paper）
- 区分"作者自述的局限性"与"对前人工作的批评"

#### LimitationExtractor (`LimitationExtractor.py`)

**职责**：结合章节定位和引用分析提取论文局限性

**工作流程**：
1. 使用 SectionLocatorAgent 定位 limitation 章节
2. 从定位的章节中提取 limitation（第一部分）
3. 使用 CitationDetectiveAgent 从引用中提取（第二部分，可选）
4. 合并两部分结果
5. 使用 CriticAgent 审查并自动重试（可选）

**核心方法**：
- `extract(paper, paper_id, feedback=None)`: 提取 limitation，返回 ExtractionResult

**特点**：
- 双流合并：论文自述 + 同行评价
- 自动区分"本文局限性"与"对前人工作的批评"
- 支持 CriticAgent 质量验证与迭代改进

#### FutureWorkExtractor (`FutureWorkExtractor.py`)

**职责**：提取论文的未来工作方向

**工作流程**：
1. 使用 SectionLocatorAgent 定位 future work 章节
2. 从定位的章节中提取 future work
3. 使用 CriticAgent 审查并自动重试（可选）

**核心方法**：
- `extract(paper, feedback=None)`: 提取 future work，返回 ExtractionResult

**识别技巧**：
- 关键词：future, next, further, explore, plan, will, could, would
- 从 limitation 推断出的改进方向
- Conclusion/Discussion 末尾的展望

#### CitationDetectiveAgent (`CitationDetectiveAgent.py`)

**职责**：从引用论文中挖掘真实的 peer 评价

**核心方法**：
- `analyze_citations(paper_id)`: 获取引用信息并分析，返回 CitationAnalysisResult

**工作流程**：
1. 通过 Semantic Scholar API 获取引用论文列表
2. 筛选包含批判性关键词的引用上下文
3. 使用 LLM 从引用上下文中提取 limitation

**批判性关键词**：
- 转折词：however, although, but, nevertheless
- 负面评价：limitation, weakness, drawback, problem, challenge
- 能力限制：cannot, fails to, unable to, insufficient

**特点**：
- 自动过滤无关引用（纯粹的 cite）
- 聚焦批判性评价
- 支持 ArXiv ID 和 DOI

#### CriticAgent (`critic_agent.py`) ⭐ 核心创新

**职责**：验证提取质量，触发迭代改进

**核心方法**：
- `critique(extraction, paper, scope, evaluation_level)`: 审查提取结果，返回 CriticFeedback
- `critique_and_retry(extraction, paper, extractor_func, scope)`: 审查并自动重试（主入口）

**检测问题类型**：
1. **空结果**（Recall 问题）：未提取到任何内容
2. **错误目标**（Precision 问题）：提取了前人工作的局限性，而非本文的
3. **过于宽泛**（Quality 问题）：描述太泛，缺乏具体性

**Reflection Loop 机制**：
```
初始提取 → Critic 审查 → [不通过] → 改进指令 → 重新提取 → Critic 审查 → ...
                       → [通过] → 返回最终结果
```

**特点**：
- 自动质量控制，无需人工干预
- 提供具体改进指令（不是简单的"重试"）
- 支持严格模式和宽容模式（适应不同难度的论文）
- 最大迭代次数保护（默认 3 次）

#### DeepPaper2Orchestrator (`orchestrator.py`)

**职责**：协调所有组件，完成端到端的论文分析

**核心方法**：
- `analyze_paper(paper_document, paper_id, output_dir)`: 执行完整分析流程，返回 FinalReport

**工作流程**：
1. 使用 LogicAnalystAgent 提取 Problem & Method
2. 使用 LimitationExtractor 提取 Limitation
3. 使用 FutureWorkExtractor 提取 Future Work
4. 整合结果，生成 FinalReport
5. 保存报告（JSON + Markdown）

---

## 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│                    输入：论文 PDF 解析后的内容                    │
│              (标题、摘要、作者、章节列表、元数据)                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 1: 提取 Problem & Method                              │
│                                                             │
│  LogicAnalystAgent                                          │
│  ├── 分析论文全文（Abstract + Intro + Method）              │
│  ├── 识别核心痛点（The Lock）                               │
│  ├── 找出核心机制（The Key）                                │
│  ├── 生成解释（Key 如何解决 Lock）                           │
│  └── 输出: Problem-Solution Pairs                           │
│                                                             │
│  CriticAgent（可选）                                         │
│  └── 审查提取质量 → [不通过] → 改进指令 → 重新提取           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: 提取 Limitation                                    │
│                                                             │
│  LimitationExtractor                                        │
│  ├── SectionLocatorAgent → 定位 Limitation 章节             │
│  ├── 从定位章节提取 Limitation（第一部分）                   │
│  ├── CitationDetectiveAgent → 从引用中提取（第二部分，可选） │
│  ├── 合并结果                                               │
│  └── CriticAgent → 审查并迭代改进                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: 提取 Future Work                                   │
│                                                             │
│  FutureWorkExtractor                                        │
│  ├── SectionLocatorAgent → 定位 Future Work 章节            │
│  ├── 从定位章节提取 Future Work                              │
│  └── CriticAgent → 审查并迭代改进                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 4: 整合结果                                            │
│                                                             │
│  DeepPaper2Orchestrator                                     │
│  ├── 整合 Problem, Method, Limitation, Future Work          │
│  ├── 生成 FinalReport                                       │
│  ├── 保存 JSON 报告（机器可读）                              │
│  └── 保存 Markdown 报告（人类可读）                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    输出：最终报告                             │
│          (deeppaper2_{paper_id}.json + .md)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 安装与配置

### 环境要求

- Python 3.8+
- 依赖库：
  ```bash
  pip install requests  # 用于引用分析（可选）
  ```

### LLM 配置

需要配置 LLM 客户端（参考 `src/llm_config.py`）：

```python
from src.llm_config import LLMClient, LLMConfig

config = LLMConfig.from_file("/path/to/llm_config.json")
llm_client = LLMClient(config)
```

LLM 配置文件示例（`llm_config.json`）：

```json
{
  "model": "gpt-4",
  "api_key": "your-api-key",
  "temperature": 0.7,
  "max_tokens": 2000
}
```

---

## 使用方法

### 1. 准备输入数据

论文需要转换为 JSON 格式，包含以下字段：

```json
{
  "paper_id": "arxiv:2024.12345",
  "title": "论文标题",
  "abstract": "摘要内容...",
  "authors": ["作者1", "作者2"],
  "year": 2024,
  "sections": [
    {
      "title": "Introduction",
      "content": "章节内容...",
      "page_num": 1,
      "section_type": "introduction"
    },
    {
      "title": "Related Work",
      "content": "章节内容...",
      "page_num": 2,
      "section_type": "related_work"
    },
    {
      "title": "Method",
      "content": "章节内容...",
      "page_num": 3,
      "section_type": "method"
    },
    {
      "title": "Discussion",
      "content": "章节内容...",
      "page_num": 7,
      "section_type": "discussion"
    },
    {
      "title": "Conclusion",
      "content": "章节内容...",
      "page_num": 8,
      "section_type": "conclusion"
    }
  ],
  "metadata": {}
}
```

**章节类型说明**：
- `abstract`: 摘要
- `introduction`: 引言
- `related_work`: 相关工作
- `method`: 方法
- `experiment`: 实验
- `result`: 结果
- `discussion`: 讨论
- `conclusion`: 结论
- `other`: 其他

### 2. 运行完整分析

```bash
python orchestrator.py \
  --config /path/to/llm_config.json \
  --paper /path/to/paper.json \
  --output ./output \
  --paper-id arxiv:2024.xxxxx \  # 可选，用于引用分析
  --use-citation  # 可选，启用引用分析
```

### 3. 编程接口

```python
from DeepPaper_Agent2_0 import DeepPaper2Orchestrator
from DeepPaper_Agent2_0.data_structures import PaperDocument, PaperSection
from src.llm_config import LLMClient, LLMConfig

# 初始化 LLM 客户端
config = LLMConfig.from_file("llm_config.json")
llm_client = LLMClient(config)

# 加载论文文档
paper = PaperDocument(
    paper_id="arxiv:2024.xxxxx",
    title="论文标题",
    abstract="摘要...",
    authors=["作者1", "作者2"],
    year=2024,
    sections=[
        PaperSection(title="Introduction", content="...", page_num=1, section_type="introduction"),
        # ...
    ]
)

# 创建协调器
orchestrator = DeepPaper2Orchestrator(
    llm_client=llm_client,
    use_citation_analysis=True  # 是否使用引用分析
)

# 执行分析
report = orchestrator.analyze_paper(
    paper_document=paper,
    paper_id="arxiv:2024.xxxxx",
    output_dir="./output"
)

# 访问结果
print(f"Problem: {report.problem}")
print(f"Method: {report.method}")
print(f"Limitation: {report.limitation}")
print(f"Future Work: {report.future_work}")
```

### 4. 独立运行各组件

每个 Agent 都可以独立运行进行测试：

#### LogicAnalystAgent

```bash
python LogicAnalystAgent.py \
  --config /path/to/llm_config.json \
  --paper /path/to/paper.txt \
  --output logic_analysis_results.json \
  --format json
```

#### SectionLocatorAgent

```bash
python SectionLocatorAgent.py \
  --config /path/to/llm_config.json \
  --paper /path/to/paper.json \
  --field limitation
```

#### LimitationExtractor

```bash
python LimitationExtractor.py \
  --config /path/to/llm_config.json \
  --paper /path/to/paper.json \
  --paper-id arxiv:2024.xxxxx \
  --use-citation \
  --output limitation_results.json
```

#### FutureWorkExtractor

```bash
python FutureWorkExtractor.py \
  --config /path/to/llm_config.json \
  --paper /path/to/paper.json \
  --output future_work_results.json
```

#### CriticAgent

CriticAgent 通常不单独运行，而是集成在其他组件中。如需测试：

```python
from critic_agent import CriticAgent
from data_structures import ExtractionResult, FieldType

critic = CriticAgent(llm_client, max_iterations=3, strict_mode=False)
feedback = critic.validate(extraction_result, field_type=FieldType.LIMITATION)

if not feedback.approved:
    print(f"需要改进: {feedback.critique}")
    print(f"改进指令: {feedback.improvement_instruction}")
```

---

## 输出示例

系统会在输出目录生成两种格式的报告：

- `deeppaper2_{paper_id}.json` - JSON 格式（机器可读）
- `deeppaper2_{paper_id}.md` - Markdown 格式（人类可读）

### JSON 报告示例

```json
{
  "paper_id": "arxiv:2024.xxxxx",
  "title": "论文标题",
  "problem": "本文要解决的核心问题是...",
  "method": "本文提出的方法是...\n\n**Explanation:** 该方法通过...来解决上述问题。",
  "limitation": "**从论文章节提取的局限性:**\n- 局限性1\n- 局限性2\n\n**从引用分析提取的局限性:**\n- 局限性3（来自后续研究）",
  "future_work": "- 未来工作方向1\n- 未来工作方向2",
  "problem_evidence": [
    {"text": "证据引用..."}
  ],
  "method_evidence": [
    {"text": "证据引用..."}
  ],
  "limitation_evidence": [
    {"section_index": 7, "text": "证据引用..."}
  ],
  "future_work_evidence": [
    {"section_index": 8, "text": "证据引用..."}
  ],
  "metadata": {
    "authors": ["作者1", "作者2"],
    "year": 2024,
    "extraction_methods": {
      "problem": "logic_analyst",
      "method": "logic_analyst",
      "limitation": "section_locator + citation_detective",
      "future_work": "section_locator"
    },
    "confidences": {
      "problem": 0.85,
      "method": 0.85,
      "limitation": 0.80,
      "future_work": 0.80
    }
  }
}
```

### Markdown 报告示例

```markdown
# 论文标题

## Paper Information
- **Paper ID**: arxiv:2024.xxxxx
- **Authors**: 作者1, 作者2
- **Year**: 2024

---

## Problem

本文要解决的核心问题是...

---

## Method

本文提出的方法是...

**Explanation:** 该方法通过...来解决上述问题。

---

## Limitation

**从论文章节提取的局限性:**
- 局限性1
- 局限性2

**从引用分析提取的局限性:**
- 局限性3（来自后续研究）

---

## Future Work

- 未来工作方向1
- 未来工作方向2

---

## Metadata

### Extraction Methods
- **problem**: logic_analyst
- **method**: logic_analyst
- **limitation**: section_locator + citation_detective
- **future_work**: section_locator

### Confidences
- **problem**: 0.85
- **method**: 0.85
- **limitation**: 0.80
- **future_work**: 0.80
```

---

## 核心组件详解

### Reflection Loop（反思循环）机制

**问题背景**：
传统提取系统存在三大问题：
1. **Recall 问题**：提取为空，漏掉重要信息
2. **Precision 问题**：提取错误，如将 baseline 的局限性当作本文的
3. **Quality 问题**：提取内容过于宽泛，缺乏具体性

**解决方案**：
引入 CriticAgent，实现自动质量验证与迭代改进：

```python
# 伪代码示例
current_extraction = extractor.extract(paper)  # 初始提取
iteration = 0

while iteration < max_iterations:
    feedback = critic.critique(current_extraction)  # 审查

    if feedback.approved:
        return current_extraction  # 通过，返回结果

    # 不通过，使用改进指令重新提取
    current_extraction = extractor.extract(paper, feedback=feedback)
    iteration += 1

return current_extraction  # 达到最大迭代次数，返回当前最佳结果
```

**CriticAgent 的判断逻辑**：

1. **检测空结果**（Recall 问题）：
   ```python
   if len(content.strip()) < 50:
       return CriticFeedback(
           approved=False,
           feedback_type="empty_retry",
           feedback_message="提取为空，请重试",
           retry_prompt="请仔细查找 Discussion/Conclusion 末尾的内容"
       )
   ```

2. **检测错误目标**（Precision 问题）：
   ```python
   # 使用 LLM 判断提取内容是否针对本文
   llm_check = llm.check_target(content, paper)
   if llm_check == "wrong_target":
       return CriticFeedback(
           approved=False,
           feedback_type="wrong_target",
           feedback_message="提取了前人工作的局限性，而非本文的",
           retry_prompt="请关注 'our method', 'we', 'this paper' 等自指"
       )
   ```

3. **检测过于宽泛**（Quality 问题）：
   ```python
   if "generic" in llm.check_quality(content):
       return CriticFeedback(
           approved=False,
           feedback_type="too_generic",
           feedback_message="内容过于宽泛，缺乏具体性",
           retry_prompt="请提供更具体的描述，避免泛泛而谈"
       )
   ```

**价值**：
- 自动化质量控制，无需人工干预
- 显著提升 Precision、Recall 和 Quality
- ACL 级别的创新点

### Problem-Solution Pair 逻辑分析

**核心思想**：
论文的本质是"痛点"与"机制"的映射关系：
- **The Lock（痛点）**：作者试图解决的核心问题
- **The Key（机制）**：作者设计的核心解法
- **因果关系**：机制如何具体解决问题

**示例**：

```json
{
  "problem": "现有方法在处理长文本时，注意力机制的计算复杂度为 O(n²)，导致效率低下",
  "solution": "提出线性注意力机制，使用核技巧将复杂度降低到 O(n)",
  "explanation": "通过将注意力计算分解为两个独立的线性操作，避免了显式计算完整的注意力矩阵，从而将复杂度从 O(n²) 降低到 O(n)。",
  "confidence": 0.9,
  "evidence": "As shown in Section 3.2, our linear attention mechanism reduces the computational complexity..."
}
```

**对比传统方法**：
- 传统：简单罗列 Problem 和 Method，缺乏因果关系
- DeepPaper 2.0：提取 Problem-Solution Pair，自动生成解释

### 双流合并：章节分析 + 引用分析

**为什么需要引用分析？**
- 作者在论文中可能不会完全坦诚地描述局限性
- 后续研究的引用上下文中往往包含真实的评价和批评

**引用分析工作流程**：

1. **获取引用论文列表**（通过 Semantic Scholar API）
2. **筛选批判性引用**（包含转折词、负面评价的上下文）
3. **提取 Limitation**（使用 LLM 从引用上下文中提取）

**批判性关键词**：
- 转折词：however, although, but, nevertheless
- 负面评价：limitation, weakness, drawback, problem
- 能力限制：cannot, fails to, unable to

**示例**：

论文 A 的 Limitation 提取结果：

```markdown
**从论文章节提取的局限性:**
- 本文方法仅在英文数据集上进行了测试，泛化性有待验证
- 训练过程需要大量标注数据，成本较高

**从引用分析提取的局限性:**
- 后续研究指出，该方法在处理短文本时性能下降明显（来自 Paper B, 2025）
- 有研究发现，该方法对噪声数据敏感，鲁棒性不足（来自 Paper C, 2025）
```

**价值**：
- 提供更全面、客观的 Limitation 信息
- 挖掘作者未明确提及的潜在问题

---

## 特性

### 1. 智能章节定位

- 支持 LLM 智能定位和规则匹配两种模式
- 自动识别章节类型（introduction, discussion, conclusion 等）
- 特别关注 Discussion/Conclusion 末尾的转折词
- 区分"作者自述的局限性"与"对前人工作的批评"

### 2. 引用分析（可选）

- 通过 CitationDetectiveAgent 从引用论文中提取真实的 Limitation
- 支持 ArXiv ID 和 DOI
- 使用 Semantic Scholar API 获取引用信息
- 自动过滤批判性关键词（however, limitation, drawback 等）

### 3. 质量验证与反思循环（Reflection Loop）⭐

- CriticAgent 实现自动质量控制
- 检测提取问题：空结果、错误内容、过于宽泛
- 提供具体改进指令，触发迭代重试
- 持续优化直到达到质量标准或达到最大重试次数
- ACL 级别创新：自动化的 Precision、Recall 和 Quality 提升

### 4. Problem-Solution Pair 逻辑分析

- 聚焦"痛点"与"机制"的因果关系
- 自动生成解释：机制如何解决问题
- 支持多个 P-S Pairs（聚焦最核心的 1-3 个）

### 5. 双流合并

- Limitation 提取结合论文章节和引用分析两部分
- 提供更全面、客观的局限性信息

### 6. 多格式输出

- JSON 格式：便于机器处理和后续分析
- Markdown 格式：便于人类阅读和展示

---

## 与 DeepPaper 1.0 的区别

| 特性 | DeepPaper 1.0 | DeepPaper 2.0 |
|------|---------------|---------------|
| **Problem/Method 提取** | Navigator + Extractor + Critic | LogicAnalystAgent + CriticAgent（可选） |
| **逻辑分析** | 无 | Problem-Solution Pair 映射 |
| **Limitation 提取** | Navigator + Extractor | SectionLocator + CitationDetective + CriticAgent |
| **Future Work 提取** | Navigator + Extractor | SectionLocator + Extractor + CriticAgent |
| **引用分析** | 无 | 有（可选） |
| **质量验证机制** | 单轮 Critic 反馈 | 迭代式 Reflection Loop |
| **自动重试机制** | 有（Critic 反馈） | 增强（自动检测 + 具体指令 + 迭代改进） |
| **输出格式** | JSON | JSON + Markdown |
| **模块化设计** | 一般 | 高（每个 Agent 可独立运行） |

---

## 注意事项

1. **LLM 配置**：需要配置有效的 LLM 客户端（支持 OpenAI API 或兼容接口）
2. **引用分析**：需要网络连接访问 Semantic Scholar API
3. **输入格式**：确保论文 JSON 格式正确，特别是 `sections` 字段
4. **章节类型**：建议使用 GROBID 等工具解析 PDF，自动识别章节类型
5. **CriticAgent**：启用质量验证会增加 LLM 调用次数和运行时间，但能显著提升提取质量
6. **最大迭代次数**：建议设置为 2-3 次，平衡质量和效率
7. **API 限制**：
   - Semantic Scholar API 有频率限制（100 请求/5分钟）
   - LLM API 可能有 token 限制和费用
8. **论文格式**：
   - 需要预先将 PDF 解析为结构化 JSON
   - 推荐使用 GROBID 或 Science Parse 等工具

---

## TODO

- [ ] 支持批量论文处理
- [ ] 添加更多引用源（CrossRef, OpenCitations）
- [ ] 优化 LLM 提示词
- [ ] 添加置信度阈值过滤
- [ ] 支持更多输出格式（HTML, LaTeX）
- [ ] 优化 CriticAgent 的判断标准
- [ ] 添加详细的提取日志和质量报告
- [ ] 支持多语言论文（中文、法语等）
- [ ] 添加可视化界面
- [ ] 支持增量提取（避免重复分析）

---

## 贡献

欢迎提交 Issue 和 Pull Request！

### 贡献指南

1. Fork 本仓库
2. 创建新分支 (`git checkout -b feature/your-feature`)
3. 提交更改 (`git commit -am 'Add some feature'`)
4. 推送到分支 (`git push origin feature/your-feature`)
5. 创建 Pull Request

---

## 许可证

MIT License

---

## 引用

如果您使用了 DeepPaper 2.0，请引用：

```bibtex
@software{deeppaper2,
  title={DeepPaper 2.0: A Multi-Agent System for Deep Paper Information Extraction},
  author={DeepPaper Team},
  year={2024},
  url={https://github.com/your-repo/DeepPaper_Agent2.0}
}
```

---

## 联系方式

如有问题或建议，请通过以下方式联系：

- GitHub Issues: [提交 Issue](https://github.com/your-repo/DeepPaper_Agent2.0/issues)
- Email: your-email@example.com

---

## 更新日志

### v2.0.0 (2024-12-18)

- 引入 Reflection Loop 质量验证机制（CriticAgent）
- 添加 Problem-Solution Pair 逻辑分析（LogicAnalystAgent）
- 支持引用分析（CitationDetectiveAgent）
- 双流合并：章节分析 + 引用分析
- 多格式输出：JSON + Markdown
- 模块化设计，每个 Agent 可独立运行

### v1.0.0

- 基础的论文信息提取功能
- Navigator + Extractor + Critic 架构
