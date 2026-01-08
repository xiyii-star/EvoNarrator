# DeepPaper Multi-Agent System

基于迭代式多Agent架构的深度论文解析系统

## 🎯 核心特性

### 1. **Multi-Agent架构**

放弃传统的"单次RAG",采用四个专业Agent协同工作:

- **Navigator Agent (导航员)**

  - 分析论文结构,定位关键信息位置
  - 解决"找不到位置"的问题
- **Extractor Agent (提取员)**

  - 深度阅读指定章节,提取关键内容
  - 使用Sliding Window + LLM语义理解
  - 解决"关键词匹配不准"的问题
- **Critic Agent (审查员)** ⭐ **核心创新**

  - 验证提取质量,自动发现问题
  - 处理三种场景:
    - 提取为空 (Recall提升)
    - 提取错误 (Precision提升,如区分本文vs前人工作)
    - 内容太泛 (Quality提升)
  - 给出具体改进指令,指导重试
- **Synthesizer Agent (总结员)**

  - 整合所有结果,输出结构化JSON
  - 附带原文Evidence,可人工核查

### 2. **Reflection Loop (反思循环)**

实现智能重试机制，包含多重优化策略：

```
For each field (Problem/Method/Limitation/Future Work):
    Navigator → 定位章节范围
    │
    ↓
    Extractor → 提取内容
    │
    ↓
    Critic → 验证质量
    │
    ├─ 通过 ✅ → 继续下一个field
    │
    └─ 未通过 ⚠️ → 根据反馈重试 (最多max_retries次)
         │
         ├─ 检测重复提取 → 自动终止无效重试
         ├─ 扩展章节范围 → 避免重复扩展相同章节
         └─ 根据Critic反馈调整提取策略
```

**优化机制**:

- **重复检测**: 自动识别相同提取结果，避免无效迭代 (orchestrator.py:185-188)
- **智能章节扩展**: 只添加新章节，避免重复搜索 (orchestrator.py:222-232)
- **反馈驱动**: Critic提供具体改进指令，而非盲目重试

### 3. **提取四个关键字段**

- **Problem**: 研究问题/挑战
- **Method**: 主要方法/技术创新点
- **Limitation**: 局限性 (区分本文vs前人工作!)
- **Future Work**: 未来工作方向

### 4. **Evidence-based输出**

每个提取结果都附带原文引用:

```json
{
  "problem": {
    "content": "提取总结的内容",
    "evidence": [
      {
        "section": "Introduction",
        "text": "原文句子...",
        "page": 2
      }
    ]
  },
  "method": {
    "content": "主要方法和技术创新",
    "evidence": [...]
  },
  "limitation": {
    "content": "本文方法的局限性",
    "evidence": [...]
  },
  "future_work": {
    "content": "未来研究方向",
    "evidence": [...]
  }
}
```

## 📁 项目结构

```
CLwithRAG/KGdemo/
├── DeepPaper_Agent/          # Multi-Agent核心模块
│   ├── __init__.py
│   ├── data_structures.py     # 数据结构定义
│   ├── navigator_agent.py     # 导航员
│   ├── extractor_agent.py     # 提取员
│   ├── critic_agent.py        # 审查员 ⭐
│   ├── synthesizer_agent.py   # 总结员
│   └── orchestrator.py        # 协调器
│
├── src/
│   ├── DeepPaper.py           # 入口文件 ⭐
│   ├── llm_config.py          # LLM配置管理
│   ├── grobid_parser.py       # PDF解析(GROBID)
│   └── openalex_client.py     # OpenAlex API客户端
│
├── config/
│   └── config.yaml            # 统一配置文件
│
└── output/deep_paper/         # 输出目录
    ├── deep_paper_*.json      # JSON报告
    └── deep_paper_*.md        # Markdown报告
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd KGdemo
pip install -r requirements.txt
```

### 2. 配置LLM

编辑 `config/config.yaml`:

```yaml
llm:
  enabled: true
  provider: "ollama"  # 或 "openai", "anthropic"
  model: "qwen2.5:14b"
  base_url: "http://localhost:11434"
  # api_key: "your-key"  # OpenAI/Anthropic需要

grobid:
  enabled: true
  url: "http://localhost:8070"
```

**关键参数**:

- `max_retries`: Critic重试次数，默认2次 (orchestrator.py:49)
- `max_context_length`: LLM上下文窗口，默认3000 tokens (orchestrator.py:50)

### 3. 运行示例

#### 分析单篇论文 (通过OpenAlex ID)

```bash
python src/DeepPaper.py --paper-id W2741809807
```

#### 分析PDF文件

```bash
python src/DeepPaper.py --pdf data/papers/example.pdf
```

#### 批量分析主题

```bash
python src/DeepPaper.py --batch --topic "transformer" --max-papers 5
```

## 📊 输出示例

### JSON格式

```json
{
  "paper_id": "W2741809807",
  "title": "Attention Is All You Need",
  "problem": {
    "content": "Sequence transduction models based on RNNs are difficult to parallelize...",
    "evidence": [...]
  },
  "method": {
    "content": "We propose the Transformer, a novel architecture relying entirely on attention...",
    "evidence": [...]
  },
  "limitation": {
    "content": "The model requires large amounts of training data...",
    "evidence": [...]
  },
  "future_work": {
    "content": "We plan to extend the Transformer to other modalities...",
    "evidence": [...]
  },
  "metadata": {
    "extraction_quality": {
      "problem": 0.85,
      "method": 0.92,
      "limitation": 0.78,
      "future_work": 0.81
    },
    "iteration_count": {
      "problem": 1,
      "method": 1,
      "limitation": 2,
      "future_work": 1
    },
    "timestamp": "2025-01-15T10:30:00Z"
  }
}
```

### Markdown格式

自动生成人类可读的报告,包含:

- 提取的四个字段
- 原文引用(Evidence)
- 质量评分

## 🔧 高级配置

### 调整Critic重试次数

```bash
python src/DeepPaper.py --paper-id W2741809807 --max-retries 3
```

**说明**: 默认 `max_retries=2`，配合重复检测和智能章节扩展，通常能在3次内收敛

### 批量处理论文

使用 `batch_analyze_papers`方法批量处理多篇论文 (orchestrator.py:267-313):

```python
from DeepPaper_Agent import DeepPaperOrchestrator

orchestrator = DeepPaperOrchestrator(llm_client, max_retries=2)

# 批量分析
reports = orchestrator.batch_analyze_papers(
    papers=paper_list,  # OpenAlex格式的论文列表
    pdf_dir="data/papers",  # PDF文件夹(可选)
    output_dir="output/deep_paper"  # 输出目录
)
```

**特性**:

- 自动从PDF提取章节，失败则降级到摘要 (orchestrator.py:339-359)
- 支持OpenAlex格式输入
- 逐篇处理，错误隔离，不影响其他论文
- 自动保存JSON和Markdown报告

### 使用GROBID解析PDF

需要先启动GROBID服务:

```bash
docker run -d -p 8070:8070 lfoppiano/grobid:0.8.0
```

然后在运行时指定:

```bash
python src/DeepPaper.py --pdf paper.pdf --grobid-url http://localhost:8070
```

**PDF处理流程** (orchestrator.py:383-404):

1. 尝试使用GROBID提取章节结构
2. 失败则降级到简单PDF解析
3. 如无PDF，使用Title和Abstract

## 🆚 与传统RAG的对比

| 特性              | 传统RAG              | DeepPaper Multi-Agent           |
| ----------------- | -------------------- | ------------------------------- |
| 架构              | 单次向量检索+LLM     | 四Agent迭代式协同               |
| 质量控制          | 无                   | Critic自动验证+智能重试         |
| 错误处理          | 手动发现             | 自动重试机制(最多N次)           |
| 重复检测          | 无                   | 自动识别相同提取，终止无效重试  |
| 章节扩展          | 静态检索范围         | 动态扩展，只添加新章节          |
| 精度 (Limitation) | 易混淆本文/前人工作  | Critic智能区分本文vs前人        |
| 可验证性          | 无Evidence           | 附带原文引用(section+text+page) |
| Recall            | 依赖向量检索(易遗漏) | Navigator+Extractor协同定位     |
| 迭代透明度        | 无                   | 记录每个字段的迭代次数          |

## 💡 核心创新点

### 1. Critic Agent的三种场景处理

Critic Agent实现智能质量检测，针对不同问题类型提供精准反馈：

#### 场景A: 提取为空 (Recall提升)

```python
# Critic检测到空提取
if extraction.content == "未找到相关信息":
    feedback = CriticFeedback(
        feedback_type="empty_retry",
        retry_prompt="请检查Discussion和Conclusion结尾的转折词(However, Future work)...",
        suggested_sections=[5, 6]  # 扩展搜索范围
    )
```

#### 场景B: 提取错误 (Precision提升)

```python
# Critic检测到提取了baseline的limitation
if "LSTM" in extraction.content and field == "limitation":
    feedback = CriticFeedback(
        feedback_type="wrong_target",
        retry_prompt="请只提取本文方法(our/proposed)的局限性,不要提取LSTM的缺点"
    )
```

#### 场景C: 内容太泛 (Quality提升)

```python
# Critic检测到内容太笼统
if "needs more data" in extraction.content and len(extraction.content) < 50:
    feedback = CriticFeedback(
        feedback_type="too_generic",
        retry_prompt="请明确需要什么类型的数据?在什么场景下表现不佳?"
    )
```

### 2. Navigator的智能定位

Navigator不只是关键词匹配,而是:

- 读取论文目录和首尾段
- 使用LLM推理隐藏位置 (如Limitation隐藏在Conclusion的"However"后)
- 输出推理过程和章节索引
- 返回 `SectionScope`对象，包含目标章节和置信度

### 3. 智能重试优化

Orchestrator实现多重优化策略，避免无效迭代：

#### 重复检测机制 (orchestrator.py:185-188)

```python
# 检测重复提取，避免无效重试
if previous_content and current_extraction.content == previous_content:
    logger.warning("检测到重复提取，重试无效，终止循环")
    return current_extraction
```

#### 智能章节扩展 (orchestrator.py:222-232)

```python
# 只添加新章节，避免重复搜索
new_sections = [s for s in feedback.suggested_sections
                if s not in scope.target_sections]
if new_sections:
    scope.target_sections = feedback.suggested_sections
else:
    logger.info("搜索范围未变化，使用更强的retry_prompt")
```

### 4. Evidence链路

从原文到输出全程可追溯:

```
原文段落 → Extractor提取 → Critic验证 → Synthesizer整理 → JSON输出(带Evidence)
```

每个提取结果都包含：

- `content`: 提取和总结的内容
- `evidence`: 原文片段列表，包含section、text、page信息
- `confidence`: 提取置信度
- `iterations`: 实际迭代次数

## 🔬 适用场景

- ✅ 学术论文深度分析
- ✅ 文献综述自动化
- ✅ 论文知识图谱构建 (与KGdemo主流程集成)
- ✅ 研究趋势分析
- ✅ ACL/NeurIPS等顶会论文批量处理
- ✅ 需要可验证Evidence的论文信息提取

## 🏗️ 系统架构细节

### Orchestrator工作流程

[orchestrator.py](DeepPaper_Agent/orchestrator.py:84-143) 实现完整的分析流程：

1. **初始化** (orchestrator.py:46-74)

   - 创建四个Agent实例
   - 配置 `max_retries`和 `max_context_length`
   - 定义提取字段列表
2. **逐字段处理** (orchestrator.py:111-122)

   ```python
   for field in [PROBLEM, METHOD, LIMITATION, FUTURE_WORK]:
       extraction = self._extract_field_with_retry(paper, field)
   ```
3. **Reflection Loop** (orchestrator.py:145-237)

   - Navigator定位章节 → Extractor提取 → Critic验证
   - 通过则继续，不通过则重试
   - 重复检测和智能章节扩展优化
4. **结果整合** (orchestrator.py:129-133)

   - Synthesizer生成最终报告
   - 包含所有字段+Evidence+元数据
5. **报告保存** (orchestrator.py:239-265)

   - JSON格式 (机器可读)
   - Markdown格式 (人类可读)

### 核心数据结构

详见 [data_structures.py](DeepPaper_Agent/data_structures.py):

- **`PaperDocument`**: 论文完整结构，包含sections列表
- **`SectionScope`**: Navigator返回的章节范围，包含推理过程
- **`ExtractionResult`**: Extractor提取结果，包含content+evidence
- **`CriticFeedback`**: Critic反馈，包含feedback_type和retry_prompt
- **`FinalReport`**: 最终输出，四个字段+evidence+metadata

### 关键优化点总结

1. **重复检测** (orchestrator.py:185-188): 比较 `previous_content`和 `current_extraction.content`
2. **智能扩展** (orchestrator.py:222-232): 只添加不在 `target_sections`中的新章节
3. **迭代记录** (orchestrator.py:122, 202): 每个字段记录实际迭代次数
4. **降级处理** (orchestrator.py:339-359): PDF失败自动降级到Abstract

## 🛠 故障排查

### 问题1: LLM连接失败

```bash
# 检查LLM服务是否运行
curl http://localhost:11434/api/tags  # Ollama

# 检查配置文件
cat config/config.yaml
```

### 问题2: PDF解析失败

```bash
# 如果GROBID不可用,系统会自动降级到PyPDF2
# 建议使用GROBID获得更好的章节识别

# 启动GROBID:
docker run -d -p 8070:8070 lfoppiano/grobid:0.8.0
```

### 问题3: Critic总是不通过

```bash
# 调整max_retries避免过度迭代
python src/DeepPaper.py --paper-id W2741809807 --max-retries 1

# 查看详细日志
python src/DeepPaper.py --paper-id W2741809807 2>&1 | tee deep_paper.log
```

**说明**:

- 系统已实现重复检测 (orchestrator.py:185)，会自动终止无效重试
- 如果连续重试都提取到相同内容，会自动终止循环
- 建议保持默认 `max_retries=2`，配合智能优化通常足够

### 问题4: 批量处理中断

```bash
# 批量处理时单篇失败不影响其他论文
# 检查日志查看具体失败原因
grep "❌" deep_paper.log

# 查看成功率统计
grep "成功:" deep_paper.log
```

## 📚 相关文档

- [orchestrator.py](DeepPaper_Agent/orchestrator.py) - 协调器核心实现
- [data_structures.py](DeepPaper_Agent/data_structures.py) - 数据结构定义
- 各Agent实现: [navigator_agent.py](DeepPaper_Agent/navigator_agent.py), [extractor_agent.py](DeepPaper_Agent/extractor_agent.py), [critic_agent.py](DeepPaper_Agent/critic_agent.py), [synthesizer_agent.py](DeepPaper_Agent/synthesizer_agent.py)

## 🤝 贡献

欢迎提交Issue和Pull Request!

**改进方向**:

- 支持更多字段提取 (实验结果、数据集等)
- 优化Critic的判断逻辑
- 支持多语言论文
- 集成更多PDF解析器

## 📄 许可证

MIT License

## 🙏 致谢

- OpenAlex API - 论文元数据
- GROBID - 学术PDF解析
- LangChain/AutoGen - Multi-Agent灵感来源

---

**DeepPaper v1.0** - 深度、准确、可验证的论文分析系统

**核心特性**:

- ✅ 四Agent协同 (Navigator→Extractor→Critic→Synthesizer)
- ✅ 智能重试优化 (重复检测+动态章节扩展)
- ✅ Evidence-based输出 (可追溯的原文引用)
- ✅ 批量处理支持 (错误隔离+自动降级)
