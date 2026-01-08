"""
科研假设生成器
使用思维链（Chain of Thought）推理来生成可行的研究创意
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import networkx as nx

try:
    from langchain_openai import ChatOpenAI
    try:
        # 尝试新版 langchain 结构 (v0.1.0+)
        from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
        from langchain_core.output_parsers import PydanticOutputParser
    except ImportError:
        # 回退到旧版 langchain 结构
        from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
        from langchain.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field
except ImportError as e:
    raise ImportError(
        "缺少必需的包。请运行: pip install langchain langchain-openai langchain-core pydantic"
    ) from e

# 日志记录器
logger = logging.getLogger(__name__)


class IdeaStatus(str, Enum):
    """生成创意的状态"""
    SUCCESS = "SUCCESS"  # 成功：方法与限制兼容
    INCOMPATIBLE = "INCOMPATIBLE"  # 不兼容：方法无法解决该限制


class InnovationIdea(BaseModel):
    """生成的研究创意的结构化输出"""
    status: IdeaStatus = Field(description="方法是否与限制兼容")
    title: Optional[str] = Field(default=None, description="吸引人的学术标题")
    abstract: Optional[str] = Field(
        default=None,
        description="标准学术摘要（背景 → 差距 → 提出的方法 → 预期结果）"
    )
    modification: Optional[str] = Field(
        default=None,
        description="所需的具体修改（即'桥接变量'）"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="展示分析过程的思维链推理"
    )
    rationale: Optional[str] = Field(
        default=None,
        description="完整的推演路径，包括Step 1-3的详细分析过程、推理链条和决策依据"
    )


@dataclass
class IdeaFragment:
    """研究片段（限制或方法）"""
    content: str  # 片段内容
    paper_id: str = ""  # 论文ID
    paper_title: str = ""  # 论文标题
    year: int = 0  # 发表年份
    cited_count: int = 0  # 被引用次数


class HypothesisGenerator:
    """
    假设生成器，使用思维链（Chain of Thought）推理

    处理流程:
    1. 分析兼容性：检查数学/理论兼容性
    2. 识别差距：确定需要什么修改
    3. 起草创意：生成结构化的研究提案
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.3,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        初始化假设生成器

        Args:
            model_name: OpenAI 模型名称（例如："gpt-4", "gpt-3.5-turbo"）
            temperature: 采样温度（越低越聚焦，越高越有创造性）
            api_key: OpenAI API 密钥（可选，默认使用 OPENAI_API_KEY 环境变量）
            base_url: API的可选基础URL（用于代理或自定义端点）
        """
        # 构建 LLM 参数
        llm_kwargs = {
            "model": model_name,
            "temperature": temperature,
        }

        if api_key:
            llm_kwargs["api_key"] = api_key
        if base_url:
            llm_kwargs["base_url"] = base_url

        # 初始化 OpenAI LLM 客户端
        self.llm = ChatOpenAI(**llm_kwargs)
        # 初始化输出解析器，用于解析结构化输出
        self.output_parser = PydanticOutputParser(pydantic_object=InnovationIdea)

        # 构建提示词模板
        self.prompt_template = self._build_prompt_template()

        logger.info(f"假设生成器已使用模型初始化: {model_name}")

    def _format_evolutionary_context(self, evolutionary_paths: Optional[List[Dict]] = None) -> str:
        """
        格式化演化路径为提示词上下文
        
        Args:
            evolutionary_paths: 演化路径列表，每个路径包含：
                - thread_type: 'chain', 'divergence', 'convergence'
                - pattern_type: 模式描述
                - title: 标题
                - narrative: 叙事文本
                - papers: 论文信息列表
                - relation_chain: 关系链（包含 Overcomes, Extends, Realizes 等）
                - routes: 对于分化/汇聚模式，包含多条路线
        
        Returns:
            格式化的演化上下文字符串
        """
        if not evolutionary_paths or len(evolutionary_paths) == 0:
            return ""
        
        context_parts = [
            "**EVOLUTIONARY CONTEXT (Learn from Research Evolution Patterns):**",
            "",
            "The following evolutionary paths show how research has evolved in this field. ",
            "Learn from these patterns to generate better research ideas:",
            ""
        ]
        
        for i, path in enumerate(evolutionary_paths[:3], 1):  # 最多使用前3条路径
            thread_type = path.get('thread_type', 'unknown')
            pattern_type = path.get('pattern_type', 'Unknown Pattern')
            title = path.get('title', '')
            narrative = path.get('narrative', '')
            relation_chain = path.get('relation_chain', [])
            
            context_parts.append(f"**Evolutionary Path {i}: {pattern_type}**")
            context_parts.append(f"Title: {title}")
            context_parts.append(f"Narrative: {narrative[:500]}...")  # 限制长度
            
            # 提取关键演化逻辑
            if relation_chain:
                context_parts.append("Key Evolution Logic:")
                for rel in relation_chain[:3]:  # 最多显示3个关系
                    from_paper = rel.get('from_paper', {}).get('title', 'Unknown')
                    to_paper = rel.get('to_paper', {}).get('title', 'Unknown')
                    relation_type = rel.get('relation_type', 'Unknown')
                    narrative_rel = rel.get('narrative_relation', relation_type)
                    
                    context_parts.append(
                        f"  - {from_paper[:60]}... "
                        f"--[{narrative_rel}]--> "
                        f"{to_paper[:60]}..."
                    )
            
            # 对于分化/汇聚模式，提取路线信息
            if thread_type in ['divergence', 'convergence']:
                routes = path.get('routes', [])
                if routes:
                    context_parts.append(f"Routes ({len(routes)} branches):")
                    for route_idx, route in enumerate(routes[:2], 1):  # 最多显示2条路线
                        route_papers = route.get('papers', [])
                        if route_papers:
                            first_paper = route_papers[0].get('title', 'Unknown')[:50]
                            context_parts.append(f"  Route {route_idx}: {first_paper}...")
            
            context_parts.append("")
        
        context_parts.append(
            "**Learning Instructions:** "
            "Analyze how limitations were overcome, how methods were extended, "
            "and how cross-domain adaptations worked. Apply similar logic to generate your idea."
        )
        
        return "\n".join(context_parts)

    def _build_prompt_template(self) -> ChatPromptTemplate:
        """构建思维链提示词模板（支持演化路径上下文）"""

        system_message = SystemMessagePromptTemplate.from_template(
            """You are a **Senior Principal Researcher** with deep expertise in analyzing research problems and generating innovative solutions.

Your task is to evaluate whether a candidate method can solve a given research limitation, following a rigorous Chain of Thought reasoning process.

**Your reasoning must follow these three steps:**

**Step 1: Analyze Compatibility**
- Examine the mathematical, algorithmic, and theoretical properties of the method
- Check if these properties align with the constraints and requirements of the limitation
- Consider: computational complexity, applicability domain, underlying assumptions
- If fundamentally incompatible, output status="INCOMPATIBLE" and stop

**Step 2: Identify the Gap**
- Determine what specific modifications are needed to bridge the gap
- Identify the "Bridging Variable" - the key innovation that makes the connection work
- Ask: What needs to change in the method to address this new problem context?

**Step 3: Draft the Idea**
- Create a catchy, academic title
- Write a structured abstract following: Background → Gap → Proposed Method → Expected Result
- Clearly state the core innovation in one sentence

**Rationale (Complete Reasoning Path):**
You must provide a comprehensive rationale that documents the complete reasoning path, including:
- **Step 1 Analysis**: Detailed compatibility analysis with specific technical considerations
- **Step 2 Analysis**: Detailed gap identification with the bridging variable explanation
- **Step 3 Analysis**: Detailed idea drafting process and innovation justification
- **Decision Chain**: The logical chain of reasoning from limitation → method compatibility → gap → solution
- **Evidence**: Key evidence or patterns from evolutionary paths (if provided) that support the reasoning

The rationale should be a complete, structured narrative that allows readers to understand the full reasoning process behind the generated idea.

**Evolutionary Context (if provided):**
When evolutionary paths are provided, you should:
- Learn from the evolution patterns (Chain, Divergence, Convergence)
- Understand how previous research has evolved: what limitations were overcome, what methods were extended
- Apply similar evolutionary logic: if Method A overcame Limitation X, and Method B extends Method A, consider how Method B might address similar limitations
- Identify patterns: successful combinations, common adaptation strategies, cross-domain transfers

{format_instructions}

Be rigorous and honest. If something won't work, say INCOMPATIBLE. Only output SUCCESS for truly feasible ideas."""
        )

        human_message = HumanMessagePromptTemplate.from_template(
            """**LIMITATION (Current Research Bottleneck):**
{limitation}

**CANDIDATE METHOD (Potential Solution):**
{method}

{evolutionary_context}

Now, follow the three-step Chain of Thought process:

1. **Compatibility Analysis**: Are the mathematical/algorithmic properties of this method suitable for the limitation's constraints?

2. **Gap Identification**: What specific modification or adaptation is needed? Consider evolutionary patterns if provided.

3. **Idea Drafting**: If feasible, create the title, abstract, and describe the core innovation. Learn from how similar problems were solved in the evolutionary paths.

**Important**: You must provide a comprehensive "rationale" field that documents the complete reasoning path, including detailed analysis for each step, the decision chain, and supporting evidence. This rationale should be a complete narrative that explains the full thought process.

Provide your complete reasoning and final output in the specified JSON format."""
        )

        return ChatPromptTemplate.from_messages([system_message, human_message])

    def generate_innovation_idea(
        self,
        limitation: str,
        method: str,
        evolutionary_paths: Optional[List[Dict]] = None,
        verbose: bool = False
    ) -> Dict:
        """
        从限制和候选方法生成研究创新创意

        Args:
            limitation: 研究瓶颈/限制的描述
            method: 候选方法的描述
            evolutionary_paths: 可选的演化路径列表，用于提供上下文和学习演化逻辑
            verbose: 如果为True，打印详细的推理过程

        Returns:
            包含以下结构的字典:
            {
                "status": "SUCCESS" 或 "INCOMPATIBLE",
                "title": "...",
                "abstract": "...",
                "modification": "...",
                "reasoning": "...",
                "rationale": "..."  # 完整的推演路径，包括Step 1-3的详细分析
            }
        """
        try:
            # 格式化演化上下文
            evolutionary_context = self._format_evolutionary_context(evolutionary_paths)
            
            # 格式化提示词
            formatted_prompt = self.prompt_template.format_messages(
                limitation=limitation,
                method=method,
                evolutionary_context=evolutionary_context,
                format_instructions=self.output_parser.get_format_instructions()
            )

            if verbose:
                logger.info("=" * 80)
                logger.info("正在生成创新创意...")
                logger.info(f"限制: {limitation[:100]}...")
                logger.info(f"方法: {method[:100]}...")

            # 调用 LLM
            response = self.llm.invoke(formatted_prompt)

            # 解析结构化输出
            idea = self.output_parser.parse(response.content)

            if verbose:
                logger.info(f"状态: {idea.status}")
                if idea.status == IdeaStatus.SUCCESS:
                    logger.info(f"标题: {idea.title}")
                    logger.info(f"修改: {idea.modification}")
                    if idea.rationale:
                        logger.info(f"推演路径长度: {len(idea.rationale)} 字符")
                logger.info("=" * 80)

            # 转换为字典
            result = {
                "status": idea.status,
                "title": idea.title,
                "abstract": idea.abstract,
                "modification": idea.modification,
                "reasoning": idea.reasoning,
                "rationale": idea.rationale  # 完整的推演路径
            }

            return result

        except Exception as e:
            logger.error(f"生成创新创意时出错: {e}")
            return {
                "status": "ERROR",
                "title": None,
                "abstract": None,
                "modification": None,
                "reasoning": f"生成过程中出错: {str(e)}",
                "rationale": None
            }

    def batch_generate(
        self,
        unsolved_limitations: List[str],
        candidate_methods: List[str],
        evolutionary_paths: Optional[List[Dict]] = None,
        max_ideas: int = 10,
        verbose: bool = False
    ) -> List[Dict]:
        """
        通过将限制与方法配对来生成多个创意

        Args:
            unsolved_limitations: 限制描述列表
            candidate_methods: 方法描述列表
            evolutionary_paths: 可选的演化路径列表，用于提供上下文和学习演化逻辑
            max_ideas: 要生成的最大创意数量
            verbose: 如果为True，打印进度

        Returns:
            生成的创意列表（仅成功的创意）
        """
        ideas = []
        count = 0

        for limitation in unsolved_limitations:
            if count >= max_ideas:
                break

            for method in candidate_methods:
                if count >= max_ideas:
                    break

                if verbose:
                    logger.info(f"\n正在生成创意 {count + 1}/{max_ideas}...")

                idea = self.generate_innovation_idea(
                    limitation, 
                    method, 
                    evolutionary_paths=evolutionary_paths,
                    verbose=False
                )

                # 仅保留成功的创意
                if idea["status"] == "SUCCESS":
                    ideas.append({
                        "limitation": limitation,
                        "method": method,
                        **idea
                    })
                    count += 1

                    if verbose:
                        logger.info(f"✓ 成功: {idea['title']}")
                else:
                    if verbose:
                        logger.info(f"✗ 不兼容: 方法不合适")

        return ideas


class KnowledgeGraphExtractor:
    """
    知识图谱数据提取器

    从包含论文信息的知识图谱中提取研究局限性(Limitations)和候选方法(Methods)。
    这些提取的内容将用于研究创意生成,通过将未解决的局限性与候选方法进行组合来产生新的研究方向。

    工作原理:
        1. 遍历知识图谱中的所有节点
        2. 从节点属性中提取 limitations (研究瓶颈/需要解决的问题)
        3. 从节点属性中提取 methods (潜在的解决方案/贡献)
        4. 对提取的内容进行过滤和去重

    使用场景:
        - 在生成研究创意之前,从已构建的文献知识图谱中提取原材料
        - 为 HypothesisGenerator 准备输入数据
    """

    @staticmethod
    def extract_from_graph(
        graph: nx.Graph,
        min_text_length: int = 50
    ) -> tuple[List[str], List[str]]:
        """
        从 NetworkX 知识图谱中提取局限性和方法（基于引用关系类型的碎片池化）

        🔌 碎片池化策略 (Fragment Pooling)：
        通过分析引用关系类型（Socket Matching 的结果），智能筛选高质量的研究碎片。

        📦 四大碎片池：
        - Pool A (Unsolved Limitations): 未被 Overcomes 的 Limitation
          → 这些是尚未解决的研究瓶颈，最值得攻克
        - Pool B (Successful Methods): 被 Extends 多次的 Method
          → 这些方法被多次扩展，证明是成熟可靠的基础技术
        - Pool C (Cross-Domain Methods): 来自 Adapts_to 源头的 Method
          → 这些方法已证明具有跨领域迁移能力，适合新场景
        - Pool D (Unrealized Future Work): 未被 Realizes 的 Future Work
          → 这些是前人设想但尚未实现的研究方向

        🔗 Limitation 来源：Pool A + Pool D
        🔧 Method 来源：Pool B + Pool C

        Args:
            graph: NetworkX 图对象，节点包含论文信息，边包含引用关系类型
                   预期的节点属性:
                   - rag_limitation (str): RAG 提取的局限性
                   - rag_future_work (str): RAG 提取的未来工作
                   - rag_method (str): RAG 提取的贡献/方法
                   预期的边属性:
                   - edge_type (str): 引用关系类型 (Overcomes, Realizes, Extends, etc.)
            min_text_length: 有效文本的最小长度，默认 50 字符

        Returns:
            tuple[List[str], List[str]]:
                - unsolved_limitations: 高质量的未解决局限性列表（Pool A + Pool D）
                - candidate_methods: 高质量的候选方法列表（Pool B + Pool C）

        Example:
            >>> G = nx.Graph()
            >>> G.add_node('W1', rag_limitation='High complexity', rag_method='Method A')
            >>> G.add_node('W2', rag_method='Method B')
            >>> G.add_edge('W2', 'W1', edge_type='Extends')
            >>> limitations, methods = KnowledgeGraphExtractor.extract_from_graph(G)
        """
        logger.info("🔌 开始碎片池化提取 (Fragment Pooling based on Socket Matching)")

        # ===== 统计边类型信息 =====
        # 统计每个节点被哪些类型的边指向，以及指向哪些节点
        node_incoming_edges = {}  # 节点被哪些边指向 {node_id: [(source, edge_type), ...]}
        node_outgoing_edges = {}  # 节点指向哪些边 {node_id: [(target, edge_type), ...]}

        for source, target, edge_data in graph.edges(data=True):
            edge_type = edge_data.get('edge_type', 'Unknown')

            # 记录 target 被 source 通过 edge_type 引用
            if target not in node_incoming_edges:
                node_incoming_edges[target] = []
            node_incoming_edges[target].append((source, edge_type))

            # 记录 source 通过 edge_type 引用了 target
            if source not in node_outgoing_edges:
                node_outgoing_edges[source] = []
            node_outgoing_edges[source].append((target, edge_type))

        # ===== Pool A: Unsolved Limitations (未被 Overcomes 的 Limitation) =====
        pool_a_limitations = []

        for node_id, node_data in graph.nodes(data=True):
            # 提取 limitation
            limitation_text = node_data.get('rag_limitation', '')
            if not isinstance(limitation_text, str) or len(limitation_text.strip()) <= min_text_length:
                continue

            # 检查是否被 Overcomes
            incoming_edges = node_incoming_edges.get(node_id, [])
            is_overcome = any(edge_type == 'Overcomes' for _, edge_type in incoming_edges)

            if not is_overcome:
                # 未被解决的 limitation
                pool_a_limitations.append(limitation_text.strip())

        logger.info(f"📦 Pool A (Unsolved Limitations): {len(pool_a_limitations)} 条")

        # ===== Pool D: Unrealized Future Work (未被 Realizes 的 Future Work) =====
        pool_d_limitations = []

        for node_id, node_data in graph.nodes(data=True):
            # 提取 future_work
            future_work_text = node_data.get('rag_future_work', '')
            if not isinstance(future_work_text, str) or len(future_work_text.strip()) <= min_text_length:
                continue

            # 检查是否被 Realizes
            incoming_edges = node_incoming_edges.get(node_id, [])
            is_realized = any(edge_type == 'Realizes' for _, edge_type in incoming_edges)

            if not is_realized:
                # 未实现的 future work
                pool_d_limitations.append(future_work_text.strip())

        logger.info(f"📦 Pool D (Unrealized Future Work): {len(pool_d_limitations)} 条")

        # ===== Pool B: Successful Methods (被 Extends 多次的 Method) =====
        pool_b_methods = []
        extends_threshold = 2  # 至少被 Extends 2 次才算成熟方法

        for node_id, node_data in graph.nodes(data=True):
            # 提取 contribution (method)
            contribution_text = node_data.get('rag_method', '')
            if not isinstance(contribution_text, str) or len(contribution_text.strip()) <= min_text_length:
                continue

            # 统计被 Extends 的次数
            incoming_edges = node_incoming_edges.get(node_id, [])
            extends_count = sum(1 for _, edge_type in incoming_edges if edge_type == 'Extends')

            if extends_count >= extends_threshold:
                # 被多次扩展的成熟方法
                pool_b_methods.append(contribution_text.strip())

        logger.info(f"📦 Pool B (Successful Methods, Extends≥{extends_threshold}): {len(pool_b_methods)} 条")

        # ===== Pool C: Cross-Domain Methods (来自 Adapts_to 源头的 Method) =====
        pool_c_methods = []

        # 找出所有 Adapts_to 边的源节点
        adapts_to_sources = set()
        for source, target, edge_data in graph.edges(data=True):
            if edge_data.get('edge_type') == 'Adapts_to':
                adapts_to_sources.add(target)  # target 是被迁移的源论文

        # 提取这些源节点的 method
        for node_id in adapts_to_sources:
            node_data = graph.nodes[node_id]
            contribution_text = node_data.get('rag_method', '')
            if isinstance(contribution_text, str) and len(contribution_text.strip()) > min_text_length:
                pool_c_methods.append(contribution_text.strip())

        logger.info(f"📦 Pool C (Cross-Domain Methods from Adapts_to): {len(pool_c_methods)} 条")

        # ===== 合并池化结果 =====
        # Limitations = Pool A + Pool D
        unsolved_limitations = pool_a_limitations + pool_d_limitations
        # Methods = Pool B + Pool C
        candidate_methods = pool_b_methods + pool_c_methods

        # 去重
        unsolved_limitations = list(set(unsolved_limitations))
        candidate_methods = list(set(candidate_methods))

        # ===== 降级策略：如果碎片池化结果不足，补充传统方法 =====
        if len(unsolved_limitations) < 3 or len(candidate_methods) < 3:
            logger.warning("⚠️ 碎片池化结果不足，启用降级策略（补充传统提取）")
            fallback_limitations, fallback_methods = KnowledgeGraphExtractor._fallback_extract(
                graph, min_text_length
            )

            # 补充到现有池中
            unsolved_limitations.extend(fallback_limitations)
            candidate_methods.extend(fallback_methods)

            # 再次去重
            unsolved_limitations = list(set(unsolved_limitations))
            candidate_methods = list(set(candidate_methods))

            logger.info(f"  补充后 Limitations: {len(unsolved_limitations)} 条")
            logger.info(f"  补充后 Methods: {len(candidate_methods)} 条")

        # 输出最终统计
        logger.info(f"\\n✅ 碎片池化完成:")
        logger.info(f"  📊 Limitations (Pool A + Pool D): {len(unsolved_limitations)} 条")
        logger.info(f"  🔧 Methods (Pool B + Pool C): {len(candidate_methods)} 条")

        return unsolved_limitations, candidate_methods

    @staticmethod
    def _fallback_extract(
        graph: nx.Graph,
        min_text_length: int = 50
    ) -> tuple[List[str], List[str]]:
        """
        降级提取策略：当碎片池化结果不足时，使用传统方法补充

        简单地从所有节点提取 limitation 和 contribution，不考虑引用关系

        Args:
            graph: NetworkX 图对象
            min_text_length: 最小文本长度

        Returns:
            tuple[List[str], List[str]]: (limitations, methods)
        """
        fallback_limitations = []
        fallback_methods = []

        for _, node_data in graph.nodes(data=True):
            # 提取 limitation
            limitation_text = node_data.get('rag_limitation', '')
            if isinstance(limitation_text, str) and len(limitation_text.strip()) > min_text_length:
                fallback_limitations.append(limitation_text.strip())

            # 提取 contribution
            contribution_text = node_data.get('rag_method', '')
            if isinstance(contribution_text, str) and len(contribution_text.strip()) > min_text_length:
                fallback_methods.append(contribution_text.strip())

        return list(set(fallback_limitations)), list(set(fallback_methods))


class ResearchIdeaGenerator:
    """
    研究创意生成器 - 两步流程：获取 → 生成

    📋 核心流程（两步）：
    ┌────────────────────────────────────────────────────────────┐
    │  Step 1: 获取 Limitation 和 Method                         │
    │  ──────────────────────────────────────────────────────   │
    │  - KnowledgeGraphExtractor.extract_from_graph()           │
    │  - 碎片池化：基于引用关系类型（Socket Matching）           │
    │  - Pool A: 未被 Overcomes 的 Limitation                   │
    │  - Pool B: 被 Extends ≥2 次的 Method                      │
    │  - Pool C: 来自 Adapts_to 的 Method                       │
    │  - Pool D: 未被 Realizes 的 Future Work                   │
    └────────────────────────────────────────────────────────────┘
                                ↓
    ┌────────────────────────────────────────────────────────────┐
    │  Step 2: 创意生成（含演化路径学习 + 自动过滤）              │
    │  ──────────────────────────────────────────────────────   │
    │  - HypothesisGenerator.batch_generate()                   │
    │  - Limitation × Method 笛卡尔积                            │
    │  - Chain of Thought 推理：                                 │
    │    1. Compatibility Analysis（兼容性分析）                 │
    │    2. Gap Identification（差距识别）                       │
    │    3. Idea Drafting（创意草拟）                            │
    │  - 演化路径学习（新增）：                                   │
    │    * 整合 deep_survey_analyzer 的演化路径结果              │
    │    * 学习演化脉络逻辑（Chain/Divergence/Convergence）       │
    │    * 学习如何组合 Limitation 和 Method                     │
    │    * 参考历史成功案例的演化模式                             │
    │  - 自动过滤：只保留 status="SUCCESS" 的创意                │
    └────────────────────────────────────────────────────────────┘

    架构设计:
        ResearchIdeaGenerator (高层接口 - 协调两步流程)
            ├── KnowledgeGraphExtractor (Step 1: 获取)
            │   └── extract_from_graph() - 碎片池化
            └── HypothesisGenerator (Step 2: 生成)
                └── batch_generate() - CoT 推理 + 自动过滤

    使用场景:
        - 文献综述后的创意生成
        - 从知识图谱发现研究机会
        - 批量生成和筛选研究假设

    Example:
        >>> # 初始化生成器
        >>> config = {'model_name': 'gpt-4o', 'max_ideas': 10}
        >>> generator = ResearchIdeaGenerator(config=config)
        >>>
        >>> # 从知识图谱生成创意（两步流程自动执行）
        >>> result = generator.generate_from_knowledge_graph(
        ...     graph=citation_graph,
        ...     topic="Transformer Optimization"
        ... )
        >>>
        >>> # 查看结果
        >>> print(f"Step 1: {result['pools']['unsolved_limitations']} limitations")
        >>> print(f"Step 1: {result['pools']['candidate_methods']} methods")
        >>> print(f"Step 2: {result['successful_ideas']} successful ideas")
    """

    def __init__(
        self,
        config: Dict = None,
        llm_client = None,
        critic_agent = None
    ):
        """
        初始化研究创意生成器

        注意:
            - 为了保持向后兼容性,保留了 llm_client 和 critic_agent 参数
            - 这些参数在当前实现中被忽略,因为使用了新的 HypothesisGenerator
            - 如果你是新用户,只需要传入 config 参数即可

        Args:
            config: 配置字典,支持以下键值:
                - model_name (str): OpenAI 模型名称,默认 'gpt-4o'
                  支持: gpt-4o, gpt-4, gpt-3.5-turbo 等
                - temperature (float): 采样温度,默认 0.3
                  范围: 0.0-1.0 (越低越确定,越高越有创造性)
                - openai_api_key (str): OpenAI API 密钥
                  如果不提供,将使用环境变量 OPENAI_API_KEY
                - openai_base_url (str): API 基础 URL (可选)
                  用于代理或自定义端点
                - max_ideas (int): 最大生成创意数量,默认 10
            llm_client: (已废弃) 旧版 LLM 客户端,保留用于向后兼容
            critic_agent: (已废弃) 旧版评判代理,保留用于向后兼容

        """
        # 加载配置,如果未提供则使用空字典
        self.config = config or {}

        # ===== 提取 OpenAI 相关配置 =====
        # 从 config 中提取各项配置,如果不存在则使用默认值
        # 优先从 llm 节点读取配置，如果没有则从顶层读取
        llm_config = self.config.get('llm', {})
        model_name = llm_config.get('model', self.config.get('model_name', 'gpt-4o'))  # 默认使用 gpt-4o
        temperature = llm_config.get('temperature', self.config.get('temperature', 0.3))  # 默认温度 0.3
        api_key = llm_config.get('api_key') or self.config.get('openai_api_key')  # API 密钥 (可选)
        base_url = llm_config.get('base_url') or self.config.get('openai_base_url')  # 基础 URL (可选)

        # 如果仍然没有 API key，尝试从环境变量读取
        if not api_key:
            import os
            api_key = os.getenv('OPENAI_API_KEY')

        # ===== 初始化核心组件 =====
        # 创建 HypothesisGenerator 实例,这是实际执行创意生成的核心组件
        self.hypothesis_generator = HypothesisGenerator(
            model_name=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url
        )

        # 设置最大创意数量限制,避免生成过多创意
        # 优先从 research_idea.max_ideas 读取，其次从顶层 max_ideas，最后默认10
        research_idea_config = self.config.get('research_idea', {})
        self.max_ideas = research_idea_config.get('max_ideas', self.config.get('max_ideas', 10))

        # 创建知识图谱提取器实例,用于从图谱中提取数据
        self.kg_extractor = KnowledgeGraphExtractor()

        logger.info(f"ResearchIdeaGenerator initialized with HypothesisGenerator (max_ideas={self.max_ideas})")

    def generate_from_knowledge_graph(
        self,
        graph: nx.Graph,
        topic: str = "",
        min_text_length: int = 50,
        evolutionary_paths: Optional[List[Dict]] = None,
        verbose: bool = True
    ) -> Dict:
        """
        从知识图谱直接生成研究创意（两步流程）

        📋 整体流程：
        ┌────────────────────────────────────────────────────────────┐
        │  Step 1: 获取 Limitation 和 Method                         │
        │  ──────────────────────────────────────────────────────   │
        │  输入：知识图谱（含引用关系类型）                           │
        │  处理：KnowledgeGraphExtractor.extract_from_graph()       │
        │  - Pool A: 未被 Overcomes 的 Limitation                   │
        │  - Pool B: 被 Extends ≥2 次的 Method                      │
        │  - Pool C: 来自 Adapts_to 的 Method                       │
        │  - Pool D: 未被 Realizes 的 Future Work                   │
        │  输出：(unsolved_limitations, candidate_methods)           │
        └────────────────────────────────────────────────────────────┘
                                    ↓
        ┌────────────────────────────────────────────────────────────┐
        │  Step 2: 创意生成（含演化路径学习 + 自动过滤）              │
        │  ──────────────────────────────────────────────────────   │
        │  输入：Limitation × Method 笛卡尔积 + 演化路径（可选）      │
        │  处理：HypothesisGenerator.batch_generate()               │
        │  - 兼容性分析 (Compatibility Analysis)                     │
        │  - 差距识别 (Gap Identification)                           │
        │  - 创意草拟 (Idea Drafting)                                │
        │  - 演化路径学习（新增）：                                   │
        │    * 学习演化脉络逻辑（Chain/Divergence/Convergence）       │
        │    * 学习如何组合 Limitation 和 Method                     │
        │    * 参考历史成功案例的演化模式                             │
        │  - 自动过滤：只保留 status="SUCCESS" 的创意                │
        │  输出：高质量可行创意列表                                   │
        └────────────────────────────────────────────────────────────┘

        Args:
            graph: NetworkX 图对象，节点应包含论文信息
                   必需的节点属性（碎片池化）:
                   - rag_limitation (str): RAG 提取的局限性
                   - rag_future_work (str): RAG 提取的未来工作
                   - rag_method (str): RAG 提取的贡献/方法
                   必需的边属性（碎片池化）:
                   - edge_type (str): 引用关系类型 (Overcomes, Realizes, Extends, Adapts_to)
            topic: 研究主题，用于记录和输出（可选）
            min_text_length: 文本最小长度阈值，默认 50
                            用于过滤过短的文本片段
            evolutionary_paths: 可选的演化路径列表（来自 deep_survey_analyzer），
                               用于提供演化上下文和学习演化逻辑
            verbose: 是否输出详细日志，默认 True

        Returns:
            Dict: 包含生成结果和统计信息的字典
                {
                    "topic": str,                    # 研究主题
                    "total_ideas": int,              # Step 2 生成的总创意数
                    "successful_ideas": int,         # Step 2 过滤后的可行创意数
                    "ideas": List[Dict],             # 可行创意列表（只含 SUCCESS）
                    "pools": {
                        "unsolved_limitations": int, # Step 1 提取的局限性数量
                        "candidate_methods": int     # Step 1 提取的方法数量
                    }
                }

        Error Handling:
            - 空图谱：返回空结果字典
            - Step 1 数据不足（limitations 或 methods 为空）：返回空结果字典并警告

        Example:
            >>> # 初始化生成器
            >>> generator = ResearchIdeaGenerator(config={'max_ideas': 10})
            >>>
            >>> # 从知识图谱生成创意（两步流程自动执行）
            >>> # 方式1: 不使用演化路径（向后兼容）
            >>> result = generator.generate_from_knowledge_graph(
            ...     graph=citation_graph,
            ...     topic="Transformer Optimization"
            ... )
            >>>
            >>> # 方式2: 使用演化路径（推荐，更智能）
            >>> from deep_survey_analyzer import DeepSurveyAnalyzer
            >>> analyzer = DeepSurveyAnalyzer(config)
            >>> deep_survey_result = analyzer.analyze(citation_graph, "Transformer Optimization")
            >>> evolutionary_paths = deep_survey_result.get('evolutionary_paths', [])
            >>> 
            >>> result = generator.generate_from_knowledge_graph(
            ...     graph=citation_graph,
            ...     topic="Transformer Optimization",
            ...     evolutionary_paths=evolutionary_paths  # 整合演化路径
            ... )
            >>>
            >>> # 查看结果
            >>> print(f"Step 1: 提取了 {result['pools']['unsolved_limitations']} 个限制")
            >>> print(f"Step 1: 提取了 {result['pools']['candidate_methods']} 个方法")
            >>> print(f"Step 2: 生成了 {result['total_ideas']} 个候选创意")
            >>> print(f"Step 2: 过滤后剩余 {result['successful_ideas']} 个可行创意")
        """
        # ===== Step 1: 获取 Limitation 和 Method（碎片池化）=====
        # 从知识图谱中提取高质量的研究碎片
        logger.info("📋 Step 1: 从知识图谱提取 Limitation 和 Method")

        # 检查图谱是否为空
        if len(graph.nodes()) == 0:
            logger.warning("Knowledge graph is empty, cannot generate ideas")
            # 返回空结果结构
            return {
                "topic": topic,
                "total_ideas": 0,
                "successful_ideas": 0,
                "ideas": [],
                "pools": {
                    "unsolved_limitations": 0,
                    "candidate_methods": 0
                }
            }

        # 使用 KnowledgeGraphExtractor 从图谱中提取 limitations 和 methods
        unsolved_limitations, candidate_methods = self.kg_extractor.extract_from_graph(
            graph, min_text_length
        )

        # 验证数据充分性
        # 需要至少 1 个 limitation 和 1 个 method 才能进行创意生成
        if len(unsolved_limitations) == 0 or len(candidate_methods) == 0:
            logger.warning(
                f"Step 1 数据不足: "
                f"{len(unsolved_limitations)} limitations, "
                f"{len(candidate_methods)} methods (need at least 1 of each)"
            )
            # 返回空结果，但包含提取的数量信息
            return {
                "topic": topic,
                "total_ideas": 0,
                "successful_ideas": 0,
                "ideas": [],
                "pools": {
                    "unsolved_limitations": len(unsolved_limitations),
                    "candidate_methods": len(candidate_methods)
                }
            }

        logger.info(f"✅ Step 1 完成: {len(unsolved_limitations)} limitations, {len(candidate_methods)} methods")

        # ===== Step 2: 创意生成（含自动过滤）=====
        # 调用底层的 generate_from_pools() 方法
        # 该方法会进行 limitation × method 的笛卡尔积组合
        # 并使用 Chain of Thought 推理筛选可行的创意
        # 同时整合演化路径信息，学习演化逻辑
        logger.info("📋 Step 2: 创意生成（Limitation × Method + CoT 推理 + 演化路径学习 + 自动过滤）")
        
        if evolutionary_paths:
            logger.info(f"  整合 {len(evolutionary_paths)} 条演化路径作为上下文")

        return self.generate_from_pools(
            unsolved_limitations=unsolved_limitations,
            candidate_methods=candidate_methods,
            topic=topic,
            evolutionary_paths=evolutionary_paths,
            verbose=verbose
        )

    def generate_from_pools(
        self,
        unsolved_limitations: List[str],
        candidate_methods: List[str],
        topic: str = "",
        evolutionary_paths: Optional[List[Dict]] = None,
        verbose: bool = True
    ) -> Dict:
        """
        从 Limitation 和 Method 池生成研究创意（Step 2 的实现）

        该方法执行 Step 2 的完整流程：
        1. Limitation × Method 笛卡尔积组合
        2. Chain of Thought 推理（兼容性分析 → 差距识别 → 创意草拟）
        3. 演化路径学习：利用演化脉络的逻辑，学习如何组合 Limitation 和 Method
        4. 自动过滤（只保留 status="SUCCESS" 的创意）

        Args:
            unsolved_limitations: Limitation 列表（来自 Step 1 碎片池化）
            candidate_methods: Method 列表（来自 Step 1 碎片池化）
            topic: 研究主题（可选）
            evolutionary_paths: 可选的演化路径列表（来自 deep_survey_analyzer），
                               用于学习演化逻辑和提供上下文
            verbose: 是否输出详细进度日志

        Returns:
            Dict: 包含生成结果和统计信息的字典
                {
                    "topic": str,
                    "total_ideas": int,              # 生成的可行创意总数
                    "successful_ideas": int,         # 同 total_ideas（已过滤）
                    "ideas": List[Dict],             # 只含 SUCCESS 状态的创意
                    "pools": {
                        "unsolved_limitations": int,
                        "candidate_methods": int
                    }
                }
        """
        if verbose:
            logger.info(f"Generating research ideas for topic: {topic}")
            logger.info(f"Limitations pool: {len(unsolved_limitations)}")
            logger.info(f"Methods pool: {len(candidate_methods)}")
            if evolutionary_paths:
                logger.info(f"Evolutionary paths: {len(evolutionary_paths)} paths provided")

        # 调用 HypothesisGenerator 进行批量生成
        # batch_generate 内部会：
        # 1. 进行 limitation × method 笛卡尔积遍历
        # 2. 对每个组合调用 Chain of Thought 推理
        # 3. 整合演化路径信息，学习演化逻辑
        # 4. 自动过滤，只返回 status="SUCCESS" 的创意
        ideas = self.hypothesis_generator.batch_generate(
            unsolved_limitations=unsolved_limitations,
            candidate_methods=candidate_methods,
            evolutionary_paths=evolutionary_paths,
            max_ideas=self.max_ideas,
            verbose=verbose
        )

        return {
            "topic": topic,
            "total_ideas": len(ideas),
            "successful_ideas": len([i for i in ideas if i["status"] == "SUCCESS"]),
            "ideas": ideas,
            "pools": {
                "unsolved_limitations": len(unsolved_limitations),
                "candidate_methods": len(candidate_methods)
            }
        }


# 便捷函数，用于直接使用
def generate_innovation_idea(
    limitation: str,
    method: str,
    model_name: str = "gpt-4o",
    temperature: float = 0.3,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    evolutionary_paths: Optional[List[Dict]] = None,
    verbose: bool = False
) -> Dict:
    """
    便捷函数，用于生成单个创新创意

    Args:
        limitation: 研究限制/瓶颈描述
        method: 候选方法描述
        model_name: 要使用的 OpenAI 模型
        temperature: 采样温度
        api_key: OpenAI API 密钥
        base_url: 可选的 API 基础 URL
        evolutionary_paths: 可选的演化路径列表，用于提供上下文和学习演化逻辑
        verbose: 打印详细输出

    Returns:
        包含 status、title、abstract、modification、reasoning 和 rationale 的字典
        rationale 字段包含完整的推演路径，包括Step 1-3的详细分析

    Example:
        >>> idea = generate_innovation_idea(
        ...     limitation="标准注意力机制具有 O(n²) 复杂度",
        ...     method="FlashAttention 使用分块来减少内存 IO 操作"
        ... )
        >>> print(idea["status"])  # "SUCCESS" 或 "INCOMPATIBLE"
        >>> print(idea["title"])
        >>> print(idea["abstract"])
    """
    generator = HypothesisGenerator(
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url
    )

    return generator.generate_innovation_idea(
        limitation, 
        method, 
        evolutionary_paths=evolutionary_paths,
        verbose=verbose
    )


if __name__ == "__main__":
    # 示例用法
    logging.basicConfig(level=logging.INFO)

    # 示例 1: 单个创意生成
    print("\n" + "="*80)
    print("示例 1: 单个创意生成")
    print("="*80)

    limitation = "Transformer中的标准注意力机制在序列长度上具有二次计算复杂度 O(n²)，限制了其在长序列上的应用。"

    method = "FlashAttention 使用分块和重计算策略来减少内存IO操作，在保持精确注意力计算的同时实现了显著的加速。"

    idea = generate_innovation_idea(limitation, method, verbose=True)

    print("\n结果:")
    print(json.dumps(idea, indent=2, ensure_ascii=False))

    # 示例 2: 批量生成
    print("\n\n" + "="*80)
    print("示例 2: 批量创意生成")
    print("="*80)

    limitations = [
        "当前的视觉Transformer需要大量训练数据，在小数据集上表现不佳。",
        "图神经网络在堆叠多层时会出现过度平滑问题。",
        "强化学习算法在稀疏奖励环境中具有高样本复杂度。"
    ]

    methods = [
        "使用对比目标的自监督学习可以在没有标签的情况下学习有用的表示。",
        "注意力机制可以选择性地关注输入的相关部分。",
        "元学习算法可以用少量示例快速适应新任务。"
    ]

    generator = HypothesisGenerator(model_name="gpt-4o", temperature=0.3)
    ideas = generator.batch_generate(
        unsolved_limitations=limitations,
        candidate_methods=methods,
        max_ideas=5,
        verbose=True
    )

    print(f"\n\n生成了 {len(ideas)} 个成功的创意:")
    for i, idea in enumerate(ideas, 1):
        print(f"\n{'='*80}")
        print(f"创意 {i}")
        print(f"{'='*80}")
        print(f"标题: {idea['title']}")
        print(f"摘要: {idea['abstract'][:200]}...")
        print(f"关键修改: {idea['modification']}")
