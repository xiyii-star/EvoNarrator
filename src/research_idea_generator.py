"""
科学研究假设生成器
使用思维链推理生成可行的研究创意
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
        # Try new langchain structure (v0.1.0+)
        from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
        from langchain_core.output_parsers import PydanticOutputParser
    except ImportError:
        # Fallback to old langchain structure
        from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
        from langchain.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field
except ImportError as e:
    raise ImportError(
        "未安装所需包。运行: pip install langchain langchain-openai langchain-core pydantic"
    ) from e

logger = logging.getLogger(__name__)


class IdeaStatus(str, Enum):
    """生成创意的状态"""
    SUCCESS = "SUCCESS"
    INCOMPATIBLE = "INCOMPATIBLE"


class InnovationIdea(BaseModel):
    """生成的研究创意的结构化输出"""
    status: IdeaStatus = Field(description="方法是否与局限性兼容")
    title: Optional[str] = Field(default=None, description="吸引人的学术标题")
    abstract: Optional[str] = Field(
        default=None,
        description="标准学术摘要（背景 -> 差距 -> 提出的方法 -> 预期结果）"
    )
    modification: Optional[str] = Field(
        default=None,
        description="所需的具体修改（'桥接变量'）"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="显示分析过程的思维链推理"
    )


@dataclass
class IdeaFragment:
    """研究片段（局限性或方法）"""
    content: str
    paper_id: str = ""
    paper_title: str = ""
    year: int = 0
    cited_count: int = 0


class HypothesisGenerator:
    """
    使用思维链推理的假设生成器

    流程：
    1. 分析兼容性：检查数学/理论兼容性
    2. 识别差距：确定需要什么修改
    3. 起草创意：生成结构化的研究提案
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.3,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        use_step_structure: bool = True
    ):
        """
        初始化假设生成器

        参数：
            model_name: OpenAI 模型名称（例如 "gpt-4"、"gpt-3.5-turbo"）
            temperature: 采样温度（越低越专注，越高越有创意）
            api_key: OpenAI API 密钥（可选，默认为 OPENAI_API_KEY 环境变量）
            base_url: API 的可选基础 URL（用于代理或自定义端点）
            use_step_structure: 如果为 True，在提示中使用步骤 1/2/3 结构（默认：True）
        """
        llm_kwargs = {
            "model": model_name,
            "temperature": temperature,
        }

        if api_key:
            llm_kwargs["api_key"] = api_key
        if base_url:
            llm_kwargs["base_url"] = base_url

        self.llm = ChatOpenAI(**llm_kwargs)
        self.output_parser = PydanticOutputParser(pydantic_object=InnovationIdea)
        self.use_step_structure = use_step_structure

        # 构建提示模板
        self.prompt_template = self._build_prompt_template()

        logger.info(f"HypothesisGenerator initialized with model: {model_name}, use_step_structure: {use_step_structure}")

    def _build_prompt_template(self) -> ChatPromptTemplate:
        """构建思维链提示模板"""

        if self.use_step_structure:
            # 完整的三步结构化提示（基线）
            system_message = SystemMessagePromptTemplate.from_template(
                """你是一位**资深首席研究员**，在分析研究问题和生成创新解决方案方面拥有深厚的专业知识。

你的任务是评估候选方法是否能解决给定的研究局限性，遵循严格的思维链推理过程。

**你的推理必须遵循以下三个步骤：**

**步骤 1：分析兼容性**
- 检查方法的数学、算法和理论属性
- 检查这些属性是否与局限性的约束和要求一致
- 考虑：计算复杂度、适用领域、基本假设
- 如果根本不兼容，输出 status="INCOMPATIBLE" 并停止

**步骤 2：识别差距**
- 确定需要哪些具体修改来弥合差距
- 识别"桥接变量" - 使连接起作用的关键创新
- 问：方法需要改变什么才能解决这个新的问题背景？

**步骤 3：起草创意**
- 创建一个吸引人的学术标题
- 编写遵循以下结构的摘要：背景 → 差距 → 提出的方法 → 预期结果
- 用一句话清楚地陈述核心创新

{format_instructions}

要严格和诚实。如果某些东西不起作用，就说 INCOMPATIBLE。只对真正可行的创意输出 SUCCESS。"""
            )

            human_message = HumanMessagePromptTemplate.from_template(
                """**局限性（当前研究瓶颈）：**
{limitation}

**候选方法（潜在解决方案）：**
{method}

现在，遵循三步思维链过程：

1. **兼容性分析**：该方法的数学/算法属性是否适合局限性的约束？

2. **差距识别**：需要什么具体的修改或适应？

3. **创意起草**：如果可行，创建标题、摘要并描述核心创新。

以指定的 JSON 格式提供你的完整推理和最终输出。"""
            )
        else:
            # 简化提示（用于消融研究 - 没有步骤 1/2/3 结构）
            system_message = SystemMessagePromptTemplate.from_template(
                """你是一位**资深首席研究员**，在分析研究问题和生成创新解决方案方面拥有深厚的专业知识。

你的任务是评估候选方法是否能解决给定的研究局限性。

**你的分析应考虑：**
- 数学、算法和理论兼容性
- 弥合差距所需的具体修改
- 使连接起作用的关键创新
- 吸引人的学术标题和结构化摘要

{format_instructions}

要严格和诚实。如果某些东西不起作用，就说 INCOMPATIBLE。只对真正可行的创意输出 SUCCESS。"""
            )

            human_message = HumanMessagePromptTemplate.from_template(
                """**局限性（当前研究瓶颈）：**
{limitation}

**候选方法（潜在解决方案）：**
{method}

分析该方法是否能解决局限性。考虑兼容性、所需修改和核心创新。以指定的 JSON 格式提供你的推理和最终输出。"""
            )

        return ChatPromptTemplate.from_messages([system_message, human_message])

    def generate_innovation_idea(
        self,
        limitation: str,
        method: str,
        verbose: bool = False
    ) -> Dict:
        """
        从局限性和候选方法生成研究创新创意

        参数：
            limitation: 研究瓶颈/局限性的描述
            method: 候选方法的描述
            verbose: 如果为 True，打印详细推理

        返回：
            具有以下结构的字典：
            {
                "status": "SUCCESS" 或 "INCOMPATIBLE",
                "title": "...",
                "abstract": "...",
                "modification": "...",
                "reasoning": "..."
            }
        """
        try:
            # 格式化提示
            formatted_prompt = self.prompt_template.format_messages(
                limitation=limitation,
                method=method,
                format_instructions=self.output_parser.get_format_instructions()
            )

            if verbose:
                logger.info("=" * 80)
                logger.info("正在生成创新创意...")
                logger.info(f"局限性: {limitation[:100]}...")
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
                logger.info("=" * 80)

            # 转换为字典
            result = {
                "status": idea.status,
                "title": idea.title,
                "abstract": idea.abstract,
                "modification": idea.modification,
                "reasoning": idea.reasoning
            }

            return result

        except Exception as e:
            logger.error(f"生成创新创意时出错: {e}")
            return {
                "status": "ERROR",
                "title": None,
                "abstract": None,
                "modification": None,
                "reasoning": f"生成过程中出错: {str(e)}"
            }

    def batch_generate(
        self,
        unsolved_limitations: List[str],
        candidate_methods: List[str],
        max_ideas: int = 10,
        verbose: bool = False
    ) -> List[Dict]:
        """
        通过将局限性与方法配对来生成多个创意

        参数：
            unsolved_limitations: 局限性描述列表
            candidate_methods: 方法描述列表
            max_ideas: 要生成的最大创意数量
            verbose: 如果为 True，打印进度

        返回：
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

                idea = self.generate_innovation_idea(limitation, method, verbose=False)

                # 只保留成功的创意
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

    从包含论文信息的知识图谱中提取研究局限性和候选方法。
    提取的内容用于研究创意生成，通过将未解决的局限性与候选方法结合来产生新的研究方向。

    工作原理：
        1. 遍历知识图谱中的所有节点
        2. 从节点属性中提取局限性（研究瓶颈/待解决的问题）
        3. 从节点属性中提取方法（潜在解决方案/贡献）
        4. 过滤和去重提取的内容

    用例：
        - 在生成研究创意之前从构建的文献知识图谱中提取原始材料
        - 为 HypothesisGenerator 准备输入数据
    """

    @staticmethod
    def extract_from_graph(
        graph: nx.Graph,
        min_text_length: int = 50
    ) -> tuple[List[str], List[str]]:
        """
        从 NetworkX 知识图谱中提取局限性和方法（基于引用关系类型的片段池化）

        片段池化策略：
        通过分析引用关系类型（来自插座匹配的结果）智能过滤高质量的研究片段。

        四个片段池：
        - 池 A（未解决的局限性）：未被任何论文克服的局限性
          → 这些是未解决的研究瓶颈，最值得解决
        - 池 B（成功的方法）：被多次扩展的方法
          → 这些方法已被多次扩展，证明是成熟可靠的基础技术
        - 池 C（跨领域方法）：来自 Adapts_to 来源的方法
          → 这些方法已证明具有跨领域迁移能力，适合新场景
        - 池 D（未实现的未来工作）：未被任何论文实现的未来工作
          → 这些是前人设想但尚未实现的研究方向

        局限性来源：池 A + 池 D
        方法来源：池 B + 池 C

        参数：
            graph: NetworkX 图对象，节点包含论文信息，边包含引用关系类型
                   预期的节点属性：
                   - rag_limitation (str): RAG 提取的局限性
                   - rag_future_work (str): RAG 提取的未来工作
                   - rag_method (str): RAG 提取的贡献/方法
                   预期的边属性：
                   - edge_type (str): 引用关系类型（Overcomes、Realizes、Extends 等）
            min_text_length: 有效文本的最小长度，默认 50 个字符

        返回：
            tuple[List[str], List[str]]:
                - unsolved_limitations: 高质量的未解决局限性列表（池 A + 池 D）
                - candidate_methods: 高质量的候选方法列表（池 B + 池 C）

        示例：
            >>> G = nx.Graph()
            >>> G.add_node('W1', rag_limitation='高复杂度', rag_method='方法 A')
            >>> G.add_node('W2', rag_method='方法 B')
            >>> G.add_edge('W2', 'W1', edge_type='Extends')
            >>> limitations, methods = KnowledgeGraphExtractor.extract_from_graph(G)
        """
        logger.info("🔌 开始片段池化提取（基于插座匹配的片段池化）")

        # ===== 收集边类型统计 =====
        # 跟踪哪些边类型指向每个节点以及每个节点指向哪些节点
        node_incoming_edges = {}  # 指向节点的边 {node_id: [(source, edge_type), ...]}
        node_outgoing_edges = {}  # 来自节点的边 {node_id: [(target, edge_type), ...]}

        for source, target, edge_data in graph.edges(data=True):
            edge_type = edge_data.get('edge_type', 'Unknown')

            # 记录 target 被 source 通过 edge_type 引用
            if target not in node_incoming_edges:
                node_incoming_edges[target] = []
            node_incoming_edges[target].append((source, edge_type))

            # 记录 source 通过 edge_type 引用 target
            if source not in node_outgoing_edges:
                node_outgoing_edges[source] = []
            node_outgoing_edges[source].append((target, edge_type))

        # ===== 池 A：未解决的局限性（未被 Overcomes 克服的局限性）=====
        pool_a_limitations = []

        for node_id, node_data in graph.nodes(data=True):
            # 提取局限性
            limitation_text = node_data.get('rag_limitation', '')
            if not isinstance(limitation_text, str) or len(limitation_text.strip()) <= min_text_length:
                continue

            # 检查是否被 Overcomes 边克服
            incoming_edges = node_incoming_edges.get(node_id, [])
            is_overcome = any(edge_type == 'Overcomes' for _, edge_type in incoming_edges)

            if not is_overcome:
                # 未解决的局限性
                pool_a_limitations.append(limitation_text.strip())

        logger.info(f"📦 池 A（未解决的局限性）：{len(pool_a_limitations)} 项")

        # ===== 池 D：未实现的未来工作（未被 Realizes 实现的未来工作）=====
        pool_d_limitations = []

        for node_id, node_data in graph.nodes(data=True):
            # 提取未来工作
            future_work_text = node_data.get('rag_future_work', '')
            if not isinstance(future_work_text, str) or len(future_work_text.strip()) <= min_text_length:
                continue

            # 检查是否被 Realizes 边实现
            incoming_edges = node_incoming_edges.get(node_id, [])
            is_realized = any(edge_type == 'Realizes' for _, edge_type in incoming_edges)

            if not is_realized:
                # 未实现的未来工作
                pool_d_limitations.append(future_work_text.strip())

        logger.info(f"📦 池 D（未实现的未来工作）：{len(pool_d_limitations)} 项")

        # ===== 池 B：成功的方法（被多次扩展的方法）=====
        pool_b_methods = []
        extends_threshold = 2  # 必须至少被扩展 2 次才被认为是成熟的方法

        for node_id, node_data in graph.nodes(data=True):
            # 提取贡献（方法）
            contribution_text = node_data.get('rag_method', '')
            if not isinstance(contribution_text, str) or len(contribution_text.strip()) <= min_text_length:
                continue

            # 计算被 Extends 边扩展的次数
            incoming_edges = node_incoming_edges.get(node_id, [])
            extends_count = sum(1 for _, edge_type in incoming_edges if edge_type == 'Extends')

            if extends_count >= extends_threshold:
                # 被多次扩展的成熟方法
                pool_b_methods.append(contribution_text.strip())

        logger.info(f"📦 池 B（成功的方法，Extends≥{extends_threshold}）：{len(pool_b_methods)} 项")

        # ===== 池 C：跨领域方法（来自 Adapts_to 来源的方法）=====
        pool_c_methods = []

        # 查找所有 Adapts_to 边的源节点
        adapts_to_sources = set()
        for source, target, edge_data in graph.edges(data=True):
            if edge_data.get('edge_type') == 'Adapts_to':
                adapts_to_sources.add(target)  # target 是被适应的源论文

        # 从这些源节点提取方法
        for node_id in adapts_to_sources:
            node_data = graph.nodes[node_id]
            contribution_text = node_data.get('rag_method', '')
            if isinstance(contribution_text, str) and len(contribution_text.strip()) > min_text_length:
                pool_c_methods.append(contribution_text.strip())

        logger.info(f"📦 池 C（来自 Adapts_to 的跨领域方法）：{len(pool_c_methods)} 项")

        # ===== 合并池化结果 =====
        # 局限性 = 池 A + 池 D
        unsolved_limitations = pool_a_limitations + pool_d_limitations
        # 方法 = 池 B + 池 C
        candidate_methods = pool_b_methods + pool_c_methods

        # 去重
        unsolved_limitations = list(set(unsolved_limitations))
        candidate_methods = list(set(candidate_methods))

        # ===== 回退策略：如果片段池化结果不足，用传统方法补充 =====
        if len(unsolved_limitations) < 3 or len(candidate_methods) < 3:
            logger.warning("⚠️ 片段池化结果不足，启用回退策略（用传统提取补充）")
            fallback_limitations, fallback_methods = KnowledgeGraphExtractor._fallback_extract(
                graph, min_text_length
            )

            # 补充现有池
            unsolved_limitations.extend(fallback_limitations)
            candidate_methods.extend(fallback_methods)

            # 再次去重
            unsolved_limitations = list(set(unsolved_limitations))
            candidate_methods = list(set(candidate_methods))

            logger.info(f"  补充后的局限性：{len(unsolved_limitations)} 项")
            logger.info(f"  补充后的方法：{len(candidate_methods)} 项")

        # 输出最终统计
        logger.info(f"\\n✅ 片段池化完成：")
        logger.info(f"  📊 局限性（池 A + 池 D）：{len(unsolved_limitations)} 项")
        logger.info(f"  🔧 方法（池 B + 池 C）：{len(candidate_methods)} 项")

        return unsolved_limitations, candidate_methods

    @staticmethod
    def _fallback_extract(
        graph: nx.Graph,
        min_text_length: int = 50
    ) -> tuple[List[str], List[str]]:
        """
        回退提取策略：当片段池化结果不足时使用传统方法补充

        简单地从所有节点提取局限性和贡献，不考虑引用关系

        参数：
            graph: NetworkX 图对象
            min_text_length: 最小文本长度

        返回：
            tuple[List[str], List[str]]: (局限性, 方法)
        """
        fallback_limitations = []
        fallback_methods = []

        for _, node_data in graph.nodes(data=True):
            # 提取局限性
            limitation_text = node_data.get('rag_limitation', '')
            if isinstance(limitation_text, str) and len(limitation_text.strip()) > min_text_length:
                fallback_limitations.append(limitation_text.strip())

            # 提取贡献
            contribution_text = node_data.get('rag_method', '')
            if isinstance(contribution_text, str) and len(contribution_text.strip()) > min_text_length:
                fallback_methods.append(contribution_text.strip())

        return list(set(fallback_limitations)), list(set(fallback_methods))


class ResearchIdeaGenerator:
    """
    研究创意生成器 - 两步过程：检索 → 生成

    核心流程（两步）：
    ┌────────────────────────────────────────────────────────────┐
    │  步骤 1：检索局限性和方法                                    │
    │  ──────────────────────────────────────────────────────   │
    │  - KnowledgeGraphExtractor.extract_from_graph()           │
    │  - 片段池化：基于引用关系类型                                │
    │    （插座匹配）                                              │
    │  - 池 A：未被 Overcomes 克服的局限性                         │
    │  - 池 B：被扩展 ≥2 次的方法                                 │
    │  - 池 C：来自 Adapts_to 来源的方法                          │
    │  - 池 D：未被 Realizes 实现的未来工作                        │
    └────────────────────────────────────────────────────────────┘
                                ↓
    ┌────────────────────────────────────────────────────────────┐
    │  步骤 2：创意生成（带自动过滤）                              │
    │  ──────────────────────────────────────────────────────   │
    │  - HypothesisGenerator.batch_generate()                   │
    │  - 局限性 × 方法笛卡尔积                                     │
    │  - 思维链推理：                                              │
    │    1. 兼容性分析                                            │
    │    2. 差距识别                                              │
    │    3. 创意起草                                              │
    │  - 自动过滤：只保留 status="SUCCESS" 的创意                  │
    └────────────────────────────────────────────────────────────┘

    架构设计：
        ResearchIdeaGenerator（高级接口 - 协调两步过程）
            ├── KnowledgeGraphExtractor（步骤 1：检索）
            │   └── extract_from_graph() - 片段池化
            └── HypothesisGenerator（步骤 2：生成）
                └── batch_generate() - CoT 推理 + 自动过滤

    用例：
        - 文献综述后的创意生成
        - 从知识图谱中发现研究机会
        - 批量生成和过滤研究假设

    示例：
        >>> # 初始化生成器
        >>> config = {'model_name': 'gpt-4o', 'max_ideas': 10}
        >>> generator = ResearchIdeaGenerator(config=config)
        >>>
        >>> # 从知识图谱生成创意（两步过程自动执行）
        >>> result = generator.generate_from_knowledge_graph(
        ...     graph=citation_graph,
        ...     topic="Transformer 优化"
        ... )
        >>>
        >>> # 查看结果
        >>> print(f"步骤 1: {result['pools']['unsolved_limitations']} 个局限性")
        >>> print(f"步骤 1: {result['pools']['candidate_methods']} 个方法")
        >>> print(f"步骤 2: {result['successful_ideas']} 个成功的创意")
    """

    def __init__(
        self,
        config: Dict = None,
        llm_client = None,
        critic_agent = None
    ):
        """
        初始化研究创意生成器

        注意：
            - 为了向后兼容，保留了 llm_client 和 critic_agent 参数
            - 这些参数在当前实现中被忽略，因为使用了新的 HypothesisGenerator
            - 如果你是新用户，只需传递 config 参数

        参数：
            config: 配置字典，支持以下键：
                - model_name (str): OpenAI 模型名称，默认 'gpt-4o'
                  支持：gpt-4o、gpt-4、gpt-3.5-turbo 等
                - temperature (float): 采样温度，默认 0.3
                  范围：0.0-1.0（越低越确定，越高越有创意）
                - openai_api_key (str): OpenAI API 密钥
                  如果未提供，将使用 OPENAI_API_KEY 环境变量
                - openai_base_url (str): API 基础 URL（可选）
                  用于代理或自定义端点
                - max_ideas (int): 要生成的最大创意数量，默认 10
            llm_client: （已弃用）旧版 LLM 客户端，为向后兼容而保留
            critic_agent: （已弃用）旧版评论代理，为向后兼容而保留

        示例：
            >>> # 基本用法
            >>> config = {
            ...     'model_name': 'gpt-4o',
            ...     'temperature': 0.3,
            ...     'max_ideas': 5
            ... }
            >>> generator = ResearchIdeaGenerator(config=config)
            >>>
            >>> # 使用自定义 API 配置
            >>> config = {
            ...     'openai_api_key': 'your-api-key',
            ...     'openai_base_url': 'https://your-proxy.com/v1',
            ...     'max_ideas': 10
            ... }
            >>> generator = ResearchIdeaGenerator(config=config)
        """
        # 加载配置，如果未提供则使用空字典
        self.config = config or {}

        # ===== 提取 OpenAI 相关配置 =====
        # 从 config 中提取配置项，如果不存在则使用默认值
        # 优先级：首先从 llm 节点读取，然后从顶层读取
        llm_config = self.config.get('llm', {})
        model_name = llm_config.get('model', self.config.get('model_name', 'gpt-4o'))  # 默认为 gpt-4o
        temperature = llm_config.get('temperature', self.config.get('temperature', 0.3))  # 默认温度 0.3
        api_key = llm_config.get('api_key') or self.config.get('openai_api_key')  # API 密钥（可选）
        base_url = llm_config.get('base_url') or self.config.get('openai_base_url')  # 基础 URL（可选）

        # 如果仍然没有 API 密钥，尝试从环境变量读取
        if not api_key:
            import os
            api_key = os.getenv('OPENAI_API_KEY')

        # 获取 use_step_structure 配置（用于消融研究）
        use_step_structure = self.config.get('use_step_structure', True)

        # ===== 初始化核心组件 =====
        # 创建 HypothesisGenerator 实例，实际创意生成的核心组件
        self.hypothesis_generator = HypothesisGenerator(
            model_name=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            use_step_structure=use_step_structure
        )

        # 设置最大创意数量限制，避免生成过多创意
        # 优先级：从 research_idea.max_ideas 读取，然后是顶层 max_ideas，最后默认为 10
        research_idea_config = self.config.get('research_idea', {})
        self.max_ideas = research_idea_config.get('max_ideas', self.config.get('max_ideas', 10))

        # 创建知识图谱提取器实例，用于从图中提取数据
        self.kg_extractor = KnowledgeGraphExtractor()

        logger.info(f"ResearchIdeaGenerator initialized with HypothesisGenerator (max_ideas={self.max_ideas})")

    def generate_from_knowledge_graph(
        self,
        graph: nx.Graph,
        topic: str = "",
        min_text_length: int = 50,
        verbose: bool = True
    ) -> Dict:
        """
        直接从知识图谱生成研究创意（两步过程）

        整体流程：
        ┌────────────────────────────────────────────────────────────┐
        │  步骤 1：检索局限性和方法                                    │
        │  ──────────────────────────────────────────────────────   │
        │  输入：知识图谱（带引用关系类型）                             │
        │  处理：KnowledgeGraphExtractor.extract_from_graph()       │
        │  - 池 A：未被 Overcomes 克服的局限性                         │
        │  - 池 B：被扩展 ≥2 次的方法                                 │
        │  - 池 C：来自 Adapts_to 来源的方法                          │
        │  - 池 D：未被 Realizes 实现的未来工作                        │
        │  输出：(unsolved_limitations, candidate_methods)           │
        └────────────────────────────────────────────────────────────┘
                                    ↓
        ┌────────────────────────────────────────────────────────────┐
        │  步骤 2：创意生成（带自动过滤）                              │
        │  ──────────────────────────────────────────────────────   │
        │  输入：局限性 × 方法笛卡尔积                                 │
        │  处理：HypothesisGenerator.batch_generate()               │
        │  - 兼容性分析                                              │
        │  - 差距识别                                                │
        │  - 创意起草                                                │
        │  - 自动过滤：只保留 status="SUCCESS" 的创意                  │
        │  输出：高质量可行创意列表                                    │
        └────────────────────────────────────────────────────────────┘

        参数：
            graph: NetworkX 图对象，节点应包含论文信息
                   所需的节点属性（用于片段池化）：
                   - rag_limitation (str): RAG 提取的局限性
                   - rag_future_work (str): RAG 提取的未来工作
                   - rag_method (str): RAG 提取的贡献/方法
                   所需的边属性（用于片段池化）：
                   - edge_type (str): 引用关系类型（Overcomes、Realizes、Extends、Adapts_to）
            topic: 研究主题，用于日志和输出（可选）
            min_text_length: 最小文本长度阈值，默认 50
                            用于过滤过短的文本片段
            verbose: 是否输出详细日志，默认 True

        返回：
            Dict: 包含生成结果和统计信息的字典
                {
                    "topic": str,                    # 研究主题
                    "total_ideas": int,              # 步骤 2 中生成的总创意数
                    "successful_ideas": int,         # 步骤 2 过滤后的可行创意数
                    "ideas": List[Dict],             # 可行创意列表（仅 SUCCESS）
                    "pools": {
                        "unsolved_limitations": int, # 步骤 1 中提取的局限性数量
                        "candidate_methods": int     # 步骤 1 中提取的方法数量
                    }
                }

        错误处理：
            - 空图：返回空结果字典
            - 步骤 1 数据不足（局限性或方法为空）：返回带警告的空结果字典

        示例：
            >>> # 初始化生成器
            >>> generator = ResearchIdeaGenerator(config={'max_ideas': 10})
            >>>
            >>> # 从知识图谱生成创意（两步过程自动执行）
            >>> result = generator.generate_from_knowledge_graph(
            ...     graph=citation_graph,
            ...     topic="Transformer 优化"
            ... )
            >>>
            >>> # 查看结果
            >>> print(f"步骤 1：提取了 {result['pools']['unsolved_limitations']} 个局限性")
            >>> print(f"步骤 1：提取了 {result['pools']['candidate_methods']} 个方法")
            >>> print(f"步骤 2：生成了 {result['total_ideas']} 个候选创意")
            >>> print(f"步骤 2：过滤后有 {result['successful_ideas']} 个可行创意")
        """
        # ===== 步骤 1：检索局限性和方法（片段池化）=====
        # 从知识图谱中提取高质量的研究片段
        logger.info("📋 步骤 1：从知识图谱中提取局限性和方法")

        # 检查图是否为空
        if len(graph.nodes()) == 0:
            logger.warning("知识图谱为空，无法生成创意")
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

        # 使用 KnowledgeGraphExtractor 从图中提取局限性和方法
        unsolved_limitations, candidate_methods = self.kg_extractor.extract_from_graph(
            graph, min_text_length
        )

        # 验证数据充足性
        # 需要至少 1 个局限性和 1 个方法才能继续生成创意
        if len(unsolved_limitations) == 0 or len(candidate_methods) == 0:
            logger.warning(
                f"步骤 1 数据不足: "
                f"{len(unsolved_limitations)} 个局限性, "
                f"{len(candidate_methods)} 个方法（每个至少需要 1 个）"
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

        logger.info(f"✅ 步骤 1 完成：{len(unsolved_limitations)} 个局限性，{len(candidate_methods)} 个方法")

        # ===== 步骤 2：创意生成（带自动过滤）=====
        # 调用底层的 generate_from_pools() 方法
        # 该方法将执行局限性 × 方法笛卡尔积组合
        # 并使用思维链推理过滤可行的创意
        logger.info("📋 步骤 2：创意生成（局限性 × 方法 + CoT 推理 + 自动过滤）")

        return self.generate_from_pools(
            unsolved_limitations=unsolved_limitations,
            candidate_methods=candidate_methods,
            topic=topic,
            verbose=verbose
        )

    def generate_from_pools(
        self,
        unsolved_limitations: List[str],
        candidate_methods: List[str],
        topic: str = "",
        verbose: bool = True
    ) -> Dict:
        """
        从局限性和方法池生成研究创意（步骤 2 实现）

        该方法执行完整的步骤 2 过程：
        1. 局限性 × 方法笛卡尔积组合
        2. 思维链推理（兼容性分析 → 差距识别 → 创意起草）
        3. 自动过滤（只保留 status="SUCCESS" 的创意）

        参数：
            unsolved_limitations: 局限性列表（来自步骤 1 片段池化）
            candidate_methods: 方法列表（来自步骤 1 片段池化）
            topic: 研究主题（可选）
            verbose: 是否输出详细进度日志

        返回：
            Dict: 包含生成结果和统计信息的字典
                {
                    "topic": str,
                    "total_ideas": int,              # 生成的总可行创意数
                    "successful_ideas": int,         # 与 total_ideas 相同（已过滤）
                    "ideas": List[Dict],             # 仅包含 SUCCESS 状态的创意
                    "pools": {
                        "unsolved_limitations": int,
                        "candidate_methods": int
                    }
                }
        """
        if verbose:
            logger.info(f"为主题生成研究创意: {topic}")
            logger.info(f"局限性池: {len(unsolved_limitations)}")
            logger.info(f"方法池: {len(candidate_methods)}")

        # 调用 HypothesisGenerator 进行批量生成
        # batch_generate 内部将：
        # 1. 执行局限性 × 方法笛卡尔积遍历
        # 2. 对每个组合调用思维链推理
        # 3. 自动过滤，只返回 status="SUCCESS" 的创意
        ideas = self.hypothesis_generator.batch_generate(
            unsolved_limitations=unsolved_limitations,
            candidate_methods=candidate_methods,
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
    verbose: bool = False
) -> Dict:
    """
    便捷函数，用于生成单个创新创意

    参数：
        limitation: 研究局限性/瓶颈描述
        method: 候选方法描述
        model_name: 要使用的 OpenAI 模型
        temperature: 采样温度
        api_key: OpenAI API 密钥
        base_url: 可选的 API 基础 URL
        verbose: 打印详细输出

    返回：
        包含 status、title、abstract、modification 和 reasoning 的字典

    示例：
        >>> idea = generate_innovation_idea(
        ...     limitation="标准注意力机制具有 O(n²) 复杂度",
        ...     method="FlashAttention 使用分块来减少内存 IO 操作"
        ... )
        >>> print(idea["status"])  # "SUCCESS" or "INCOMPATIBLE"
        >>> print(idea["title"])
        >>> print(idea["abstract"])
    """
    generator = HypothesisGenerator(
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url
    )

    return generator.generate_innovation_idea(limitation, method, verbose=verbose)


if __name__ == "__main__":
    # 示例用法
    logging.basicConfig(level=logging.INFO)

    # 示例 1：单个创意生成
    print("\n" + "="*80)
    print("示例 1：单个创意生成")
    print("="*80)

    limitation = "Standard attention mechanisms in transformers have quadratic computational complexity O(n²) with respect to sequence length, limiting their application to long sequences."

    method = "FlashAttention uses tiling and recomputation strategies to reduce memory IO operations, achieving significant speedups while maintaining exact attention computation."

    idea = generate_innovation_idea(limitation, method, verbose=True)

    print("\n结果:")
    print(json.dumps(idea, indent=2, ensure_ascii=False))

    # 示例 2：批量生成
    print("\n\n" + "="*80)
    print("示例 2：批量创意生成")
    print("="*80)

    limitations = [
        "当前的视觉 Transformer 需要大量训练数据，在小数据集上表现不佳。",
        "图神经网络在堆叠多层时会出现过度平滑问题。",
        "强化学习算法在稀疏奖励环境中具有高样本复杂度。"
    ]

    methods = [
        "自监督学习与对比目标使得无需标签即可学习有用的表示。",
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
