"""
DeepPaper Multi-Agent System
基于迭代式多Agent架构的深度论文解析系统

核心Agent:
- Navigator: 导航员 - 分析论文结构,定位关键信息
- Extractor: 提取员 - 深度阅读章节,提取关键内容
- Critic: 审查员 - 验证提取质量,指导重试
- Synthesizer: 总结员 - 整合信息,输出JSON

设计理念:
- 放弃单次RAG,采用迭代式多轮精炼
- Critic驱动的Reflection Loop确保高质量
- 支持复杂推理和上下文理解
"""

from .navigator_agent import NavigatorAgent
from .extractor_agent import ExtractorAgent
from .critic_agent import CriticAgent
from .synthesizer_agent import SynthesizerAgent
from .orchestrator import DeepPaperOrchestrator
from .data_structures import (
    PaperDocument,
    SectionScope,
    ExtractionResult,
    CriticFeedback,
    FinalReport
)

__version__ = "1.0.0"
__all__ = [
    "NavigatorAgent",
    "ExtractorAgent",
    "CriticAgent",
    "SynthesizerAgent",
    "DeepPaperOrchestrator",
    "PaperDocument",
    "SectionScope",
    "ExtractionResult",
    "CriticFeedback",
    "FinalReport"
]
