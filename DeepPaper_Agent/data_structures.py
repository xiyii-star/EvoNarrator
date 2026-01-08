"""
数据结构定义
Multi-Agent系统中使用的核心数据结构
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from enum import Enum


class FieldType(Enum):
    """提取字段类型"""
    PROBLEM = "problem"
    METHOD = "method"
    LIMITATION = "limitation"
    FUTURE_WORK = "future_work"


class SectionType(Enum):
    """章节类型"""
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHOD = "method"
    EXPERIMENT = "experiment"
    DISCUSSION = "discussion"
    LIMITATION = "limitation"
    CONCLUSION = "conclusion"
    FUTURE_WORK = "future_work"
    OTHER = "other"


@dataclass
class PaperSection:
    """论文章节"""
    title: str
    content: str
    page_num: int
    section_type: str

    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'content': self.content,
            'page_num': self.page_num,
            'section_type': self.section_type
        }


@dataclass
class PaperDocument:
    """
    论文文档结构
    包含完整的论文信息和章节
    """
    paper_id: str
    title: str
    abstract: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    sections: List[PaperSection] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def get_sections_by_type(self, section_types: List[str]) -> List[PaperSection]:
        """按类型筛选章节"""
        return [s for s in self.sections if s.section_type in section_types]

    def get_section_titles(self) -> List[str]:
        """获取所有章节标题"""
        return [s.title for s in self.sections]

    def get_full_text(self) -> str:
        """获取全文本"""
        parts = [f"Title: {self.title}"]
        if self.abstract:
            parts.append(f"Abstract: {self.abstract}")
        for section in self.sections:
            parts.append(f"\n## {section.title}\n{section.content}")
        return "\n\n".join(parts)


@dataclass
class SectionScope:
    """
    Navigator返回的章节范围
    指示Extractor应该去哪里找信息
    """
    field: FieldType
    target_sections: List[int]  # 章节索引列表
    section_titles: List[str]  # 对应的章节标题
    reasoning: str  # Navigator的推理过程
    confidence: float = 0.0  # 置信度(0-1)

    def to_dict(self) -> Dict:
        return {
            'field': self.field.value,
            'target_sections': self.target_sections,
            'section_titles': self.section_titles,
            'reasoning': self.reasoning,
            'confidence': self.confidence
        }


@dataclass
class ExtractionResult:
    """
    Extractor返回的提取结果
    包含原文引用和提取的内容
    """
    field: FieldType
    content: str  # 提取的内容(已总结)
    evidence: List[Dict] = field(default_factory=list)  # 原文证据
    # evidence格式: [{'section': str, 'text': str, 'page': int}]
    extraction_method: str = "llm"  # 提取方法
    confidence: float = 0.0
    iterations: int = 1  # 迭代次数(用于跟踪重试)

    def to_dict(self) -> Dict:
        return {
            'field': self.field.value,
            'content': self.content,
            'evidence': self.evidence,
            'extraction_method': self.extraction_method,
            'confidence': self.confidence
        }


@dataclass
class CriticFeedback:
    """
    Critic返回的反馈
    用于指导Extractor重试
    """
    field: FieldType
    approved: bool  # 是否通过
    feedback_type: Literal[
        "approved",  # 通过
        "empty_retry",  # 提取为空,需要重试
        "wrong_target",  # 提取错误(如提取了baseline的limitation)
        "too_generic",  # 内容太泛化,需要更具体
        "missing_context"  # 缺少上下文
    ]
    feedback_message: str  # 具体的反馈和改进建议
    suggested_sections: List[int] = field(default_factory=list)  # 建议的新章节
    retry_prompt: Optional[str] = None  # 重试的prompt

    def to_dict(self) -> Dict:
        return {
            'field': self.field.value,
            'approved': self.approved,
            'feedback_type': self.feedback_type,
            'feedback_message': self.feedback_message,
            'suggested_sections': self.suggested_sections,
            'retry_prompt': self.retry_prompt
        }


@dataclass
class FinalReport:
    """
    最终输出报告
    Synthesizer生成的结构化JSON
    """
    paper_id: str
    title: str

    # 四个核心字段
    problem: str = ""
    problem_evidence: List[Dict] = field(default_factory=list)

    method: str = ""
    method_evidence: List[Dict] = field(default_factory=list)

    limitation: str = ""
    limitation_evidence: List[Dict] = field(default_factory=list)

    future_work: str = ""
    future_work_evidence: List[Dict] = field(default_factory=list)

    # 元数据
    extraction_quality: Dict = field(default_factory=dict)  # 每个字段的质量评分
    iteration_count: Dict = field(default_factory=dict)  # 每个字段的迭代次数
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            'paper_id': self.paper_id,
            'title': self.title,
            'problem': {
                'content': self.problem,
                'evidence': self.problem_evidence
            },
            'method': {
                'content': self.method,
                'evidence': self.method_evidence
            },
            'limitation': {
                'content': self.limitation,
                'evidence': self.limitation_evidence
            },
            'future_work': {
                'content': self.future_work,
                'evidence': self.future_work_evidence
            },
            'metadata': {
                'extraction_quality': self.extraction_quality,
                'iteration_count': self.iteration_count,
                'timestamp': self.timestamp
            }
        }
