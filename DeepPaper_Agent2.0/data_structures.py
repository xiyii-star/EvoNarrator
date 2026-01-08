"""
数据结构定义
用于 DeepPaper 2.0 系统的各种数据类型
"""

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
from enum import Enum


class FieldType(Enum):
    """提取字段类型"""
    PROBLEM = "problem"
    METHOD = "method"
    LIMITATION = "limitation"
    FUTURE_WORK = "future_work"


@dataclass
class PaperSection:
    """论文章节"""
    title: str  # 章节标题
    content: str  # 章节内容
    page_num: int = 0  # 页码
    section_type: str = "other"  # 章节类型 (abstract, introduction, method, discussion, etc.)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class PaperDocument:
    """论文文档"""
    paper_id: str  # 论文ID
    title: str  # 标题
    abstract: str  # 摘要
    authors: List[str]  # 作者列表
    year: Optional[int]  # 年份
    sections: List[PaperSection]  # 章节列表
    metadata: Optional[Dict[str, Any]] = None  # 额外元数据

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "year": self.year,
            "sections": [s.to_dict() for s in self.sections],
            "metadata": self.metadata
        }


@dataclass
class SectionScope:
    """章节范围 (Navigator输出)"""
    field: FieldType  # 目标字段
    target_sections: List[int]  # 目标章节的索引列表
    section_titles: List[str]  # 章节标题列表
    reasoning: str = ""  # 推理过程
    confidence: float = 0.0  # 置信度


@dataclass
class ExtractionResult:
    """提取结果"""
    field: FieldType  # 字段类型
    content: str  # 提取的内容
    evidence: List[Dict[str, Any]] = field(default_factory=list)  # 证据列表
    extraction_method: str = "unknown"  # 提取方法
    confidence: float = 0.0  # 置信度
    iterations: int = 1  # 迭代次数

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "field": self.field.value,
            "content": self.content,
            "evidence": self.evidence,
            "extraction_method": self.extraction_method,
            "confidence": self.confidence,
            "iterations": self.iterations
        }


@dataclass
class CriticFeedback:
    """Critic反馈"""
    field: FieldType  # 字段类型
    approved: bool  # 是否通过
    feedback_type: str  # 反馈类型: "approved", "empty_retry", "wrong_target", "too_generic", "missing_context"
    feedback_message: str  # 反馈信息
    retry_prompt: Optional[str] = None  # 重试提示
    suggested_sections: Optional[List[int]] = None  # 建议的章节


@dataclass
class FinalReport:
    """最终报告"""
    paper_id: str  # 论文ID
    title: str  # 标题
    problem: str  # 研究问题
    method: str  # 方法
    limitation: str  # 局限性
    future_work: str  # 未来工作
    problem_evidence: List[Dict[str, Any]] = field(default_factory=list)
    method_evidence: List[Dict[str, Any]] = field(default_factory=list)
    limitation_evidence: List[Dict[str, Any]] = field(default_factory=list)
    future_work_evidence: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "problem": self.problem,
            "method": self.method,
            "limitation": self.limitation,
            "future_work": self.future_work,
            "problem_evidence": self.problem_evidence,
            "method_evidence": self.method_evidence,
            "limitation_evidence": self.limitation_evidence,
            "future_work_evidence": self.future_work_evidence,
            "metadata": self.metadata
        }
