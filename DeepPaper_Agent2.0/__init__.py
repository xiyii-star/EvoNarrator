"""
DeepPaper 2.0 - Deep Paper Information Extraction System

This package provides a Multi-Agent system for extracting deep information from research papers.

Main Components:
- LogicAnalystAgent: Extract Problem-Solution Pairs
- SectionLocatorAgent: Locate relevant sections
- LimitationExtractor: Extract limitations (section + citation analysis)
- FutureWorkExtractor: Extract future work directions
- DeepPaper2Orchestrator: Coordinate all components

Usage:
    from DeepPaper_Agent2_0 import DeepPaper2Orchestrator
    from DeepPaper_Agent2_0.data_structures import PaperDocument

    orchestrator = DeepPaper2Orchestrator(llm_client)
    report = orchestrator.analyze_paper(paper_document)
"""

from .data_structures import (
    FieldType,
    PaperSection,
    PaperDocument,
    SectionScope,
    ExtractionResult,
    CriticFeedback,
    FinalReport
)

from .LogicAnalystAgent import LogicAnalystAgent, ProblemSolutionPair
from .SectionLocatorAgent import SectionLocatorAgent
from .LimitationExtractor import LimitationExtractor
from .FutureWorkExtractor import FutureWorkExtractor
from .CitationDetectiveAgent import CitationDetectiveAgent, CitationContext, CitationAnalysisResult
from .orchestrator import DeepPaper2Orchestrator

__version__ = "2.0.0"
__author__ = "DeepPaper Team"

__all__ = [
    # Data structures
    "FieldType",
    "PaperSection",
    "PaperDocument",
    "SectionScope",
    "ExtractionResult",
    "CriticFeedback",
    "FinalReport",

    # Agents
    "LogicAnalystAgent",
    "ProblemSolutionPair",
    "SectionLocatorAgent",
    "LimitationExtractor",
    "FutureWorkExtractor",
    "CitationDetectiveAgent",
    "CitationContext",
    "CitationAnalysisResult",

    # Orchestrator
    "DeepPaper2Orchestrator",
]
