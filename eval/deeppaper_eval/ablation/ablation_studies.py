#!/usr/bin/env python3
"""
消融实验 (Ablation Studies) - 完整版本

三个消融实验:
1. Ablation 1: Reflection Loop (CriticAgent) 的有效性
   - Variant A: DeepPaper 2.0 (Full) - 使用CriticAgent进行迭代优化
   - Variant B: DeepPaper 2.0 w/o CriticAgent - 单次提取，无迭代

2. Ablation 2: LogicAnalystAgent vs. 简单提取
   - Variant A: 使用LogicAnalystAgent (寻找因果链)
   - Variant B: 使用普通的Summarization Prompt

3. Ablation 3: Citation Detective (双流合并) 的增益
   - Variant A: Full System (Section + Citation)
   - Variant B: Only Section Analysis

关注指标:
- Precision: 提取的准确性
- Recall: 召回率 (特别是implicit limitations)
- Format Compliance: 格式符合度
- Pairing Accuracy: Problem和Method的对应准确性
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
import sys
import logging

# Add parent directory to path
kg_demo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(kg_demo_root))
deeppaper2_path = kg_demo_root / "DeepPaper_Agent2.0"
sys.path.insert(0, str(deeppaper2_path))

from data_structures import PaperDocument, PaperSection, FieldType, ExtractionResult
from src.llm_config import LLMClient, LLMConfig
from src.grobid_parser import GrobidPDFParser

logger = logging.getLogger(__name__)


def load_paper_from_txt(txt_path: str) -> PaperDocument:
    """
    从已解析的TXT文件加载论文

    TXT文件格式:
    ================================================================================
    Section Title
    ================================================================================
    Section content...

    Args:
        txt_path: TXT文件路径

    Returns:
        PaperDocument对象
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 分割章节
    sections = []
    title = "Unknown"
    abstract = ""

    # 使用分隔符分割
    separator = '=' * 80
    parts = content.split(separator)

    current_section_title = None
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue

        # 奇数索引是标题,偶数索引是内容
        if i % 2 == 1:
            current_section_title = part.strip()
        elif i % 2 == 0 and current_section_title:
            section_content = part.strip()

            # 确定章节类型
            section_type = 'other'
            title_lower = current_section_title.lower()

            if 'abstract' in title_lower:
                section_type = 'abstract'
                abstract = section_content
            elif 'introduction' in title_lower or 'intro' in title_lower:
                section_type = 'introduction'
            elif 'conclusion' in title_lower:
                section_type = 'conclusion'
            elif 'reference' in title_lower:
                section_type = 'references'

            # 如果是第一个章节且标题不是abstract,可能是论文标题
            if len(sections) == 0 and section_type == 'other' and len(current_section_title) < 200:
                title = current_section_title

            sections.append(PaperSection(
                title=current_section_title,
                content=section_content,
                page_num=len(sections),
                section_type=section_type
            ))

            current_section_title = None

    # 如果没有章节,使用整个文本作为一个章节
    if not sections:
        sections.append(PaperSection(
            title="Full Text",
            content=content[:15000],
            page_num=0,
            section_type='other'
        ))

    # 提取paper_id
    paper_id = Path(txt_path).stem

    return PaperDocument(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        authors=[],  # TXT文件中没有作者信息
        year=None,   # TXT文件中没有年份信息
        sections=sections
    )


def extract_paper_from_pdf(
    pdf_path: str,
    grobid_url: Optional[str] = None
) -> PaperDocument:
    """从PDF提取论文信息"""
    sections = []
    title = "Unknown"
    abstract = ""

    # 尝试使用GROBID解析
    if grobid_url:
        try:
            parser = GrobidPDFParser(grobid_url)
            sections = parser.extract_sections_from_pdf(pdf_path)

            if sections:
                for section in sections:
                    if section.section_type == 'title':
                        title = section.content
                    elif section.section_type == 'abstract':
                        abstract = section.content
                print(f"         ✅ GROBID解析成功: {len(sections)} 个章节")
        except Exception as e:
            print(f"         ⚠️ GROBID解析失败: {e}")

    # 降级到PyPDF2
    if not sections:
        print(f"         使用PyPDF2解析...")
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()

                sections.append(PaperSection(
                    title="Full Text",
                    content=text[:15000],
                    page_num=0,
                    section_type='other'
                ))
                print(f"         ✅ PyPDF2解析完成")
        except Exception as e:
            print(f"         ❌ PDF解析失败: {e}")
            sections.append(PaperSection(
                title="Error",
                content=f"Failed to parse PDF: {e}",
                page_num=0,
                section_type='other'
            ))

    paper_id = Path(pdf_path).stem.split('_')[0] if '_' in Path(pdf_path).stem else Path(pdf_path).stem

    return PaperDocument(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        authors=[],  # PDF解析中没有作者信息
        year=None,   # PDF解析中没有年份信息
        sections=sections
    )


# ============================================================================
# 消融实验 1: Reflection Loop (CriticAgent) 的有效性
# ============================================================================

def ablation_1_with_critic(
    paper: PaperDocument,
    llm_client: LLMClient,
    paper_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    消融实验1 - Variant A: 使用完整的Critic反馈循环

    使用DeepPaper 2.0完整流程，包含CriticAgent的迭代优化
    """
    print(f"      > [Ablation 1A] Full system with CriticAgent...")

    from orchestrator import DeepPaper2Orchestrator

    orchestrator = DeepPaper2Orchestrator(
        llm_client=llm_client,
        use_citation_analysis=False  # 本实验不关注citation
    )

    # 追踪迭代过程
    iterations_data = []

    try:
        report = orchestrator.analyze_paper(
            paper_document=paper,
            paper_id=paper_id,
            output_dir=None
        )

        # 提取迭代信息
        metadata = report.metadata or {}

        return {
            'problem': report.problem,
            'method': report.method,
            'limitation': report.limitation,
            'future_work': report.future_work,
            'metadata': {
                'use_critic': True,
                'iterations': metadata.get('iterations', {}),
                'confidences': metadata.get('confidences', {}),
                'extraction_methods': metadata.get('extraction_methods', {})
            }
        }
    except Exception as e:
        logger.error(f"Ablation 1A failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'problem': '',
            'method': '',
            'limitation': '',
            'future_work': '',
            'error': str(e),
            'metadata': {'use_critic': True}
        }


def ablation_1_without_critic(
    paper: PaperDocument,
    llm_client: LLMClient,
    paper_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    消融实验1 - Variant B: 移除Critic，单次提取

    使用相同的组件但禁用Critic反馈循环
    """
    print(f"      > [Ablation 1B] Without CriticAgent (single-pass)...")

    from LogicAnalystAgent import LogicAnalystAgent
    from LimitationExtractor import LimitationExtractor
    from FutureWorkExtractor import FutureWorkExtractor

    # 初始化组件，但禁用Critic
    logic_analyst = LogicAnalystAgent(llm_client)  # LogicAnalystAgent不支持use_critic参数
    limitation_extractor = LimitationExtractor(
        llm_client,
        use_citation_analysis=False,
        use_critic=False  # 关键: 禁用Critic
    )
    future_work_extractor = FutureWorkExtractor(llm_client, use_critic=False)

    try:
        # 单次提取，无迭代
        # LogicAnalystAgent使用analyze方法，不是extract
        # 准备论文内容
        paper_content = f"{paper.title}\n\n{paper.abstract}\n\n"
        for section in paper.sections[:10]:  # 使用前10个章节
            paper_content += f"## {section.title}\n{section.content}\n\n"

        metadata = {
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.year
        }

        # 使用analyze方法
        pairs = logic_analyst.analyze(
            paper_content=paper_content,
            paper_metadata=metadata
        )

        # 解析Problem和Method
        if pairs:
            main_pair = pairs[0]
            problem_content = main_pair.problem
            method_content = f"{main_pair.solution}\n\n**Explanation:** {main_pair.explanation}"
            problem_confidence = main_pair.confidence
            method_confidence = main_pair.confidence
        else:
            problem_content = "未找到明确的研究问题描述"
            method_content = "未找到明确的方法描述"
            problem_confidence = 0.0
            method_confidence = 0.0

        limitation_result = limitation_extractor.extract(paper, paper_id)
        future_work_result = future_work_extractor.extract(paper)

        return {
            'problem': problem_content,
            'method': method_content,
            'limitation': limitation_result.content,
            'future_work': future_work_result.content,
            'metadata': {
                'use_critic': False,
                'iterations': {
                    'problem': 1,
                    'method': 1,
                    'limitation': 1,
                    'future_work': 1
                },
                'confidences': {
                    'problem': problem_confidence,
                    'method': method_confidence,
                    'limitation': limitation_result.confidence,
                    'future_work': future_work_result.confidence
                }
            }
        }
    except Exception as e:
        logger.error(f"Ablation 1B failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'problem': '',
            'method': '',
            'limitation': '',
            'future_work': '',
            'error': str(e),
            'metadata': {'use_critic': False}
        }


# ============================================================================
# 消融实验 2: LogicAnalystAgent vs. 简单提取
# ============================================================================

def ablation_2_with_logic_analyst(
    paper: PaperDocument,
    llm_client: LLMClient
) -> Dict[str, Any]:
    """
    消融实验2 - Variant A: 使用LogicAnalystAgent (寻找因果链)

    LogicAnalystAgent能够理解Problem和Method之间的因果关系，
    精准对齐问题和解决方案
    """
    print(f"      > [Ablation 2A] With LogicAnalystAgent (causal reasoning)...")

    from LogicAnalystAgent import LogicAnalystAgent

    logic_analyst = LogicAnalystAgent(llm_client)  # LogicAnalystAgent不支持use_critic参数

    try:
        # 准备论文内容
        paper_content = f"{paper.title}\n\n{paper.abstract}\n\n"
        for section in paper.sections[:10]:  # 使用前10个章节
            paper_content += f"## {section.title}\n{section.content}\n\n"

        metadata = {
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.year
        }

        # 使用analyze方法
        pairs = logic_analyst.analyze(
            paper_content=paper_content,
            paper_metadata=metadata
        )

        # 解析Problem和Method
        if pairs:
            main_pair = pairs[0]
            problem_content = main_pair.problem
            method_content = f"{main_pair.solution}\n\n**Explanation:** {main_pair.explanation}"
            problem_confidence = main_pair.confidence
            method_confidence = main_pair.confidence
        else:
            problem_content = "未找到明确的研究问题描述"
            method_content = "未找到明确的方法描述"
            problem_confidence = 0.0
            method_confidence = 0.0

        return {
            'problem': problem_content,
            'method': method_content,
            'metadata': {
                'extractor_type': 'logic_analyst',
                'causal_reasoning': True,
                'problem_method_paired': True,
                'confidences': {
                    'problem': problem_confidence,
                    'method': method_confidence
                }
            }
        }
    except Exception as e:
        logger.error(f"Ablation 2A failed: {e}")
        return {
            'problem': '',
            'method': '',
            'error': str(e),
            'metadata': {'extractor_type': 'logic_analyst'}
        }


def ablation_2_simple_summarization(
    paper: PaperDocument,
    llm_client: LLMClient
) -> Dict[str, Any]:
    """
    消融实验2 - Variant B: 使用简单的Summarization Prompt

    不使用因果推理，只是简单地从文本中总结Problem和Method
    容易出现Problem A配Method B的错配问题
    """
    print(f"      > [Ablation 2B] Simple summarization (no causal reasoning)...")

    # 构建简单的summarization prompt
    simple_system_prompt = """你是一个学术论文分析助手。请从论文中提取关键信息。"""

    try:
        # 提取Problem (简单版)
        problem_prompt = f"""论文标题: {paper.title}

请总结这篇论文要解决的主要问题。用2-3句话概括。

论文摘要:
{paper.abstract}

论文内容 (前2000字):
{' '.join([s.content for s in paper.sections[:3]])[:2000]}

输出格式: 直接输出问题描述，不要添加"问题是..."等前缀。
"""

        problem_response = llm_client.generate(
            prompt=problem_prompt,
            system_prompt=simple_system_prompt,
            temperature=0.3,
            max_tokens=500
        )

        # 提取Method (简单版)
        method_prompt = f"""论文标题: {paper.title}

请总结这篇论文提出的方法或贡献。用2-3句话概括。

论文摘要:
{paper.abstract}

论文内容 (前2000字):
{' '.join([s.content for s in paper.sections[:3]])[:2000]}

输出格式: 直接输出方法描述，不要添加"方法是..."等前缀。
"""

        method_response = llm_client.generate(
            prompt=method_prompt,
            system_prompt=simple_system_prompt,
            temperature=0.3,
            max_tokens=500
        )

        return {
            'problem': problem_response.strip(),
            'method': method_response.strip(),
            'metadata': {
                'extractor_type': 'simple_summarization',
                'causal_reasoning': False,
                'problem_method_paired': False,
                'confidences': {
                    'problem': 0.5,  # 固定置信度
                    'method': 0.5
                }
            }
        }
    except Exception as e:
        logger.error(f"Ablation 2B failed: {e}")
        return {
            'problem': '',
            'method': '',
            'error': str(e),
            'metadata': {'extractor_type': 'simple_summarization'}
        }


# ============================================================================
# 消融实验 3: Citation Detective (双流合并) 的增益
# ============================================================================

def ablation_3_with_citation(
    paper: PaperDocument,
    llm_client: LLMClient,
    paper_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    消融实验3 - Variant A: Full System (Section + Citation)

    使用双流合并: 从论文章节提取 + 从引用分析提取
    能够发现作者自己没承认但同行指出的隐式局限性
    """
    print(f"      > [Ablation 3A] With Citation Detective (dual-stream)...")

    from LimitationExtractor import LimitationExtractor

    limitation_extractor = LimitationExtractor(
        llm_client,
        use_citation_analysis=True,  # 关键: 启用引用分析
        use_critic=False  # 为了公平对比，禁用Critic
    )

    try:
        result = limitation_extractor.extract(paper, paper_id)

        # 统计来源
        evidence = result.evidence or []
        section_evidence_count = len([e for e in evidence if 'section' in e])

        return {
            'limitation': result.content,
            'metadata': {
                'use_citation': True,
                'extraction_method': result.extraction_method,
                'confidence': result.confidence,
                'evidence_count': len(evidence),
                'section_evidence_count': section_evidence_count,
                'citation_evidence_count': len(evidence) - section_evidence_count
            }
        }
    except Exception as e:
        logger.error(f"Ablation 3A failed: {e}")
        return {
            'limitation': '',
            'error': str(e),
            'metadata': {'use_citation': True}
        }


def ablation_3_section_only(
    paper: PaperDocument,
    llm_client: LLMClient,
    paper_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    消融实验3 - Variant B: Only Section Analysis

    只从论文章节提取局限性，不使用引用分析
    只能找到作者明确承认的局限性
    """
    print(f"      > [Ablation 3B] Section only (no citation analysis)...")

    from LimitationExtractor import LimitationExtractor

    limitation_extractor = LimitationExtractor(
        llm_client,
        use_citation_analysis=False,  # 关键: 禁用引用分析
        use_critic=False
    )

    try:
        result = limitation_extractor.extract(paper, paper_id)

        evidence = result.evidence or []

        return {
            'limitation': result.content,
            'metadata': {
                'use_citation': False,
                'extraction_method': result.extraction_method,
                'confidence': result.confidence,
                'evidence_count': len(evidence),
                'section_evidence_count': len(evidence),
                'citation_evidence_count': 0
            }
        }
    except Exception as e:
        logger.error(f"Ablation 3B failed: {e}")
        return {
            'limitation': '',
            'error': str(e),
            'metadata': {'use_citation': False}
        }


# ============================================================================
# 主运行函数
# ============================================================================

def run_all_ablation_studies(
    papers_dir: str,
    output_dir: str,
    config_path: str = None,
    grobid_url: str = None,
    limit: int = None,
    ablation_choice: str = 'all'
):
    """
    运行所有消融实验

    Args:
        papers_dir: PDF文件目录
        output_dir: 输出目录
        config_path: LLM配置文件路径
        grobid_url: GROBID服务URL
        limit: 限制处理论文数量
        ablation_choice: 选择运行哪个消融实验 ('1', '2', '3', 'all')
    """
    print("="*80)
    print("消融实验 (Ablation Studies)")
    print("="*80)
    print(f"运行实验: {ablation_choice}")
    print("="*80)

    # 加载LLM client
    if not config_path:
        config_path = str(kg_demo_root / "config" / "config.yaml")

    try:
        config = LLMConfig.from_file(config_path)
        llm_client = LLMClient(config)
        print(f"✅ LLM Client loaded: {config.provider} - {config.model}")
    except Exception as e:
        print(f"❌ Failed to load LLM client: {e}")
        return

    # 加载论文 (支持PDF和TXT)
    papers_path = Path(papers_dir)

    # 检查是PDF目录还是TXT目录
    pdf_files = sorted(list(papers_path.glob("*.pdf")))
    txt_files = sorted(list(papers_path.glob("*.txt")))

    if pdf_files and txt_files:
        print(f"⚠️ 警告: 目录同时包含PDF和TXT文件,将优先使用TXT文件")
        paper_files = txt_files
        file_type = 'txt'
    elif txt_files:
        paper_files = txt_files
        file_type = 'txt'
    elif pdf_files:
        paper_files = pdf_files
        file_type = 'pdf'
    else:
        print(f"❌ 错误: 目录中没有找到PDF或TXT文件")
        return

    if limit:
        paper_files = paper_files[:limit]

    print(f"📚 Found {len(paper_files)} {file_type.upper()} files")

    # 准备结果存储
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 决定运行哪些实验
    run_ablation_1 = ablation_choice in ['1', 'all']
    run_ablation_2 = ablation_choice in ['2', 'all']
    run_ablation_3 = ablation_choice in ['3', 'all']

    ablation_1_results_a = []
    ablation_1_results_b = []
    ablation_2_results_a = []
    ablation_2_results_b = []
    ablation_3_results_a = []
    ablation_3_results_b = []

    # 处理每篇论文
    for i, paper_file in enumerate(paper_files, 1):
        paper_id = paper_file.stem.split('_')[0] if '_' in paper_file.stem else paper_file.stem

        print(f"\n[{i}/{len(paper_files)}] {paper_id}")
        print(f"   File: {paper_file.name}")

        try:
            # 根据文件类型解析论文
            if file_type == 'txt':
                print(f"         📝 Loading from parsed TXT file...")
                paper = load_paper_from_txt(str(paper_file))
                print(f"         ✅ Loaded {len(paper.sections)} sections")
            else:  # pdf
                print(f"         📄 Parsing PDF file...")
                paper = extract_paper_from_pdf(str(paper_file), grobid_url)

            # ========== 消融实验 1: CriticAgent ==========
            if run_ablation_1:
                print(f"   🔬 Ablation 1: Reflection Loop (CriticAgent)")

                start_time = time.time()
                result_1a = ablation_1_with_critic(paper, llm_client, paper_id)
                time_1a = time.time() - start_time

                ablation_1_results_a.append({
                    'paper_id': paper_id,
                    'problem': result_1a.get('problem', ''),
                    'method': result_1a.get('method', ''),
                    'limitation': result_1a.get('limitation', ''),
                    'future_work': result_1a.get('future_work', ''),
                    'extraction_time': time_1a,
                    'metadata': result_1a.get('metadata', {})
                })

                start_time = time.time()
                result_1b = ablation_1_without_critic(paper, llm_client, paper_id)
                time_1b = time.time() - start_time

                ablation_1_results_b.append({
                    'paper_id': paper_id,
                    'problem': result_1b.get('problem', ''),
                    'method': result_1b.get('method', ''),
                    'limitation': result_1b.get('limitation', ''),
                    'future_work': result_1b.get('future_work', ''),
                    'extraction_time': time_1b,
                    'metadata': result_1b.get('metadata', {})
                })

                print(f"      ✅ Ablation 1 completed (A: {time_1a:.1f}s, B: {time_1b:.1f}s)")

            # ========== 消融实验 2: LogicAnalystAgent ==========
            if run_ablation_2:
                print(f"   🔬 Ablation 2: LogicAnalyst vs. Simple Summarization")

                start_time = time.time()
                result_2a = ablation_2_with_logic_analyst(paper, llm_client)
                time_2a = time.time() - start_time

                ablation_2_results_a.append({
                    'paper_id': paper_id,
                    'problem': result_2a.get('problem', ''),
                    'method': result_2a.get('method', ''),
                    'extraction_time': time_2a,
                    'metadata': result_2a.get('metadata', {})
                })

                start_time = time.time()
                result_2b = ablation_2_simple_summarization(paper, llm_client)
                time_2b = time.time() - start_time

                ablation_2_results_b.append({
                    'paper_id': paper_id,
                    'problem': result_2b.get('problem', ''),
                    'method': result_2b.get('method', ''),
                    'extraction_time': time_2b,
                    'metadata': result_2b.get('metadata', {})
                })

                print(f"      ✅ Ablation 2 completed (A: {time_2a:.1f}s, B: {time_2b:.1f}s)")

            # ========== 消融实验 3: Citation Detective ==========
            if run_ablation_3:
                print(f"   🔬 Ablation 3: Citation Detective (Dual-Stream)")

                start_time = time.time()
                result_3a = ablation_3_with_citation(paper, llm_client, paper_id)
                time_3a = time.time() - start_time

                ablation_3_results_a.append({
                    'paper_id': paper_id,
                    'limitation': result_3a.get('limitation', ''),
                    'extraction_time': time_3a,
                    'metadata': result_3a.get('metadata', {})
                })

                start_time = time.time()
                result_3b = ablation_3_section_only(paper, llm_client, paper_id)
                time_3b = time.time() - start_time

                ablation_3_results_b.append({
                    'paper_id': paper_id,
                    'limitation': result_3b.get('limitation', ''),
                    'extraction_time': time_3b,
                    'metadata': result_3b.get('metadata', {})
                })

                print(f"      ✅ Ablation 3 completed (A: {time_3a:.1f}s, B: {time_3b:.1f}s)")

        except Exception as e:
            print(f"   ❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Rate limiting
        time.sleep(0.5)

    # 保存结果
    print(f"\n{'='*80}")
    print("保存结果...")
    print(f"{'='*80}")

    if run_ablation_1:
        output_1 = {
            'ablation_1_with_critic': ablation_1_results_a,
            'ablation_1_without_critic': ablation_1_results_b,
            'summary': {
                'total_papers': len(paper_files),
                'description': 'Ablation Study 1: Impact of CriticAgent (Reflection Loop)',
                'variant_a': 'Full system with iterative refinement',
                'variant_b': 'Single-pass extraction without Critic'
            }
        }
        output_file_1 = output_path / 'ablation_1_critic_results.json'
        with open(output_file_1, 'w', encoding='utf-8') as f:
            json.dump(output_1, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Ablation 1 results saved: {output_file_1}")

    if run_ablation_2:
        output_2 = {
            'ablation_2_logic_analyst': ablation_2_results_a,
            'ablation_2_simple_summarization': ablation_2_results_b,
            'summary': {
                'total_papers': len(paper_files),
                'description': 'Ablation Study 2: LogicAnalystAgent vs. Simple Summarization',
                'variant_a': 'LogicAnalystAgent with causal reasoning',
                'variant_b': 'Simple summarization without reasoning'
            }
        }
        output_file_2 = output_path / 'ablation_2_logic_analyst_results.json'
        with open(output_file_2, 'w', encoding='utf-8') as f:
            json.dump(output_2, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Ablation 2 results saved: {output_file_2}")

    if run_ablation_3:
        output_3 = {
            'ablation_3_with_citation': ablation_3_results_a,
            'ablation_3_section_only': ablation_3_results_b,
            'summary': {
                'total_papers': len(paper_files),
                'description': 'Ablation Study 3: Impact of Citation Detective',
                'variant_a': 'Dual-stream (Section + Citation analysis)',
                'variant_b': 'Section-only (no citation analysis)'
            }
        }
        output_file_3 = output_path / 'ablation_3_citation_results.json'
        with open(output_file_3, 'w', encoding='utf-8') as f:
            json.dump(output_3, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Ablation 3 results saved: {output_file_3}")

    print(f"\n{'='*80}")
    print("✅ 所有消融实验完成!")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='DeepPaper 2.0 消融实验 (Ablation Studies)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
消融实验说明:

1. Ablation 1: Reflection Loop (CriticAgent) 的有效性
   测试迭代优化对提取质量的影响

2. Ablation 2: LogicAnalystAgent vs. 简单提取
   测试因果推理对Problem-Method配对的影响

3. Ablation 3: Citation Detective (双流合并) 的增益
   测试引用分析对发现隐式局限性的影响

示例:
    # 运行所有消融实验 (使用TXT文件)
    python ablation_studies.py --papers_dir ../data/papers_txt --output_dir ./results/ablation

    # 运行所有消融实验 (使用PDF文件)
    python ablation_studies.py --papers_dir ../data/papers_pdf --output_dir ./results/ablation

    # 只运行实验1
    python ablation_studies.py --papers_dir ../data/papers_txt --output_dir ./results/ablation --ablation 1

    # 测试模式 (前3篇)
    python ablation_studies.py --papers_dir ../data/papers_txt --output_dir ./results/ablation --limit 3
        """
    )

    parser.add_argument(
        '--papers_dir',
        type=str,
        required=True,
        help='论文文件目录 (支持PDF或已解析的TXT文件)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results/ablation',
        help='输出目录 (default: ./results/ablation)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='LLM配置文件路径'
    )
    parser.add_argument(
        '--grobid_url',
        type=str,
        default=None,
        help='GROBID服务URL'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='限制处理论文数量 (用于测试)'
    )
    parser.add_argument(
        '--ablation',
        type=str,
        default='all',
        choices=['1', '2', '3', 'all'],
        help='选择运行哪个消融实验 (1, 2, 3, 或 all)'
    )

    args = parser.parse_args()

    run_all_ablation_studies(
        papers_dir=args.papers_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        grobid_url=args.grobid_url,
        limit=args.limit,
        ablation_choice=args.ablation
    )


if __name__ == '__main__':
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    main()
