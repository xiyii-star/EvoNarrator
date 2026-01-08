#!/usr/bin/env python3
"""
Ablation Study: No Critic (单次提取，无迭代优化)
消融实验：移除Critic组件，只进行一次提取，无反馈循环

This tests the impact of the Critic component by:
- Using Navigator for section localization (kept)
- Using Extractor for extraction (kept)
- Using Dual-Stream for Problem/Method (kept)
- Skipping Critic feedback and refinement loops (removed)

对比实验设计:
- 完整系统: Navigator -> Extractor -> Critic (迭代) -> Synthesizer
- 本消融版本: Navigator -> Extractor (单次) -> Synthesizer
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import sys
import logging

# Add parent directory to path to import agents
kg_demo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(kg_demo_root))

from DeepPaper_Agent.navigator_agent import NavigatorAgent
from DeepPaper_Agent.extractor_agent import ExtractorAgent
from DeepPaper_Agent.fast_stream_extractor import FastStreamExtractor
from DeepPaper_Agent.dual_stream_synthesizer import DualStreamSynthesizer
from DeepPaper_Agent.data_structures import (
    PaperDocument,
    PaperSection,
    FieldType,
    ExtractionResult
)
from src.llm_config import LLMClient, LLMConfig
from src.grobid_parser import GrobidPDFParser

logger = logging.getLogger(__name__)


def extract_paper_from_pdf(
    pdf_path: str,
    grobid_url: Optional[str] = None
) -> PaperDocument:
    """
    从PDF提取论文信息

    Args:
        pdf_path: PDF文件路径
        grobid_url: GROBID服务URL

    Returns:
        PaperDocument
    """
    sections = []
    title = "Unknown"
    abstract = ""

    # 尝试使用GROBID解析
    if grobid_url:
        try:
            print(f"         🔧 尝试GROBID解析...")
            parser = GrobidPDFParser(grobid_url)
            sections = parser.extract_sections_from_pdf(pdf_path)

            if sections:
                # 提取标题和摘要
                for section in sections:
                    if section.section_type == 'title':
                        title = section.content
                    elif section.section_type == 'abstract':
                        abstract = section.content

                print(f"         ✅ GROBID解析成功: {len(sections)} 个章节")
            else:
                print(f"         ⚠️ GROBID返回空结果")

        except Exception as e:
            print(f"         ⚠️ GROBID解析失败: {e}")
            import traceback
            traceback.print_exc()

    # 降级到PyPDF2（如果需要且sections为空）
    if not sections:
        if not grobid_url:
            print(f"         使用PyPDF2解析（未配置GROBID）...")
        else:
            print(f"         降级到PyPDF2解析...")

        try:
            import PyPDF2
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()

                # 简单分割
                sections.append(PaperSection(
                    title="Full Text",
                    content=text[:10000],  # 限制长度
                    page_num=0,
                    section_type='other'
                ))

                print(f"         ✅ PyPDF2解析完成")

        except Exception as e:
            print(f"         ❌ PDF解析失败: {e}")
            # 返回空文档
            sections.append(PaperSection(
                title="Error",
                content=f"Failed to parse PDF: {e}",
                page_num=0,
                section_type='other'
            ))

    # 生成paper_id (从文件名提取)
    paper_id = Path(pdf_path).stem.split('_')[0] if '_' in Path(pdf_path).stem else Path(pdf_path).stem

    return PaperDocument(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        sections=sections
    )


def create_paper_document_from_text(paper_id: str, paper_text: str, title: str = "") -> PaperDocument:
    """
    Convert plain text paper to PaperDocument format

    Args:
        paper_id: Paper identifier
        paper_text: Full paper text
        title: Paper title (optional)

    Returns:
        PaperDocument instance
    """
    # Simple section splitting based on common headers
    sections = []
    lines = paper_text.split('\n')
    current_section_title = "Full Text"
    current_section_content = []

    common_headers = ['abstract', 'introduction', 'related work', 'methodology',
                      'method', 'approach', 'experiments', 'results', 'discussion',
                      'conclusion', 'future work', 'limitations', 'references']

    for line in lines:
        line_stripped = line.strip()
        line_lower = line_stripped.lower()

        # Skip separator lines (=====, -----, etc.)
        if line_stripped and all(c in '=-_*' for c in line_stripped):
            continue

        # Check if this line is a section header
        is_header = False
        for header in common_headers:
            if line_lower == header or line_lower.startswith(header + ' '):
                is_header = True
                break

        if is_header:
            # Save previous section if it has content
            if current_section_content:
                # Determine section type for previous section
                section_type_for_previous = 'other'
                title_lower = current_section_title.lower()
                if 'abstract' in title_lower:
                    section_type_for_previous = 'abstract'
                elif 'introduction' in title_lower or 'intro' in title_lower:
                    section_type_for_previous = 'introduction'
                elif 'conclusion' in title_lower:
                    section_type_for_previous = 'conclusion'
                elif 'reference' in title_lower:
                    section_type_for_previous = 'references'

                sections.append(PaperSection(
                    title=current_section_title,
                    content='\n'.join(current_section_content),
                    page_num=len(sections),
                    section_type=section_type_for_previous
                ))

            # Start new section
            current_section_title = line_stripped
            current_section_content = []
        elif line_stripped:  # Only add non-empty lines
            current_section_content.append(line)

    # Add last section
    if current_section_content:
        # Determine section type for last section
        section_type_for_last = 'other'
        title_lower = current_section_title.lower()
        if 'abstract' in title_lower:
            section_type_for_last = 'abstract'
        elif 'introduction' in title_lower or 'intro' in title_lower:
            section_type_for_last = 'introduction'
        elif 'conclusion' in title_lower:
            section_type_for_last = 'conclusion'
        elif 'reference' in title_lower:
            section_type_for_last = 'references'

        sections.append(PaperSection(
            title=current_section_title,
            content='\n'.join(current_section_content),
            page_num=len(sections),
            section_type=section_type_for_last
        ))

    # If no sections found, use whole text as one section
    if not sections:
        sections.append(PaperSection(
            title="Full Text",
            content=paper_text,
            page_num=0,
            section_type='other'
        ))

    # Extract abstract from sections if available
    abstract = ""
    for section in sections:
        if section.section_type == 'abstract':
            abstract = section.content
            break

    return PaperDocument(
        paper_id=paper_id,
        title=title or f"Paper {paper_id}",
        abstract=abstract,
        sections=sections
    )


def extract_without_critic(
    paper: PaperDocument,
    llm_client: LLMClient,
    use_dual_stream: bool = True
) -> Dict[FieldType, ExtractionResult]:
    """
    Extract information without Critic component (single-pass extraction)

    使用与完整系统相同的流程，但移除Critic反馈循环:
    - 对于Problem/Method: 使用Dual-Stream (Fast + Slow)
    - 对于Limitation/Future Work: 使用传统流程 (Navigator + Extractor)
    - 所有字段都只提取一次，不进行Critic验证和重试

    Args:
        paper: Paper document
        llm_client: LLM client instance
        use_dual_stream: Whether to use dual-stream for Problem/Method

    Returns:
        Dict mapping FieldType to ExtractionResult
    """
    print(f"   🔍 Extracting without Critic (single-pass, no refinement)...")

    # Initialize agents
    navigator = NavigatorAgent(llm_client)
    extractor = ExtractorAgent(llm_client)

    # Dual-stream components (if enabled)
    fast_stream_extractor = FastStreamExtractor(llm_client) if use_dual_stream else None
    dual_stream_synthesizer = DualStreamSynthesizer(llm_client) if use_dual_stream else None

    results = {}
    fields = [FieldType.PROBLEM, FieldType.METHOD, FieldType.LIMITATION, FieldType.FUTURE_WORK]

    for field in fields:
        print(f"      Processing {field.value}...")

        # 判断是否使用Dual-Stream
        if use_dual_stream and field in [FieldType.PROBLEM, FieldType.METHOD]:
            # Dual-Stream策略
            print(f"         → Using Dual-Stream strategy")

            # Step 1: Fast Stream - 提取高层锚点
            fast_result = fast_stream_extractor.extract_anchor(paper, field)

            # Step 2: Slow Stream - Navigator + Extractor (单次，无Critic)
            scope = navigator.navigate(paper, field)
            slow_result = extractor.extract(
                paper=paper,
                scope=scope,
                retry_prompt=None,
                anchor_guidance=fast_result.content
            )

            # Step 3: Dual-Stream Synthesizer - 合并结果
            merged_result = dual_stream_synthesizer.merge(
                fast_result=fast_result,
                slow_result=slow_result,
                field=field
            )

            results[field] = merged_result

        else:
            # 传统流程: Navigator + Extractor (单次，无Critic)
            print(f"         → Using traditional flow")

            # Step 1: Navigate to relevant sections
            scope = navigator.navigate(paper, field)

            # Step 2: Extract (single pass, no refinement)
            extraction = extractor.extract(
                paper=paper,
                scope=scope,
                retry_prompt=None
            )

            results[field] = extraction

        print(f"         ✅ Extracted: {results[field].content[:80]}...")

    return results


def run_ablation_no_critic(
    golden_set_path: str,
    papers_dir: str,
    output_path: str,
    config_path: str = None,
    grobid_url: str = None,
    use_dual_stream: bool = True
):
    """
    Run ablation study without Critic

    Args:
        golden_set_path: Path to golden set Excel
        papers_dir: Directory with paper text files
        output_path: Where to save results
        config_path: LLM config file path
        use_dual_stream: Whether to use dual-stream for Problem/Method
    """
    import pandas as pd

    print("="*80)
    print("Ablation Study: No Critic (Single-Pass Extraction)")
    print("="*80)
    print(f"Dual-Stream: {'Enabled' if use_dual_stream else 'Disabled'}")
    print("="*80)

    # Load LLM client
    if not config_path:
        config_path = str(Path(__file__).parent.parent.parent.parent / "config" / "config.yaml")

    try:
        config = LLMConfig.from_file(config_path)
        llm_client = LLMClient(config)
        print(f"✅ LLM Client loaded: {config.provider} - {config.model}")
    except Exception as e:
        print(f"❌ Failed to load LLM client: {e}")
        print(f"   Config path: {config_path}")
        return

    # Load golden set
    df = pd.read_excel(golden_set_path)

    # Filter annotated papers (支持不同的列名格式)
    problem_col = 'human_problem' if 'human_problem' in df.columns else 'Human_Problem'
    contribution_col = 'human_contribution' if 'human_contribution' in df.columns else 'Human_Contribution'

    annotated = df[
        df[problem_col].notna() & (df[problem_col] != '') &
        df[contribution_col].notna() & (df[contribution_col] != '')
    ]

    print(f"📚 Found {len(annotated)} annotated papers")

    # Process each paper
    results = []
    papers_path = Path(papers_dir)

    for i, (idx, row) in enumerate(annotated.iterrows(), 1):
        paper_id = row['Paper_ID']
        title = row.get('Paper_Title', '')
        openalex_id = row.get('OpenAlex_ID', '')

        print(f"\n[{i}/{len(annotated)}] Processing {paper_id}: {title[:60]}...")

        start_time = time.time()

        # 尝试多种方式加载论文
        paper = None

        # 方式1: 从PDF加载 (优先，如果papers_dir包含PDF)
        pdf_file = None
        for pdf_pattern in [f"{openalex_id}*.pdf", f"{paper_id}.pdf"]:
            pdf_matches = list(papers_path.glob(pdf_pattern))
            if pdf_matches:
                pdf_file = pdf_matches[0]
                break

        if pdf_file and pdf_file.exists():
            print(f"   📄 Loading from PDF: {pdf_file.name}")
            paper = extract_paper_from_pdf(str(pdf_file), grobid_url)

        # 方式2: 从TXT加载 (降级)
        if paper is None:
            text_file = papers_path / f"{paper_id}.txt"
            alt_text_file = papers_path / f"{paper_id}_text.txt"

            if text_file.exists():
                print(f"   📝 Loading from TXT: {text_file.name}")
                with open(text_file, 'r', encoding='utf-8') as f:
                    paper_text = f.read()
                paper = create_paper_document_from_text(paper_id, paper_text, title)
            elif alt_text_file.exists():
                print(f"   📝 Loading from TXT: {alt_text_file.name}")
                with open(alt_text_file, 'r', encoding='utf-8') as f:
                    paper_text = f.read()
                paper = create_paper_document_from_text(paper_id, paper_text, title)

        if paper is None:
            print(f"   ⚠️  No paper file found (PDF or TXT), skipping...")
            continue

        # Extract without Critic
        extractions = extract_without_critic(
            paper=paper,
            llm_client=llm_client,
            use_dual_stream=use_dual_stream
        )

        extraction_time = time.time() - start_time

        result = {
            'paper_id': paper_id,
            'problem': extractions.get(FieldType.PROBLEM, ExtractionResult(
                field=FieldType.PROBLEM, content='', evidence=[]
            )).content,
            'method': extractions.get(FieldType.METHOD, ExtractionResult(
                field=FieldType.METHOD, content='', evidence=[]
            )).content,
            'limitation': extractions.get(FieldType.LIMITATION, ExtractionResult(
                field=FieldType.LIMITATION, content='', evidence=[]
            )).content,
            'future_work': extractions.get(FieldType.FUTURE_WORK, ExtractionResult(
                field=FieldType.FUTURE_WORK, content='', evidence=[]
            )).content,
            'extraction_time': extraction_time,
            'metadata': {
                'paper_id': paper_id,
                'title': title,
                'method': 'ablation_no_critic',
                'use_dual_stream': use_dual_stream,
                'iterations': 1  # Always 1 since no Critic refinement
            }
        }

        results.append(result)
        print(f"   ✅ Completed in {extraction_time:.1f}s")

    # Save results
    method_name = "ablation_no_critic_dual_stream" if use_dual_stream else "ablation_no_critic"
    output = {method_name: results}

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {output_file}")
    print(f"✅ Processed {len(results)} papers")

    avg_time = sum(r['extraction_time'] for r in results) / len(results) if results else 0
    print(f"\n📊 Average extraction time: {avg_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(description='Ablation study: extraction without Critic')
    parser.add_argument('--golden_set', type=str, required=True,
                        help='Path to golden set Excel file')
    parser.add_argument('--papers_dir', type=str, required=True,
                        help='Directory containing paper files (PDF or TXT)')
    parser.add_argument('--output', type=str, default='./results/ablation_no_critic_results.json',
                        help='Output JSON file path')
    parser.add_argument('--config', type=str, default=None,
                        help='LLM config file path')
    parser.add_argument('--grobid_url', type=str, default=None,
                        help='GROBID service URL (e.g., http://localhost:8070)')
    parser.add_argument('--no-dual-stream', action='store_true',
                        help='Disable dual-stream for Problem/Method')

    args = parser.parse_args()

    run_ablation_no_critic(
        golden_set_path=args.golden_set,
        papers_dir=args.papers_dir,
        output_path=args.output,
        config_path=args.config,
        grobid_url=args.grobid_url,
        use_dual_stream=not args.no_dual_stream
    )

    print("\n💡 This ablation shows the impact of Critic's iterative refinement")
    print("   Compare with full architecture to see if feedback loops improve quality")
    print("\n🔬 Key Comparison:")
    print("   - Full System: Navigator -> Extractor -> Critic (iterate) -> Final Result")
    print("   - This Ablation: Navigator -> Extractor (single-pass) -> Final Result")


if __name__ == '__main__':
    main()
