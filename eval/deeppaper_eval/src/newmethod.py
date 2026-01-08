#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
New Method: DeepPaper 2.0 Multi-Agent Architecture
Based on DeepPaper_Agent2.0 for deep information extraction

Workflow:
1. Problem & Method: LogicAnalystAgent extracts problems and methods
2. Limitation: LimitationExtractor (SectionLocator + CitationDetective)
3. Future Work: FutureWorkExtractor (SectionLocator)
4. Integrate results and output final report

Difference from mymethod.py:
- mymethod.py uses DeepPaper_Agent (1.0)
- newmethod.py uses DeepPaper_Agent2.0 (latest architecture)
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Add paths for imports
_project_root = Path(__file__).resolve().parent.parent.parent.parent
_deeppaper2_path = _project_root / "DeepPaper_Agent2.0"

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_deeppaper2_path) not in sys.path:
    sys.path.insert(0, str(_deeppaper2_path))

# Import DeepPaper 2.0 components
from orchestrator import DeepPaper2Orchestrator
from data_structures import PaperDocument, PaperSection
from src.llm_config import LLMClient, LLMConfig
from src.grobid_parser import GrobidPDFParser


def extract_paper_from_pdf(
    pdf_path: str,
    grobid_url: Optional[str] = None
) -> PaperDocument:
    """
    Extract paper information from PDF using GROBID or PyPDF2

    Args:
        pdf_path: PDF file path
        grobid_url: GROBID service URL

    Returns:
        PaperDocument
    """
    sections = []
    title = "Unknown"
    abstract = ""
    authors = []
    year = None

    # Try GROBID parsing
    if grobid_url:
        try:
            parser = GrobidPDFParser(grobid_url)
            sections = parser.extract_sections_from_pdf(pdf_path)

            # Extract title and abstract
            for section in sections:
                if section.section_type == 'title':
                    title = section.content
                elif section.section_type == 'abstract':
                    abstract = section.content

            print(f"         GROBID parsing successful: {len(sections)} sections")

        except Exception as e:
            print(f"         GROBID parsing failed: {e}")

    # Fallback to PyPDF2
    if not sections:
        print(f"         Using PyPDF2 fallback parsing...")
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()

                # Simple section split
                sections.append(PaperSection(
                    title="Full Text",
                    content=text[:15000],  # Limit length
                    page_num=0,
                    section_type='other'
                ))

                print(f"         PyPDF2 parsing complete")

        except Exception as e:
            print(f"         PDF parsing failed: {e}")
            # Return error section
            sections.append(PaperSection(
                title="Error",
                content=f"Failed to parse PDF: {e}",
                page_num=0,
                section_type='other'
            ))

    # Generate paper_id (extract from filename)
    paper_id = Path(pdf_path).stem.split('_')[0] if '_' in Path(pdf_path).stem else Path(pdf_path).stem

    return PaperDocument(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        authors=authors,
        year=year,
        sections=sections,
        metadata={}
    )


def extract_with_deeppaper2(
    paper_doc: PaperDocument,
    llm_client: LLMClient,
    use_citation_analysis: bool = False,
    paper_id: Optional[str] = None
) -> Dict[str, str]:
    """
    Extract deep information using DeepPaper 2.0 Multi-Agent system

    Args:
        paper_doc: Paper document
        llm_client: LLM client
        use_citation_analysis: Whether to use citation analysis for limitation
        paper_id: Paper ID (for citation analysis)

    Returns:
        Dictionary containing four dimensions
    """
    print(f"      > DeepPaper 2.0 Multi-Agent analysis...")
    print(f"         Section count: {len(paper_doc.sections)}")
    if use_citation_analysis:
        print(f"         Citation analysis: Enabled")

    # Initialize DeepPaper 2.0 orchestrator
    orchestrator = DeepPaper2Orchestrator(
        llm_client=llm_client,
        use_citation_analysis=use_citation_analysis
    )

    # Run Multi-Agent analysis
    try:
        report = orchestrator.analyze_paper(
            paper_document=paper_doc,
            paper_id=paper_id,
            output_dir=None  # Don't save intermediate files
        )

        return {
            'problem': report.problem,
            'method': report.method,
            'limitation': report.limitation,
            'future_work': report.future_work,
            'metadata': {
                'extraction_methods': report.metadata.get('extraction_methods', {}),
                'confidences': report.metadata.get('confidences', {}),
                'sections_count': len(paper_doc.sections)
            }
        }

    except Exception as e:
        print(f"      Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'problem': '',
            'method': '',
            'limitation': '',
            'future_work': '',
            'error': str(e),
            'metadata': {
                'sections_count': len(paper_doc.sections)
            }
        }


def load_pdfs_from_directory(pdf_dir: str) -> List[Dict]:
    """
    Load all PDF files from specified directory

    Args:
        pdf_dir: PDF file directory

    Returns:
        List of PDF file information
    """
    pdf_path = Path(pdf_dir)

    if not pdf_path.exists():
        print(f"Directory does not exist: {pdf_dir}")
        return []

    # Get all PDF files
    pdf_files = sorted(list(pdf_path.glob("*.pdf")))

    print(f"Found {len(pdf_files)} PDF files")

    papers = []
    for pdf_file in pdf_files:
        # Extract paper_id from filename (format: W1234567890_Title.pdf)
        paper_id = pdf_file.stem.split('_')[0] if '_' in pdf_file.stem else pdf_file.stem

        papers.append({
            'paper_id': paper_id,
            'pdf_path': str(pdf_file),
            'filename': pdf_file.name
        })

    return papers


def run_newmethod(
    golden_set_path: str,
    papers_dir: str,
    output_path: str,
    config_path: str = None,
    grobid_url: str = None,
    use_citation: bool = False,
    limit: int = None
):
    """
    Run DeepPaper 2.0 Multi-Agent method to analyze PDF papers

    Args:
        golden_set_path: Golden standard Excel file (optional, ignored here)
        papers_dir: PDF file directory
        output_path: Output JSON file path
        config_path: LLM configuration file path
        grobid_url: GROBID service URL
        use_citation: Whether to use citation analysis for limitation
        limit: Limit number of papers to process (for testing)
    """
    print("="*80)
    print("New Method: DeepPaper 2.0 Multi-Agent Architecture")
    print("="*80)
    print("Architecture: LogicAnalyst + SectionLocator + LimitationExtractor + FutureWorkExtractor")
    if use_citation:
        print("Citation analysis: Enabled (CitationDetectiveAgent)")
    print("="*80)

    # Load LLM client
    if not config_path:
        config_path = str(Path(__file__).parent.parent.parent.parent / "config" / "config.yaml")

    try:
        config = LLMConfig.from_file(config_path)
        llm_client = LLMClient(config)
        print(f"LLM client loaded successfully: {config.provider} - {config.model}")
    except Exception as e:
        print(f"LLM client loading failed: {e}")
        print(f"   Config file: {config_path}")
        return

    # Check GROBID
    if grobid_url:
        print(f"Using GROBID service: {grobid_url}")
    else:
        print(f"GROBID not configured, will use PyPDF2 fallback")

    # Load PDF files
    print(f"\nProcessing directory: {papers_dir}")
    papers = load_pdfs_from_directory(papers_dir)

    if not papers:
        print("No PDF files found!")
        return

    # Limit number (for testing)
    if limit:
        papers = papers[:limit]
        print(f"Limiting to first {limit} papers (test mode)")

    print(f"\n{'='*80}")
    print(f"Starting to process {len(papers)} papers")
    print(f"{'='*80}\n")

    # Extract each paper
    results = []
    success_count = 0
    pdf_count = 0

    for i, paper in enumerate(papers, 1):
        paper_id = paper['paper_id']
        pdf_path = paper['pdf_path']

        print(f"\n[{i}/{len(papers)}] {paper_id}")
        print(f"   File: {paper['filename']}")

        start_time = time.time()

        try:
            # Step 1: Extract paper document from PDF (using GROBID)
            paper_doc = extract_paper_from_pdf(pdf_path, grobid_url)

            # Step 2: Use DeepPaper 2.0 Multi-Agent analysis
            extraction = extract_with_deeppaper2(
                paper_doc=paper_doc,
                llm_client=llm_client,
                use_citation_analysis=use_citation,
                paper_id=paper_id
            )

            extraction_time = time.time() - start_time

            # Build result
            result = {
                'paper_id': paper_id,
                'problem': extraction['problem'],
                'method': extraction['method'],
                'limitation': extraction['limitation'],
                'future_work': extraction['future_work'],
                'extraction_time': extraction_time,
                'metadata': {
                    'paper_id': paper_id,
                    'filename': paper['filename'],
                    'method': 'newmethod_deeppaper2',
                    'use_citation_analysis': use_citation,
                    'grobid_enabled': grobid_url is not None,
                    **extraction.get('metadata', {})
                }
            }

            if 'error' in extraction:
                result['metadata']['error'] = extraction['error']
            else:
                success_count += 1

            results.append(result)

            print(f"   Completed (time: {extraction_time:.1f}s)")

            # Show confidence
            if 'confidences' in extraction.get('metadata', {}):
                confidences = extraction['metadata']['confidences']
                avg_confidence = sum(confidences.values()) / len(confidences) if confidences else 0
                print(f"   Average confidence: {avg_confidence:.2f}")

            # Count papers with PDF sections
            if extraction.get('metadata', {}).get('sections_count', 0) > 1:
                pdf_count += 1

        except Exception as e:
            print(f"   Processing failed: {e}")
            import traceback
            traceback.print_exc()

            # Add failure record
            results.append({
                'paper_id': paper_id,
                'problem': '',
                'method': '',
                'limitation': '',
                'future_work': '',
                'extraction_time': time.time() - start_time,
                'metadata': {
                    'paper_id': paper_id,
                    'filename': paper['filename'],
                    'method': 'newmethod_deeppaper2',
                    'error': str(e)
                }
            })

        # Rate limiting
        time.sleep(0.5)

    # Save results
    output = {
        "newmethod_deeppaper2": results,
        "summary": {
            "total_papers": len(papers),
            "successful": success_count,
            "failed": len(papers) - success_count,
            "with_pdf_sections": pdf_count,
            "success_rate": success_count / len(papers) if papers else 0,
            "use_citation_analysis": use_citation
        }
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    print(f"Statistics:")
    print(f"   Total papers: {len(papers)}")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {len(papers) - success_count}")
    print(f"   With PDF sections: {pdf_count}")
    print(f"   Success rate: {success_count / len(papers) * 100:.1f}%")

    # Average time
    avg_time = sum(r['extraction_time'] for r in results) / len(results) if results else 0
    print(f"   Average time: {avg_time:.1f}s/paper")

    # Average confidence
    all_confidences = []
    for r in results:
        if 'confidences' in r.get('metadata', {}):
            confs = r['metadata']['confidences']
            all_confidences.extend(confs.values())
    if all_confidences:
        avg_conf = sum(all_confidences) / len(all_confidences)
        print(f"   Average confidence: {avg_conf:.2f}")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='DeepPaper 2.0 Multi-Agent method: Extract deep information from PDFs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Called via run_all_experiments.py
    python newmethod.py --golden_set data/golden_set.xlsx --papers_dir data/papers_pdf --output result/newmethod.json

    # Direct call
    python newmethod.py --golden_set '' --papers_dir ../data/papers_pdf --output ../result/newmethod_results.json

    # Using GROBID
    python newmethod.py --golden_set '' --papers_dir ../data/papers_pdf --grobid_url http://localhost:8070

    # Enable citation analysis
    python newmethod.py --golden_set '' --papers_dir ../data/papers_pdf --use_citation

    # Test mode (process first 3)
    python newmethod.py --golden_set '' --papers_dir ../data/papers_pdf --limit 3
        """
    )

    parser.add_argument(
        '--golden_set',
        type=str,
        required=True,
        help='Golden standard Excel file path (compatibility parameter, not actually used)'
    )
    parser.add_argument(
        '--papers_dir',
        type=str,
        required=True,
        help='PDF file directory path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../result/newmethod_results.json',
        help='Output JSON file path (default: ../result/newmethod_results.json)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='LLM configuration file path (default: auto-find config.yaml)'
    )
    parser.add_argument(
        '--grobid_url',
        type=str,
        default=None,
        help='GROBID service URL (example: http://localhost:8070)'
    )
    parser.add_argument(
        '--use_citation',
        action='store_true',
        help='Whether to use citation analysis for limitation (CitationDetectiveAgent)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of papers to process (for testing, default: process all)'
    )

    args = parser.parse_args()

    run_newmethod(
        golden_set_path=args.golden_set,
        papers_dir=args.papers_dir,
        output_path=args.output,
        config_path=args.config,
        grobid_url=args.grobid_url,
        use_citation=args.use_citation,
        limit=args.limit
    )

    print("This is the new method based on DeepPaper 2.0 architecture")
    print("   Core components:")
    print("   - LogicAnalystAgent: Extract Problem & Method")
    print("   - LimitationExtractor: Extract Limitation (section location + citation analysis)")
    print("   - FutureWorkExtractor: Extract Future Work (section location)")
    print("   Output four dimensions: Problem, Method, Limitation, Future Work")


if __name__ == '__main__':
    main()
