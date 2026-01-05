"""
DeepPaper - Multi-Agent Based Deep Paper Analysis System
Entry file

Usage:
    python src/DeepPaper.py --paper-id W2741809807
    python src/DeepPaper.py --pdf path/to/paper.pdf
    python src/DeepPaper.py --batch --topic "transformer"

Core features:
- Iterative Multi-Agent architecture (Navigator → Extractor → Critic → Synthesizer)
- Reflection Loop automatic quality control
- Output structured JSON with Evidence
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DeepPaper_Agent import DeepPaperOrchestrator
from DeepPaper_Agent.data_structures import PaperDocument, PaperSection

# Import existing modules
from src.llm_config import LLMClient, LLMConfig
from src.grobid_parser import GrobidPDFParser
from src.openalex_client import OpenAlexClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_llm_client(config_path: str = "./config/config.yaml") -> Optional[LLMClient]:
    """
    Load LLM client

    Args:
        config_path: Configuration file path

    Returns:
        LLMClient instance or None
    """
    try:
        config = LLMConfig.from_file(config_path)
        client = LLMClient(config)
        logger.info(f"✅ LLM client loaded successfully: {config.provider} - {config.model}")
        return client
    except Exception as e:
        logger.error(f"❌ LLM client loading failed: {e}")
        logger.error("   Please check configuration file: config/config.yaml")
        sys.exit(1)


def extract_paper_from_pdf(
    pdf_path: str,
    grobid_url: Optional[str] = None
) -> PaperDocument:
    """
    Extract paper information from PDF

    Args:
        pdf_path: PDF file path
        grobid_url: GROBID service URL

    Returns:
        PaperDocument
    """
    logger.info(f"📄 Loading paper from PDF: {pdf_path}")

    # Use GROBID parsing
    sections = []
    title = "Unknown"
    abstract = ""

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

            logger.info(f"  ✅ GROBID parsing successful, extracted {len(sections)} sections")

        except Exception as e:
            logger.warning(f"  ⚠️ GROBID parsing failed: {e}")

    # If GROBID fails, fallback to PyPDF2
    if not sections:
        logger.info("  Using PyPDF2 parsing...")
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()

                # Simple split
                sections.append(PaperSection(
                    title="Full Text",
                    content=text[:5000],  # Limit length
                    page_num=0,
                    section_type='other'
                ))

                logger.info("  ✅ PyPDF2 parsing completed")

        except Exception as e:
            logger.error(f"  ❌ PDF parsing failed: {e}")
            sys.exit(1)

    # Generate paper_id
    paper_id = Path(pdf_path).stem

    return PaperDocument(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        sections=sections
    )


def fetch_paper_from_openalex(paper_id: str) -> PaperDocument:
    """
    Fetch paper information from OpenAlex

    Args:
        paper_id: OpenAlex paper ID

    Returns:
        PaperDocument
    """
    logger.info(f"🔍 Fetching paper from OpenAlex: {paper_id}")

    try:
        client = OpenAlexClient()
        paper_data = client.get_paper_by_id(paper_id)

        if not paper_data:
            logger.error(f"❌ Paper not found: {paper_id}")
            sys.exit(1)

        # Build PaperDocument
        title = paper_data.get('title', 'Untitled')
        abstract = paper_data.get('abstract', '')

        # Build simple sections from abstract
        sections = []
        if title:
            sections.append(PaperSection(
                title='Title',
                content=title,
                page_num=0,
                section_type='title'
            ))
        if abstract:
            sections.append(PaperSection(
                title='Abstract',
                content=abstract,
                page_num=0,
                section_type='abstract'
            ))

        authors = [
            author.get('author', {{}}).get('display_name', 'Unknown')
            for author in paper_data.get('authorships', [])
        ]

        logger.info(f"  ✅ Paper information retrieved successfully: {title}")
        more_text = '...' if len(authors) > 3 else ''
        logger.info(f"     Authors: {', '.join(authors[:3])}{more_text}")

        return PaperDocument(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            year=paper_data.get('publication_year'),
            sections=sections,
            metadata=paper_data
        )

    except Exception as e:
        logger.error(f"❌ Failed to fetch paper: {e}")
        sys.exit(1)


def batch_analyze_from_topic(
    topic: str,
    max_papers: int,
    llm_client: LLMClient,
    output_dir: str
):
    """
    Batch analyze papers on a topic

    Args:
        topic: Topic keyword
        max_papers: Maximum number of papers
        llm_client: LLM client
        output_dir: Output directory
    """
    logger.info(f"🔍 Searching topic: '{topic}'")

    # Search papers
    openalex_client = OpenAlexClient()
    papers = openalex_client.search_papers(
        topic=topic,
        max_results=max_papers,
        sort_by="cited_by_count"
    )

    logger.info(f"✅ Found {len(papers)} papers")

    # Batch analysis
    orchestrator = DeepPaperOrchestrator(
        llm_client=llm_client,
        max_retries=2
    )

    reports = orchestrator.batch_analyze_papers(
        papers=papers,
        output_dir=output_dir
    )

    logger.info(f"\n✅ Batch analysis completed! Successfully analyzed {len(reports)} papers")
    logger.info(f"📂 Output directory: {output_dir}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="DeepPaper - Multi-Agent Based Deep Paper Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single paper (via OpenAlex ID)
  python src/DeepPaper.py --paper-id W2741809807

  # Analyze PDF file
  python src/DeepPaper.py --pdf data/papers/example.pdf

  # Batch analyze topic
  python src/DeepPaper.py --batch --topic "transformer" --max-papers 5

Output:
  - JSON report: output/deep_paper_ID.json
  - Markdown report: output/deep_paper_ID.md
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--paper-id',
        help='OpenAlex paper ID (e.g., W2741809807)'
    )
    input_group.add_argument(
        '--pdf',
        help='PDF file path'
    )
    input_group.add_argument(
        '--batch',
        action='store_true',
        help='Batch analysis mode'
    )

    # Batch mode parameters
    parser.add_argument(
        '--topic',
        help='Topic keyword for batch mode'
    )
    parser.add_argument(
        '--max-papers',
        type=int,
        default=10,
        help='Maximum number of papers in batch mode (default: 10)'
    )

    # Configuration parameters
    parser.add_argument(
        '--config',
        default='./config/config.yaml',
        help='LLM configuration file path (default: ./config/config.yaml)'
    )
    parser.add_argument(
        '--grobid-url',
        help='GROBID service URL (e.g., http://localhost:8070)'
    )
    parser.add_argument(
        '--output-dir',
        default='./output/deep_paper',
        help='Output directory (default: ./output/deep_paper)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=2,
        help='Maximum retry attempts for Critic (default: 2)'
    )

    args = parser.parse_args()

    # Check batch mode parameters
    if args.batch and not args.topic:
        parser.error("--batch mode requires --topic")

    # Print banner
    print_banner()

    # Load LLM client
    llm_client = load_llm_client(args.config)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.batch:
            # Batch analysis mode
            batch_analyze_from_topic(
                topic=args.topic,
                max_papers=args.max_papers,
                llm_client=llm_client,
                output_dir=str(output_dir)
            )

        else:
            # Single paper analysis
            # Get paper document
            if args.pdf:
                paper_doc = extract_paper_from_pdf(args.pdf, args.grobid_url)
            else:
                paper_doc = fetch_paper_from_openalex(args.paper_id)

            # Create orchestrator and analyze
            orchestrator = DeepPaperOrchestrator(
                llm_client=llm_client,
                max_retries=args.max_retries
            )

            report = orchestrator.analyze_paper(
                paper_document=paper_doc,
                output_dir=str(output_dir)
            )

            # Display result summary
            print_result_summary(report)

    except KeyboardInterrupt:
        logger.info("\n❌ User interrupted execution")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\n❌ Execution error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def print_banner():
    """Print banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                      🤖 DeepPaper Multi-Agent                        ║
║                   Deep Paper Analysis System v1.0                    ║
║                                                                      ║
║  Navigator → Extractor → Critic → Synthesizer                       ║
║  (Navigator)  (Extractor)  (Critic)  (Synthesizer)                  ║
║                                                                      ║
║  Core Features:                                                      ║
║  ✓ Iterative Reflection Loop                                        ║
║  ✓ Automatic Quality Control (Critic-driven)                        ║
║  ✓ Evidence-based Output (Verifiable source citations)              ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_result_summary(report):
    """Print result summary"""
    print("\n" + "="*80)
    print("📊 Analysis Result Summary")
    print("="*80)
    print(f"📖 Paper: {report.title}")
    print(f"🆔 ID: {report.paper_id}")
    print("\nExtraction Quality:")

    for field, score in report.extraction_quality.items():
        status = "✅" if score >= 0.6 else "⚠️" if score >= 0.3 else "❌"
        iterations = report.iteration_count.get(field, 0)
        print(f"  {status} {field}: {score:.2f} ({iterations} iterations)")

    print("\nContent Preview:")
    print(f"  🎯 Problem: {report.problem[:100]}...")
    print(f"  💡 Contribution: {report.contribution[:100]}...")
    print(f"  ⚠️ Limitation: {report.limitation[:100]}...")
    print(f"  🔮 Future Work: {report.future_work[:100]}...")

    print("\n" + "="*80)
    print(f"📂 Complete report saved")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
