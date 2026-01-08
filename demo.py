"""
Usage:
  python demo.py [topic_keywords]

Examples:
  python demo.py "transformer"
  python demo.py "computer vision"
  python demo.py "natural language processing"
"""

import sys
import logging
from pathlib import Path
import yaml
import argparse
from datetime import datetime

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from pipeline import PaperGraphPipeline


def setup_logging():
    """Setup logging"""
    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"demo_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    return logger


def load_config():
    """Load configuration file"""
    config_file = Path('./config/config.yaml')

    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"⚠️  Failed to load config file: {e}")
            return {}
    else:
        print("⚠️  Config file not found, using default configuration")
        return {}


def print_banner():
    """Print banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║              📚 Paper Knowledge Graph Demo                    ║
║                   (Deep Analysis via RAG)                     ║
║                                                              ║
║  🔍 Paper Search → 📥 PDF Download → 🧠 RAG Analysis → 🕸️ KG ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Paper Knowledge Graph Construction Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python demo.py transformer
        python demo.py "computer vision"
        python demo.py "natural language processing"

        Output files:
        - output/papers_*.json         # Paper data
        - output/graph_data_*.json     # Graph data
        - output/graph_viz_*.html      # Interactive visualization
        - output/analysis_report_*.json # Analysis report
        """
    )

    parser.add_argument(
        'topic',
        help='Research topic keywords (e.g., transformer, computer vision)'
    )

    parser.add_argument(
        '--max-papers',
        type=int,
        default=15,
        help='Maximum number of papers (default: 15)'
    )

    parser.add_argument(
        '--skip-pdf',
        action='store_true',
        help='Skip PDF download'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode (reduce number of papers and citations)'
    )

    parser.add_argument(
        '--use-llm',
        action='store_true',
        help='Use LLM to enhance paper analysis (requires config.yaml)'
    )

    parser.add_argument(
        '--use-deep-paper',
        action='store_true',
        help='Use DeepPaper Multi-Agent system (recommended, with Reflection Loop)'
    )

    parser.add_argument(
        '--use-snowball',
        action='store_true',
        help='Use snowball search mode (five-step: seed→successor→ancestor→SOTA→closure)'
    )

    parser.add_argument(
        '--llm-config',
        type=str,
        default='./config/config.yaml',
        help='LLM config file path (default: ./config/config.yaml)'
    )

    return parser.parse_args()


def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    logger = setup_logging()

    # Print banner
    print_banner()

    try:
        # Load configuration
        yaml_config = load_config()

        # Flatten nested YAML config to the format required by pipeline
        config = {}
        if 'search' in yaml_config:
            config.update(yaml_config['search'])
        if 'pdf' in yaml_config:
            config.update({
                'download_pdfs': yaml_config['pdf'].get('download_enabled', True),
                'max_pdf_downloads': yaml_config['pdf'].get('max_downloads', 5),
                'timeout': yaml_config['pdf'].get('timeout', 60),
                'download_dir': yaml_config['pdf'].get('download_dir', './data/papers')
            })
        if 'output' in yaml_config:
            config.update({
                'output_dir': yaml_config['output']['base_dir'],
                'save_intermediate': yaml_config['output']['save_intermediate'],
                'generate_visualization': yaml_config['output']['generate_visualization']
            })
        if 'graph' in yaml_config:
            config.update({
                'max_nodes_in_viz': yaml_config['graph'].get('max_nodes_in_viz', 100),
                'enable_clustering': yaml_config['graph'].get('enable_clustering', True),
                'min_cluster_size': yaml_config['graph'].get('min_cluster_size', 3)
            })

        # Adjust configuration based on command line arguments
        if args.max_papers:
            config['max_papers'] = args.max_papers

        if args.skip_pdf:
            config['download_pdfs'] = False

        if args.quick:
            config.update({
                'max_papers': 8,
                'max_citations': 2,
                'max_references': 2,
                'max_pdf_downloads': 3
            })
            logger.info("🚀 Using quick mode")

        # Configure analysis method
        # 1. Priority: command line arguments > config.yaml configuration
        # 2. Default to DeepPaper if LLM configuration is available

        # Check if DeepPaper should be used
        use_deep_paper = False
        if args.use_deep_paper:
            use_deep_paper = True
            logger.info(f"🤖 Command line specified: Using DeepPaper Multi-Agent")
        elif 'deep_paper' in yaml_config and yaml_config['deep_paper'].get('enabled', False):
            use_deep_paper = True
            logger.info(f"🤖 Config file specified: Using DeepPaper Multi-Agent")
        elif 'llm' in yaml_config and yaml_config['llm'].get('enabled', False):
            # If LLM is available, default to DeepPaper
            use_deep_paper = True
            logger.info(f"🤖 Default: Using DeepPaper Multi-Agent (LLM available)")

        if use_deep_paper:
            config['use_deep_paper'] = True
            config['use_llm'] = True  # DeepPaper requires LLM
            config['llm_config_file'] = './config/config.yaml'

            # DeepPaper specific configuration
            if 'deep_paper' in yaml_config:
                config['deep_paper_max_retries'] = yaml_config['deep_paper'].get('max_retries', 2)
                config['save_deep_paper_reports'] = yaml_config['deep_paper'].get('save_individual_reports', False)
            else:
                config['deep_paper_max_retries'] = 2
                config['save_deep_paper_reports'] = False

            # GROBID configuration
            if 'grobid' in yaml_config and yaml_config['grobid'].get('enabled', False):
                config['grobid_url'] = yaml_config['grobid'].get('url')

            logger.info(f"   Provider: {yaml_config['llm'].get('provider', 'N/A')}")
            logger.info(f"   Model: {yaml_config['llm'].get('model', 'N/A')}")
            logger.info(f"   Max Retries: {config['deep_paper_max_retries']}")
            if config.get('grobid_url'):
                logger.info(f"   GROBID: {config['grobid_url']}")

        # Traditional RAG mode
        elif args.use_llm or ('llm' in yaml_config and yaml_config['llm'].get('enabled', False)):
            config['use_deep_paper'] = False
            config['use_llm'] = True
            config['llm_config_file'] = './config/config.yaml'

            if 'grobid' in yaml_config and yaml_config['grobid'].get('enabled', False):
                config['grobid_url'] = yaml_config['grobid'].get('url')

            logger.info(f"🔧 Using traditional RAG analysis")
            logger.info(f"   Provider: {yaml_config['llm'].get('provider', 'N/A')}")
            logger.info(f"   Model: {yaml_config['llm'].get('model', 'N/A')}")
            if config.get('grobid_url'):
                logger.info(f"   GROBID: {config['grobid_url']}")

        # No LLM mode
        else:
            config['use_deep_paper'] = False
            config['use_llm'] = False
            logger.info(f"⚠️ Not using LLM analysis (basic mode)")

        # Configure snowball search mode
        if args.use_snowball:
            logger.info(f"📊 Enabling snowball search mode (command line specified)")
            config['use_snowball'] = True  # Mark using snowball mode
            logger.info(f"   Five-step search: seed→successor→ancestor→SOTA→closure")
        elif 'snowball' in yaml_config and yaml_config['snowball'].get('enabled', False):
            logger.info(f"📊 Snowball search already enabled in config file")
            config['use_snowball'] = True

        # Display configuration information
        logger.info("📋 Configuration:")
        logger.info(f"  Topic: {args.topic}")
        logger.info(f"  Search mode: {'Snowball (five-step)' if config.get('use_snowball', False) else 'Traditional'}")
        logger.info(f"  Max papers: {config.get('max_papers', 15)}")
        logger.info(f"  Download PDF: {config.get('download_pdfs', True)}")
        logger.info(f"  Max PDF downloads: {config.get('max_pdf_downloads', 5)}")
        logger.info(f"  LLM enhanced: {config.get('use_llm', False)}")

        # Create and run pipeline
        logger.info("🚀 Starting paper knowledge graph construction pipeline...")

        pipeline = PaperGraphPipeline(config)
        results = pipeline.run(args.topic)

        # Display results
        print("\n" + "="*60)
        print("🎉 Run completed! Results summary:")
        print("="*60)
        print(f"📖 Topic: {results['topic']}")
        print(f"📊 Total papers: {results['summary']['total_papers']}")
        print(f"✅ Successful analysis: {results['summary']['successful_analysis']}")
        print(f"🧠 Analysis method: {results['summary'].get('analysis_method', 'N/A').upper()}")
        print(f"🔗 Citation edges: {results['summary']['citation_edges']}")
        print(f"📈 Graph nodes: {results['summary']['graph_nodes']}")
        print(f"📈 Graph edges: {results['summary']['graph_edges']}")

        print("\n📂 Generated files:")
        for file_type, file_path in results['files'].items():
            print(f"  {file_type}: {file_path}")

        viz_file = results['files']['visualization']
        print(f"\n💡 Open visualization file to view knowledge graph:")
        print(f"   file://{Path(viz_file).resolve()}")

        print("\n" + "="*60)

    except KeyboardInterrupt:
        logger.info("❌ User interrupted execution")
        sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Execution error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()