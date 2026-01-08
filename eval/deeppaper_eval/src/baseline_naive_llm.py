#!/usr/bin/env python3
"""
Naive LLM Baseline for Deep Information Extraction
直接将论文全文(或截断)扔给LLM，让它总结四个字段

Baseline Model:
- Takes full text or truncated text (first N tokens due to context limit)
- Single prompt to extract all four fields at once
- No navigation, no critic feedback
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List
import argparse
from datetime import datetime
import pandas as pd

# Add parent directory to path to import LLM config
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.llm_config import LLMClient, LLMConfig


def create_naive_extraction_prompt(paper_text: str, max_length: int = 12000) -> str:
    """
    Create prompt for naive LLM extraction

    Args:
        paper_text: Full paper text
        max_length: Maximum characters to include (for context limit)

    Returns:
        Prompt string
    """
    # Truncate if needed
    if len(paper_text) > max_length:
        paper_text = paper_text[:max_length] + "\n\n[...truncated due to length...]"

    prompt = f"""You are a research paper analyzer. Please read the following paper and extract the following information:

1. **Problem**: What is the main problem or research question this paper addresses?
2. **Method**: What are the key methods, contributions or innovations of this paper?
3. **Limitation**: What are the limitations, weaknesses, or potential issues mentioned or implied in the paper?
4. **Future Work**: What future research directions or open problems does the paper suggest?

Please provide concise but comprehensive answers for each field. Be honest about limitations even if not explicitly stated.

Paper Text:
{paper_text}

Please respond in the following JSON format:
{{
    "problem": "Your answer here",
    "method": "Your answer here",
    "limitation": "Your answer here",
    "future_work": "Your answer here"
}}
"""
    return prompt


def extract_with_naive_llm(
    paper_text: str,
    llm_client: LLMClient,
    max_length: int = 12000
) -> Dict[str, str]:
    """
    Extract information using naive LLM approach

    Args:
        paper_text: Full paper text
        llm_client: LLM client instance
        max_length: Max characters to send to model

    Returns:
        Dict with extracted fields
    """
    prompt = create_naive_extraction_prompt(paper_text, max_length)

    try:
        # Use LLM client to generate
        result_text = llm_client.generate(
            prompt=prompt,
            system_prompt="You are a helpful research paper analysis assistant."
        )

        # Check if response is empty or error message
        if not result_text or result_text.strip() == "":
            raise ValueError("LLM returned empty response")

        if "LLM调用失败" in result_text or "LLM未配置" in result_text:
            raise ValueError(f"LLM error: {result_text}")

        # Try to parse JSON from response
        # Remove markdown code blocks if present
        result_text = result_text.strip()

        # More robust markdown code block removal
        if result_text.startswith("```json"):
            result_text = result_text[7:].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:].strip()

        if result_text.endswith("```"):
            result_text = result_text[:-3].strip()

        result = json.loads(result_text.strip())

        return {
            'problem': result.get('problem', ''),
            'method': result.get('method', ''),  # 统一使用method
            'limitation': result.get('limitation', ''),
            'future_work': result.get('future_work', ''),
            'raw_response': result_text
        }

    except Exception as e:
        print(f"   ❌ Error during extraction: {e}")
        print(f"      Raw response length: {len(result_text) if 'result_text' in locals() else 'N/A'}")
        if 'result_text' in locals() and result_text:
            print(f"      First 200 chars: {result_text[:200]}")
        return {
            'problem': '',
            'method': '',  # 统一使用method
            'limitation': '',
            'future_work': '',
            'error': str(e),
            'raw_response': result_text if 'result_text' in locals() else ''
        }


def load_papers_for_evaluation(golden_set_path: str, papers_dir: str) -> List[Dict]:
    """
    Load papers that need to be evaluated

    Args:
        golden_set_path: Path to golden set Excel
        papers_dir: Directory containing paper PDFs or text files

    Returns:
        List of dicts with paper_id and text
    """
    # Load golden set to get paper IDs
    df = pd.read_excel(golden_set_path)

    # Filter annotated papers (支持不同的列名格式)
    # 检测列名，支持多种格式（包括typo）
    problem_col = None
    contribution_col = None
    paper_id_col = None
    title_col = None

    # Find problem column (支持 human_problem, Human_Problem, Hunman_Problem等)
    for col in ['human_problem', 'Human_Problem', 'Hunman_Problem']:
        if col in df.columns:
            problem_col = col
            break

    # Find contribution column
    for col in ['human_contribution', 'Human_Contribution', 'Hunman_Contribution']:
        if col in df.columns:
            contribution_col = col
            break

    # Find paper_id column
    for col in ['paper_id', 'Paper_ID', 'Paper_Id']:
        if col in df.columns:
            paper_id_col = col
            break

    # Find title column
    for col in ['title', 'Title', 'Paper_Title']:
        if col in df.columns:
            title_col = col
            break

    if not problem_col or not contribution_col or not paper_id_col:
        raise ValueError(f"Could not find required columns. Available: {list(df.columns)}")

    annotated = df[
        df[problem_col].notna() & (df[problem_col] != '') &
        df[contribution_col].notna() & (df[contribution_col] != '')
    ]

    print(f"📚 Found {len(annotated)} annotated papers in golden set")

    papers = []
    papers_path = Path(papers_dir)

    for idx, row in annotated.iterrows():
        paper_id = row[paper_id_col]
        doi = row.get('DOI', row.get('doi', ''))
        openalex_id = row.get('OpenAlex_ID', row.get('openalex_id', ''))

        # Try to find paper text file
        # Expected format: papers/P001.txt or papers/P001_text.txt
        text_file = papers_path / f"{paper_id}.txt"
        alt_text_file = papers_path / f"{paper_id}_text.txt"

        paper_text = None
        if text_file.exists():
            with open(text_file, 'r', encoding='utf-8') as f:
                paper_text = f.read()
        elif alt_text_file.exists():
            with open(alt_text_file, 'r', encoding='utf-8') as f:
                paper_text = f.read()
        else:
            print(f"   ⚠️  Text file not found for {paper_id}")
            continue

        papers.append({
            'paper_id': paper_id,
            'title': row.get('Paper_Title', ''),
            'text': paper_text,
            'doi': doi,
            'openalex_id': openalex_id
        })

    print(f"✅ Loaded {len(papers)} paper texts")
    return papers


def run_naive_baseline(
    golden_set_path: str,
    papers_dir: str,
    output_path: str,
    config_path: str = None,
    max_length: int = 12000
):
    """
    Run naive LLM baseline on all papers

    Args:
        golden_set_path: Path to golden set Excel
        papers_dir: Directory with paper text files
        output_path: Where to save results
        config_path: LLM config file path
        max_length: Max chars to send to model
    """
    print("="*80)
    print("Naive LLM Baseline")
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

    # Load papers
    papers = load_papers_for_evaluation(golden_set_path, papers_dir)

    if not papers:
        print("❌ No papers found to evaluate!")
        return

    # Extract for each paper
    results = []

    for i, paper in enumerate(papers, 1):
        paper_id = paper['paper_id']
        print(f"\n[{i}/{len(papers)}] Processing {paper_id}: {paper['title'][:60]}...")

        start_time = time.time()

        extraction = extract_with_naive_llm(
            paper_text=paper['text'],
            llm_client=llm_client,
            max_length=max_length
        )

        extraction_time = time.time() - start_time

        result = {
            'paper_id': paper_id,
            'problem': extraction['problem'],
            'method': extraction['method'],  # 统一使用method
            'limitation': extraction['limitation'],
            'future_work': extraction['future_work'],
            'extraction_time': extraction_time,
            'metadata': {
                'paper_id': paper_id,
                'model': config.model,
                'max_length': max_length,
                'title': paper['title']
            }
        }

        if 'error' in extraction:
            result['metadata']['error'] = extraction['error']

        results.append(result)

        print(f"   ✅ Completed in {extraction_time:.1f}s")

        # Rate limiting
        time.sleep(1.0)

    # Save results
    output = {
        f"naive_llm_{config.model}": results
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {output_file}")
    print(f"✅ Processed {len(results)} papers")

    # Summary stats
    avg_time = sum(r['extraction_time'] for r in results) / len(results)
    print(f"\n📊 Average extraction time: {avg_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(description='Naive LLM baseline for information extraction')
    parser.add_argument('--golden_set', type=str, required=True,
                        help='Path to golden set Excel file')
    parser.add_argument('--papers_dir', type=str, required=True,
                        help='Directory containing paper text files')
    parser.add_argument('--output', type=str, default='../result/naive_baseline_results.json',
                        help='Output JSON file path')
    parser.add_argument('--config', type=str, default=None,
                        help='LLM config file path')
    parser.add_argument('--max_length', type=int, default=12000,
                        help='Max characters to send to model')

    args = parser.parse_args()

    run_naive_baseline(
        golden_set_path=args.golden_set,
        papers_dir=args.papers_dir,
        output_path=args.output,
        config_path=args.config,
        max_length=args.max_length
    )

    print("\n💡 This is a simple baseline - direct full-text extraction")
    print("   Compare with Multi-Agent methods to see the improvement")


if __name__ == '__main__':
    main()
