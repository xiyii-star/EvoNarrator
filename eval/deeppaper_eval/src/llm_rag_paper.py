#!/usr/bin/env python3
"""
LLM + RAG Method for Deep Information Extraction
使用传统RAG方法 (类似pipeline._analyze_with_traditional_rag)
处理papers_pdf目录下的PDF文件
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Add parent directory to path
kg_demo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(kg_demo_root))

from src.llm_rag_paper_analyzer import LLMRAGPaperAnalyzer


def load_pdfs_from_directory(pdf_dir: str) -> List[Dict]:
    """
    从指定目录加载所有PDF文件

    Args:
        pdf_dir: PDF文件目录

    Returns:
        PDF文件信息列表
    """
    pdf_path = Path(pdf_dir)

    if not pdf_path.exists():
        print(f"❌ 目录不存在: {pdf_dir}")
        return []

    # 获取所有PDF文件
    pdf_files = sorted(list(pdf_path.glob("*.pdf")))

    print(f"✅ 找到 {len(pdf_files)} 个PDF文件")

    papers = []
    for pdf_file in pdf_files:
        # 从文件名提取paper_id (格式: W1234567890_Title.pdf)
        paper_id = pdf_file.stem.split('_')[0] if '_' in pdf_file.stem else pdf_file.stem

        # 构建OpenAlex格式的paper字典
        paper = {
            'id': paper_id,
            'title': pdf_file.stem,  # 使用文件名作为标题
            'abstract': '',
            'pdf_path': str(pdf_file),
            'filename': pdf_file.name
        }

        papers.append(paper)

    return papers


def run_llm_rag_method(
    golden_set_path: str,
    papers_dir: str,
    output_path: str,
    config_path: str = None,
    grobid_url: str = None,
    limit: int = None
):
    """
    运行LLM + RAG方法分析PDF论文
    (类似pipeline._analyze_with_traditional_rag)

    Args:
        golden_set_path: 金标准Excel文件 (兼容参数，未使用)
        papers_dir: PDF文件目录
        output_path: 输出JSON文件路径
        config_path: LLM配置文件路径
        grobid_url: GROBID服务URL
        limit: 限制处理的论文数量
    """
    print("="*80)
    print("LLM + RAG Method (Traditional RAG)")
    print("="*80)

    # 加载PDF文件
    print(f"\n📂 处理目录: {papers_dir}")
    papers = load_pdfs_from_directory(papers_dir)

    if not papers:
        print("❌ 未找到PDF文件!")
        return

    # 限制数量
    if limit:
        papers = papers[:limit]
        print(f"⚠️  限制处理前 {limit} 篇论文 (测试模式)")

    # 初始化分析器
    if not config_path:
        config_path = str(Path(__file__).parent.parent.parent.parent / "config" / "config.yaml")

    prompts_path = str(Path(__file__).parent.parent.parent.parent / "prompts")
    local_model_path = str(Path(__file__).parent.parent.parent.parent / "model" / "sentence-transformers" / "all-MiniLM-L6-v2")

    # 打印配置信息
    print(f"\n📋 配置信息:")
    print(f"   LLM配置文件: {config_path}")
    print(f"   本地模型路径: {local_model_path}")
    print(f"   GROBID服务: {grobid_url if grobid_url else '未配置 (将使用PyPDF2)'}")
    print()

    try:
        analyzer = LLMRAGPaperAnalyzer(
            llm_config_path=config_path,
            embedding_model="all-MiniLM-L6-v2",
            use_modelscope=True,
            max_context_length=3000,
            prompts_dir=prompts_path,
            local_model_path=local_model_path,
            grobid_url=grobid_url
        )
        print(f"\n✅ LLM RAG分析器初始化成功")
        print(f"   Embedding模型: {'已启用 (本地模型)' if analyzer.use_embeddings else '未启用 (关键词检索)'}")
        if grobid_url:
            print(f"   GROBID解析: 已启用")
        else:
            print(f"   PDF解析: PyPDF2 (降级模式)")
    except Exception as e:
        print(f"❌ 分析器初始化失败: {e}")
        print(f"   配置文件: {config_path}")
        return

    print(f"\n{'='*80}")
    print(f"开始处理 {len(papers)} 篇论文")
    print(f"{'='*80}\n")

    # 批量分析 (类似pipeline中的batch_analyze_papers)
    results = []
    success_count = 0
    with_pdf_count = 0

    for i, paper in enumerate(papers, 1):
        paper_id = paper['id']
        pdf_path = paper['pdf_path']

        print(f"\n[{i}/{len(papers)}] {paper_id}")
        print(f"   文件: {paper['filename']}")

        start_time = time.time()

        try:
            # 使用LLMRAGPaperAnalyzer分析 (传入PDF路径)
            enriched_paper = analyzer.analyze_paper(paper, pdf_path=pdf_path)

            extraction_time = time.time() - start_time

            # 提取结果
            rag_analysis = enriched_paper.get('rag_analysis', {})

            result = {
                'paper_id': paper_id,
                'problem': rag_analysis.get('problem', ''),
                'method': rag_analysis.get('method', rag_analysis.get('contribution', '')),  # 兼容旧字段
                'limitation': rag_analysis.get('limitation', ''),
                'future_work': rag_analysis.get('future_work', ''),
                'extraction_time': extraction_time,
                'metadata': {
                    'paper_id': paper_id,
                    'filename': paper['filename'],
                    'method': 'llm_rag',
                    'sections_extracted': enriched_paper.get('sections_extracted', 0),
                    'analysis_method': enriched_paper.get('analysis_method', 'unknown'),
                    'grobid_enabled': grobid_url is not None
                }
            }

            # 检查是否有错误
            if 'error' not in rag_analysis:
                success_count += 1

            results.append(result)

            print(f"   ✅ 完成 (耗时 {extraction_time:.1f}s)")
            print(f"   📊 提取章节数: {enriched_paper.get('sections_extracted', 0)}")

            # 统计有PDF章节的论文
            if enriched_paper.get('sections_extracted', 0) > 0:
                with_pdf_count += 1

        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()

            result = {
                'paper_id': paper_id,
                'problem': '',
                'method': '',  # 统一使用method
                'limitation': '',
                'future_work': '',
                'extraction_time': time.time() - start_time,
                'metadata': {
                    'paper_id': paper_id,
                    'filename': paper['filename'],
                    'method': 'llm_rag',
                    'error': str(e)
                }
            }
            results.append(result)

        # Rate limiting
        time.sleep(0.5)

    # 保存结果
    output = {
        "llm_rag": results,
        "summary": {
            "total_papers": len(papers),
            "successful": success_count,
            "failed": len(papers) - success_count,
            "with_pdf_sections": with_pdf_count,
            "success_rate": success_count / len(papers) if papers else 0
        }
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"✅ 结果已保存到: {output_file}")
    print(f"{'='*80}")
    print(f"📊 统计:")
    print(f"   总论文数: {len(papers)}")
    print(f"   成功: {success_count}")
    print(f"   失败: {len(papers) - success_count}")
    print(f"   有PDF章节: {with_pdf_count}")
    print(f"   成功率: {success_count / len(papers) * 100:.1f}%")

    # 平均耗时
    avg_time = sum(r['extraction_time'] for r in results) / len(results) if results else 0
    print(f"   平均耗时: {avg_time:.1f}s/篇")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='LLM + RAG方法: 从PDF提取深度信息',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 通过run_all_experiments.py调用
    python llm_rag_paper.py --golden_set data/golden_set.xlsx --papers_dir data/papers_pdf --output result/llm_rag.json

    # 使用GROBID
    python llm_rag_paper.py --golden_set '' --papers_dir ../data/papers_pdf --grobid_url http://localhost:8070

    # 测试模式
    python llm_rag_paper.py --golden_set '' --papers_dir ../data/papers_pdf --limit 3
        """
    )

    parser.add_argument(
        '--golden_set',
        type=str,
        required=True,
        help='金标准Excel文件路径 (兼容参数，实际未使用)'
    )
    parser.add_argument(
        '--papers_dir',
        type=str,
        required=True,
        help='PDF文件目录路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../result/llm_rag_results.json',
        help='输出JSON文件路径'
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
        help='GROBID服务URL (例如: http://localhost:8070)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='限制处理的论文数量 (用于测试)'
    )

    args = parser.parse_args()

    run_llm_rag_method(
        golden_set_path=args.golden_set,
        papers_dir=args.papers_dir,
        output_path=args.output,
        config_path=args.config,
        grobid_url=args.grobid_url,
        limit=args.limit
    )

    print("💡 这是传统LLM + RAG方法")
    print("   输出四个维度: Problem, Contribution, Limitation, Future Work")


if __name__ == '__main__':
    main()
