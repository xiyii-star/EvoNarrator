#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
My Method: DeepPaper Multi-Agent Architecture
Navigator → Extractor → Critic → Synthesizer

Modified to process PDFs directly from papers_pdf/ directory
using GROBID parser (similar to pipeline._phase4_paper_rag_analysis)
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Add parent directory to path to import DeepPaper agents
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from DeepPaper_Agent.orchestrator import DeepPaperOrchestrator
from DeepPaper_Agent.data_structures import PaperDocument, PaperSection
from src.llm_config import LLMClient, LLMConfig
from src.grobid_parser import GrobidPDFParser


def extract_paper_from_pdf(
    pdf_path: str,
    grobid_url: Optional[str] = None
) -> PaperDocument:
    """
    从PDF提取论文信息 (类似pipeline中的_convert_to_paper_document)

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
            parser = GrobidPDFParser(grobid_url)
            sections = parser.extract_sections_from_pdf(pdf_path)

            # 提取标题和摘要
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
        print(f"         使用PyPDF2降级解析...")
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


def extract_with_deep_paper(
    paper_doc: PaperDocument,
    llm_client: LLMClient,
    max_retries: int = 2
) -> Dict[str, str]:
    """
    使用DeepPaper Multi-Agent系统提取深度信息
    (类似pipeline._analyze_with_deep_paper)

    Args:
        paper_doc: 论文文档
        llm_client: LLM客户端
        max_retries: Critic最大重试次数

    Returns:
        包含四个维度的字典
    """
    print(f"      > DeepPaper Multi-Agent分析...")
    print(f"         章节数: {len(paper_doc.sections)}")

    # 初始化协调器
    orchestrator = DeepPaperOrchestrator(
        llm_client=llm_client,
        max_retries=max_retries
    )

    # 运行Multi-Agent分析
    try:
        report = orchestrator.analyze_paper(paper_doc, output_dir=None)

        return {
            'problem': report.problem,
            'method': report.method,  # FinalReport使用method字段
            'limitation': report.limitation,
            'future_work': report.future_work,
            'metadata': {
                'extraction_quality': report.extraction_quality,
                'iteration_count': report.iteration_count,
                'sections_count': len(paper_doc.sections)
            }
        }

    except Exception as e:
        print(f"      ❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'problem': '',
            'method': '',  # 统一使用method
            'limitation': '',
            'future_work': '',
            'error': str(e),
            'metadata': {
                'sections_count': len(paper_doc.sections)
            }
        }


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

        papers.append({
            'paper_id': paper_id,
            'pdf_path': str(pdf_file),
            'filename': pdf_file.name
        })

    return papers


def run_mymethod(
    golden_set_path: str,
    papers_dir: str,
    output_path: str,
    config_path: str = None,
    grobid_url: str = None,
    max_retries: int = 2,
    limit: int = None
):
    """
    运行DeepPaper Multi-Agent方法分析PDF论文
    (类似pipeline._phase4_paper_rag_analysis的逻辑)

    Args:
        golden_set_path: 金标准Excel文件 (可选，这里忽略)
        papers_dir: PDF文件目录
        output_path: 输出JSON文件路径
        config_path: LLM配置文件路径
        grobid_url: GROBID服务URL
        max_retries: Critic最大重试次数
        limit: 限制处理的论文数量 (用于测试)
    """
    print("="*80)
    print("My Method: DeepPaper Multi-Agent Architecture")
    print("="*80)

    # 加载LLM客户端
    if not config_path:
        config_path = str(Path(__file__).parent.parent.parent.parent / "config" / "config.yaml")

    try:
        config = LLMConfig.from_file(config_path)
        llm_client = LLMClient(config)
        print(f"✅ LLM客户端加载成功: {config.provider} - {config.model}")
    except Exception as e:
        print(f"❌ LLM客户端加载失败: {e}")
        print(f"   配置文件: {config_path}")
        return

    # 检查GROBID
    if grobid_url:
        print(f"✅ 使用GROBID服务: {grobid_url}")
    else:
        print(f"⚠️  未配置GROBID, 将使用PyPDF2降级解析")

    # 加载PDF文件
    print(f"\n📂 处理目录: {papers_dir}")
    papers = load_pdfs_from_directory(papers_dir)

    if not papers:
        print("❌ 未找到PDF文件!")
        return

    # 限制数量 (用于测试)
    if limit:
        papers = papers[:limit]
        print(f"⚠️  限制处理前 {limit} 篇论文 (测试模式)")

    print(f"\n{'='*80}")
    print(f"开始处理 {len(papers)} 篇论文")
    print(f"{'='*80}\n")

    # 提取每篇论文 (类似pipeline中的批量分析)
    results = []
    success_count = 0
    pdf_count = 0

    for i, paper in enumerate(papers, 1):
        paper_id = paper['paper_id']
        pdf_path = paper['pdf_path']

        print(f"\n[{i}/{len(papers)}] {paper_id}")
        print(f"   文件: {paper['filename']}")

        start_time = time.time()

        try:
            # 步骤1: 从PDF提取论文文档 (使用GROBID)
            paper_doc = extract_paper_from_pdf(pdf_path, grobid_url)

            # 步骤2: 使用DeepPaper Multi-Agent分析
            extraction = extract_with_deep_paper(
                paper_doc=paper_doc,
                llm_client=llm_client,
                max_retries=max_retries
            )

            extraction_time = time.time() - start_time

            # 构建结果
            result = {
                'paper_id': paper_id,
                'problem': extraction['problem'],
                'method': extraction['method'],  # 统一使用method
                'limitation': extraction['limitation'],
                'future_work': extraction['future_work'],
                'extraction_time': extraction_time,
                'metadata': {
                    'paper_id': paper_id,
                    'filename': paper['filename'],
                    'method': 'mymethod_deeppaper',
                    'max_retries': max_retries,
                    'grobid_enabled': grobid_url is not None,
                    **extraction.get('metadata', {})
                }
            }

            if 'error' in extraction:
                result['metadata']['error'] = extraction['error']
            else:
                success_count += 1

            results.append(result)

            print(f"   ✅ 完成 (耗时 {extraction_time:.1f}s)")

            # 显示提取质量
            if 'extraction_quality' in extraction.get('metadata', {}):
                quality = extraction['metadata']['extraction_quality']
                avg_quality = sum(quality.values()) / len(quality) if quality else 0
                print(f"   📊 平均质量: {avg_quality:.2f}")

            # 统计有PDF章节的论文
            if extraction.get('metadata', {}).get('sections_count', 0) > 1:
                pdf_count += 1

        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()

            # 添加失败记录
            results.append({
                'paper_id': paper_id,
                'problem': '',
                'method': '',  # 统一使用method
                'limitation': '',
                'future_work': '',
                'extraction_time': time.time() - start_time,
                'metadata': {
                    'paper_id': paper_id,
                    'filename': paper['filename'],
                    'method': 'mymethod_deeppaper',
                    'error': str(e)
                }
            })

        # Rate limiting
        time.sleep(0.5)

    # 保存结果
    output = {
        "mymethod_deeppaper": results,
        "summary": {
            "total_papers": len(papers),
            "successful": success_count,
            "failed": len(papers) - success_count,
            "with_pdf_sections": pdf_count,
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
    print(f"   有PDF章节: {pdf_count}")
    print(f"   成功率: {success_count / len(papers) * 100:.1f}%")

    # 平均耗时
    avg_time = sum(r['extraction_time'] for r in results) / len(results) if results else 0
    print(f"   平均耗时: {avg_time:.1f}s/篇")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='DeepPaper Multi-Agent方法: 从PDF提取深度信息',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 通过run_all_experiments.py调用
    python mymethod.py --golden_set data/golden_set.xlsx --papers_dir data/papers_pdf --output result/mymethod.json

    # 直接调用
    python mymethod.py --golden_set '' --papers_dir ../data/papers_pdf --output ../result/mymethod_results.json

    # 使用GROBID
    python mymethod.py --golden_set '' --papers_dir ../data/papers_pdf --grobid_url http://localhost:8070

    # 测试模式 (处理前3篇)
    python mymethod.py --golden_set '' --papers_dir ../data/papers_pdf --limit 3
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
        default='../result/mymethod_results.json',
        help='输出JSON文件路径 (默认: ../result/mymethod_results.json)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='LLM配置文件路径 (默认: 自动查找config.yaml)'
    )
    parser.add_argument(
        '--grobid_url',
        type=str,
        default=None,
        help='GROBID服务URL (例如: http://localhost:8070)'
    )
    parser.add_argument(
        '--max_retries',
        type=int,
        default=2,
        help='Critic最大重试次数 (默认: 2)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='限制处理的论文数量 (用于测试, 默认: 处理全部)'
    )

    args = parser.parse_args()

    run_mymethod(
        golden_set_path=args.golden_set,
        papers_dir=args.papers_dir,
        output_path=args.output,
        config_path=args.config,
        grobid_url=args.grobid_url,
        max_retries=args.max_retries,
        limit=args.limit
    )

    print("💡 这是我们提出的完整Multi-Agent架构方法")
    print("   输出四个维度: Problem, Method, Limitation, Future Work")


if __name__ == '__main__':
    main()
