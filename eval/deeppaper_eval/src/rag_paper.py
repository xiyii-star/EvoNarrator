#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure RAG Method for Deep Paper Information Extraction
纯RAG方法：使用向量检索提取论文深度信息

输入：论文PDF解析后的文本内容
输出：Problem / Contribution / Limitation / Future_work

方法：
1. 将论文文本分块并构建向量索引
2. 针对每个字段构建查询，检索最相关的文本块
3. 直接返回检索到的文本作为提取结果（无LLM生成）
"""

import json
import re
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np


class SimpleRAGExtractor:
    """纯RAG信息提取器（不使用LLM）"""

    def __init__(self, chunk_size: int = 500, top_k: int = 3):
        """
        初始化RAG提取器

        Args:
            chunk_size: 文本分块大小（按字符数）
            top_k: 每个查询检索的相关块数量
        """
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.embeddings = None
        self.chunks = []

    def _chunk_text(self, text: str) -> List[str]:
        """
        将文本分块

        Args:
            text: 论文全文

        Returns:
            文本块列表
        """
        # 按段落分割
        paragraphs = text.split('\n\n')

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果当前块加上新段落超过大小限制，保存当前块
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _simple_embedding(self, text: str) -> np.ndarray:
        """
        简单的嵌入方法：基于关键词的向量表示

        Args:
            text: 输入文本

        Returns:
            向量表示
        """
        # 定义关键词集合
        keywords = {
            # Problem相关
            'problem': ['problem', 'challenge', 'issue', 'difficulty', 'gap',
                       'limitation', 'drawback', 'weakness'],
            # Method相关 (原Contribution)
            'method': ['contribution', 'propose', 'present', 'introduce',
                           'develop', 'novel', 'innovation', 'approach', 'method'],
            # Limitation相关
            'limitation': ['limitation', 'constraint', 'shortcoming', 'drawback',
                         'weakness', 'not address', 'future work', 'improve'],
            # Future work相关
            'future': ['future', 'future work', 'extend', 'improve', 'enhancement',
                      'further', 'next step', 'open question', 'remain']
        }

        text_lower = text.lower()

        # 构建特征向量
        features = []
        for category, words in keywords.items():
            score = sum(text_lower.count(word) for word in words)
            features.append(score)

        # 添加位置特征（文本在整体中的位置）
        features.extend([len(text), text_lower.count('.')])

        return np.array(features, dtype=float)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _retrieve_relevant_chunks(self, query_embedding: np.ndarray) -> List[str]:
        """
        检索相关文本块

        Args:
            query_embedding: 查询向量

        Returns:
            最相关的文本块列表
        """
        if not self.chunks or self.embeddings is None:
            return []

        # 计算相似度
        similarities = []
        for i, chunk_emb in enumerate(self.embeddings):
            sim = self._cosine_similarity(query_embedding, chunk_emb)
            similarities.append((sim, i, self.chunks[i]))

        # 按相似度排序
        similarities.sort(reverse=True, key=lambda x: x[0])

        # 返回top-k
        return [chunk for _, _, chunk in similarities[:self.top_k]]

    def extract_deep_info(self, paper_text: str) -> Dict[str, str]:
        """
        提取论文深度信息

        Args:
            paper_text: 论文全文（PDF解析后的文本）

        Returns:
            包含四个字段的字典
        """
        # 1. 文本分块
        self.chunks = self._chunk_text(paper_text)

        if not self.chunks:
            return {
                'problem': '',
                'method': '',  # 统一使用method
                'limitation': '',
                'future_work': ''
            }

        # 2. 构建嵌入
        self.embeddings = [self._simple_embedding(chunk) for chunk in self.chunks]

        # 3. 为每个字段构建查询并检索
        results = {}

        # Problem查询
        problem_query = "What is the main problem, research question, or challenge that this paper addresses?"
        problem_emb = self._simple_embedding(problem_query)
        problem_chunks = self._retrieve_relevant_chunks(problem_emb)
        results['problem'] = self._synthesize_chunks(problem_chunks)

        # Method查询 (原Contribution)
        method_query = "What are the key methods, contributions, innovations, or proposed approaches in this paper?"
        method_emb = self._simple_embedding(method_query)
        method_chunks = self._retrieve_relevant_chunks(method_emb)
        results['method'] = self._synthesize_chunks(method_chunks)

        # Limitation查询
        limit_query = "What are the limitations, weaknesses, or constraints of this work?"
        limit_emb = self._simple_embedding(limit_query)
        limit_chunks = self._retrieve_relevant_chunks(limit_emb)
        results['limitation'] = self._synthesize_chunks(limit_chunks)

        # Future work查询
        future_query = "What future work, open problems, or research directions are suggested?"
        future_emb = self._simple_embedding(future_query)
        future_chunks = self._retrieve_relevant_chunks(future_emb)
        results['future_work'] = self._synthesize_chunks(future_chunks)

        return results

    def _synthesize_chunks(self, chunks: List[str]) -> str:
        """
        合成检索到的文本块

        Args:
            chunks: 文本块列表

        Returns:
            合成后的文本
        """
        if not chunks:
            return ""

        # 简单拼接，去重相似内容
        result = "\n\n".join(chunks)

        # 限制长度
        max_length = 500
        if len(result) > max_length:
            result = result[:max_length] + "..."

        return result


def extract_from_paper_text(paper_text: str,
                            chunk_size: int = 500,
                            top_k: int = 3) -> Dict[str, str]:
    """
    从论文文本提取深度信息（纯RAG方法）

    Args:
        paper_text: 论文PDF解析后的文本内容
        chunk_size: 文本分块大小
        top_k: 检索的相关块数量

    Returns:
        包含提取信息的字典：
        {
            'problem': str,
            'method': str,
            'limitation': str,
            'future_work': str
        }
    """
    extractor = SimpleRAGExtractor(chunk_size=chunk_size, top_k=top_k)
    return extractor.extract_deep_info(paper_text)


def batch_extract(papers: List[Dict[str, str]],
                  output_path: Optional[str] = None) -> List[Dict]:
    """
    批量处理多篇论文

    Args:
        papers: 论文列表，每个元素为 {'id': str, 'text': str}
        output_path: 输出JSON文件路径（可选）

    Returns:
        提取结果列表
    """
    results = []

    for i, paper in enumerate(papers, 1):
        paper_id = paper.get('id', f'paper_{i}')
        paper_text = paper.get('text', '')

        print(f"Processing [{i}/{len(papers)}]: {paper_id}")

        # 提取信息
        extracted = extract_from_paper_text(paper_text)

        results.append({
            'paper_id': paper_id,
            **extracted
        })

    # 保存结果
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({'pure_rag': results}, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {output_path}")

    return results


def load_papers_for_evaluation(golden_set_path: str, papers_dir: str) -> List[Dict]:
    """
    Load papers for evaluation

    Args:
        golden_set_path: Path to golden set Excel
        papers_dir: Directory containing paper text files

    Returns:
        List of paper dicts with id and text
    """
    import pandas as pd
    from column_utils import get_standard_columns

    # Load golden set
    df = pd.read_excel(golden_set_path)

    # Get standardized column names
    cols = get_standard_columns(df)
    problem_col = cols['problem_col']
    contribution_col = cols['contribution_col']
    paper_id_col = cols['paper_id_col']
    title_col = cols['title_col']

    annotated = df[
        df[problem_col].notna() & (df[problem_col] != '') &
        df[contribution_col].notna() & (df[contribution_col] != '')
    ]

    print(f"📚 Found {len(annotated)} annotated papers in golden set")

    papers = []
    papers_path = Path(papers_dir)

    for _, row in annotated.iterrows():
        paper_id = row[paper_id_col]
        title = row.get(title_col, '') if title_col else ''

        # Try to find paper text file
        text_file = papers_path / f"{paper_id}.txt"

        if not text_file.exists():
            print(f"   ⚠️  Text file not found for {paper_id}")
            continue

        # Read text
        with open(text_file, 'r', encoding='utf-8') as f:
            paper_text = f.read()

        papers.append({
            'id': paper_id,
            'title': title,
            'text': paper_text
        })

    print(f"✅ Loaded {len(papers)} papers")
    return papers


def run_pure_rag_method(
    golden_set_path: str,
    papers_dir: str,
    output_path: str
):
    """
    Run Pure RAG method on all papers

    Args:
        golden_set_path: Path to golden set Excel
        papers_dir: Directory with paper text files
        output_path: Where to save results
    """
    print("="*80)
    print("Pure RAG Method (Retrieval Only)")
    print("="*80)

    # Load papers
    papers = load_papers_for_evaluation(golden_set_path, papers_dir)

    if not papers:
        print("❌ No papers found to evaluate!")
        return

    # Extract for each paper
    results = []
    import time

    for i, paper in enumerate(papers, 1):
        paper_id = paper['id']
        print(f"\n[{i}/{len(papers)}] Processing {paper_id}: {paper['title'][:60]}...")

        start_time = time.time()

        try:
            extracted = extract_from_paper_text(paper['text'])

            result = {
                'paper_id': paper_id,
                'problem': extracted['problem'],
                'method': extracted['method'],  # 统一使用method
                'limitation': extracted['limitation'],
                'future_work': extracted['future_work'],
                'extraction_time': time.time() - start_time,
                'metadata': {
                    'paper_id': paper_id,
                    'title': paper['title'],
                    'method': 'pure_rag'
                }
            }

        except Exception as e:
            print(f"   ❌ Error during extraction: {e}")
            result = {
                'paper_id': paper_id,
                'problem': '',
                'method': '',  # 统一使用method
                'limitation': '',
                'future_work': '',
                'extraction_time': time.time() - start_time,
                'metadata': {
                    'paper_id': paper_id,
                    'title': paper['title'],
                    'method': 'pure_rag',
                    'error': str(e)
                }
            }

        results.append(result)
        print(f"   ✅ Completed in {result['extraction_time']:.1f}s")

    # Save results
    output = {
        "pure_rag": results
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
    import argparse

    parser = argparse.ArgumentParser(description='Pure RAG method for information extraction')
    parser.add_argument('--golden_set', type=str, required=True,
                        help='Path to golden set Excel file')
    parser.add_argument('--papers_dir', type=str, required=True,
                        help='Directory containing paper text files')
    parser.add_argument('--output', type=str, default='../result/pure_rag_results.json',
                        help='Output JSON file path')

    args = parser.parse_args()

    run_pure_rag_method(
        golden_set_path=args.golden_set,
        papers_dir=args.papers_dir,
        output_path=args.output
    )

    print("\n💡 This is pure RAG - retrieval only, no LLM generation")
    print("   Compare with LLM+RAG to see the benefit of generation")


if __name__ == '__main__':
    main()
