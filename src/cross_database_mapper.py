"""
跨库ID映射模块
将arXiv/Semantic Scholar论文映射到OpenAlex ID系统

核心功能：
1. arXiv ID -> OpenAlex ID映射
2. DOI -> OpenAlex ID映射
3. Title搜索映射
4. Concept过滤验证（确保论文属于目标领域）
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from openalex_client import OpenAlexClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAlex概念ID映射
CONCEPT_IDS = {
    "Computer Science": "C41008148",
    "Artificial Intelligence": "C154945302",
    "Machine Learning": "C119857082",
    "Natural Language Processing": "C204321447",
    "Computer Vision": "C73124549",
    "Neural Networks": "C50644808",
    "Deep Learning": "C78519656",
    "Information Retrieval": "C46978859",
}


class CrossDatabaseMapper:
    """
    跨数据库论文映射器
    将arXiv论文映射到OpenAlex，并进行概念过滤
    """

    def __init__(
        self,
        client: Optional[OpenAlexClient] = None,
        min_concept_score: float = 0.3,
        required_concepts: Optional[List[str]] = None
    ):
        """
        初始化映射器

        Args:
            client: OpenAlex客户端
            min_concept_score: 最小概念置信度分数（0-1）
            required_concepts: 必需的概念列表（概念名称，如 ["Computer Science", "Artificial Intelligence"]）
        """
        self.client = client or OpenAlexClient()
        self.min_concept_score = min_concept_score
        self.required_concepts = required_concepts or ["Computer Science"]

        # 统计信息
        self.stats = {
            'total_papers': 0,
            'mapped': 0,
            'failed': 0,
            'filtered_by_concept': 0,
            'mapping_methods': {
                'arxiv_id': 0,
                'doi': 0,
                'title_search': 0
            }
        }

        logger.info("跨库映射器初始化完成")
        logger.info(f"  min_concept_score={min_concept_score}")
        logger.info(f"  required_concepts={required_concepts}")

    def map_arxiv_to_openalex(
        self,
        arxiv_papers: List[Dict],
        verify_concepts: bool = True
    ) -> Tuple[List[Dict], Dict]:
        """
        将arXiv论文批量映射到OpenAlex

        Args:
            arxiv_papers: arXiv论文列表
            verify_concepts: 是否验证概念（过滤非目标领域论文）

        Returns:
            (映射成功的论文列表, 统计信息)
        """
        logger.info(f"开始映射 {len(arxiv_papers)} 篇arXiv论文到OpenAlex...")
        if not verify_concepts:
            logger.info("  ℹ️ 已禁用概念验证（arXiv阶段已完成领域过滤）")

        self.stats['total_papers'] = len(arxiv_papers)
        mapped_papers = []

        for i, arxiv_paper in enumerate(arxiv_papers, 1):
            logger.info(f"\n[{i}/{len(arxiv_papers)}] 映射: {arxiv_paper['title'][:50]}...")

            # 尝试映射
            openalex_paper = self._map_single_paper(arxiv_paper)

            if not openalex_paper:
                logger.warning(f"  ❌ 映射失败")
                self.stats['failed'] += 1
                continue

            # 验证概念（可选）
            if verify_concepts:
                is_valid, concept_info = self._verify_concepts(openalex_paper)
                if not is_valid:
                    logger.warning(
                        f"  ⚠️ 概念过滤: 不属于目标领域 "
                        f"(匹配概念: {concept_info})"
                    )
                    self.stats['filtered_by_concept'] += 1
                    continue

                logger.info(f"  ✓ 概念验证通过: {concept_info}")

            # 合并arXiv和OpenAlex数据
            merged_paper = self._merge_paper_data(arxiv_paper, openalex_paper)
            mapped_papers.append(merged_paper)

            self.stats['mapped'] += 1
            logger.info(
                f"  ✅ 映射成功: OpenAlex ID = {openalex_paper['id']}, "
                f"引用数 = {openalex_paper['cited_by_count']}"
            )

            # 避免API限流
            time.sleep(0.15)

        logger.info(f"\n✅ 映射完成:")
        logger.info(f"  总论文数: {self.stats['total_papers']}")
        logger.info(f"  映射成功: {self.stats['mapped']}")
        logger.info(f"  映射失败: {self.stats['failed']}")
        if verify_concepts:
            logger.info(f"  概念过滤: {self.stats['filtered_by_concept']}")

        return mapped_papers, self.stats

    def _map_single_paper(self, arxiv_paper: Dict) -> Optional[Dict]:
        """
        映射单篇论文到OpenAlex

        尝试顺序:
        1. arXiv ID查询
        2. DOI查询
        3. Title精确搜索

        Args:
            arxiv_paper: arXiv论文数据

        Returns:
            OpenAlex论文数据，失败返回None
        """
        # 方法1: 通过arXiv ID查询
        arxiv_id = arxiv_paper.get('arxiv_id')
        if arxiv_id:
            openalex_paper = self._query_by_arxiv_id(arxiv_id)
            if openalex_paper:
                self.stats['mapping_methods']['arxiv_id'] += 1
                logger.info(f"  → 方法: arXiv ID")
                return openalex_paper

        # 方法2: 通过DOI查询
        doi = arxiv_paper.get('doi')
        if doi:
            openalex_paper = self._query_by_doi(doi)
            if openalex_paper:
                self.stats['mapping_methods']['doi'] += 1
                logger.info(f"  → 方法: DOI")
                return openalex_paper

        # 方法3: 通过标题精确搜索
        title = arxiv_paper.get('title')
        if title:
            openalex_paper = self._query_by_title(title)
            if openalex_paper:
                self.stats['mapping_methods']['title_search'] += 1
                logger.info(f"  → 方法: Title搜索")
                return openalex_paper

        return None

    def _query_by_arxiv_id(self, arxiv_id: str) -> Optional[Dict]:
        """
        通过arXiv ID查询OpenAlex

        Args:
            arxiv_id: arXiv ID（如 "2301.12345" 或 "2301.12345v2"）

        Returns:
            OpenAlex论文数据
        """
        # 清理arXiv ID（去除版本号）
        clean_id = arxiv_id.split('v')[0]

        # OpenAlex支持通过external_ids.arxiv过滤
        # 尝试不同的arXiv ID格式
        arxiv_formats = [
            clean_id,                              # 2301.12345
            f"arXiv:{clean_id}",                   # arXiv:2301.12345
        ]

        for arxiv_format in arxiv_formats:
            params = {
                'filter': f'ids.arxiv:{arxiv_format}',
                'per-page': 1
            }

            try:
                data = self.client._make_request('works', params)
                results = data.get('results', [])

                if results:
                    logger.debug(f"arXiv ID查询成功，格式: {arxiv_format}")
                    return self.client._parse_paper(results[0])

            except Exception as e:
                logger.debug(f"arXiv ID查询失败（格式: {arxiv_format}）: {e}")
                continue

        return None

    def _query_by_doi(self, doi: str) -> Optional[Dict]:
        """
        通过DOI查询OpenAlex

        Args:
            doi: DOI字符串

        Returns:
            OpenAlex论文数据
        """
        # 清理DOI（移除前缀）
        clean_doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')

        params = {
            'filter': f'doi:{clean_doi}',
            'per-page': 1
        }

        try:
            data = self.client._make_request('works', params)
            results = data.get('results', [])

            if results:
                return self.client._parse_paper(results[0])

        except Exception as e:
            logger.debug(f"DOI查询失败: {e}")

        return None

    def _query_by_title(self, title: str) -> Optional[Dict]:
        """
        通过标题搜索查询OpenAlex

        Args:
            title: 论文标题

        Returns:
            OpenAlex论文数据
        """
        params = {
            'filter': f'title.search:{title}',
            'per-page': 5  # 取前5个，选择最匹配的
        }

        try:
            data = self.client._make_request('works', params)
            results = data.get('results', [])

            if not results:
                return None

            # 找到标题最匹配的论文
            best_match = None
            best_score = 0.0

            for result in results:
                result_title = result.get('title', '').lower()
                query_title = title.lower()

                # 简单的相似度计算（Jaccard相似度）
                similarity = self._compute_title_similarity(result_title, query_title)

                if similarity > best_score:
                    best_score = similarity
                    best_match = result

            # 只接受高相似度的匹配（阈值0.7）
            if best_match and best_score >= 0.7:
                return self.client._parse_paper(best_match)

        except Exception as e:
            logger.debug(f"Title搜索失败: {e}")

        return None

    def _compute_title_similarity(self, title1: str, title2: str) -> float:
        """
        计算标题相似度（Jaccard相似度）

        Args:
            title1: 标题1
            title2: 标题2

        Returns:
            相似度分数（0-1）
        """
        # 分词
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())

        # 移除常见停用词
        stopwords = {'a', 'an', 'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for'}
        words1 = words1 - stopwords
        words2 = words2 - stopwords

        # Jaccard相似度
        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _verify_concepts(self, openalex_paper: Dict) -> Tuple[bool, str]:
        """
        验证论文的概念标签（确保属于目标领域）

        Args:
            openalex_paper: OpenAlex论文数据

        Returns:
            (是否通过验证, 概念信息描述)
        """
        # 获取论文的概念标签（从原始API数据中提取）
        # 注意：_parse_paper可能没有包含concepts字段，需要重新获取
        paper_id = openalex_paper['id']

        try:
            # 确保paper_id格式正确（去除可能已存在的W前缀）
            if paper_id.startswith('W'):
                clean_id = paper_id
            else:
                clean_id = f'W{paper_id}'

            # 重新获取详细信息（包含concepts）
            data = self.client._make_request(f'works/{clean_id}')

            if not data:
                return False, "无法获取论文详情"

            concepts = data.get('concepts', [])

            if not concepts:
                return False, "无概念标签"

            # 检查必需的概念
            matched_concepts = []
            for concept in concepts:
                concept_name = concept.get('display_name', '')
                concept_score = concept.get('score', 0.0)

                # 检查是否匹配必需概念
                for required in self.required_concepts:
                    if required.lower() in concept_name.lower():
                        if concept_score >= self.min_concept_score:
                            matched_concepts.append(f"{concept_name}({concept_score:.2f})")

            # 至少匹配一个必需概念
            if matched_concepts:
                return True, ", ".join(matched_concepts)
            else:
                # 列出实际的概念
                actual_concepts = [
                    f"{c.get('display_name', 'Unknown')}({c.get('score', 0.0):.2f})"
                    for c in concepts[:3]
                ]
                return False, ", ".join(actual_concepts)

        except Exception as e:
            logger.debug(f"概念验证失败: {e}")
            return False, "验证失败"

    def _merge_paper_data(self, arxiv_paper: Dict, openalex_paper: Dict) -> Dict:
        """
        合并arXiv和OpenAlex数据

        Args:
            arxiv_paper: arXiv论文数据
            openalex_paper: OpenAlex论文数据

        Returns:
            合并后的论文数据
        """
        # 以OpenAlex数据为基础
        merged = openalex_paper.copy()

        # 补充arXiv特有字段
        merged['arxiv_id'] = arxiv_paper.get('arxiv_id')
        merged['arxiv_categories'] = arxiv_paper.get('categories', [])
        merged['arxiv_primary_category'] = arxiv_paper.get('primary_category')
        merged['arxiv_published_date'] = arxiv_paper.get('published_date')

        # 如果OpenAlex没有摘要，使用arXiv的
        if not merged.get('abstract') and arxiv_paper.get('abstract'):
            merged['abstract'] = arxiv_paper['abstract']

        # 如果OpenAlex没有PDF链接，使用arXiv的
        if not merged.get('pdf_url') and arxiv_paper.get('pdf_url'):
            merged['pdf_url'] = arxiv_paper['pdf_url']

        # 标记来源
        merged['source'] = 'arxiv+openalex'

        return merged

    def get_statistics(self) -> Dict:
        """
        获取映射统计信息

        Returns:
            统计信息字典
        """
        stats = self.stats.copy()
        if stats['total_papers'] > 0:
            stats['success_rate'] = stats['mapped'] / stats['total_papers']
            stats['filter_rate'] = stats['filtered_by_concept'] / stats['total_papers']
        else:
            stats['success_rate'] = 0.0
            stats['filter_rate'] = 0.0

        return stats


if __name__ == "__main__":
    # 测试代码
    from arxiv_seed_retriever import ArxivSeedRetriever

    # 1. 检索arXiv种子论文
    print("=" * 80)
    print("步骤1: 检索arXiv种子论文")
    print("=" * 80)

    arxiv_retriever = ArxivSeedRetriever(
        max_results_per_query=20,
        years_back=3,
        min_relevance_score=0.6
    )

    arxiv_papers = arxiv_retriever.retrieve_seed_papers(
        topic="Natural Language Processing",
        keywords=["transformer", "language model"],
        max_seeds=10
    )

    print(f"\n找到 {len(arxiv_papers)} 篇arXiv种子论文")

    # 2. 映射到OpenAlex
    print("\n" + "=" * 80)
    print("步骤2: 映射到OpenAlex")
    print("=" * 80)

    mapper = CrossDatabaseMapper(
        min_concept_score=0.3,
        required_concepts=["Computer Science", "Artificial Intelligence"]
    )

    mapped_papers, stats = mapper.map_arxiv_to_openalex(
        arxiv_papers,
        verify_concepts=True
    )

    # 3. 显示结果
    print("\n" + "=" * 80)
    print("映射结果")
    print("=" * 80)
    print(f"总论文数: {stats['total_papers']}")
    print(f"映射成功: {stats['mapped']} ({stats['success_rate']:.1%})")
    print(f"映射失败: {stats['failed']}")
    print(f"概念过滤: {stats['filtered_by_concept']} ({stats['filter_rate']:.1%})")
    print(f"\n映射方法统计:")
    for method, count in stats['mapping_methods'].items():
        print(f"  {method}: {count}")

    print(f"\n映射成功的论文样例（前3篇）:")
    for i, paper in enumerate(mapped_papers[:3], 1):
        print(f"\n[{i}] {paper['title']}")
        print(f"    arXiv ID: {paper.get('arxiv_id', 'N/A')}")
        print(f"    OpenAlex ID: {paper['id']}")
        print(f"    年份: {paper['year']}")
        print(f"    引用数: {paper['cited_by_count']}")
        print(f"    来源: {paper['source']}")
