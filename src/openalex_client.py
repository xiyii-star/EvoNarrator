"""
简化的OpenAlex API客户端
用于论文检索和引用关系获取
"""

import requests
import time
from typing import List, Dict, Optional
import logging
from urllib.parse import urlencode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAlex概念ID常量
CONCEPT_COMPUTER_SCIENCE = "C41008148"  # 计算机科学
CONCEPT_ARTIFICIAL_INTELLIGENCE = "C154945302"  # 人工智能


class OpenAlexClient:
    """
    OpenAlex API客户端 - 简化版本
    """

    def __init__(
        self,
        email: Optional[str] = "1743623557@qq.com",
        base_url: str = "https://api.openalex.org",
        rate_limit_delay: float = 0.1
    ):
        self.base_url = base_url.rstrip('/')
        self.email = email
        self.rate_limit_delay = rate_limit_delay

        self.session = requests.Session()
        if email:
            self.session.params = {'mailto': email}

        logger.info(f"OpenAlex客户端初始化完成 (email: {email})")

    def _make_request(self, endpoint: str, params: Dict = None, silent_404: bool = False) -> Dict:
        """
        发送API请求

        Args:
            endpoint: API端点
            params: 请求参数
            silent_404: 是否静默404错误（默认False）

        Returns:
            响应JSON数据
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        params = params or {}

        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            # 404错误可以选择静默（很多新论文在OpenAlex中不存在是正常的）
            if e.response.status_code == 404 and silent_404:
                logger.debug(f"论文不存在(404): {endpoint}")
                return {}
            else:
                logger.error(f"请求失败: {e.response.status_code} {e.response.reason} for url: {e.response.url}")
                return {}
        except Exception as e:
            logger.error(f"请求失败: {e}")
            return {}

    def search_papers(
        self,
        topic: str,
        max_results: int = 10,
        sort_by: str = "cited_by_count",
        min_citations: int = 10,
        year_filter: Optional[str] = None,
        additional_filters: Optional[List[str]] = None,
        require_cs_ai: bool = False,
        concept_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        搜索论文（支持高级过滤）

        Args:
            topic: 搜索主题
            max_results: 最大结果数
            sort_by: 排序方式
            min_citations: 最小引用数
            year_filter: 年份过滤（如 "<2023", ">2022", "2020-2023"）
            additional_filters: 额外的过滤条件列表
            require_cs_ai: 是否限定CS/AI领域（默认False，不限制）
                          设置为True时会过滤CS OR AI,只要有其中一个标签即可
            concept_filter: 自定义概念过滤（如 "C41008148" 表示Computer Science）
                          指定后会覆盖默认的CS OR AI过滤
                          支持的概念：
                          - C41008148: Computer Science（计算机科学）
                          - C154945302: Artificial Intelligence（人工智能）

        Returns:
            论文列表
        """
        logger.info(f"搜索论文: '{topic}' (最多{max_results}篇)")

        # 构建过滤条件
        filters = [f'cited_by_count:>{min_citations}']

        # 强制限定CS/AI领域
        if require_cs_ai:
            if concept_filter:
                # 使用自定义概念过滤
                filters.append(f'concepts.id:{concept_filter}')
                logger.info(f"应用概念过滤: {concept_filter}")
            else:
                # 默认使用Computer Science OR Artificial Intelligence概念
                # 只要论文有CS或AI标签之一即可(使用|表示OR逻辑)
                filters.append('concepts.id:C41008148|C154945302')
                logger.info("应用默认过滤: Computer Science (C41008148) OR Artificial Intelligence (C154945302)")

        if year_filter:
            filters.append(f'publication_year:{year_filter}')

        if additional_filters:
            filters.extend(additional_filters)

        params = {
            'search': topic,
            'per-page': min(max_results, 25),
            'sort': f'{sort_by}:desc',
            'filter': ','.join(filters)
        }

        try:
            data = self._make_request('works', params)
            results = data.get('results', [])

            papers = []
            for result in results[:max_results]:
                paper = self._parse_paper(result)
                papers.append(paper)

            logger.info(f"找到 {len(papers)} 篇论文")
            return papers

        except Exception as e:
            logger.error(f"搜索论文失败: {e}")
            return []

    def get_citations(self, paper_id: str, max_results: int = 5) -> List[Dict]:
        """
        获取引用该论文的论文（正向滚雪球）

        Args:
            paper_id: 论文ID（OpenAlex Work ID）
            max_results: 最大结果数

        Returns:
            引用该论文的论文列表
        """
        logger.info(f"获取引用该论文的论文: {paper_id}")

        # 确保paper_id格式正确
        if not paper_id.startswith('W'):
            paper_id = f"W{paper_id}"

        # 使用cites过滤器：找到引用了该论文的论文
        params = {
            'filter': f'cites:{paper_id}',
            'per-page': max_results,
            'sort': 'cited_by_count:desc'
        }

        try:
            data = self._make_request('works', params)
            results = data.get('results', [])

            citations = []
            for result in results:
                citation = self._parse_paper(result)
                citations.append(citation)

            logger.info(f"  → 找到 {len(citations)} 篇引用该论文的论文")
            return citations

        except Exception as e:
            logger.error(f"获取引用论文失败: {e}")
            return []

    def get_references(self, paper_id: str, max_results: int = 5) -> List[Dict]:
        """
        获取该论文引用的参考文献（反向滚雪球）

        Args:
            paper_id: 论文ID（OpenAlex Work ID）
            max_results: 最大结果数

        Returns:
            该论文引用的参考文献列表
        """
        logger.info(f"获取该论文的参考文献: {paper_id}")

        # 确保paper_id格式正确
        if not paper_id.startswith('W'):
            paper_id = f"W{paper_id}"

        try:
            # 方法1：先获取论文详情，从referenced_works字段中提取ID列表
            paper_data = self._make_request(f'works/{paper_id}', silent_404=True)

            if not paper_data:
                logger.debug(f"  论文详情不存在或无法获取: {paper_id}")
                return []

            # 获取referenced_works ID列表
            referenced_work_ids = paper_data.get('referenced_works', [])

            if not referenced_work_ids:
                logger.debug(f"  该论文没有参考文献")
                return []

            # 截取前max_results个
            referenced_work_ids = referenced_work_ids[:max_results]

            logger.debug(f"  → 找到 {len(referenced_work_ids)} 个参考文献ID，正在获取详情...")

            # 批量获取这些论文的详细信息
            references = []
            for ref_id in referenced_work_ids:
                # 提取干净的ID（去掉URL前缀）
                clean_ref_id = ref_id.split('/')[-1] if '/' in ref_id else ref_id

                try:
                    ref_data = self._make_request(f'works/{clean_ref_id}', silent_404=True)
                    if ref_data:
                        ref_paper = self._parse_paper(ref_data)
                        references.append(ref_paper)
                except Exception as e:
                    logger.debug(f"    跳过参考文献 {clean_ref_id}: {e}")
                    continue

            logger.info(f"  → 成功获取 {len(references)} 篇参考文献详情")
            return references

        except Exception as e:
            logger.error(f"获取参考文献失败: {e}")
            return []

    def _parse_paper(self, raw_data: Dict) -> Dict:
        """解析论文数据为标准格式"""
        # 提取作者信息
        authors = []
        for authorship in raw_data.get('authorships', [])[:3]:  # 只取前3个作者
            author = authorship.get('author', {})
            authors.append(author.get('display_name', 'Unknown'))

        # 提取PDF链接
        pdf_url = None
        open_access = raw_data.get('open_access', {})
        if open_access.get('is_oa') and open_access.get('oa_url'):
            pdf_url = open_access.get('oa_url')

        # 标准化论文数据
        paper = {
            'id': raw_data.get('id', '').split('/')[-1] if raw_data.get('id') else '',
            'title': raw_data.get('title', 'Untitled'),
            'authors': authors,
            'year': raw_data.get('publication_year', 0),
            'cited_by_count': raw_data.get('cited_by_count', 0),
            'doi': raw_data.get('doi', ''),
            'pdf_url': pdf_url,
            'abstract': self._reconstruct_abstract(raw_data.get('abstract_inverted_index', {})),
            'venue': raw_data.get('host_venue', {}).get('display_name', ''),
            'is_open_access': open_access.get('is_oa', False)
        }

        return paper

    def get_work_with_concepts(self, work_id: str) -> Optional[Dict]:
        """
        获取单篇论文的详细信息（包含完整的概念标签）

        Args:
            work_id: 论文ID

        Returns:
            包含concepts字段的论文详细信息字典
        """
        if not work_id.startswith('W'):
            work_id = f"W{work_id}"

        try:
            data = self._make_request(f'works/{work_id}')
            if data:
                # 返回完整数据（包含concepts）
                paper = self._parse_paper(data)
                # 添加concepts字段
                paper['concepts'] = data.get('concepts', [])
                paper['primary_topic'] = data.get('primary_topic', {})
                return paper
            return None
        except Exception as e:
            logger.error(f"获取论文详情失败: {e}")
            return None

    def search_by_arxiv_id(self, arxiv_id: str) -> Optional[Dict]:
        """
        通过arXiv ID搜索论文

        Args:
            arxiv_id: arXiv ID（如 "2301.12345" 或 "2301.12345v2"）

        Returns:
            论文信息字典
        """
        # 清理arXiv ID（去除版本号）
        clean_id = arxiv_id.split('v')[0]

        params = {
            'filter': f'ids.arxiv:{clean_id}',
            'per-page': 1
        }

        try:
            data = self._make_request('works', params)
            results = data.get('results', [])

            if results:
                paper = self._parse_paper(results[0])
                paper['arxiv_id'] = arxiv_id
                return paper
            return None

        except Exception as e:
            logger.error(f"通过arXiv ID搜索失败: {e}")
            return None

    def search_by_doi(self, doi: str) -> Optional[Dict]:
        """
        通过DOI搜索论文

        Args:
            doi: DOI字符串

        Returns:
            论文信息字典
        """
        # 清理DOI（移除前缀）
        clean_doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')

        params = {
            'filter': f'doi:{clean_doi}',
            'per-page': 1
        }

        try:
            data = self._make_request('works', params)
            results = data.get('results', [])

            if results:
                return self._parse_paper(results[0])
            return None

        except Exception as e:
            logger.error(f"通过DOI搜索失败: {e}")
            return None

    def filter_by_concepts(
        self,
        papers: List[Dict],
        required_concepts: List[str],
        min_score: float = 0.3
    ) -> List[Dict]:
        """
        根据概念标签过滤论文

        Args:
            papers: 论文列表
            required_concepts: 必需的概念列表（概念名称）
            min_score: 最小置信度分数

        Returns:
            过滤后的论文列表
        """
        filtered = []

        for paper in papers:
            paper_id = paper['id']

            # 获取完整概念信息
            full_paper = self.get_work_with_concepts(paper_id)
            if not full_paper:
                continue

            concepts = full_paper.get('concepts', [])
            if not concepts:
                continue

            # 检查是否匹配必需概念
            matched = False
            for concept in concepts:
                concept_name = concept.get('display_name', '')
                concept_score = concept.get('score', 0.0)

                for required in required_concepts:
                    if required.lower() in concept_name.lower():
                        if concept_score >= min_score:
                            matched = True
                            break

                if matched:
                    break

            if matched:
                filtered.append(full_paper)

        logger.info(f"概念过滤: {len(papers)} -> {len(filtered)} 篇")
        return filtered

    def get_work_details(self, work_id: str) -> Optional[Dict]:
        """
        获取单篇论文的详细信息（兼容旧代码）

        Args:
            work_id: 论文ID

        Returns:
            论文详细信息字典
        """
        return self.get_work_with_concepts(work_id)

    def get_paper_by_id(self, paper_id: str) -> Optional[Dict]:
        """
        通过ID获取单篇论文（别名方法）

        Args:
            paper_id: 论文ID

        Returns:
            论文信息字典
        """
        return self.get_work_with_concepts(paper_id)

    def _reconstruct_abstract(self, inverted_index: Dict) -> str:
        """重构摘要文本"""
        if not inverted_index:
            return ""

        try:
            # 将倒排索引转换为文本
            words_with_pos = []
            for word, positions in inverted_index.items():
                for pos in positions:
                    words_with_pos.append((pos, word))

            # 按位置排序并连接
            words_with_pos.sort(key=lambda x: x[0])
            abstract = ' '.join([word for pos, word in words_with_pos])

            # 截断过长的摘要
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."

            return abstract
        except:
            return ""


if __name__ == "__main__":
    # 测试代码
    client = OpenAlexClient()

    print("=" * 60)
    print("示例1: 搜索 'Attention Mechanism'（默认CS OR AI过滤）")
    print("=" * 60)
    papers = client.search_papers("Attention Mechanism", max_results=3)

    for i, paper in enumerate(papers, 1):
        print(f"\n[{i}] {paper['title']}")
        print(f"    作者: {', '.join(paper['authors'])}")
        print(f"    年份: {paper['year']}")
        print(f"    引用数: {paper['cited_by_count']}")
        if paper['pdf_url']:
            print(f"    PDF: {paper['pdf_url']}")

    print("\n" + "=" * 60)
    print("示例2: 搜索 'Machine Learning'（只过滤AI领域）")
    print("=" * 60)
    # 只使用AI概念进行过滤
    papers = client.search_papers(
        "Machine Learning",
        max_results=3,
        concept_filter="C154945302"  # 只要Artificial Intelligence
    )

    for i, paper in enumerate(papers, 1):
        print(f"\n[{i}] {paper['title']}")
        print(f"    作者: {', '.join(paper['authors'])}")
        print(f"    年份: {paper['year']}")
        print(f"    引用数: {paper['cited_by_count']}")

    print("\n" + "=" * 60)
    print("示例3: 搜索 'Neural Network'（不限定领域）")
    print("=" * 60)
    # 关闭CS/AI过滤
    papers = client.search_papers(
        "Neural Network",
        max_results=3,
        require_cs_ai=False
    )

    for i, paper in enumerate(papers, 1):
        print(f"\n[{i}] {paper['title']}")
        print(f"    年份: {paper['year']}")
        print(f"    引用数: {paper['cited_by_count']}")