"""
高级论文检索模块 - 基于滚雪球方法的多阶段检索策略

实现优化的六步检索流程（基于arXiv种子 + OpenAlex扩展）：
1. 高质量种子获取 (High-Quality Seed Retrieval) - 使用arXiv API + Categories过滤
2. 跨库ID映射 (ID Mapping) - arXiv -> OpenAlex，严格Concept验证
3. 正向滚雪球 (Forward Snowballing) - Seed -> 谁引用了Seed? -> 得到子节点
4. 反向滚雪球 (Backward Snowballing) - 谁被Seed引用了? <- Seed -> 得到父节点/祖先
5. 横向补充/共引挖掘 (Co-citation Mining) - 在子节点和父节点中，谁被大家反复提及但还不在库里?
6. 补充SOTA (Add Recent Frontiers) - arXiv最新论文（6-12个月）+ 相似度过滤
7. 构建闭包 (Closure Construction) - 建立连接
"""

import logging
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import yaml
from openalex_client import OpenAlexClient
from arxiv_seed_retriever import ArxivSeedRetriever
from cross_database_mapper import CrossDatabaseMapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SnowballRetrieval:
    """
    滚雪球论文检索系统
    基于引用关系的多阶段论文发现和关系构建
    """

    def __init__(
        self,
        client: Optional[OpenAlexClient] = None,
        seed_count: Optional[int] = None,
        citations_per_seed: Optional[int] = None,
        recent_count: Optional[int] = None,
        seed_keywords: Optional[List[str]] = None,
        enable_second_round: Optional[bool] = None,
        second_round_limit: Optional[int] = None,
        use_arxiv_seeds: Optional[bool] = None,
        arxiv_years_back: Optional[int] = None,
        llm_client = None,
        config_path: str = './config/config.yaml'
    ):
        """
        初始化滚雪球检索系统

        优先级：传入参数 > config.yaml配置 > 默认值

        Args:
            client: OpenAlex API客户端
            seed_count: 基石种子论文数量（默认: 5）
            citations_per_seed: 每个种子论文选取的引用论文数量（默认: 8）
            recent_count: 最新论文数量（默认: 10）
            seed_keywords: 种子关键词列表，用于相关性过滤（默认: []）
            enable_second_round: 是否启用第二轮滚雪球（默认: True）
            second_round_limit: 第二轮每篇论文的扩展数量限制（默认: 3）
            use_arxiv_seeds: 是否使用arXiv种子检索（默认: True）
            arxiv_years_back: arXiv种子回溯年数（默认: 5）
            llm_client: LLM客户端（用于智能查询生成）
            config_path: 配置文件路径（默认: './config/config.yaml'）
        """
        # 加载配置文件
        snowball_config = self._load_config(config_path)

        # 初始化客户端
        self.client = client or OpenAlexClient()
        self.llm_client = llm_client

        # 参数优先级：传入参数 > config.yaml > 默认值
        self.seed_count = seed_count if seed_count is not None else snowball_config.get('seed_count', 5)
        self.citations_per_seed = citations_per_seed if citations_per_seed is not None else snowball_config.get('citations_per_seed', 8)
        self.recent_count = recent_count if recent_count is not None else snowball_config.get('recent_count', 10)
        self.seed_keywords = seed_keywords if seed_keywords is not None else snowball_config.get('seed_keywords', [])
        self.enable_second_round = enable_second_round if enable_second_round is not None else snowball_config.get('enable_second_round', True)
        self.second_round_limit = second_round_limit if second_round_limit is not None else snowball_config.get('second_round_limit', 3)

        # 新增：arXiv种子检索参数
        self.use_arxiv_seeds = use_arxiv_seeds if use_arxiv_seeds is not None else snowball_config.get('use_arxiv_seeds', True)
        self.arxiv_years_back = arxiv_years_back if arxiv_years_back is not None else snowball_config.get('arxiv_years_back', 5)

        # 初始化arXiv检索器和跨库映射器（如果启用）
        if self.use_arxiv_seeds:
            self.arxiv_retriever = ArxivSeedRetriever(
                max_results_per_query=self.seed_count * 2,  # 多取一些，映射后可能会减少
                years_back=self.arxiv_years_back,
                min_relevance_score=0.6,
                llm_client=self.llm_client,  # 传递LLM客户端
                use_llm_query_generation=True  # 启用LLM查询生成
            )
            self.cross_mapper = CrossDatabaseMapper(
                client=self.client,
                min_concept_score=0.3,
                required_concepts=["Computer Science"]
            )
        else:
            self.arxiv_retriever = None
            self.cross_mapper = None

        # 存储检索到的论文
        self.seed_papers: List[Dict] = []
        self.citing_papers: List[Dict] = []  # 子节点：引用种子的论文
        self.ancestor_papers: List[Dict] = []  # 父节点：被种子引用的论文
        self.cocitation_papers: List[Dict] = []  # 共引论文：被反复提及的论文
        self.recent_papers: List[Dict] = []
        self.all_papers: Dict[str, Dict] = {}  # paper_id -> paper_data

        # 存储引用关系
        self.citation_edges: Set[Tuple[str, str]] = set()  # (citing_id, cited_id)

        # 第二轮统计（用于报告生成）
        self.first_round_citing_count: int = 0
        self.first_round_ancestor_count: int = 0
        self.first_round_cocitation_count: int = 0
        self.second_round_citing_count: int = 0
        self.second_round_ancestor_count: int = 0
        self.second_round_cocitation_count: int = 0

        logger.info("滚雪球检索系统初始化完成")
        logger.info(f"  配置来源: {config_path if Path(config_path).exists() else '默认值'}")
        logger.info(f"  种子检索模式: {'arXiv优先' if self.use_arxiv_seeds else 'OpenAlex直接搜索'}")
        logger.info(f"  seed_count={self.seed_count}, citations_per_seed={self.citations_per_seed}")
        logger.info(f"  recent_count={self.recent_count}, enable_second_round={self.enable_second_round}")
        if self.enable_second_round:
            logger.info(f"  second_round_limit={self.second_round_limit}")
        if self.use_arxiv_seeds:
            logger.info(f"  arxiv_years_back={self.arxiv_years_back}")

    def _load_config(self, config_path: str) -> Dict:
        """
        从配置文件加载滚雪球检索配置

        Args:
            config_path: 配置文件路径

        Returns:
            滚雪球配置字典（snowball部分）
        """
        config_file = Path(config_path)

        if not config_file.exists():
            logger.debug(f"配置文件不存在: {config_path}，将使用默认配置")
            return {}

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                full_config = yaml.safe_load(f)

            snowball_config = full_config.get('snowball', {}) if full_config else {}
            logger.debug(f"成功加载滚雪球配置: {config_path}")
            return snowball_config

        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}，将使用默认配置")
            return {}

    def execute_full_pipeline(
        self,
        topic: str,
        content_keyword: str,
        seed_year_threshold: int = 2023
    ) -> Dict:
        """
        执行完整的六步检索流程

        Args:
            topic: 主题关键词
            content_keyword: 内容关键词
            seed_year_threshold: 种子论文的年份阈值（小于此年份）

        Returns:
            包含所有论文和引用关系的字典
        """
        logger.info(f"开始执行完整检索流程: topic='{topic}', content='{content_keyword}'")

        # 第一步：基石种子
        logger.info("\n" + "=" * 60)
        logger.info("第一步：基石种子 (Seed Papers)")
        logger.info("=" * 60)
        self.seed_papers = self._select_seed_papers(
            topic=topic,
            content_keyword=content_keyword,
            year_threshold=seed_year_threshold
        )

        # 第二步：正向滚雪球 - 找子节点
        logger.info("\n" + "=" * 60)
        logger.info("第二步：正向滚雪球 (Forward Snowballing)")
        logger.info("Seed -> 谁引用了Seed? -> 得到子节点")
        logger.info("=" * 60)
        self.citing_papers = self._forward_snowballing(self.seed_papers)

        # 第三步：反向滚雪球 - 找父节点/祖先
        logger.info("\n" + "=" * 60)
        logger.info("第三步：反向滚雪球 (Backward Snowballing)")
        logger.info("谁被Seed引用了? <- Seed -> 得到父节点/祖先")
        logger.info("=" * 60)
        self.ancestor_papers = self._backward_snowballing(self.seed_papers)

        # 第四步：横向补充/共引挖掘
        logger.info("\n" + "=" * 60)
        logger.info("第四步：横向补充/共引挖掘 (Co-citation Mining)")
        logger.info("在子节点和父节点中，谁被大家反复提及但还不在库里?")
        logger.info("=" * 60)
        self.cocitation_papers = self._cocitation_mining(
            self.citing_papers,
            self.ancestor_papers
        )

        # 记录第一轮统计
        self.first_round_citing_count = len(self.citing_papers)
        self.first_round_ancestor_count = len(self.ancestor_papers)
        self.first_round_cocitation_count = len(self.cocitation_papers)

        # 第二轮滚雪球（如果启用）
        if self.enable_second_round:
            logger.info("\n" + "=" * 80)
            logger.info("🔄 开始第二轮滚雪球扩展")
            logger.info("=" * 80)
            self._execute_second_round_snowballing()

        # 第五步：补充最新SOTA
        logger.info("\n" + "=" * 60)
        logger.info("第五步：补充最新SOTA (Recent Frontiers)")
        logger.info("=" * 60)
        self.recent_papers = self._add_recent_frontiers(
            topic=topic,
            content_keyword=content_keyword,
            year_threshold=seed_year_threshold
        )

        # 第六步：构建闭包
        logger.info("\n" + "=" * 60)
        logger.info("第六步：构建引用闭包 (Closure Construction)")
        logger.info("=" * 60)
        self._build_closure()

        # 生成统计报告
        result = self._generate_report()
        logger.info("\n检索流程完成！")
        return result

    def _select_seed_papers(
        self,
        topic: str,
        content_keyword: str,
        year_threshold: int
    ) -> List[Dict]:
        """
        第一步：选定基石种子论文（优化版）

        策略：
        - 如果启用arXiv：使用arXiv Categories + 关键词 -> 映射到OpenAlex -> Concept验证
        - 如果不启用：直接在OpenAlex搜索（传统模式）

        Args:
            topic: 主题关键词
            content_keyword: 内容关键词
            year_threshold: 年份阈值（arXiv模式下会被覆盖）

        Returns:
            种子论文列表
        """
        if self.use_arxiv_seeds:
            logger.info("🎯 使用arXiv优先种子检索策略")
            return self._select_seeds_from_arxiv(topic, content_keyword)
        else:
            logger.info("🔍 使用OpenAlex直接搜索策略")
            return self._select_seeds_from_openalex(topic, content_keyword, year_threshold)

    def _select_seeds_from_arxiv(
        self,
        topic: str,
        content_keyword: str
    ) -> List[Dict]:
        """
        从arXiv检索种子论文并映射到OpenAlex

        流程：
        1. arXiv检索（使用Categories + 关键词）
        2. 跨库映射（arXiv -> OpenAlex）
        3. Concept过滤（确保CS/AI领域）

        Args:
            topic: 主题
            content_keyword: 内容关键词

        Returns:
            映射后的种子论文列表（OpenAlex格式）
        """
        logger.info("步骤1a: 从arXiv检索高质量种子")
        logger.info(f"  主题: '{topic}', 关键词: '{content_keyword}'")
        logger.info(f"  回溯年数: {self.arxiv_years_back}年")

        # 1. 检索arXiv论文
        keywords = [content_keyword] + self.seed_keywords if self.seed_keywords else [content_keyword]
        arxiv_papers = self.arxiv_retriever.retrieve_seed_papers(
            topic=topic,
            keywords=keywords,
            max_seeds=self.seed_count * 2  # 多取一些，映射后会减少
        )

        logger.info(f"  ✓ arXiv检索到 {len(arxiv_papers)} 篇候选论文")

        # 2. 映射到OpenAlex
        logger.info("\n步骤1b: 跨库映射（arXiv -> OpenAlex）")
        mapped_papers, stats = self.cross_mapper.map_arxiv_to_openalex(
            arxiv_papers,
            verify_concepts=False  # arXiv阶段已通过类别过滤，无需再验证概念
        )

        logger.info(f"  ✓ 成功映射 {len(mapped_papers)} 篇论文")
        if stats.get('filtered_by_concept', 0) > 0:
            logger.info(f"  ℹ️ 概念过滤: {stats.get('filtered_by_concept', 0)} 篇（已禁用）")

        # 3. 存储到all_papers
        seeds = []
        for paper in mapped_papers[:self.seed_count]:
            self.all_papers[paper['id']] = paper
            seeds.append(paper)
            logger.info(
                f"  ✓ [{paper['year']}] {paper['title'][:60]}... "
                f"(引用数: {paper['cited_by_count']}, arXiv: {paper.get('arxiv_id', 'N/A')})"
            )

        logger.info(f"\n✅ 共选定 {len(seeds)} 篇高质量种子论文（arXiv验证）")
        return seeds

    def _select_seeds_from_openalex(
        self,
        topic: str,
        content_keyword: str,
        year_threshold: int
    ) -> List[Dict]:
        """
        从OpenAlex直接检索种子论文（传统模式）

        Args:
            topic: 主题关键词
            content_keyword: 内容关键词
            year_threshold: 年份阈值

        Returns:
            种子论文列表
        """
        query = f"{topic} {content_keyword}"
        logger.info(f"搜索查询: '{query}'")
        logger.info(f"筛选条件: publication_year < {year_threshold}, sorted by citations")

        params = {
            'search': query,
            'per-page': self.seed_count,
            'sort': 'cited_by_count:desc',
            'filter': f'publication_year:<{year_threshold},cited_by_count:>50'
        }

        try:
            data = self.client._make_request('works', params)
            results = data.get('results', [])

            seeds = []
            for result in results[:self.seed_count]:
                paper = self.client._parse_paper(result)
                seeds.append(paper)
                self.all_papers[paper['id']] = paper
                logger.info(
                    f"  ✓ [{paper['year']}] {paper['title'][:60]}... "
                    f"(引用数: {paper['cited_by_count']})"
                )

            logger.info(f"共找到 {len(seeds)} 篇基石论文")
            return seeds

        except Exception as e:
            logger.error(f"选定基石论文失败: {e}")
            return []

    def _deduplicate_and_log(
        self,
        new_papers: List[Dict],
        existing_dict: Dict[str, Dict],
        paper_type: str
    ) -> Tuple[List[Dict], int, int]:
        """
        去重并记录统计信息

        Args:
            new_papers: 新检索到的论文列表
            existing_dict: 已存在的论文字典 {paper_id: paper_data}
            paper_type: 论文类型描述（用于日志）

        Returns:
            (去重后的新论文列表, 原始数量, 重复数量)
        """
        original_count = len(new_papers)
        duplicates = 0
        deduplicated = []

        for paper in new_papers:
            paper_id = paper['id']
            if paper_id not in existing_dict:
                deduplicated.append(paper)
                existing_dict[paper_id] = paper
            else:
                duplicates += 1

        final_count = len(deduplicated)

        logger.info(f"📊 {paper_type}去重统计:")
        logger.info(f"   原本数量: {original_count} 篇")
        logger.info(f"   检测到重复: {duplicates} 篇")
        logger.info(f"   去重后进入下一步: {final_count} 篇")

        return deduplicated, original_count, duplicates

    def _forward_snowballing(
        self,
        seed_papers: List[Dict],
        max_per_paper: Optional[int] = None
    ) -> List[Dict]:
        """
        第二步：正向滚雪球 - 找继承者
        找出谁引用了这些基石论文

        Args:
            seed_papers: 种子论文列表
            max_per_paper: 每篇论文最多扩展的数量（None表示使用默认值）

        Returns:
            引用论文列表
        """
        if max_per_paper is None:
            max_per_paper = self.citations_per_seed

        citing_papers_list = []  # 收集所有论文

        for seed in seed_papers:
            seed_id = seed['id']
            seed_year = seed['year']

            logger.info(f"\n处理种子论文: {seed['title'][:50]}...")
            logger.info(f"  种子ID: {seed_id}, 年份: {seed_year}")

            # 获取引用此论文的所有文献
            citing = self._get_filtered_citations(
                work_id=seed_id,
                min_year=seed_year,
                keywords=self.seed_keywords,
                max_results=max_per_paper
            )

            logger.info(f"  找到 {len(citing)} 篇相关引用论文")

            for paper in citing:
                citing_papers_list.append(paper)
                # 记录引用关系
                self.citation_edges.add((paper['id'], seed_id))

        # 统一去重
        result, _, _ = self._deduplicate_and_log(
            citing_papers_list,
            self.all_papers,
            "正向滚雪球"
        )

        logger.info(f"\n正向滚雪球完成，共收集 {len(result)} 篇继承者论文（去重后）")
        return result

    def _backward_snowballing(
        self,
        seed_papers: List[Dict],
        max_per_paper: Optional[int] = None
    ) -> List[Dict]:
        """
        第三步：反向滚雪球 - 找父节点/祖先
        找出这些基石论文引用了谁（它们的参考文献）

        Args:
            seed_papers: 种子论文列表
            max_per_paper: 每篇论文最多扩展的数量（None表示使用默认值）

        Returns:
            祖先论文列表
        """
        if max_per_paper is None:
            max_per_paper = self.citations_per_seed

        ancestor_papers_list = []  # 收集所有论文

        for seed in seed_papers:
            seed_id = seed['id']
            logger.info(f"\n处理种子论文: {seed['title'][:50]}...")
            logger.info(f"  种子ID: {seed_id}")

            # 获取此论文引用的参考文献
            references = self._get_references(
                work_id=seed_id,
                max_results=max_per_paper
            )

            logger.info(f"  找到 {len(references)} 篇参考文献（父节点）")

            for ref in references:
                ancestor_papers_list.append(ref)
                # 记录引用关系：种子引用了祖先
                self.citation_edges.add((seed_id, ref['id']))

        # 统一去重
        result, _, _ = self._deduplicate_and_log(
            ancestor_papers_list,
            self.all_papers,
            "反向滚雪球"
        )

        logger.info(f"\n反向滚雪球完成，共收集 {len(result)} 篇父节点/祖先论文（去重后）")
        return result

    def _cocitation_mining(
        self,
        citing_papers: List[Dict],
        ancestor_papers: List[Dict]
    ) -> List[Dict]:
        """
        第四步：横向补充/共引挖掘
        在子节点和父节点中，找出被大家反复提及但还不在库里的论文

        Args:
            citing_papers: 子节点论文列表
            ancestor_papers: 父节点论文列表

        Returns:
            共引论文列表
        """
        reference_counter = Counter()  # 统计每篇参考文献被引用的次数
        all_references = []

        # 合并子节点和父节点（去重）
        seen_ids = set()
        all_nodes = []
        for paper in citing_papers + ancestor_papers:
            if paper['id'] not in seen_ids:
                all_nodes.append(paper)
                seen_ids.add(paper['id'])

        logger.info(f"分析 {len(all_nodes)} 篇论文的共引模式（已去重）...")

        # 收集所有论文的参考文献
        analysis_limit = min(30, len(all_nodes))  # 限制数量以控制API调用
        for i, paper in enumerate(all_nodes[:analysis_limit], 1):
            logger.info(f"  [{i}/{analysis_limit}] 分析: {paper['title'][:40]}...")

            refs = self._get_references(paper['id'], max_results=10)
            for ref in refs:
                ref_id = ref['id']
                all_references.append(ref)
                reference_counter[ref_id] += 1

        # 找出被多次引用的论文
        cocitation_papers_list = []
        threshold = 3  # 至少被3篇论文引用

        logger.info(f"\n共引分析: 找出被频繁提及的论文（阈值: {threshold}次）")
        for ref_id, count in reference_counter.most_common(30):
            if count >= threshold:
                # 找到对应的论文详情
                ref_paper = next((r for r in all_references if r['id'] == ref_id), None)
                if ref_paper:
                    cocitation_papers_list.append(ref_paper)
                    logger.info(
                        f"  ✓ 候选共引论文: {ref_paper['title'][:50]}... "
                        f"(被引用{count}次, 总引用数: {ref_paper['cited_by_count']})"
                    )

        # 统一去重
        result, _, _ = self._deduplicate_and_log(
            cocitation_papers_list,
            self.all_papers,
            "共引挖掘"
        )

        logger.info(f"\n共引挖掘完成，找到 {len(result)} 篇被反复提及的论文（去重后）")
        return result

    def _execute_second_round_snowballing(self):
        """
        执行第二轮滚雪球：对第一轮得到的论文再进行一轮扩展
        包括：citing_papers, ancestor_papers, cocitation_papers
        """
        # 合并第一轮得到的所有论文（去重）
        seen_ids = set()
        first_round_papers = []
        for paper in self.citing_papers + self.ancestor_papers + self.cocitation_papers:
            if paper['id'] not in seen_ids:
                first_round_papers.append(paper)
                seen_ids.add(paper['id'])

        logger.info(f"第一轮共得到 {len(first_round_papers)} 篇论文（已去重）")
        logger.info(f"每篇论文最多扩展 {self.second_round_limit} 个引用")

        # 第二轮正向滚雪球
        logger.info("\n" + "-" * 60)
        logger.info("第二轮正向滚雪球：从第一轮论文找子节点")
        logger.info("-" * 60)

        # 使用统一的方法，传入限制参数
        second_citing = self._forward_snowballing(
            first_round_papers,
            max_per_paper=self.second_round_limit
        )
        self.second_round_citing_count = len(second_citing)

        # 合并到第一轮（使用字典去重）
        citing_dict = {p['id']: p for p in self.citing_papers}
        before_merge = len(citing_dict)
        citing_dict.update({p['id']: p for p in second_citing})
        after_merge = len(citing_dict)
        self.citing_papers = list(citing_dict.values())

        logger.info(f"📊 第二轮正向滚雪球与第一轮合并统计:")
        logger.info(f"   第一轮子节点: {len(self.citing_papers) - len(second_citing)} 篇")
        logger.info(f"   第二轮新增: {self.second_round_citing_count} 篇")
        logger.info(f"   合并时重复: {before_merge + self.second_round_citing_count - after_merge} 篇")
        logger.info(f"   合并后总计: {len(self.citing_papers)} 篇")

        # 第二轮反向滚雪球
        logger.info("\n" + "-" * 60)
        logger.info("第二轮反向滚雪球：从第一轮论文找父节点")
        logger.info("-" * 60)

        second_ancestor = self._backward_snowballing(
            first_round_papers,
            max_per_paper=self.second_round_limit
        )
        self.second_round_ancestor_count = len(second_ancestor)

        # 合并到第一轮
        ancestor_dict = {p['id']: p for p in self.ancestor_papers}
        before_merge = len(ancestor_dict)
        ancestor_dict.update({p['id']: p for p in second_ancestor})
        after_merge = len(ancestor_dict)
        self.ancestor_papers = list(ancestor_dict.values())

        logger.info(f"📊 第二轮反向滚雪球与第一轮合并统计:")
        logger.info(f"   第一轮父节点: {len(self.ancestor_papers) - len(second_ancestor)} 篇")
        logger.info(f"   第二轮新增: {self.second_round_ancestor_count} 篇")
        logger.info(f"   合并时重复: {before_merge + self.second_round_ancestor_count - after_merge} 篇")
        logger.info(f"   合并后总计: {len(self.ancestor_papers)} 篇")

        # 第二轮共引挖掘
        logger.info("\n" + "-" * 60)
        logger.info("第二轮共引挖掘：分析第二轮论文的共引模式")
        logger.info("-" * 60)

        second_cocitation = self._cocitation_mining(
            second_citing,
            second_ancestor
        )
        self.second_round_cocitation_count = len(second_cocitation)

        # 合并到第一轮
        cocitation_dict = {p['id']: p for p in self.cocitation_papers}
        before_merge = len(cocitation_dict)
        cocitation_dict.update({p['id']: p for p in second_cocitation})
        after_merge = len(cocitation_dict)
        self.cocitation_papers = list(cocitation_dict.values())

        logger.info(f"📊 第二轮共引挖掘与第一轮合并统计:")
        logger.info(f"   第一轮共引: {len(self.cocitation_papers) - len(second_cocitation)} 篇")
        logger.info(f"   第二轮新增: {self.second_round_cocitation_count} 篇")
        logger.info(f"   合并时重复: {before_merge + self.second_round_cocitation_count - after_merge} 篇")
        logger.info(f"   合并后总计: {len(self.cocitation_papers)} 篇")

        logger.info("\n" + "=" * 80)
        logger.info(f"✅ 第二轮滚雪球完成（已与第一轮去重合并）")
        logger.info(f"   最终论文总数（去重后）:")
        logger.info(f"     - 子节点: {len(self.citing_papers)} 篇")
        logger.info(f"     - 父节点: {len(self.ancestor_papers)} 篇")
        logger.info(f"     - 共引论文: {len(self.cocitation_papers)} 篇")
        logger.info(f"     - 合计: {len(self.citing_papers) + len(self.ancestor_papers) + len(self.cocitation_papers)} 篇")
        logger.info("=" * 80)

    def _add_recent_frontiers(
        self,
        topic: str,
        content_keyword: str,
        year_threshold: int
    ) -> List[Dict]:
        """
        第五步：补充最新SOTA论文（优化版）

        策略：
        - 如果启用arXiv：使用arXiv检索最新论文（6-12个月）-> 映射到OpenAlex
        - 如果不启用：使用OpenAlex搜索最新论文

        Args:
            topic: 主题关键词
            content_keyword: 内容关键词
            year_threshold: 年份阈值（大于等于此年份）

        Returns:
            最新论文列表
        """
        if self.use_arxiv_seeds:
            logger.info("🎯 使用arXiv检索最新SOTA论文")
            return self._add_recent_from_arxiv(topic, content_keyword)
        else:
            logger.info("🔍 使用OpenAlex检索最新论文")
            return self._add_recent_from_openalex(topic, content_keyword, year_threshold)

    def _add_recent_from_arxiv(
        self,
        topic: str,
        content_keyword: str
    ) -> List[Dict]:
        """
        从arXiv检索最新论文并映射到OpenAlex

        Args:
            topic: 主题
            content_keyword: 内容关键词

        Returns:
            最新论文列表（OpenAlex格式）
        """
        logger.info("  从arXiv检索最新6-12个月的前沿论文")

        # 1. 检索arXiv最新论文
        keywords = [content_keyword] + self.seed_keywords if self.seed_keywords else [content_keyword]
        arxiv_recent = self.arxiv_retriever.retrieve_recent_papers(
            topic=topic,
            keywords=keywords,
            max_results=self.recent_count * 2,  # 多取一些
            months_back=12
        )

        logger.info(f"  ✓ arXiv检索到 {len(arxiv_recent)} 篇最新论文")

        # 2. 映射到OpenAlex（不强制Concept验证，新论文可能还未被标注）
        mapped_recent, stats = self.cross_mapper.map_arxiv_to_openalex(
            arxiv_recent,
            verify_concepts=False  # 最新论文放宽验证
        )

        logger.info(f"  ✓ 成功映射 {len(mapped_recent)} 篇最新论文")

        # 3. 去重并存储
        recent_papers_list = []
        for paper in mapped_recent[:self.recent_count]:
            recent_papers_list.append(paper)
            logger.info(
                f"  ✓ 最新: [{paper.get('published_date', paper['year'])}] "
                f"{paper['title'][:60]}... (arXiv: {paper.get('arxiv_id', 'N/A')})"
            )

        # 去重
        result, _, _ = self._deduplicate_and_log(
            recent_papers_list,
            self.all_papers,
            "最新SOTA论文（arXiv）"
        )

        logger.info(f"✅ 共添加 {len(result)} 篇最新SOTA论文（arXiv）")
        return result

    def _add_recent_from_openalex(
        self,
        topic: str,
        content_keyword: str,
        year_threshold: int
    ) -> List[Dict]:
        """
        从OpenAlex检索最新论文（传统模式）

        Args:
            topic: 主题关键词
            content_keyword: 内容关键词
            year_threshold: 年份阈值（大于等于此年份）

        Returns:
            最新论文列表
        """
        query = f"{topic} {content_keyword}"
        logger.info(f"搜索最新论文: '{query}'")
        logger.info(f"筛选条件: publication_year >= {year_threshold}")

        params = {
            'search': query,
            'per-page': self.recent_count,
            'sort': 'cited_by_count:desc',  # 在最新论文中选引用数高的
            'filter': f'publication_year:>{year_threshold},cited_by_count:>5'
        }

        try:
            data = self.client._make_request('works', params)
            results = data.get('results', [])

            recent_papers_list = []
            for result in results[:self.recent_count]:
                paper = self.client._parse_paper(result)
                recent_papers_list.append(paper)
                logger.info(
                    f"  ✓ 候选最新论文: [{paper['year']}] {paper['title'][:60]}... "
                    f"(引用数: {paper['cited_by_count']})"
                )

            # 统一去重
            result, _, _ = self._deduplicate_and_log(
                recent_papers_list,
                self.all_papers,
                "最新SOTA论文"
            )

            logger.info(f"共添加 {len(result)} 篇最新SOTA论文（去重后）")
            return result

        except Exception as e:
            logger.error(f"补充最新论文失败: {e}")
            return []

    def _build_closure(self):
        """
        第五步：构建引用闭包
        为所有论文构建完整的引用关系网络
        """
        paper_ids = list(self.all_papers.keys())
        total_papers = len(paper_ids)

        logger.info(f"开始为 {total_papers} 篇论文构建引用闭包...")

        # 为每篇论文获取其引用关系
        for i, paper_id in enumerate(paper_ids, 1):
            paper = self.all_papers[paper_id]
            logger.info(
                f"  [{i}/{total_papers}] 处理: {paper['title'][:40]}..."
            )

            # 获取该论文引用的其他论文（在我们的集合中）
            cited_papers = self._get_references(paper_id, max_results=20)

            for cited in cited_papers:
                cited_id = cited['id']
                # 只记录集合内的引用关系
                if cited_id in self.all_papers and cited_id != paper_id:
                    edge = (paper_id, cited_id)
                    if edge not in self.citation_edges:
                        self.citation_edges.add(edge)
                        logger.debug(f"    添加边: {paper_id} -> {cited_id}")

        logger.info(f"引用闭包构建完成！共建立 {len(self.citation_edges)} 条引用关系")

    def _get_filtered_citations(
        self,
        work_id: str,
        min_year: int,
        keywords: List[str],
        max_results: int
    ) -> List[Dict]:
        """
        获取经过过滤的引用论文

        Args:
            work_id: 论文ID
            min_year: 最小年份
            keywords: 关键词列表（用于相关性过滤）
            max_results: 最大结果数

        Returns:
            过滤后的引用论文列表
        """
        if not work_id.startswith('W'):
            work_id = f"W{work_id}"

        # 构建过滤条件
        filters = [
            f'cites:{work_id}',
            f'publication_year:>{min_year}'
        ]

        params = {
            'filter': ','.join(filters),
            'per-page': max_results * 2,  # 多取一些，后续再过滤
            'sort': 'cited_by_count:desc'
        }

        try:
            data = self.client._make_request('works', params)
            results = data.get('results', [])

            # 解析并过滤论文
            filtered = []
            for result in results:
                paper = self.client._parse_paper(result)

                # 如果有关键词要求，进行相关性过滤
                if keywords and not self._is_relevant(paper, keywords):
                    continue

                filtered.append(paper)
                if len(filtered) >= max_results:
                    break

            return filtered

        except Exception as e:
            logger.error(f"获取过滤后的引用失败: {e}")
            return []

    def _get_references(self, work_id: str, max_results: int = 10) -> List[Dict]:
        """获取论文的参考文献"""
        if not work_id.startswith('W'):
            work_id = f"W{work_id}"

        params = {
            'filter': f'cited_by:{work_id}',
            'per-page': max_results,
            'sort': 'cited_by_count:desc'
        }

        try:
            data = self.client._make_request('works', params)
            results = data.get('results', [])

            references = []
            for result in results:
                ref = self.client._parse_paper(result)
                references.append(ref)

            return references

        except Exception as e:
            logger.error(f"获取参考文献失败: {e}")
            return []

    def _is_relevant(self, paper: Dict, keywords: List[str]) -> bool:
        """
        检查论文是否与关键词相关

        Args:
            paper: 论文数据
            keywords: 关键词列表

        Returns:
            是否相关
        """
        # 合并标题和摘要进行匹配
        text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()

        # 至少匹配一个关键词
        return any(keyword.lower() in text for keyword in keywords)

    def _generate_report(self) -> Dict:
        """
        生成检索报告

        Returns:
            包含所有数据和统计信息的字典
        """
        report = {
            'statistics': {
                'total_papers': len(self.all_papers),
                'seed_papers': len(self.seed_papers),
                'citing_papers': len(self.citing_papers),
                'ancestor_papers': len(self.ancestor_papers),
                'cocitation_papers': len(self.cocitation_papers),
                'recent_papers': len(self.recent_papers),
                'total_edges': len(self.citation_edges),
                # 第一轮统计
                'first_round_citing': self.first_round_citing_count,
                'first_round_ancestor': self.first_round_ancestor_count,
                'first_round_cocitation': self.first_round_cocitation_count,
                # 第二轮统计
                'second_round_citing': self.second_round_citing_count,
                'second_round_ancestor': self.second_round_ancestor_count,
                'second_round_cocitation': self.second_round_cocitation_count,
                # 第二轮是否启用
                'second_round_enabled': self.enable_second_round
            },
            'papers': self.all_papers,
            'citation_edges': list(self.citation_edges),
            'seed_ids': [p['id'] for p in self.seed_papers],
            'citing_ids': [p['id'] for p in self.citing_papers],
            'ancestor_ids': [p['id'] for p in self.ancestor_papers],
            'cocitation_ids': [p['id'] for p in self.cocitation_papers],
            'recent_ids': [p['id'] for p in self.recent_papers]
        }

        # 打印统计信息
        logger.info("\n" + "=" * 60)
        logger.info("检索统计报告")
        logger.info("=" * 60)
        logger.info(f"总论文数: {report['statistics']['total_papers']}")
        logger.info(f"  1. 基石种子: {report['statistics']['seed_papers']}")
        logger.info(f"  2. 子节点(引用种子): {report['statistics']['citing_papers']}")
        logger.info(f"     - 第一轮: {report['statistics']['first_round_citing']}")
        if self.enable_second_round:
            logger.info(f"     - 第二轮: {report['statistics']['second_round_citing']}")
        logger.info(f"  3. 父节点(被种子引用): {report['statistics']['ancestor_papers']}")
        logger.info(f"     - 第一轮: {report['statistics']['first_round_ancestor']}")
        if self.enable_second_round:
            logger.info(f"     - 第二轮: {report['statistics']['second_round_ancestor']}")
        logger.info(f"  4. 共引论文(横向补充): {report['statistics']['cocitation_papers']}")
        logger.info(f"     - 第一轮: {report['statistics']['first_round_cocitation']}")
        if self.enable_second_round:
            logger.info(f"     - 第二轮: {report['statistics']['second_round_cocitation']}")
        logger.info(f"  5. 最新SOTA: {report['statistics']['recent_papers']}")
        logger.info(f"总引用关系数: {report['statistics']['total_edges']}")
        logger.info(f"平均每篇论文的连接数: {report['statistics']['total_edges'] / max(report['statistics']['total_papers'], 1):.2f}")
        logger.info("=" * 60)

        return report

    def export_to_graph_format(self) -> Dict:
        """
        导出为图数据格式（便于可视化）

        Returns:
            包含节点和边的字典
        """
        nodes = []
        for paper_id, paper in self.all_papers.items():
            # 确定节点类型（优先级顺序）
            if paper_id in [p['id'] for p in self.seed_papers]:
                node_type = 'seed'
            elif paper_id in [p['id'] for p in self.ancestor_papers]:
                node_type = 'ancestor'
            elif paper_id in [p['id'] for p in self.citing_papers]:
                node_type = 'citing'
            elif paper_id in [p['id'] for p in self.cocitation_papers]:
                node_type = 'cocitation'
            elif paper_id in [p['id'] for p in self.recent_papers]:
                node_type = 'recent'
            else:
                node_type = 'other'

            nodes.append({
                'id': paper_id,
                'label': paper['title'][:50],
                'type': node_type,
                'year': paper['year'],
                'citations': paper['cited_by_count'],
                'authors': paper['authors']
            })

        edges = [
            {'source': source, 'target': target}
            for source, target in self.citation_edges
        ]

        return {
            'nodes': nodes,
            'edges': edges
        }


if __name__ == "__main__":
    # 测试代码
    logger.info("开始测试滚雪球检索系统...")

    # 创建检索系统
    retrieval = SnowballRetrieval(
        seed_count=5,
        citations_per_seed=6,
        recent_count=10,
        seed_keywords=["reasoning", "chain of thought", "prompting", "thinking"]
    )

    # 执行完整流程
    result = retrieval.execute_full_pipeline(
        topic="Large Language Models",
        content_keyword="Reasoning",
        seed_year_threshold=2023
    )

    # 导出图数据
    graph_data = retrieval.export_to_graph_format()
    logger.info(f"\n图数据导出完成：{len(graph_data['nodes'])} 个节点，{len(graph_data['edges'])} 条边")

    # 显示部分结果
    logger.info("\n示例论文（前5篇）：")
    for i, (paper_id, paper) in enumerate(list(result['papers'].items())[:5], 1):
        logger.info(f"{i}. [{paper['year']}] {paper['title']}")
        logger.info(f"   引用数: {paper['cited_by_count']}, 作者: {', '.join(paper['authors'])}")
