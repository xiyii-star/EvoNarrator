"""
Complete 8-Step Literature Retrieval Pipeline with Citation Network Construction

完整的8步文献检索流程与引用网络构建

8-Step Pipeline:
---------------
步骤 1+2: 种子检索与OpenAlex映射 (Seed Retrieval & OpenAlex Mapping - Combined)
    策略优化：放宽检索 + 确保映射
    - 阶段1: 使用arXiv API放宽检索条件，获取3倍于目标数量的候选论文（无年份限制）
    - 阶段2: 对所有候选论文批量进行OpenAlex映射
    - 阶段3: 只保留映射成功的论文，按质量(引用数+相关性)排序选择最佳种子
    - 优势: 确保所有种子都能在OpenAlex引用网络中使用，同时保持高召回率

步骤 3: 正向滚雪球 (Forward Snowballing)
    - Seed -> 谁引用了Seed? -> 子节点
    - 获取被引用论文的详细信息

步骤 4: 反向滚雪球 (Backward Snowballing)
    - 谁被Seed引用了? <- Seed -> 父节点/祖先
    - 获取引用论文的详细信息

步骤 5: 横向补充/共引挖掘 (Co-citation Mining)
    - 在子节点和父节点中，谁被大家反复提及?
    - 共引阈值过滤高价值论文

步骤 6 [可选]: 第二轮滚雪球 (Second-Round Snowballing)
    - 对第一轮论文再进行一轮受控扩展
    - 控制扩展规模

步骤 7: 补充最新SOTA (Recent Frontiers Supplementation)
    - arXiv最近6-12个月论文
    - 相似度过滤

步骤 8: 构建引用闭包 (Citation Closure Construction)
    - 建立完整网络
    - 填补缺失的引用关系连接引用

Date: 2025-12-09
Version: 2.0 (Combined Step 1+2 for better seed quality)
"""

import logging
import time
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import yaml

# 导入依赖模块
from openalex_client import OpenAlexClient
from arxiv_seed_retriever import ArxivSeedRetriever
from cross_database_mapper import CrossDatabaseMapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PaperSearchPipeline:
    """
    完整的8步文献检索流程（优化版：步骤1+2合并）

    策略优化：
    - 步骤1+2合并：放宽arXiv检索 → 批量OpenAlex映射 → 确保所有种子可用
    - 整合arXiv种子检索、OpenAlex引用扩展、共引挖掘
    - 确保所有种子论文都在OpenAlex有映射，可进行引用网络扩展
    """

    def __init__(
        self,
        openalex_client: Optional[OpenAlexClient] = None,
        config_path: str = './config/config.yaml',
        llm_client = None
    ):
        """
        初始化流程

        Args:
            openalex_client: OpenAlex客户端
            config_path: 配置文件路径
            llm_client: LLM客户端(用于查询生成)
        """
        # 初始化客户端
        self.openalex_client = openalex_client or OpenAlexClient()
        self.llm_client = llm_client

        # 加载配置
        self.config = self._load_config(config_path)

        # 初始化检索器
        self._init_retrievers()

        # 数据存储结构
        self.papers = {}  # paper_id -> paper_dict
        self.citation_edges = []  # [(source_id, target_id), ...]

        # 各步骤结果缓存(用于调试和统计)
        self.seed_papers = []  # 步骤1: 种子论文
        self.mapped_seeds = []  # 步骤2: 映射成功的种子
        self.unmapped_seeds = []  # 步骤2: 映射失败的种子
        self.forward_papers = []  # 步骤3: 正向引用论文
        self.backward_papers = []  # 步骤4: 反向引用论文
        self.cocitation_papers = []  # 步骤5: 共引论文
        self.second_round_papers = []  # 步骤6: 第二轮扩展论文（总计）
        self.second_round_citing = []  # 步骤6: 第二轮正向引用论文
        self.second_round_ancestor = []  # 步骤6: 第二轮反向引用论文
        self.recent_papers = []  # 步骤7: 最新论文

        # 统计信息
        self.statistics = {
            'seed_papers': 0,
            'arxiv_mapped': 0,
            'arxiv_unmapped': 0,
            'manual_citations_built': 0,
            'first_round_citing': 0,
            'first_round_ancestor': 0,
            'first_round_cocitation': 0,
            'second_round_enabled': False,
            'second_round_citing': 0,
            'second_round_ancestor': 0,
            'recent_papers': 0,
            'total_papers': 0,
            'total_edges': 0
        }

        logger.info("="*70)
        logger.info("初始化论文检索流程")
        logger.info("="*70)
        logger.info(f"配置参数:")
        logger.info(f"  - 种子数量: {self.config['seed_count']}")
        logger.info(f"  - 每种子引用数: {self.config['citations_per_seed']}")
        logger.info(f"  - 共引阈值: {self.config['cocitation_threshold']}")
        logger.info(f"  - 第二轮扩展: {'启用' if self.config['enable_second_round'] else '禁用'}")
        logger.info(f"  - 最新论文数: {self.config['recent_count']}")
        logger.info("="*70)

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
                return self._default_config()

            with open(config_file, 'r', encoding='utf-8') as f:
                full_config = yaml.safe_load(f)
                snowball_config = full_config.get('snowball', {})

                config = {
                    # 种子检索
                    'seed_count': snowball_config.get('seed_count', 10),
                    'arxiv_years_back': snowball_config.get('arxiv_years_back', 5),

                    # 引用数量
                    'citations_per_seed': snowball_config.get('citations_per_seed', 15),
                    'references_per_seed': snowball_config.get('references_per_seed', 10),

                    # 共引
                    'cocitation_threshold': snowball_config.get('cocitation_threshold', 3),
                    'max_cocitation_papers': snowball_config.get('max_cocitation_papers', 20),

                    # 第二轮扩展
                    'enable_second_round': snowball_config.get('enable_second_round', True),
                    'second_round_limit': snowball_config.get('second_round_limit', 5),
                    'second_round_max_papers': snowball_config.get('second_round_max_papers', 50),

                    # 最新论文
                    'recent_months': snowball_config.get('recent_months', 12),
                    'recent_count': snowball_config.get('recent_count', 10),

                    # 其他
                    'use_llm_query': snowball_config.get('use_llm_query', True),
                    'min_citation_count': snowball_config.get('min_citation_count', 5),

                    # LLM语义扩展
                    'llm_semantic_expansion': snowball_config.get('llm_semantic_expansion', True),
                    'expansion_max_topics': snowball_config.get('expansion_max_topics', 4),
                    'expansion_max_keywords': snowball_config.get('expansion_max_keywords', 8),
                }

                logger.info(f"成功加载配置文件: {config_path}")
                return config

        except Exception as e:
            logger.warning(f"加载配置失败: {e}，使用默认配置")
            return self._default_config()

    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'seed_count': 10,
            'arxiv_years_back': 5,
            'citations_per_seed': 15,
            'references_per_seed': 10,
            'cocitation_threshold': 3,
            'max_cocitation_papers': 20,
            'enable_second_round': True,
            'second_round_limit': 5,
            'second_round_max_papers': 50,
            'recent_months': 12,
            'recent_count': 10,
            'use_llm_query': True,
            'min_citation_count': 5,
            'llm_semantic_expansion': True,
            'expansion_max_topics': 4,
            'expansion_max_keywords': 8,
        }

    def _init_retrievers(self):
        """初始化检索器"""
        # arXiv种子检索器
        self.arxiv_retriever = ArxivSeedRetriever(
            max_results_per_query=self.config['seed_count'] * 2,
            years_back=self.config['arxiv_years_back'],
            min_relevance_score=0.5,
            llm_client=self.llm_client,
            use_llm_query_generation=self.config['use_llm_query'],
            enable_semantic_expansion=self.config.get('llm_semantic_expansion', True),
            expansion_max_topics=self.config.get('expansion_max_topics', 4),
            expansion_max_keywords=self.config.get('expansion_max_keywords', 8)
        )

        # 跨库映射器
        self.cross_mapper = CrossDatabaseMapper(
            client=self.openalex_client,
            min_concept_score=0.3,
            required_concepts=["Computer Science"]
        )

        logger.info("检索器初始化完成")

    def execute_full_pipeline(
        self,
        topic: str,
        keywords: Optional[List[str]] = None,
        categories: Optional[List[str]] = None
    ) -> Dict:
        """
        执行完整的8步检索流程

        Args:
            topic: 研究主题
            keywords: 关键词列表(可选)
            categories: arXiv分类列表(可选)

        Returns:
            {
                'papers': {paper_id: paper_dict},
                'citation_edges': [(source_id, target_id), ...],
                'statistics': {...}
            }
        """
        logger.info("\n" + "="*70)
        logger.info(f"开始执行8步文献检索流程")
        logger.info(f"主题: {topic}")
        logger.info("="*70 + "\n")

        start_time = time.time()

        # 步骤1: 高质量种子获取
        self._step1_seed_retrieval(topic, keywords, categories)

        # 步骤2: 跨库ID映射
        self._step2_cross_database_mapping()

        # 步骤3: 正向滚雪球
        self._step3_forward_snowballing()

        # 步骤4: 反向滚雪球
        self._step4_backward_snowballing()

        # 步骤5: 共引挖掘
        self._step5_cocitation_mining()

        # 步骤6: 第二轮扩展(可选)
        if self.config['enable_second_round']:
            self._step6_second_round_snowballing()

        # 步骤7: 补充最新SOTA
        self._step7_recent_frontiers(topic, keywords, categories)

        # 步骤8: 构建引用闭包
        self._step8_citation_closure()

        # 更新统计信息
        self._finalize_statistics()

        elapsed_time = time.time() - start_time
        logger.info("\n" + "="*70)
        logger.info(f"8步检索流程完成，耗时: {elapsed_time:.2f}秒")
        logger.info("="*70)
        self._print_summary()

        return {
            'papers': self.papers,
            'citation_edges': self.citation_edges,
            'statistics': self.statistics
        }

    def _step1_seed_retrieval(
        self,
        topic: str,
        keywords: Optional[List[str]],
        categories: Optional[List[str]]
    ):
        """
        步骤1: 高质量种子获取（结合步骤2：确保OpenAlex映射）

        策略：
        1. 放宽arXiv检索条件，获取更多候选论文（扩大初始检索范围，无年份限制）
        2. 对所有候选论文进行OpenAlex映射
        3. 只保留映射成功的论文作为最终种子
        4. 按质量排序（引用数+相关性）选择最佳种子
        """
        logger.info("\n" + "="*70)
        logger.info("步骤1+2: 种子检索与OpenAlex映射（组合流程）")
        logger.info("="*70)
        logger.info("策略: 放宽arXiv检索 → 批量OpenAlex映射 → 保留映射成功的论文")

        target_seed_count = self.config['seed_count']

        # 放宽检索条件：检索更多候选论文（3倍于目标数量）
        # 因为考虑到映射成功率，需要更多候选
        candidate_count = target_seed_count * 3

        logger.info(f"\n阶段1: arXiv候选论文检索（放宽条件）")
        logger.info(f"  - 目标种子数: {target_seed_count}")
        logger.info(f"  - 候选检索数: {candidate_count}")

        try:
            # 临时降低相关性阈值以获取更多候选
            original_threshold = self.arxiv_retriever.min_relevance_score
            self.arxiv_retriever.min_relevance_score = 0.3  # 放宽到0.3

            # 使用arXiv检索器获取候选论文
            arxiv_candidates = self.arxiv_retriever.retrieve_seed_papers(
                topic=topic,
                keywords=keywords,
                categories=categories,
                max_seeds=candidate_count
            )

            # 恢复原阈值
            self.arxiv_retriever.min_relevance_score = original_threshold

            # 不再过滤年份，接受所有候选论文
            filtered_candidates = arxiv_candidates

            logger.info(f"  ✓ arXiv检索到 {len(filtered_candidates)} 篇候选论文（所有年份）")

            if not filtered_candidates:
                logger.warning("  ⚠️ 没有找到符合条件的arXiv候选论文")
                self.seed_papers = []
                self.mapped_seeds = []
                self.unmapped_seeds = []
                return

            # 阶段2: 批量映射到OpenAlex
            logger.info(f"\n阶段2: 批量映射到OpenAlex")
            logger.info(f"  - 候选论文数: {len(filtered_candidates)}")

            # 使用映射器进行ID映射（禁用概念验证，因为arXiv阶段已过滤）
            mapped_papers, mapping_stats = self.cross_mapper.map_arxiv_to_openalex(
                arxiv_papers=filtered_candidates,
                verify_concepts=False  # arXiv论文已经过滤
            )

            logger.info(f"\n映射结果:")
            logger.info(f"  - 映射成功: {len(mapped_papers)} 篇")
            logger.info(f"  - 映射失败: {mapping_stats['failed']} 篇")
            logger.info(f"  - 成功率: {mapping_stats.get('success_rate', 0):.1%}")

            # 阶段3: 按质量排序，选择最佳种子
            logger.info(f"\n阶段3: 选择高质量种子")

            # 按引用数和相关性综合排序
            for paper in mapped_papers:
                # 综合得分 = 归一化引用数 * 0.6 + 相关性得分 * 0.4
                cited_count = paper.get('cited_by_count', 0)
                relevance = paper.get('relevance_score', 0.5)

                # 简单归一化（log scale）
                normalized_citation = min(1.0, cited_count / 100.0) if cited_count > 0 else 0
                paper['quality_score'] = normalized_citation * 0.6 + relevance * 0.4

            # 排序
            mapped_papers.sort(key=lambda x: x.get('quality_score', 0), reverse=True)

            # 选择top N作为最终种子
            final_seeds = mapped_papers[:target_seed_count]

            logger.info(f"  - 选择前 {len(final_seeds)} 篇作为最终种子")

            # 更新种子列表
            self.seed_papers = filtered_candidates  # 保留原始arXiv检索结果用于统计
            self.mapped_seeds = final_seeds  # 映射成功的种子（用于后续流程）
            self.unmapped_seeds = [
                p for p in filtered_candidates
                if not any(m.get('arxiv_id') == p.get('arxiv_id') for m in mapped_papers)
            ]

            # 将映射成功的种子论文加入papers字典，并标记为种子节点
            for paper in self.mapped_seeds:
                paper_id = paper['id']
                paper['is_seed'] = True  # 添加种子节点标记
                self.papers[paper_id] = paper

            logger.info(f"\n最终种子统计:")
            logger.info(f"  - arXiv候选数: {len(filtered_candidates)} （所有年份）")
            logger.info(f"  - OpenAlex映射成功: {len(mapped_papers)}")
            logger.info(f"  - 最终种子数: {len(final_seeds)}")
            logger.info(f"  - 映射失败(丢弃): {len(self.unmapped_seeds)}")

            # 打印种子论文信息和ID
            logger.info(f"\n🌱 最终种子论文（Top {min(5, len(final_seeds))}）:")
            for i, paper in enumerate(final_seeds[:5], 1):
                logger.info(
                    f"  [{i}] {paper['title'][:60]}... "
                    f"({paper.get('year', 'N/A')}, "
                    f"引用:{paper.get('cited_by_count', 0)}, "
                    f"质量分:{paper.get('quality_score', 0):.2f})"
                )
                logger.info(f"      ID: {paper['id']}")
            if len(final_seeds) > 5:
                logger.info(f"  ... 还有 {len(final_seeds) - 5} 篇")

            # 记录所有种子节点ID
            seed_ids = [p['id'] for p in final_seeds]
            logger.info(f"\n🔑 种子节点ID列表: {seed_ids}")

            # 如果种子数量不足，发出警告
            if len(final_seeds) < target_seed_count:
                logger.warning(
                    f"\n⚠️ 警告: 最终种子数({len(final_seeds)}) "
                    f"少于目标数({target_seed_count})，"
                    f"建议放宽检索条件或调整年份限制"
                )

        except Exception as e:
            logger.error(f"种子检索与映射失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.seed_papers = []
            self.mapped_seeds = []
            self.unmapped_seeds = []

    def _step2_cross_database_mapping(self):
        """
        步骤2: 跨库ID映射（已整合到步骤1）

        注意：此步骤已与步骤1合并，此方法仅作为占位符保留
        实际映射逻辑已在 _step1_seed_retrieval 中完成
        """
        logger.info("\n" + "-"*70)
        logger.info("步骤2: 跨库ID映射 (已整合到步骤1)")
        logger.info("-"*70)
        logger.info("✓ 映射已在步骤1完成，跳过")

        # 统计信息已在步骤1更新，这里无需额外操作
        pass

    def _handle_unmapped_seeds(self):
        """处理映射失败的种子论文"""
        manual_count = 0

        for seed in self.unmapped_seeds:
            try:
                # 尝试在OpenAlex中搜索
                title = seed.get('title', '')
                if not title:
                    continue

                search_results = self.openalex_client.search_papers(
                    topic=title,
                    max_results=1,
                    sort_by="relevance"
                )

                if search_results:
                    paper = search_results[0]
                    paper_id = paper['id']
                    self.papers[paper_id] = paper
                    manual_count += 1
                    logger.info(f"  手动搜索成功: {title[:50]}")

            except Exception as e:
                logger.debug(f"  手动搜索失败: {seed.get('title', 'Unknown')[:50]} - {e}")

        self.statistics['manual_citations_built'] = manual_count
        logger.info(f"  - 手动搜索成功: {manual_count} 篇")

    def _step3_forward_snowballing(self):
        """
        步骤3: 正向滚雪球
        找到哪些论文引用了种子论文的详细信息
        """
        logger.info("\n" + "-"*70)
        logger.info("步骤3: 正向滚雪球 (Forward Snowballing)")
        logger.info("-"*70)
        logger.info("策略: Seed -> 谁引用了Seed? -> 子节点")

        if not self.mapped_seeds:
            logger.warning("没有可用的映射种子，跳过正向滚雪球")
            return

        citing_papers = []
        citations_per_seed = self.config['citations_per_seed']

        logger.info(f"开始处理 {len(self.mapped_seeds)} 个种子论文，每个获取最多 {citations_per_seed} 篇引用...")

        for i, seed in enumerate(self.mapped_seeds, 1):
            seed_id = seed['id']
            seed_title = seed.get('title', 'Unknown')

            try:
                # 获取引用该种子论文的论文
                citations = self.openalex_client.get_citations(
                    paper_id=seed_id,
                    max_results=citations_per_seed
                )

                # 添加引用论文
                new_papers_count = 0
                for citation in citations:
                    citation_id = citation['id']

                    # 添加到论文集合
                    if citation_id not in self.papers:
                        self.papers[citation_id] = citation
                        citing_papers.append(citation)
                        new_papers_count += 1

                    # 添加引用边: citation -> seed
                    edge = (citation_id, seed_id)
                    if edge not in self.citation_edges:
                        self.citation_edges.append(edge)

                # 简化输出：只显示关键信息
                logger.info(f"  [{i}/{len(self.mapped_seeds)}] {seed_title[:50]}... → +{new_papers_count} 新论文 (共{len(citations)}篇引用)")

            except Exception as e:
                logger.warning(f"  [{i}/{len(self.mapped_seeds)}] 获取引用失败: {seed_title[:40]}... - {e}")

        self.forward_papers = citing_papers
        logger.info(f"\n✅ 正向滚雪球完成: 新增 {len(citing_papers)} 篇引用论文")

    def _step4_backward_snowballing(self):
        """
        步骤4: 反向滚雪球
        找到种子论文引用了哪些论文
        """
        logger.info("\n" + "-"*70)
        logger.info("步骤4: 反向滚雪球 (Backward Snowballing)")
        logger.info("-"*70)
        logger.info("策略: 谁被Seed引用了? <- Seed -> 父节点/祖先")

        if not self.mapped_seeds:
            logger.warning("没有可用的映射种子，跳过反向滚雪球")
            return

        referenced_papers = []
        references_per_seed = self.config['references_per_seed']

        logger.info(f"开始处理 {len(self.mapped_seeds)} 个种子论文，每个获取最多 {references_per_seed} 篇参考文献...")

        for i, seed in enumerate(self.mapped_seeds, 1):
            seed_id = seed['id']
            seed_title = seed.get('title', 'Unknown')

            try:
                # 获取该种子论文引用的论文
                references = self.openalex_client.get_references(
                    paper_id=seed_id,
                    max_results=references_per_seed
                )

                # 添加引用论文
                new_papers_count = 0
                for reference in references:
                    reference_id = reference['id']

                    # 添加到论文集合
                    if reference_id not in self.papers:
                        self.papers[reference_id] = reference
                        referenced_papers.append(reference)
                        new_papers_count += 1

                    # 添加引用边: seed -> reference
                    edge = (seed_id, reference_id)
                    if edge not in self.citation_edges:
                        self.citation_edges.append(edge)

                # 简化输出：只显示关键信息
                logger.info(f"  [{i}/{len(self.mapped_seeds)}] {seed_title[:50]}... → +{new_papers_count} 新论文 (共{len(references)}篇参考)")

            except Exception as e:
                logger.warning(f"  [{i}/{len(self.mapped_seeds)}] 获取参考文献失败: {seed_title[:40]}... - {e}")

        self.backward_papers = referenced_papers
        logger.info(f"\n✅ 反向滚雪球完成: 新增 {len(referenced_papers)} 篇祖先论文")

    def _step5_cocitation_mining(self):
        """
        步骤5: 共引挖掘
        找到在子节点和父节点中被反复提及的高价值论文
        """
        logger.info("\n" + "-"*70)
        logger.info("步骤5: 共引挖掘 (Co-citation Mining)")
        logger.info("-"*70)
        logger.info("策略: 统计共同引用的论文")

        # 合并第一轮的种子论文 + 引用论文
        first_round_papers = self.forward_papers + self.backward_papers

        if not first_round_papers:
            logger.warning("没有可用的第一轮论文，跳过共引挖掘")
            return

        # 统计论文被引用的次数
        cocitation_counter = Counter()

        for paper in first_round_papers:
            paper_id = paper['id']

            try:
                # 获取该论文的参考文献
                references = self.openalex_client.get_references(
                    paper_id=paper_id,
                    max_results=20
                )

                # 统计参考文献被引用次数
                for ref in references:
                    ref_id = ref['id']
                    # 只统计不在当前论文集合中的论文
                    if ref_id not in self.papers:
                        cocitation_counter[ref_id] += 1

            except Exception as e:
                logger.debug(f"  获取参考文献失败: {paper['title'][:50]} - {e}")

        # 根据共引次数过滤论文
        threshold = self.config['cocitation_threshold']
        max_papers = self.config['max_cocitation_papers']

        cocited_paper_ids = [
            paper_id for paper_id, count in cocitation_counter.most_common()
            if count >= threshold
        ][:max_papers]

        logger.info(f"找到 {len(cocited_paper_ids)} 个共引论文(阈值≥{threshold})")

        # 获取共引论文的详细信息
        cocitation_papers = []
        for paper_id in cocited_paper_ids:
            try:
                paper = self.openalex_client.get_paper_by_id(paper_id)
                if paper:
                    self.papers[paper_id] = paper
                    cocitation_papers.append(paper)

                    # 添加引用边(从引用共引论文的论文到共引论文)
                    for citing_paper in first_round_papers:
                        citing_id = citing_paper['id']
                        try:
                            refs = self.openalex_client.get_references(citing_id, max_results=50)
                            if any(r['id'] == paper_id for r in refs):
                                edge = (citing_id, paper_id)
                                if edge not in self.citation_edges:
                                    self.citation_edges.append(edge)
                        except:
                            pass

            except Exception as e:
                logger.debug(f"  获取共引论文失败: {paper_id} - {e}")

        self.cocitation_papers = cocitation_papers
        logger.info(f"共引挖掘获取 {len(cocitation_papers)} 篇共引论文")

    def _step6_second_round_snowballing(self):
        """
        步骤6: 第二轮滚雪球(可选)
        对第一轮论文再进行一轮受控扩展
        """
        logger.info("\n" + "-"*70)
        logger.info("步骤6: 第二轮滚雪球 (Second-Round Snowballing)")
        logger.info("-"*70)
        logger.info("策略: 对第一轮论文再进行一轮受控扩展")

        self.statistics['second_round_enabled'] = True

        # 选择高质量的第一轮论文进行扩展
        first_round_papers = self.forward_papers + self.backward_papers + self.cocitation_papers

        # 选择引用量最高的top论文
        sorted_papers = sorted(
            first_round_papers,
            key=lambda p: p.get('cited_by_count', 0),
            reverse=True
        )

        max_expand = self.config['second_round_max_papers']
        papers_to_expand = sorted_papers[:max_expand]

        logger.info(f"选择 {len(papers_to_expand)} 篇高引用论文进行第二轮扩展")

        second_round_citing = []
        second_round_ancestor = []
        limit_per_paper = self.config['second_round_limit']

        for i, paper in enumerate(papers_to_expand, 1):
            paper_id = paper['id']
            paper_title = paper.get('title', 'Unknown')

            try:
                # 正向滚雪球(获取引用该论文的论文)
                citations = self.openalex_client.get_citations(
                    paper_id=paper_id,
                    max_results=limit_per_paper
                )

                for citation in citations:
                    citation_id = citation['id']
                    if citation_id not in self.papers:
                        self.papers[citation_id] = citation
                        second_round_citing.append(citation)

                    edge = (citation_id, paper_id)
                    if edge not in self.citation_edges:
                        self.citation_edges.append(edge)

                # 反向滚雪球(获取该论文引用的论文)
                references = self.openalex_client.get_references(
                    paper_id=paper_id,
                    max_results=limit_per_paper
                )

                for reference in references:
                    reference_id = reference['id']
                    if reference_id not in self.papers:
                        self.papers[reference_id] = reference
                        second_round_ancestor.append(reference)

                    edge = (paper_id, reference_id)
                    if edge not in self.citation_edges:
                        self.citation_edges.append(edge)

                if i <= 5 or i % 10 == 0:
                    logger.info(f"  [{i}/{len(papers_to_expand)}] {paper_title[:50]}... -> {len(citations)}引用 + {len(references)}参考")

            except Exception as e:
                logger.debug(f"  第二轮扩展失败: {paper_title[:50]} - {e}")

        # 保存第二轮结果
        self.second_round_citing = second_round_citing
        self.second_round_ancestor = second_round_ancestor
        self.second_round_papers = second_round_citing + second_round_ancestor

        logger.info(f"第二轮滚雪球结果:")
        logger.info(f"  - 正向子节点: {len(second_round_citing)} 篇")
        logger.info(f"  - 反向祖先: {len(second_round_ancestor)} 篇")

    def _step7_recent_frontiers(
        self,
        topic: str,
        keywords: Optional[List[str]],
        categories: Optional[List[str]]
    ):
        """
        步骤7: 补充最新SOTA
        从arXiv获取最近6-12个月的最新论文
        """
        logger.info("\n" + "-"*70)
        logger.info("步骤7: 补充最新SOTA (Recent Frontiers Supplementation)")
        logger.info("-"*70)
        logger.info(f"策略: arXiv最近{self.config['recent_months']}个月论文")

        try:
            # 使用arXiv检索器获取最新论文
            try:
                import arxiv
            except ImportError:
                logger.warning("arxiv包未安装，跳过最新论文补充")
                self.recent_papers = []
                return

            # 计算日期范围
            months_back = self.config['recent_months']
            start_date = datetime.now() - timedelta(days=30 * months_back)

            logger.info(f"  - 时间范围: >= {start_date.strftime('%Y-%m-%d')}")
            logger.info(f"  - 目标数量: {self.config['recent_count']} 篇")

            # 临时降低相关性阈值以获取更多最新论文
            original_threshold = self.arxiv_retriever.min_relevance_score
            self.arxiv_retriever.min_relevance_score = 0.25  # 进一步放宽到0.25（最新论文可能相关性评分偏低）

            recent_papers_raw = self.arxiv_retriever.retrieve_seed_papers(
                topic=topic,
                keywords=keywords,
                categories=categories,
                max_seeds=self.config['recent_count'] * 3,  # 多取一些
                sort_by=arxiv.SortCriterion.SubmittedDate  # 按提交日期排序
            )

            # 恢复原阈值
            self.arxiv_retriever.min_relevance_score = original_threshold

            logger.info(f"  → arXiv检索到 {len(recent_papers_raw)} 篇候选论文")

            # 过滤只保留最近时间的论文
            recent_filtered = []
            for paper in recent_papers_raw:
                # 使用published_date字段进行精确的时间过滤
                published_date = paper.get('published_date')

                if published_date:
                    # 如果published_date是datetime对象
                    if isinstance(published_date, datetime):
                        pub_date = published_date
                    else:
                        # 尝试解析字符串
                        try:
                            pub_date = datetime.fromisoformat(str(published_date).replace('Z', '+00:00'))
                        except:
                            # 回退到年份比较
                            pub_year = paper.get('year', 0)
                            if pub_year >= start_date.year:
                                recent_filtered.append(paper)
                            continue

                    # 确保pub_date是naive datetime（移除时区信息以便与start_date比较）
                    if pub_date.tzinfo is not None:
                        pub_date = pub_date.replace(tzinfo=None)

                    # 比较完整的日期
                    if pub_date >= start_date:
                        recent_filtered.append(paper)
                        logger.debug(f"  ✓ 保留: {paper['title'][:50]}... ({pub_date.strftime('%Y-%m-%d')})")
                    else:
                        logger.debug(f"  × 过滤: {paper['title'][:50]}... ({pub_date.strftime('%Y-%m-%d')}，早于{start_date.strftime('%Y-%m-%d')})")
                else:
                    # 没有日期信息，使用年份作为后备
                    pub_year = paper.get('year', 0)
                    if pub_year >= start_date.year:
                        recent_filtered.append(paper)
                        logger.debug(f"  ✓ 保留: {paper['title'][:50]}... (年份:{pub_year})")

            recent_filtered = recent_filtered[:self.config['recent_count']]

            logger.info(f"  → 过滤后保留 {len(recent_filtered)} 篇最近{months_back}个月的论文")

            # 映射到OpenAlex
            if recent_filtered:
                logger.info(f"\n开始映射到OpenAlex...")
                mapped_recent, _ = self.cross_mapper.map_arxiv_to_openalex(
                    arxiv_papers=recent_filtered,
                    verify_concepts=False
                )

                logger.info(f"  → 映射成功 {len(mapped_recent)} 篇")

                # 添加到论文集合
                for paper in mapped_recent:
                    paper_id = paper['id']
                    if paper_id not in self.papers:
                        self.papers[paper_id] = paper
                        self.recent_papers.append(paper)

                        # 尝试连接引用关系论文
                        self._connect_recent_paper(paper)

            logger.info(f"\n✅ 补充最新论文: 新增 {len(self.recent_papers)} 篇最新论文")

        except Exception as e:
            logger.error(f"补充最新论文失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.recent_papers = []

    def _connect_recent_paper(self, paper: Dict):
        """将最新论文连接到引用论文网络"""
        paper_id = paper['id']

        try:
            # 获取该论文的参考文献
            references = self.openalex_client.get_references(
                paper_id=paper_id,
                max_results=20
            )

            # 如果参考文献在我们的论文集合中，添加引用边
            for ref in references:
                ref_id = ref['id']
                if ref_id in self.papers:
                    edge = (paper_id, ref_id)
                    if edge not in self.citation_edges:
                        self.citation_edges.append(edge)

        except Exception as e:
            logger.debug(f"  连接最新论文失败: {paper['title'][:50]} - {e}")

    def _step8_citation_closure(self):
        """
        步骤8: 构建引用闭包
        填补论文集合之间缺失的引用关系，构建完整网络
        """
        logger.info("\n" + "-"*70)
        logger.info("步骤8: 构建引用闭包 (Citation Closure Construction)")
        logger.info("-"*70)
        logger.info("策略: 补全论文集合之间引用关系")

        initial_edges = len(self.citation_edges)
        paper_ids = list(self.papers.keys())

        # 优化: 使用集合加速查找
        paper_ids_set = set(paper_ids)
        citation_edges_set = set(self.citation_edges)

        logger.info(f"  当前论文数: {len(paper_ids)}")
        logger.info(f"  当前引用边数: {initial_edges}")

        # 计算实际检查数量
        max_check = min(50, len(paper_ids))
        logger.info(f"  将检查前 {max_check} 篇论文的引用关系\n")

        # 检查论文之间的引用关系缺失引用关系
        new_edges = 0
        checked_papers = 0
        failed_papers = 0
        start_time = time.time()

        for i, source_id in enumerate(paper_ids[:max_check]):
            try:
                # 获取该论文的参考文献
                references = self.openalex_client.get_references(
                    paper_id=source_id,
                    max_results=50
                )

                # 批量检查引用关系
                for ref in references:
                    ref_id = ref['id']
                    if ref_id in paper_ids_set:
                        edge = (source_id, ref_id)
                        if edge not in citation_edges_set:
                            self.citation_edges.append(edge)
                            citation_edges_set.add(edge)
                            new_edges += 1

                checked_papers += 1

                # 优化输出频率: 每20%或每10篇输出一次
                if checked_papers % max(1, max_check // 5) == 0 or checked_papers % 10 == 0:
                    progress = (checked_papers / max_check) * 100
                    elapsed = time.time() - start_time
                    rate = checked_papers / elapsed if elapsed > 0 else 0
                    eta = (max_check - checked_papers) / rate if rate > 0 else 0

                    logger.info(
                        f"  进度: [{checked_papers}/{max_check}] {progress:.0f}% | "
                        f"新增边: {new_edges} | "
                        f"失败: {failed_papers} | "
                        f"速度: {rate:.1f}篇/s | "
                        f"预计剩余: {eta:.0f}s"
                    )

            except Exception as e:
                failed_papers += 1
                logger.debug(f"  跳过论文 {source_id[:20]}... : {str(e)[:50]}")

        # 最终统计
        elapsed_total = time.time() - start_time
        logger.info(f"\n✅ 引用闭包构建完成 (耗时 {elapsed_total:.1f}s):")
        logger.info(f"  检查论文: {checked_papers}/{max_check}")
        logger.info(f"  失败论文: {failed_papers}")
        logger.info(f"  初始引用边: {initial_edges}")
        logger.info(f"  新增引用边: {new_edges}")
        logger.info(f"  最终引用边: {len(self.citation_edges)}")

        if new_edges > 0:
            logger.info(f"  增长率: +{(new_edges/initial_edges*100):.1f}%")

        # 计算网络密度
        if len(paper_ids) > 1:
            max_possible_edges = len(paper_ids) * (len(paper_ids) - 1)
            density = len(self.citation_edges) / max_possible_edges * 100
            logger.info(f"  网络密度: {density:.2f}%")

    def _finalize_statistics(self):
        """更新最终统计信息"""
        self.statistics.update({
            'seed_papers': len(self.seed_papers),
            'arxiv_mapped': len(self.mapped_seeds),
            'arxiv_unmapped': len(self.unmapped_seeds),
            'first_round_citing': len(self.forward_papers),
            'first_round_ancestor': len(self.backward_papers),
            'first_round_cocitation': len(self.cocitation_papers),
            # 修复：使用单独缓存的列表
            'second_round_citing': len(self.second_round_citing),
            'second_round_ancestor': len(self.second_round_ancestor),
            'recent_papers': len(self.recent_papers),
            'total_papers': len(self.papers),
            'total_edges': len(self.citation_edges),
            # 添加种子节点ID列表
            'seed_ids': [p['id'] for p in self.mapped_seeds]
        })

    def _print_summary(self):
        """打印最终统计摘要"""
        stats = self.statistics

        logger.info("\n" + "="*70)
        logger.info("8步检索流程统计摘要")
        logger.info("="*70)
        logger.info(f"种子论文")
        logger.info(f"  - 总种子数: {stats['seed_papers']}")
        logger.info(f"  - arXiv映射成功: {stats['arxiv_mapped']}")
        logger.info(f"  - arXiv映射失败: {stats['arxiv_unmapped']}")
        if stats.get('seed_ids'):
            logger.info(f"  - 种子节点ID: {stats['seed_ids'][:3]}{'...' if len(stats['seed_ids']) > 3 else ''}")
        if stats['manual_citations_built'] > 0:
            logger.info(f"  - 手动搜索补充: {stats['manual_citations_built']}")

        logger.info(f"第一轮滚雪球")
        logger.info(f"  - 正向子节点: {stats['first_round_citing']}")
        logger.info(f"  - 反向祖先: {stats['first_round_ancestor']}")
        logger.info(f"  - 共引论文: {stats['first_round_cocitation']}")

        if stats['second_round_enabled']:
            logger.info(f"第二轮滚雪球")
            logger.info(f"  - 正向子节点: {stats['second_round_citing']}")
            logger.info(f"  - 反向祖先: {stats['second_round_ancestor']}")

        logger.info(f"最新SOTA")
        logger.info(f"  - 最新论文: {stats['recent_papers']}")

        logger.info(f"最终结果")
        logger.info(f"  - 总论文数: {stats['total_papers']}")
        logger.info(f"  - 引用关系数: {stats['total_edges']}")

        if stats['total_papers'] > 0:
            avg_degree = stats['total_edges'] / stats['total_papers']
            logger.info(f"  - 平均连接度: {avg_degree:.2f}")

        logger.info("="*70)

    def get_statistics(self) -> Dict:
        """返回统计信息"""
        return self.statistics.copy()

    def get_papers(self) -> Dict[str, Dict]:
        """返回所有论文"""
        return self.papers.copy()

    def get_citation_edges(self) -> List[Tuple[str, str]]:
        """返回所有引用关系"""
        return self.citation_edges.copy()


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    # 初始化流程
    pipeline = PaperSearchPipeline(
        config_path='./config/config.yaml'
    )

    # 执行完整的8步检索
    result = pipeline.execute_full_pipeline(
        topic="Natural Language Processing",
        keywords=["transformer", "attention", "BERT", "GPT"],
        categories=["cs.CL", "cs.AI"]
    )

    # 输出结果
    print("\n" + "="*70)
    print("最终检索结果:")
    print("="*70)
    print(f"总论文数: {len(result['papers'])}")
    print(f"引用关系数: {len(result['citation_edges'])}")
    print(f"平均连接度: {len(result['citation_edges']) / len(result['papers']):.2f}")
    print("="*70)
