"""
arXiv种子论文检索模块
基于arXiv API和分类系统获取高质量种子论文

核心策略：
1. 使用arXiv Categories进行精准检索
2. 结合关键词（title, abstract）过滤
3. 限定时间范围（近3-5年）
4. 按相关性和引用量排序
"""

import arxiv
import logging
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from llm_config import LLMConfig, LLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# arXiv分类映射表（CS子领域）
ARXIV_CATEGORY_MAP = {
    "NLP": ["cs.CL"],  # Computation and Language
    "Machine Learning": ["cs.LG", "stat.ML"],  # Machine Learning
    "Computer Vision": ["cs.CV"],  # Computer Vision
    "Artificial Intelligence": ["cs.AI"],  # Artificial Intelligence
    "Robotics": ["cs.RO"],  # Robotics
    "Information Retrieval": ["cs.IR"],  # Information Retrieval
    "Neural Networks": ["cs.NE"],  # Neural and Evolutionary Computing
    "Cryptography": ["cs.CR"],  # Cryptography and Security
    "Software Engineering": ["cs.SE"],  # Software Engineering
    "Databases": ["cs.DB"],  # Databases
    "Distributed Computing": ["cs.DC"],  # Distributed Computing
    "Human-Computer Interaction": ["cs.HC"],  # Human-Computer Interaction
}


class ArxivSeedRetriever:
    """
    arXiv种子论文检索器
    专注于获取高质量的领域种子论文
    """

    def __init__(
        self,
        max_results_per_query: int = 50,
        years_back: int = 5,
        min_relevance_score: float = 0.4,  # 降低阈值,提高召回率
        llm_client: Optional[LLMClient] = None,
        use_llm_query_generation: bool = True,
        enable_semantic_expansion: bool = True,
        expansion_max_topics: int = 4,
        expansion_max_keywords: int = 8
    ):
        """
        初始化arXiv种子检索器

        Args:
            max_results_per_query: 每次查询的最大结果数
            years_back: 回溯年数（从当前年份往前）
            min_relevance_score: 最小相关性分数（0-1），默认0.4以提高召回率
            llm_client: LLM客户端（可选，用于智能生成查询）
            use_llm_query_generation: 是否使用LLM生成查询（默认True）
            enable_semantic_expansion: 是否启用语义扩展（默认True）
            expansion_max_topics: 最多扩展主题数（默认4）
            expansion_max_keywords: 最多扩展关键词数（默认8）
        """
        self.client = arxiv.Client()
        self.max_results_per_query = max_results_per_query
        self.years_back = years_back
        self.min_relevance_score = min_relevance_score
        self.llm_client = llm_client
        self.use_llm_query_generation = use_llm_query_generation
        self.enable_semantic_expansion = enable_semantic_expansion
        self.expansion_max_topics = expansion_max_topics
        self.expansion_max_keywords = expansion_max_keywords

        logger.info("arXiv种子检索器初始化完成")
        logger.info(f"  max_results_per_query={max_results_per_query}")
        logger.info(f"  years_back={years_back}")
        logger.info(f"  min_relevance_score={min_relevance_score}")
        logger.info(f"  use_llm_query_generation={use_llm_query_generation}")
        if use_llm_query_generation and llm_client:
            logger.info(f"  enable_semantic_expansion={enable_semantic_expansion}")
            if enable_semantic_expansion:
                logger.info(f"  expansion_max_topics={expansion_max_topics}")
                logger.info(f"  expansion_max_keywords={expansion_max_keywords}")

    def retrieve_seed_papers(
        self,
        topic: str,
        keywords: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        max_seeds: int = 100,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance
    ) -> List[Dict]:
        """
        检索种子论文（高质量核心论文）

        Args:
            topic: 主题名称（如 "Natural Language Processing"）
            keywords: 关键词列表（用于标题/摘要匹配）
            categories: arXiv分类列表（如 ["cs.CL", "cs.AI"]）
            max_seeds: 最大种子论文数量
            sort_by: 排序方式

        Returns:
            种子论文列表
        """
        logger.info(f"开始检索种子论文: topic='{topic}'")

        # 1. 自动推断categories
        if not categories:
            categories = self._infer_categories(topic)
            logger.info(f"自动推断arXiv分类: {categories}")

        # 2. 构建查询
        query = self._build_arxiv_query(
            topic=topic,
            keywords=keywords,
            categories=categories
        )
        logger.info(f"arXiv查询: {query}")

        # 3. 设置时间范围
        # 设定时间范围为1995年到2022年
        start_date = datetime(1995, 1, 1, tzinfo=timezone.utc)
        end_date = datetime.now(timezone.utc) # 直到现在
        logger.info(f"时间范围: >= {start_date.strftime('%Y-%m-%d')} 到 <= {end_date.strftime('%Y-%m-%d')}")

        # 4. 执行检索
        search = arxiv.Search(
            query=query,
            max_results=self.max_results_per_query,
            sort_by=sort_by,
            sort_order=arxiv.SortOrder.Descending
        )

        papers = []
        try:
            # 显式转换为列表,捕获网络异常
            results = list(self.client.results(search))
            logger.info(f"  成功获取 {len(results)} 条原始结果")
        except Exception as e:
            logger.error(f"  ❌ arXiv API请求失败: {e}")
            logger.warning("  提示: 国内访问arXiv可能需要代理,或稍后重试")
            return []

        for result in results:
            # 时间过滤
            if result.published < start_date or result.published > end_date:
                continue

            # 转换为标准格式
            paper = self._parse_arxiv_result(result)

            # 相关性过滤
            relevance_score = self._compute_relevance(paper, topic, keywords)
            paper['relevance_score'] = relevance_score

            if relevance_score >= self.min_relevance_score:
                papers.append(paper)
                logger.info(
                    f"  ✓ [{paper['year']}] {paper['title'][:60]}... "
                    f"(相关性: {relevance_score:.2f})"
                )

            if len(papers) >= max_seeds:
                break

        # 5. 按相关性排序
        papers.sort(key=lambda x: x['relevance_score'], reverse=True)

        logger.info(f"✅ 检索到 {len(papers)} 篇高质量种子论文")
        return papers[:max_seeds]

    def _infer_categories(self, topic: str) -> List[str]:
        """
        根据主题推断arXiv分类

        Args:
            topic: 主题名称

        Returns:
            推断的分类列表
        """
        topic_lower = topic.lower()

        # 尝试匹配预定义映射
        for key, cats in ARXIV_CATEGORY_MAP.items():
            if key.lower() in topic_lower:
                return cats

        # 关键词匹配
        if any(kw in topic_lower for kw in ["nlp", "language", "text", "translation"]):
            return ["cs.CL"]
        elif any(kw in topic_lower for kw in ["vision", "image", "video"]):
            return ["cs.CV"]
        elif any(kw in topic_lower for kw in ["learning", "neural", "deep"]):
            return ["cs.LG"]
        elif any(kw in topic_lower for kw in ["ai", "intelligence", "agent"]):
            return ["cs.AI"]

        # 默认返回通用CS分类
        return ["cs.AI", "cs.LG"]

    def _build_arxiv_query(
        self,
        topic: str,
        keywords: Optional[List[str]],
        categories: List[str]
    ) -> str:
        """
        构建arXiv查询字符串

        如果启用LLM且客户端可用,则使用LLM生成查询
        否则使用传统规则方法

        Args:
            topic: 主题
            keywords: 关键词列表
            categories: arXiv分类列表

        Returns:
            查询字符串
        """
        # 尝试使用LLM生成查询
        if self.use_llm_query_generation and self.llm_client:
            try:
                llm_query = self._generate_query_with_llm(topic, keywords, categories)
                if llm_query:
                    logger.info(f"✨ 使用LLM生成的查询: {llm_query}")
                    return llm_query
            except Exception as e:
                logger.warning(f"LLM查询生成失败，回退到传统方法: {e}")

        # 传统规则方法
        return self._build_arxiv_query_traditional(topic, keywords, categories)

    def _expand_semantic_concepts(
        self,
        topic: str,
        keywords: Optional[List[str]]
    ) -> Dict:
        """
        阶段1: 语义扩展
        使用LLM作为领域专家，扩展相关概念、同义词和子领域

        Args:
            topic: 研究主题
            keywords: 关键词列表(可选)

        Returns:
            扩展后的概念字典，格式:
            {
                'expanded_topics': [...],     # 相关主题
                'expanded_keywords': [...],   # 扩展关键词
                'synonyms': [...],            # 同义词
                'subfields': [...]            # 子领域
            }
        """
        logger.info(f"🔍 [阶段1] 语义扩展: topic='{topic}'")

        system_prompt = """You are a domain expert in computer science research.
Your task is to expand research topics and keywords by providing semantically
related concepts, synonyms, subfields, and alternative terminology.

Focus on:
- Computer Science, AI, and Machine Learning domains
- Academic and technical terminology
- Both broad and specific related concepts"""

        user_prompt = f"""Research Topic: {topic}
Current Keywords: {', '.join(keywords) if keywords else 'None'}

Please expand this research area by providing:
1. Related Topics: 2-{self.expansion_max_topics} semantically similar or overlapping research topics
2. Expanded Keywords: 5-{self.expansion_max_keywords} additional relevant technical terms, methods, or concepts
3. Synonyms: 2-4 alternative terms or abbreviations for the main topic
4. Subfields: 2-3 more specific subfields or applications within this area

Important:
- Focus on computer science and AI-related terms
- Use technical/academic terminology
- Keep each item concise (1-5 words)
- Avoid generic terms like "research", "study", "analysis"

Output ONLY valid JSON in this exact format (no markdown, no code blocks):
{{
  "expanded_topics": ["topic1", "topic2", ...],
  "expanded_keywords": ["keyword1", "keyword2", ...],
  "synonyms": ["synonym1", "synonym2", ...],
  "subfields": ["subfield1", "subfield2", ...]
}}"""

        try:
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=500
            )

            # 清理响应（移除可能的markdown代码块标记）
            response = response.strip()
            if response.startswith('```'):
                # 移除 ```json 和 ```
                lines = response.split('\n')
                response = '\n'.join(lines[1:-1]) if len(lines) > 2 else response
            response = response.strip()

            # 解析JSON
            expanded = json.loads(response)

            # 验证并限制数量
            expanded_topics = expanded.get('expanded_topics', [])[:self.expansion_max_topics]
            expanded_keywords = expanded.get('expanded_keywords', [])[:self.expansion_max_keywords]
            synonyms = expanded.get('synonyms', [])[:4]
            subfields = expanded.get('subfields', [])[:3]

            result = {
                'expanded_topics': expanded_topics,
                'expanded_keywords': expanded_keywords,
                'synonyms': synonyms,
                'subfields': subfields
            }

            # 输出扩展结果
            logger.info(f"  ✅ 语义扩展成功:")
            logger.info(f"    - 相关主题({len(expanded_topics)}): {', '.join(expanded_topics[:3])}{'...' if len(expanded_topics) > 3 else ''}")
            logger.info(f"    - 扩展关键词({len(expanded_keywords)}): {', '.join(expanded_keywords[:5])}{'...' if len(expanded_keywords) > 5 else ''}")
            logger.info(f"    - 同义词({len(synonyms)}): {', '.join(synonyms)}")
            logger.info(f"    - 子领域({len(subfields)}): {', '.join(subfields)}")

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"  ⚠️ JSON解析失败: {e}")
            logger.warning(f"  LLM原始响应: {response[:200]}...")
            return {}
        except Exception as e:
            logger.warning(f"  ⚠️ 语义扩展失败: {e}")
            return {}

    def _generate_query_with_llm(
        self,
        topic: str,
        keywords: Optional[List[str]],
        categories: List[str]
    ) -> Optional[str]:
        """
        使用LLM智能生成arXiv查询字符串（两阶段方法）

        如果启用语义扩展，则执行：
          阶段1: 语义扩展 - 作为领域专家扩展相关概念
          阶段2: 查询构建 - 作为图书管理员构建精确查询
        否则，直接生成查询（传统单阶段方法）

        Args:
            topic: 主题
            keywords: 关键词列表
            categories: arXiv分类列表

        Returns:
            LLM生成的查询字符串，失败则返回None
        """
        # 检查是否启用语义扩展
        if self.enable_semantic_expansion:
            logger.info("\n" + "="*70)
            logger.info("🚀 使用两阶段LLM查询生成（语义扩展 + 查询构建）")
            logger.info("="*70)

            # 阶段1: 语义扩展
            try:
                expanded = self._expand_semantic_concepts(topic, keywords)
            except Exception as e:
                logger.warning(f"⚠️ 语义扩展失败: {e}，使用原始输入")
                expanded = {}

            # 合并原始输入和扩展结果
            all_topics = [topic]
            if expanded:
                all_topics.extend(expanded.get('expanded_topics', []))
                all_topics.extend(expanded.get('synonyms', []))

            all_keywords = list(keywords) if keywords else []
            if expanded:
                all_keywords.extend(expanded.get('expanded_keywords', []))

            logger.info(f"\n📦 合并结果: {len(all_topics)} 个主题, {len(all_keywords)} 个关键词")

            # 阶段2: 查询构建
            try:
                query = self._construct_arxiv_query_with_llm(
                    original_topic=topic,
                    all_topics=all_topics,
                    all_keywords=all_keywords,
                    categories=categories
                )
                if query:
                    logger.info("="*70 + "\n")
                    return query
            except Exception as e:
                logger.warning(f"⚠️ 查询构建失败: {e}")

            logger.info("="*70 + "\n")
            return None

        else:
            # 传统单阶段方法（原有逻辑）
            logger.info("💡 使用单阶段LLM查询生成（传统方法）")

            # 构建prompt
            system_prompt = """You are an expert at constructing arXiv API queries.
Your task is to generate effective search queries that will find relevant academic papers.

arXiv Query Syntax Rules:
1. Categories: Use "cat:cs.AI" or "cat:cs.LG" format
2. Title search: Use "ti:keyword" (without quotes for flexible matching)
3. Abstract search: Use "abs:keyword" (without quotes for flexible matching)
4. Boolean operators: AND, OR, ANDNOT
5. Parentheses for grouping: (cat:cs.AI OR cat:cs.LG)
6. For multi-word phrases: Break into key terms or use without quotes for flexible matching

Important Tips:
- Avoid overly strict exact phrase matching (don't use quotes for long phrases)
- Extract core keywords from long phrases
- Use OR to connect related terms
- Balance between precision and recall"""

            user_prompt = f"""Generate an arXiv search query for the following research topic:

Topic: {topic}
Additional Keywords: {', '.join(keywords) if keywords else 'None'}
Target Categories: {', '.join(categories)}

Requirements:
1. Include category filters: {' OR '.join([f'cat:{cat}' for cat in categories])}
2. Extract 3-5 core keywords from the topic (ignore stopwords like 'for', 'the', 'with')
3. Use flexible matching (no quotes for multi-word terms)
4. Connect keywords with OR for broader coverage
5. Use AND to combine category filters with keyword filters

Output ONLY the final query string, no explanation."""

            try:
                response = self.llm_client.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.3,
                    max_tokens=200
                )

                # 清理响应（移除可能的多余文本）
                query = response.strip()
                # 移除可能的markdown代码块标记
                if query.startswith('```'):
                    query = '\n'.join(query.split('\n')[1:-1])
                query = query.strip()

                return query

            except Exception as e:
                logger.error(f"LLM查询生成出错: {e}")
                return None

    def _construct_arxiv_query_with_llm(
        self,
        original_topic: str,
        all_topics: List[str],
        all_keywords: List[str],
        categories: List[str]
    ) -> Optional[str]:
        """
        [优化版] 阶段2: 
        1. LLM 挑选 3-5 个最关键的检索词（短语）
        2. Python 代码自动将其包装为 (ti:"..." OR abs:"...") 格式
        这避免了 LLM 生成语法错误或遗漏 abs 标签
        """
        
        # 1. 构建 Prompt，只要求返回关键词列表
        system_prompt = """You are an expert arXiv search optimizer.
Your task is to select the 3-5 MOST CRITICAL search terms from a list of candidates.
Select terms that will maximize the retrieval of high-quality papers.

Rules:
1. Include the full topic name (e.g., "Natural Language Processing").
2. Include the most common acronym (e.g., "NLP").
3. Include 1-2 core technical synonyms (e.g., "Computational Linguistics").
4. Output specific phrases, not generic words.
5. Return ONLY a Python-style list of strings."""

        user_prompt = f"""Task: Select search terms for arXiv.

Original Topic: {original_topic}
Candidates: {', '.join((all_topics + all_keywords)[:20])}

Output Format: ["term1", "term2", "term3"]
Output ONLY the list."""

        try:
            # 2. 调用 LLM
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.1, # 降低随机性，就要最准的
                max_tokens=100
            )
            
            # 3. 解析 LLM 返回的列表字符串
            import ast
            cleaned_response = response.strip()
            # 处理可能的 markdown 标记
            if "```" in cleaned_response:
                cleaned_response = cleaned_response.split("```")[1].replace("json", "").replace("python", "").strip()
            
            try:
                # 安全地将字符串转为列表
                selected_terms = ast.literal_eval(cleaned_response)
                if not isinstance(selected_terms, list):
                    raise ValueError("Output is not a list")
            except:
                # 兜底：如果解析失败，简单的按逗号分割，或者直接使用原始 Topic
                logger.warning(f"解析LLM列表失败: {cleaned_response}, 回退到原始Topic")
                selected_terms = [original_topic]

            logger.info(f"🧠 LLM选定的核心词: {selected_terms}")

            # 4. [关键步骤] Python 负责严格构建语法
            # 格式: (cat:...) AND ((ti:"A" OR abs:"A") OR (ti:"B" OR abs:"B")...)
            
            # 4.1 构建分类部分
            cat_part = " OR ".join([f"cat:{cat}" for cat in categories])
            
            # 4.2 构建内容部分 (自动为每个词加上引号和双字段检索)
            content_parts = []
            for term in selected_terms:
                term = term.strip()
                if not term: continue
                # 强制加上引号，处理特殊字符
                safe_term = f'"{term}"' 
                # 生成 (ti:"term" OR abs:"term")
                part = f'(ti:{safe_term} OR abs:{safe_term})'
                content_parts.append(part)
            
            if not content_parts:
                content_parts = [f'(ti:"{original_topic}" OR abs:"{original_topic}")']

            content_query = " OR ".join(content_parts)
            
            # 5. 组合最终查询
            final_query = f"({cat_part}) AND ({content_query})"
            
            logger.info(f"✅ Python组装查询成功: {final_query}")
            return final_query

        except Exception as e:
            logger.error(f"❌ 查询构建过程出错: {e}")
            return None

    def _build_arxiv_query_traditional(
        self,
        topic: str,
        keywords: Optional[List[str]],
        categories: List[str]
    ) -> str:
        """
        传统规则方法构建arXiv查询字符串

        优化策略:
        1. 去掉引号,使用宽松匹配 (ti:keyword 而非 ti:"keyword")
        2. 使用 OR 连接 topic 和 keywords (提高召回率)
        3. 依赖后续的 _compute_relevance 进行精细过滤

        Args:
            topic: 主题
            keywords: 关键词列表
            categories: arXiv分类列表

        Returns:
            查询字符串
        """
        query_parts = []

        # 添加分类约束
        if categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            query_parts.append(f"({cat_query})")

        # 构建内容查询: topic OR keywords (宽松匹配)
        content_parts = []

        # 添加主题 (不加引号,宽松匹配)
        if topic:
            content_parts.append(f"ti:{topic}")
            content_parts.append(f"abs:{topic}")

        # 添加关键词 (不加引号,宽松匹配)
        if keywords:
            for kw in keywords:
                content_parts.append(f"ti:{kw}")
                content_parts.append(f"abs:{kw}")

        # 使用 OR 连接所有内容部分
        if content_parts:
            content_query = " OR ".join(content_parts)
            query_parts.append(f"({content_query})")

        # 使用AND连接分类和关键词
        return " AND ".join(query_parts)

    def _parse_arxiv_result(self, result: arxiv.Result) -> Dict:
        """
        解析arXiv结果为标准格式

        Args:
            result: arxiv.Result对象

        Returns:
            标准论文字典
        """
        return {
            'arxiv_id': result.get_short_id(),
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'abstract': result.summary,
            'year': result.published.year,
            'published_date': result.published,
            'updated_date': result.updated,
            'categories': result.categories,
            'primary_category': result.primary_category,
            'pdf_url': result.pdf_url,
            'doi': result.doi,
            'comment': result.comment,
            'journal_ref': result.journal_ref,
            # 用于后续映射
            'source': 'arxiv',
            'openalex_id': None  # 待映射
        }

    def _compute_relevance(
        self,
        paper: Dict,
        topic: str,
        keywords: Optional[List[str]]
    ) -> float:
        """
        计算论文与主题的相关性分数

        Args:
            paper: 论文数据
            topic: 主题
            keywords: 关键词列表

        Returns:
            相关性分数（0-1）
        """
        text = (paper['title'] + ' ' + paper['abstract']).lower()
        topic_lower = topic.lower()

        score = 0.0

        # 1. 主题匹配（权重: 0.4）
        # 改进：拆分主题为单词，计算词汇覆盖率
        topic_words = [w for w in topic_lower.split() if len(w) > 2]  # 过滤短词
        if topic_words:
            matched_topic_words = sum(1 for word in topic_words if word in text)
            topic_coverage = matched_topic_words / len(topic_words)
            score += 0.4 * topic_coverage
        else:
            # 如果主题为空或太短，检查完整匹配
            if topic_lower in text:
                score += 0.4

        # 2. 关键词匹配（权重: 0.4）
        if keywords:
            matched_keywords = sum(1 for kw in keywords if kw.lower() in text)
            score += 0.4 * (matched_keywords / len(keywords))

        # 3. 分类匹配（权重: 0.2）
        primary_cat = paper.get('primary_category', '')
        if primary_cat.startswith('cs.'):
            score += 0.2

        return min(score, 1.0)

    def retrieve_recent_papers(
        self,
        topic: str,
        keywords: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        max_results: int = 20,
        months_back: int = 12
    ) -> List[Dict]:
        """
        检索最新前沿论文（用于步骤4：补充SOTA）

        Args:
            topic: 主题
            keywords: 关键词列表
            categories: arXiv分类列表
            max_results: 最大结果数
            months_back: 回溯月数

        Returns:
            最新论文列表
        """
        logger.info(f"检索最新论文: topic='{topic}', 回溯{months_back}个月")

        # 自动推断categories
        if not categories:
            categories = self._infer_categories(topic)

        # 构建查询
        query = self._build_arxiv_query(topic, keywords, categories)

        # 设置时间范围
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=30 * months_back)

        # 执行检索（按提交日期排序）
        search = arxiv.Search(
            query=query,
            max_results=max_results * 2,  # 多取一些，后续过滤
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )

        papers = []
        try:
            # 显式转换为列表,捕获网络异常
            results = list(self.client.results(search))
            logger.info(f"  成功获取 {len(results)} 条最新论文原始结果")
        except Exception as e:
            logger.error(f"  ❌ arXiv API请求失败: {e}")
            logger.warning("  提示: 国内访问arXiv可能需要代理,或稍后重试")
            return []

        for result in results:
            # 只要最新的
            if result.published < cutoff_date:
                continue

            paper = self._parse_arxiv_result(result)
            relevance_score = self._compute_relevance(paper, topic, keywords)
            paper['relevance_score'] = relevance_score

            if relevance_score >= self.min_relevance_score:
                papers.append(paper)
                logger.info(
                    f"  ✓ [{paper['published_date'].strftime('%Y-%m')}] "
                    f"{paper['title'][:60]}... (相关性: {relevance_score:.2f})"
                )

            if len(papers) >= max_results:
                break

        logger.info(f"✅ 检索到 {len(papers)} 篇最新论文")
        return papers


if __name__ == "__main__":
    # 测试代码
    retriever = ArxivSeedRetriever(
        max_results_per_query=50,
        years_back=5,
        min_relevance_score=0.6
    )

    # 示例1: 检索NLP种子论文
    print("=" * 80)
    print("示例1: 检索NLP种子论文")
    print("=" * 80)
    seeds = retriever.retrieve_seed_papers(
        topic="Natural Language Processing",
        keywords=["transformer", "attention", "language model"],
        max_seeds=10
    )

    for i, paper in enumerate(seeds[:5], 1):
        print(f"\n[{i}] {paper['title']}")
        print(f"    arXiv ID: {paper['arxiv_id']}")
        print(f"    年份: {paper['year']}")
        print(f"    分类: {paper['primary_category']}")
        print(f"    相关性: {paper['relevance_score']:.2f}")

    # 示例2: 检索最新论文
    print("\n" + "=" * 80)
    print("示例2: 检索最新论文（近6个月）")
    print("=" * 80)
    recent = retriever.retrieve_recent_papers(
        topic="Large Language Models",
        keywords=["reasoning", "chain of thought"],
        max_results=10,
        months_back=6
    )

    for i, paper in enumerate(recent[:5], 1):
        print(f"\n[{i}] {paper['title']}")
        print(f"    arXiv ID: {paper['arxiv_id']}")
        print(f"    发布日期: {paper['published_date'].strftime('%Y-%m-%d')}")
        print(f"    相关性: {paper['relevance_score']:.2f}")
