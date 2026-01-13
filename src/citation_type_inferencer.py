"""
Socket Matching 引用关系类型推断器
基于深度论文信息的"接口对接"方法推断引用关系的语义类型

核心思想：
将提取的论文深度信息（Problem, Method, Limitation, Future_Work）作为"接口（Socket）"，
通过 LLM Agent 判断这些接口是否能够对接，从而推断引用关系的语义类型。

支持的关系类型（Socket Matching - 6种）：（Overcomes、Realizes、Extends、Alternative、Adapts_to、Baselines）
1. Overcomes - 攻克/优化（纵向深化）
   来源：Match 1 (Limitation→Problem)
2. Realizes - 实现愿景（科研传承）
   来源：Match 2 (Future_Work→Problem)
3. Extends - 方法扩展（微创新）
   来源：Match 3 Extension
4. Alternative - 另辟蹊径（颠覆创新）
   来源：Match 3 Alternative
5. Adapts_to - 技术迁移（横向扩散）
   来源：Match 4 (Problem→Problem 跨域)
6. Baselines - 基线对比（背景噪音）
   来源：无匹配

逻辑对接矩阵（4个Match → 6种类型）：
- Match 1: A.Limitation ↔ B.Problem → Overcomes
- Match 2: A.Future_Work ↔ B.Problem → Realizes
- Match 3: (Problem一致)A.Method ↔ B.Method → Extends(Extension) / Alternative
- Match 4: A.Problem ↔ B.Problem(跨域) → Adapts_to
- 无匹配 → Baselines
"""

import json
import logging
import os
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# 导入LLM配置模块
try:
    from llm_config import create_llm_client
except ImportError:
    create_llm_client = None

logger = logging.getLogger(__name__)


@dataclass
class SocketMatchResult:
    """Socket 匹配结果"""
    match_type: str  # "limitation_problem", "future_work_problem", "method_extension", "problem_adaptation"
    is_match: bool
    confidence: float
    reasoning: str
    evidence: str
    additional_info: Dict = None  # 额外信息（如 relationship_type, source_domain, target_domain）


@dataclass
class CitationRelationship:
    """引用关系"""
    citing_id: str
    cited_id: str
    relationship_type: str  # Overcomes, Realizes, Extends, Alternative, Adapts_to, Baselines (6种类型)
    confidence: float
    reasoning: str
    evidence: str
    match_results: List[SocketMatchResult]


class CitationTypeInferencer:
    """
    Socket Matching 引用关系类型推断器

    使用 LLM Agent 进行深度语义分析，通过"接口对接"的方式判断引用关系类型
    """

    def __init__(self, llm_client=None, config_path: str = None, prompts_dir: str = "./prompts"):
        """
        初始化推断器

        Args:
            llm_client: LLM客户端（如果为None则使用基于规则的方法）
            config_path: LLM配置文件路径（如果提供且llm_client为None，则从此文件加载）
            prompts_dir: 提示词目录
        """
        # 如果提供了config_path但没有提供llm_client，尝试从配置文件加载
        if llm_client is None and config_path:
            if create_llm_client is None:
                logger.warning("无法导入create_llm_client，将使用规则方法")
                self.llm_client = None
            else:
                try:
                    config_file = Path(config_path)
                    if config_file.exists():
                        self.llm_client = create_llm_client(str(config_file))
                        logger.info(f"✅ 从配置文件加载LLM客户端: {config_path}")
                    else:
                        logger.warning(f"配置文件不存在: {config_path}，将使用规则方法")
                        self.llm_client = None
                except Exception as e:
                    logger.warning(f"加载LLM客户端失败: {e}，将使用规则方法")
                    self.llm_client = None
        else:
            self.llm_client = llm_client

        self.prompts_dir = Path(prompts_dir)
        self.prompts_cache = {}

        # 加载提示词
        self._load_prompts()

        # 关系类型优先级（用于规则方法和冲突解决）
        self.relationship_priority = {
            "Overcomes": 6,     # 最高优先级 - 直接解决问题
            "Realizes": 5,      # 次高优先级 - 实现愿景
            "Adapts_to": 4,     # 高优先级 - 技术迁移
            "Extends": 3,       # 中高优先级 - 方法扩展
            "Alternative": 2,   # 中优先级 - 另辟蹊径
            "Baselines": 1      # 最低优先级 - 基线对比
        }

        logger.info("CitationTypeInferencer 初始化完成")
        if self.llm_client:
            logger.info("  模式: LLM Socket Matching")
        else:
            logger.info("  模式: 基于规则的方法（降级模式）")

    def _load_prompts(self):
        """加载所有提示词"""
        prompt_files = {
            'match_limitation_problem': 'match_limitation_problem.txt',
            'match_future_work_problem': 'match_future_work_problem.txt',
            'match_method_extension': 'match_method_extension.txt',
            'match_problem_adaptation': 'match_problem_adaptation.txt',
        }

        for key, filename in prompt_files.items():
            file_path = self.prompts_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.prompts_cache[key] = f.read().strip()
                    logger.debug(f"  加载提示词: {key}")
                except Exception as e:
                    logger.warning(f"  加载提示词失败 ({key}): {e}")
            else:
                logger.warning(f"  提示词文件不存在: {filename}")

        logger.info(f"加载 {len(self.prompts_cache)} 个提示词模板")

    def infer_edge_types(
        self,
        papers: List[Dict],
        citation_edges: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str, str]], Dict[str, int]]:
        """
        批量推断引用关系类型

        Args:
            papers: 论文列表（必须包含 rag_analysis 或 deep_analysis）
            citation_edges: 引用关系列表 [(citing_id, cited_id), ...]

        Returns:
            (typed_edges, statistics)
            - typed_edges: 带类型的引用关系 [(citing_id, cited_id, edge_type), ...]
            - statistics: 类型统计 {edge_type: count}
        """
        logger.info(f"开始推断 {len(citation_edges)} 条引用关系的类型...")

        # 构建论文字典
        papers_dict = {paper['id']: paper for paper in papers}

        # 推断每条边的类型
        typed_edges = []
        statistics = {}
        relationships = []

        for i, (citing_id, cited_id) in enumerate(citation_edges):
            logger.info(f"处理引用关系 {i+1}/{len(citation_edges)}: {citing_id} -> {cited_id}")

            if citing_id in papers_dict and cited_id in papers_dict:
                relationship = self.infer_single_edge_type(
                    papers_dict[citing_id],
                    papers_dict[cited_id]
                )
                relationships.append(relationship)
                edge_type = relationship.relationship_type
            else:
                # 论文不在字典中，使用默认类型
                edge_type = "Baselines"
                logger.warning(f"  论文不在字典中，使用默认类型: {edge_type}")

            typed_edges.append((citing_id, cited_id, edge_type))
            statistics[edge_type] = statistics.get(edge_type, 0) + 1

        logger.info(f"✅ 引用类型推断完成")
        logger.info(f"  总引用关系: {len(typed_edges)} 条")
        logger.info(f"\n📊 引用类型分布:")
        for edge_type, count in sorted(statistics.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(typed_edges)) * 100 if typed_edges else 0
            logger.info(f"  • {edge_type}: {count} 条 ({percentage:.1f}%)")

        return typed_edges, statistics

    def infer_single_edge_type(
        self,
        citing_paper: Dict,
        cited_paper: Dict
    ) -> CitationRelationship:
        """
        推断单条引用关系的类型（Socket Matching）

        Args:
            citing_paper: 引用论文（Paper B）
            cited_paper: 被引用论文（Paper A）

        Returns:
            CitationRelationship 对象
        """
        # 提取深度分析信息
        citing_analysis = self._extract_deep_analysis(citing_paper)
        cited_analysis = self._extract_deep_analysis(cited_paper)

        # 提取引用上下文
        citation_context = self._extract_citation_context(citing_paper, cited_paper)

        # 如果没有LLM客户端，使用基于规则的方法
        if not self.llm_client:
            return self._rule_based_inference(
                citing_paper, cited_paper, citing_analysis, cited_analysis
            )

        # Socket Matching: 执行4个匹配检测
        match_results = []

        # Match 1: A.Limitation ↔ B.Problem
        if cited_analysis.get('limitation') and citing_analysis.get('problem'):
            match = self._check_limitation_problem_match(
                cited_paper, citing_paper, cited_analysis, citing_analysis, citation_context
            )
            if match:
                match_results.append(match)

        # Match 2: A.Future_Work ↔ B.Problem
        if cited_analysis.get('future_work') and citing_analysis.get('problem'):
            match = self._check_future_work_problem_match(
                cited_paper, citing_paper, cited_analysis, citing_analysis, citation_context
            )
            if match:
                match_results.append(match)

        # Match 3: A.Method ↔ B.Method
        if cited_analysis.get('method') and citing_analysis.get('method'):
            match = self._check_method_extension_match(
                cited_paper, citing_paper, cited_analysis, citing_analysis, citation_context
            )
            if match:
                match_results.append(match)

        # Match 4: A.Problem ↔ B.Problem (但场景不同)
        if cited_analysis.get('problem') and citing_analysis.get('problem'):
            match = self._check_problem_adaptation_match(
                cited_paper, citing_paper, cited_analysis, citing_analysis, citation_context
            )
            if match:
                match_results.append(match)

        # 综合所有匹配结果，最终分类
        relationship = self._classify_relationship(
            citing_paper, cited_paper, match_results, citation_context
        )

        return relationship

    def _check_limitation_problem_match(
        self,
        cited_paper: Dict,
        citing_paper: Dict,
        cited_analysis: Dict,
        citing_analysis: Dict,
        citation_context: str
    ) -> Optional[SocketMatchResult]:
        """
        Match 1: 检查 A.Limitation ↔ B.Problem
        判断 B 是否解决了 A 的局限性
        """
        prompt_template = self.prompts_cache.get('match_limitation_problem')
        if not prompt_template:
            logger.warning("缺少 match_limitation_problem 提示词，跳过")
            return None

        # 填充提示词
        prompt = prompt_template.format(
            cited_title=cited_paper.get('title', 'Unknown'),
            cited_limitation=cited_analysis.get('limitation', 'N/A'),
            citing_title=citing_paper.get('title', 'Unknown'),
            citing_problem=citing_analysis.get('problem', 'N/A'),
            citation_context=citation_context
        )

        # 调用LLM
        try:
            response = self.llm_client.generate(prompt)

            # 提取JSON内容
            json_str = self._extract_json_from_response(response)
            result = json.loads(json_str)

            return SocketMatchResult(
                match_type="limitation_problem",
                is_match=result.get('is_match', False),
                confidence=result.get('confidence', 0.0),
                reasoning=result.get('reasoning', ''),
                evidence=result.get('evidence', '')
            )
        except Exception as e:
            logger.error(f"Match 1 (Limitation-Problem) 失败: {e}")
            return None

    def _check_future_work_problem_match(
        self,
        cited_paper: Dict,
        citing_paper: Dict,
        cited_analysis: Dict,
        citing_analysis: Dict,
        citation_context: str
    ) -> Optional[SocketMatchResult]:
        """
        Match 2: 检查 A.Future_Work ↔ B.Problem
        判断 B 是否实现了 A 的未来工作建议
        """
        # 如果 A 的 Future Work 提取为空，或者太短，直接跳过 Match 2
        future_work = cited_analysis.get('future_work', '')
        if not future_work or len(future_work) < 5 or future_work == "N/A":
            logger.info("    → Match 2 跳过: A的Future Work为空或过短")
            return None

        prompt_template = self.prompts_cache.get('match_future_work_problem')
        if not prompt_template:
            logger.warning("缺少 match_future_work_problem 提示词，跳过")
            return None

        prompt = prompt_template.format(
            cited_title=cited_paper.get('title', 'Unknown'),
            cited_year=cited_paper.get('year', 'N/A'),
            cited_future_work=future_work,
            citing_title=citing_paper.get('title', 'Unknown'),
            citing_year=citing_paper.get('year', 'N/A'),
            citing_problem=citing_analysis.get('problem', 'N/A'),
            citation_context=citation_context
        )

        try:
            response = self.llm_client.generate(prompt)
            json_str = self._extract_json_from_response(response)
            result = json.loads(json_str)

            # 双重过滤：区分真传承(Realizes) vs 假客套(Extends/Baselines)
            is_match = result.get('is_match', False)
            specificity = result.get('specificity', 'low')
            confidence = result.get('confidence', 0.0)

            # 场景1: LLM认为匹配 + 建议很具体(high specificity) → 真正的Realizes
            if is_match and specificity == "high" and confidence > 0.6:
                logger.info(f"    → Match 2 具体性检查: ✓ 高具体性 (specificity=high, conf={confidence:.2f})")
                return SocketMatchResult(
                    match_type="future_work_problem",
                    is_match=True,
                    confidence=confidence,
                    reasoning=result.get('reasoning', ''),
                    evidence=result.get('evidence', ''),
                    additional_info={'specificity': 'high'}
                )

            # 场景2: LLM认为匹配 + 但建议很宽泛(low specificity) → 假客套,降级
            elif is_match and specificity == "low":
                logger.info(f"    → Match 2 具体性检查: ✗ 低具体性 (specificity=low, conf={confidence:.2f}) - 疑似客套话,不计入Realizes")
                return SocketMatchResult(
                    match_type="future_work_problem",
                    is_match=False,  # 强制标记为不匹配
                    confidence=0.0,
                    reasoning=f"[过滤] A的Future Work过于宽泛,不符合Realizes标准。{result.get('reasoning', '')}",
                    evidence=result.get('evidence', ''),
                    additional_info={'specificity': 'low', 'filtered': True}
                )

            # 场景3: LLM认为不匹配
            else:
                logger.info(f"    → Match 2: 不匹配 (is_match=False)")
                return SocketMatchResult(
                    match_type="future_work_problem",
                    is_match=False,
                    confidence=0.0,
                    reasoning=result.get('reasoning', ''),
                    evidence=result.get('evidence', '')
                )
        except Exception as e:
            logger.error(f"Match 2 (FutureWork-Problem) 失败: {e}")
            return None

    def _check_method_extension_match(
        self,
        cited_paper: Dict,
        citing_paper: Dict,
        cited_analysis: Dict,
        citing_analysis: Dict,
        citation_context: str
    ) -> Optional[SocketMatchResult]:
        """
        Match 3: 检查 A.Method ↔ B.Method
        判断是 Extension（扩展）还是 Alternative（另辟蹊径）
        """
        prompt_template = self.prompts_cache.get('match_method_extension')
        if not prompt_template:
            logger.warning("缺少 match_method_extension 提示词，跳过")
            return None

        prompt = prompt_template.format(
            cited_title=cited_paper.get('title', 'Unknown'),
            cited_year=cited_paper.get('year', 'N/A'),
            cited_problem=cited_analysis.get('problem', 'N/A'),
            cited_method=cited_analysis.get('method', 'N/A'),
            citing_title=citing_paper.get('title', 'Unknown'),
            citing_year=citing_paper.get('year', 'N/A'),
            citing_problem=citing_analysis.get('problem', 'N/A'),
            citing_method=citing_analysis.get('method', 'N/A'),
            citation_context=citation_context
        )

        try:
            response = self.llm_client.generate(prompt)
            json_str = self._extract_json_from_response(response)
            result = json.loads(json_str)

            return SocketMatchResult(
                match_type="method_extension",
                is_match=(result.get('relationship_type') != 'none'),
                confidence=result.get('confidence', 0.0),
                reasoning=result.get('reasoning', ''),
                evidence=result.get('evidence', ''),
                additional_info={'relationship_type': result.get('relationship_type', 'none')}
            )
        except Exception as e:
            logger.error(f"Match 3 (Method Extension) 失败: {e}")
            return None

    def _check_problem_adaptation_match(
        self,
        cited_paper: Dict,
        citing_paper: Dict,
        cited_analysis: Dict,
        citing_analysis: Dict,
        citation_context: str
    ) -> Optional[SocketMatchResult]:
        """
        Match 4: 检查 A.Problem ↔ B.Problem (但场景不同)
        判断是否是技术迁移/泛化
        """
        prompt_template = self.prompts_cache.get('match_problem_adaptation')
        if not prompt_template:
            logger.warning("缺少 match_problem_adaptation 提示词，跳过")
            return None

        prompt = prompt_template.format(
            cited_title=cited_paper.get('title', 'Unknown'),
            cited_year=cited_paper.get('year', 'N/A'),
            cited_problem=cited_analysis.get('problem', 'N/A'),
            cited_method=cited_analysis.get('method', 'N/A'),
            citing_title=citing_paper.get('title', 'Unknown'),
            citing_year=citing_paper.get('year', 'N/A'),
            citing_problem=citing_analysis.get('problem', 'N/A'),
            citing_method=citing_analysis.get('method', 'N/A'),
            citation_context=citation_context
        )

        try:
            response = self.llm_client.generate(prompt)
            json_str = self._extract_json_from_response(response)
            result = json.loads(json_str)

            # 双重过滤：区分真跨域迁移(Adapts_to) vs 换数据集(Extends)
            is_adaptation = result.get('is_adaptation', False)
            domain_shift_type = result.get('domain_shift_type', 'none')
            confidence = result.get('confidence', 0.0)

            # 场景1: 真正的跨域迁移 (cross-task/cross-modality) → Adapts_to
            if is_adaptation and domain_shift_type in ['cross-task', 'cross-modality']:
                logger.info(f"    → Match 4 领域跨度检查: ✓ 真跨域迁移 (type={domain_shift_type}, conf={confidence:.2f})")
                return SocketMatchResult(
                    match_type="problem_adaptation",
                    is_match=True,
                    confidence=confidence,
                    reasoning=result.get('reasoning', ''),
                    evidence=result.get('evidence', ''),
                    additional_info={
                        'source_domain': result.get('source_domain', ''),
                        'target_domain': result.get('target_domain', ''),
                        'domain_shift_type': domain_shift_type
                    }
                )

            # 场景2: 只是换数据集 (same-task-new-data) → 不算Adapts_to,降级
            elif is_adaptation and domain_shift_type == 'same-task-new-data':
                logger.info(f"    → Match 4 领域跨度检查: ✗ 仅换数据集 (type={domain_shift_type}, conf={confidence:.2f}) - 不算真正的Adapts_to")
                return SocketMatchResult(
                    match_type="problem_adaptation",
                    is_match=False,  # 强制标记为不匹配
                    confidence=0.0,
                    reasoning=f"[过滤] 仅在同类任务上更换数据集,不符合真正的领域迁移。{result.get('reasoning', '')}",
                    evidence=result.get('evidence', ''),
                    additional_info={
                        'domain_shift_type': domain_shift_type,
                        'filtered': True
                    }
                )

            # 场景3: 不是adaptation或domain_shift_type为none
            else:
                logger.info(f"    → Match 4: 不匹配 (is_adaptation={is_adaptation}, type={domain_shift_type})")
                return SocketMatchResult(
                    match_type="problem_adaptation",
                    is_match=False,
                    confidence=0.0,
                    reasoning=result.get('reasoning', ''),
                    evidence=result.get('evidence', '')
                )
        except Exception as e:
            logger.error(f"Match 4 (Problem Adaptation) 失败: {e}")
            return None

    def _classify_relationship(
        self,
        citing_paper: Dict,
        cited_paper: Dict,
        match_results: List[SocketMatchResult],
        citation_context: str  # 保留以备未来使用
    ) -> CitationRelationship:
        """
        综合所有匹配结果，最终分类关系类型
        使用基于优先级的决策树逻辑（不再依赖LLM）

        决策树逻辑（按优先级）：
        1. Match 1 (Limitation→Problem) 成功 → Overcomes
        2. Match 2 (Future_Work→Problem) 成功 → Realizes
        3. Match 4 (Problem→Problem 跨域) 成功 → Adapts_to
        4. Match 3 (Method→Method) 成功:
           - extension → Extends
           - alternative → Alternative
           - none → Baselines
        5. 无任何匹配 → Baselines

        优先级排序：Overcomes > Realizes > Adapts_to > Extends > Alternative > Baselines
        """
        # 如果没有匹配结果，默认为 Baselines
        if not match_results:
            logger.info("  无匹配结果 -> Baselines")
            return CitationRelationship(
                citing_id=citing_paper['id'],
                cited_id=cited_paper['id'],
                relationship_type="Baselines",
                confidence=0.3,
                reasoning="无明确的深度关系，仅作为基线对比",
                evidence="",
                match_results=[]
            )

        # 将匹配结果按类型组织
        matches_by_type = {}
        for match in match_results:
            if match.is_match:
                matches_by_type[match.match_type] = match

        # 按优先级顺序检查匹配结果
        relationship_type = "Baselines"
        confidence = 0.3
        reasoning = "无明确的深度关系，仅作为基线对比"
        evidence = ""
        relationship_decided = False  # 标记是否已确定关系类型

        # 优先级1: Match 1 (Limitation→Problem) → Overcomes
        if not relationship_decided and "limitation_problem" in matches_by_type:
            match = matches_by_type["limitation_problem"]
            relationship_type = "Overcomes"
            confidence = match.confidence
            reasoning = f"B解决了A的局限性。{match.reasoning}"
            evidence = match.evidence
            relationship_decided = True
            logger.info(f"  ✓ Match 1 (Limitation→Problem) 匹配成功 -> Overcomes (置信度: {confidence:.2f})")

        # 优先级2: Match 2 (Future_Work→Problem) → Realizes
        # 特别注意：必须是高具体性的future work，不能是客套话
        if not relationship_decided and "future_work_problem" in matches_by_type:
            match = matches_by_type["future_work_problem"]

            # 双重验证：检查specificity
            specificity = match.additional_info.get('specificity', 'low') if match.additional_info else 'low'

            if specificity == "high" and match.confidence > 0.6:
                # 真正的Realizes：A挖坑 B填坑
                relationship_type = "Realizes"
                confidence = match.confidence
                reasoning = f"B实现了A设想的具体未来工作方向。{match.reasoning}"
                evidence = match.evidence
                relationship_decided = True
                logger.info(f"  ✓ Match 2 (Future_Work→Problem) 匹配成功 -> Realizes (置信度: {confidence:.2f}) [高具体性]")
            else:
                # 低具体性或低置信度：不算Realizes，继续检查其他Match
                logger.info(f"  ⚠ Match 2 检测到但具体性不足 (specificity={specificity}, conf={match.confidence:.2f}) - 跳过Realizes，检查其他Match")

        # 优先级3: Match 4 (Problem→Problem 跨域) → Adapts_to
        # 特别注意：必须是真正的跨任务/跨模态，不能只是换数据集
        if not relationship_decided and "problem_adaptation" in matches_by_type:
            match = matches_by_type["problem_adaptation"]

            # 双重验证：检查domain_shift_type
            domain_shift_type = match.additional_info.get('domain_shift_type', 'none') if match.additional_info else 'none'

            if domain_shift_type in ['cross-task', 'cross-modality']:
                # 真正的跨域迁移：技术横向扩散
                relationship_type = "Adapts_to"
                confidence = match.confidence
                source_domain = match.additional_info.get('source_domain', '') if match.additional_info else ''
                target_domain = match.additional_info.get('target_domain', '') if match.additional_info else ''
                reasoning = f"B将A的方法迁移到不同领域（{source_domain} → {target_domain}，{domain_shift_type}）。{match.reasoning}"
                evidence = match.evidence
                relationship_decided = True
                logger.info(f"  ✓ Match 4 (Problem→Problem 跨域) 匹配成功 -> Adapts_to (置信度: {confidence:.2f}) [{domain_shift_type}]")
            else:
                # 仅换数据集或无明显跨度：不算Adapts_to
                logger.info(f"  ⚠ Match 4 检测到但领域跨度不足 (type={domain_shift_type}, conf={match.confidence:.2f}) - 跳过Adapts_to")

        # 优先级4-5: Match 3 (Method→Method) → Extends / Alternative
        if not relationship_decided and "method_extension" in matches_by_type:
            match = matches_by_type["method_extension"]
            rel_type = match.additional_info.get('relationship_type', 'none') if match.additional_info else 'none'

            if rel_type == "extension":
                relationship_type = "Extends"
                confidence = match.confidence
                reasoning = f"B在A的方法基础上做了增量改进。{match.reasoning}"
                evidence = match.evidence
                relationship_decided = True
                logger.info(f"  ✓ Match 3 (Method Extension) 匹配成功 -> Extends (置信度: {confidence:.2f})")

            elif rel_type == "alternative":
                relationship_type = "Alternative"
                confidence = match.confidence
                reasoning = f"B使用不同范式解决类似问题。{match.reasoning}"
                evidence = match.evidence
                relationship_decided = True
                logger.info(f"  ✓ Match 3 (Method Alternative) 匹配成功 -> Alternative (置信度: {confidence:.2f})")

            else:  # rel_type == "none"
                relationship_type = "Baselines"
                confidence = 0.4
                reasoning = "方法之间无明确继承或改进关系，仅作为基线对比"
                evidence = match.evidence
                relationship_decided = True
                logger.info(f"  ✓ Match 3 (Method None) -> Baselines (置信度: {confidence:.2f})")

        # 优先级6: 无有效匹配 → Baselines
        if not relationship_decided:
            logger.info("  所有匹配均未成功 -> Baselines")

        return CitationRelationship(
            citing_id=citing_paper['id'],
            cited_id=cited_paper['id'],
            relationship_type=relationship_type,
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            match_results=match_results
        )

    def _extract_deep_analysis(self, paper: Dict) -> Dict:
        """
        提取论文的深度分析信息
        优先级：deep_analysis > rag_analysis > 空字典
        """
        if 'deep_analysis' in paper:
            return paper['deep_analysis']
        elif 'rag_analysis' in paper:
            return paper['rag_analysis']
        else:
            return {}

    def _extract_citation_context(self, citing_paper: Dict, cited_paper: Dict) -> str:
        """
        提取引用上下文 - 从PDF文件中提取引用A的具体句子

        优先级:
        1. 从PDF文件中提取引用上下文(基于作者名和年份匹配)
        2. 如果PDF不可用或提取失败，返回简单描述

        Args:
            citing_paper: 引用论文(Paper B)
            cited_paper: 被引论文(Paper A)

        Returns:
            引用上下文字符串，可能包含多个引用点
        """
        # 尝试从PDF提取引用上下文
        try:
            contexts = self._extract_citation_from_pdf(citing_paper, cited_paper)
            if contexts:
                # 返回前3个引用上下文
                context_str = " | ".join([
                    f"[p.{ctx['page']}] {ctx['context']}"
                    for ctx in contexts[:3]
                ])
                logger.debug(f"从PDF提取到 {len(contexts)} 个引用上下文")
                return context_str
        except Exception as e:
            logger.warning(f"从PDF提取引用上下文失败: {e}")

        # 降级方案：返回简单描述
        return f"{citing_paper.get('title', 'Paper B')} 引用了 {cited_paper.get('title', 'Paper A')}"

    def _extract_citation_from_pdf(self, citing_paper: Dict, cited_paper: Dict) -> List[Dict]:
        """
        从PDF文件中提取引用上下文

        策略:
        1. 定位PDF文件路径
        2. 提取PDF全文
        3. 使用引用模式匹配(基于作者名和年份)
        4. 提取引用前后的上下文

        Args:
            citing_paper: 引用论文
            cited_paper: 被引论文

        Returns:
            引用上下文列表，每个元素包含 {'page': int, 'context': str}
        """
        # 1. 定位PDF文件
        pdf_path = self._get_pdf_path(citing_paper)
        if not pdf_path or not os.path.exists(pdf_path):
            logger.debug(f"PDF文件不存在: {pdf_path}")
            return []

        # 2. 获取被引论文的识别信息
        cited_info = self._extract_citation_identifiers(cited_paper)
        if not cited_info:
            logger.debug(f"无法提取被引论文的识别信息: {cited_paper.get('id')}")
            return []

        # 3. 尝试使用PyMuPDF提取
        try:
            import fitz  # PyMuPDF
            contexts = []

            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    text = page.get_text()

                    # 查找引用模式
                    matches = self._find_citation_patterns(text, cited_info)

                    for match in matches:
                        # 提取上下文(引用前后各100个字符)
                        start = max(0, match['start'] - 100)
                        end = min(len(text), match['end'] + 100)
                        context = text[start:end].strip()

                        # 清理上下文(移除多余换行和空格)
                        context = ' '.join(context.split())

                        contexts.append({
                            'page': page_num + 1,
                            'context': context,
                            'citation_text': match['citation']
                        })

            return contexts

        except ImportError:
            # PyMuPDF未安装，尝试使用PyPDF2
            logger.debug("PyMuPDF未安装，尝试使用PyPDF2")
            return self._extract_citation_from_pdf_pypdf2(pdf_path, cited_info)
        except Exception as e:
            logger.warning(f"使用PyMuPDF提取失败: {e}")
            return []

    def _extract_citation_from_pdf_pypdf2(self, pdf_path: str, cited_info: Dict) -> List[Dict]:
        """使用PyPDF2提取引用上下文(降级方案)"""
        try:
            import PyPDF2
            contexts = []

            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)

                for page_num, page in enumerate(reader.pages):
                    try:
                        text = page.extract_text()
                        if not text:
                            continue

                        # 查找引用模式
                        matches = self._find_citation_patterns(text, cited_info)

                        for match in matches:
                            start = max(0, match['start'] - 100)
                            end = min(len(text), match['end'] + 100)
                            context = text[start:end].strip()
                            context = ' '.join(context.split())

                            contexts.append({
                                'page': page_num + 1,
                                'context': context,
                                'citation_text': match['citation']
                            })
                    except Exception as e:
                        logger.debug(f"处理第{page_num+1}页失败: {e}")
                        continue

            return contexts

        except ImportError:
            logger.warning("PyPDF2未安装，无法提取PDF内容")
            return []
        except Exception as e:
            logger.warning(f"使用PyPDF2提取失败: {e}")
            return []

    def _get_pdf_path(self, paper: Dict) -> Optional[str]:
        """
        获取论文PDF文件路径

        策略:
        1. 检查paper对象中的pdf_path字段
        2. 基于paper_id在默认目录中查找
        3. 基于title在默认目录中查找
        """
        # 策略1: 使用paper中的pdf_path
        if paper.get('pdf_path') and os.path.exists(paper['pdf_path']):
            return paper['pdf_path']

        # 策略2: 基于paper_id查找
        paper_id = paper.get('id', '')
        if paper_id:
            # 默认PDF目录
            pdf_dir = Path('./data/papers')
            if not pdf_dir.exists():
                pdf_dir = Path('/home/lexy/下载/CLwithRAG/KGdemo/data/papers')

            if pdf_dir.exists():
                # 查找以paper_id开头的PDF文件
                for pdf_file in pdf_dir.glob(f'{paper_id}*.pdf'):
                    return str(pdf_file)

        # 策略3: 基于title查找(如果paper_id查找失败)
        title = paper.get('title', '')
        if title and pdf_dir.exists():
            # 将标题转换为安全文件名格式
            safe_title = re.sub(r'[^\w\s-]', '', title).strip()
            safe_title = re.sub(r'[\s]+', '_', safe_title)[:50]

            for pdf_file in pdf_dir.glob(f'*{safe_title}*.pdf'):
                return str(pdf_file)

        return None

    def _extract_citation_identifiers(self, paper: Dict) -> Optional[Dict]:
        """
        提取论文的引用识别信息

        Returns:
            包含识别信息的字典:
            {
                'authors': ['Smith', 'Jones'],  # 主要作者姓氏
                'year': '2020',
                'first_author': 'Smith',
                'title_keywords': ['deep', 'learning']
            }
        """
        info = {}

        # 提取年份
        year = paper.get('year') or paper.get('publication_year')
        if year:
            info['year'] = str(year)

        # 提取作者信息
        authors = paper.get('authors', [])
        if authors:
            # 支持多种作者格式
            if isinstance(authors, list):
                if authors and isinstance(authors[0], dict):
                    # 格式: [{'name': 'John Smith'}, ...]
                    author_names = [a.get('name', '') or a.get('author', '') for a in authors]
                else:
                    # 格式: ['John Smith', ...]
                    author_names = authors

                # 提取姓氏
                surnames = []
                for name in author_names[:3]:  # 只取前3个作者
                    if name:
                        # 提取姓氏(假设姓氏是最后一个单词)
                        parts = name.strip().split()
                        if parts:
                            surnames.append(parts[-1])

                if surnames:
                    info['authors'] = surnames
                    info['first_author'] = surnames[0]

        # 提取标题关键词
        title = paper.get('title', '')
        if title:
            # 提取有意义的单词(长度>3)
            words = re.findall(r'\b\w{4,}\b', title.lower())
            info['title_keywords'] = words[:5]  # 取前5个关键词

        return info if info else None

    def _find_citation_patterns(self, text: str, cited_info: Dict) -> List[Dict]:
        """
        在文本中查找引用模式

        支持的引用格式:
        1. [Author, Year] - [Smith, 2020]
        2. (Author, Year) - (Smith, 2020)
        3. Author (Year) - Smith (2020)
        4. [1], [2], etc. - 数字引用(仅当上下文中提到作者时)
        5. Author et al., Year - Smith et al., 2020

        Args:
            text: PDF文本内容
            cited_info: 被引论文的识别信息

        Returns:
            匹配列表，每个元素包含 {'start': int, 'end': int, 'citation': str}
        """
        matches = []

        first_author = cited_info.get('first_author', '')
        year = cited_info.get('year', '')

        if not first_author or not year:
            return matches

        # 构建引用模式(不区分大小写)
        patterns = [
            # [Author, Year] 或 [Author et al., Year]
            rf'\[{first_author}(?:\s+et\s+al\.?)?,?\s*{year}\]',
            # (Author, Year) 或 (Author et al., Year)
            rf'\({first_author}(?:\s+et\s+al\.?)?,?\s*{year}\)',
            # Author (Year) 或 Author et al. (Year)
            rf'{first_author}(?:\s+et\s+al\.)?\s*\({year}\)',
            # Author et al., Year
            rf'{first_author}\s+et\s+al\.,?\s*{year}',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append({
                    'start': match.start(),
                    'end': match.end(),
                    'citation': match.group(0)
                })

        # 按位置排序并去重
        matches = sorted(matches, key=lambda x: x['start'])

        # 去除重叠的匹配
        unique_matches = []
        last_end = -1
        for match in matches:
            if match['start'] >= last_end:
                unique_matches.append(match)
                last_end = match['end']

        return unique_matches

    def _rule_based_inference(
        self,
        citing_paper: Dict,
        cited_paper: Dict,
        citing_analysis: Dict,
        cited_analysis: Dict
    ) -> CitationRelationship:
        """
        基于规则的推断（当没有LLM时使用）
        """
        # 提取基本信息
        citing_year = citing_paper.get('year', 0)
        cited_year = cited_paper.get('year', 0)
        year_diff = citing_year - cited_year if citing_year > 0 and cited_year > 0 else 0

        # 简单规则
        relationship_type = "Baselines"
        confidence = 0.3
        reasoning = "基于规则的简单推断"

        # 规则1: 如果有limitation和problem，可能是Overcomes
        if cited_analysis.get('limitation') and citing_analysis.get('problem'):
            if self._text_similarity(cited_analysis['limitation'], citing_analysis['problem']) > 0.3:
                relationship_type = "Overcomes"
                confidence = 0.6
                reasoning = "B的问题与A的局限性相关"

        # 规则2: 如果有future_work和problem，可能是Realizes
        if cited_analysis.get('future_work') and citing_analysis.get('problem'):
            if self._text_similarity(cited_analysis['future_work'], citing_analysis['problem']) > 0.3:
                relationship_type = "Realizes"
                confidence = 0.6
                reasoning = "B实现了A建议的未来工作"

        logger.info(f"  规则推断: {relationship_type} (置信度: {confidence:.2f})")

        return CitationRelationship(
            citing_id=citing_paper['id'],
            cited_id=cited_paper['id'],
            relationship_type=relationship_type,
            confidence=confidence,
            reasoning=reasoning,
            evidence="",
            match_results=[]
        )

    def _text_similarity(self, text1: str, text2: str) -> float:
        """简单的文本相似度计算"""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _extract_json_from_response(self, response: str) -> str:
        """
        从LLM响应中提取JSON内容
        处理可能包含markdown代码块的情况
        """
        import re

        # 去除首尾空白
        response = response.strip()

        # 尝试提取markdown代码块中的JSON
        # 匹配 ```json ... ``` 或 ``` ... ```
        json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        match = re.search(json_block_pattern, response, re.DOTALL)

        if match:
            return match.group(1).strip()

        # 如果没有代码块，直接返回原响应
        return response


if __name__ == "__main__":
    # 测试代码
    import logging
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 检查命令行参数
    use_rule_based = len(sys.argv) > 1 and sys.argv[1] == '--rule-based'

    # 创建推断器（默认使用LLM）
    if use_rule_based:
        print("\n📏 使用规则方法模式（无LLM）")
        print("提示: 不使用参数则默认启用LLM模式\n")
        inferencer = CitationTypeInferencer(llm_client=None)
    else:
        print("\n🔌 使用 LLM Socket Matching 模式（默认）")
        print("从 config/config.yaml 加载LLM配置...")
        print("提示: 使用 --rule-based 参数切换到规则方法\n")
        inferencer = CitationTypeInferencer(config_path="config/config.yaml")

    # 测试论文数据
    test_papers = [
        {
            'id': 'W1',
            'title': 'Attention Is All You Need',
            'year': 2017,
            'cited_by_count': 50000,
            'deep_analysis': {
                'problem': 'Existing sequence models are difficult to parallelize',
                'method': 'Proposed Transformer model based entirely on attention mechanisms',
                'limitation': 'Limited to fixed-length sequences, requires large amounts of training data',
                'future_work': 'Explore Transformer applications in other domains like computer vision'
            }
        },
        {
            'id': 'W2',
            'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
            'year': 2018,
            'cited_by_count': 40000,
            'deep_analysis': {
                'problem': 'Existing pre-training models can only model in one direction',
                'method': 'Proposed bidirectional Transformer pre-training method BERT',
                'limitation': 'BERT is computationally expensive for fine-tuning',
                'future_work': 'Investigate more efficient pre-training methods'
            }
        },
        {
            'id': 'W3',
            'title': 'Vision Transformer (ViT)',
            'year': 2020,
            'cited_by_count': 15000,
            'deep_analysis': {
                'problem': 'Applying Transformer to computer vision tasks',
                'method': 'Demonstrated that pure Transformer can work well on image classification',
                'limitation': 'Requires very large datasets to train effectively',
                'future_work': 'Apply to other vision tasks like detection and segmentation'
            }
        }
    ]

    # 测试引用关系
    test_edges = [
        ('W2', 'W1'),  # BERT引用Transformer (应该是 Overcomes 或 Baselines)
        ('W3', 'W1'),  # ViT引用Transformer (应该是 Realizes - 实现了未来工作建议)
    ]

    # 推断引用类型
    print("\n" + "="*80)
    print("Socket Matching 引用关系类型推断测试")
    print("="*80)

    typed_edges, statistics = inferencer.infer_edge_types(test_papers, test_edges)

    print("\n引用关系类型推断结果:")
    print("="*80)
    for citing_id, cited_id, edge_type in typed_edges:
        citing_paper = next(p for p in test_papers if p['id'] == citing_id)
        cited_paper = next(p for p in test_papers if p['id'] == cited_id)
        print(f"\n{citing_paper['title']}")
        print(f"  → {cited_paper['title']}")
        print(f"  关系类型: {edge_type}")
    print("="*80)
