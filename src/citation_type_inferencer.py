"""
Socket Matching Citation Relationship Type Inferencer
Infers semantic types of citation relationships using the "socket matching" method based on deep paper information

Core Idea:
Treat extracted deep paper information (Problem, Method, Limitation, Future_Work) as "sockets",
and use LLM Agent to determine if these sockets can connect, thereby inferring the semantic type of citation relationships.

Supported Relationship Types (Socket Matching - 6 types): (Overcomes, Realizes, Extends, Alternative, Adapts_to, Baselines)
1. Overcomes - Tackles/Optimizes (Vertical Deepening)
   Source: Match 1 (Limitation→Problem)
2. Realizes - Realizes Vision (Research Inheritance)
   Source: Match 2 (Future_Work→Problem)
3. Extends - Method Extension (Incremental Innovation)
   Source: Match 3 Extension
4. Alternative - Alternative Approach (Disruptive Innovation)
   Source: Match 3 Alternative
5. Adapts_to - Technology Transfer (Horizontal Diffusion)
   Source: Match 4 (Problem→Problem Cross-domain)
6. Baselines - Baseline Comparison (Background Noise)
   Source: No match

Logic Matching Matrix (4 Matches → 6 types):
- Match 1: A.Limitation ↔ B.Problem → Overcomes
- Match 2: A.Future_Work ↔ B.Problem → Realizes
- Match 3: (Same Problem) A.Method ↔ B.Method → Extends(Extension) / Alternative
- Match 4: A.Problem ↔ B.Problem (Cross-domain) → Adapts_to
- No match → Baselines
"""

import json
import logging
import os
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# Import LLM configuration module
try:
    from llm_config import create_llm_client
except ImportError:
    create_llm_client = None

logger = logging.getLogger(__name__)


@dataclass
class SocketMatchResult:
    """Socket matching result"""
    match_type: str  # "limitation_problem", "future_work_problem", "method_extension", "problem_adaptation"
    is_match: bool
    confidence: float
    reasoning: str
    evidence: str
    additional_info: Dict = None  # Additional information (e.g., relationship_type, source_domain, target_domain)


@dataclass
class CitationRelationship:
    """Citation relationship"""
    citing_id: str
    cited_id: str
    relationship_type: str  # Overcomes, Realizes, Extends, Alternative, Adapts_to, Baselines (6 types)
    confidence: float
    reasoning: str
    evidence: str
    match_results: List[SocketMatchResult]


class CitationTypeInferencer:
    """
    Socket Matching Citation Relationship Type Inferencer

    Uses LLM Agent for deep semantic analysis to determine citation relationship types through "socket matching"
    """

    def __init__(self, llm_client=None, config_path: str = None, prompts_dir: str = "./prompts"):
        """
        Initialize the inferencer

        Args:
            llm_client: LLM client (if None, use rule-based method)
            config_path: LLM configuration file path (if provided and llm_client is None, load from this file)
            prompts_dir: Prompts directory
        """
        # If config_path is provided but llm_client is not, try to load from config file
        if llm_client is None and config_path:
            if create_llm_client is None:
                logger.warning("Unable to import create_llm_client, will use rule-based method")
                self.llm_client = None
            else:
                try:
                    config_file = Path(config_path)
                    if config_file.exists():
                        self.llm_client = create_llm_client(str(config_file))
                        logger.info(f"Successfully loaded LLM client from config file: {config_path}")
                    else:
                        logger.warning(f"Config file does not exist: {config_path}, will use rule-based method")
                        self.llm_client = None
                except Exception as e:
                    logger.warning(f"Failed to load LLM client: {e}, will use rule-based method")
                    self.llm_client = None
        else:
            self.llm_client = llm_client

        self.prompts_dir = Path(prompts_dir)
        self.prompts_cache = {}

        # Load prompts
        self._load_prompts()

        # Relationship type priority (for rule-based method and conflict resolution)
        self.relationship_priority = {
            "Overcomes": 6,     # Highest priority - directly solves problem
            "Realizes": 5,      # Second highest priority - realizes vision
            "Adapts_to": 4,     # High priority - technology transfer
            "Extends": 3,       # Medium-high priority - method extension
            "Alternative": 2,   # Medium priority - alternative approach
            "Baselines": 1      # Lowest priority - baseline comparison
        }

        logger.info("CitationTypeInferencer initialization complete")
        if self.llm_client:
            logger.info("  Mode: LLM Socket Matching")
        else:
            logger.info("  Mode: Rule-based method (fallback mode)")

    def _load_prompts(self):
        """Load all prompts"""
        prompt_files = {
            'match_limitation_problem': 'match_limitation_problem.txt',
            'match_future_work_problem': 'match_future_work_problem.txt',
            'match_method_extension': 'match_method_extension.txt',
            'match_problem_adaptation': 'match_problem_adaptation.txt',
            'classify_relationship': 'classify_relationship.txt'
        }

        for key, filename in prompt_files.items():
            file_path = self.prompts_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.prompts_cache[key] = f.read().strip()
                    logger.debug(f"  Loaded prompt: {key}")
                except Exception as e:
                    logger.warning(f"  Failed to load prompt ({key}): {e}")
            else:
                logger.warning(f"  Prompt file does not exist: {filename}")

        logger.info(f"Loaded {len(self.prompts_cache)} prompt templates")

    def infer_edge_types(
        self,
        papers: List[Dict],
        citation_edges: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str, str]], Dict[str, int]]:
        """
        Batch infer citation relationship types

        Args:
            papers: List of papers (must contain rag_analysis or deep_analysis)
            citation_edges: List of citation relationships [(citing_id, cited_id), ...]

        Returns:
            (typed_edges, statistics)
            - typed_edges: Citation relationships with types [(citing_id, cited_id, edge_type), ...]
            - statistics: Type statistics {edge_type: count}
        """
        logger.info(f"Starting to infer types for {len(citation_edges)} citation relationships...")

        # Build paper dictionary
        papers_dict = {paper['id']: paper for paper in papers}

        # Infer type for each edge
        typed_edges = []
        statistics = {}
        relationships = []

        for i, (citing_id, cited_id) in enumerate(citation_edges):
            logger.info(f"Processing citation relationship {i+1}/{len(citation_edges)}: {citing_id} -> {cited_id}")

            if citing_id in papers_dict and cited_id in papers_dict:
                relationship = self.infer_single_edge_type(
                    papers_dict[citing_id],
                    papers_dict[cited_id]
                )
                relationships.append(relationship)
                edge_type = relationship.relationship_type
            else:
                # Paper not in dictionary, use default type
                edge_type = "Baselines"
                logger.warning(f"  Paper not in dictionary, using default type: {edge_type}")

            typed_edges.append((citing_id, cited_id, edge_type))
            statistics[edge_type] = statistics.get(edge_type, 0) + 1

        logger.info(f"Citation type inference complete")
        logger.info(f"  Total citation relationships: {len(typed_edges)}")
        logger.info(f"\nCitation type distribution:")
        for edge_type, count in sorted(statistics.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(typed_edges)) * 100 if typed_edges else 0
            logger.info(f"  - {edge_type}: {count} ({percentage:.1f}%)")

        return typed_edges, statistics

    def infer_single_edge_type(
        self,
        citing_paper: Dict,
        cited_paper: Dict
    ) -> CitationRelationship:
        """
        Infer the type of a single citation relationship (Socket Matching)

        Args:
            citing_paper: Citing paper (Paper B)
            cited_paper: Cited paper (Paper A)

        Returns:
            CitationRelationship object
        """
        # Extract deep analysis information
        citing_analysis = self._extract_deep_analysis(citing_paper)
        cited_analysis = self._extract_deep_analysis(cited_paper)

        # Extract citation context
        citation_context = self._extract_citation_context(citing_paper, cited_paper)

        # If no LLM client, use rule-based method
        if not self.llm_client:
            return self._rule_based_inference(
                citing_paper, cited_paper, citing_analysis, cited_analysis
            )

        # Socket Matching: Execute 4 match detections
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

        # Match 4: A.Problem ↔ B.Problem (but different scenarios)
        if cited_analysis.get('problem') and citing_analysis.get('problem'):
            match = self._check_problem_adaptation_match(
                cited_paper, citing_paper, cited_analysis, citing_analysis, citation_context
            )
            if match:
                match_results.append(match)

        # Synthesize all match results for final classification
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
        Match 1: Check A.Limitation ↔ B.Problem
        Determine if B solves A's limitations
        """
        prompt_template = self.prompts_cache.get('match_limitation_problem')
        if not prompt_template:
            logger.warning("Missing match_limitation_problem prompt, skipping")
            return None

        # Fill in the prompt
        prompt = prompt_template.format(
            cited_title=cited_paper.get('title', 'Unknown'),
            cited_limitation=cited_analysis.get('limitation', 'N/A'),
            citing_title=citing_paper.get('title', 'Unknown'),
            citing_problem=citing_analysis.get('problem', 'N/A'),
            citation_context=citation_context
        )

        # Call LLM
        try:
            response = self.llm_client.generate(prompt)

            # Extract JSON content
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
            logger.error(f"Match 1 (Limitation-Problem) failed: {e}")
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
        Match 2: Check A.Future_Work ↔ B.Problem
        Determine if B implements A's future work suggestions
        """
        # If A's Future Work is empty or too short, skip Match 2
        future_work = cited_analysis.get('future_work', '')
        if not future_work or len(future_work) < 5 or future_work == "N/A":
            logger.info("    → Match 2 skipped: A's Future Work is empty or too short")
            return None

        prompt_template = self.prompts_cache.get('match_future_work_problem')
        if not prompt_template:
            logger.warning("Missing match_future_work_problem prompt, skipping")
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

            # Double filtering: Distinguish true inheritance (Realizes) vs false courtesy (Extends/Baselines)
            is_match = result.get('is_match', False)
            specificity = result.get('specificity', 'low')
            confidence = result.get('confidence', 0.0)

            # Scenario 1: LLM thinks it matches + suggestion is specific (high specificity) → True Realizes
            if is_match and specificity == "high" and confidence > 0.6:
                logger.info(f"    → Match 2 specificity check: ✓ High specificity (specificity=high, conf={confidence:.2f})")
                return SocketMatchResult(
                    match_type="future_work_problem",
                    is_match=True,
                    confidence=confidence,
                    reasoning=result.get('reasoning', ''),
                    evidence=result.get('evidence', ''),
                    additional_info={'specificity': 'high'}
                )

            # Scenario 2: LLM thinks it matches + but suggestion is vague (low specificity) → False courtesy, downgrade
            elif is_match and specificity == "low":
                logger.info(f"    → Match 2 specificity check: ✗ Low specificity (specificity=low, conf={confidence:.2f}) - Suspected courtesy, not counted as Realizes")
                return SocketMatchResult(
                    match_type="future_work_problem",
                    is_match=False,  # Force mark as no match
                    confidence=0.0,
                    reasoning=f"[Filtered] A's Future Work is too vague, does not meet Realizes criteria. {result.get('reasoning', '')}",
                    evidence=result.get('evidence', ''),
                    additional_info={'specificity': 'low', 'filtered': True}
                )

            # Scenario 3: LLM thinks it doesn't match
            else:
                logger.info(f"    → Match 2: No match (is_match=False)")
                return SocketMatchResult(
                    match_type="future_work_problem",
                    is_match=False,
                    confidence=0.0,
                    reasoning=result.get('reasoning', ''),
                    evidence=result.get('evidence', '')
                )
        except Exception as e:
            logger.error(f"Match 2 (FutureWork-Problem) failed: {e}")
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
        Match 3: Check A.Method ↔ B.Method
        Determine if it's Extension or Alternative
        """
        prompt_template = self.prompts_cache.get('match_method_extension')
        if not prompt_template:
            logger.warning("Missing match_method_extension prompt, skipping")
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
            logger.error(f"Match 3 (Method Extension) failed: {e}")
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
        Match 4: Check A.Problem ↔ B.Problem (but different scenarios)
        Determine if it's technology transfer/generalization
        """
        prompt_template = self.prompts_cache.get('match_problem_adaptation')
        if not prompt_template:
            logger.warning("Missing match_problem_adaptation prompt, skipping")
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

            # Double filtering: Distinguish true cross-domain transfer (Adapts_to) vs dataset change (Extends)
            is_adaptation = result.get('is_adaptation', False)
            domain_shift_type = result.get('domain_shift_type', 'none')
            confidence = result.get('confidence', 0.0)

            # Scenario 1: True cross-domain transfer (cross-task/cross-modality) → Adapts_to
            if is_adaptation and domain_shift_type in ['cross-task', 'cross-modality']:
                logger.info(f"    → Match 4 domain span check: ✓ True cross-domain transfer (type={domain_shift_type}, conf={confidence:.2f})")
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

            # Scenario 2: Just changing dataset (same-task-new-data) → Not Adapts_to, downgrade
            elif is_adaptation and domain_shift_type == 'same-task-new-data':
                logger.info(f"    → Match 4 domain span check: ✗ Only dataset change (type={domain_shift_type}, conf={confidence:.2f}) - Not true Adapts_to")
                return SocketMatchResult(
                    match_type="problem_adaptation",
                    is_match=False,  # Force mark as no match
                    confidence=0.0,
                    reasoning=f"[Filtered] Only changing dataset on same task type, does not meet true domain transfer. {result.get('reasoning', '')}",
                    evidence=result.get('evidence', ''),
                    additional_info={
                        'domain_shift_type': domain_shift_type,
                        'filtered': True
                    }
                )

            # Scenario 3: Not adaptation or domain_shift_type is none
            else:
                logger.info(f"    → Match 4: No match (is_adaptation={is_adaptation}, type={domain_shift_type})")
                return SocketMatchResult(
                    match_type="problem_adaptation",
                    is_match=False,
                    confidence=0.0,
                    reasoning=result.get('reasoning', ''),
                    evidence=result.get('evidence', '')
                )
        except Exception as e:
            logger.error(f"Match 4 (Problem Adaptation) failed: {e}")
            return None

    def _classify_relationship(
        self,
        citing_paper: Dict,
        cited_paper: Dict,
        match_results: List[SocketMatchResult],
        citation_context: str  # Reserved for future use
    ) -> CitationRelationship:
        """
        Synthesize all match results for final relationship type classification
        Uses priority-based decision tree logic (no longer depends on LLM)

        Decision tree logic (by priority):
        1. Match 1 (Limitation→Problem) success → Overcomes
        2. Match 2 (Future_Work→Problem) success → Realizes
        3. Match 4 (Problem→Problem cross-domain) success → Adapts_to
        4. Match 3 (Method→Method) success:
           - extension → Extends
           - alternative → Alternative
           - none → Baselines
        5. No match → Baselines

        Priority order: Overcomes > Realizes > Adapts_to > Extends > Alternative > Baselines
        """
        # If no match results, default to Baselines
        if not match_results:
            logger.info("  No match results -> Baselines")
            return CitationRelationship(
                citing_id=citing_paper['id'],
                cited_id=cited_paper['id'],
                relationship_type="Baselines",
                confidence=0.3,
                reasoning="No clear deep relationship, only used as baseline comparison",
                evidence="",
                match_results=[]
            )

        # Organize match results by type
        matches_by_type = {}
        for match in match_results:
            if match.is_match:
                matches_by_type[match.match_type] = match

        # Check match results in priority order
        relationship_type = "Baselines"
        confidence = 0.3
        reasoning = "No clear deep relationship, only used as baseline comparison"
        evidence = ""
        relationship_decided = False  # Mark whether relationship type has been determined

        # Priority 1: Match 1 (Limitation→Problem) → Overcomes
        if not relationship_decided and "limitation_problem" in matches_by_type:
            match = matches_by_type["limitation_problem"]
            relationship_type = "Overcomes"
            confidence = match.confidence
            reasoning = f"B solved A's limitations. {match.reasoning}"
            evidence = match.evidence
            relationship_decided = True
            logger.info(f"  ✓ Match 1 (Limitation→Problem) matched successfully -> Overcomes (confidence: {confidence:.2f})")

        # Priority 2: Match 2 (Future_Work→Problem) → Realizes
        # Special note: Must be high specificity future work, not courtesy language
        if not relationship_decided and "future_work_problem" in matches_by_type:
            match = matches_by_type["future_work_problem"]

            # Double verification: Check specificity
            specificity = match.additional_info.get('specificity', 'low') if match.additional_info else 'low'

            if specificity == "high" and match.confidence > 0.6:
                # True Realizes: A digs the hole, B fills it
                relationship_type = "Realizes"
                confidence = match.confidence
                reasoning = f"B implemented the specific future work direction envisioned by A. {match.reasoning}"
                evidence = match.evidence
                relationship_decided = True
                logger.info(f"  ✓ Match 2 (Future_Work→Problem) matched successfully -> Realizes (confidence: {confidence:.2f}) [high specificity]")
            else:
                # Low specificity or low confidence: Not Realizes, continue checking other Matches
                logger.info(f"  ⚠ Match 2 detected but insufficient specificity (specificity={specificity}, conf={match.confidence:.2f}) - Skip Realizes, check other Matches")

        # Priority 3: Match 4 (Problem→Problem cross-domain) → Adapts_to
        # Special note: Must be true cross-task/cross-modality, not just dataset change
        if not relationship_decided and "problem_adaptation" in matches_by_type:
            match = matches_by_type["problem_adaptation"]

            # Double verification: Check domain_shift_type
            domain_shift_type = match.additional_info.get('domain_shift_type', 'none') if match.additional_info else 'none'

            if domain_shift_type in ['cross-task', 'cross-modality']:
                # True cross-domain transfer: Technology horizontal diffusion
                relationship_type = "Adapts_to"
                confidence = match.confidence
                source_domain = match.additional_info.get('source_domain', '') if match.additional_info else ''
                target_domain = match.additional_info.get('target_domain', '') if match.additional_info else ''
                reasoning = f"B transferred A's method to a different domain ({source_domain} → {target_domain}, {domain_shift_type}). {match.reasoning}"
                evidence = match.evidence
                relationship_decided = True
                logger.info(f"  ✓ Match 4 (Problem→Problem cross-domain) matched successfully -> Adapts_to (confidence: {confidence:.2f}) [{domain_shift_type}]")
            else:
                # Only dataset change or no significant span: Not Adapts_to
                logger.info(f"  ⚠ Match 4 detected but insufficient domain span (type={domain_shift_type}, conf={match.confidence:.2f}) - Skip Adapts_to")

        # Priority 4-5: Match 3 (Method→Method) → Extends / Alternative
        if not relationship_decided and "method_extension" in matches_by_type:
            match = matches_by_type["method_extension"]
            rel_type = match.additional_info.get('relationship_type', 'none') if match.additional_info else 'none'

            if rel_type == "extension":
                relationship_type = "Extends"
                confidence = match.confidence
                reasoning = f"B made incremental improvements based on A's method. {match.reasoning}"
                evidence = match.evidence
                relationship_decided = True
                logger.info(f"  ✓ Match 3 (Method Extension) matched successfully -> Extends (confidence: {confidence:.2f})")

            elif rel_type == "alternative":
                relationship_type = "Alternative"
                confidence = match.confidence
                reasoning = f"B uses a different paradigm to solve similar problems. {match.reasoning}"
                evidence = match.evidence
                relationship_decided = True
                logger.info(f"  ✓ Match 3 (Method Alternative) matched successfully -> Alternative (confidence: {confidence:.2f})")

            else:  # rel_type == "none"
                relationship_type = "Baselines"
                confidence = 0.4
                reasoning = "No clear inheritance or improvement relationship between methods, only used as baseline comparison"
                evidence = match.evidence
                relationship_decided = True
                logger.info(f"  ✓ Match 3 (Method None) -> Baselines (confidence: {confidence:.2f})")

        # Priority 6: No valid match → Baselines
        if not relationship_decided:
            logger.info("  All matches failed -> Baselines")

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
        Extract deep analysis information from paper
        Priority: deep_analysis > rag_analysis > empty dict
        """
        if 'deep_analysis' in paper:
            return paper['deep_analysis']
        elif 'rag_analysis' in paper:
            return paper['rag_analysis']
        else:
            return {}

    def _extract_citation_context(self, citing_paper: Dict, cited_paper: Dict) -> str:
        """
        Extract citation context - extract specific sentences citing A from PDF file

        Priority:
        1. Extract citation context from PDF file (based on author name and year matching)
        2. If PDF is unavailable or extraction fails, return simple description

        Args:
            citing_paper: Citing paper (Paper B)
            cited_paper: Cited paper (Paper A)

        Returns:
            Citation context string, may contain multiple citation points
        """
        # Try to extract citation context from PDF
        try:
            contexts = self._extract_citation_from_pdf(citing_paper, cited_paper)
            if contexts:
                # Return first 3 citation contexts
                context_str = " | ".join([
                    f"[p.{ctx['page']}] {ctx['context']}"
                    for ctx in contexts[:3]
                ])
                logger.debug(f"Extracted {len(contexts)} citation contexts from PDF")
                return context_str
        except Exception as e:
            logger.warning(f"Failed to extract citation context from PDF: {e}")

        # Fallback: Return simple description
        return f"{citing_paper.get('title', 'Paper B')} cited {cited_paper.get('title', 'Paper A')}"

    def _extract_citation_from_pdf(self, citing_paper: Dict, cited_paper: Dict) -> List[Dict]:
        """
        Extract citation context from PDF file

        Strategy:
        1. Locate PDF file path
        2. Extract full text from PDF
        3. Use citation pattern matching (based on author name and year)
        4. Extract context before and after citation

        Args:
            citing_paper: Citing paper
            cited_paper: Cited paper

        Returns:
            List of citation contexts, each element contains {'page': int, 'context': str}
        """
        # 1. Locate PDF file
        pdf_path = self._get_pdf_path(citing_paper)
        if not pdf_path or not os.path.exists(pdf_path):
            logger.debug(f"PDF file does not exist: {pdf_path}")
            return []

        # 2. Get identification information of cited paper
        cited_info = self._extract_citation_identifiers(cited_paper)
        if not cited_info:
            logger.debug(f"Unable to extract identification information of cited paper: {cited_paper.get('id')}")
            return []

        # 3. Try to extract using PyMuPDF
        try:
            import fitz  # PyMuPDF
            contexts = []

            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    text = page.get_text()

                    # Find citation patterns
                    matches = self._find_citation_patterns(text, cited_info)

                    for match in matches:
                        # Extract context (100 characters before and after citation)
                        start = max(0, match['start'] - 100)
                        end = min(len(text), match['end'] + 100)
                        context = text[start:end].strip()

                        # Clean context (remove extra newlines and spaces)
                        context = ' '.join(context.split())

                        contexts.append({
                            'page': page_num + 1,
                            'context': context,
                            'citation_text': match['citation']
                        })

            return contexts

        except ImportError:
            # PyMuPDF not installed, try using PyPDF2
            logger.debug("PyMuPDF not installed, trying PyPDF2")
            return self._extract_citation_from_pdf_pypdf2(pdf_path, cited_info)
        except Exception as e:
            logger.warning(f"Extraction using PyMuPDF failed: {e}")
            return []

    def _extract_citation_from_pdf_pypdf2(self, pdf_path: str, cited_info: Dict) -> List[Dict]:
        """Extract citation context using PyPDF2 (fallback method)"""
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

                        # Find citation patterns
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
                        logger.debug(f"Processing page {page_num+1} failed: {e}")
                        continue

            return contexts

        except ImportError:
            logger.warning("PyPDF2 not installed, unable to extract PDF content")
            return []
        except Exception as e:
            logger.warning(f"Extraction using PyPDF2 failed: {e}")
            return []

    def _get_pdf_path(self, paper: Dict) -> Optional[str]:
        """
        Get paper PDF file path

        Strategy:
        1. Check pdf_path field in paper object
        2. Search in default directory based on paper_id
        3. Search in default directory based on title
        """
        # Strategy 1: Use pdf_path from paper
        if paper.get('pdf_path') and os.path.exists(paper['pdf_path']):
            return paper['pdf_path']

        # Strategy 2: Search based on paper_id
        paper_id = paper.get('id', '')
        if paper_id:
            # Default PDF directory
            pdf_dir = Path('./data/papers')
            if not pdf_dir.exists():
                pdf_dir = Path('/home/lexy/下载/CLwithRAG/KGdemo/data/papers')

            if pdf_dir.exists():
                # Find PDF files starting with paper_id
                for pdf_file in pdf_dir.glob(f'{paper_id}*.pdf'):
                    return str(pdf_file)

        # Strategy 3: Search based on title (if paper_id search fails)
        title = paper.get('title', '')
        if title and pdf_dir.exists():
            # Convert title to safe filename format
            safe_title = re.sub(r'[^\w\s-]', '', title).strip()
            safe_title = re.sub(r'[\s]+', '_', safe_title)[:50]

            for pdf_file in pdf_dir.glob(f'*{safe_title}*.pdf'):
                return str(pdf_file)

        return None

    def _extract_citation_identifiers(self, paper: Dict) -> Optional[Dict]:
        """
        Extract citation identification information from paper

        Returns:
            Dictionary containing identification information:
            {
                'authors': ['Smith', 'Jones'],  # Main author surnames
                'year': '2020',
                'first_author': 'Smith',
                'title_keywords': ['deep', 'learning']
            }
        """
        info = {}

        # Extract year
        year = paper.get('year') or paper.get('publication_year')
        if year:
            info['year'] = str(year)

        # Extract author information
        authors = paper.get('authors', [])
        if authors:
            # Support multiple author formats
            if isinstance(authors, list):
                if authors and isinstance(authors[0], dict):
                    # Format: [{'name': 'John Smith'}, ...]
                    author_names = [a.get('name', '') or a.get('author', '') for a in authors]
                else:
                    # Format: ['John Smith', ...]
                    author_names = authors

                # Extract surnames
                surnames = []
                for name in author_names[:3]:  # Only take first 3 authors
                    if name:
                        # Extract surname (assume surname is the last word)
                        parts = name.strip().split()
                        if parts:
                            surnames.append(parts[-1])

                if surnames:
                    info['authors'] = surnames
                    info['first_author'] = surnames[0]

        # Extract title keywords
        title = paper.get('title', '')
        if title:
            # Extract meaningful words (length > 3)
            words = re.findall(r'\b\w{4,}\b', title.lower())
            info['title_keywords'] = words[:5]  # Take first 5 keywords

        return info if info else None

    def _find_citation_patterns(self, text: str, cited_info: Dict) -> List[Dict]:
        """
        Find citation patterns in text

        Supported citation formats:
        1. [Author, Year] - [Smith, 2020]
        2. (Author, Year) - (Smith, 2020)
        3. Author (Year) - Smith (2020)
        4. [1], [2], etc. - Numeric citations (only when author is mentioned in context)
        5. Author et al., Year - Smith et al., 2020

        Args:
            text: PDF text content
            cited_info: Identification information of cited paper

        Returns:
            List of matches, each element contains {'start': int, 'end': int, 'citation': str}
        """
        matches = []

        first_author = cited_info.get('first_author', '')
        year = cited_info.get('year', '')

        if not first_author or not year:
            return matches

        # Build citation patterns (case insensitive)
        patterns = [
            # [Author, Year] or [Author et al., Year]
            rf'\[{first_author}(?:\s+et\s+al\.?)?,?\s*{year}\]',
            # (Author, Year) or (Author et al., Year)
            rf'\({first_author}(?:\s+et\s+al\.?)?,?\s*{year}\)',
            # Author (Year) or Author et al. (Year)
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

        # Sort by position and deduplicate
        matches = sorted(matches, key=lambda x: x['start'])

        # Remove overlapping matches
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
        Rule-based inference (used when no LLM available)
        """
        # Extract basic information
        citing_year = citing_paper.get('year', 0)
        cited_year = cited_paper.get('year', 0)
        year_diff = citing_year - cited_year if citing_year > 0 and cited_year > 0 else 0

        # Simple rules
        relationship_type = "Baselines"
        confidence = 0.3
        reasoning = "Simple rule-based inference"

        # Rule 1: If there's limitation and problem, might be Overcomes
        if cited_analysis.get('limitation') and citing_analysis.get('problem'):
            if self._text_similarity(cited_analysis['limitation'], citing_analysis['problem']) > 0.3:
                relationship_type = "Overcomes"
                confidence = 0.6
                reasoning = "B's problem is related to A's limitations"

        # Rule 2: If there's future_work and problem, might be Realizes
        if cited_analysis.get('future_work') and citing_analysis.get('problem'):
            if self._text_similarity(cited_analysis['future_work'], citing_analysis['problem']) > 0.3:
                relationship_type = "Realizes"
                confidence = 0.6
                reasoning = "B implemented the future work suggested by A"

        logger.info(f"  Rule-based inference: {relationship_type} (confidence: {confidence:.2f})")

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
        """Simple text similarity calculation"""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON content from LLM response
        Handle cases where markdown code blocks may be included
        """
        import re

        # Remove leading and trailing whitespace
        response = response.strip()

        # Try to extract JSON from markdown code block
        # Match ```json ... ``` or ``` ... ```
        json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        match = re.search(json_block_pattern, response, re.DOTALL)

        if match:
            return match.group(1).strip()

        # If no code block, return original response directly
        return response


if __name__ == "__main__":
    # Test code
    import logging
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Check command line arguments
    use_rule_based = len(sys.argv) > 1 and sys.argv[1] == '--rule-based'

    # Create inferencer (default uses LLM)
    if use_rule_based:
        print("\n Using rule-based method mode (no LLM)")
        print("Tip: Use without parameters to enable LLM mode by default\n")
        inferencer = CitationTypeInferencer(llm_client=None)
    else:
        print("\n Using LLM Socket Matching mode (default)")
        print("Loading LLM configuration from config/config.yaml...")
        print("Tip: Use --rule-based parameter to switch to rule-based method\n")
        inferencer = CitationTypeInferencer(config_path="config/config.yaml")

    # Test paper data
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

    # Test citation relationships
    test_edges = [
        ('W2', 'W1'),  # BERT cites Transformer (should be Overcomes or Baselines)
        ('W3', 'W1'),  # ViT cites Transformer (should be Realizes - implements future work suggestion)
    ]

    # Infer citation types
    print("\n" + "="*80)
    print("Socket Matching Citation Relationship Type Inference Test")
    print("="*80)

    typed_edges, statistics = inferencer.infer_edge_types(test_papers, test_edges)

    print("\nCitation Relationship Type Inference Results:")
    print("="*80)
    for citing_id, cited_id, edge_type in typed_edges:
        citing_paper = next(p for p in test_papers if p['id'] == citing_id)
        cited_paper = next(p for p in test_papers if p['id'] == cited_id)
        print(f"\n{citing_paper['title']}")
        print(f"  → {cited_paper['title']}")
        print(f"  Relationship Type: {edge_type}")
    print("="*80)
