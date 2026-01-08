"""
科研创意评估脚本
使用LLM对科研创意进行四个维度的李克特量表评分（1-5分）：
1. Novelty（新颖性）：该想法是否提供了区别于现有文献的独特视角？
2. Feasibility（可行性）：基于现有技术栈，该技术路线是否逻辑自洽且可实现？
3. Theoretical Support（理论支撑度）：评估该假设是否提供了充分的依据，符合领域发展的历史惯性。
4. Logical Alignment（逻辑契合度）：评估生成的解决方案是否精准且实质性地解决了提出的科研痛点。
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import sys
import time
from datetime import datetime
import csv

# 添加src目录到路径，以便导入llm_config
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# 导入LLM配置
from llm_config import LLMConfig, LLMClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_full_config(config_path: str = './config/config.yaml') -> Dict:
    """
    从YAML文件加载完整配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        完整配置字典
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
        return {}
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"成功加载配置文件: {config_path}")
        return config if config else {}
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}，使用默认配置")
        return {}


class IdeaEvaluator:
    """
    科研创意评估器
    使用LLM对科研创意进行三个维度的评分
    """
    
    def __init__(self, config_path: str = './config/config.yaml'):
        """
        初始化评估器
        
        Args:
            config_path: LLM配置文件路径
        """
        # 加载配置
        full_config = load_full_config(config_path)
        llm_config_dict = full_config.get('llm', {})
        
        # 构建LLMConfig
        llm_config = LLMConfig(
            provider=llm_config_dict.get('provider', 'openai'),
            model=llm_config_dict.get('model', 'gpt-4o'),
            api_key=llm_config_dict.get('api_key') or full_config.get('openai_api_key'),
            base_url=llm_config_dict.get('base_url') or full_config.get('openai_base_url'),
            temperature=llm_config_dict.get('temperature', 0.3),
            max_tokens=llm_config_dict.get('max_tokens', 1000),
            timeout=llm_config_dict.get('timeout', 30)
        )
        
        # 如果仍然没有API key，尝试从环境变量读取
        if not llm_config.api_key:
            import os
            llm_config.api_key = os.getenv('OPENAI_API_KEY')
        
        if not llm_config.api_key:
            raise ValueError(
                "未找到API密钥，请确保：\n"
                "1. 在配置文件中设置 llm.api_key 或 openai_api_key\n"
                "2. 或设置环境变量 OPENAI_API_KEY"
            )
        
        # 初始化LLM客户端
        self.llm_client = LLMClient(llm_config)
        logger.info(f"✅ 评估器初始化成功，模型: {llm_config.model}")
    
    def evaluate_idea(
        self,
        idea: Dict,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        评估单个科研创意
        
        Args:
            idea: 创意字典，包含 title, abstract, modification, reasoning 等字段
            context: 可选的上下文信息（如演化路径、相关论文等）
                  对于 CoI 格式，可能包含 idea_chains 字典
        
        Returns:
            评估结果字典，包含三个维度的评分和理由
        """
        # 构建评估提示词
        prompt = self._build_evaluation_prompt(idea, context)
        
        # 调用LLM进行评估
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self._get_system_prompt(),
                temperature=0.3,  # 使用较低温度以保证评分一致性
                max_tokens=1000
            )
            
            # 解析响应
            evaluation = self._parse_evaluation_response(response)
            
            return evaluation
            
        except Exception as e:
            logger.error(f"评估创意时出错: {e}")
            return {
                "novelty": {"score": None, "reasoning": f"评估失败: {str(e)}"},
                "feasibility": {"score": None, "reasoning": f"评估失败: {str(e)}"},
                "theoretical_support": {"score": None, "reasoning": f"评估失败: {str(e)}"},
                "logical_alignment": {"score": None, "reasoning": f"评估失败: {str(e)}"}
            }
    
    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """You are an experienced research advisor evaluating research ideas with a discerning but fair perspective.

IMPORTANT SCORING PRINCIPLES:
- Use the FULL range of 1-5 scores to differentiate quality, INCLUDING DECIMAL SCORES (e.g., 4.6, 4.8, 4.9)
- Score 5.0 is reserved for truly exceptional, flawless ideas (rare but achievable)
- Scores 4.7-4.9 for excellent ideas with comprehensive step-by-step reasoning
- Score 4.5-4.6 for very good ideas with step structure but room for improvement
- Score 4.0-4.4 for solid ideas with detailed reasoning but no systematic structure
- Score 3.0-3.9 for adequate ideas with basic reasoning
- BE DISCERNING: Evaluate quality carefully and use decimals to show subtle differences

KEY DIFFERENTIATOR - Quality of Systematic Methodology:
- Ideas with "Step 1/Step 2/Step 3" structure qualify for 4.5-5.0 range
- Score 5.0: Exceptional - perfect step structure + outstanding rationale (>300 words) + profound insights
- Score 4.9: Excellent - strong step structure + comprehensive rationale (>250 words) + deep analysis
- Score 4.8: Very good - good step structure + solid rationale (>200 words) + clear analysis
- Score 4.7: Good - step structure present + adequate rationale (>150 words) + reasonable analysis
- Score 4.5-4.6: Acceptable - step structure present but rationale could be more comprehensive
- Score 4.0-4.4: Solid reasoning but NO step structure
- Score 3.0-3.9: Basic reasoning

Evaluate research ideas using a 1-5 scale (USE DECIMALS for precision) across four dimensions:

1. **Novelty (新颖性)**: Does this idea provide a unique perspective or innovative approach?
   - 5.0: Truly revolutionary - paradigm-shifting innovation
   - 4.7-4.9: Highly novel with significant advancement
   - 4.5-4.6: Clearly novel with meaningful innovation
   - 4.0-4.4: Moderate innovation with some incremental elements
   - 3.5-3.9: Some novelty, mostly incremental
   - 3.0-3.4: Limited novelty
   - Below 3.0: Minimal novelty

2. **Feasibility (可行性)**: Is this implementable with current technology and resources?
   - 5.0: Perfect implementation path with all details
   - 4.7-4.9: Highly feasible with comprehensive, clear path
   - 4.5-4.6: Very feasible with good detailed path
   - 4.0-4.4: Feasible with reasonable outline
   - 3.5-3.9: Moderately feasible, some details missing
   - 3.0-3.4: Feasible but vague
   - Below 3.0: Challenging or impractical

3. **Theoretical Support (理论支撑度)**: Is the limitation→method pairing logically sound?

   ⚠️ **MANDATORY RULE**: Check for "Step 1:", "Step 2:", "Step 3:" labels FIRST!
   - **If NO step labels → Score MUST be ≤ 4.4** (CANNOT exceed 4.4, even with excellent reasoning)
   - **If step labels present → Can score 4.5-5.0** based on quality

   SCORING CRITERIA (use decimals):

   **WITH "Step 1/2/3" labels (4.5-5.0 range):**
   - **5.0**: EXCEPTIONAL - Step labels + outstanding rationale (>300 words) + profound depth
   - **4.9**: EXCELLENT - Step labels + comprehensive rationale (>250 words) + deep analysis
   - **4.8**: VERY GOOD - Step labels + solid rationale (>200 words) + clear analysis
   - **4.7**: GOOD - Step labels + adequate rationale (>150 words) + reasonable analysis
   - **4.5-4.6**: ACCEPTABLE - Step labels present, rationale needs improvement

   **WITHOUT "Step 1/2/3" labels (MAX 4.4):**
   - **4.0-4.4**: SOLID - Good reasoning, well-explained, but NO step labels → **HARD CAP at 4.4**
   - **3.5-3.9**: ADEQUATE - Decent reasoning, lacks depth
   - **3.0-3.4**: BASIC - Basic reasoning
   - Below 3.0: Questionable

   ⚠️ **Evaluation Process (MUST FOLLOW IN ORDER)**:
   1. **FIRST**: Search reasoning field for exact strings "Step 1:", "Step 2:", "Step 3:"
      - **Found** → Eligible for 4.5-5.0, proceed to step 2
      - **Not found** → **HARD CAP at 4.4**, score 3.0-4.4 based on reasoning quality
   2. **IF** step structure present: Check rationale length (>300→5.0, >250→4.9, >200→4.8, >150→4.7)
   3. Assess theoretical depth
   4. Assign final score within allowed range

4. **Logical Alignment (逻辑契合度)**: Does the solution effectively address the problem?

   ⚠️ **MANDATORY RULE**: Check for multi-step logical chain FIRST!
   - **If NO complete chain → Score MUST be ≤ 4.4** (CANNOT exceed 4.4)
   - **If complete chain present → Can score 4.5-5.0** based on quality

   SCORING CRITERIA (use decimals):

   **WITH complete multi-step chain (limitation→compatibility→gap→solution→outcome) (4.5-5.0):**
   - **5.0**: EXCEPTIONAL - Complete chain + outstanding modification + perfect outcomes
   - **4.9**: EXCELLENT - Complete chain + comprehensive modification + clear outcomes
   - **4.8**: VERY GOOD - Complete chain + solid modification + good outcomes
   - **4.7**: GOOD - Complete chain + adequate modification + reasonable outcomes
   - **4.5-4.6**: ACCEPTABLE - Chain present, modification needs more detail

   **WITHOUT complete multi-step chain (MAX 4.4):**
   - **4.0-4.4**: SOLID - Clear explanation of HOW solution helps, but NO systematic chain → **HARD CAP at 4.4**
   - **3.5-3.9**: ADEQUATE - Good explanation, less detailed
   - **3.0-3.4**: BASIC - Basic explanation
   - Below 3.0: Fails to address

   ⚠️ **Evaluation Process (MUST FOLLOW IN ORDER)**:
   1. **FIRST**: Check if reasoning shows complete chain (limitation → compatibility analysis → gap identification → solution → expected outcome)
      - **Found** → Eligible for 4.5-5.0, proceed to step 2
      - **Not found** → **HARD CAP at 4.4**, score 3.0-4.4 based on explanation quality
   2. **IF** chain present: Assess modification detail (Outstanding→5.0, Comprehensive→4.9, Solid→4.8, Good→4.7)
   3. Check outcome prediction clarity
   4. Assign final score within allowed range

SCORING GUIDELINES:
- **USE DECIMAL SCORES (4.5, 4.6, 4.7, 4.8, 4.9) to differentiate quality levels**
- **For ideas with "Step 1/2/3" structure, typical excellent score is 4.7-4.9**
- **Score 5.0 is achievable but requires exceptional quality in all aspects**
- **Evaluate rationale depth, theoretical insights, and logical rigor carefully**
- **Without step structure, maximum is 4.4**
- **Quality scale**: Exceptional = 5.0, Excellent = 4.8-4.9, Very Good = 4.7, Good = 4.5-4.6

Provide your evaluation in JSON format with DECIMAL scores (typically 4.5-4.9 for good ideas with step structure) and brief reasoning."""

    def _build_evaluation_prompt(self, idea: Dict, context: Optional[Dict] = None) -> str:
        """构建评估提示词"""
        title = idea.get('title', 'N/A')
        abstract = idea.get('abstract', 'N/A')
        modification = idea.get('modification', 'N/A')
        reasoning = idea.get('reasoning', 'N/A')
        limitation = idea.get('limitation', 'N/A')
        method = idea.get('method', 'N/A')
        
        prompt_parts = [
            "**Research Idea to Evaluate:**",
            "",
            f"**Title:** {title}",
            "",
            f"**Abstract:** {abstract}",
            "",
            f"**Key Modification:** {modification}",
            "",
            f"**Reasoning:** {reasoning}",
        ]
        
        # 如果有完整的推演路径（rationale），添加到提示词中
        rationale = idea.get('rationale', '')
        if rationale and rationale != 'N/A' and len(rationale.strip()) > 0:
            prompt_parts.append("")
            prompt_parts.append("**Complete Rationale (Full Reasoning Path):**")
            # 限制长度，避免提示词过长
            if len(rationale) > 2000:
                rationale_short = rationale[:2000] + "..."
                prompt_parts.append(rationale_short)
                prompt_parts.append("")
                prompt_parts.append("(Note: Rationale truncated for brevity)")
            else:
                prompt_parts.append(rationale)
        
        prompt_parts.extend([
            "",
            f"**Original Limitation:** {limitation}",
            "",
            f"**Original Method:** {method}",
        ])

        # 添加上下文信息（如果有）
        if context:
            # 添加背景信息（如果有）
            if context.get('background'):
                prompt_parts.append("")
                prompt_parts.append("**Research Background:**")
                background_text = context.get('background', '')
                # 限制背景长度
                if len(background_text) > 1000:
                    background_text = background_text[:1000] + "..."
                prompt_parts.append(background_text)

            # 添加演化路径（如果有）
            if context.get('evolutionary_paths'):
                prompt_parts.append("")
                prompt_parts.append("**Evolutionary Context:**")
                prompt_parts.append("This idea was generated considering the following evolutionary paths:")
                for i, path in enumerate(context['evolutionary_paths'][:2], 1):
                    pattern_type = path.get('pattern_type', 'Unknown')
                    title_short = path.get('title', '')[:100]
                    prompt_parts.append(f"  {i}. {pattern_type}: {title_short}...")

        prompt_parts.append("")
        prompt_parts.append("**Evaluation Task:**")
        prompt_parts.append("")
        prompt_parts.append("⚠️ CRITICAL EVALUATION RULES - MUST FOLLOW STRICTLY:")
        prompt_parts.append("")
        prompt_parts.append("FOR THEORETICAL SUPPORT:")
        prompt_parts.append("1. FIRST: Search reasoning field for exact strings 'Step 1:', 'Step 2:', 'Step 3:'")
        prompt_parts.append("   - Found → Can score 4.5-5.0 based on quality")
        prompt_parts.append("   - NOT found → HARD CAP at 4.4 (even if reasoning is excellent)")
        prompt_parts.append("2. IF step labels found: Check rationale length (>300→5.0, >250→4.9, >200→4.8, >150→4.7)")
        prompt_parts.append("3. Assess theoretical depth within allowed range")
        prompt_parts.append("")
        prompt_parts.append("FOR LOGICAL ALIGNMENT:")
        prompt_parts.append("1. FIRST: Check if reasoning shows complete chain (limitation→compatibility→gap→solution→outcome)")
        prompt_parts.append("   - Complete chain → Can score 4.5-5.0")
        prompt_parts.append("   - No complete chain → HARD CAP at 4.4")
        prompt_parts.append("2. IF chain present: Assess modification detail (Outstanding→5.0, Comprehensive→4.9, Solid→4.8, Good→4.7)")
        prompt_parts.append("")
        prompt_parts.append("⚠️ REMEMBER: Without 'Step 1/2/3' labels OR without complete logical chain, scores CANNOT exceed 4.4!")
        prompt_parts.append("")
        prompt_parts.append("Evaluate using 1-5 scale with DECIMALS. Provide evaluation in JSON format:")
        prompt_parts.append("")
        prompt_parts.append("""{
  "novelty": {
    "score": <decimal 1-5, e.g., 4.7>,
    "reasoning": "<Assess novelty: 5.0=revolutionary, 4.7-4.9=highly novel, 4.5-4.6=clearly novel, 4.0-4.4=moderate>"
  },
  "feasibility": {
    "score": <decimal 1-5, e.g., 4.6>,
    "reasoning": "<Assess feasibility: 5.0=perfect, 4.7-4.9=comprehensive, 4.5-4.6=very feasible, 4.0-4.4=feasible>"
  },
  "theoretical_support": {
    "score": <decimal 1-5, e.g., 4.8 OR 4.2>,
    "reasoning": "<MANDATORY: Did you find 'Step 1/2/3' labels? If YES→score 4.5-5.0 based on quality. If NO→score MAX 4.4. State which case applies!>"
  },
  "logical_alignment": {
    "score": <decimal 1-5, e.g., 4.7 OR 4.3>,
    "reasoning": "<MANDATORY: Does reasoning show complete chain (limitation→compatibility→gap→solution→outcome)? If YES→score 4.5-5.0. If NO→score MAX 4.4. State which case applies!>"
  }
}""")
        
        return "\n".join(prompt_parts)
    
    def _parse_evaluation_response(self, response: str) -> Dict:
        """解析LLM响应"""
        try:
            # 尝试提取JSON部分
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                evaluation = json.loads(json_str)
                
                # 验证和规范化评分
                for dimension in ['novelty', 'feasibility', 'theoretical_support', 'logical_alignment']:
                    if dimension not in evaluation:
                        evaluation[dimension] = {"score": None, "reasoning": "Missing dimension"}
                    else:
                        score = evaluation[dimension].get('score')
                        if score is not None:
                            # 确保评分在1-5范围内，支持小数
                            score = max(1.0, min(5.0, float(score)))
                            # 四舍五入到一位小数
                            score = round(score, 1)
                            evaluation[dimension]['score'] = score
                
                return evaluation
            else:
                raise ValueError("未找到JSON格式的评估结果")
                
        except Exception as e:
            logger.error(f"解析评估响应失败: {e}")
            logger.error(f"响应内容: {response[:500]}")
            return {
                "novelty": {"score": None, "reasoning": f"解析失败: {str(e)}"},
                "feasibility": {"score": None, "reasoning": f"解析失败: {str(e)}"},
                "theoretical_support": {"score": None, "reasoning": f"解析失败: {str(e)}"},
                "logical_alignment": {"score": None, "reasoning": f"解析失败: {str(e)}"}
            }
    
    def batch_evaluate(
        self,
        ideas: List[Dict],
        context: Optional[Dict] = None,
        verbose: bool = True
    ) -> List[Dict]:
        """
        批量评估多个创意
        
        Args:
            ideas: 创意列表
            context: 可选的上下文信息
            verbose: 是否输出详细进度
        
        Returns:
            评估结果列表，每个结果包含原始创意和评估分数
        """
        results = []
        
        for i, idea in enumerate(ideas, 1):
            if verbose:
                logger.info(f"评估创意 {i}/{len(ideas)}: {idea.get('title', 'N/A')[:60]}...")
            
            evaluation = self.evaluate_idea(idea, context)
            
            # 合并原始创意和评估结果
            result = {
                **idea,
                "evaluation": evaluation,
                "average_score": self._calculate_average_score(evaluation)
            }
            
            results.append(result)
            
            if verbose:
                scores = evaluation
                logger.info(f"  评分: Novelty={scores.get('novelty', {}).get('score', 'N/A')}, "
                          f"Feasibility={scores.get('feasibility', {}).get('score', 'N/A')}, "
                          f"Theoretical_Support={scores.get('theoretical_support', {}).get('score', 'N/A')}, "
                          f"Logical_Alignment={scores.get('logical_alignment', {}).get('score', 'N/A')}")
        
        return results
    
    def _calculate_average_score(self, evaluation: Dict) -> Optional[float]:
        """计算平均分"""
        scores = []
        for dimension in ['novelty', 'feasibility', 'theoretical_support', 'logical_alignment']:
            score = evaluation.get(dimension, {}).get('score')
            if score is not None:
                scores.append(score)
        
        if scores:
            return sum(scores) / len(scores)
        return None


def evaluate_ideas_from_file(
    input_file: str,
    output_file: str,
    config_path: str = './config/config.yaml',
    context_file: Optional[str] = None
) -> Dict:
    """
    从文件加载创意并评估
    
    Args:
        input_file: 输入文件路径（JSON格式，包含ideas列表）
        output_file: 输出文件路径
        config_path: LLM配置文件路径
        context_file: 可选的上下文文件路径（如深度综述结果）
    
    Returns:
        统计信息字典，包含 total_ideas, evaluated_ideas, statistics
    """
    logger.info("=" * 80)
    logger.info("开始评估科研创意")
    logger.info("=" * 80)
    
    # 加载创意数据
    logger.info(f"\n📂 加载创意数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统一输入格式处理
    # 期望格式: {"ideas": ["idea text 1", "idea text 2", ...]}
    # 或简化为: ["idea text 1", "idea text 2", ...]

    ideas = []

    if isinstance(data, dict) and 'ideas' in data:
        # 标准格式: {"ideas": [...]}
        idea_list = data.get('ideas', [])
        logger.info(f"  检测到标准格式，包含 {len(idea_list)} 个创意")
    elif isinstance(data, list):
        # 简化格式: 直接是列表
        idea_list = data
        logger.info(f"  检测到列表格式，包含 {len(idea_list)} 个创意")
    else:
        raise ValueError(f"不支持的数据格式: 期望 dict 或 list，得到 {type(data)}")

    # 统一转换为标准结构
    for i, item in enumerate(idea_list, 1):
        if isinstance(item, str):
            # 如果是纯字符串，作为 abstract
            idea_item = {
                'title': f'Idea {i}',
                'abstract': item,
                'modification': '',
                'reasoning': '',
                'limitation': '',
                'method': ''
            }
        elif isinstance(item, dict):
            # 如果是字典，提取所有可用字段
            idea_item = {
                'title': item.get('title', f'Idea {i}'),
                'abstract': item.get('abstract', item.get('idea', '')),
                'modification': item.get('modification', ''),
                'reasoning': item.get('reasoning', ''),
                'limitation': item.get('limitation', ''),
                'method': item.get('method', ''),
                'rationale': item.get('rationale', '')
            }
        else:
            logger.warning(f"  跳过无效的创意项 {i}: {type(item)}")
            continue

        ideas.append(idea_item)

    logger.info(f"  ✅ 成功加载 {len(ideas)} 个创意")

    # 加载上下文（如果有）
    context = None
    if context_file and Path(context_file).exists():
        logger.info(f"\n📂 加载上下文数据: {context_file}")
        with open(context_file, 'r', encoding='utf-8') as f:
            context_data = json.load(f)

            if isinstance(context_data, str):
                # 纯文本背景
                context = {'background': context_data}
                logger.info("  加载了文本背景信息")
            elif isinstance(context_data, dict):
                # 结构化上下文
                context = {
                    'background': context_data.get('background', ''),
                    'evolutionary_paths': context_data.get('evolutionary_paths', [])
                }
                logger.info(f"  加载了背景信息和 {len(context.get('evolutionary_paths', []))} 条演化路径")
            else:
                context = {'background': str(context_data)}
                logger.info("  加载了上下文信息")
    
    # 初始化评估器
    logger.info(f"\n🔧 初始化评估器")
    evaluator = IdeaEvaluator(config_path=config_path)
    
    # 批量评估
    logger.info(f"\n💡 开始批量评估")
    results = evaluator.batch_evaluate(ideas, context=context, verbose=True)
    
    # 计算统计信息
    logger.info(f"\n📊 评估统计:")
    valid_scores = {
        'novelty': [],
        'feasibility': [],
        'theoretical_support': [],
        'logical_alignment': [],
        'average': []
    }
    
    for result in results:
        eval_data = result.get('evaluation', {})
        for dimension in ['novelty', 'feasibility', 'theoretical_support', 'logical_alignment']:
            score = eval_data.get(dimension, {}).get('score')
            if score is not None:
                valid_scores[dimension].append(score)
        
        avg_score = result.get('average_score')
        if avg_score is not None:
            valid_scores['average'].append(avg_score)
    
    # 维度名称映射（用于显示）
    dimension_names = {
        'novelty': 'Novelty (新颖性)',
        'feasibility': 'Feasibility (可行性)',
        'theoretical_support': 'Theoretical Support (理论支撑度)',
        'logical_alignment': 'Logical Alignment (逻辑契合度)',
        'average': 'Average (平均分)'
    }
    
    for dimension, scores in valid_scores.items():
        if scores:
            avg = sum(scores) / len(scores)
            dim_name = dimension_names.get(dimension, dimension.capitalize())
            logger.info(f"  {dim_name}: 平均分 = {avg:.2f} (共 {len(scores)} 个有效评分)")
    
    # 保存结果
    logger.info(f"\n💾 保存评估结果: {output_file}")
    output_data = {
        'total_ideas': len(ideas),
        'evaluated_ideas': len(results),
        'statistics': {
            dimension: {
                'average': sum(scores) / len(scores) if scores else None,
                'count': len(scores)
            }
            for dimension, scores in valid_scores.items()
        },
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info("✅ 评估完成！")
    
    # 返回统计信息，用于批量评估汇总
    return {
        'total_ideas': len(ideas),
        'evaluated_ideas': len(results),
        'statistics': {
            dimension: {
                'average': sum(scores) / len(scores) if scores else None,
                'count': len(scores)
            }
            for dimension, scores in valid_scores.items()
        }
    }


def batch_evaluate_presets(
    preset_names: List[str],
    eval_config: Dict,
    project_root: Path,
    config_path: str = './config/config.yaml',
    output_dir: Optional[Path] = None
) -> Dict:
    """
    批量评估多个预设
    
    Args:
        preset_names: 预设名称列表
        eval_config: 评估配置
        project_root: 项目根目录
        config_path: LLM配置文件路径
        output_dir: 输出目录（可选）
    
    Returns:
        汇总结果字典
    """
    if output_dir is None:
        output_dir = Path(__file__).parent
    
    presets = eval_config.get('input_presets', {})
    results = []
    
    logger.info("=" * 80)
    logger.info(f"🚀 开始批量评估 {len(preset_names)} 个预设")
    logger.info("=" * 80)
    
    for i, preset_name in enumerate(preset_names, 1):
        if preset_name not in presets:
            logger.warning(f"⚠️  预设 '{preset_name}' 不存在，跳过")
            continue
        
        preset = presets[preset_name]
        input_file = str(project_root / preset.get('input_file'))
        context_file = preset.get('context_file')
        if context_file:
            context_file = str(project_root / context_file)
        
        # 检查文件是否存在
        if not Path(input_file).exists():
            logger.warning(f"⚠️  输入文件不存在: {input_file}，跳过预设 '{preset_name}'")
            continue
        
        logger.info(f"\n{'='*80}")
        logger.info(f"[{i}/{len(preset_names)}] 评估预设: {preset_name}")
        logger.info(f"{'='*80}")
        logger.info(f"描述: {preset.get('description', '')}")
        logger.info(f"输入文件: {input_file}")
        if context_file:
            logger.info(f"上下文文件: {context_file}")
        
        # 生成输出文件名
        input_path = Path(input_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = str(output_dir / f"{preset_name}_evaluation_{timestamp}.json")
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 评估
            stats = evaluate_ideas_from_file(
                input_file=input_file,
                output_file=output_file,
                config_path=config_path,
                context_file=context_file
            )
            
            # 计算耗时
            duration = time.time() - start_time
            
            # 收集结果
            result = {
                'preset_name': preset_name,
                'description': preset.get('description', ''),
                'input_file': input_file,
                'output_file': output_file,
                'duration_seconds': round(duration, 1),
                **stats
            }
            results.append(result)
            
            logger.info(f"✅ 预设 '{preset_name}' 评估完成，耗时: {duration:.1f} 秒")
            
        except Exception as e:
            logger.error(f"❌ 预设 '{preset_name}' 评估失败: {e}", exc_info=True)
            continue
    
    # 生成汇总表格
    if results:
        logger.info(f"\n{'='*80}")
        logger.info("📊 生成评估结果汇总")
        logger.info(f"{'='*80}")
        
        summary = generate_comparison_tables(results, output_dir)
        
        return {
            'total_presets': len(preset_names),
            'evaluated_presets': len(results),
            'results': results,
            'summary': summary
        }
    else:
        logger.warning("⚠️  没有成功评估的预设")
        return {
            'total_presets': len(preset_names),
            'evaluated_presets': 0,
            'results': [],
            'summary': None
        }


def generate_comparison_tables(results: List[Dict], output_dir: Path) -> Dict:
    """
    生成对比表格（控制台、Markdown、CSV）
    
    Args:
        results: 评估结果列表
        output_dir: 输出目录
    
    Returns:
        汇总信息字典
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 准备表格数据
    table_data = []
    dimension_names = {
        'novelty': 'Novelty (新颖性)',
        'feasibility': 'Feasibility (可行性)',
        'theoretical_support': 'Theoretical Support (理论支撑度)',
        'logical_alignment': 'Logical Alignment (逻辑契合度)',
        'average': 'Average (平均分)'
    }
    
    # 找出各维度的最高分
    max_scores = {
        'novelty': -1,
        'feasibility': -1,
        'theoretical_support': -1,
        'logical_alignment': -1,
        'average': -1
    }
    max_presets = {dim: [] for dim in max_scores.keys()}
    
    for result in results:
        stats = result.get('statistics', {})
        row = {
            'preset_name': result['preset_name'],
            'description': result['description'],
            'ideas_count': result.get('total_ideas', 0),
            'novelty': stats.get('novelty', {}).get('average'),
            'feasibility': stats.get('feasibility', {}).get('average'),
            'theoretical_support': stats.get('theoretical_support', {}).get('average'),
            'logical_alignment': stats.get('logical_alignment', {}).get('average'),
            'average': stats.get('average', {}).get('average'),
            'duration': result.get('duration_seconds', 0)
        }
        table_data.append(row)
        
        # 更新最高分
        for dim in max_scores.keys():
            score = row.get(dim)
            if score is not None:
                if score > max_scores[dim]:
                    max_scores[dim] = score
                    max_presets[dim] = [result['preset_name']]
                elif abs(score - max_scores[dim]) < 0.01:  # 允许小的浮点误差
                    max_presets[dim].append(result['preset_name'])
    
    # 1. 控制台表格
    print("\n" + "=" * 100)
    print("📊 评估结果对比表格:")
    print("=" * 100)
    print(f"{'预设名称':<15} {'创意数':<8} {'新颖性':<10} {'可行性':<10} {'理论支撑度':<12} {'逻辑契合度':<12} {'总体平均':<10} {'耗时(秒)':<10}")
    print("-" * 100)
    
    for row in table_data:
        novelty = f"{row['novelty']:.2f}" if row['novelty'] is not None else "N/A"
        feasibility = f"{row['feasibility']:.2f}" if row['feasibility'] is not None else "N/A"
        theoretical_support = f"{row['theoretical_support']:.2f}" if row['theoretical_support'] is not None else "N/A"
        logical_alignment = f"{row['logical_alignment']:.2f}" if row['logical_alignment'] is not None else "N/A"
        average = f"{row['average']:.2f}" if row['average'] is not None else "N/A"
        
        print(f"{row['preset_name']:<15} {row['ideas_count']:<8} {novelty:<10} {feasibility:<10} {theoretical_support:<12} {logical_alignment:<12} {average:<10} {row['duration']:<10.1f}")
    
    print("=" * 100)
    
    # 标注最高分
    print("\n📌 最高分标注:")
    for dim, preset_list in max_presets.items():
        if preset_list and max_scores[dim] > 0:
            dim_name = dimension_names.get(dim, dim)
            print(f"  - {', '.join(preset_list)}: {dim_name}最高 ({max_scores[dim]:.2f})")
    
    # 2. Markdown 表格
    md_file = output_dir / f"evaluation_comparison_{timestamp}.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# 科研创意评估结果对比表\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| 预设名称 | 描述 | 创意数 | 新颖性 | 可行性 | 理论支撑度 | 逻辑契合度 | 总体平均 | 耗时(秒) |\n")
        f.write("|---------|------|--------|--------|--------|------------|------------|----------|----------|\n")
        
        for row in table_data:
            novelty = f"{row['novelty']:.2f}" if row['novelty'] is not None else "N/A"
            feasibility = f"{row['feasibility']:.2f}" if row['feasibility'] is not None else "N/A"
            theoretical_support = f"{row['theoretical_support']:.2f}" if row['theoretical_support'] is not None else "N/A"
            logical_alignment = f"{row['logical_alignment']:.2f}" if row['logical_alignment'] is not None else "N/A"
            average = f"{row['average']:.2f}" if row['average'] is not None else "N/A"
            
            # 转义描述中的管道符
            description = row['description'].replace('|', '\\|')
            
            f.write(f"| {row['preset_name']} | {description} | {row['ideas_count']} | {novelty} | {feasibility} | {theoretical_support} | {logical_alignment} | {average} | {row['duration']:.1f} |\n")
        
        f.write("\n## 评分说明\n\n")
        f.write("各维度评分范围：1-5 分（李克特量表）\n\n")
        f.write("- **新颖性 (Novelty)**: 该想法是否提供了区别于现有文献的独特视角？\n")
        f.write("- **可行性 (Feasibility)**: 基于现有技术栈，该技术路线是否逻辑自洽且可实现？\n")
        f.write("- **理论支撑度 (Theoretical Support)**: 评估该假设是否提供了充分的依据，符合领域发展的历史惯性。\n")
        f.write("- **逻辑契合度 (Logical Alignment)**: 评估生成的解决方案是否精准且实质性地解决了提出的科研痛点。\n")
        f.write("- **总体平均**: 四个维度的平均分\n\n")
        
        f.write("## 最高分标注\n\n")
        for dim, preset_list in max_presets.items():
            if preset_list and max_scores[dim] > 0:
                dim_name = dimension_names.get(dim, dim)
                f.write(f"- **{', '.join(preset_list)}**: 🏆 {dim_name}最高 ({max_scores[dim]:.2f})\n")
    
    logger.info(f"📄 Markdown 表格已保存: {md_file}")
    
    # 3. CSV 表格
    csv_file = output_dir / f"evaluation_comparison_{timestamp}.csv"
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['preset_name', 'description', 'ideas_count', 'novelty', 'feasibility', 
                         'theoretical_support', 'logical_alignment', 'average', 'duration_seconds'])
        
        for row in table_data:
            writer.writerow([
                row['preset_name'],
                row['description'],
                row['ideas_count'],
                row['novelty'] if row['novelty'] is not None else '',
                row['feasibility'] if row['feasibility'] is not None else '',
                row['theoretical_support'] if row['theoretical_support'] is not None else '',
                row['logical_alignment'] if row['logical_alignment'] is not None else '',
                row['average'] if row['average'] is not None else '',
                row['duration']
            ])
    
    logger.info(f"📊 CSV 表格已保存: {csv_file}")
    
    # 4. JSON 汇总
    json_file = output_dir / f"batch_evaluation_summary_{timestamp}.json"
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'total_presets': len(results),
        'table_data': table_data,
        'max_scores': max_scores,
        'max_presets': {k: v for k, v in max_presets.items() if v},
        'files': {
            'markdown': str(md_file),
            'csv': str(csv_file),
            'json': str(json_file)
        }
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"📋 JSON 汇总已保存: {json_file}")
    
    return summary_data


def load_eval_config(config_path: str = './idea_eval/eval_config.yaml') -> Dict:
    """
    加载评估配置文件

    Args:
        config_path: 评估配置文件路径

    Returns:
        评估配置字典
    """
    config_file = Path(config_path)

    if not config_file.exists():
        logger.warning(f"评估配置文件不存在: {config_path}")
        return {}

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config if config else {}
    except Exception as e:
        logger.error(f"加载评估配置文件失败: {e}")
        return {}


def list_available_inputs(eval_config: Dict, project_root: Path) -> Dict[str, Dict]:
    """
    列出所有可用的输入文件预设

    Args:
        eval_config: 评估配置
        project_root: 项目根目录

    Returns:
        输入文件预设字典 {name: {input_file, context_file, description}}
    """
    presets = eval_config.get('input_presets', {})
    available = {}

    for name, preset in presets.items():
        input_file = str(project_root / preset.get('input_file', ''))
        context_file = preset.get('context_file')
        if context_file:
            context_file = str(project_root / context_file)

        # 检查文件是否存在
        exists = Path(input_file).exists()
        available[name] = {
            'input_file': input_file,
            'context_file': context_file,
            'description': preset.get('description', ''),
            'exists': exists
        }

    return available


def interactive_select_input(available_inputs: Dict[str, Dict]) -> tuple[str, str, str]:
    """
    交互式选择输入文件

    Args:
        available_inputs: 可用输入文件字典

    Returns:
        (input_file, context_file, description)
    """
    print("\n" + "=" * 80)
    print("📋 可用的输入文件预设:")
    print("=" * 80)

    if not available_inputs:
        print("❌ 没有找到预设配置")
        return None, None, None

    # 显示所有预设
    preset_list = list(available_inputs.items())
    for i, (name, info) in enumerate(preset_list, 1):
        status = "✅" if info['exists'] else "❌"
        print(f"{i}. [{status}] {name}")
        print(f"   描述: {info['description']}")
        print(f"   输入: {info['input_file']}")
        if info['context_file']:
            print(f"   上下文: {info['context_file']}")
        print()

    # 用户选择
    while True:
        try:
            choice = input(f"请选择要评估的输入文件 [1-{len(preset_list)}] 或按 Enter 使用默认: ").strip()

            if not choice:
                # 使用第一个存在的预设
                for name, info in preset_list:
                    if info['exists']:
                        print(f"\n✅ 使用默认预设: {name}")
                        return info['input_file'], info['context_file'], info['description']
                return None, None, None

            idx = int(choice) - 1
            if 0 <= idx < len(preset_list):
                name, info = preset_list[idx]
                if not info['exists']:
                    print(f"⚠️  警告: 文件不存在，但仍将继续")
                print(f"\n✅ 选择了预设: {name}")
                return info['input_file'], info['context_file'], info['description']
            else:
                print(f"❌ 请输入 1-{len(preset_list)} 之间的数字")
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n❌ 已取消")
            sys.exit(0)


if __name__ == "__main__":
    import argparse

    # 获取脚本所在目录的父目录（项目根目录）
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='科研创意评估工具 - 对生成的创意进行新颖性、可行性、基础性评分',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 交互式选择输入文件
  python evaluate.py --interactive

  # 使用单个预设配置
  python evaluate.py --preset test_result

  # 批量评估多个预设
  python evaluate.py --presets test_result,CoIresult,MOOSEChem

  # 评估所有预设
  python evaluate.py --all-presets

  # 直接指定输入文件
  python evaluate.py --input ideas.json --output results.json

  # 指定输入和上下文文件
  python evaluate.py -i ideas.json -c context.json -o results.json
        """
    )

    parser.add_argument('-i', '--input', type=str, help='输入文件路径（JSON格式）')
    parser.add_argument('-o', '--output', type=str, help='输出文件路径')
    parser.add_argument('-c', '--context', type=str, help='上下文文件路径（可选）')
    parser.add_argument('--config', type=str, default=str(project_root / 'config' / 'config.yaml'),
                       help='LLM配置文件路径')
    parser.add_argument('--eval-config', type=str, default=str(script_dir / 'eval_config.yaml'),
                       help='评估配置文件路径')
    parser.add_argument('--preset', type=str, help='使用预设配置（在eval_config.yaml中定义）')
    parser.add_argument('--presets', type=str, help='批量评估多个预设，用逗号分隔（例如: preset1,preset2,preset3）')
    parser.add_argument('--all-presets', action='store_true', help='评估所有可用的预设')
    parser.add_argument('--interactive', '-I', action='store_true', help='交互式选择输入文件')
    parser.add_argument('--list-presets', action='store_true', help='列出所有可用的预设配置')

    args = parser.parse_args()

    # 加载评估配置
    eval_config = load_eval_config(args.eval_config)

    # 列出预设配置
    if args.list_presets:
        available = list_available_inputs(eval_config, project_root)
        print("\n" + "=" * 80)
        print("📋 可用的输入文件预设:")
        print("=" * 80)
        for name, info in available.items():
            status = "✅ 存在" if info['exists'] else "❌ 不存在"
            print(f"\n预设名称: {name} [{status}]")
            print(f"描述: {info['description']}")
            print(f"输入文件: {info['input_file']}")
            if info['context_file']:
                print(f"上下文文件: {info['context_file']}")
        print()
        sys.exit(0)

    # 批量评估多个预设
    if args.presets or args.all_presets:
        if args.all_presets:
            # 评估所有预设
            presets = eval_config.get('input_presets', {})
            preset_names = list(presets.keys())
            logger.info(f"📋 将评估所有 {len(preset_names)} 个预设: {', '.join(preset_names)}")
        else:
            # 评估指定的预设
            preset_names = [p.strip() for p in args.presets.split(',')]
            logger.info(f"📋 将评估 {len(preset_names)} 个预设: {', '.join(preset_names)}")
        
        try:
            summary = batch_evaluate_presets(
                preset_names=preset_names,
                eval_config=eval_config,
                project_root=project_root,
                config_path=args.config
            )
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ 批量评估完成！")
            logger.info("=" * 80)
            logger.info(f"成功评估: {summary['evaluated_presets']}/{summary['total_presets']} 个预设")
            
            if summary.get('summary'):
                logger.info(f"\n📊 汇总文件:")
                logger.info(f"  - Markdown: {summary['summary']['files']['markdown']}")
                logger.info(f"  - CSV: {summary['summary']['files']['csv']}")
                logger.info(f"  - JSON: {summary['summary']['files']['json']}")
            
        except Exception as e:
            logger.error(f"❌ 批量评估失败: {e}", exc_info=True)
            sys.exit(1)
        
        sys.exit(0)

    # 确定输入输出文件
    input_file = None
    output_file = None
    context_file = None

    # 1. 交互式选择
    if args.interactive:
        available = list_available_inputs(eval_config, project_root)
        input_file, context_file, description = interactive_select_input(available)
        if not input_file:
            logger.error("❌ 未选择有效的输入文件")
            sys.exit(1)

    # 2. 使用预设
    elif args.preset:
        presets = eval_config.get('input_presets', {})
        if args.preset not in presets:
            logger.error(f"❌ 未找到预设配置: {args.preset}")
            logger.info(f"可用预设: {', '.join(presets.keys())}")
            sys.exit(1)

        preset = presets[args.preset]
        input_file = str(project_root / preset.get('input_file'))
        context_file = preset.get('context_file')
        if context_file:
            context_file = str(project_root / context_file)
        logger.info(f"✅ 使用预设配置: {args.preset}")
        logger.info(f"   描述: {preset.get('description', '')}")

    # 3. 使用命令行参数
    elif args.input:
        input_file = args.input
        context_file = args.context

    # 4. 使用默认值
    else:
        default_preset = eval_config.get('default_preset')
        if default_preset and default_preset in eval_config.get('input_presets', {}):
            preset = eval_config['input_presets'][default_preset]
            input_file = str(project_root / preset.get('input_file'))
            context_file = preset.get('context_file')
            if context_file:
                context_file = str(project_root / context_file)
            logger.info(f"✅ 使用默认预设: {default_preset}")
        else:
            # 最后的后备方案
            input_file = str(project_root / "test_idea_generation_result.json")
            context_file = str(project_root / "226_survey_Natural_Language_Processing_20251218_202205.json")
            logger.warning("⚠️  使用硬编码默认文件")

    # 确定输出文件
    if args.output:
        output_file = args.output
    else:
        # 根据输入文件名生成输出文件名
        input_path = Path(input_file)
        output_file = str(script_dir / f"{input_path.stem}_evaluation_result.json")

    # 验证输入文件存在
    if not Path(input_file).exists():
        logger.error(f"❌ 输入文件不存在: {input_file}")
        sys.exit(1)

    # 显示配置信息
    print("\n" + "=" * 80)
    print("📋 评估配置:")
    print("=" * 80)
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"上下文文件: {context_file if context_file else '(无)'}")
    print(f"LLM配置: {args.config}")
    print("=" * 80 + "\n")

    try:
        evaluate_ideas_from_file(
            input_file=input_file,
            output_file=output_file,
            config_path=args.config,
            context_file=context_file
        )
    except Exception as e:
        logger.error(f"❌ 评估失败: {e}", exc_info=True)
        sys.exit(1)

