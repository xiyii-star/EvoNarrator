"""
Naive LLM 方法：直接使用 prompt 让大模型生成科研创意
不依赖知识图谱、演化路径等复杂信息，仅基于主题生成创意
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
import yaml
import sys
from datetime import datetime

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


def generate_idea_with_naive_llm(
    topic: str = "Natural Language Processing",
    config_path: str = './config/config.yaml',
    num_ideas: int = 1
) -> Dict:
    """
    使用 Naive LLM 方法生成科研创意
    
    该方法直接使用 prompt 让大模型基于主题生成科研创意，不依赖知识图谱、
    演化路径等复杂信息。适合作为 baseline 方法。
    
    Args:
        topic: 研究主题，默认为 "Natural Language Processing"
        config_path: LLM配置文件路径
        num_ideas: 要生成的创意数量，默认为 1
    
    Returns:
        包含生成的创意和背景信息的字典：
        {
            "topic": str,
            "ideas": [
                {
                    "idea": str,  # 生成的科研创意
                    "background": str  # 研究背景
                }
            ]
        }
    """
    logger.info("=" * 80)
    logger.info(f"🚀 使用 Naive LLM 方法生成科研创意")
    logger.info("=" * 80)
    logger.info(f"主题: {topic}")
    logger.info(f"生成数量: {num_ideas}")
    
    # 加载配置
    full_config = load_full_config(config_path)
    llm_config_dict = full_config.get('llm', {})
    
    # 构建LLMConfig
    llm_config = LLMConfig(
        provider=llm_config_dict.get('provider', 'openai'),
        model=llm_config_dict.get('model', 'gpt-4o'),
        api_key=llm_config_dict.get('api_key') or full_config.get('openai_api_key'),
        base_url=llm_config_dict.get('base_url') or full_config.get('openai_base_url'),
        temperature=llm_config_dict.get('temperature', 0.7),  # 使用稍高的温度以增加创造性
        max_tokens=llm_config_dict.get('max_tokens', 2000),
        timeout=llm_config_dict.get('timeout', 60)
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
    llm_client = LLMClient(llm_config)
    logger.info(f"✅ LLM客户端初始化成功，模型: {llm_config.model}")
    
    # 构建系统提示词
    system_prompt = """You are an expert research scientist with deep knowledge in academic research methodology and scientific innovation.

Your task is to generate novel and feasible research ideas based on a given research topic. You should provide:
1. A comprehensive research background that explains the current state of the field
2. A novel research idea that addresses an important problem or gap in the field

Your ideas should be:
- Novel: Provide a unique perspective distinct from existing literature
- Feasible: Logically coherent and achievable with current technology
- Well-grounded: Based on solid theoretical foundations and aligned with field evolution
- Clearly articulated: With specific technical approaches and expected outcomes"""

    # 生成创意
    ideas = []
    
    for i in range(num_ideas):
        logger.info(f"\n生成创意 {i+1}/{num_ideas}...")
        
        # 构建用户提示词
        user_prompt = f"""Please generate a novel research idea in the field of **{topic}**.

**Requirements:**
1. **Background (研究背景)**: Provide a comprehensive background (2-3 paragraphs) that:
   - Describes the current state of research in {topic}
   - Identifies key challenges, limitations, or gaps in existing approaches
   - Explains why addressing these issues is important

2. **Research Idea (科研创意)**: Propose a novel research idea (1-2 paragraphs) that:
   - Addresses a specific problem or gap identified in the background
   - Provides a clear technical approach or methodology
   - Explains the expected contributions and outcomes
   - Demonstrates novelty compared to existing work

**Output Format:**
Please provide your response in the following JSON format:
{{
    "background": "<comprehensive research background in 2-3 paragraphs>",
    "idea": "<novel research idea in 1-2 paragraphs, including technical approach and expected outcomes>"
}}

Make sure the JSON is valid and properly formatted."""

        try:
            # 调用LLM生成
            response = llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=2000
            )
            
            # 尝试解析JSON响应
            try:
                # 提取JSON部分（可能包含markdown代码块）
                import re
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    json_str = json_match.group(0)
                    idea_data = json.loads(json_str)
                    
                    ideas.append({
                        "idea": idea_data.get("idea", response),
                        "background": idea_data.get("background", "")
                    })
                    logger.info(f"✅ 创意 {i+1} 生成成功")
                else:
                    # 如果没有找到JSON，尝试直接使用响应
                    logger.warning(f"⚠️  无法解析JSON，使用原始响应")
                    ideas.append({
                        "idea": response,
                        "background": ""
                    })
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️  JSON解析失败: {e}，使用原始响应")
                # 尝试智能分割：假设前半部分是background，后半部分是idea
                parts = response.split("\n\n")
                if len(parts) >= 2:
                    ideas.append({
                        "background": "\n\n".join(parts[:-1]),
                        "idea": parts[-1]
                    })
                else:
                    ideas.append({
                        "background": "",
                        "idea": response
                    })
            
        except Exception as e:
            logger.error(f"❌ 生成创意 {i+1} 时出错: {e}")
            ideas.append({
                "idea": f"生成失败: {str(e)}",
                "background": ""
            })
    
    # 构建结果
    result = {
        "topic": topic,
        "method": "naive_llm",
        "timestamp": datetime.now().isoformat(),
        "num_ideas": len(ideas),
        "ideas": ideas
    }
    
    logger.info(f"\n✅ 成功生成 {len(ideas)} 个创意")
    
    return result


def save_result(result: Dict, output_file: Optional[str] = None) -> str:
    """
    保存生成结果到文件
    
    Args:
        result: 生成结果字典
        output_file: 输出文件路径（可选，默认自动生成）
    
    Returns:
        输出文件路径
    """
    if output_file is None:
        topic_safe = result["topic"].replace(" ", "_").replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = str(Path(__file__).parent / f"naive_llm_{topic_safe}_{timestamp}.json")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"💾 结果已保存: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Naive LLM 方法：直接使用 prompt 生成科研创意',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 生成单个创意（默认主题：Natural Language Processing）
  python naive_llm.py

  # 指定主题和数量
  python naive_llm.py --topic "Machine Learning" --num-ideas 3

  # 指定输出文件
  python naive_llm.py --output my_ideas.json
        """
    )
    
    parser.add_argument('--topic', type=str, default='Natural Language Processing',
                       help='研究主题（默认: Natural Language Processing）')
    parser.add_argument('--num-ideas', type=int, default=1,
                       help='要生成的创意数量（默认: 1）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径（可选，默认自动生成）')
    parser.add_argument('--config', type=str, default='./config/config.yaml',
                       help='LLM配置文件路径')
    
    args = parser.parse_args()
    
    try:
        # 生成创意
        result = generate_idea_with_naive_llm(
            topic=args.topic,
            config_path=args.config,
            num_ideas=args.num_ideas
        )
        
        # 保存结果
        output_file = save_result(result, args.output)
        
        # 显示结果摘要
        print("\n" + "=" * 80)
        print("📋 生成结果摘要:")
        print("=" * 80)
        print(f"主题: {result['topic']}")
        print(f"方法: {result['method']}")
        print(f"生成数量: {result['num_ideas']}")
        print(f"输出文件: {output_file}")
        print("\n生成的创意:")
        for i, idea_item in enumerate(result['ideas'], 1):
            print(f"\n  [{i}] Idea:")
            print(f"      {idea_item['idea'][:200]}...")
            if idea_item.get('background'):
                print(f"      Background: {idea_item['background'][:200]}...")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ 生成失败: {e}", exc_info=True)
        sys.exit(1)

