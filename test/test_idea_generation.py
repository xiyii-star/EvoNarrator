"""
测试科研创意生成功能
使用已生成的数据文件测试 research_idea_generator with survey.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import networkx as nx
import yaml

# 配置日志（需要在其他函数之前）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入 research_idea_generator with survey.py
import importlib.util
_research_idea_spec = importlib.util.spec_from_file_location(
    "research_idea_generator_with_survey",
    str(Path(__file__).parent / "src" / "research_idea_generator with survey.py")
)
if _research_idea_spec and _research_idea_spec.loader:
    _research_idea_module = importlib.util.module_from_spec(_research_idea_spec)
    _research_idea_spec.loader.exec_module(_research_idea_module)
    ResearchIdeaGenerator = _research_idea_module.ResearchIdeaGenerator
else:
    raise ImportError("无法导入 research_idea_generator with survey.py")


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


def load_papers_data(papers_file: str) -> List[Dict]:
    """
    加载论文数据
    
    Args:
        papers_file: 论文数据JSON文件路径
    
    Returns:
        论文列表
    """
    logger.info(f"加载论文数据: {papers_file}")
    with open(papers_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    logger.info(f"  成功加载 {len(papers)} 篇论文")
    return papers


def load_graph_data(graph_file: str) -> nx.DiGraph:
    """
    从JSON文件加载知识图谱数据并构建NetworkX图
    
    Args:
        graph_file: 图数据JSON文件路径
    
    Returns:
        NetworkX有向图
    """
    logger.info(f"加载图数据: {graph_file}")
    
    with open(graph_file, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加节点
    nodes = graph_data.get('nodes', [])
    logger.info(f"  添加 {len(nodes)} 个节点...")
    
    for node in nodes:
        node_id = node.get('id')
        if not node_id:
            continue
        
        # 提取节点属性
        node_attrs = {
            'title': node.get('title', ''),
            'year': node.get('year', 0),
            'cited_by_count': node.get('cited_by_count', 0),
            'venue': node.get('venue', ''),
            'is_open_access': node.get('is_open_access', False),
            'is_seed': node.get('is_seed', False),
        }
        
        # 提取RAG分析结果（兼容多种格式）
        deep_analysis = node.get('deep_analysis', {})
        rag_analysis = node.get('rag_analysis', {})
        
        # 优先使用deep_analysis，如果没有则使用rag_analysis
        if deep_analysis:
            node_attrs['rag_problem'] = deep_analysis.get('problem', '')
            node_attrs['rag_method'] = deep_analysis.get('method', '')
            node_attrs['rag_limitation'] = deep_analysis.get('limitation', '')
            node_attrs['rag_future_work'] = deep_analysis.get('future_work', '')
        elif rag_analysis:
            node_attrs['rag_problem'] = rag_analysis.get('problem', '')
            node_attrs['rag_method'] = rag_analysis.get('method', '')
            node_attrs['rag_limitation'] = rag_analysis.get('limitation', '')
            node_attrs['rag_future_work'] = rag_analysis.get('future_work', '')
        else:
            # 如果没有分析结果，使用空字符串
            node_attrs['rag_problem'] = ''
            node_attrs['rag_method'] = ''
            node_attrs['rag_limitation'] = ''
            node_attrs['rag_future_work'] = ''
        
        G.add_node(node_id, **node_attrs)
    
    # 添加边
    edges = graph_data.get('edges', [])
    logger.info(f"  添加 {len(edges)} 条边...")
    
    for edge in edges:
        source = edge.get('source') or edge.get('from')
        target = edge.get('target') or edge.get('to')
        edge_type = edge.get('edge_type') or edge.get('type', 'CITES')
        
        if source and target:
            G.add_edge(source, target, edge_type=edge_type, weight=1)
    
    logger.info(f"✅ 图构建完成: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
    return G


def load_survey_data(survey_file: str) -> Dict:
    """
    加载深度综述数据
    
    Args:
        survey_file: 深度综述JSON文件路径
    
    Returns:
        深度综述数据字典
    """
    logger.info(f"加载深度综述数据: {survey_file}")
    with open(survey_file, 'r', encoding='utf-8') as f:
        survey_data = json.load(f)
    
    # 提取演化路径
    evolutionary_paths = survey_data.get('evolutionary_paths', [])
    logger.info(f"  提取到 {len(evolutionary_paths)} 条演化路径")
    
    return survey_data


def test_idea_generation(
    papers_file: str,
    graph_file: str,
    survey_file: str,
    config: Optional[Dict] = None
) -> Dict:
    """
    测试科研创意生成
    
    Args:
        papers_file: 论文数据JSON文件路径
        graph_file: 图数据JSON文件路径
        survey_file: 深度综述JSON文件路径
        config: 配置参数字典（可选）
    
    Returns:
        生成结果字典
    """
    logger.info("=" * 80)
    logger.info("开始测试科研创意生成")
    logger.info("=" * 80)
    
    # 1. 加载数据
    logger.info("\n📂 步骤1: 加载数据文件")
    papers = load_papers_data(papers_file)
    graph = load_graph_data(graph_file)
    survey_data = load_survey_data(survey_file)
    
    # 2. 提取演化路径
    logger.info("\n🔍 步骤2: 提取演化路径")
    evolutionary_paths = survey_data.get('evolutionary_paths', [])
    if evolutionary_paths:
        logger.info(f"  找到 {len(evolutionary_paths)} 条演化路径")
        for i, path in enumerate(evolutionary_paths[:3], 1):
            pattern_type = path.get('pattern_type', 'Unknown')
            title = path.get('title', '')[:60]
            logger.info(f"    路径 {i}: {pattern_type} - {title}...")
    else:
        logger.warning("  未找到演化路径")
    
    # 3. 初始化创意生成器
    logger.info("\n🔧 步骤3: 初始化科研创意生成器")
    
    # 加载配置文件（如果config为None或需要补充配置）
    config_path = './config/config.yaml'
    if config is None:
        # 从配置文件加载
        full_config = load_full_config(config_path)
        config = full_config
        logger.info(f"  从配置文件加载: {config_path}")
    else:
        # 合并用户提供的配置和配置文件
        full_config = load_full_config(config_path)
        # 合并配置：用户配置优先，然后使用配置文件中的值
        merged_config = full_config.copy()
        for key, value in config.items():
            if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
                merged_config[key].update(value)
            else:
                merged_config[key] = value
        config = merged_config
        logger.info(f"  合并用户配置和配置文件: {config_path}")
    
    # 检查API密钥
    api_key = None
    if config.get('llm', {}).get('api_key'):
        api_key = config['llm']['api_key']
    elif config.get('openai_api_key'):
        api_key = config['openai_api_key']
    else:
        import os
        api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        logger.warning("⚠️  未找到API密钥，请确保:")
        logger.warning("  1. 在配置文件中设置 llm.api_key 或 openai_api_key")
        logger.warning("  2. 或设置环境变量 OPENAI_API_KEY")
        logger.warning("  3. 或在config参数中传入 api_key")
    
    generator = ResearchIdeaGenerator(config=config)
    logger.info(f"  模型: {config.get('llm', {}).get('model', 'gpt-4o')}")
    logger.info(f"  最大创意数: {config.get('research_idea', {}).get('max_ideas', 10)}")
    if api_key:
        logger.info(f"  API密钥: {'*' * 20}...{api_key[-4:] if len(api_key) > 4 else '****'}")
    
    # 4. 生成科研创意
    logger.info("\n💡 步骤4: 生成科研创意（使用演化路径学习）")
    topic = survey_data.get('topic', 'Natural Language Processing')
    
    result = generator.generate_from_knowledge_graph(
        graph=graph,
        topic=topic,
        evolutionary_paths=evolutionary_paths,
        verbose=True
    )
    
    # 5. 输出结果
    logger.info("\n" + "=" * 80)
    logger.info("✅ 测试完成")
    logger.info("=" * 80)
    logger.info(f"研究主题: {result.get('topic', 'N/A')}")
    logger.info(f"Step 1 - 提取的局限性: {result.get('pools', {}).get('unsolved_limitations', 0)} 条")
    logger.info(f"Step 1 - 提取的方法: {result.get('pools', {}).get('candidate_methods', 0)} 条")
    logger.info(f"Step 2 - 生成的创意总数: {result.get('total_ideas', 0)} 个")
    logger.info(f"Step 2 - 成功创意数: {result.get('successful_ideas', 0)} 个")
    
    # 显示生成的创意
    ideas = result.get('ideas', [])
    if ideas:
        logger.info(f"\n📋 生成的创意列表:")
        for i, idea in enumerate(ideas, 1):
            logger.info(f"\n  创意 {i}:")
            logger.info(f"    标题: {idea.get('title', 'N/A')}")
            logger.info(f"    状态: {idea.get('status', 'N/A')}")
            if idea.get('modification'):
                logger.info(f"    关键修改: {idea.get('modification', '')[:100]}...")
    
    return result


def save_results(result: Dict, output_file: str) -> None:
    """
    保存生成结果到JSON文件
    
    Args:
        result: 生成结果字典
        output_file: 输出文件路径
    """
    logger.info(f"\n💾 保存结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("✅ 结果已保存")


if __name__ == "__main__":
    # 数据文件路径
    base_dir = Path(__file__).parent
    papers_file = base_dir / "226_papers_natural_language_processing_20251218_032842.json"
    graph_file = base_dir / "226_graph_data_natural_language_processing_20251218_032842.json"
    survey_file = base_dir / "226_survey_Natural_Language_Processing_20251218_202205.json"
    
    # 检查文件是否存在
    if not papers_file.exists():
        logger.error(f"论文数据文件不存在: {papers_file}")
        exit(1)
    if not graph_file.exists():
        logger.error(f"图数据文件不存在: {graph_file}")
        exit(1)
    if not survey_file.exists():
        logger.error(f"深度综述文件不存在: {survey_file}")
        exit(1)
    
    # 配置参数（优先从config.yaml加载，这里可以覆盖部分配置）
    # 如果不需要覆盖，可以设置为 None，完全使用配置文件
    # config = None  # 设置为 None 以完全使用配置文件
    # 或者提供部分覆盖配置：
    config = {
        'llm': {
            'model': 'gpt-4o',  # 覆盖配置文件中的模型
            'temperature': 0.3
        },
        'research_idea': {
            'max_ideas': 20  # 最大生成创意数
        }
    }
    
    # 运行测试
    try:
        result = test_idea_generation(
            papers_file=str(papers_file),
            graph_file=str(graph_file),
            survey_file=str(survey_file),
            config=config
        )
        
        # 保存结果
        output_file = base_dir / "test_idea_generation_result.json"
        save_results(result, str(output_file))
        
        logger.info(f"\n🎉 测试完成！结果已保存到: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}", exc_info=True)
        exit(1)


