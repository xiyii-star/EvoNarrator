"""
合并所有评估结果（包括消融实验）到一个综合对比表格
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import argparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_ablation_results(ablation_dir: Path) -> List[Dict]:
    """
    加载消融实验结果

    Args:
        ablation_dir: 消融实验结果目录

    Returns:
        消融实验结果列表
    """
    results = []

    # 定义消融实验变体
    variants = {
        'baseline': {
            'name': 'MyMethod (Full)',
            'description': '完整方法（包含所有组件）',
            'eval_pattern': 'evaluation_baseline_*.json'
        },
        'w1_no_deep_extraction': {
            'name': 'w/o Deep Extraction',
            'description': '移除深度提取组件',
            'eval_pattern': 'evaluation_w1_*.json'
        },
        'w2_no_socket_match': {
            'name': 'w/o Socket Match',
            'description': '移除插槽匹配组件',
            'eval_pattern': 'evaluation_w2_*.json'
        }
    }

    for variant_key, variant_info in variants.items():
        variant_dir = ablation_dir / variant_key
        if not variant_dir.exists():
            logger.warning(f"消融实验目录不存在: {variant_dir}")
            continue

        # 查找最新的评估结果文件
        eval_files = list(variant_dir.glob('evaluation_*.json'))
        if not eval_files:
            logger.warning(f"未找到评估结果: {variant_dir}")
            continue

        # 选择最新的文件
        latest_file = max(eval_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"加载消融实验结果: {latest_file}")

        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results.append({
            'variant_key': variant_key,
            'name': variant_info['name'],
            'description': variant_info['description'],
            'type': 'ablation',
            'data': data
        })

    return results


def load_baseline_results(baseline_summary_file: Path) -> List[Dict]:
    """
    加载基线方法评估结果

    Args:
        baseline_summary_file: 基线评估汇总文件

    Returns:
        基线方法结果列表
    """
    if not baseline_summary_file.exists():
        logger.warning(f"基线评估汇总文件不存在: {baseline_summary_file}")
        return []

    with open(baseline_summary_file, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    results = []
    for row in summary.get('table_data', []):
        results.append({
            'variant_key': row['preset_name'],
            'name': row['preset_name'],
            'description': row['description'],
            'type': 'baseline',
            'data': {
                'total_ideas': row['ideas_count'],
                'evaluated_ideas': row['ideas_count'],
                'statistics': {
                    'novelty': {'average': row['novelty']},
                    'feasibility': {'average': row['feasibility']},
                    'theoretical_support': {'average': row['theoretical_support']},
                    'logical_alignment': {'average': row['logical_alignment']},
                    'average': {'average': row['average']}
                }
            }
        })

    return results


def generate_merged_table(
    all_results: List[Dict],
    output_dir: Path,
    include_descriptions: bool = True
) -> Dict:
    """
    生成合并后的对比表格

    Args:
        all_results: 所有评估结果
        output_dir: 输出目录
        include_descriptions: 是否包含描述列

    Returns:
        汇总信息
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 准备表格数据
    table_data = []
    dimension_names = {
        'novelty': 'Novelty',
        'feasibility': 'Feasibility',
        'theoretical_support': 'Theoretical Support',
        'logical_alignment': 'Logical Alignment',
        'average': 'Average'
    }

    for result in all_results:
        stats = result['data'].get('statistics', {})
        row = {
            'name': result['name'],
            'type': result['type'],
            'description': result['description'],
            'ideas_count': result['data'].get('total_ideas', 0),
            'novelty': stats.get('novelty', {}).get('average'),
            'feasibility': stats.get('feasibility', {}).get('average'),
            'theoretical_support': stats.get('theoretical_support', {}).get('average'),
            'logical_alignment': stats.get('logical_alignment', {}).get('average'),
            'average': stats.get('average', {}).get('average')
        }
        table_data.append(row)

    # 找出各维度的最高分
    max_scores = {
        'novelty': -1,
        'feasibility': -1,
        'theoretical_support': -1,
        'logical_alignment': -1,
        'average': -1
    }
    max_methods = {dim: [] for dim in max_scores.keys()}

    for row in table_data:
        for dim in max_scores.keys():
            score = row.get(dim)
            if score is not None:
                if score > max_scores[dim]:
                    max_scores[dim] = score
                    max_methods[dim] = [row['name']]
                elif abs(score - max_scores[dim]) < 0.01:
                    max_methods[dim].append(row['name'])

    # 1. 控制台表格
    print("\n" + "=" * 120)
    print("📊 综合评估结果对比表格（包含消融实验）:")
    print("=" * 120)

    if include_descriptions:
        print(f"{'方法名称':<25} {'类型':<15} {'创意数':<8} {'新颖性':<10} {'可行性':<10} {'理论支撑度':<12} {'逻辑契合度':<12} {'总体平均':<10}")
    else:
        print(f"{'方法名称':<30} {'类型':<15} {'创意数':<8} {'新颖性':<10} {'可行性':<10} {'理论支撑度':<12} {'逻辑契合度':<12} {'总体平均':<10}")

    print("-" * 120)

    for row in table_data:
        type_label = "消融实验" if row['type'] == 'ablation' else "基线方法"
        novelty = f"{row['novelty']:.2f}" if row['novelty'] is not None else "N/A"
        feasibility = f"{row['feasibility']:.2f}" if row['feasibility'] is not None else "N/A"
        theoretical_support = f"{row['theoretical_support']:.2f}" if row['theoretical_support'] is not None else "N/A"
        logical_alignment = f"{row['logical_alignment']:.2f}" if row['logical_alignment'] is not None else "N/A"
        average = f"{row['average']:.2f}" if row['average'] is not None else "N/A"

        if include_descriptions:
            print(f"{row['name']:<25} {type_label:<15} {row['ideas_count']:<8} {novelty:<10} {feasibility:<10} {theoretical_support:<12} {logical_alignment:<12} {average:<10}")
        else:
            print(f"{row['name']:<30} {type_label:<15} {row['ideas_count']:<8} {novelty:<10} {feasibility:<10} {theoretical_support:<12} {logical_alignment:<12} {average:<10}")

    print("=" * 120)

    # 标注最高分
    print("\n📌 最高分标注:")
    for dim, method_list in max_methods.items():
        if method_list and max_scores[dim] > 0:
            dim_name = dimension_names.get(dim, dim)
            print(f"  - {', '.join(method_list)}: {dim_name}最高 ({max_scores[dim]:.2f})")

    # 2. Markdown 表格
    md_file = output_dir / f"merged_evaluation_comparison_{timestamp}.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# 综合评估结果对比表（包含消融实验）\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if include_descriptions:
            f.write("| 方法名称 | 类型 | 描述 | 创意数 | 新颖性 | 可行性 | 理论支撑度 | 逻辑契合度 | 总体平均 |\n")
            f.write("|---------|------|------|--------|--------|--------|------------|------------|----------|\n")
        else:
            f.write("| 方法名称 | 类型 | 创意数 | 新颖性 | 可行性 | 理论支撑度 | 逻辑契合度 | 总体平均 |\n")
            f.write("|---------|------|--------|--------|--------|------------|------------|----------|\n")

        for row in table_data:
            type_label = "消融实验" if row['type'] == 'ablation' else "基线方法"
            novelty = f"{row['novelty']:.2f}" if row['novelty'] is not None else "N/A"
            feasibility = f"{row['feasibility']:.2f}" if row['feasibility'] is not None else "N/A"
            theoretical_support = f"{row['theoretical_support']:.2f}" if row['theoretical_support'] is not None else "N/A"
            logical_alignment = f"{row['logical_alignment']:.2f}" if row['logical_alignment'] is not None else "N/A"
            average = f"{row['average']:.2f}" if row['average'] is not None else "N/A"

            # 转义描述中的管道符
            description = row['description'].replace('|', '\\|') if include_descriptions else ""

            if include_descriptions:
                f.write(f"| {row['name']} | {type_label} | {description} | {row['ideas_count']} | {novelty} | {feasibility} | {theoretical_support} | {logical_alignment} | {average} |\n")
            else:
                f.write(f"| {row['name']} | {type_label} | {row['ideas_count']} | {novelty} | {feasibility} | {theoretical_support} | {logical_alignment} | {average} |\n")

        f.write("\n## 评分说明\n\n")
        f.write("各维度评分范围：1-5 分（李克特量表）\n\n")
        f.write("- **新颖性 (Novelty)**: 该想法是否提供了区别于现有文献的独特视角？\n")
        f.write("- **可行性 (Feasibility)**: 基于现有技术栈，该技术路线是否逻辑自洽且可实现？\n")
        f.write("- **理论支撑度 (Theoretical Support)**: 评估该假设是否提供了充分的依据，符合领域发展的历史惯性。\n")
        f.write("- **逻辑契合度 (Logical Alignment)**: 评估生成的解决方案是否精准且实质性地解决了提出的科研痛点。\n")
        f.write("- **总体平均**: 四个维度的平均分\n\n")

        f.write("## 方法类型说明\n\n")
        f.write("- **消融实验**: MyMethod的消融变体，用于验证各组件的贡献\n")
        f.write("  - MyMethod (Full): 完整方法\n")
        f.write("  - w/o Deep Extraction: 移除深度提取组件\n")
        f.write("  - w/o Socket Match: 移除插槽匹配组件\n")
        f.write("- **基线方法**: 其他方法的评估结果（CoI, MOOSEChem, naive_llm等）\n\n")

        f.write("## 最高分标注\n\n")
        for dim, method_list in max_methods.items():
            if method_list and max_scores[dim] > 0:
                dim_name = dimension_names.get(dim, dim)
                f.write(f"- **{', '.join(method_list)}**: 🏆 {dim_name}最高 ({max_scores[dim]:.2f})\n")

    logger.info(f"📄 Markdown 表格已保存: {md_file}")

    # 3. CSV 表格
    csv_file = output_dir / f"merged_evaluation_comparison_{timestamp}.csv"
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        if include_descriptions:
            fieldnames = ['name', 'type', 'description', 'ideas_count', 'novelty', 'feasibility',
                         'theoretical_support', 'logical_alignment', 'average']
        else:
            fieldnames = ['name', 'type', 'ideas_count', 'novelty', 'feasibility',
                         'theoretical_support', 'logical_alignment', 'average']

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in table_data:
            csv_row = {
                'name': row['name'],
                'type': '消融实验' if row['type'] == 'ablation' else '基线方法',
                'ideas_count': row['ideas_count'],
                'novelty': row['novelty'] if row['novelty'] is not None else '',
                'feasibility': row['feasibility'] if row['feasibility'] is not None else '',
                'theoretical_support': row['theoretical_support'] if row['theoretical_support'] is not None else '',
                'logical_alignment': row['logical_alignment'] if row['logical_alignment'] is not None else '',
                'average': row['average'] if row['average'] is not None else ''
            }
            if include_descriptions:
                csv_row['description'] = row['description']
            writer.writerow(csv_row)

    logger.info(f"📊 CSV 表格已保存: {csv_file}")

    # 4. JSON 汇总
    json_file = output_dir / f"merged_evaluation_summary_{timestamp}.json"
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'total_methods': len(all_results),
        'ablation_methods': len([r for r in all_results if r['type'] == 'ablation']),
        'baseline_methods': len([r for r in all_results if r['type'] == 'baseline']),
        'table_data': table_data,
        'max_scores': max_scores,
        'max_methods': {k: v for k, v in max_methods.items() if v},
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


def main():
    parser = argparse.ArgumentParser(
        description='合并所有评估结果（包括消融实验）到一个综合对比表格',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认路径
  python merge_all_results.py

  # 指定消融实验目录和基线汇总文件
  python merge_all_results.py --ablation-dir ablation_results --baseline-summary idea_eval/batch_evaluation_summary_20260101_211623.json

  # 指定输出目录
  python merge_all_results.py --output idea_eval/merged_results

  # 不包含描述列
  python merge_all_results.py --no-descriptions
        """
    )

    # 获取脚本所在目录的父目录（项目根目录）
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    parser.add_argument('--ablation-dir', type=str,
                       default=str(project_root / 'ablation_results'),
                       help='消融实验结果目录')
    parser.add_argument('--baseline-summary', type=str,
                       help='基线评估汇总文件路径（JSON格式）')
    parser.add_argument('--output', type=str,
                       default=str(script_dir),
                       help='输出目录')
    parser.add_argument('--no-descriptions', action='store_true',
                       help='不包含描述列')

    args = parser.parse_args()

    # 转换为Path对象
    ablation_dir = Path(args.ablation_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载消融实验结果
    logger.info(f"\n{'='*80}")
    logger.info("📂 加载消融实验结果")
    logger.info(f"{'='*80}")
    ablation_results = load_ablation_results(ablation_dir)
    logger.info(f"✅ 加载了 {len(ablation_results)} 个消融实验结果")

    # 加载基线方法结果
    logger.info(f"\n{'='*80}")
    logger.info("📂 加载基线方法结果")
    logger.info(f"{'='*80}")

    if args.baseline_summary:
        baseline_summary_file = Path(args.baseline_summary)
    else:
        # 查找最新的基线评估汇总文件
        baseline_files = list(script_dir.glob('batch_evaluation_summary_*.json'))
        if baseline_files:
            baseline_summary_file = max(baseline_files, key=lambda f: f.stat().st_mtime)
            logger.info(f"自动选择最新的基线评估汇总文件: {baseline_summary_file}")
        else:
            logger.warning("未找到基线评估汇总文件")
            baseline_summary_file = None

    baseline_results = []
    if baseline_summary_file:
        baseline_results = load_baseline_results(baseline_summary_file)
        logger.info(f"✅ 加载了 {len(baseline_results)} 个基线方法结果")

    # 合并所有结果
    all_results = ablation_results + baseline_results

    if not all_results:
        logger.error("❌ 没有找到任何评估结果")
        return

    logger.info(f"\n{'='*80}")
    logger.info(f"总共 {len(all_results)} 个方法（消融实验: {len(ablation_results)}, 基线方法: {len(baseline_results)}）")
    logger.info(f"{'='*80}")

    # 生成合并后的表格
    logger.info(f"\n{'='*80}")
    logger.info("📊 生成综合对比表格")
    logger.info(f"{'='*80}")

    summary = generate_merged_table(
        all_results=all_results,
        output_dir=output_dir,
        include_descriptions=not args.no_descriptions
    )

    logger.info(f"\n{'='*80}")
    logger.info("✅ 综合对比表格生成完成！")
    logger.info(f"{'='*80}")
    logger.info(f"\n📊 汇总文件:")
    logger.info(f"  - Markdown: {summary['files']['markdown']}")
    logger.info(f"  - CSV: {summary['files']['csv']}")
    logger.info(f"  - JSON: {summary['files']['json']}")


if __name__ == "__main__":
    main()
