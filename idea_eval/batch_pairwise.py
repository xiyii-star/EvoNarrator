"""
批量运行Pairwise Win-Rate比较
自动对比MyMethod vs 所有baseline/ablation方法
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from idea_eval.pairwise_compare import (
    PairwiseComparator,
    load_ideas_from_file,
    save_comparison_results,
    generate_comparison_report
)
from src.llm_config import LLMConfig
import yaml

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_all_pairwise_comparisons(
    mymethod_file: str,
    comparison_files: dict,
    config_path: str = 'config/config.yaml',
    output_dir: str = 'pairwise_results',
    max_pairs: int = None
):
    """
    批量运行所有Pairwise比较

    Args:
        mymethod_file: MyMethod的创意文件
        comparison_files: 对比方法的文件字典 {method_name: file_path}
        config_path: LLM配置文件
        output_dir: 输出目录
        max_pairs: 每组最大比较对数
    """
    start_time = datetime.now()

    logger.info("=" * 80)
    logger.info("批量Pairwise Win-Rate比较")
    logger.info("=" * 80)
    logger.info(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"MyMethod文件: {mymethod_file}")
    logger.info(f"对比方法数: {len(comparison_files)}")
    logger.info("=" * 80 + "\n")

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    llm_config_dict = config.get('llm', {})
    llm_config = LLMConfig(
        provider=llm_config_dict.get('provider', 'openai'),
        model=llm_config_dict.get('model', 'gpt-4o'),
        api_key=llm_config_dict.get('api_key'),
        base_url=llm_config_dict.get('base_url'),
        temperature=0.3,
        max_tokens=800
    )

    # 初始化比较器
    comparator = PairwiseComparator(llm_config)

    # 加载MyMethod创意
    logger.info("加载MyMethod创意...")
    mymethod_ideas = load_ideas_from_file(mymethod_file)

    # 运行所有比较
    all_results = {}

    for method_name, method_file in comparison_files.items():
        logger.info("\n" + "=" * 80)
        logger.info(f"对比: MyMethod vs {method_name}")
        logger.info("=" * 80)

        try:
            # 加载对比方法创意
            comparison_ideas = load_ideas_from_file(method_file)

            # 运行比较
            result = comparator.batch_compare(
                ideas_a=mymethod_ideas,
                ideas_b=comparison_ideas,
                method_a_name="MyMethod",
                method_b_name=method_name,
                max_pairs=max_pairs
            )

            # 保存结果
            output_file = Path(output_dir) / f"pairwise_mymethod_vs_{method_name.lower().replace(' ', '_')}.json"
            save_comparison_results(result, str(output_file))

            # 生成报告
            report_file = output_file.with_suffix('.md')
            generate_comparison_report(result, str(report_file))

            all_results[method_name] = result

            logger.info(f"✅ 完成对比: MyMethod vs {method_name}")
            logger.info(f"   胜率: {result['win_rate_a']:.1f}% vs {result['win_rate_b']:.1f}%")

        except Exception as e:
            logger.error(f"❌ 对比失败 ({method_name}): {e}", exc_info=True)
            continue

    # 生成综合报告
    if all_results:
        logger.info("\n" + "=" * 80)
        logger.info("生成综合对比报告")
        logger.info("=" * 80)

        summary_file = Path(output_dir) / f"pairwise_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        generate_summary_report(all_results, str(summary_file))

        logger.info(f"✅ 综合报告已生成: {summary_file}")

    # 输出总结
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("\n" + "=" * 80)
    logger.info("所有Pairwise比较完成！")
    logger.info("=" * 80)
    logger.info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"总耗时: {duration:.1f}秒 ({duration/60:.1f}分钟)")
    logger.info(f"完成对比数: {len(all_results)}/{len(comparison_files)}")
    logger.info(f"\n胜率汇总:")

    for method_name, result in all_results.items():
        logger.info(f"  MyMethod vs {method_name}: {result['win_rate_a']:.1f}%")

    return all_results


def generate_summary_report(all_results: dict, output_file: str):
    """生成综合对比报告"""
    logger.info(f"生成综合报告: {output_file}")

    report_lines = [
        "# Pairwise Win-Rate 综合对比报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 胜率汇总表",
        "",
        "| 对比 | MyMethod胜率 | 对手胜率 | 平局率 | 总比较数 |",
        "|------|-------------|----------|--------|----------|"
    ]

    # 按胜率降序排列
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]['win_rate_a'],
        reverse=True
    )

    for method_name, result in sorted_results:
        report_lines.append(
            f"| MyMethod vs {method_name} | "
            f"**{result['win_rate_a']:.1f}%** | "
            f"{result['win_rate_b']:.1f}% | "
            f"{result['tie_rate']:.1f}% | "
            f"{result['total_comparisons']} |"
        )

    report_lines.extend([
        "",
        "## 胜率可视化",
        "",
        "```"
    ])

    for method_name, result in sorted_results:
        bar_length = int(result['win_rate_a'] / 2)  # 每2%一个字符
        report_lines.append(f"{method_name:20s}: {'█' * bar_length} {result['win_rate_a']:.1f}%")

    report_lines.extend([
        "```",
        "",
        "## 关键发现",
        ""
    ])

    # 自动生成关键发现
    if sorted_results:
        best_match = sorted_results[0]
        worst_match = sorted_results[-1]

        report_lines.extend([
            f"1. **最高胜率**: MyMethod vs {best_match[0]} = **{best_match[1]['win_rate_a']:.1f}%**",
            f"   - 说明: MyMethod在对比{best_match[0]}时表现最优",
            "",
            f"2. **最低胜率**: MyMethod vs {worst_match[0]} = **{worst_match[1]['win_rate_a']:.1f}%**",
            f"   - 说明: {worst_match[0]}是最强的baseline",
            "",
            "3. **整体表现**:",
        ])

        avg_win_rate = sum(r['win_rate_a'] for r in all_results.values()) / len(all_results)
        report_lines.append(f"   - 平均胜率: {avg_win_rate:.1f}%")

        # 统计显著优势的对比（胜率>70%）
        strong_wins = [name for name, r in all_results.items() if r['win_rate_a'] >= 70]
        if strong_wins:
            report_lines.append(f"   - 显著优势（胜率≥70%）: {len(strong_wins)}/{len(all_results)} 个对比")
            report_lines.append(f"     - {', '.join(strong_wins)}")

    report_lines.extend([
        "",
        "## 详细对比结果",
        "",
        "各对比的详细结果请参见单独的报告文件:",
        ""
    ])

    for method_name in all_results.keys():
        filename = f"pairwise_mymethod_vs_{method_name.lower().replace(' ', '_')}.md"
        report_lines.append(f"- [{method_name}]({filename})")

    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    logger.info("✅ 综合报告已生成")


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description='批量运行Pairwise Win-Rate比较',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 自动对比MyMethod vs 所有baseline/ablation
  python idea_eval/batch_pairwise.py \\
    --mymethod results/mymethod_ideas.json \\
    --output pairwise_results

  # 需要手动指定对比文件，可以创建一个配置文件或使用命令行参数
        """
    )

    parser.add_argument('--mymethod', type=str, required=True, help='MyMethod创意文件')
    parser.add_argument('--output', type=str, default='pairwise_results', help='输出目录')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='LLM配置文件')
    parser.add_argument('--max-pairs', type=int, help='每组最大比较对数')

    # 对比文件
    parser.add_argument('--naive-llm', type=str, help='Naive LLM创意文件')
    parser.add_argument('--standard-rag', type=str, help='Standard RAG创意文件')
    parser.add_argument('--w1', type=str, help='w/o Deep Extraction创意文件')
    parser.add_argument('--w2', type=str, help='w/o SocketMatch创意文件')
    parser.add_argument('--w3', type=str, help='w/o Dual-Track创意文件')

    args = parser.parse_args()

    # 检查MyMethod文件
    if not Path(args.mymethod).exists():
        logger.error(f"❌ MyMethod文件不存在: {args.mymethod}")
        sys.exit(1)

    # 收集对比文件
    comparison_files = {}

    if args.naive_llm:
        if Path(args.naive_llm).exists():
            comparison_files['Naive LLM'] = args.naive_llm
        else:
            logger.warning(f"⚠️  跳过Naive LLM: 文件不存在 {args.naive_llm}")

    if args.standard_rag:
        if Path(args.standard_rag).exists():
            comparison_files['Standard RAG'] = args.standard_rag
        else:
            logger.warning(f"⚠️  跳过Standard RAG: 文件不存在 {args.standard_rag}")

    if args.w1:
        if Path(args.w1).exists():
            comparison_files['w/o Deep Extraction'] = args.w1
        else:
            logger.warning(f"⚠️  跳过w1: 文件不存在 {args.w1}")

    if args.w2:
        if Path(args.w2).exists():
            comparison_files['w/o SocketMatch'] = args.w2
        else:
            logger.warning(f"⚠️  跳过w2: 文件不存在 {args.w2}")

    if args.w3:
        if Path(args.w3).exists():
            comparison_files['w/o Dual-Track'] = args.w3
        else:
            logger.warning(f"⚠️  跳过w3: 文件不存在 {args.w3}")

    if not comparison_files:
        logger.error("❌ 没有找到任何对比文件，请使用 --naive-llm, --standard-rag 等参数指定")
        logger.info("\n或者自动搜索常见位置...")

        # 尝试自动发现文件
        auto_discovered = {}
        search_patterns = {
            'Standard RAG': ['standardrag_ideas.json', 'results/standardrag_*.json'],
            'w/o Deep Extraction': ['ablation_results/w1_*/ideas_w1_*.json'],
            'w/o SocketMatch': ['ablation_results/w2_*/ideas_w2_*.json'],
            'w/o Dual-Track': ['ablation_results/w3_*/ideas_w3_*.json']
        }

        for method_name, patterns in search_patterns.items():
            for pattern in patterns:
                matches = list(Path('.').glob(pattern))
                if matches:
                    auto_discovered[method_name] = str(matches[0])
                    logger.info(f"  发现: {method_name} -> {matches[0]}")
                    break

        if auto_discovered:
            comparison_files = auto_discovered
        else:
            sys.exit(1)

    try:
        results = run_all_pairwise_comparisons(
            mymethod_file=args.mymethod,
            comparison_files=comparison_files,
            config_path=args.config,
            output_dir=args.output,
            max_pairs=args.max_pairs
        )

        logger.info(f"\n🎉 所有比较完成！结果保存在: {args.output}")

    except Exception as e:
        logger.error(f"❌ 批量比较失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
