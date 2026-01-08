#!/usr/bin/env python
"""
从已有的评估结果重新生成美观的报告输出
"""

import json
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent / "src"))
from metrics_calculator import MetricsCalculator

def main():
    if len(sys.argv) < 2:
        print("用法: python regenerate_report.py <结果目录>")
        print("示例: python regenerate_report.py results/20251210_114707")
        sys.exit(1)

    result_dir = Path(sys.argv[1])

    if not result_dir.exists():
        print(f"❌ 错误: 目录不存在: {result_dir}")
        sys.exit(1)

    # 加载报告
    baseline_report_path = result_dir / "baseline_report.json"
    socketmatch_report_path = result_dir / "socketmatch_report.json"

    if not baseline_report_path.exists() or not socketmatch_report_path.exists():
        print(f"❌ 错误: 报告文件不存在")
        print(f"   需要: {baseline_report_path}")
        print(f"   需要: {socketmatch_report_path}")
        sys.exit(1)

    # 读取报告
    with open(baseline_report_path, 'r', encoding='utf-8') as f:
        baseline_report = json.load(f)

    with open(socketmatch_report_path, 'r', encoding='utf-8') as f:
        socketmatch_report = json.load(f)

    # 创建计算器
    calculator = MetricsCalculator()

    print("\n" + "🎉"*50)
    print("引用关系分类评估 - 完整报告")
    print("🎉"*50)

    # 打印 Baseline 报告
    calculator.print_report(baseline_report, show_explanation=True)

    # 打印 SocketMatch 报告
    calculator.print_report(socketmatch_report, show_explanation=False)

    # 打印对比
    comparison = calculator.compare_methods([baseline_report, socketmatch_report])
    calculator.print_comparison(comparison)

    print("\n✅ 报告生成完成！")
    print(f"📁 结果目录: {result_dir}")

if __name__ == "__main__":
    main()
