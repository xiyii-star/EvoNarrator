"""
生成论文格式的实验结果表格
支持 LaTeX、Markdown、CSV 等多种格式
"""

import json
import csv
from pathlib import Path
from typing import Dict, List


class ExperimentTableGenerator:
    """实验结果表格生成器"""

    def __init__(self, relation_types: List[str] = None):
        if relation_types is None:
            self.relation_types = [
                "Overcomes",
                "Realizes",
                "Extends",
                "Alternative",
                "Adapts_to",
                "Baselines"
            ]
        else:
            self.relation_types = relation_types

    def generate_main_result_table_latex(
        self,
        reports: List[Dict],
        output_path: str = None
    ) -> str:
        """
        生成主结果对比表 (LaTeX格式)

        Table 1: Overall Performance Comparison

        Method              | Accuracy | Macro F1 | Weighted F1
        --------------------|----------|----------|------------
        Baseline (Abstract) | 0.5174   | 0.2402   | 0.5590
        Ours (SocketMatch)  | 0.7435   | 0.1421   | 0.6341
        """
        latex = []
        latex.append("% Table 1: Overall Performance Comparison")
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Overall Performance Comparison of Different Methods}")
        latex.append("\\label{tab:main_results}")
        latex.append("\\begin{tabular}{lccc}")
        latex.append("\\toprule")
        latex.append("\\textbf{Method} & \\textbf{Accuracy} & \\textbf{Macro F1} & \\textbf{Weighted F1} \\\\")
        latex.append("\\midrule")

        for report in reports:
            method_name = report['method_name']
            acc = report['overall_metrics']['accuracy']
            macro_f1 = report['overall_metrics']['macro_f1']
            weighted_f1 = report['overall_metrics']['weighted_f1']

            latex.append(f"{method_name} & {acc:.4f} & {macro_f1:.4f} & {weighted_f1:.4f} \\\\")

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        result = "\n".join(latex)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)

        return result

    def generate_classwise_table_latex(
        self,
        reports: List[Dict],
        output_path: str = None
    ) -> str:
        """
        生成各类别详细对比表 (LaTeX格式)

        Table 2: Per-Class Performance Comparison (F1-Score)

        Citation Type | Baseline | Ours | Improvement
        --------------|----------|------|-------------
        Overcomes     | 0.0645   | 0.XXX| +X.XX%
        Realizes      | 0.0800   | 0.XXX| +X.XX%
        ...
        """
        latex = []
        latex.append("% Table 2: Per-Class Performance Comparison")
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Per-Class F1-Score Comparison Across Citation Relationship Types}")
        latex.append("\\label{tab:classwise_results}")

        # 构建表头
        num_methods = len(reports)
        col_format = "l" + "c" * num_methods + "c"  # 类别 + 方法们 + 提升
        latex.append(f"\\begin{{tabular}}{{{col_format}}}")
        latex.append("\\toprule")

        header = "\\textbf{Citation Type}"
        for report in reports:
            method_name = report['method_name'].replace('(', '\\small(').replace(')', ')')
            header += f" & \\textbf{{{method_name}}}"
        if num_methods == 2:
            header += " & \\textbf{Improvement}"
        header += " \\\\"
        latex.append(header)
        latex.append("\\midrule")

        # 填充数据
        for rel_type in self.relation_types:
            row = f"{rel_type}"
            f1_scores = []

            for report in reports:
                f1 = report['per_class_metrics'].get(rel_type, {}).get('f1', 0.0)
                f1_scores.append(f1)
                row += f" & {f1:.4f}"

            # 计算提升
            if num_methods == 2:
                diff = f1_scores[1] - f1_scores[0]
                if f1_scores[0] > 0:
                    pct = (diff / f1_scores[0]) * 100
                    if diff > 0:
                        row += f" & \\textcolor{{ForestGreen}}{{+{diff:.4f} ({pct:+.1f}\\%)}}"
                    elif diff < 0:
                        row += f" & \\textcolor{{red}}{{{diff:.4f} ({pct:.1f}\\%)}}"
                    else:
                        row += " & --"
                else:
                    row += f" & +{diff:.4f}"

            row += " \\\\"
            latex.append(row)

        latex.append("\\midrule")

        # 添加平均行
        avg_row = "\\textbf{Average (Macro)}"
        for report in reports:
            macro_f1 = report['overall_metrics']['macro_f1']
            avg_row += f" & \\textbf{{{macro_f1:.4f}}}"

        if num_methods == 2:
            diff = reports[1]['overall_metrics']['macro_f1'] - reports[0]['overall_metrics']['macro_f1']
            base = reports[0]['overall_metrics']['macro_f1']
            if base > 0:
                pct = (diff / base) * 100
                if diff > 0:
                    avg_row += f" & \\textbf{{\\textcolor{{ForestGreen}}{{+{diff:.4f} ({pct:+.1f}\\%)}}}}"
                else:
                    avg_row += f" & \\textbf{{\\textcolor{{red}}{{{diff:.4f} ({pct:.1f}\\%)}}}}"
            else:
                avg_row += f" & \\textbf{{+{diff:.4f}}}"

        avg_row += " \\\\"
        latex.append(avg_row)

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        result = "\n".join(latex)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)

        return result

    def generate_precision_recall_table_latex(
        self,
        reports: List[Dict],
        output_path: str = None
    ) -> str:
        """
        生成详细的 Precision/Recall/F1 表格 (LaTeX格式)

        适合放在附录或补充材料中
        """
        latex = []

        for idx, report in enumerate(reports, 1):
            latex.append(f"% Table: Detailed Metrics for {report['method_name']}")
            latex.append("\\begin{table}[htbp]")
            latex.append("\\centering")
            latex.append(f"\\caption{{Detailed Performance Metrics: {report['method_name']}}}")
            latex.append(f"\\label{{tab:detailed_metrics_{idx}}}")
            latex.append("\\begin{tabular}{lcccc}")
            latex.append("\\toprule")
            latex.append("\\textbf{Citation Type} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} & \\textbf{Support} \\\\")
            latex.append("\\midrule")

            for rel_type in self.relation_types:
                metrics = report['per_class_metrics'].get(rel_type, {})
                precision = metrics.get('precision', 0.0)
                recall = metrics.get('recall', 0.0)
                f1 = metrics.get('f1', 0.0)
                support = metrics.get('support', 0)

                latex.append(f"{rel_type} & {precision:.4f} & {recall:.4f} & {f1:.4f} & {support} \\\\")

            latex.append("\\bottomrule")
            latex.append("\\end{tabular}")
            latex.append("\\end{table}")
            latex.append("")  # 空行分隔

        result = "\n".join(latex)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)

        return result

    def generate_csv_tables(
        self,
        reports: List[Dict],
        output_dir: str
    ):
        """
        生成CSV格式的表格（便于Excel查看和编辑）
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Table 1: 主结果表
        with open(output_dir / "table1_main_results.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Method', 'Accuracy', 'Macro F1', 'Weighted F1'])
            for report in reports:
                writer.writerow([
                    report['method_name'],
                    f"{report['overall_metrics']['accuracy']:.4f}",
                    f"{report['overall_metrics']['macro_f1']:.4f}",
                    f"{report['overall_metrics']['weighted_f1']:.4f}"
                ])

        # Table 2: 各类别F1对比
        with open(output_dir / "table2_classwise_f1.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['Citation Type'] + [r['method_name'] for r in reports]
            if len(reports) == 2:
                header.append('Improvement')
            writer.writerow(header)

            for rel_type in self.relation_types:
                row = [rel_type]
                f1_scores = []
                for report in reports:
                    f1 = report['per_class_metrics'].get(rel_type, {}).get('f1', 0.0)
                    f1_scores.append(f1)
                    row.append(f"{f1:.4f}")

                if len(reports) == 2:
                    diff = f1_scores[1] - f1_scores[0]
                    pct = (diff / f1_scores[0] * 100) if f1_scores[0] > 0 else 0
                    row.append(f"{diff:+.4f} ({pct:+.1f}%)")

                writer.writerow(row)

            # 平均行
            avg_row = ['Average (Macro)']
            for report in reports:
                avg_row.append(f"{report['overall_metrics']['macro_f1']:.4f}")
            if len(reports) == 2:
                diff = reports[1]['overall_metrics']['macro_f1'] - reports[0]['overall_metrics']['macro_f1']
                pct = (diff / reports[0]['overall_metrics']['macro_f1'] * 100) if reports[0]['overall_metrics']['macro_f1'] > 0 else 0
                avg_row.append(f"{diff:+.4f} ({pct:+.1f}%)")
            writer.writerow(avg_row)

        # Table 3: 详细指标
        for idx, report in enumerate(reports, 1):
            filename = f"table3_detailed_metrics_{idx}.csv"
            with open(output_dir / filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Method', report['method_name']])
                writer.writerow(['Citation Type', 'Precision', 'Recall', 'F1-Score', 'Support'])

                for rel_type in self.relation_types:
                    metrics = report['per_class_metrics'].get(rel_type, {})
                    writer.writerow([
                        rel_type,
                        f"{metrics.get('precision', 0.0):.4f}",
                        f"{metrics.get('recall', 0.0):.4f}",
                        f"{metrics.get('f1', 0.0):.4f}",
                        metrics.get('support', 0)
                    ])

    def generate_markdown_tables(
        self,
        reports: List[Dict],
        output_path: str = None
    ) -> str:
        """
        生成Markdown格式的表格（适合GitHub README）
        """
        md = []

        # Table 1: 主结果
        md.append("## Table 1: Overall Performance Comparison")
        md.append("")
        md.append("| Method | Accuracy | Macro F1 | Weighted F1 |")
        md.append("|--------|----------|----------|-------------|")
        for report in reports:
            md.append(f"| {report['method_name']} | {report['overall_metrics']['accuracy']:.4f} | "
                     f"{report['overall_metrics']['macro_f1']:.4f} | "
                     f"{report['overall_metrics']['weighted_f1']:.4f} |")
        md.append("")

        # Table 2: 各类别对比
        md.append("## Table 2: Per-Class F1-Score Comparison")
        md.append("")
        header = "| Citation Type |"
        sep = "|---------------|"
        for report in reports:
            header += f" {report['method_name']} |"
            sep += "----------|"
        if len(reports) == 2:
            header += " Improvement |"
            sep += "-------------|"
        md.append(header)
        md.append(sep)

        for rel_type in self.relation_types:
            row = f"| **{rel_type}** |"
            f1_scores = []
            for report in reports:
                f1 = report['per_class_metrics'].get(rel_type, {}).get('f1', 0.0)
                f1_scores.append(f1)
                row += f" {f1:.4f} |"

            if len(reports) == 2:
                diff = f1_scores[1] - f1_scores[0]
                pct = (diff / f1_scores[0] * 100) if f1_scores[0] > 0 else 0
                row += f" {diff:+.4f} ({pct:+.1f}%) |"

            md.append(row)

        # 平均行
        avg_row = "| **Average** |"
        for report in reports:
            avg_row += f" **{report['overall_metrics']['macro_f1']:.4f}** |"
        if len(reports) == 2:
            diff = reports[1]['overall_metrics']['macro_f1'] - reports[0]['overall_metrics']['macro_f1']
            pct = (diff / reports[0]['overall_metrics']['macro_f1'] * 100) if reports[0]['overall_metrics']['macro_f1'] > 0 else 0
            avg_row += f" **{diff:+.4f} ({pct:+.1f}%)** |"
        md.append(avg_row)
        md.append("")

        result = "\n".join(md)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)

        return result

    def generate_all_tables(
        self,
        reports: List[Dict],
        output_dir: str
    ):
        """
        生成所有格式的表格
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"📊 正在生成实验结果表格...")

        # LaTeX 表格
        latex_dir = output_dir / "latex"
        latex_dir.mkdir(exist_ok=True)

        print("  ✓ 生成 LaTeX Table 1 (主结果对比)")
        self.generate_main_result_table_latex(reports, str(latex_dir / "table1_main_results.tex"))

        print("  ✓ 生成 LaTeX Table 2 (各类别F1对比)")
        self.generate_classwise_table_latex(reports, str(latex_dir / "table2_classwise_f1.tex"))

        print("  ✓ 生成 LaTeX Table 3 (详细指标)")
        self.generate_precision_recall_table_latex(reports, str(latex_dir / "table3_detailed_metrics.tex"))

        # CSV 表格
        csv_dir = output_dir / "csv"
        print("  ✓ 生成 CSV 表格")
        self.generate_csv_tables(reports, str(csv_dir))

        # Markdown 表格
        print("  ✓ 生成 Markdown 表格")
        self.generate_markdown_tables(reports, str(output_dir / "tables.md"))

        print(f"\n✅ 所有表格已生成到: {output_dir}")
        print(f"   • LaTeX 表格: {latex_dir}")
        print(f"   • CSV 表格: {csv_dir}")
        print(f"   • Markdown 表格: {output_dir / 'tables.md'}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python generate_tables.py <结果目录>")
        print("示例: python generate_tables.py results/20251210_114707")
        sys.exit(1)

    result_dir = Path(sys.argv[1])

    # 加载报告
    with open(result_dir / "baseline_report.json", 'r') as f:
        baseline_report = json.load(f)

    with open(result_dir / "socketmatch_report.json", 'r') as f:
        socketmatch_report = json.load(f)

    # 生成表格
    generator = ExperimentTableGenerator()
    generator.generate_all_tables(
        [baseline_report, socketmatch_report],
        str(result_dir / "tables")
    )
