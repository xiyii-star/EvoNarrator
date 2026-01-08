"""
评估指标计算模块
计算 Accuracy 和 F1-Score
"""

import logging
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """评估指标计算器"""

    def __init__(self, relation_types: List[str] = None):
        """
        初始化计算器

        Args:
            relation_types: 关系类型列表（如果为None则使用默认）
        """
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

    def calculate_accuracy(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> float:
        """
        计算准确率

        Args:
            predictions: 预测标签列表
            ground_truths: 真实标签列表

        Returns:
            准确率 (0.0-1.0)
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("预测标签和真实标签数量不一致")

        if len(predictions) == 0:
            return 0.0

        correct = sum(1 for pred, true in zip(predictions, ground_truths) if pred == true)
        return correct / len(predictions)

    def calculate_precision_recall_f1(
        self,
        predictions: List[str],
        ground_truths: List[str],
        label: str
    ) -> Tuple[float, float, float]:
        """
        计算单个类别的精确率、召回率和F1分数

        Args:
            predictions: 预测标签列表
            ground_truths: 真实标签列表
            label: 目标类别

        Returns:
            (precision, recall, f1)
        """
        # True Positives: 预测为该类且实际为该类
        tp = sum(1 for pred, true in zip(predictions, ground_truths)
                 if pred == label and true == label)

        # False Positives: 预测为该类但实际不是该类
        fp = sum(1 for pred, true in zip(predictions, ground_truths)
                 if pred == label and true != label)

        # False Negatives: 预测不是该类但实际是该类
        fn = sum(1 for pred, true in zip(predictions, ground_truths)
                 if pred != label and true == label)

        # 计算精确率
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # 计算召回率
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # 计算F1分数
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    def calculate_macro_f1(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> float:
        """
        计算宏平均F1分数（Macro-averaged F1）

        Args:
            predictions: 预测标签列表
            ground_truths: 真实标签列表

        Returns:
            宏平均F1分数
        """
        f1_scores = []

        for label in self.relation_types:
            _, _, f1 = self.calculate_precision_recall_f1(predictions, ground_truths, label)
            f1_scores.append(f1)

        return np.mean(f1_scores)

    def calculate_weighted_f1(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> float:
        """
        计算加权平均F1分数（Weighted-averaged F1）
        权重为每个类别在真实标签中的比例

        Args:
            predictions: 预测标签列表
            ground_truths: 真实标签列表

        Returns:
            加权平均F1分数
        """
        # 统计真实标签中每个类别的数量
        label_counts = Counter(ground_truths)
        total = len(ground_truths)

        weighted_f1 = 0.0

        for label in self.relation_types:
            _, _, f1 = self.calculate_precision_recall_f1(predictions, ground_truths, label)
            weight = label_counts.get(label, 0) / total
            weighted_f1 += f1 * weight

        return weighted_f1

    def calculate_per_class_metrics(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        计算每个类别的详细指标

        Args:
            predictions: 预测标签列表
            ground_truths: 真实标签列表

        Returns:
            每个类别的指标字典
        """
        per_class_metrics = {}

        for label in self.relation_types:
            precision, recall, f1 = self.calculate_precision_recall_f1(
                predictions, ground_truths, label
            )

            # 统计支持数（真实标签中该类别的数量）
            support = sum(1 for true in ground_truths if true == label)

            per_class_metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support
            }

        return per_class_metrics

    def calculate_confusion_matrix(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> Dict:
        """
        计算混淆矩阵

        Args:
            predictions: 预测标签列表
            ground_truths: 真实标签列表

        Returns:
            混淆矩阵字典
        """
        # 初始化混淆矩阵
        confusion_matrix = {
            true_label: {pred_label: 0 for pred_label in self.relation_types}
            for true_label in self.relation_types
        }

        # 填充混淆矩阵
        for pred, true in zip(predictions, ground_truths):
            if true in confusion_matrix and pred in confusion_matrix[true]:
                confusion_matrix[true][pred] += 1

        return confusion_matrix

    def generate_full_report(
        self,
        predictions: List[Dict],
        method_name: str = "Method"
    ) -> Dict:
        """
        生成完整的评估报告

        Args:
            predictions: 预测结果列表，每项包含:
                - predicted_type
                - ground_truth
                - confidence (optional)
            method_name: 方法名称

        Returns:
            完整的评估报告字典
        """
        # 提取预测和真实标签
        pred_labels = [p['predicted_type'] for p in predictions]
        true_labels = [p['ground_truth'] for p in predictions]

        # 计算整体指标
        accuracy = self.calculate_accuracy(pred_labels, true_labels)
        macro_f1 = self.calculate_macro_f1(pred_labels, true_labels)
        weighted_f1 = self.calculate_weighted_f1(pred_labels, true_labels)

        # 计算每个类别的指标
        per_class_metrics = self.calculate_per_class_metrics(pred_labels, true_labels)

        # 计算混淆矩阵
        confusion_matrix = self.calculate_confusion_matrix(pred_labels, true_labels)

        # 统计预测分布
        pred_distribution = Counter(pred_labels)
        true_distribution = Counter(true_labels)

        # 构建报告
        report = {
            'method_name': method_name,
            'total_samples': len(predictions),
            'overall_metrics': {
                'accuracy': accuracy,
                'macro_f1': macro_f1,
                'weighted_f1': weighted_f1
            },
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': confusion_matrix,
            'distribution': {
                'predictions': dict(pred_distribution),
                'ground_truth': dict(true_distribution)
            }
        }

        return report

    def print_report(self, report: Dict, show_explanation: bool = True):
        """
        打印评估报告

        Args:
            report: 评估报告字典
            show_explanation: 是否显示指标说明
        """
        print("\n" + "="*100)
        print(f"📊 评估报告: {report['method_name']}")
        print("="*100)

        # 整体指标
        print(f"\n┌{'─' * 98}┐")
        print(f"│ {'整体性能指标':<95} │")
        print(f"├{'─' * 98}┤")
        print(f"│ {'样本总数:':<30} {report['total_samples']:<67} │")
        print(f"│ {'Accuracy (准确率):':<30} {report['overall_metrics']['accuracy']:<8.4f} {'- 所有预测中正确的比例':<57} │")
        print(f"│ {'Macro F1 (宏平均F1):':<30} {report['overall_metrics']['macro_f1']:<8.4f} {'- 各类别F1的平均值，平等对待每个类别':<57} │")
        print(f"│ {'Weighted F1 (加权F1):':<30} {report['overall_metrics']['weighted_f1']:<8.4f} {'- 按类别样本量加权的F1，关注样本多的类别':<57} │")
        print(f"└{'─' * 98}┘")

        # 每个类别的详细指标
        print(f"\n┌{'─' * 98}┐")
        print(f"│ {'各类别详细指标':<95} │")
        print(f"├{'─' * 98}┤")
        print(f"│ {'类别':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10} {'正确/总预测':<15} {'正确/总真实':<15} │")
        print(f"├{'─' * 98}┤")

        # 计算每个类别的正确数
        for label, metrics in sorted(report['per_class_metrics'].items(),
                                      key=lambda x: x[1]['f1'],
                                      reverse=True):
            support = metrics['support']
            precision = metrics['precision']
            recall = metrics['recall']
            f1 = metrics['f1']

            # 计算正确预测数和总预测数
            pred_count = report['distribution']['predictions'].get(label, 0)
            true_positives = int(precision * pred_count) if pred_count > 0 else 0
            pred_ratio = f"{true_positives}/{pred_count}" if pred_count > 0 else "0/0"

            # 计算召回的比例
            true_positives_recall = int(recall * support) if support > 0 else 0
            recall_ratio = f"{true_positives_recall}/{support}" if support > 0 else "0/0"

            print(f"│ {label:<15} "
                  f"{precision:<12.4f} "
                  f"{recall:<12.4f} "
                  f"{f1:<12.4f} "
                  f"{support:<10} "
                  f"{pred_ratio:<15} "
                  f"{recall_ratio:<15} │")

        print(f"└{'─' * 98}┘")

        # 指标说明
        if show_explanation:
            print(f"\n💡 指标含义说明:")
            print(f"   • Precision (精确率): 在预测为该类别的样本中，真正属于该类别的比例")
            print(f"     └─ 高精确率 = 预测该类别时很少出错")
            print(f"   • Recall (召回率): 在真实属于该类别的样本中，被正确预测的比例")
            print(f"     └─ 高召回率 = 能找出该类别的大部分样本")
            print(f"   • F1-Score: Precision和Recall的调和平均，综合评价指标")
            print(f"     └─ F1 = 2 × (Precision × Recall) / (Precision + Recall)")
            print(f"   • Support: 该类别在真实标签中的样本数量")

        # 类别分布对比
        print(f"\n┌{'─' * 98}┐")
        print(f"│ {'类别分布对比':<95} │")
        print(f"├{'─' * 98}┤")
        print(f"│ {'类别':<20} {'真实标签数量':<25} {'预测标签数量':<25} {'差异':<25} │")
        print(f"├{'─' * 98}┤")

        all_labels = set(report['distribution']['ground_truth'].keys()) | \
                     set(report['distribution']['predictions'].keys())

        for label in sorted(all_labels,
                           key=lambda x: report['distribution']['ground_truth'].get(x, 0),
                           reverse=True):
            true_count = report['distribution']['ground_truth'].get(label, 0)
            pred_count = report['distribution']['predictions'].get(label, 0)
            diff = pred_count - true_count

            true_pct = (true_count / report['total_samples']) * 100
            pred_pct = (pred_count / report['total_samples']) * 100

            true_str = f"{true_count:>4} ({true_pct:>5.1f}%)"
            pred_str = f"{pred_count:>4} ({pred_pct:>5.1f}%)"
            diff_str = f"{diff:+5} ({(diff/report['total_samples']*100):+6.1f}%)" if report['total_samples'] > 0 else "0"

            print(f"│ {label:<20} {true_str:<25} {pred_str:<25} {diff_str:<25} │")

        print(f"└{'─' * 98}┘")
        print("="*100)

    def compare_methods(
        self,
        reports: List[Dict]
    ) -> Dict:
        """
        比较多个方法的性能

        Args:
            reports: 多个评估报告列表

        Returns:
            比较结果字典
        """
        comparison = {
            'methods': [r['method_name'] for r in reports],
            'accuracy': [r['overall_metrics']['accuracy'] for r in reports],
            'macro_f1': [r['overall_metrics']['macro_f1'] for r in reports],
            'weighted_f1': [r['overall_metrics']['weighted_f1'] for r in reports],
            'per_class_metrics': {}
        }

        # 添加每个类别的比较
        relation_types = self.relation_types
        for rel_type in relation_types:
            comparison['per_class_metrics'][rel_type] = {
                'f1_scores': []
            }
            for report in reports:
                f1 = report['per_class_metrics'].get(rel_type, {}).get('f1', 0.0)
                comparison['per_class_metrics'][rel_type]['f1_scores'].append(f1)

        return comparison

    def print_comparison(self, comparison: Dict):
        """
        打印方法比较结果

        Args:
            comparison: 比较结果字典
        """
        print("\n" + "="*100)
        print("🔬 方法性能对比分析")
        print("="*100)

        # 整体指标比较
        print(f"\n┌{'─' * 98}┐")
        print(f"│ {'整体性能对比':<95} │")
        print(f"├{'─' * 98}┤")
        print(f"│ {'方法':<35} {'Accuracy':<20} {'Macro F1':<20} {'Weighted F1':<20} │")
        print(f"├{'─' * 98}┤")

        for i, method in enumerate(comparison['methods']):
            acc = comparison['accuracy'][i]
            macro_f1 = comparison['macro_f1'][i]
            weighted_f1 = comparison['weighted_f1'][i]

            # 标记最佳值
            acc_mark = " 🏆" if acc == max(comparison['accuracy']) else ""
            f1_mark = " 🏆" if macro_f1 == max(comparison['macro_f1']) else ""
            wf1_mark = " 🏆" if weighted_f1 == max(comparison['weighted_f1']) else ""

            print(f"│ {method:<35} "
                  f"{acc:.4f}{acc_mark:<15} "
                  f"{macro_f1:.4f}{f1_mark:<15} "
                  f"{weighted_f1:.4f}{wf1_mark:<15} │")

        print(f"└{'─' * 98}┘")

        # 性能提升分析（如果有两个方法）
        if len(comparison['methods']) == 2:
            method1 = comparison['methods'][0]
            method2 = comparison['methods'][1]
            print(f"\n┌{'─' * 98}┐")
            print(f"│ 性能提升分析 ({method2} vs {method1})"[:97].ljust(97) + " │")
            print(f"├{'─' * 98}┤")

            acc_diff = comparison['accuracy'][1] - comparison['accuracy'][0]
            f1_diff = comparison['macro_f1'][1] - comparison['macro_f1'][0]
            wf1_diff = comparison['weighted_f1'][1] - comparison['weighted_f1'][0]

            acc_pct = (acc_diff / comparison['accuracy'][0] * 100) if comparison['accuracy'][0] > 0 else 0
            f1_pct = (f1_diff / comparison['macro_f1'][0] * 100) if comparison['macro_f1'][0] > 0 else 0
            wf1_pct = (wf1_diff / comparison['weighted_f1'][0] * 100) if comparison['weighted_f1'][0] > 0 else 0

            def format_improvement(diff, pct):
                sign = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
                return f"{sign} {diff:+.4f} ({pct:+.2f}%)"

            print(f"│ {'Accuracy 变化:':<30} {format_improvement(acc_diff, acc_pct):<67} │")
            print(f"│ {'Macro F1 变化:':<30} {format_improvement(f1_diff, f1_pct):<67} │")
            print(f"│ {'Weighted F1 变化:':<30} {format_improvement(wf1_diff, wf1_pct):<67} │")
            print(f"└{'─' * 98}┘")

        # 每个类别的F1对比
        if 'per_class_metrics' in comparison:
            print(f"\n┌{'─' * 98}┐")
            print(f"│ {'各类别F1分数对比':<95} │")
            print(f"├{'─' * 98}┤")

            # 表头
            header = f"│ {'类别':<20}"
            for method in comparison['methods']:
                header += f" {method[:18]:<20}"
            if len(comparison['methods']) == 2:
                header += f" {'提升':<15}"
            header += " │"
            print(header)
            print(f"├{'─' * 98}┤")

            # 每个类别的数据
            for rel_type in self.relation_types:
                if rel_type in comparison['per_class_metrics']:
                    f1_scores = comparison['per_class_metrics'][rel_type]['f1_scores']
                    row = f"│ {rel_type:<20}"

                    for f1 in f1_scores:
                        row += f" {f1:<20.4f}"

                    # 添加提升
                    if len(f1_scores) == 2:
                        diff = f1_scores[1] - f1_scores[0]
                        pct = (diff / f1_scores[0] * 100) if f1_scores[0] > 0 else 0
                        symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
                        row += f" {symbol}{abs(diff):.4f} ({pct:+.1f}%)"[:15].ljust(15)

                    row += " │"
                    print(row)

            print(f"└{'─' * 98}┘")

        # 最佳方法推荐
        best_acc_idx = np.argmax(comparison['accuracy'])
        best_f1_idx = np.argmax(comparison['macro_f1'])

        print(f"\n💡 推荐结论:")
        if best_acc_idx == best_f1_idx:
            print(f"   🎯 最佳方法: {comparison['methods'][best_acc_idx]}")
            print(f"      ├─ Accuracy: {comparison['accuracy'][best_acc_idx]:.4f} (最高)")
            print(f"      └─ Macro F1: {comparison['macro_f1'][best_f1_idx]:.4f} (最高)")
        else:
            print(f"   ⚖️  方法各有优势:")
            print(f"      ├─ {comparison['methods'][best_acc_idx]} 在 Accuracy 上表现最佳 ({comparison['accuracy'][best_acc_idx]:.4f})")
            print(f"      └─ {comparison['methods'][best_f1_idx]} 在 Macro F1 上表现最佳 ({comparison['macro_f1'][best_f1_idx]:.4f})")

        print("="*100)


if __name__ == "__main__":
    # 测试
    print("\n评估指标计算器测试\n")

    calculator = MetricsCalculator()

    # 模拟预测结果
    predictions = [
        {'predicted_type': 'Overcomes', 'ground_truth': 'Overcomes', 'confidence': 0.9},
        {'predicted_type': 'Extends', 'ground_truth': 'Overcomes', 'confidence': 0.7},
        {'predicted_type': 'Realizes', 'ground_truth': 'Realizes', 'confidence': 0.8},
        {'predicted_type': 'Baselines', 'ground_truth': 'Extends', 'confidence': 0.6},
        {'predicted_type': 'Extends', 'ground_truth': 'Extends', 'confidence': 0.85},
    ]

    # 生成报告
    report = calculator.generate_full_report(predictions, method_name="Test Method")

    # 打印报告
    calculator.print_report(report)
