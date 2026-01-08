#!/usr/bin/env python3
"""
Regenerate evaluation report with LLM metrics from existing JSON
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Load existing evaluation JSON
json_path = Path("result1217/evaluation_report_llm.json")
with open(json_path, 'r', encoding='utf-8') as f:
    all_evaluations = json.load(f)

output_path = "result1217/evaluation_report_llm_regenerated"
golden_set_path = "data/golden_set_79papers.xlsx"

# Configuration flags
use_llm_eval = True
use_bertscore = False

# Count annotated papers
golden_df = pd.read_excel(golden_set_path)
problem_col = 'human_problem' if 'human_problem' in golden_df.columns else 'Human_Problem'
contribution_col = 'human_contribution' if 'human_contribution' in golden_df.columns else 'Human_Contribution'
annotated_df = golden_df[
    golden_df[problem_col].notna() & (golden_df[problem_col] != '') &
    golden_df[contribution_col].notna() & (golden_df[contribution_col] != '')
]

# Generate comparison table
rows = []
for method_name, evaluation in all_evaluations.items():
    overall = evaluation['overall']
    row = {
        'Method': method_name,
        'ROUGE-1': f"{overall['rouge1_f_mean']:.4f}",
        'ROUGE-2': f"{overall['rouge2_f_mean']:.4f}",
        'ROUGE-L': f"{overall['rougeL_f_mean']:.4f}",
        'BLEU': f"{overall.get('bleu_mean', 0):.4f}",
        'Num Papers': evaluation['num_papers']
    }

    if use_bertscore:
        row['BERTScore'] = f"{overall.get('bertscore_f1_mean', 0):.4f}"

    if use_llm_eval:
        # Compute overall LLM metrics from per-field data
        llm_similarity_values = []
        llm_coverage_values = []
        llm_accuracy_values = []

        for field in ['problem', 'method', 'limitation', 'future_work']:
            if field in evaluation['by_field']:
                field_metrics = evaluation['by_field'][field]
                if 'llm_similarity_mean' in field_metrics:
                    llm_similarity_values.append(field_metrics['llm_similarity_mean'])
                if 'llm_coverage_mean' in field_metrics:
                    llm_coverage_values.append(field_metrics['llm_coverage_mean'])
                if 'llm_accuracy_mean' in field_metrics:
                    llm_accuracy_values.append(field_metrics['llm_accuracy_mean'])

        # Compute averages
        row['LLM_Similarity'] = f"{sum(llm_similarity_values) / len(llm_similarity_values) if llm_similarity_values else 0:.4f}"
        row['LLM_Coverage'] = f"{sum(llm_coverage_values) / len(llm_coverage_values) if llm_coverage_values else 0:.4f}"
        row['LLM_Accuracy'] = f"{sum(llm_accuracy_values) / len(llm_accuracy_values) if llm_accuracy_values else 0:.4f}"

    rows.append(row)

comparison_df = pd.DataFrame(rows)

# Sort by ROUGE-L
comparison_df['ROUGE-L_float'] = comparison_df['ROUGE-L'].astype(float)
comparison_df = comparison_df.sort_values('ROUGE-L_float', ascending=False)
comparison_df = comparison_df.drop('ROUGE-L_float', axis=1)

# Save CSV
csv_file = Path(output_path).with_suffix('.csv')
csv_file.parent.mkdir(parents=True, exist_ok=True)
comparison_df.to_csv(csv_file, index=False)
print(f"💾 CSV comparison saved to: {csv_file}")

# Generate markdown report
md_file = Path(output_path).with_suffix('.md')
with open(md_file, 'w', encoding='utf-8') as f:
    f.write("# Deep Paper Evaluation Report\n\n")
    f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Golden Set:** {golden_set_path}\n\n")
    f.write(f"**Annotated Papers:** {len(annotated_df)}\n\n")

    f.write(f"**Metrics Used:**\n")
    f.write("- ROUGE-1/2/L\n")
    f.write("- BLEU\n")
    if use_bertscore:
        f.write("- BERTScore\n")
    if use_llm_eval:
        f.write(f"- LLM Evaluation (gpt-4o-2024-11-20)\n")
    f.write("\n")

    f.write("## Overall Comparison\n\n")
    f.write(comparison_df.to_markdown(index=False))
    f.write("\n\n")

    # Detailed results for each method
    for method_name, evaluation in all_evaluations.items():
        f.write(f"## {method_name}\n\n")
        f.write(f"**Papers Evaluated:** {evaluation['num_papers']}\n\n")

        f.write("### Overall Metrics\n\n")
        overall = evaluation['overall']
        f.write(f"- ROUGE-1 F1: {overall['rouge1_f_mean']:.4f}\n")
        f.write(f"- ROUGE-2 F1: {overall['rouge2_f_mean']:.4f}\n")
        f.write(f"- ROUGE-L F1: {overall['rougeL_f_mean']:.4f}\n")
        f.write(f"- BLEU: {overall.get('bleu_mean', 0):.4f}\n")
        if use_bertscore:
            f.write(f"- BERTScore F1: {overall.get('bertscore_f1_mean', 0):.4f}\n")

        # Compute overall LLM metrics
        if use_llm_eval:
            llm_similarity_values = []
            llm_coverage_values = []
            llm_accuracy_values = []

            for field in ['problem', 'method', 'limitation', 'future_work']:
                if field in evaluation['by_field']:
                    field_metrics = evaluation['by_field'][field]
                    if 'llm_similarity_mean' in field_metrics:
                        llm_similarity_values.append(field_metrics['llm_similarity_mean'])
                    if 'llm_coverage_mean' in field_metrics:
                        llm_coverage_values.append(field_metrics['llm_coverage_mean'])
                    if 'llm_accuracy_mean' in field_metrics:
                        llm_accuracy_values.append(field_metrics['llm_accuracy_mean'])

            if llm_similarity_values:
                f.write(f"- LLM Similarity: {sum(llm_similarity_values) / len(llm_similarity_values):.4f}\n")
            if llm_coverage_values:
                f.write(f"- LLM Coverage: {sum(llm_coverage_values) / len(llm_coverage_values):.4f}\n")
            if llm_accuracy_values:
                f.write(f"- LLM Accuracy: {sum(llm_accuracy_values) / len(llm_accuracy_values):.4f}\n")
        f.write("\n")

        f.write("### Per-Field Metrics\n\n")
        for field in ['problem', 'method', 'limitation', 'future_work']:
            if field in evaluation['by_field']:
                field_metrics = evaluation['by_field'][field]
                f.write(f"#### {field.upper()}\n\n")
                f.write(f"- ROUGE-1: {field_metrics.get('rouge1_f_mean', 0):.4f} ± {field_metrics.get('rouge1_f_std', 0):.4f}\n")
                f.write(f"- ROUGE-2: {field_metrics.get('rouge2_f_mean', 0):.4f} ± {field_metrics.get('rouge2_f_std', 0):.4f}\n")
                f.write(f"- ROUGE-L: {field_metrics.get('rougeL_f_mean', 0):.4f} ± {field_metrics.get('rougeL_f_std', 0):.4f}\n")
                f.write(f"- BLEU: {field_metrics.get('bleu_mean', 0):.4f} ± {field_metrics.get('bleu_std', 0):.4f}\n")
                if use_bertscore and 'bertscore_f1_mean' in field_metrics:
                    f.write(f"- BERTScore F1: {field_metrics.get('bertscore_f1_mean', 0):.4f} ± {field_metrics.get('bertscore_f1_std', 0):.4f}\n")
                if use_llm_eval:
                    if 'llm_similarity_mean' in field_metrics:
                        f.write(f"- LLM Similarity: {field_metrics.get('llm_similarity_mean', 0):.4f} ± {field_metrics.get('llm_similarity_std', 0):.4f}\n")
                    if 'llm_coverage_mean' in field_metrics:
                        f.write(f"- LLM Coverage: {field_metrics.get('llm_coverage_mean', 0):.4f} ± {field_metrics.get('llm_coverage_std', 0):.4f}\n")
                    if 'llm_accuracy_mean' in field_metrics:
                        f.write(f"- LLM Accuracy: {field_metrics.get('llm_accuracy_mean', 0):.4f} ± {field_metrics.get('llm_accuracy_std', 0):.4f}\n")
                f.write("\n")

print(f"💾 Markdown report saved to: {md_file}")

# Print comparison table
print("\n" + "="*120)
print("📊 COMPARISON TABLE WITH LLM METRICS")
print("="*120)
print(comparison_df.to_string(index=False))
print("="*120)
print("\n✅ Report regeneration complete!")
