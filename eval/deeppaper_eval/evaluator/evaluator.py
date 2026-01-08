#!/usr/bin/env python3
"""
Main Evaluator for Deep Paper Information Extraction
主评估器 - 对比所有方法与人工标注的结果

Evaluates:
- My Method (DeepPaper Multi-Agent)
- LLM + RAG
- Pure RAG
- Baseline: Naive LLM
- Ablation: No Navigator
- Ablation: No Critic

Against human annotations in golden set
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
import argparse
from datetime import datetime
import sys

# Add parent to path
sys.path.append(str(Path(__file__).parent))

from metrics import compute_all_metrics, aggregate_metrics, compute_bertscore


class DeepPaperEvaluator:
    """评估器类"""

    def __init__(
        self,
        golden_set_path: str,
        use_llm_eval: bool = False,
        use_bertscore: bool = False,
        llm_model: str = "gpt-4o-2024-11-20",
        bertscore_model_path: str = None,
        config_path: str = None
    ):
        """
        Initialize evaluator

        Args:
            golden_set_path: Path to golden set Excel with human annotations
            use_llm_eval: Whether to use LLM-based evaluation (slower, more expensive)
            use_bertscore: Whether to include BERTScore (requires bert-score package)
            llm_model: LLM model for evaluation
            bertscore_model_path: Optional local path to BERTScore model
            config_path: Path to config.yaml file (optional)
        """
        self.golden_set_path = golden_set_path
        self.use_llm_eval = use_llm_eval
        self.use_bertscore = use_bertscore
        self.llm_model = llm_model
        self.bertscore_model_path = bertscore_model_path
        self.llm_config = None

        # Load LLM configuration if use_llm_eval is enabled
        if self.use_llm_eval:
            self.llm_config = self._load_llm_config(config_path)

        # Load golden set
        print(f"📚 Loading golden set from: {golden_set_path}")
        self.golden_df = pd.read_excel(golden_set_path)

        # 检测列名格式 (兼容大小写)
        problem_col = 'human_problem' if 'human_problem' in self.golden_df.columns else 'Human_Problem'
        contribution_col = 'human_contribution' if 'human_contribution' in self.golden_df.columns else 'Human_Contribution'
        limitation_col = 'Human_Limitation' if 'Human_Limitation' in self.golden_df.columns else 'human_limitation'
        future_work_col = 'Human_Future_Work' if 'Human_Future_Work' in self.golden_df.columns else 'human_future_work'

        # Filter annotated papers
        self.annotated_df = self.golden_df[
            self.golden_df[problem_col].notna() & (self.golden_df[problem_col] != '') &
            self.golden_df[contribution_col].notna() & (self.golden_df[contribution_col] != '')
        ]

        print(f"✅ Loaded {len(self.annotated_df)} annotated papers")

        # Create reference dict (support both Paper_ID and OpenAlex_ID)
        self.references = {}
        for _, row in self.annotated_df.iterrows():
            # 使用Paper_ID作为主键
            paper_id = row['Paper_ID']
            ref_data = {
                'problem': row.get(problem_col, ''),
                'method': row.get(contribution_col, ''),  # 将Contribution映射到method
                'limitation': row.get(limitation_col, ''),
                'future_work': row.get(future_work_col, '')
            }
            self.references[paper_id] = ref_data

            # 同时用OpenAlex_ID作为键(如果存在)
            if 'OpenAlex_ID' in row and pd.notna(row['OpenAlex_ID']):
                openalex_id = row['OpenAlex_ID']
                self.references[openalex_id] = ref_data

    def _load_llm_config(self, config_path: str = None) -> Dict:
        """
        Load LLM configuration from config file

        Args:
            config_path: Path to config file (optional)

        Returns:
            Dict with LLM configuration
        """
        try:
            import yaml

            # Determine config path
            if config_path is None:
                # Use default path: project_root/config/config.yaml
                project_root = Path(__file__).parent.parent.parent.parent
                config_path = project_root / "config" / "config.yaml"
            else:
                config_path = Path(config_path)

            # Load config file
            if not config_path.exists():
                print(f"⚠️  Config file not found: {config_path}")
                print("    Will try to use environment variables for LLM config")
                return None

            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            llm_cfg = config.get('llm', {})

            if not llm_cfg.get('enabled', False):
                print("⚠️  LLM not enabled in config.yaml")
                return None

            llm_config = {
                'api_key': llm_cfg.get('api_key'),
                'base_url': llm_cfg.get('base_url'),
                'model': llm_cfg.get('model', 'gpt-4'),
                'temperature': llm_cfg.get('temperature', 0.3),
                'max_tokens': llm_cfg.get('max_tokens', 4096)
            }

            # Override model if specified in command line
            if self.llm_model != "gpt-4":  # "gpt-4" is the default
                llm_config['model'] = self.llm_model

            print(f"✅ Loaded LLM config from: {config_path}")
            print(f"   Model: {llm_config['model']}")
            print(f"   Base URL: {llm_config['base_url'] or 'default'}")

            return llm_config

        except Exception as e:
            print(f"⚠️  Error loading LLM config: {e}")
            print("    Will try to use environment variables for LLM config")
            return None

    def load_results(self, results_path: str) -> Dict:
        """
        Load results from JSON file

        Args:
            results_path: Path to results JSON

        Returns:
            Dict with method results
        """
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def evaluate_method(
        self,
        method_name: str,
        results: List[Dict]
    ) -> Dict:
        """
        Evaluate a single method against golden set with optional BERTScore

        Args:
            method_name: Name of the method
            results: List of result dicts with paper_id and extracted fields

        Returns:
            Dict with evaluation metrics
        """
        print(f"\n{'='*80}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*80}")

        fields = ['problem', 'method', 'limitation', 'future_work']
        field_metrics = {field: [] for field in fields}
        field_texts = {field: {'predictions': [], 'references': []} for field in fields}
        paper_scores = []

        # Evaluate each paper
        for result in results:
            paper_id = result['paper_id']

            if paper_id not in self.references:
                print(f"  ⚠️  {paper_id} not in golden set, skipping...")
                continue

            reference = self.references[paper_id]
            paper_score = {'paper_id': paper_id}

            # Evaluate each field
            for field in fields:
                pred = result.get(field, '')
                ref = reference.get(field, '')

                # Skip if reference is empty
                if not ref or ref.strip() == '':
                    continue

                # Compute standard metrics first
                metrics = compute_all_metrics(
                    prediction=pred,
                    reference=ref,
                    use_llm=self.use_llm_eval,
                    llm_model=self.llm_model,
                    llm_config=self.llm_config
                )

                # Store metrics
                field_metrics[field].append(metrics)

                # Store for batch BERTScore computation (must be after metrics to maintain index alignment)
                field_texts[field]['predictions'].append(pred)
                field_texts[field]['references'].append(ref)

                paper_score[field] = metrics.get('rouge1_f', 0.0)

            paper_scores.append(paper_score)

        # Compute BERTScore for each field (batch processing)
        if self.use_bertscore:
            print("\n🤖 Computing BERTScore (this may take a few minutes)...")
            for field in fields:
                if field_texts[field]['predictions']:
                    predictions = field_texts[field]['predictions']
                    references = field_texts[field]['references']

                    bert_scores = compute_bertscore(
                        predictions,
                        references,
                        model_path=self.bertscore_model_path
                    )

                    # Add BERTScore to metrics
                    for i, metrics in enumerate(field_metrics[field]):
                        if i < len(bert_scores['f1']):
                            metrics['bertscore_p'] = bert_scores['precision'][i]
                            metrics['bertscore_r'] = bert_scores['recall'][i]
                            metrics['bertscore_f1'] = bert_scores['f1'][i]

                    print(f"  ✅ BERTScore computed for {field}: {len(predictions)} samples")

        # Aggregate metrics for each field
        aggregated = {}
        for field in fields:
            if field_metrics[field]:
                field_agg = aggregate_metrics(field_metrics[field])
                aggregated[field] = field_agg
                print(f"\n{field.upper()}:")
                print(f"  ROUGE-1 F1: {field_agg.get('rouge1_f_mean', 0):.4f} ± {field_agg.get('rouge1_f_std', 0):.4f}")
                print(f"  ROUGE-2 F1: {field_agg.get('rouge2_f_mean', 0):.4f} ± {field_agg.get('rouge2_f_std', 0):.4f}")
                print(f"  ROUGE-L F1: {field_agg.get('rougeL_f_mean', 0):.4f} ± {field_agg.get('rougeL_f_std', 0):.4f}")
                print(f"  BLEU: {field_agg.get('bleu_mean', 0):.4f} ± {field_agg.get('bleu_std', 0):.4f}")

                if self.use_bertscore and 'bertscore_f1_mean' in field_agg:
                    print(f"  BERTScore F1: {field_agg.get('bertscore_f1_mean', 0):.4f} ± {field_agg.get('bertscore_f1_std', 0):.4f}")

                if self.use_llm_eval and 'llm_similarity_mean' in field_agg:
                    print(f"  LLM Similarity: {field_agg.get('llm_similarity_mean', 0):.4f}")

        # Overall average
        all_rouge1 = []
        all_rouge2 = []
        all_rougeL = []
        all_bleu = []
        all_bertscore = []
        all_llm_similarity = []
        all_llm_coverage = []
        all_llm_accuracy = []

        for field in fields:
            if field in aggregated:
                all_rouge1.append(aggregated[field].get('rouge1_f_mean', 0))
                all_rouge2.append(aggregated[field].get('rouge2_f_mean', 0))
                all_rougeL.append(aggregated[field].get('rougeL_f_mean', 0))
                all_bleu.append(aggregated[field].get('bleu_mean', 0))
                if self.use_bertscore:
                    all_bertscore.append(aggregated[field].get('bertscore_f1_mean', 0))
                if self.use_llm_eval:
                    all_llm_similarity.append(aggregated[field].get('llm_similarity_mean', 0))
                    all_llm_coverage.append(aggregated[field].get('llm_coverage_mean', 0))
                    all_llm_accuracy.append(aggregated[field].get('llm_accuracy_mean', 0))

        overall = {
            'rouge1_f_mean': sum(all_rouge1) / len(all_rouge1) if all_rouge1 else 0,
            'rouge2_f_mean': sum(all_rouge2) / len(all_rouge2) if all_rouge2 else 0,
            'rougeL_f_mean': sum(all_rougeL) / len(all_rougeL) if all_rougeL else 0,
            'bleu_mean': sum(all_bleu) / len(all_bleu) if all_bleu else 0
        }

        if self.use_bertscore and all_bertscore:
            overall['bertscore_f1_mean'] = sum(all_bertscore) / len(all_bertscore)

        if self.use_llm_eval:
            if all_llm_similarity:
                overall['llm_similarity_mean'] = sum(all_llm_similarity) / len(all_llm_similarity)
            if all_llm_coverage:
                overall['llm_coverage_mean'] = sum(all_llm_coverage) / len(all_llm_coverage)
            if all_llm_accuracy:
                overall['llm_accuracy_mean'] = sum(all_llm_accuracy) / len(all_llm_accuracy)

        print(f"\nOVERALL (average across all fields):")
        print(f"  ROUGE-1 F1: {overall['rouge1_f_mean']:.4f}")
        print(f"  ROUGE-2 F1: {overall['rouge2_f_mean']:.4f}")
        print(f"  ROUGE-L F1: {overall['rougeL_f_mean']:.4f}")
        print(f"  BLEU: {overall['bleu_mean']:.4f}")
        if self.use_bertscore:
            print(f"  BERTScore F1: {overall.get('bertscore_f1_mean', 0):.4f}")
        if self.use_llm_eval:
            print(f"  LLM Similarity: {overall.get('llm_similarity_mean', 0):.4f}")
            print(f"  LLM Coverage: {overall.get('llm_coverage_mean', 0):.4f}")
            print(f"  LLM Accuracy: {overall.get('llm_accuracy_mean', 0):.4f}")

        return {
            'method_name': method_name,
            'overall': overall,
            'by_field': aggregated,
            'paper_scores': paper_scores,
            'num_papers': len(paper_scores)
        }

    def evaluate_all_methods(self, results_dir: str) -> Dict:
        """
        Evaluate all methods in results directory

        Args:
            results_dir: Directory containing result JSON files

        Returns:
            Dict with all evaluation results
        """
        results_path = Path(results_dir)

        # Find all result files
        result_files = list(results_path.glob("*_results.json"))

        if not result_files:
            print(f"❌ No result files found in {results_dir}")
            return {}

        print(f"\n📊 Found {len(result_files)} result files")

        all_evaluations = {}

        for result_file in result_files:
            print(f"\n📄 Loading {result_file.name}...")

            try:
                results_data = self.load_results(str(result_file))

                # Results file may have multiple methods
                for method_name, method_results in results_data.items():
                    # Skip non-list results (e.g., summary dict)
                    if not isinstance(method_results, list):
                        print(f"  ⚠️  Skipping '{method_name}' (not a results list)")
                        continue

                    evaluation = self.evaluate_method(method_name, method_results)
                    all_evaluations[method_name] = evaluation

            except Exception as e:
                print(f"❌ Error processing {result_file.name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        return all_evaluations

    def generate_comparison_table(self, all_evaluations: Dict) -> pd.DataFrame:
        """
        Generate comparison table across all methods

        Args:
            all_evaluations: Dict with evaluation results for all methods

        Returns:
            DataFrame with comparison
        """
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

            if self.use_bertscore:
                row['BERTScore'] = f"{overall.get('bertscore_f1_mean', 0):.4f}"

            if self.use_llm_eval:
                row['LLM_Similarity'] = f"{overall.get('llm_similarity_mean', 0):.4f}"
                row['LLM_Coverage'] = f"{overall.get('llm_coverage_mean', 0):.4f}"
                row['LLM_Accuracy'] = f"{overall.get('llm_accuracy_mean', 0):.4f}"

            # Add per-field scores
            for field in ['problem', 'method', 'limitation', 'future_work']:
                if field in evaluation['by_field']:
                    field_metrics = evaluation['by_field'][field]
                    row[f'{field}_rouge1'] = f"{field_metrics.get('rouge1_f_mean', 0):.4f}"

            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by ROUGE-L
        df['ROUGE-L_float'] = df['ROUGE-L'].astype(float)
        df = df.sort_values('ROUGE-L_float', ascending=False)
        df = df.drop('ROUGE-L_float', axis=1)

        return df

    def save_evaluation_report(
        self,
        all_evaluations: Dict,
        output_path: str
    ):
        """
        Save evaluation report

        Args:
            all_evaluations: Dict with all evaluation results
            output_path: Where to save report
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_file = output_file.with_suffix('.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(all_evaluations, f, indent=2, ensure_ascii=False)
        print(f"\n💾 JSON report saved to: {json_file}")

        # Generate comparison table
        comparison_df = self.generate_comparison_table(all_evaluations)

        # Save CSV
        csv_file = output_file.with_suffix('.csv')
        comparison_df.to_csv(csv_file, index=False)
        print(f"💾 CSV comparison saved to: {csv_file}")

        # Generate markdown report
        md_file = output_file.with_suffix('.md')
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# Deep Paper Evaluation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Golden Set:** {self.golden_set_path}\n\n")
            f.write(f"**Annotated Papers:** {len(self.annotated_df)}\n\n")

            f.write(f"**Metrics Used:**\n")
            f.write("- ROUGE-1/2/L\n")
            f.write("- BLEU\n")
            if self.use_bertscore:
                f.write("- BERTScore\n")
            if self.use_llm_eval:
                f.write(f"- LLM Evaluation ({self.llm_model})\n")
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
                if self.use_bertscore:
                    f.write(f"- BERTScore F1: {overall.get('bertscore_f1_mean', 0):.4f}\n")
                if self.use_llm_eval:
                    f.write(f"- LLM Similarity: {overall.get('llm_similarity_mean', 0):.4f}\n")
                    f.write(f"- LLM Coverage: {overall.get('llm_coverage_mean', 0):.4f}\n")
                    f.write(f"- LLM Accuracy: {overall.get('llm_accuracy_mean', 0):.4f}\n")
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
                        if self.use_bertscore and 'bertscore_f1_mean' in field_metrics:
                            f.write(f"- BERTScore F1: {field_metrics.get('bertscore_f1_mean', 0):.4f} ± {field_metrics.get('bertscore_f1_std', 0):.4f}\n")
                        if self.use_llm_eval:
                            if 'llm_similarity_mean' in field_metrics:
                                f.write(f"- LLM Similarity: {field_metrics.get('llm_similarity_mean', 0):.4f} ± {field_metrics.get('llm_similarity_std', 0):.4f}\n")
                            if 'llm_coverage_mean' in field_metrics:
                                f.write(f"- LLM Coverage: {field_metrics.get('llm_coverage_mean', 0):.4f} ± {field_metrics.get('llm_coverage_std', 0):.4f}\n")
                            if 'llm_accuracy_mean' in field_metrics:
                                f.write(f"- LLM Accuracy: {field_metrics.get('llm_accuracy_mean', 0):.4f} ± {field_metrics.get('llm_accuracy_std', 0):.4f}\n")
                        f.write("\n")

        print(f"💾 Markdown report saved to: {md_file}")

        # Print comparison table
        print("\n" + "="*80)
        print("📊 COMPARISON TABLE")
        print("="*80)
        print(comparison_df.to_string(index=False))
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate deep paper extraction methods with enhanced metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard evaluation (ROUGE + BLEU)
  python evaluator.py --golden_set data/golden_set_79papers.xlsx --results_dir result

  # With BERTScore
  python evaluator.py --golden_set data/golden_set_79papers.xlsx --results_dir result --use_bertscore

  # With LLM evaluation
  export OPENAI_API_KEY="your-key"
  python evaluator.py --golden_set data/golden_set_79papers.xlsx --results_dir result --use_llm_eval
        """
    )
    parser.add_argument('--golden_set', type=str, required=True,
                        help='Path to golden set Excel file')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing result JSON files')
    parser.add_argument('--output', type=str, default='./result/evaluation_report',
                        help='Output report path (without extension)')
    parser.add_argument('--use_bertscore', action='store_true',
                        help='Include BERTScore (requires bert-score package)')
    parser.add_argument('--bertscore_model', type=str, default=None,
                        help='Path to local BERTScore model (optional)')
    parser.add_argument('--use_llm_eval', action='store_true',
                        help='Use LLM-based evaluation (slower, more expensive)')
    parser.add_argument('--llm_model', type=str, default='gpt-4o-2024-11-20',
                        help='LLM model for evaluation')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml file (optional, will use project default if not specified)')

    args = parser.parse_args()

    print("="*80)
    print("Deep Paper Evaluation System")
    print("="*80)

    if args.use_bertscore:
        print("\n🤖 BERTScore enabled (this will take longer)")
        # Auto-detect local model path if not specified
        if args.bertscore_model is None:
            # Try to find local model in project directory
            project_root = Path(__file__).parent.parent.parent.parent

            # First try: specific DeBERTa-v2 model directory
            specific_model_path = project_root / "model" / "bertscore" / "microsoft" / "deberta-v2-xlarge-mnli"
            if specific_model_path.exists():
                args.bertscore_model = str(specific_model_path)
                print(f"   Using local model: {args.bertscore_model}")
            else:
                # Second try: general bertscore directory
                default_model_path = project_root / "model" / "bertscore"
                if default_model_path.exists():
                    args.bertscore_model = str(default_model_path)
                    print(f"   Using local model: {args.bertscore_model}")
                else:
                    print("   Warning: No local model specified, will try to download from HuggingFace")

    if args.use_llm_eval:
        print(f"\n🤖 LLM evaluation enabled using {args.llm_model}")

    # Initialize evaluator
    evaluator = DeepPaperEvaluator(
        golden_set_path=args.golden_set,
        use_llm_eval=args.use_llm_eval,
        use_bertscore=args.use_bertscore,
        llm_model=args.llm_model,
        bertscore_model_path=args.bertscore_model,
        config_path=args.config
    )

    # Evaluate all methods
    all_evaluations = evaluator.evaluate_all_methods(args.results_dir)

    if not all_evaluations:
        print("❌ No evaluations completed!")
        return

    # Save report
    evaluator.save_evaluation_report(all_evaluations, args.output)

    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
