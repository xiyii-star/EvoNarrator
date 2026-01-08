#!/usr/bin/env python3
"""
Run All Experiments Script
运行所有对比实验并进行评估

Experiments:
1. My Method (DeepPaper Multi-Agent)
2. LLM + RAG
3. Pure RAG
4. Baseline: Naive LLM
5. Ablation: No Navigator
6. Ablation: No Critic

Then run evaluation to compare all methods
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime


class ExperimentRunner:
    """实验运行器"""

    def __init__(
        self,
        golden_set: str,
        papers_dir: str,
        results_dir: str,
        config_path: str = None,
        grobid_url: str = None
    ):
        """
        Initialize experiment runner

        Args:
            golden_set: Path to golden set Excel
            papers_dir: Directory with paper text files
            results_dir: Directory to save results
            config_path: LLM config file path
            grobid_url: GROBID service URL
        """
        self.golden_set = golden_set
        self.papers_dir = papers_dir
        self.results_dir = results_dir
        self.config_path = config_path
        self.grobid_url = grobid_url

        # Create results directory
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        # Get src directory
        self.src_dir = Path(__file__).parent / "src"

    def run_command(self, cmd: list, description: str) -> bool:
        """
        Run a command and capture output

        Args:
            cmd: Command as list
            description: Description of what's running

        Returns:
            True if successful, False otherwise
        """
        print("\n" + "="*80)
        print(f"🚀 {description}")
        print("="*80)
        print(f"Command: {' '.join(cmd)}")
        print("")

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True
            )

            elapsed = time.time() - start_time
            print(f"\n✅ {description} completed in {elapsed:.1f}s")
            return True

        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            print(f"\n❌ {description} failed after {elapsed:.1f}s")
            print(f"Error: {e}")
            return False

        except KeyboardInterrupt:
            print(f"\n⚠️  {description} interrupted by user")
            raise

    def run_mymethod(self) -> bool:
        """Run my method (DeepPaper Multi-Agent)"""
        output_file = Path(self.results_dir) / "mymethod_results.json"

        cmd = [
            sys.executable,
            str(self.src_dir / "mymethod.py"),
            "--golden_set", self.golden_set,
            "--papers_dir", self.papers_dir,
            "--output", str(output_file)
        ]

        if self.config_path:
            cmd.extend(["--config", self.config_path])

        if self.grobid_url:
            cmd.extend(["--grobid_url", self.grobid_url])

        return self.run_command(cmd, "My Method (DeepPaper Multi-Agent)")

    def run_llm_rag(self) -> bool:
        """Run LLM + RAG method"""
        output_file = Path(self.results_dir) / "llm_rag_results.json"

        cmd = [
            sys.executable,
            str(self.src_dir / "llm_rag_paper.py"),
            "--golden_set", self.golden_set,
            "--papers_dir", self.papers_dir,
            "--output", str(output_file)
        ]

        if self.config_path:
            cmd.extend(["--config", self.config_path])

        if self.grobid_url:
            cmd.extend(["--grobid_url", self.grobid_url])

        return self.run_command(cmd, "LLM + RAG Method")

    def run_pure_rag(self) -> bool:
        """Run pure RAG method"""
        output_file = Path(self.results_dir) / "pure_rag_results.json"

        cmd = [
            sys.executable,
            str(self.src_dir / "rag_paper.py"),
            "--golden_set", self.golden_set,
            "--papers_dir", self.papers_dir,
            "--output", str(output_file)
        ]

        return self.run_command(cmd, "Pure RAG Method")

    def run_naive_baseline(self) -> bool:
        """Run naive LLM baseline"""
        output_file = Path(self.results_dir) / "naive_baseline_results.json"

        cmd = [
            sys.executable,
            str(self.src_dir / "baseline_naive_llm.py"),
            "--golden_set", self.golden_set,
            "--papers_dir", self.papers_dir,
            "--output", str(output_file)
        ]

        if self.config_path:
            cmd.extend(["--config", self.config_path])

        return self.run_command(cmd, "Baseline: Naive LLM")

    def run_ablation_no_navigator(self) -> bool:
        """Run ablation study: no navigator"""
        output_file = Path(self.results_dir) / "ablation_no_navigator_results.json"

        cmd = [
            sys.executable,
            str(self.src_dir / "ablation_no_navigator.py"),
            "--golden_set", self.golden_set,
            "--papers_dir", self.papers_dir,
            "--output", str(output_file)
        ]

        if self.config_path:
            cmd.extend(["--config", self.config_path])

        return self.run_command(cmd, "Ablation: No Navigator")

    def run_ablation_no_critic(self) -> bool:
        """Run ablation study: no critic"""
        output_file = Path(self.results_dir) / "ablation_no_critic_results.json"

        cmd = [
            sys.executable,
            str(self.src_dir / "ablation_no_critic.py"),
            "--golden_set", self.golden_set,
            "--papers_dir", self.papers_dir,
            "--output", str(output_file)
        ]

        if self.config_path:
            cmd.extend(["--config", self.config_path])

        return self.run_command(cmd, "Ablation: No Critic")

    def run_evaluation(self, use_llm_eval: bool = False) -> bool:
        """Run evaluation on all results"""
        evaluator_dir = Path(__file__).parent / "evaluator"
        output_file = Path(self.results_dir) / "evaluation_report"

        cmd = [
            sys.executable,
            str(evaluator_dir / "evaluator.py"),
            "--golden_set", self.golden_set,
            "--results_dir", self.results_dir,
            "--output", str(output_file)
        ]

        if use_llm_eval:
            cmd.append("--use_llm_eval")

        if self.config_path:
            cmd.extend(["--config", self.config_path])

        return self.run_command(cmd, "Evaluation")

    def run_all(
        self,
        skip_existing: bool = False,
        run_evaluation: bool = True,
        use_llm_eval: bool = False,
        methods: list = None
    ):
        """
        Run all experiments

        Args:
            skip_existing: Skip methods that already have results
            run_evaluation: Whether to run evaluation after experiments
            use_llm_eval: Use LLM-based evaluation
            methods: List of specific methods to run (None = all)
        """
        print("\n" + "▓"*80)
        print("Deep Paper Evaluation Experiment Suite")
        print("▓"*80)
        print(f"\nGolden Set: {self.golden_set}")
        print(f"Papers Dir: {self.papers_dir}")
        print(f"Results Dir: {self.results_dir}")
        print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("▓"*80)

        # Define all methods
        all_methods = {
            'mymethod': self.run_mymethod,
            'llm_rag': self.run_llm_rag,
            'pure_rag': self.run_pure_rag,
            'naive_baseline': self.run_naive_baseline,
            'ablation_no_navigator': self.run_ablation_no_navigator,
            'ablation_no_critic': self.run_ablation_no_critic
        }

        # Filter methods if specified
        if methods:
            methods_to_run = {k: v for k, v in all_methods.items() if k in methods}
        else:
            methods_to_run = all_methods

        # Track results
        results = {}
        start_time = time.time()

        # Run each method
        for method_name, method_func in methods_to_run.items():
            try:
                success = method_func()
                results[method_name] = "✅ Success" if success else "❌ Failed"
            except KeyboardInterrupt:
                print("\n\n⚠️  Experiment suite interrupted by user")
                results[method_name] = "⚠️  Interrupted"
                break
            except Exception as e:
                print(f"\n❌ Unexpected error in {method_name}: {e}")
                results[method_name] = f"❌ Error: {e}"

        # Run evaluation
        if run_evaluation:
            try:
                eval_success = self.run_evaluation(use_llm_eval)
                results['evaluation'] = "✅ Success" if eval_success else "❌ Failed"
            except Exception as e:
                print(f"\n❌ Evaluation error: {e}")
                results['evaluation'] = f"❌ Error: {e}"

        # Print summary
        total_time = time.time() - start_time
        print("\n\n" + "▓"*80)
        print("EXPERIMENT SUMMARY")
        print("▓"*80)
        print(f"\nTotal Time: {total_time/60:.1f} minutes")
        print("\nResults:")
        for method, status in results.items():
            print(f"  {method:25s} {status}")

        print("\n" + "▓"*80)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("▓"*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run all deep paper evaluation experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python run_all_experiments.py --golden_set data/golden_set_79papers.xlsx --papers_dir data/papers

  # Run only specific methods
  python run_all_experiments.py --golden_set data/golden_set_79papers.xlsx --papers_dir data/papers --methods mymethod llm_rag

  # Skip evaluation
  python run_all_experiments.py --golden_set data/golden_set_79papers.xlsx --papers_dir data/papers --no-eval

Available methods:
  - mymethod: DeepPaper Multi-Agent (our proposed method)
  - llm_rag: LLM + RAG
  - pure_rag: Pure RAG (retrieval only)
  - naive_baseline: Naive LLM
  - ablation_no_navigator: Ablation without Navigator
  - ablation_no_critic: Ablation without Critic
        """
    )

    parser.add_argument('--golden_set', type=str, required=True,
                        help='Path to golden set Excel file')
    parser.add_argument('--papers_dir', type=str, required=True,
                        help='Directory containing paper text files')
    parser.add_argument('--results_dir', type=str, default='./result',
                        help='Directory to save results (default: ./result)')
    parser.add_argument('--config', type=str, default=None,
                        help='LLM config file path')
    parser.add_argument('--grobid_url', type=str, default=None,
                        help='GROBID service URL (e.g., http://localhost:8070)')
    parser.add_argument('--methods', nargs='+', default=None,
                        choices=['mymethod', 'llm_rag', 'pure_rag', 'naive_baseline',
                                'ablation_no_navigator', 'ablation_no_critic'],
                        help='Specific methods to run (default: all)')
    parser.add_argument('--no-eval', action='store_true',
                        help='Skip evaluation after running experiments')
    parser.add_argument('--use_llm_eval', action='store_true',
                        help='Use LLM-based evaluation (slower, more expensive)')

    args = parser.parse_args()

    # Initialize runner
    runner = ExperimentRunner(
        golden_set=args.golden_set,
        papers_dir=args.papers_dir,
        results_dir=args.results_dir,
        config_path=args.config,
        grobid_url=args.grobid_url
    )

    # Run experiments
    try:
        runner.run_all(
            run_evaluation=not args.no_eval,
            use_llm_eval=args.use_llm_eval,
            methods=args.methods
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment suite terminated by user")
        sys.exit(1)


if __name__ == '__main__':
    main()
