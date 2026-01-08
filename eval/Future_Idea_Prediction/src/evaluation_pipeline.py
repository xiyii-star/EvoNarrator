"""
主评估流程：
整合向量检索、LLM评估和指标计算的完整pipeline
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from vector_retrieval import EmbeddingModel, PaperVectorRetrieval
from llm_evaluator import LLMEvaluator
from metrics_calculator import MetricsCalculator
from config_loader import ConfigLoader


class EvaluationPipeline:
    """完整的评估流程"""

    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        初始化评估pipeline

        Args:
            embedding_model_name: Embedding模型名称（如果为None则从配置文件读取）
            llm_model_name: LLM模型名称（如果为None则从配置文件读取）
            llm_api_key: LLM API密钥（如果为None则从配置文件读取）
            llm_base_url: LLM API基础URL（如果为None则从配置文件读取）
            config_path: 配置文件路径（如果为None则使用默认路径）
        """
        print("Initializing Evaluation Pipeline...")

        # 加载配置
        try:
            self.config_loader = ConfigLoader(config_path)
            llm_config = self.config_loader.get_llm_config()
            embedding_config = self.config_loader.get_embedding_config()

            print("\n" + "="*60)
            print("Loading configuration from config file...")
            self.config_loader.print_config_summary()
            print("="*60 + "\n")

            # 使用配置文件的值，除非明确提供了参数
            final_embedding_model = embedding_model_name or embedding_config['model']
            final_llm_config = {
                'model': llm_model_name or llm_config['model'],
                'api_key': llm_api_key or llm_config['api_key'],
                'base_url': llm_base_url or llm_config['base_url'],
                'temperature': llm_config['temperature'],
                'max_tokens': llm_config['max_tokens']
            }

        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            print("Using command-line parameters or defaults...")
            final_embedding_model = embedding_model_name or "sentence-transformers/all-MiniLM-L6-v2"
            final_llm_config = {
                'model': llm_model_name or "gpt-4",
                'api_key': llm_api_key,
                'base_url': llm_base_url,
                'temperature': 0.3,
                'max_tokens': 500
            }

        # 初始化组件
        self.embedding_model = EmbeddingModel(final_embedding_model)
        self.retrieval_system = PaperVectorRetrieval(self.embedding_model)
        self.llm_evaluator = LLMEvaluator(config_dict=final_llm_config)
        self.metrics_calculator = MetricsCalculator()

        print("\nPipeline initialized successfully!\n")

    def build_vector_database(
        self,
        jsonl_files: List[str],
        save_dir: str,
        force_rebuild: bool = False
    ):
        """
        构建或加载向量数据库

        Args:
            jsonl_files: JSONL文件路径列表
            save_dir: 向量数据库保存目录
            force_rebuild: 是否强制重建
        """
        save_path = Path(save_dir)

        # 检查是否已存在
        if save_path.exists() and not force_rebuild:
            print(f"Loading existing vector database from {save_dir}")
            self.retrieval_system.load_database(save_dir)
            return

        print("Building vector database from scratch...")

        # 加载论文
        papers = self.retrieval_system.load_papers_from_jsonl(jsonl_files)

        # 构建向量数据库
        self.retrieval_system.build_vector_database(papers)

        # 保存
        self.retrieval_system.save_database(save_dir)

    def load_generated_ideas(self, ideas_file: str) -> List[Dict]:
        """
        加载生成的ideas

        Args:
            ideas_file: ideas文件路径（JSON或JSONL）

        Returns:
            ideas列表，每个包含idea_id和idea_text
        """
        print(f"Loading generated ideas from {ideas_file}")

        ideas_path = Path(ideas_file)

        # 如果是相对路径且文件不存在，尝试从父目录查找
        if not ideas_path.is_absolute() and not ideas_path.exists():
            # 尝试从当前脚本的父目录查找
            parent_path = Path(__file__).parent.parent / ideas_file
            if parent_path.exists():
                ideas_path = parent_path

        if not ideas_path.exists():
            raise FileNotFoundError(f"Ideas file not found: {ideas_file} (also tried: {ideas_path})")

        ideas = []

        if ideas_path.suffix == ".json":
            with open(ideas_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                # 处理不同的JSON格式
                if isinstance(data, dict):
                    # 检查是否有ideas字段
                    if "ideas" in data:
                        raw_ideas = data["ideas"]
                    else:
                        # 如果是字典格式，尝试提取key-value
                        raw_ideas = [{"idea_id": k, "idea_text": v} for k, v in data.items()]
                elif isinstance(data, list):
                    raw_ideas = data
                else:
                    raise ValueError(f"Unexpected JSON structure in {ideas_file}")

                # 标准化ideas格式
                for i, idea in enumerate(raw_ideas):
                    if isinstance(idea, dict):
                        # 如果已经有idea_id和idea_text，直接使用
                        if "idea_id" in idea and "idea_text" in idea:
                            ideas.append(idea)
                        # 否则尝试从其他字段提取
                        else:
                            idea_id = idea.get("idea_id") or f"idea_{i+1:03d}"
                            # 尝试从多个可能的字段提取文本
                            idea_text = (
                                idea.get("idea_text") or
                                idea.get("abstract") or
                                idea.get("title", "") + "\n" + idea.get("abstract", "")
                            )
                            if idea_text:
                                ideas.append({
                                    "idea_id": idea_id,
                                    "idea_text": idea_text.strip()
                                })
                    elif isinstance(idea, str):
                        ideas.append({
                            "idea_id": f"idea_{i+1:03d}",
                            "idea_text": idea
                        })

        elif ideas_path.suffix == ".jsonl":
            with open(ideas_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    idea = json.loads(line.strip())
                    if "idea_id" not in idea:
                        idea["idea_id"] = f"idea_{i+1:03d}"
                    if "idea_text" not in idea and "abstract" in idea:
                        idea["idea_text"] = idea["abstract"]
                    ideas.append(idea)

        print(f"Loaded {len(ideas)} ideas")
        return ideas

    def retrieve_candidates(
        self,
        ideas: List[Dict],
        top_k: int = 5
    ) -> Dict[str, List]:
        """
        为每个idea检索候选论文

        Args:
            ideas: ideas列表
            top_k: 每个idea检索的论文数量

        Returns:
            Dict[idea_id, List[(paper, similarity_score)]]
        """
        print(f"Retrieving top-{top_k} candidates for each idea...")

        retrieval_results = {}

        for idea in tqdm(ideas, desc="Retrieving"):
            idea_id = idea["idea_id"]
            idea_text = idea["idea_text"]

            candidates = self.retrieval_system.retrieve_similar_papers(
                idea_text,
                top_k=top_k
            )

            retrieval_results[idea_id] = candidates

        return retrieval_results

    def evaluate_ideas(
        self,
        ideas: List[Dict],
        retrieval_results: Dict[str, List],
        threshold: float = 7.0
    ) -> List[Dict]:
        """
        使用LLM评估ideas

        Args:
            ideas: ideas列表
            retrieval_results: 检索结果
            threshold: 匹配阈值

        Returns:
            评估结果列表
        """
        print("Evaluating ideas with LLM...")

        evaluation_results = self.llm_evaluator.batch_evaluate_ideas(
            ideas,
            retrieval_results,
            threshold=threshold
        )

        return evaluation_results

    def calculate_metrics(
        self,
        evaluation_results: List[Dict],
        k_values: List[int] = [1, 3, 5, 10],
        threshold: float = 7.0
    ) -> Dict:
        """
        计算评估指标

        Args:
            evaluation_results: 评估结果列表
            k_values: K值列表
            threshold: 匹配阈值

        Returns:
            评估报告
        """
        print("Calculating metrics...")

        report = self.metrics_calculator.generate_report(
            evaluation_results,
            k_values=k_values,
            threshold=threshold
        )

        return report

    def run_full_evaluation(
        self,
        jsonl_files: List[str],
        ideas_file: str,
        vector_db_dir: str,
        results_dir: str,
        top_k: int = 5,
        threshold: float = 7.0,
        k_values: List[int] = [1, 3, 5, 10],
        force_rebuild_db: bool = False
    ):
        """
        运行完整的评估流程

        Args:
            jsonl_files: 论文JSONL文件列表
            ideas_file: 生成的ideas文件
            vector_db_dir: 向量数据库目录
            results_dir: 结果保存目录
            top_k: 检索的候选论文数量
            threshold: 匹配阈值
            k_values: Hit Rate计算的K值列表
            force_rebuild_db: 是否强制重建向量数据库
        """
        print("\n" + "="*80)
        print("STARTING FULL EVALUATION PIPELINE".center(80))
        print("="*80 + "\n")

        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        # Step 1: 构建/加载向量数据库
        print("\nStep 1: Building/Loading Vector Database")
        print("-" * 80)
        self.build_vector_database(jsonl_files, vector_db_dir, force_rebuild_db)

        # Step 2: 加载生成的ideas
        print("\nStep 2: Loading Generated Ideas")
        print("-" * 80)
        ideas = self.load_generated_ideas(ideas_file)

        # Step 3: 检索候选论文
        print("\nStep 3: Retrieving Candidate Papers")
        print("-" * 80)
        retrieval_results = self.retrieve_candidates(ideas, top_k=top_k)

        # 保存检索结果
        retrieval_save_path = results_path / "retrieval_results.json"
        with open(retrieval_save_path, "w", encoding="utf-8") as f:
            # 转换为可序列化的格式
            serializable_results = {
                idea_id: [
                    {"paper": paper, "similarity": float(score)}
                    for paper, score in results
                ]
                for idea_id, results in retrieval_results.items()
            }
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        print(f"Retrieval results saved to {retrieval_save_path}")

        # Step 4: LLM评估
        print("\nStep 4: LLM Evaluation")
        print("-" * 80)
        evaluation_results = self.evaluate_ideas(
            ideas,
            retrieval_results,
            threshold=threshold
        )

        # 保存评估结果
        eval_save_path = results_path / "evaluation_results.json"
        with open(eval_save_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        print(f"Evaluation results saved to {eval_save_path}")

        # Step 5: 计算指标
        print("\nStep 5: Calculating Metrics")
        print("-" * 80)
        report = self.calculate_metrics(
            evaluation_results,
            k_values=k_values,
            threshold=threshold
        )

        # 保存报告
        report_save_path = results_path / "metrics_report.json"
        with open(report_save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Metrics report saved to {report_save_path}")

        print("\n" + "="*80)
        print("EVALUATION PIPELINE COMPLETED".center(80))
        print("="*80 + "\n")

        return {
            "retrieval_results": retrieval_results,
            "evaluation_results": evaluation_results,
            "metrics_report": report
        }


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="Future Idea Prediction Evaluation Pipeline")

    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to config.yaml file (default: ../../config/config.yaml)"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/lexy/下载/CLwithRAG/KGdemo/eval/Future_Idea_Prediction/data/iclr",
        help="Directory containing JSONL files"
    )

    parser.add_argument(
        "--ideas_file",
        type=str,
        required=True,
        help="Path to generated ideas file (JSON or JSONL)"
    )

    parser.add_argument(
        "--vector_db_dir",
        type=str,
        default="/home/lexy/下载/CLwithRAG/KGdemo/eval/Future_Idea_Prediction/data/vector_db",
        help="Directory for vector database"
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="/home/lexy/下载/CLwithRAG/KGdemo/eval/Future_Idea_Prediction/results",
        help="Directory to save results"
    )

    parser.add_argument(
        "--embedding_model",
        type=str,
        default=None,
        help="Embedding model name (overrides config file)"
    )

    parser.add_argument(
        "--llm_model",
        type=str,
        default=None,
        help="LLM model name (overrides config file)"
    )

    parser.add_argument(
        "--llm_api_key",
        type=str,
        default=None,
        help="LLM API key (overrides config file)"
    )

    parser.add_argument(
        "--llm_base_url",
        type=str,
        default=None,
        help="LLM API base URL (overrides config file)"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of candidate papers to retrieve"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=7.0,
        help="Match score threshold"
    )

    parser.add_argument(
        "--force_rebuild_db",
        action="store_true",
        help="Force rebuild vector database"
    )

    args = parser.parse_args()

    # 构建JSONL文件列表
    data_dir = Path(args.data_dir)
    jsonl_files = [
        str(data_dir / "iclr_2023_submitted.jsonl"),
        str(data_dir / "iclr2024_submissions.jsonl"),
        str(data_dir / "iclr2025_submissions.jsonl"),
    ]

    # 初始化pipeline（优先使用config.yaml配置）
    pipeline = EvaluationPipeline(
        embedding_model_name=args.embedding_model,
        llm_model_name=args.llm_model,
        llm_api_key=args.llm_api_key,
        llm_base_url=args.llm_base_url,
        config_path=args.config_path
    )

    # 运行评估
    pipeline.run_full_evaluation(
        jsonl_files=jsonl_files,
        ideas_file=args.ideas_file,
        vector_db_dir=args.vector_db_dir,
        results_dir=args.results_dir,
        top_k=args.top_k,
        threshold=args.threshold,
        k_values=[1, 3, 5, 10, 20],
        force_rebuild_db=args.force_rebuild_db
    )


if __name__ == "__main__":
    main()
