"""
向量检索模块：
1. 加载论文abstract并转换为向量
2. 构建向量库
3. 对生成的idea进行Top-K检索
"""

import json
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm


class EmbeddingModel:
    """本地Embedding模型封装"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        初始化本地Embedding模型

        Args:
            model_name: HuggingFace模型名称或本地路径
        """
        try:
            import os
            from sentence_transformers import SentenceTransformer

            # 设置环境变量，避免联网检查
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'

            # 检查是否是本地路径
            local_model_path = self._get_local_model_path(model_name)

            if local_model_path and local_model_path.exists():
                print(f"Loading embedding model from local path: {local_model_path}")
                # 直接使用本地路径，禁用在线模式
                self.model = SentenceTransformer(str(local_model_path), device='cpu')
            else:
                print(f"Loading embedding model: {model_name}")
                print("Warning: Model not found locally, will try to download from HuggingFace")
                self.model = SentenceTransformer(model_name, device='cpu')

            print("Embedding model loaded successfully")
        except ImportError:
            raise ImportError(
                "Please install sentence-transformers: "
                "pip install sentence-transformers"
            )

    def _get_local_model_path(self, model_name: str) -> Path:
        """
        获取本地模型路径

        Args:
            model_name: 模型名称

        Returns:
            本地模型路径，如果不存在返回None
        """
        # 如果已经是绝对路径，直接返回
        model_path = Path(model_name)
        if model_path.is_absolute() and model_path.exists():
            return model_path

        # 检查项目根目录下的model文件夹
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent  # 回到KGdemo根目录

        # 尝试多种可能的本地路径
        possible_paths = [
            Path("/home/lexy/下载/CLwithRAG/KGdemo/model/sentence-transformers/all-MiniLM-L6-v2"),  # 直接硬编码路径
            project_root / "model" / "sentence-transformers" / "all-MiniLM-L6-v2",
            project_root / "model" / model_name,  # model/sentence-transformers/all-MiniLM-L6-v2
            project_root / "model" / model_name.split('/')[-1],  # model/all-MiniLM-L6-v2
        ]

        for path in possible_paths:
            if path.exists():
                # 验证是否包含必要的模型文件
                required_files = ['config.json', 'modules.json']
                if all((path / f).exists() for f in required_files):
                    return path

        return None

    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        批量编码文本为向量

        Args:
            texts: 文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条

        Returns:
            numpy数组，shape为(len(texts), embedding_dim)
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )


class VectorDatabase:
    """简单的内存向量数据库"""

    def __init__(self):
        """初始化向量数据库"""
        self.vectors = None  # numpy数组: (N, D)
        self.metadata = []   # 元数据列表

    def add(self, vectors: np.ndarray, metadata: List[Dict]):
        """
        添加向量和元数据到数据库

        Args:
            vectors: 向量数组，shape为(N, D)
            metadata: 元数据列表，长度为N
        """
        if self.vectors is None:
            self.vectors = vectors
            self.metadata = metadata
        else:
            self.vectors = np.vstack([self.vectors, vectors])
            self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        搜索最相似的向量

        Args:
            query_vector: 查询向量，shape为(D,)
            top_k: 返回最相似的K个结果

        Returns:
            List of (metadata, similarity_score) tuples
        """
        if self.vectors is None or len(self.vectors) == 0:
            return []

        # 计算余弦相似度
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        vectors_norm = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(vectors_norm, query_norm)

        # 获取top_k索引
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # 返回结果
        results = [
            (self.metadata[idx], float(similarities[idx]))
            for idx in top_indices
        ]

        return results

    def save(self, save_dir: str):
        """保存向量数据库到磁盘"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        if self.vectors is not None:
            np.save(save_path / "vectors.npy", self.vectors)

        with open(save_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        print(f"Vector database saved to {save_dir}")

    def load(self, load_dir: str):
        """从磁盘加载向量数据库"""
        load_path = Path(load_dir)

        vectors_file = load_path / "vectors.npy"
        if vectors_file.exists():
            self.vectors = np.load(vectors_file)

        metadata_file = load_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        print(f"Vector database loaded from {load_dir}")
        print(f"Total vectors: {len(self.vectors) if self.vectors is not None else 0}")


class PaperVectorRetrieval:
    """论文向量检索系统"""

    def __init__(self, embedding_model: EmbeddingModel):
        """
        初始化检索系统

        Args:
            embedding_model: Embedding模型实例
        """
        self.embedding_model = embedding_model
        self.vector_db = VectorDatabase()

    def load_papers_from_jsonl(self, jsonl_files: List[str]) -> List[Dict]:
        """
        从多个JSONL文件加载论文数据

        Args:
            jsonl_files: JSONL文件路径列表

        Returns:
            论文列表，每个论文包含id, title, abstract, year等字段
        """
        papers = []

        for jsonl_file in jsonl_files:
            print(f"Loading papers from {jsonl_file}")
            year = self._extract_year_from_filename(jsonl_file)

            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc=f"Reading {Path(jsonl_file).name}"):
                    try:
                        data = json.loads(line.strip())
                        paper = self._extract_paper_info(data, year)
                        if paper and paper.get("abstract"):
                            papers.append(paper)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line: {e}")
                        continue

        print(f"Total papers loaded: {len(papers)}")
        return papers

    def _extract_year_from_filename(self, filename: str) -> str:
        """从文件名提取年份"""
        import re
        match = re.search(r'(2023|2024|2025)', filename)
        return match.group(1) if match else "unknown"

    def _extract_paper_info(self, data: Dict, year: str) -> Dict:
        """
        从JSON数据提取论文信息

        Args:
            data: JSON数据
            year: 年份

        Returns:
            包含paper_id, title, abstract, year的字典
        """
        # 兼容不同的JSON格式
        if "content" in data:
            content = data["content"]

            # 提取title - 可能是字符串或字典
            title = content.get("title", "")
            if isinstance(title, dict):
                title = title.get("value", "")

            # 提取abstract - 可能是字符串或字典
            abstract = content.get("abstract", "")
            if isinstance(abstract, dict):
                abstract = abstract.get("value", "")

            return {
                "paper_id": data.get("id", ""),
                "title": title,
                "abstract": abstract,
                "year": year
            }
        else:
            # 备用格式
            return {
                "paper_id": data.get("id", ""),
                "title": data.get("title", ""),
                "abstract": data.get("abstract", ""),
                "year": year
            }

    def build_vector_database(self, papers: List[Dict], batch_size: int = 32):
        """
        构建向量数据库

        Args:
            papers: 论文列表
            batch_size: 批处理大小
        """
        print("Building vector database...")

        # 提取abstracts
        abstracts = [paper["abstract"] for paper in papers]

        # 批量编码
        print("Encoding abstracts...")
        vectors = self.embedding_model.encode(abstracts, batch_size=batch_size)

        # 构建元数据
        metadata = [
            {
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "abstract": paper["abstract"],
                "year": paper["year"]
            }
            for paper in papers
        ]

        # 添加到向量数据库
        self.vector_db.add(vectors, metadata)
        print(f"Vector database built with {len(vectors)} papers")

    def retrieve_similar_papers(
        self,
        idea_text: str,
        top_k: int = 5
    ) -> List[Tuple[Dict, float]]:
        """
        检索与idea最相似的论文

        Args:
            idea_text: 生成的idea文本
            top_k: 返回最相似的K篇论文

        Returns:
            List of (paper_metadata, similarity_score) tuples
        """
        # 编码idea
        idea_vector = self.embedding_model.encode([idea_text], show_progress=False)[0]

        # 检索
        results = self.vector_db.search(idea_vector, top_k=top_k)

        return results

    def save_database(self, save_dir: str):
        """保存向量数据库"""
        self.vector_db.save(save_dir)

    def load_database(self, load_dir: str):
        """加载向量数据库"""
        self.vector_db.load(load_dir)


def main():
    """示例：构建向量数据库"""

    # 配置路径
    data_dir = Path("/home/lexy/下载/CLwithRAG/KGdemo/eval/Future_Idea_Prediction/data/iclr")
    jsonl_files = [
        str(data_dir / "iclr_2023_submitted.jsonl"),
        str(data_dir / "iclr2024_submissions.jsonl"),
        str(data_dir / "iclr2025_submissions.jsonl"),
    ]

    save_dir = "/home/lexy/下载/CLwithRAG/KGdemo/eval/Future_Idea_Prediction/data/vector_db"

    # 初始化
    embedding_model = EmbeddingModel()
    retrieval_system = PaperVectorRetrieval(embedding_model)

    # 加载论文
    papers = retrieval_system.load_papers_from_jsonl(jsonl_files)

    # 构建向量数据库
    retrieval_system.build_vector_database(papers)

    # 保存
    retrieval_system.save_database(save_dir)

    print("\nVector database construction completed!")

    # 测试检索
    print("\n" + "="*50)
    print("Testing retrieval with sample idea...")
    test_idea = "We propose a novel attention mechanism for transformer models that improves efficiency."
    results = retrieval_system.retrieve_similar_papers(test_idea, top_k=3)

    print(f"\nTop-3 similar papers for test idea:")
    for i, (paper, score) in enumerate(results, 1):
        print(f"\n{i}. Similarity: {score:.4f}")
        print(f"   Year: {paper['year']}")
        print(f"   Title: {paper['title']}")
        print(f"   Abstract: {paper['abstract'][:150]}...")


if __name__ == "__main__":
    main()
