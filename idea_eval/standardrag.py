
"""
Pairwise Win-Rate 评估 (含可视化绘图)
用于对比两个方法生成的创意，计算胜率并生成美观的趋势图。

使用场景:
- MyMethod (Ours) vs Naive LLM
- MyMethod (Ours) vs Standard RAG
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
from datetime import datetime
import re

# --- 绘图库导入 ---
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    # 尝试导入 seaborn 用于美化，如果没有也没关系
    try:
        import seaborn as sns
        HAS_SEABORN = True
    except ImportError:
        HAS_SEABORN = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# 添加项目根目录到路径 (假设 src 在上一级目录)
sys.path.insert(0, str(Path(__file__).parent.parent))

# 尝试导入 LLMConfig，如果不在路径中，提供一个Mock以便代码结构完整
try:
    from src.llm_config import LLMClient, LLMConfig
except ImportError:
    # 仅用于演示，如果找不到模块
    class LLMConfig:
        def __init__(self, **kwargs): pass
    class LLMClient:
        def __init__(self, config): pass
        def generate(self, **kwargs): return '{"winner": "A", "reasoning": "Mock"}'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PairwiseComparator:
    """
    成对比较评估器
    模拟导师决策场景："如果您是导师，您会建议学生尝试哪个Idea？"
    """

    def __init__(self, llm_config: LLMConfig):
        self.llm_client = LLMClient(llm_config)
        logger.info(f"✅ Pairwise比较器初始化成功")

    def compare_pair(
        self,
        idea_a: Dict,
        idea_b: Dict,
        method_a_name: str = "Method A",
        method_b_name: str = "Method B",
        randomize_order: bool = True
    ) -> Dict:
        """比较两个创意，返回胜者"""
        # 随机化顺序避免位置偏差
        if randomize_order and random.random() < 0.5:
            presented_first = 'B'
            presented_second = 'A'
            first_idea = idea_b
            second_idea = idea_a
        else:
            presented_first = 'A'
            presented_second = 'B'
            first_idea = idea_a
            second_idea = idea_b

        # 构建比较提示词
        prompt = self._build_comparison_prompt(
            first_idea, second_idea,
            presented_first, presented_second
        )

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self._get_system_prompt(),
                temperature=0.3,
                max_tokens=800
            )

            result = self._parse_comparison_response(
                response,
                presented_first,
                presented_second
            )

            # 转换回原始标签
            if result['winner'] == 'A':
                actual_winner = 'A' if presented_first == 'A' else 'B'
            elif result['winner'] == 'B':
                actual_winner = 'B' if presented_second == 'B' else 'A'
            else:
                actual_winner = 'TIE'

            result['winner'] = actual_winner
            result['method_a_name'] = method_a_name
            result['method_b_name'] = method_b_name

            return result

        except Exception as e:
            logger.error(f"比较创意时出错: {e}")
            return {
                'winner': 'ERROR',
                'reasoning': f"比较失败: {str(e)}",
                'confidence': None,
                'method_a_name': method_a_name,
                'method_b_name': method_b_name
            }

    def _get_system_prompt(self) -> str:
        return """You are an experienced research advisor evaluating research ideas.

Your role: Imagine you are a Ph.D. advisor. Two students present their research ideas to you.
Your task: Decide which idea you would recommend the student to pursue.

Evaluation criteria (in order of importance):
1. **Impact Potential**: Which idea could make a more significant contribution to the field?
2. **Feasibility**: Which idea is more likely to succeed given current resources and technology?
3. **Novelty**: Which idea is more innovative and less incremental?
4. **Clarity**: Which idea has a clearer research plan and stronger logical reasoning?

Decision guidelines:
- Be decisive: Choose one idea as the winner (or declare a tie only if truly equivalent)
- Provide your decision in JSON format:
{
  "winner": "A" | "B" | "TIE",
  "reasoning": "<explanation>",
  "confidence": "high" | "medium" | "low"
}"""

    def _build_comparison_prompt(self, idea_1: Dict, idea_2: Dict, label_1: str, label_2: str) -> str:
        return f"""Two students present their research ideas. Please decide which one you would recommend.

**Idea {label_1}:**
Title: {idea_1.get('title', 'N/A')}
Abstract: {idea_1.get('abstract', 'N/A')}
Modification: {idea_1.get('modification', 'N/A')}

---

**Idea {label_2}:**
Title: {idea_2.get('title', 'N/A')}
Abstract: {idea_2.get('abstract', 'N/A')}
Modification: {idea_2.get('modification', 'N/A')}

---

**Your Decision:**
Which idea (A or B) would you recommend? Provide JSON output."""

    def _parse_comparison_response(self, response: str, label_1: str, label_2: str) -> Dict:
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group(0))
                winner = result.get('winner', '').upper()
                if winner not in ['A', 'B', 'TIE']:
                    winner = 'TIE' # 简单回退
                result['winner'] = winner
                return result
            else:
                raise ValueError("No JSON found")
        except Exception:
            # 简单回退策略
            lower_resp = response.lower()
            if 'winner": "a"' in lower_resp or 'winner":"a"' in lower_resp: return {'winner': 'A'}
            if 'winner": "b"' in lower_resp or 'winner":"b"' in lower_resp: return {'winner': 'B'}
            return {'winner': 'TIE'}

    def batch_compare(
        self,
        ideas_a: List[Dict],
        ideas_b: List[Dict],
        method_a_name: str = "Method A",
        method_b_name: str = "Method B",
        max_pairs: int = None,
        num_rounds: int = 1
    ) -> Dict:
        logger.info(f"开始比较: {method_a_name} vs {method_b_name} ({num_rounds} 轮)")
      
        n_pairs = min(len(ideas_a), len(ideas_b))
        if max_pairs: n_pairs = min(n_pairs, max_pairs)

        all_round_results = []
        total_wins_a = 0
        total_wins_b = 0
        total_ties = 0

        for round_num in range(num_rounds):
            logger.info(f"--- Round {round_num + 1} ---")
            results = []
            wins_a = 0
            wins_b = 0
            ties = 0

            for i in range(n_pairs):
                comparison = self.compare_pair(
                    ideas_a[i], ideas_b[i], method_a_name, method_b_name
                )
                winner = comparison['winner']
                if winner == 'A': wins_a += 1
                elif winner == 'B': wins_b += 1
                elif winner == 'TIE': ties += 1
                results.append(comparison)
              
                # 简单日志
                if (i+1) % 5 == 0:
                    logger.info(f"Progress: {i+1}/{n_pairs}")

            total_wins_a += wins_a
            total_wins_b += wins_b
            total_ties += ties
          
            all_round_results.append({
                'round': round_num + 1,
                'comparisons': results,
                'wins_a': wins_a, 'wins_b': wins_b, 'ties': ties
            })

        total = n_pairs * num_rounds
        return {
            'method_a_name': method_a_name,
            'method_b_name': method_b_name,
            'total_comparisons': total,
            'wins_a': total_wins_a,
            'wins_b': total_wins_b,
            'ties': total_ties,
            'win_rate_a': (total_wins_a/total*100) if total else 0,
            'win_rate_b': (total_wins_b/total*100) if total else 0,
            'tie_rate': (total_ties/total*100) if total else 0,
            'round_results': all_round_results
        }


# -----------------------------------------------------------------------------
# 绘图功能 (核心修改部分)
# -----------------------------------------------------------------------------

def plot_cumulative_win_rate(results: Dict, output_file: str):
    """
    绘制累积胜率图 (美化版: Only Ours, No Red/Green, Big Fonts)
    """
    if not HAS_MATPLOTLIB:
        logger.warning("未安装matplotlib，跳过绘图")
        return

    logger.info(f"正在绘制累积胜率图: {output_file}")
  
    # 1. 数据准备
    all_comparisons = []
    if 'round_results' in results:
        for round_data in results['round_results']:
            all_comparisons.extend(round_data['comparisons'])
  
    if not all_comparisons:
        return

    x_indices = []
    y_win_rates = []
    cumulative_wins = 0
  
    # 计算 Ours (Method A) 的累积胜率
    for i, comp in enumerate(all_comparisons):
        count = i + 1
        if comp.get('winner') == 'A':
            cumulative_wins += 1
      
        # 计算胜率 (Win / Total * 100)
        win_rate = (cumulative_wins / count) * 100
        x_indices.append(count)
        y_win_rates.append(win_rate)

    # 2. 设置绘图风格
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid")
    else:
        plt.style.use('bmh')

    # 全局字体设置 (大字体)
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'font.family': 'sans-serif' # 确保兼容性
    })

    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # 3. 绘制线条 (Only Ours)
    # 颜色: Steel Blue / Deep Sky Blue (学术蓝，无红绿)
    LINE_COLOR = '#2077B4' 
  
    ax.plot(x_indices, y_win_rates, 
            color=LINE_COLOR, 
            linewidth=3, 
            marker='o', 
            markersize=6, 
            markevery=max(1, len(x_indices)//15), # 稀疏标记点
            label='Ours') # 强制命名为 Ours

    # 4. 辅助元素
    # 50% 基准线
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

    # 末尾数值标注 (例如: 65.0%)
    final_rate = y_win_rates[-1]
    ax.text(x_indices[-1] + (len(x_indices)*0.015), final_rate, 
            f'{final_rate:.1f}%', 
            color=LINE_COLOR, 
            fontweight='bold', 
            fontsize=18,
            va='center')

    # 5. 标签与装饰
    # 标题使用 "Ours vs Method_B_Name"
    method_b_clean = results['method_b_name']
    ax.set_title(f"Win-Rate Progression: Ours vs {method_b_clean}", pad=20, fontweight='bold')
    ax.set_xlabel("Number of Comparisons")
    ax.set_ylabel("Win Rate (%)")
  
    # 范围控制
    ax.set_ylim(0, 105)
    ax.set_xlim(0, len(x_indices) * 1.08) # 右侧留白给文字
  
    # 去除多余边框 (Spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
  
    # 图例
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)

    # 6. 保存
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    logger.info("✅ 图片生成完成")


# -----------------------------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------------------------

def load_ideas_from_file(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'ideas' in data: return data['ideas']
    return data


def read_paper_text(paper_path: str, max_length: int = None) -> str:
    """
    读取论文文本文件
    
    Args:
        paper_path: 论文文件路径
        max_length: 最大文本长度（用于截断，None表示不截断）
    
    Returns:
        论文文本内容
    """
    try:
        with open(paper_path, 'r', encoding='utf-8') as f:
            text = f.read()
        # 如果指定了最大长度，截断并添加提示
        if max_length and len(text) > max_length:
            text = text[:max_length] + "\n\n[...truncated due to length...]"
        return text
    except Exception as e:
        logger.error(f"读取论文文件失败 {paper_path}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50, max_chunks: int = 100) -> List[str]:
    """
    将文本切分成小块（优化版）
    
    Args:
        text: 输入文本
        chunk_size: 每个块的大小（字符数）
        chunk_overlap: 块之间的重叠大小（字符数）
        max_chunks: 最大chunk数量，避免过多chunks
    
    Returns:
        文本块列表
    """
    if not text:
        return []
    
    # 如果文本太长，先截断
    max_text_length = chunk_size * max_chunks
    if len(text) > max_text_length:
        text = text[:max_text_length]
        logger.warning(f"文本过长，已截断至 {max_text_length} 字符")
    
    chunks = []
    start = 0
    text_length = len(text)
    
    # 先按段落分割，更智能
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < 20:
            continue
        
        # 如果当前块加上新段落超过大小限制
        if current_chunk and len(current_chunk) + len(para) > chunk_size:
            chunks.append(current_chunk)
            # 如果段落本身很长，需要进一步切分
            if len(para) > chunk_size:
                # 按句子切分长段落
                sentences = re.split(r'[.!?]+\s+', para)
                temp_chunk = ""
                for sent in sentences:
                    if len(temp_chunk) + len(sent) > chunk_size and temp_chunk:
                        chunks.append(temp_chunk)
                        temp_chunk = sent
                    else:
                        temp_chunk = (temp_chunk + " " + sent).strip()
                current_chunk = temp_chunk
            else:
                current_chunk = para
        else:
            current_chunk = (current_chunk + "\n\n" + para).strip() if current_chunk else para
        
        # 限制chunks数量
        if len(chunks) >= max_chunks:
            break
    
    # 添加最后一个块
    if current_chunk and len(chunks) < max_chunks:
        chunks.append(current_chunk)
    
    # 过滤太短的块
    chunks = [chunk for chunk in chunks if len(chunk) > 50]
    
    logger.info(f"文本切分为 {len(chunks)} 个片段（最大限制: {max_chunks}）")
    return chunks


class VanillaRAGIdeaGenerator:
    """
    Vanilla RAG 创意生成器
    使用论文切片 -> 向量化 -> 检索 -> 生成的方式
    """
    
    def __init__(self, model_path: str, llm_client: LLMClient):
        """
        初始化RAG生成器
        
        Args:
            model_path: 本地embedding模型路径
            llm_client: LLM客户端
        """
        self.llm_client = llm_client
        self.embedding_model = None
        self.vector_db = None
        self._init_embedding_model(model_path)
        self._init_vector_db()
    
    def _init_embedding_model(self, model_path: str):
        """初始化embedding模型"""
        try:
            import os
            from sentence_transformers import SentenceTransformer
            
            # 设置环境变量，避免联网检查
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
            
            # 处理模型路径
            model_full_path = Path(model_path)
            if not model_full_path.is_absolute():
                # 如果是相对路径，尝试多个可能的位置
                current_file = Path(__file__)
                project_root = current_file.parent.parent.parent
                possible_paths = [
                    project_root / model_path,  # 相对于项目根目录
                    current_file.parent.parent / model_path,  # 相对于idea_eval目录
                    Path(model_path)  # 直接路径
                ]
                
                for path in possible_paths:
                    if path.exists() and (path / "modules.json").exists():
                        model_full_path = path
                        break
                else:
                    # 如果都找不到，尝试绝对路径
                    if not model_full_path.exists():
                        raise FileNotFoundError(f"模型路径不存在: {model_path}。尝试过的路径: {possible_paths}")
            elif not model_full_path.exists():
                raise FileNotFoundError(f"模型路径不存在: {model_path}")
            
            logger.info(f"加载embedding模型: {model_full_path}")
            self.embedding_model = SentenceTransformer(str(model_full_path), device='cpu')
            logger.info("✅ Embedding模型加载成功")
        except ImportError:
            raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")
        except Exception as e:
            logger.error(f"加载embedding模型失败: {e}")
            raise
    
    def _init_vector_db(self):
        """初始化向量数据库"""
        import numpy as np
        
        class SimpleVectorDB:
            def __init__(self):
                self.vectors = None
                self.metadata = []
            
            def add(self, vectors: np.ndarray, metadata: List[Dict]):
                if self.vectors is None:
                    self.vectors = vectors
                    self.metadata = metadata
                else:
                    self.vectors = np.vstack([self.vectors, vectors])
                    self.metadata.extend(metadata)
            
            def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
                if self.vectors is None or len(self.vectors) == 0:
                    return []
                
                # 计算余弦相似度
                query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
                vectors_norm = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8)
                similarities = np.dot(vectors_norm, query_norm)
                
                # 获取top_k索引
                top_k = min(top_k, len(similarities))
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                return [
                    (self.metadata[idx], float(similarities[idx]))
                    for idx in top_indices
                ]
        
        self.vector_db = SimpleVectorDB()
    
    def build_vector_database(self, paper_text: str, paper_id: str, chunk_size: int = 500, chunk_overlap: int = 50, max_chunks: int = 100):
        """
        构建论文的向量数据库（优化版）
        
        Args:
            paper_text: 论文文本
            paper_id: 论文ID
            chunk_size: 切片大小
            chunk_overlap: 切片重叠大小
            max_chunks: 最大chunk数量
        """
        import numpy as np
        
        # 1. 文本切片（优化：限制chunks数量）
        chunks = chunk_text(paper_text, chunk_size, chunk_overlap, max_chunks)
        if not chunks:
            logger.warning(f"论文 {paper_id} 切片为空")
            return
        
        logger.info(f"论文 {paper_id} 切分为 {len(chunks)} 个片段")
        
        # 2. 向量化（优化：使用更小的batch_size避免内存问题）
        try:
            embeddings = self.embedding_model.encode(
                chunks,
                batch_size=16,  # 减小batch size
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # 归一化以提高检索质量
            )
        except Exception as e:
            logger.error(f"论文 {paper_id} 向量化失败: {e}")
            return
        
        # 3. 构建元数据
        metadata = [
            {
                "paper_id": paper_id,
                "chunk_id": i,
                "text": chunk,
                "chunk_index": i
            }
            for i, chunk in enumerate(chunks)
        ]
        
        # 4. 添加到向量库
        self.vector_db.add(embeddings, metadata)
        logger.info(f"✅ 论文 {paper_id} 的向量库构建完成（{len(chunks)} 个片段）")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        检索相关文本片段
        
        Args:
            query: 查询文本
            top_k: 返回top-k个结果
        
        Returns:
            相关片段列表，每个包含text和score
        """
        import numpy as np
        
        # 向量化查询
        query_vector = self.embedding_model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True
        )[0]
        
        # 检索
        results = self.vector_db.search(query_vector, top_k=top_k)
        
        # 格式化结果
        retrieved_chunks = [
            {
                "text": meta["text"],
                "score": score,
                "paper_id": meta["paper_id"],
                "chunk_id": meta["chunk_id"]
            }
            for meta, score in results
        ]
        
        return retrieved_chunks
    
    def generate_idea(
        self,
        paper_id: str,
        retrieval_query: str = "What are the limitations, gaps, or future research directions?",
        top_k: int = 5
    ) -> Dict:
        """
        基于检索到的片段生成研究创意
        
        Args:
            paper_id: 论文ID
            retrieval_query: 检索查询
            top_k: 检索top-k个片段
        
        Returns:
            包含生成的创意的字典
        """
        # 1. 检索相关片段
        relevant_chunks = self.retrieve_relevant_chunks(retrieval_query, top_k=top_k)
        
        if not relevant_chunks:
            logger.warning(f"论文 {paper_id} 未检索到相关片段")
            return {"idea": "检索失败：未找到相关片段"}
        
        # 2. 组合检索到的片段
        retrieved_text = "\n\n".join([
            f"[片段 {i+1}, 相似度: {chunk['score']:.3f}]\n{chunk['text']}"
            for i, chunk in enumerate(relevant_chunks)
        ])
        
        logger.info(f"论文 {paper_id} 检索到 {len(relevant_chunks)} 个相关片段")
        
        # 3. 使用LLM生成创意
        system_prompt = """You are an expert research scientist with deep knowledge in academic research methodology and scientific innovation.

Your task is to read relevant text fragments from a research paper and generate a novel research idea that builds upon or extends the work. The idea should:
1. Be novel and innovative
2. Address a gap or limitation identified in the retrieved fragments
3. Be feasible and well-grounded
4. Clearly articulate the technical approach and expected outcomes

Focus on proposing a concrete research direction that could be pursued as a follow-up study."""

        user_prompt = f"""Please read the following relevant text fragments retrieved from a research paper and generate a novel research idea that builds upon or extends this work.

**Retrieved Text Fragments:**
{retrieved_text}

**Requirements:**
Generate a research idea (1-2 paragraphs) that:
- Identifies a specific gap, limitation, or unexplored direction based on the retrieved fragments
- Proposes a novel approach or extension to address this gap
- Explains the technical methodology and expected contributions
- Demonstrates how this idea advances the field beyond the current work

**Output Format:**
Please provide your response in the following JSON format:
{{
    "idea": "<your research idea in 1-2 paragraphs, including technical approach and expected outcomes>"
}}

Make sure the JSON is valid and properly formatted."""

        try:
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=2000
            )
            
            # 解析JSON响应
            try:
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    json_str = json_match.group(0)
                    idea_data = json.loads(json_str)
                    idea_text = idea_data.get("idea", response)
                    logger.info(f"✅ 论文 {paper_id} 的创意生成成功")
                    return {"idea": idea_text}
                else:
                    logger.warning(f"⚠️  论文 {paper_id} 无法解析JSON，使用原始响应")
                    return {"idea": response}
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️  论文 {paper_id} JSON解析失败: {e}，使用原始响应")
                return {"idea": response}
                
        except Exception as e:
            logger.error(f"❌ 论文 {paper_id} 生成创意时出错: {e}")
            return {"idea": f"生成失败: {str(e)}"}


def generate_idea_from_paper(
    paper_text: str,
    llm_client: LLMClient,
    paper_id: str = "unknown"
) -> Dict:
    """
    从论文文本生成研究创意
    
    Args:
        paper_text: 论文文本内容
        llm_client: LLM客户端
        paper_id: 论文ID（用于日志）
    
    Returns:
        包含生成的创意的字典: {"idea": "..."}
    """
    system_prompt = """You are an expert research scientist with deep knowledge in academic research methodology and scientific innovation.

Your task is to read a research paper and generate a novel research idea that builds upon or extends the work presented in the paper. The idea should:
1. Be novel and innovative
2. Address a gap or limitation in the current work
3. Be feasible and well-grounded
4. Clearly articulate the technical approach and expected outcomes

Focus on proposing a concrete research direction that could be pursued as a follow-up study."""

    user_prompt = f"""Please read the following research paper and generate a novel research idea that builds upon or extends this work.

**Paper Content:**
{paper_text}

**Requirements:**
Generate a research idea (1-2 paragraphs) that:
- Identifies a specific gap, limitation, or unexplored direction in the paper
- Proposes a novel approach or extension to address this gap
- Explains the technical methodology and expected contributions
- Demonstrates how this idea advances the field beyond the current paper

**Output Format:**
Please provide your response in the following JSON format:
{{
    "idea": "<your research idea in 1-2 paragraphs, including technical approach and expected outcomes>"
}}

Make sure the JSON is valid and properly formatted."""

    try:
        response = llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=2000
        )
        
        # 尝试解析JSON响应
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                idea_data = json.loads(json_str)
                idea_text = idea_data.get("idea", response)
                logger.info(f"✅ 论文 {paper_id} 的创意生成成功")
                return {"idea": idea_text}
            else:
                logger.warning(f"⚠️  论文 {paper_id} 无法解析JSON，使用原始响应")
                return {"idea": response}
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️  论文 {paper_id} JSON解析失败: {e}，使用原始响应")
            return {"idea": response}
            
    except Exception as e:
        logger.error(f"❌ 论文 {paper_id} 生成创意时出错: {e}")
        return {"idea": f"生成失败: {str(e)}"}


def generate_ideas_from_papers(
    papers_dir: str,
    output_file: str,
    config_path: str = 'config/config.yaml',
    max_papers: int = None,
    model_path: str = 'model/sentence-transformers/all-MiniLM-L6-v2',
    use_rag: bool = True,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    top_k: int = 5,
    max_chunks: int = 80
) -> None:
    """
    从论文目录读取所有论文并生成创意（使用Vanilla RAG）
    
    Args:
        papers_dir: 论文目录路径
        output_file: 输出JSON文件路径
        config_path: LLM配置文件路径
        max_papers: 最大处理论文数量（None表示处理所有）
        model_path: 本地embedding模型路径
        use_rag: 是否使用RAG（True使用RAG，False使用全文）
        chunk_size: 文本切片大小（字符数）
        chunk_overlap: 切片重叠大小（字符数）
        top_k: 检索top-k个片段
    """
    import yaml
    
    logger.info("=" * 80)
    logger.info("🚀 开始从论文生成创意")
    logger.info(f"模式: {'Vanilla RAG' if use_rag else '全文生成'}")
    logger.info("=" * 80)
    logger.info(f"论文目录: {papers_dir}")
    logger.info(f"输出文件: {output_file}")
    
    # 1. 加载LLM配置
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        llm_dict = config.get('llm', {})
        llm_config = LLMConfig(
            provider=llm_dict.get('provider', 'openai'),
            model=llm_dict.get('model', 'gpt-4o'),
            api_key=llm_dict.get('api_key'),
            base_url=llm_dict.get('base_url'),
            temperature=0.7
        )
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        sys.exit(1)
    
    if not llm_config.api_key:
        import os
        llm_config.api_key = os.getenv('OPENAI_API_KEY')
    
    if not llm_config.api_key:
        logger.error("未找到API密钥，请确保在配置文件中设置或设置环境变量 OPENAI_API_KEY")
        sys.exit(1)
    
    # 2. 初始化LLM客户端
    llm_client = LLMClient(llm_config)
    logger.info(f"✅ LLM客户端初始化成功，模型: {llm_config.model}")
    
    # 3. 初始化RAG生成器（如果使用RAG）
    rag_generator = None
    if use_rag:
        try:
            rag_generator = VanillaRAGIdeaGenerator(model_path, llm_client)
            logger.info(f"✅ RAG生成器初始化成功，模型: {model_path}")
        except Exception as e:
            logger.error(f"RAG生成器初始化失败: {e}")
            logger.warning("降级到全文生成模式")
            use_rag = False
    
    # 4. 读取所有论文文件
    papers_path = Path(papers_dir)
    if not papers_path.exists():
        logger.error(f"论文目录不存在: {papers_dir}")
        sys.exit(1)
    
    paper_files = sorted(papers_path.glob("*.txt"))
    if not paper_files:
        logger.error(f"在目录 {papers_dir} 中未找到 .txt 文件")
        sys.exit(1)
    
    if max_papers:
        paper_files = paper_files[:max_papers]
    
    logger.info(f"找到 {len(paper_files)} 篇论文")
    
    # 5. 为每篇论文生成创意
    ideas = []
    for i, paper_file in enumerate(paper_files, 1):
        paper_id = paper_file.stem
        logger.info(f"\n处理论文 {i}/{len(paper_files)}: {paper_id}")
        
        # 读取论文文本
        paper_text = read_paper_text(str(paper_file))
        if not paper_text:
            logger.warning(f"⚠️  跳过论文 {paper_id}（无法读取）")
            continue
        
        # 生成创意
        try:
            if use_rag and rag_generator:
                # 使用RAG模式
                # 为每篇论文重新构建向量库（单论文模式）
                rag_generator._init_vector_db()  # 重置向量库
                
                # 限制文本长度，避免内存问题
                max_text_length = 50000  # 限制为50k字符
                if len(paper_text) > max_text_length:
                    logger.info(f"论文 {paper_id} 文本过长 ({len(paper_text)} 字符)，截断至 {max_text_length} 字符")
                    paper_text = paper_text[:max_text_length]
                
                rag_generator.build_vector_database(paper_text, paper_id, chunk_size, chunk_overlap, max_chunks=max_chunks)
                
                # 使用检索查询生成创意
                retrieval_query = "What are the limitations, gaps, future work, or unexplored research directions?"
                idea_dict = rag_generator.generate_idea(paper_id, retrieval_query, top_k)
            else:
                # 使用全文模式（原有方法）
                # 限制文本长度
                max_text_length = 15000
                if len(paper_text) > max_text_length:
                    paper_text = paper_text[:max_text_length] + "\n\n[...truncated...]"
                idea_dict = generate_idea_from_paper(paper_text, llm_client, paper_id)
        except Exception as e:
            logger.error(f"❌ 论文 {paper_id} 处理失败: {e}")
            idea_dict = {"idea": f"处理失败: {str(e)}"}
        
        ideas.append(idea_dict)
        
        # 进度日志
        if i % 5 == 0:
            logger.info(f"进度: {i}/{len(paper_files)}")
    
    # 6. 保存结果（格式与 naive_llm.json 一致）
    result = {
        "ideas": ideas
    }
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n✅ 完成! 生成了 {len(ideas)} 个创意")
    logger.info(f"💾 结果已保存: {output_path}")

def save_comparison_results(results: Dict, output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def generate_comparison_report(results: Dict, output_file: str):
    method_a = "Ours" # 报告中也强调 Ours
    method_b = results['method_b_name']
    win_rate_a = results['win_rate_a']
  
    lines = [
        f"# Win-Rate 对比报告",
        f"**对比**: {method_a} vs {method_b}",
        f"**Ours 胜率**: {win_rate_a:.1f}%",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 统计",
        f"- Ours 胜: {results['wins_a']}",
        f"- {method_b} 胜: {results['wins_b']}",
        f"- 平局: {results['ties']}",
    ]
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description='Pairwise Win-Rate 评估 或 从论文生成创意',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 模式1: 从论文生成创意
  python test.py --generate-from-papers \\
      --papers-dir /path/to/papers_txt \\
      --output output/ideas.json \\
      --config config/config.yaml \\
      --max-papers 10

  # 模式2: Pairwise比较评估
  python test.py \\
      --method-a ideas_a.json \\
      --method-b ideas_b.json \\
      --output comparison_results.json \\
      --config config/config.yaml
        """
    )
    
    # 生成模式参数
    parser.add_argument('--generate-from-papers', action='store_true',
                       help='从论文目录生成创意（生成模式）')
    parser.add_argument('--papers-dir', type=str,
                       help='论文目录路径（生成模式必需）')
    parser.add_argument('--max-papers', type=int,
                       help='最大处理论文数量（可选）')
    parser.add_argument('--model-path', type=str,
                       default='model/sentence-transformers/all-MiniLM-L6-v2',
                       help='本地embedding模型路径（默认: model/sentence-transformers/all-MiniLM-L6-v2）')
    parser.add_argument('--use-rag', action='store_true', default=True,
                       help='使用Vanilla RAG模式（默认启用）')
    parser.add_argument('--no-rag', dest='use_rag', action='store_false',
                       help='禁用RAG，使用全文生成模式')
    parser.add_argument('--chunk-size', type=int, default=800,
                       help='文本切片大小（字符数，默认: 800）')
    parser.add_argument('--chunk-overlap', type=int, default=100,
                       help='切片重叠大小（字符数，默认: 100）')
    parser.add_argument('--top-k', type=int, default=5,
                       help='检索top-k个片段（默认: 5）')
    parser.add_argument('--max-chunks', type=int, default=80,
                       help='每篇论文最大chunk数量（默认: 80）')
    
    # 比较模式参数
    parser.add_argument('--method-a', type=str, help='Ours的方法文件（比较模式必需）')
    parser.add_argument('--method-b', type=str, help='对比方法文件（比较模式必需）')
    parser.add_argument('--name-a', type=str, default='Ours', help='方法A名称 (默认 Ours)')
    parser.add_argument('--name-b', type=str, default='Baseline', help='方法B名称')
    parser.add_argument('--max-pairs', type=int, help='最大对数')
    parser.add_argument('--num-rounds', type=int, default=1, help='轮数')
    
    # 通用参数
    parser.add_argument('--output', type=str, required=True, help='输出JSON路径')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='LLM配置')

    args = parser.parse_args()

    # 判断运行模式
    if args.generate_from_papers:
        # 生成模式：从论文生成创意
        if not args.papers_dir:
            logger.error("生成模式需要指定 --papers-dir")
            sys.exit(1)
        
        generate_ideas_from_papers(
            papers_dir=args.papers_dir,
            output_file=args.output,
            config_path=args.config,
            max_papers=args.max_papers,
            model_path=args.model_path,
            use_rag=args.use_rag,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            max_chunks=args.max_chunks
        )
    else:
        # 比较模式：Pairwise Win-Rate 评估
        if not args.method_a or not args.method_b:
            logger.error("比较模式需要指定 --method-a 和 --method-b")
            sys.exit(1)
        
        # 1. 检查文件
        for f in [args.method_a, args.method_b]:
            if not Path(f).exists():
                logger.error(f"文件不存在: {f}")
                sys.exit(1)

        # 2. 配置 LLM
        try:
            with open(args.config, 'r') as f: config = yaml.safe_load(f)
            llm_dict = config.get('llm', {})
            llm_config = LLMConfig(
                provider=llm_dict.get('provider', 'openai'),
                model=llm_dict.get('model', 'gpt-4o'),
                api_key=llm_dict.get('api_key'),
                base_url=llm_dict.get('base_url'),
                temperature=0.3
            )
        except Exception as e:
            logger.warning(f"配置加载失败 ({e})，使用默认配置")
            llm_config = LLMConfig()

        # 3. 运行比较
        comparator = PairwiseComparator(llm_config)
        ideas_a = load_ideas_from_file(args.method_a)
        ideas_b = load_ideas_from_file(args.method_b)

        results = comparator.batch_compare(
            ideas_a, ideas_b, 
            # 强制将方法A在内部逻辑中也视为 Ours，或者在绘图时处理
            method_a_name=args.name_a, 
            method_b_name=args.name_b,
            max_pairs=args.max_pairs,
            num_rounds=args.num_rounds
        )

        # 4. 保存与报告
        save_comparison_results(results, args.output)
      
        report_file = args.output.replace('.json', '_report.md')
        generate_comparison_report(results, report_file)

        # 5. 生成美观图片 (新增)
        img_file = args.output.replace('.json', '.png')
        plot_cumulative_win_rate(results, img_file)

        logger.info(f"🎉 完成! 胜率: {results['win_rate_a']:.1f}%")

if __name__ == "__main__":
    main()
