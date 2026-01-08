#!/usr/bin/env python3
"""
Evaluation Metrics for Deep Paper Information Extraction
评估指标实现

Metrics:
1. ROUGE (ROUGE-1, ROUGE-2, ROUGE-L) - 文本重叠度
2. BERTScore - 语义相似度
3. LLM-based Semantic Similarity - 使用LLM评估语义相似度
4. Coverage - 信息覆盖率
5. Precision - 信息准确率
"""

import re
from typing import Dict, List, Tuple
import numpy as np


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Remove punctuation (keep only alphanumeric and spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Remove extra spaces again
    text = ' '.join(text.split())

    return text


def compute_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    """
    Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)

    Args:
        prediction: Predicted text
        reference: Reference (gold) text

    Returns:
        Dict with ROUGE scores
    """
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, prediction)

        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge1_p': scores['rouge1'].precision,
            'rouge1_r': scores['rouge1'].recall,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rouge2_p': scores['rouge2'].precision,
            'rouge2_r': scores['rouge2'].recall,
            'rougeL_f': scores['rougeL'].fmeasure,
            'rougeL_p': scores['rougeL'].precision,
            'rougeL_r': scores['rougeL'].recall,
        }

    except ImportError:
        print("⚠️  rouge-score not installed. Install with: pip install rouge-score")
        return {
            'rouge1_f': 0.0,
            'rouge1_p': 0.0,
            'rouge1_r': 0.0,
            'rouge2_f': 0.0,
            'rouge2_p': 0.0,
            'rouge2_r': 0.0,
            'rougeL_f': 0.0,
            'rougeL_p': 0.0,
            'rougeL_r': 0.0,
        }
    except Exception as e:
        print(f"⚠️  Error computing ROUGE: {e}")
        return {
            'rouge1_f': 0.0,
            'rouge1_p': 0.0,
            'rouge1_r': 0.0,
            'rouge2_f': 0.0,
            'rouge2_p': 0.0,
            'rouge2_r': 0.0,
            'rougeL_f': 0.0,
            'rougeL_p': 0.0,
            'rougeL_r': 0.0,
        }


def compute_bertscore(predictions: List[str], references: List[str], model_path: str = None) -> Dict[str, List[float]]:
    """
    Compute BERTScore for a list of predictions

    Args:
        predictions: List of predicted texts
        references: List of reference texts
        model_path: Optional path to local model cache directory

    Returns:
        Dict with precision, recall, f1 lists
    """
    try:
        import os
        import warnings

        # Suppress the misleading SentencePiece warning from transformers
        warnings.filterwarnings('ignore', message='.*SentencePiece.*')

        # Determine full model path
        if model_path:
            # Check if it's a directory path or cache path
            if os.path.isdir(os.path.join(model_path, 'microsoft', 'deberta-v2-xlarge-mnli')):
                # It's a cache directory structure
                full_model_path = os.path.join(model_path, 'microsoft', 'deberta-v2-xlarge-mnli')
            elif os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, 'config.json')):
                # It's a direct model directory
                full_model_path = model_path
            else:
                print(f"  Warning: Model path {model_path} not found, falling back to default")
                full_model_path = None
        else:
            full_model_path = None

        # Use direct model loading with transformers
        if full_model_path:
            print(f"  Loading local BERTScore model from: {full_model_path}")
            from transformers import AutoTokenizer, AutoModel
            import torch
            import warnings

            # Disable HuggingFace Hub access completely
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'

            # Suppress regex pattern warning for DeBERTa models
            warnings.filterwarnings('ignore', message='.*incorrect regex pattern.*')
            warnings.filterwarnings('ignore', message='.*fix_mistral_regex.*')

            # Load model and tokenizer with local_files_only
            # Create a custom workaround for the vocab_file.endswith bug
            print(f"  Loading tokenizer from: {full_model_path}")

            # Patch the vocab_file attribute to avoid the None.endswith error
            # This is a workaround for a bug in transformers library
            try:
                from transformers import DebertaV2TokenizerFast
                import os as _os

                # Create a minimal mock for vocab_file to avoid AttributeError
                class TokenizerConfig:
                    def __init__(self, model_path):
                        self.vocab_file = _os.path.join(model_path, 'spm.model')

                # Try loading with fast tokenizer and explicit config
                tokenizer = DebertaV2TokenizerFast.from_pretrained(
                    full_model_path,
                    local_files_only=True
                )
                print(f"  ✓ Loaded DebertaV2TokenizerFast")

            except Exception as tok_err:
                print(f"  Warning: Fast tokenizer failed ({tok_err})")
                print(f"  Trying alternative approach with manual SentencePiece...")

                try:
                    # Alternative: Use SentencePiece directly with custom wrapper
                    import sentencepiece as spm
                    import os as _os

                    vocab_file = _os.path.join(full_model_path, 'spm.model')
                    sp = spm.SentencePieceProcessor()
                    sp.Load(vocab_file)

                    # Create a minimal tokenizer wrapper
                    class SPTokenizer:
                        def __init__(self, sp_model):
                            self.sp = sp_model

                        def __call__(self, texts, padding=True, truncation=True,
                                   max_length=512, return_tensors='pt'):
                            if isinstance(texts, str):
                                texts = [texts]

                            # Tokenize with SentencePiece
                            encoded = [self.sp.EncodeAsIds(text) for text in texts]

                            # Truncate
                            if truncation:
                                encoded = [ids[:max_length] for ids in encoded]

                            # Pad
                            if padding:
                                max_len = max(len(ids) for ids in encoded)
                                encoded = [ids + [0] * (max_len - len(ids)) for ids in encoded]

                            # Create attention mask
                            import torch
                            input_ids = torch.tensor(encoded, dtype=torch.long)
                            attention_mask = (input_ids != 0).long()

                            return {
                                'input_ids': input_ids,
                                'attention_mask': attention_mask
                            }

                    tokenizer = SPTokenizer(sp)
                    print(f"  ✓ Loaded custom SentencePiece tokenizer")

                except Exception as sp_err:
                    print(f"  Error: All tokenizer loading methods failed: {sp_err}")
                    raise RuntimeError(f"Cannot load tokenizer for BERTScore: {sp_err}")
            model = AutoModel.from_pretrained(
                full_model_path,
                local_files_only=True,
                trust_remote_code=True
            )

            # Move to device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
            model.eval()

            print(f"  Model loaded on {device}")
            print(f"  Computing BERTScore...")

            # Compute BERTScore manually using the loaded model
            # This avoids bert_score library's online model loading
            import torch.nn.functional as F

            all_preds_embeddings = []
            all_refs_embeddings = []

            # Process in batches to avoid OOM
            batch_size = 8

            for i in range(0, len(predictions), batch_size):
                batch_preds = predictions[i:i+batch_size]
                batch_refs = references[i:i+batch_size]

                # Tokenize batch
                pred_inputs = tokenizer(batch_preds, padding=True, truncation=True,
                                       max_length=512, return_tensors='pt').to(device)
                ref_inputs = tokenizer(batch_refs, padding=True, truncation=True,
                                      max_length=512, return_tensors='pt').to(device)

                with torch.no_grad():
                    # Get embeddings
                    pred_outputs = model(**pred_inputs)
                    ref_outputs = model(**ref_inputs)

                    # Use last hidden state
                    pred_emb = pred_outputs.last_hidden_state  # [batch, seq_len, hidden]
                    ref_emb = ref_outputs.last_hidden_state

                    all_preds_embeddings.append((pred_emb.cpu(), pred_inputs['attention_mask'].cpu()))
                    all_refs_embeddings.append((ref_emb.cpu(), ref_inputs['attention_mask'].cpu()))

            # Compute cosine similarity for each pair
            precision_list = []
            recall_list = []
            f1_list = []

            idx = 0
            for batch_idx in range(len(all_preds_embeddings)):
                pred_emb, pred_mask = all_preds_embeddings[batch_idx]
                ref_emb, ref_mask = all_refs_embeddings[batch_idx]

                batch_size_actual = pred_emb.size(0)

                for b in range(batch_size_actual):
                    # Get valid tokens (excluding padding)
                    pred_valid = pred_emb[b][pred_mask[b].bool()]  # [pred_len, hidden]
                    ref_valid = ref_emb[b][ref_mask[b].bool()]      # [ref_len, hidden]

                    # Normalize embeddings
                    pred_norm = F.normalize(pred_valid, p=2, dim=1)
                    ref_norm = F.normalize(ref_valid, p=2, dim=1)

                    # Compute cosine similarity matrix [pred_len, ref_len]
                    sim_matrix = torch.mm(pred_norm, ref_norm.t())

                    # Precision: for each pred token, max similarity with any ref token
                    precision = sim_matrix.max(dim=1)[0].mean().item()

                    # Recall: for each ref token, max similarity with any pred token
                    recall = sim_matrix.max(dim=0)[0].mean().item()

                    # F1 score
                    if precision + recall > 0:
                        f1 = 2 * precision * recall / (precision + recall)
                    else:
                        f1 = 0.0

                    precision_list.append(precision)
                    recall_list.append(recall)
                    f1_list.append(f1)

                    idx += 1

            return {
                'precision': precision_list,
                'recall': recall_list,
                'f1': f1_list
            }
        else:
            # Use default online mode
            from bert_score import score
            P, R, F1 = score(predictions, references, lang='en', verbose=False)

            return {
                'precision': P.tolist(),
                'recall': R.tolist(),
                'f1': F1.tolist()
            }

    except ImportError as e:
        print(f"⚠️  Required package not installed: {e}")
        print("   Install with: pip install bert-score transformers torch")
        return {
            'precision': [0.0] * len(predictions),
            'recall': [0.0] * len(predictions),
            'f1': [0.0] * len(predictions)
        }
    except Exception as e:
        print(f"⚠️  Error computing BERTScore: {e}")
        import traceback
        traceback.print_exc()
        return {
            'precision': [0.0] * len(predictions),
            'recall': [0.0] * len(predictions),
            'f1': [0.0] * len(predictions)
        }


def compute_llm_similarity(
    prediction: str,
    reference: str,
    model: str = "gpt-4",
    llm_config: Dict = None
) -> Dict[str, float]:
    """
    Use LLM to evaluate semantic similarity

    Args:
        prediction: Predicted text
        reference: Reference text
        model: LLM model to use (default: "gpt-4")
        llm_config: Dict with LLM configuration (api_key, base_url, etc.)

    Returns:
        Dict with similarity score and explanation
    """
    try:
        import os
        import json

        # Support both old and new OpenAI SDK
        try:
            from openai import OpenAI
            use_new_api = True
        except ImportError:
            import openai
            use_new_api = False

        # Load configuration from multiple sources (priority order):
        # 1. llm_config parameter (highest priority)
        # 2. Environment variables
        # 3. Project config.yaml file
        api_key = None
        base_url = None

        if llm_config:
            api_key = llm_config.get('api_key')
            base_url = llm_config.get('base_url')
            model = llm_config.get('model', model)

        if not api_key:
            api_key = os.environ.get('OPENAI_API_KEY')

        if not base_url:
            base_url = os.environ.get('OPENAI_BASE_URL')

        # If still no config, try loading from project config.yaml
        if not api_key:
            try:
                import yaml
                from pathlib import Path

                # Try to find config.yaml (assume it's in project root /config/)
                current_file = Path(__file__)
                project_root = current_file.parent.parent.parent.parent  # Go up from eval/deeppaper_eval/evaluator/metrics.py
                config_path = project_root / "config" / "config.yaml"

                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)

                    llm_cfg = config.get('llm', {})
                    if llm_cfg.get('enabled', False):
                        api_key = llm_cfg.get('api_key')
                        base_url = base_url or llm_cfg.get('base_url')
                        model = llm_cfg.get('model', model)
            except Exception as e:
                pass  # Silently continue if config loading fails

        # Default base_url if not set
        if not base_url:
            base_url = 'https://api.openai.com/v1'

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found. Please set it in config.yaml, environment variable, or pass via llm_config parameter")

        prompt = f"""You are evaluating the semantic similarity between two texts about research paper analysis.

Reference (Human annotation):
{reference}

Prediction (Model output):
{prediction}

Please evaluate:
1. Semantic similarity (0-100): How similar is the meaning?
2. Information coverage (0-100): How much of the reference information is covered?
3. Accuracy (0-100): Is the prediction factually correct (no hallucinations)?

Respond in JSON format:
{{
    "similarity_score": <0-100>,
    "coverage_score": <0-100>,
    "accuracy_score": <0-100>,
    "explanation": "Brief explanation"
}}"""

        if use_new_api:
            # New API (openai >= 1.0.0)
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for research paper analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=300
            )
            result_text = response.choices[0].message.content.strip()
        else:
            # Old API (openai < 1.0.0)
            openai.api_key = api_key
            openai.api_base = base_url
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for research paper analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=300
            )
            result_text = response.choices[0].message.content.strip()

        # Parse JSON
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]

        result_text = result_text.strip()

        # Try to parse JSON with better error handling
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError as json_err:
            # If JSON parsing fails, try to extract scores manually
            print(f"⚠️  JSON decode error: {json_err}")
            print(f"   Attempting to extract scores from response...")

            # Try to extract scores using regex
            import re
            similarity_match = re.search(r'"similarity_score"\s*:\s*(\d+(?:\.\d+)?)', result_text)
            coverage_match = re.search(r'"coverage_score"\s*:\s*(\d+(?:\.\d+)?)', result_text)
            accuracy_match = re.search(r'"accuracy_score"\s*:\s*(\d+(?:\.\d+)?)', result_text)

            result = {
                'similarity_score': float(similarity_match.group(1)) if similarity_match else 50.0,
                'coverage_score': float(coverage_match.group(1)) if coverage_match else 50.0,
                'accuracy_score': float(accuracy_match.group(1)) if accuracy_match else 50.0,
                'explanation': 'Parsed from malformed JSON'
            }

        return {
            'similarity': result.get('similarity_score', 0) / 100.0,
            'coverage': result.get('coverage_score', 0) / 100.0,
            'accuracy': result.get('accuracy_score', 0) / 100.0,
            'explanation': result.get('explanation', '')
        }

    except Exception as e:
        print(f"⚠️  Error computing LLM similarity: {e}")
        return {
            'similarity': 0.0,
            'coverage': 0.0,
            'accuracy': 0.0,
            'explanation': str(e)
        }


def compute_bleu_score(prediction: str, reference: str) -> float:
    """
    Compute BLEU score

    Args:
        prediction: Predicted text
        reference: Reference text

    Returns:
        BLEU score
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        # Tokenize
        ref_tokens = reference.split()
        pred_tokens = prediction.split()

        # Compute BLEU with smoothing
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)

        return score

    except ImportError:
        print("⚠️  nltk not installed. Install with: pip install nltk")
        return 0.0
    except Exception as e:
        print(f"⚠️  Error computing BLEU: {e}")
        return 0.0


def compute_all_metrics(
    prediction: str,
    reference: str,
    use_llm: bool = False,
    llm_model: str = "gpt-4",
    llm_config: Dict = None
) -> Dict[str, float]:
    """
    Compute all evaluation metrics

    Args:
        prediction: Predicted text
        reference: Reference text
        use_llm: Whether to use LLM-based evaluation
        llm_model: LLM model for evaluation
        llm_config: Dict with LLM configuration (api_key, base_url, etc.)

    Returns:
        Dict with all metrics
    """
    metrics = {}

    # ROUGE scores
    rouge = compute_rouge_scores(prediction, reference)
    metrics.update(rouge)

    # BLEU score
    bleu = compute_bleu_score(prediction, reference)
    metrics['bleu'] = bleu

    # LLM-based evaluation (optional, more expensive)
    if use_llm:
        llm_scores = compute_llm_similarity(prediction, reference, llm_model, llm_config)
        metrics['llm_similarity'] = llm_scores['similarity']
        metrics['llm_coverage'] = llm_scores['coverage']
        metrics['llm_accuracy'] = llm_scores['accuracy']

    return metrics


def aggregate_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple samples

    Args:
        all_metrics: List of metric dicts

    Returns:
        Dict with aggregated (mean) metrics
    """
    if not all_metrics:
        return {}

    # Get all metric names
    metric_names = all_metrics[0].keys()

    aggregated = {}
    for name in metric_names:
        values = [m[name] for m in all_metrics if name in m and isinstance(m[name], (int, float))]
        if values:
            aggregated[f"{name}_mean"] = np.mean(values)
            aggregated[f"{name}_std"] = np.std(values)
            aggregated[f"{name}_median"] = np.median(values)

    return aggregated


if __name__ == '__main__':
    # Test metrics
    ref = "This paper addresses the problem of efficient transformer models for long sequences."
    pred = "The paper tackles efficient transformers for processing long text sequences."

    print("Testing metrics...")
    print("\nReference:", ref)
    print("Prediction:", pred)

    metrics = compute_all_metrics(pred, ref, use_llm=False)

    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
