"""
显示评估结果中的Top N最佳匹配
包含完整的idea内容和匹配论文详情
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict


def load_ideas_with_content(ideas_file: str) -> Dict[str, Dict]:
    """加载ideas并建立id到内容的映射"""
    ideas_path = Path(ideas_file)

    if not ideas_path.exists():
        # 尝试从父目录查找
        parent_path = Path(__file__).parent.parent / ideas_file
        if parent_path.exists():
            ideas_path = parent_path

    ideas_map = {}

    with open(ideas_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

        if isinstance(data, dict) and "ideas" in data:
            ideas_list = data["ideas"]
        elif isinstance(data, list):
            ideas_list = data
        else:
            raise ValueError("Unexpected JSON structure")

        for i, idea in enumerate(ideas_list, 1):
            idea_id = f"idea_{i:03d}"
            ideas_map[idea_id] = idea

    return ideas_map


def show_top_matches(
    evaluation_results_file: str,
    ideas_file: str,
    top_n: int = 10,
    show_full_reason: bool = False
):
    """显示Top N最佳匹配"""

    # 加载评估结果
    with open(evaluation_results_file, 'r', encoding='utf-8') as f:
        eval_results = json.load(f)

    # 加载ideas内容
    ideas_map = load_ideas_with_content(ideas_file)

    # 按分数排序
    sorted_results = sorted(
        eval_results,
        key=lambda x: x.get('max_score', 0),
        reverse=True
    )

    print("\n" + "="*100)
    print(f"TOP {top_n} BEST MATCHING IDEAS AND PAPERS".center(100))
    print("="*100)

    for rank, result in enumerate(sorted_results[:top_n], 1):
        idea_id = result['idea_id']
        score = result['max_score']

        print(f"\n{'='*100}")
        print(f"RANK #{rank} - Idea: {idea_id} - Match Score: {score:.1f}/10")
        print('='*100)

        # 显示idea内容
        if idea_id in ideas_map:
            idea = ideas_map[idea_id]
            print(f"\n📝 IDEA TITLE:")
            print(f"   {idea.get('title', 'N/A')}")
            print(f"\n📝 IDEA ABSTRACT:")
            abstract = idea.get('abstract', 'N/A')
            # 换行显示，每行最多100字符
            for i in range(0, len(abstract), 100):
                print(f"   {abstract[i:i+100]}")

        # 显示匹配的论文
        if 'best_match' in result:
            best_match = result['best_match']
            print(f"\n📄 MATCHED PAPER:")
            print(f"   Title: {best_match.get('paper_title', 'N/A')}")
            print(f"   Year:  {best_match.get('paper_year', 'N/A')}")

            print(f"\n💡 EVALUATION REASON:")
            reason = best_match.get('reason', 'N/A')
            if show_full_reason:
                # 完整显示
                for i in range(0, len(reason), 100):
                    print(f"   {reason[i:i+100]}")
            else:
                # 截断显示
                if len(reason) > 300:
                    reason = reason[:297] + "..."
                for i in range(0, len(reason), 100):
                    print(f"   {reason[i:i+100]}")

    print("\n" + "="*100)
    print(f"Total ideas evaluated: {len(eval_results)}")
    print(f"Average max score: {sum(r.get('max_score', 0) for r in eval_results) / len(eval_results):.2f}/10")
    print("="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="显示评估结果中的Top N最佳匹配"
    )

    parser.add_argument(
        "--evaluation_results",
        type=str,
        default="../results/evaluation_results.json",
        help="评估结果文件路径"
    )

    parser.add_argument(
        "--ideas_file",
        type=str,
        default="../data/research_ideas_Natural_Language_Processing_20251210_161113.json",
        help="Ideas文件路径"
    )

    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="显示Top N个最佳匹配"
    )

    parser.add_argument(
        "--show_full_reason",
        action="store_true",
        help="显示完整的评估理由（不截断）"
    )

    args = parser.parse_args()

    show_top_matches(
        args.evaluation_results,
        args.ideas_file,
        args.top_n,
        args.show_full_reason
    )


if __name__ == "__main__":
    main()
