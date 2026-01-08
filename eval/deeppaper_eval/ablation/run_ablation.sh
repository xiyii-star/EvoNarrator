#!/bin/bash
# 消融实验运行脚本

set -e

echo "=========================================="
echo "DeepPaper 2.0 消融实验运行脚本"
echo "=========================================="

# 默认配置
PAPERS_DIR="../data/papers_txt"  # 默认使用TXT文件（更快）
OUTPUT_DIR="./results/ablation"
CONFIG_PATH="../../../config/config.yaml"
GROBID_URL=""  # TXT文件不需要GROBID
LIMIT=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --papers_dir)
            PAPERS_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --grobid_url)
            GROBID_URL="$2"
            shift 2
            ;;
        --limit)
            LIMIT="--limit $2"
            shift 2
            ;;
        --test)
            LIMIT="--limit 3"
            echo "测试模式: 只处理前3篇论文"
            shift
            ;;
        --no-grobid)
            GROBID_URL=""
            echo "禁用GROBID: 将使用PyPDF2解析"
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --papers_dir DIR     论文文件目录 (支持PDF或TXT, 默认: ../data/papers_txt)"
            echo "  --output_dir DIR     输出目录 (默认: ./results/ablation)"
            echo "  --config PATH        LLM配置文件 (默认: ../../../config/config.yaml)"
            echo "  --grobid_url URL     GROBID服务URL (仅PDF需要, 例如: http://localhost:8070)"
            echo "  --limit N            限制处理论文数量"
            echo "  --test               测试模式 (只处理前3篇)"
            echo "  --no-grobid          不使用GROBID (对TXT文件无效)"
            echo "  -h, --help           显示帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 --test                            # 测试模式 (TXT文件)"
            echo "  $0 --papers_dir /path/to/papers_txt  # 使用TXT文件"
            echo "  $0 --papers_dir /path/to/papers_pdf --grobid_url http://localhost:8070  # 使用PDF"
            echo "  $0 --limit 5                         # 只处理5篇论文"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo ""
echo "配置信息:"
echo "  Papers目录: $PAPERS_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  配置文件: $CONFIG_PATH"
if [ -n "$GROBID_URL" ]; then
    echo "  GROBID URL: $GROBID_URL"
else
    echo "  GROBID: 禁用"
fi
echo "=========================================="
echo ""

# 检查papers目录是否存在
if [ ! -d "$PAPERS_DIR" ]; then
    echo "❌ 错误: Papers目录不存在: $PAPERS_DIR"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG_PATH" ]; then
    echo "⚠️  警告: 配置文件不存在: $CONFIG_PATH"
    echo "   将使用默认配置"
fi

# 检查GROBID服务 (只有在使用PDF且指定了GROBID URL时才检查)
if [ -n "$GROBID_URL" ]; then
    echo "🔍 检查GROBID服务..."
    if curl -s -f "$GROBID_URL/api/isalive" > /dev/null 2>&1; then
        echo "   ✅ GROBID服务正常"
    else
        echo "   ⚠️  警告: GROBID服务不可用"
        echo "   将使用PyPDF2作为后备方案"
        GROBID_URL=""
    fi
else
    echo "ℹ️  未配置GROBID (如果使用TXT文件，无需GROBID)"
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/figures"

echo ""
echo "=========================================="
echo "开始运行消融实验..."
echo "=========================================="
echo ""

# 构建基础命令
BASE_CMD="python ablation_studies.py --papers_dir \"$PAPERS_DIR\" --output_dir \"$OUTPUT_DIR\""
if [ -n "$CONFIG_PATH" ]; then
    BASE_CMD="$BASE_CMD --config \"$CONFIG_PATH\""
fi
if [ -n "$GROBID_URL" ]; then
    BASE_CMD="$BASE_CMD --grobid_url \"$GROBID_URL\""
fi
if [ -n "$LIMIT" ]; then
    BASE_CMD="$BASE_CMD $LIMIT"
fi

# 运行所有消融实验
echo "📊 运行所有消融实验..."
echo "命令: $BASE_CMD"
echo ""

eval "$BASE_CMD" || {
    echo "❌ 消融实验运行失败"
    exit 1
}

echo ""
echo "=========================================="
echo "生成可视化图表..."
echo "=========================================="
echo ""

python ablation_visualization.py \
    --results_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR/figures" || {
    echo "⚠️  警告: 可视化生成失败"
}

echo ""
echo "=========================================="
echo "✅ 所有任务完成!"
echo "=========================================="
echo ""
echo "结果文件:"
echo "  - JSON结果: $OUTPUT_DIR/*.json"
echo "  - 可视化图表: $OUTPUT_DIR/figures/*.png"
echo ""
echo "下一步:"
echo "  1. 查看结果文件: ls -lh $OUTPUT_DIR/"
echo "  2. 查看图表: ls -lh $OUTPUT_DIR/figures/"
echo "  3. 运行评估脚本分析结果"
echo ""
