#!/bin/bash

# Quick start script for evaluation pipeline

echo "=========================================="
echo "Future Idea Prediction Evaluation"
echo "=========================================="
echo ""

# Check if ideas file is provided
if [ -z "$1" ]; then
    echo "Usage: ./run_evaluation.sh <path_to_ideas_file> [options]"
    echo ""
    echo "Example:"
    echo "  ./run_evaluation.sh data/sample_ideas.json"
    echo "  ./run_evaluation.sh data/my_ideas.json --llm_model gpt-4 --top_k 10"
    echo ""
    echo "To use sample data:"
    echo "  ./run_evaluation.sh data/sample_ideas.json"
    exit 1
fi

IDEAS_FILE=$1
shift  # Remove first argument

# Check if ideas file exists
if [ ! -f "$IDEAS_FILE" ]; then
    echo "Error: Ideas file not found: $IDEAS_FILE"
    exit 1
fi

# Default settings
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="${SCRIPT_DIR}/data/iclr"
VECTOR_DB_DIR="${SCRIPT_DIR}/data/vector_db"
RESULTS_DIR="${SCRIPT_DIR}/results"
CONFIG_PATH="${SCRIPT_DIR}/../../config/config.yaml"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

echo "Configuration:"
echo "  Config file: $CONFIG_PATH"
echo "  Ideas file: $IDEAS_FILE"
echo "  Data directory: $DATA_DIR"
echo "  Vector DB: $VECTOR_DB_DIR"
echo "  Results: $RESULTS_DIR"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Warning: Config file not found at $CONFIG_PATH"
    echo "Will use command-line parameters or defaults."
    echo ""
fi

echo "Starting evaluation..."
echo ""

# Run evaluation with config path
cd "$SCRIPT_DIR/src"
python evaluation_pipeline.py \
    --config_path "$CONFIG_PATH" \
    --ideas_file "$IDEAS_FILE" \
    --data_dir "$DATA_DIR" \
    --vector_db_dir "$VECTOR_DB_DIR" \
    --results_dir "$RESULTS_DIR" \
    "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Evaluation completed successfully!"
    echo "=========================================="
    echo ""
    echo "Results saved to: $RESULTS_DIR"
    echo "  - retrieval_results.json"
    echo "  - evaluation_results.json"
    echo "  - metrics_report.json"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "Evaluation failed with exit code $EXIT_CODE"
    echo "=========================================="
    echo ""
fi

exit $EXIT_CODE
