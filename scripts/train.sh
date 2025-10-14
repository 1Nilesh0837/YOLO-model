#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${PROJECT_ROOT}"

DEFAULT_MODEL="yolov8n.pt"
DEFAULT_DATA="data/data.yaml"
DEFAULT_EPOCHS=50
DEFAULT_IMGSZ=640
DEFAULT_BATCH=16
DEFAULT_NAME="baseline_run"
DEFAULT_SEED=42

MODEL=${MODEL:-$DEFAULT_MODEL}
DATA=${DATA:-$DEFAULT_DATA}
EPOCHS=${EPOCHS:-$DEFAULT_EPOCHS}
IMGSZ=${IMGSZ:-$DEFAULT_IMGSZ}
BATCH=${BATCH:-$DEFAULT_BATCH}
RUN_NAME=${RUN_NAME:-$DEFAULT_NAME}
SEED=${SEED:-$DEFAULT_SEED}

if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi

print_usage() {
    cat <<'EOF'
Usage: train.sh [--model WEIGHTS] [--data DATA] [--epochs E] [--imgsz S] [--batch B] [--name RUN_NAME] [--seed SEED]

You can also override defaults via environment variables (MODEL, DATA, EPOCHS, IMGSZ, BATCH, RUN_NAME, SEED).
EOF
}

POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"; shift 2 ;;
        --data)
            DATA="$2"; shift 2 ;;
        --epochs)
            EPOCHS="$2"; shift 2 ;;
        --imgsz)
            IMGSZ="$2"; shift 2 ;;
        --batch)
            BATCH="$2"; shift 2 ;;
        --name)
            RUN_NAME="$2"; shift 2 ;;
        --seed)
            SEED="$2"; shift 2 ;;
        -h|--help)
            print_usage; exit 0 ;;
        *)
            POSITIONAL_ARGS+=("$1"); shift ;;
    esac
done

if [[ ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
    echo "Unexpected argument(s): ${POSITIONAL_ARGS[*]}" >&2
    print_usage
    exit 1
fi

export YOLO_EVAL_SEED="${SEED}"

yolo detect train \
    model="${MODEL}" \
    data="${DATA}" \
    epochs="${EPOCHS}" \
    imgsz="${IMGSZ}" \
    batch="${BATCH}" \
    seed="${SEED}" \
    name="${RUN_NAME}" "$@"
