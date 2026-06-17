#!/bin/bash
# =============================================================================
# USDA Crop Stress Analyst - Model Provisioning
# Provisions Qwen3-VL-8B-Instruct-FP8 (~9GB)
#
# Order of preference:
#   1. ./models/Qwen3-VL-8B-Instruct-FP8 already populated -> done
#   2. HuggingFace cache (~/.cache/huggingface) has the snapshot -> symlink
#   3. Download fresh from HuggingFace Hub
# =============================================================================

set -e

echo ""
echo "============================================================"
echo "  USDA Crop Stress Analyst - Model Provisioning"
echo "============================================================"
echo ""
echo "  Model: Qwen/Qwen3-VL-8B-Instruct-FP8 (~9GB)"
echo ""

MODEL_DIR="./models/Qwen3-VL-8B-Instruct-FP8"
REPO_ID="Qwen/Qwen3-VL-8B-Instruct-FP8"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
HF_REPO_DIR="$HF_CACHE/models--Qwen--Qwen3-VL-8B-Instruct-FP8"

# -- 1. Already provisioned locally? -----------------------------------------
if [ -d "$MODEL_DIR" ] && [ -n "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    # Sanity check: confirm at least a config.json is present
    if [ -f "$MODEL_DIR/config.json" ] || [ -L "$MODEL_DIR/config.json" ]; then
        echo "  Model already provisioned at $MODEL_DIR"
        echo "  To re-provision, delete the directory first:"
        echo "     rm -rf $MODEL_DIR"
        echo ""
        exit 0
    else
        echo "  Found $MODEL_DIR but it looks incomplete (no config.json)"
        echo "  Removing and re-provisioning..."
        rm -rf "$MODEL_DIR"
    fi
fi

mkdir -p ./models

# -- 2. Check HF cache for an existing snapshot ------------------------------
if [ -d "$HF_REPO_DIR/snapshots" ]; then
    # Find the most recent snapshot directory that actually contains files
    LATEST_SNAPSHOT=$(find "$HF_REPO_DIR/snapshots" -mindepth 1 -maxdepth 1 -type d \
        -exec sh -c '[ -f "$1/config.json" ] && echo "$1"' _ {} \; 2>/dev/null \
        | head -n 1)

    if [ -n "$LATEST_SNAPSHOT" ]; then
        echo "  Found existing snapshot in HuggingFace cache:"
        echo "     $LATEST_SNAPSHOT"
        echo ""
        echo "  Symlinking to $MODEL_DIR (no re-download needed)..."

        # Resolve to absolute paths so the symlink target is unambiguous
        LATEST_SNAPSHOT_ABS=$(readlink -f "$LATEST_SNAPSHOT")
        MODEL_DIR_ABS=$(readlink -f "$(dirname "$MODEL_DIR")")/$(basename "$MODEL_DIR")

        ln -s "$LATEST_SNAPSHOT_ABS" "$MODEL_DIR_ABS"

        echo ""
        echo "  Provisioned via symlink."
        echo "  Start the demo with: ./start.sh"
        echo ""
        exit 0
    fi
fi

# -- 3. Fall back to fresh download ------------------------------------------
echo "  Model not found in HuggingFace cache."
echo "  Downloading $REPO_ID (~9GB)..."
echo "  This may take 10-15 minutes depending on network speed."
echo ""

# Create a temporary venv if not already in one
if [ -z "$VIRTUAL_ENV" ]; then
    echo "  Creating temporary Python environment..."
    python3 -m venv /tmp/hf-download-env
    source /tmp/hf-download-env/bin/activate
    pip install -q huggingface_hub
fi

python3 << EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$REPO_ID",
    local_dir="$MODEL_DIR"
)
EOF

if [ -d "$MODEL_DIR" ] && [ -n "$(ls -A "$MODEL_DIR")" ]; then
    echo ""
    echo "  Download complete!"
    echo "  Location: $MODEL_DIR"
    echo ""
    echo "  Start the demo with: ./start.sh"
    echo ""
else
    echo ""
    echo "  Download failed!"
    exit 1
fi