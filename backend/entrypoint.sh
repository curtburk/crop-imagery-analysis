#!/bin/bash
# =============================================================================
# USDA Crop Stress Analyst - Container Entrypoint
# Starts vLLM in background, waits for model load, then starts FastAPI
#
# Key delta from single-image demos: VLLM_MAX_MODEL_LEN is bumped to 8192
# because two 1024x1024 vision-token streams plus the comparison prompt
# plus the analyst output do not fit in 4096.
# =============================================================================

set -e

echo ""
echo "============================================================"
echo "  USDA Crop Stress Analyst - Starting"
echo "============================================================"
echo ""

# -- Configuration -----------------------------------------------------------
VLLM_PORT=${VLLM_PORT:-8090}
VLLM_MODEL=${VLLM_MODEL:-/models/Qwen3-VL-8B-Instruct-FP8}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}
APP_PORT=${PORT:-8000}

# -- Start vLLM in background -----------------------------------------------
echo "  Starting vLLM server on port $VLLM_PORT..."
echo "  Model:           $VLLM_MODEL"
echo "  Max model len:   $VLLM_MAX_MODEL_LEN  (sized for two-image input)"
echo ""

python3 -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL" \
    --port "$VLLM_PORT" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --trust-remote-code \
    --dtype auto \
    --gpu-memory-utilization 0.85 \
    --limit-mm-per-prompt '{"image": 2}' \
    &

VLLM_PID=$!

# -- Wait for vLLM to be ready ----------------------------------------------
echo "  Waiting for vLLM to load model (this may take 2-3 minutes)..."

MAX_WAIT=300
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -sf http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo ""
        echo "  vLLM is ready!"
        break
    fi

    # Check if vLLM process died
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "  ERROR: vLLM process died during startup"
        exit 1
    fi

    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  ... still loading ($ELAPSED seconds)"
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "  ERROR: vLLM failed to start within $MAX_WAIT seconds"
    exit 1
fi

# -- Start FastAPI app -------------------------------------------------------
echo ""
echo "  Starting FastAPI application on port $APP_PORT..."
echo ""

cd /app
exec python3 -m uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port "$APP_PORT" \
    --log-level info
