#!/bin/bash
# =============================================================================
# USDA Crop Stress Analyst - Start Script
# Detects host IP, launches the container, waits for readiness,
# then prints connection info.
# =============================================================================

set -e

CONTAINER_NAME="usda-crop-stress-analyst"
APP_HOST_PORT=8093
READINESS_TIMEOUT=600   # 10 minutes max wait
POLL_INTERVAL=30        # seconds between readiness checks

# -- Pre-flight checks -------------------------------------------------------

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Removing existing container..."
    docker rm -f "$CONTAINER_NAME" > /dev/null
fi

MODEL_DIR="./models/Qwen3-VL-8B-Instruct-FP8"
if [ ! -d "$MODEL_DIR" ]; then
    echo ""
    echo "Model not found: $MODEL_DIR"
    echo "   Download the model first:  ./download_models.sh"
    echo ""
    exit 1
fi

if ! docker info &>/dev/null; then
    echo ""
    echo "Docker daemon is not running."
    echo "   Start it with: sudo systemctl start docker"
    echo ""
    exit 1
fi

if ss -ltn 2>/dev/null | grep -q ":${APP_HOST_PORT} "; then
    echo ""
    echo "ERROR: Host port ${APP_HOST_PORT} is already in use."
    echo "  Pick a different port (edit APP_HOST_PORT in this script"
    echo "  and the docker-compose.yml ports mapping)."
    echo ""
    exit 1
fi

if docker ps --format '{{.Names}}' | grep -qE 'vllm'; then
    echo ""
    echo "  Note: another vLLM container is already running on this Nano."
    echo "  This demo launches its OWN vLLM inside its container."
    echo "  Both instances will share the GB10 GPU and compete for memory."
    echo "  Curtis's operational rule: avoid running two vLLM instances at once."
    echo ""
    read -p "  Continue anyway? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# -- Detect host LAN IP ------------------------------------------------------

if [ -z "$HOST_IP" ]; then
    HOST_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}')
fi
if [ -z "$HOST_IP" ]; then
    HOST_IP=$(hostname -I | awk '{print $1}')
fi
export HOST_IP

# -- Launch -----------------------------------------------------------------

echo ""
echo "Launching container..."
docker compose up "$@" > /dev/null 2>&1 &
COMPOSE_PID=$!

# Give docker a moment to actually create the container
sleep 3

# -- Wait for readiness ------------------------------------------------------

echo ""
echo "Waiting for vLLM to load the model and FastAPI to come up..."
echo "(First boot takes 2-3 minutes; CUDA graph compilation can add another minute)"
echo ""

ELAPSED=0
HEALTH_URL="http://localhost:${APP_HOST_PORT}/api/health"

while [ $ELAPSED -lt $READINESS_TIMEOUT ]; do
    # Bail if the container died
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        STATUS=$(docker ps -a --filter "name=${CONTAINER_NAME}" --format '{{.Status}}')
        echo ""
        echo "ERROR: Container is not running."
        echo "  Status: ${STATUS}"
        echo "  Last 30 log lines:"
        echo "  -------------------------------------------------------"
        docker logs --tail 30 "${CONTAINER_NAME}" 2>&1 | sed 's/^/    /'
        echo "  -------------------------------------------------------"
        exit 1
    fi

    # Try the health endpoint
    HEALTH_JSON=$(curl -sf --max-time 3 "${HEALTH_URL}" 2>/dev/null || echo "")
    if echo "${HEALTH_JSON}" | grep -q '"vllm_server":"ready"'; then
        break
    fi

    # Print a status line every POLL_INTERVAL seconds
    MINUTES=$((ELAPSED / 60))
    SECONDS=$((ELAPSED % 60))
    printf "  [%02d:%02d] still loading...\n" $MINUTES $SECONDS

    sleep $POLL_INTERVAL
    ELAPSED=$((ELAPSED + POLL_INTERVAL))
done

if [ $ELAPSED -ge $READINESS_TIMEOUT ]; then
    echo ""
    echo "ERROR: Service did not become ready within ${READINESS_TIMEOUT} seconds."
    echo "  Check logs: docker logs ${CONTAINER_NAME}"
    exit 1
fi

# -- Ready banner ------------------------------------------------------------

TOTAL_MINUTES=$((ELAPSED / 60))
TOTAL_SECONDS=$((ELAPSED % 60))

echo ""
echo "============================================================"
echo "  USDA Crop Stress Analyst - READY"
echo "  Qwen3-VL-8B-Instruct-FP8 | HP ZGX Nano | vLLM"
echo "============================================================"
echo "  Host IP:        $HOST_IP"
echo "  App port:       ${APP_HOST_PORT}"
echo "  Boot time:      ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo ""
echo "  Demo:    http://$HOST_IP:${APP_HOST_PORT}"
echo "  Health:  http://$HOST_IP:${APP_HOST_PORT}/api/health"
echo ""
echo "  LAN access: ensure 'sudo ufw allow ${APP_HOST_PORT}/tcp' is set."
echo "  Stop demo: docker compose down"
echo "============================================================"
echo ""