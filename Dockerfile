# =============================================================================
# USDA Crop Stress Analyst - Comparative Moisture Index Analysis
# Powered by HP ZGX Nano AI Station (NVIDIA GB10 Grace Blackwell, sm_121)
#
# Based on NVIDIA vLLM container with Qwen3-VL-8B-Instruct-FP8.
# Single container: vLLM (internal :8090) + FastAPI app (:8000)
#
# Build & run:
#   ./start.sh                     # First time auto-builds + starts
#   docker compose down            # Stop
#
# Prerequisites:
#   - NVIDIA Container Toolkit installed
#   - Model downloaded to ./models/ via download_models.sh
#   - Verified on ZGX Nano with driver 580.95.05
#     (do NOT upgrade base image past :26.01 without confirming driver)
# =============================================================================
FROM nvcr.io/nvidia/vllm:26.01-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install additional Python dependencies for the FastAPI app
COPY backend/requirements-docker.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set up application directory
WORKDIR /app

# Copy application code
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

# Copy entrypoint script
COPY backend/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Models are mounted at runtime from the host
VOLUME /models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=180s \
    CMD curl -f http://localhost:8000/api/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
