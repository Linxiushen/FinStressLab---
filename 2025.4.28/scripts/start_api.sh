#!/bin/bash
set -e

echo "Starting TimeSeriesGPT API with Prometheus monitoring..."

# Create directory for Prometheus multiprocess mode
mkdir -p ${PROMETHEUS_MULTIPROC_DIR}
chmod 777 ${PROMETHEUS_MULTIPROC_DIR}

# Configure GPU memory limit if specified
if [ "${ENABLE_GPU}" = "true" ]; then
    if [ -n "${GPU_MEMORY_LIMIT}" ]; then
        echo "Setting GPU memory limit to ${GPU_MEMORY_LIMIT}MB"
        export CUDA_VISIBLE_DEVICES=0
        export TF_MEMORY_ALLOCATION=${GPU_MEMORY_LIMIT}
        export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:${GPU_MEMORY_LIMIT}
    fi

    # Start DCGM exporter for GPU metrics
    echo "Starting NVIDIA DCGM exporter for GPU metrics collection..."
    nohup dcgm-exporter --port 9400 &
    sleep 2
fi

# Start Prometheus exporter in background
echo "Starting Prometheus metrics exporter on port ${PROMETHEUS_PORT:-8001}..."
python -m prometheus_client.exposition --multiprocess --port ${PROMETHEUS_PORT:-8001} &

# Calculate number of workers based on CPU cores
if [ -z "${MAX_WORKERS}" ]; then
    CORES=$(nproc)
    MAX_WORKERS=$((CORES * 2 + 1))
    echo "Auto-configuring to ${MAX_WORKERS} workers based on ${CORES} CPU cores"
else
    echo "Using configured MAX_WORKERS: ${MAX_WORKERS}"
fi

# Start the API with Gunicorn
echo "Starting TimeSeriesGPT API on port ${PORT:-8000}..."
exec gunicorn \
    --bind 0.0.0.0:${PORT:-8000} \
    --workers ${MAX_WORKERS} \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 300 \
    --graceful-timeout 60 \
    --keep-alive 5 \
    --log-level info \
    --access-logfile - \
    --error-logfile - \
    batch_prediction_api:app 