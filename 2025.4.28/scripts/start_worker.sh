#!/bin/bash
set -e

echo "Starting TimeSeriesGPT Celery worker..."

# Create directory for Prometheus multiprocess mode
mkdir -p ${PROMETHEUS_MULTIPROC_DIR}
chmod 777 ${PROMETHEUS_MULTIPROC_DIR}

# Configure GPU memory if enabled
if [ "${ENABLE_GPU}" = "true" ]; then
    echo "Worker running with GPU enabled"
    
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
    
    # Verify GPU is available
    echo "Verifying GPU availability..."
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Number of GPUs: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
else
    echo "Worker running in CPU-only mode"
fi

# Configure Celery worker concurrency
if [ -z "$WORKER_CONCURRENCY" ]; then
  if [ "$ENABLE_GPU" = "true" ]; then
    # Default to number of available GPUs for concurrency
    WORKER_CONCURRENCY=$(python3 -c "import torch; print(torch.cuda.device_count() or 1)")
  else
    # Default to available CPU cores or 4
    WORKER_CONCURRENCY=${CPU_COUNT:-4}
  fi
fi

# Configure memory limits
if [ "$ENABLE_GPU" = "true" ]; then
  export GPU_OPTIONS="--without-heartbeat --without-gossip"
  export WORKER_CLASS="worker.GPUWorker"
else
  export GPU_OPTIONS=""
  export WORKER_CLASS="worker.CPUWorker"
fi

echo "Starting Celery worker with concurrency: $WORKER_CONCURRENCY"

# Execute celery worker with appropriate configuration
exec celery -A worker.celery worker \
  --loglevel=${LOG_LEVEL:-INFO} \
  --concurrency=$WORKER_CONCURRENCY \
  --pool=${WORKER_POOL:-prefork} \
  $GPU_OPTIONS \
  --hostname=${HOSTNAME:-worker@%h} \
  -Q ${CELERY_QUEUE:-prediction} \
  --max-tasks-per-child=${MAX_TASKS_PER_CHILD:-100} 