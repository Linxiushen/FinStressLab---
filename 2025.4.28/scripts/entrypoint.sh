#!/bin/bash
set -e

# Default environment variables
: ${PORT:=8000}
: ${PROMETHEUS_PORT:=9090}
: ${MAX_WORKERS:=$(nproc)}
: ${WORKER_CLASS:="uvicorn.workers.UvicornWorker"}
: ${ENABLE_GPU:=true}
: ${ENABLE_AUTO_SCALING:=true}
: ${LOG_LEVEL:=info}
: ${REDIS_URL:="redis://redis:6379/0"}
: ${MODEL_CHECKPOINT_PATH:="/app/models/checkpoint"}
: ${CONFIG_PATH:="/app/configs/production.json"}

# Create necessary directories
mkdir -p /app/logs /app/models /app/data /app/configs /tmp/prometheus_multiproc_dir
chmod -R 777 /tmp/prometheus_multiproc_dir

# Configure GPU settings if enabled
if [ "$ENABLE_GPU" = "true" ]; then
    echo "Configuring for GPU use..."
    # Check NVIDIA GPU availability
    if ! command -v nvidia-smi &> /dev/null; then
        echo "WARNING: ENABLE_GPU is set to true but nvidia-smi is not available"
        export ENABLE_GPU=false
    else
        nvidia-smi
        # Set GPU memory fraction if specified
        if [ ! -z "$GPU_MEMORY_FRACTION" ]; then
            echo "Setting GPU memory fraction to $GPU_MEMORY_FRACTION"
            export CUDA_VISIBLE_DEVICES=0
            python -c "import torch; torch.cuda.set_per_process_memory_fraction($GPU_MEMORY_FRACTION, 0)"
        fi
    fi
fi

# Start DCGM exporter for GPU metrics if GPU is enabled
if [ "$ENABLE_GPU" = "true" ]; then
    echo "Starting NVIDIA DCGM exporter on port 9400..."
    dcgm-exporter &
fi

# Start Redis metrics exporter if auto-scaling is enabled
if [ "$ENABLE_AUTO_SCALING" = "true" ]; then
    echo "Starting Redis metrics exporter..."
    redis-metrics-exporter --redis.addr=${REDIS_URL} --web.listen-address=:9121 &
fi

# Handle different service types
case "$1" in
    api)
        echo "Starting API service on port $PORT..."
        exec gunicorn batch_prediction_api:app \
            --bind 0.0.0.0:$PORT \
            --workers $MAX_WORKERS \
            --worker-class $WORKER_CLASS \
            --timeout 300 \
            --graceful-timeout 60 \
            --keep-alive 75 \
            --log-level $LOG_LEVEL \
            --log-file /app/logs/api.log \
            --access-logfile /app/logs/access.log \
            --error-logfile /app/logs/error.log
        ;;
    worker)
        echo "Starting Celery worker..."
        # Determine concurrency based on resources
        if [ "$ENABLE_GPU" = "true" ]; then
            NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
            WORKER_CONCURRENCY=${WORKER_CONCURRENCY:-$NUM_GPUS}
            WORKER_POOL="prefork"
            # For GPU workers, limit max tasks per child to prevent memory leaks
            MAX_TASKS_PER_CHILD=${MAX_TASKS_PER_CHILD:-100}
        else
            WORKER_CONCURRENCY=${WORKER_CONCURRENCY:-$(nproc)}
            WORKER_POOL="prefork"
            MAX_TASKS_PER_CHILD=${MAX_TASKS_PER_CHILD:-500}
        fi
        
        exec celery -A worker worker \
            --loglevel=$LOG_LEVEL \
            --concurrency=$WORKER_CONCURRENCY \
            --pool=$WORKER_POOL \
            --max-tasks-per-child=$MAX_TASKS_PER_CHILD \
            --logfile=/app/logs/celery.log
        ;;
    monitor)
        echo "Starting monitoring dashboard..."
        # Start Grafana
        grafana-server \
            --homepath=/usr/share/grafana \
            --config=/etc/grafana/grafana.ini \
            --packaging=docker \
            cfg:default.log.mode=console \
            cfg:default.paths.data=/var/lib/grafana \
            cfg:default.paths.logs=/var/log/grafana \
            cfg:default.paths.plugins=/var/lib/grafana/plugins \
            cfg:default.paths.provisioning=/etc/grafana/provisioning &
        
        # Auto-provision the TimeSeriesGPT dashboard if available
        if [ -f "/app/configs/grafana/dashboard.json" ]; then
            echo "Auto-provisioning TimeSeriesGPT dashboard..."
            mkdir -p /etc/grafana/provisioning/dashboards
            cp /app/configs/grafana/dashboard.json /etc/grafana/provisioning/dashboards/
        fi
        
        # Keep the container running
        tail -f /app/logs/*.log
        ;;
    scheduler)
        echo "Starting Celery beat scheduler..."
        exec celery -A worker beat \
            --loglevel=$LOG_LEVEL \
            --schedule=/app/data/celerybeat-schedule \
            --logfile=/app/logs/celerybeat.log
        ;;
    incremental)
        echo "Starting incremental learning service..."
        exec python -m deploy_incremental \
            --host 0.0.0.0 \
            --port $PORT \
            --db_url ${DB_URL:-"postgresql://postgres:postgres@timescaledb:5432/tsdb"} \
            --gpu_memory_fraction ${GPU_MEMORY_FRACTION:-0.7}
        ;;
    spot-handler)
        echo "Starting spot instance termination handler..."
        # Monitor for spot instance termination notice and gracefully shutdown
        while true; do
            # AWS spot termination notice appears at this URL
            TERMINATION_TIME=$(curl -s http://169.254.169.254/latest/meta-data/spot/termination-time)
            if [[ $TERMINATION_TIME != "" ]]; then
                echo "Spot instance termination notice received: $TERMINATION_TIME"
                # Save model state
                python -c "from model import save_model_state; save_model_state('${MODEL_CHECKPOINT_PATH}')"
                # Gracefully stop services
                pkill -TERM gunicorn || true
                pkill -TERM celery || true
                sleep 5
                exit 0
            fi
            sleep 5
        done
        ;;
    *)
        echo "Unknown service type: $1"
        echo "Available options: api, worker, monitor, scheduler, incremental, spot-handler"
        exit 1
        ;;
esac 