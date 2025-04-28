###############################################
# Multi-stage production Dockerfile for TimeSeriesGPT
###############################################

####################
# Base stage with CUDA and PyTorch
####################
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.6.1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" \
    CONDA_DIR=/opt/conda

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    ca-certificates \
    cmake \
    libopenmpi-dev \
    libomp-dev \
    libjpeg-dev \
    libpng-dev \
    libgomp1 \
    zlib1g-dev \
    curl \
    gnupg \
    unzip \
    lsb-release \
    vim \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh && \
    ${CONDA_DIR}/bin/conda clean -ya

# Set conda path
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Create conda environment with Python 3.10 and PyTorch 2.3 with CUDA 12.1
RUN conda create -n tsenv python=3.10 -y && \
    conda install -n tsenv -c pytorch -c nvidia \
    pytorch=2.3.0 \
    torchvision \
    torchaudio \
    pytorch-cuda=12.1 \
    -y && \
    conda install -n tsenv -c conda-forge \
    matplotlib \
    numpy \
    pandas \
    scikit-learn \
    numba \
    fastapi \
    uvicorn \
    gunicorn \
    celery \
    redis \
    psycopg2 \
    sqlalchemy \
    alembic \
    pydantic \
    -y && \
    conda clean -ya

# Create working directory
WORKDIR /app

# Copy requirements for pip install
COPY requirements.txt .

# Activate conda environment and install pip requirements
RUN . ${CONDA_DIR}/bin/activate tsenv && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    prometheus-client \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-exporter-prometheus \
    torch-tb-profiler \
    sktime \
    mlflow && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir mkl==2023.1.0 mkl-include==2023.1.0 && \
    pip install --no-cache-dir fastapi==0.104.1 uvicorn==0.23.2 redis==5.0.1 psycopg2-binary==2.9.9 celery==5.3.4

# Create a smaller runtime image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc_dir \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    curl \
    libgomp1 \
    libopenmpi-dev \
    libomp-dev \
    gnupg \
    lsb-release \
    python3-minimal \
    python3-pip \
    tzdata \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA Container Toolkit
RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - \
    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    nvidia-container-toolkit \
    dcgm-exporter \
    && rm -rf /var/lib/apt/lists/*

# Install Grafana and Prometheus for monitoring
RUN wget -q -O /usr/share/keyrings/grafana.key https://apt.grafana.com/gpg.key \
    && echo "deb [signed-by=/usr/share/keyrings/grafana.key] https://apt.grafana.com stable main" | tee /etc/apt/sources.list.d/grafana.list \
    && apt-get update && apt-get install -y grafana \
    && rm -rf /var/lib/apt/lists/*

# Install Redis metrics exporter
RUN wget https://github.com/oliver006/redis_exporter/releases/download/v1.43.0/redis_exporter-v1.43.0.linux-amd64.tar.gz \
    && tar xvzf redis_exporter-v1.43.0.linux-amd64.tar.gz \
    && mv redis_exporter-v1.43.0.linux-amd64/redis_exporter /usr/local/bin/redis-metrics-exporter \
    && chmod +x /usr/local/bin/redis-metrics-exporter \
    && rm -rf redis_exporter-v1.43.0.linux-amd64.tar.gz redis_exporter-v1.43.0.linux-amd64

# Copy conda environment from builder
COPY --from=builder ${CONDA_DIR} ${CONDA_DIR}

# Set working directory
WORKDIR /app

# Copy application code
COPY . /app/

# Make scripts executable
RUN chmod +x /app/scripts/*.sh

# Create directory for Prometheus multiprocess metrics
RUN mkdir -p /tmp/prometheus_multiproc_dir && \
    chmod -R 777 /tmp/prometheus_multiproc_dir

# Set up volumes for model data and configurations
VOLUME ["/app/data", "/app/models", "/app/configs"]

# Expose ports for API, monitoring, and Grafana
EXPOSE 8000 9090 9100 9121 9400 3000

# Set default command to entrypoint script
ENTRYPOINT ["/app/scripts/entrypoint.sh"]

# Default to API service if not specified
CMD ["api"] 