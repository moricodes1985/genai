# Use Google mirror for reliability
ARG PY_BASE=mirror.gcr.io/library/python:3.11-slim-bullseye
FROM ${PY_BASE}

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl build-essential git libgomp1 && \
    rm -rf /var/lib/apt/lists/*

ENV PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10 \
    PIP_NO_CACHE_DIR=1 \
    # prevent resolver from opportunistically upgrading pinned wheels
    PIP_UPGRADE_STRATEGY=only-if-needed \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

# Upgrade pip tooling
RUN python -m pip install --upgrade pip setuptools wheel --retries 10 --timeout 180

# Create constraints *inside* the image and make pip always use it
RUN printf "numpy==1.26.4\nscipy<1.11\n" > /opt/constraints.txt
ENV PIP_CONSTRAINT=/opt/constraints.txt

# Install NumPy/Scipy first (honors PIP_CONSTRAINT)
RUN pip install --timeout 300 --retries 10 numpy==1.26.4 "scipy<1.11"

# Install PyTorch CPU wheels
RUN pip install --timeout 300 --retries 10 \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# Install the rest (faiss, sentence-transformers, langchain, etc.)
RUN pip install --timeout 300 --retries 10 --prefer-binary \
    faiss-cpu==1.7.4 sentence-transformers==2.5.1 \
    fastapi uvicorn \
    langchain langchain-community langchain-openai langchain-text-splitters \
    langchain-huggingface \
    openai tiktoken requests python-dotenv gradio httpx

# Sanity checks to *fail the build* if NumPy drifts
RUN python -c "import numpy; print('NumPy=', numpy.__version__); assert numpy.__version__.startswith('1.26.'), 'NumPy must be 1.26.x'"
RUN pip check

# Default CMD is overridden by compose
CMD ["python", "-c", "print('image built')"]
