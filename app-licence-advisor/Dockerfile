# Use a slim Python base
FROM python:3.11-slim

# Set envs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8001
ENV RAG_BACKEND=simple

# Set workdir
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything (so app/, ingest/, eval/, tests/, etc. are all available)
COPY . .

# Expose port
EXPOSE ${PORT}

# Start FastAPI (main.py inside app/)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
