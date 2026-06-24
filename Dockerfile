FROM harbor.intra.ke.com/keci/python:3.12.3

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Python dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

# Application source
COPY api.py ./
COPY engine/ ./engine/
COPY client/ ./client/
COPY models/ ./models/

ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    HTTP_PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

ENTRYPOINT ["python", "api.py"]
