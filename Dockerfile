FROM speaker-service-base

WORKDIR /app

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
