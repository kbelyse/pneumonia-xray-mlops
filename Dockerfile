FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 PORT=8000 MALLOC_ARENA_MAX=2
WORKDIR /app

COPY requirements.txt .
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /bin/uv
RUN uv pip install --system --no-cache -r requirements.txt

COPY src ./src
COPY ui ./ui
COPY models ./models
COPY data ./data
# Hugging Face runs the container as a normal user, so the folders the app writes to must be writable
RUN mkdir -p storage models/versions && chmod -R a+rwX models storage

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port $PORT"]
