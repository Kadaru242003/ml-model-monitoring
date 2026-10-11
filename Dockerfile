FROM python:3.12-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

# The trained artifact. Run `python scripts/train.py` before building.
COPY models/ /app/models/
ENV MODELS_DIR=/app/models PREDICTION_DB=/app/logs/predictions.db

RUN useradd --create-home appuser && mkdir -p /app/logs && chown appuser /app/logs
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "hotelmon.api:app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000"]
