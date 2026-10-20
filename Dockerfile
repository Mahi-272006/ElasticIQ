# ElasticIQ API image: builds the React dashboard, then serves API + static
# frontend from a single process.
#
# NOTE: this file was written where Docker is not installed, so the image has
# not been built locally — treat `docker build` as the first real verification.

# ---- Stage 1: frontend build -------------------------------------------------
FROM node:22-alpine AS frontend

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: API ------------------------------------------------------------
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code, migrations, and the (small, committed) trained model.
COPY src/ ./src/
COPY migrations/ ./migrations/
COPY alembic.ini ./
COPY artifacts/ ./artifacts/

# Built dashboard from stage 1 (FastAPI serves frontend/dist automatically).
COPY --from=frontend /app/frontend/dist ./frontend/dist

# Processed data is NOT baked in (240 MB raw CSVs stay outside git/images):
# mount it at /app/data/processed or run the pipeline inside the container.
VOLUME ["/app/data/processed"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
