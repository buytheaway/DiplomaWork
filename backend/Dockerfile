FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgomp1 build-essential libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# ── core dependencies (always installed) ─────────────────────────────
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ── ML dependencies (optional — set INSTALL_ML=true to include) ──────
ARG INSTALL_ML=false
COPY requirements-ml.txt /app/requirements-ml.txt
RUN if [ "$INSTALL_ML" = "true" ]; then \
      pip install --no-cache-dir -r /app/requirements-ml.txt; \
    fi

COPY app /app/app
COPY alembic.ini /app/alembic.ini
COPY entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh
RUN mkdir -p /app/data/index

ENV PYTHONPATH=/app

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
