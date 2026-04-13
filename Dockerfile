# syntax=docker/dockerfile:1

# ============================================================================
# Build stage
# ============================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies for packages that require compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create venv and install PyTorch first (CPU-only, single index)
RUN python -m venv /venv && \
    /venv/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.8.0 torchaudio==2.8.0

# Install remaining dependencies via pip (will be copied to runtime)
RUN /venv/bin/pip install --no-cache-dir \
        flask>=3.0.0 requests>=2.31.0 numpy>=1.24.0 websockets>=10.0 \
        resemblyzer>=0.1.0 sentence-transformers>=2.2.0 chromadb>=0.4.0

# Install omnivoice last (it's the heaviest)
RUN /venv/bin/pip install --no-cache-dir omnivoice>=0.1.0

# ============================================================================
# Runtime stage - minimal production image
# ============================================================================
FROM python:3.12-slim

# Build arguments
ARG VERSION=unknown

# Labels
LABEL org.opencontainers.image.title="LLM AI Dashboard"
LABEL org.opencontainers.image.description="AI Provider management, voice cloning, TTS, and AI memory for Home Assistant. Supports CPU-only inference (x86_64)."
LABEL org.opencontainers.image.source="https://github.com/double-em/llm-hass-app"
LABEL org.opencontainers.image.version="${VERSION}"

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    PYTHONPATH=/app

# Install runtime audio dependency (ffmpeg for pydub)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy venv site-packages (only site-packages, no cache, no docs)
COPY --from=builder /venv/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /venv/bin/pip /venv/bin/pip
COPY --from=builder /venv/bin/python /usr/local/bin/python
COPY --from=builder /venv/bin/python3 /usr/local/bin/python3
COPY --from=builder /venv/bin/python3.12 /usr/local/bin/python3.12

# Ensure pip is accessible
RUN ln -sf /usr/local/bin/python /venv/bin/python && \
    ln -sf /usr/local/bin/python /venv/bin/python3 && \
    ln -sf /usr/local/bin/pip /venv/bin/pip || true

# Create non-root user
RUN useradd -m -u 1000 appuser

# Create app directory
WORKDIR /app

# Copy application files
COPY *.py /app/
RUN echo '__version__ = "'"${VERSION}"'"' > /app/version.py
COPY memory/ /app/memory/
COPY templates/ /app/templates/

# Create data directory with proper ownership
RUN mkdir -p /data && chown -R appuser:appuser /app /data

EXPOSE 8000

USER appuser

CMD ["python", "app.py"]
