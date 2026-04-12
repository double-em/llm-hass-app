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
    && rm -rf /var/lib/apt/lists/*

# Pre-copy requirements for better layer caching
COPY requirements.txt .

# Install CPU-only PyTorch first (avoids CUDA dependency bloat in CPU image)
RUN python -m venv /venv && \
    /venv/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.8.0 torchaudio==2.8.0 --no-warn-script-location && \
    /venv/bin/pip install --no-cache-dir -r requirements.txt --no-warn-script-location \
        --ignore-installed torch && \
    /venv/bin/pip install --no-warn-script-location pip --upgrade

# ============================================================================
# Runtime stage
# ============================================================================
FROM python:3.12-slim

# Build arguments
ARG VERSION=unknown

# Labels
LABEL org.opencontainers.image.title="LLM AI Dashboard"
LABEL org.opencontainers.image.description="AI Provider management, voice cloning, TTS, and AI memory for Home Assistant. Supports CPU-only inference (x86_64)."
LABEL org.opencontainers.image.source="https://github.com/double-em/llm-hass-app"
LABEL org.opencontainers.image.version="${VERSION}"

# Environment - reduce image noise
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000
ENV PATH=/venv/bin:$PATH
ENV PYTHONPATH=/app

# Install runtime audio dependency (ffmpeg for pydub)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

# Create app directory
WORKDIR /app

# Copy venv from builder (only site-packages + bin, no cache, no docs)
COPY --from=builder /venv /venv

# Copy application files
COPY *.py /app/
RUN echo '__version__ = "'"${VERSION}"'"' > /app/version.py
COPY memory/ /app/memory/
COPY templates/ /app/templates/

# Create data directory
RUN mkdir -p /data/voices /data/persons /data/samples /data/memory && \
    chown -R appuser:appuser /app /data

EXPOSE 8000

# Health check - socket only, no extra imports
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.connect(('localhost',8000)); s.close()" || exit 1

USER appuser

CMD ["python", "app.py"]
