# ============================================================================
# Build stage
# ============================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ============================================================================
# Runtime stage
# ============================================================================
FROM gcr.io/distroless/python3-debian11

# Labels
LABEL org.opencontainers.image.title="LLM AI Dashboard"
LABEL org.opencontainers.image.description="AI Provider management, voice cloning, TTS, and AI memory"
LABEL org.opencontainers.image.source="https://github.com/double-em/llm-hass-app"
LABEL org.opencontainers.image.version="0.1.0"

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Create non-root user
USER root

# Create app directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /app/.local

# Copy application files
COPY --from=builder /build/requirements.txt /app/
COPY *.py /app/
COPY memory/ /app/memory/
COPY templates/ /app/templates/

# Create data directory
RUN mkdir -p /data/voices /data/persons /data/samples /data/memory

# Expose port
EXPOSE 8000

# Run as non-root user
USER default

# Set Python path
ENV PATH=/app/.local/bin:$PATH
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/health', timeout=5)"

CMD ["python", "app.py"]