# TeeUnit - Multi-Agent Arena Environment
# Compatible with HuggingFace Spaces and Northflank

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY teeunit/server/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY teeunit/ /app/teeunit/

# Expose port (HuggingFace Spaces uses 7860, Northflank uses PORT env var)
ENV PORT=7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

# Run the server
CMD ["sh", "-c", "uvicorn teeunit.server.app:app --host 0.0.0.0 --port ${PORT}"]
