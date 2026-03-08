# TeeUnit OpenEnv Environment
# Compatible with HuggingFace Spaces

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy environment files
COPY teeunit_env/ ./teeunit_env/
COPY teeunit_env/pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir \
    openenv-core>=0.2.1 \
    fastmcp>=0.1.0 \
    fastapi>=0.100.0 \
    uvicorn>=0.23.0 \
    pydantic>=2.0.0 \
    websockets>=11.0

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the server
CMD ["uvicorn", "teeunit_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
