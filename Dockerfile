# TeeUnit - Multi-Agent Arena Environment
# Runs real Teeworlds 0.7.5 server with Python bot clients
# Compatible with HuggingFace Spaces and Northflank

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Download and extract Teeworlds 0.7.5 server
ENV TEEWORLDS_VERSION=0.7.5
RUN curl -fsSL "https://github.com/teeworlds/teeworlds/releases/download/${TEEWORLDS_VERSION}/teeworlds-${TEEWORLDS_VERSION}-linux_x86_64.tar.gz" \
    | tar xz -C /opt \
    && mv "/opt/teeworlds-${TEEWORLDS_VERSION}-linux_x86_64" /opt/teeworlds \
    && chmod +x /opt/teeworlds/teeworlds_srv

# Install Python dependencies
COPY teeunit/server/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY teeunit/ /app/teeunit/
COPY pyproject.toml /app/

# Copy Teeworlds server configuration
COPY teeunit/server/teeworlds.cfg /opt/teeworlds/teeworlds.cfg

# Copy supervisor configuration
COPY teeunit/server/supervisord.conf /etc/supervisor/conf.d/teeunit.conf

# Copy entrypoint script
COPY teeunit/server/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose port (HuggingFace Spaces uses 7860, Northflank uses PORT env var)
ENV PORT=7860
ENV TEEWORLDS_PORT=8303
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

# Run both Teeworlds server and FastAPI via supervisor
ENTRYPOINT ["/entrypoint.sh"]
