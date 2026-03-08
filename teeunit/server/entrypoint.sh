#!/bin/bash
set -e

# TeeUnit Entrypoint Script
# Launches Teeworlds server and FastAPI in parallel

echo "Starting TeeUnit..."

# Start Teeworlds server in background
echo "Starting Teeworlds server on port ${TEEWORLDS_PORT:-8303}..."
cd /opt/teeworlds
./teeworlds_srv -f teeworlds.cfg &
TEEWORLDS_PID=$!

# Wait for server to start
sleep 2

# Check if Teeworlds is running
if ! kill -0 $TEEWORLDS_PID 2>/dev/null; then
    echo "ERROR: Teeworlds server failed to start"
    exit 1
fi

echo "Teeworlds server started (PID: $TEEWORLDS_PID)"

# Start FastAPI server
echo "Starting FastAPI server on port ${PORT:-7860}..."
cd /app
exec uvicorn teeunit.server.app:app \
    --host 0.0.0.0 \
    --port ${PORT:-7860} \
    --log-level info
