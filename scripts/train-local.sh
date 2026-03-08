#!/bin/bash
# TeeUnit Local Training Script
# Starts Teeworlds + FastAPI server locally, then runs training
# This eliminates network latency for much faster training

set -e

echo "=== TeeUnit Local Training ==="
echo "Starting bundled server and training..."

# Create log directory
mkdir -p /var/log/teeunit

# Start Teeworlds server in background
echo "[1/5] Starting Teeworlds server..."
cd /opt/teeworlds
./teeworlds_srv -f teeworlds.cfg > /var/log/teeunit/teeworlds.log 2>&1 &
TEEWORLDS_PID=$!

# Wait longer for Teeworlds to fully initialize
sleep 5

# Check if Teeworlds started
if ! kill -0 $TEEWORLDS_PID 2>/dev/null; then
    echo "ERROR: Teeworlds server failed to start"
    cat /var/log/teeunit/teeworlds.log
    exit 1
fi
echo "    Teeworlds started (PID: $TEEWORLDS_PID)"

# Show Teeworlds log
echo "    Teeworlds log:"
cat /var/log/teeunit/teeworlds.log | head -20 || true

# Start FastAPI server in background with more verbose logging
echo "[2/5] Starting FastAPI server on port 8000..."
cd /app
uvicorn teeunit.server.app:app --host 127.0.0.1 --port 8000 --log-level info > /var/log/teeunit/fastapi.log 2>&1 &
FASTAPI_PID=$!

# Wait for FastAPI to start
sleep 5

# Check if FastAPI started
if ! kill -0 $FASTAPI_PID 2>/dev/null; then
    echo "ERROR: FastAPI server failed to start"
    cat /var/log/teeunit/fastapi.log
    kill $TEEWORLDS_PID 2>/dev/null || true
    exit 1
fi
echo "    FastAPI started (PID: $FASTAPI_PID)"

# Wait for server to be ready (health check with retries)
echo "[3/5] Waiting for server health check..."
MAX_RETRIES=30
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "    Server is responding!"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    sleep 1
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "ERROR: Server failed to become ready"
    cat /var/log/teeunit/fastapi.log
    kill $TEEWORLDS_PID $FASTAPI_PID 2>/dev/null || true
    exit 1
fi

# Initialize environment by calling reset
echo "[4/5] Initializing environment (calling /reset)..."
RESET_RESPONSE=$(curl -s -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}')
RESET_STATUS=$?

if [ $RESET_STATUS -ne 0 ]; then
    echo "ERROR: Failed to reset environment"
    echo "Response: $RESET_RESPONSE"
    cat /var/log/teeunit/fastapi.log
    kill $TEEWORLDS_PID $FASTAPI_PID 2>/dev/null || true
    exit 1
fi

# Check if reset was successful (should have "observations" key)
if echo "$RESET_RESPONSE" | grep -q "observations"; then
    echo "    Environment initialized successfully!"
else
    echo "ERROR: Reset response doesn't contain observations"
    echo "Response: $RESET_RESPONSE"
    cat /var/log/teeunit/fastapi.log
    kill $TEEWORLDS_PID $FASTAPI_PID 2>/dev/null || true
    exit 1
fi

# Run training with local server
echo "[5/5] Starting training..."
echo "    Args: $@"
echo ""

# Pass through all arguments, but always use local server
python3.11 -m teeunit.train train "$@" --remote http://localhost:8000
TRAIN_EXIT_CODE=$?

echo ""
echo "=== Training Complete ==="
echo "Exit code: $TRAIN_EXIT_CODE"

# Show final server logs if there was an error
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "=== FastAPI server log (last 50 lines) ==="
    tail -50 /var/log/teeunit/fastapi.log || true
    echo ""
    echo "=== Teeworlds server log (last 50 lines) ==="
    tail -50 /var/log/teeunit/teeworlds.log || true
fi

# Cleanup background processes
echo "Cleaning up..."
kill $TEEWORLDS_PID $FASTAPI_PID 2>/dev/null || true

exit $TRAIN_EXIT_CODE
