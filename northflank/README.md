# TeeUnit Northflank Deployment Guide

This directory contains configuration files for deploying TeeUnit to [Northflank](https://northflank.com/).

## Deployment Options

### 1. Server Deployment (server-service.json)

Deploys the TeeUnit environment server for remote RL training.

```bash
# Deploy via Northflank CLI
northflank create --file northflank/server-service.json

# Or via API
curl -X POST "https://api.northflank.com/v1/projects/{project}/services" \
  -H "Authorization: Bearer $NORTHFLANK_TOKEN" \
  -H "Content-Type: application/json" \
  -d @northflank/server-service.json
```

### 2. GPU Training Job (training-job.json)

Runs self-play RL training on GPU.

```bash
# Deploy GPU training job
northflank create --file northflank/training-job.json
```

## Manual Deployment

### Step 1: Create Project

```bash
northflank projects create --name teeunit-rl
```

### Step 2: Deploy Server

```bash
# Build and deploy server
northflank services create \
  --name teeunit-server \
  --dockerfile Dockerfile \
  --port 7860 \
  --public
```

### Step 3: Run GPU Training

```bash
# Create GPU job
northflank jobs create \
  --name teeunit-training \
  --dockerfile Dockerfile.gpu \
  --gpu nvidia-t4 \
  --command "python -m teeunit.train train --steps 1000000 --device cuda"
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 7860 | HTTP server port |
| `TEEWORLDS_PORT` | 8303 | Teeworlds UDP port |
| `CUDA_VISIBLE_DEVICES` | 0 | GPU device index |

## Remote Training

Once the server is deployed, you can train from any machine:

```bash
# Install teeunit with RL dependencies
pip install teeunit[rl]

# Train against remote server
python -m teeunit.train train \
  --remote https://teeunit-server.northflank.app \
  --steps 1000000 \
  --device cuda
```

## Persistent Storage

The training job mounts volumes for:
- `/app/models` - Saved model checkpoints (10GB)
- `/app/logs` - TensorBoard logs (5GB)

Access saved models after training:

```bash
# List saved checkpoints
northflank storage ls teeunit-training:/app/models

# Download best model
northflank storage cp teeunit-training:/app/models/final_model.zip ./models/
```

## Monitoring

### TensorBoard

Training logs are saved to `/app/logs`. To view in TensorBoard:

```bash
# Forward logs to local TensorBoard
northflank logs teeunit-training --follow | tee training.log

# Or download and view locally
northflank storage cp teeunit-training:/app/logs ./logs/
tensorboard --logdir ./logs
```

### Training Progress

Monitor training via the API:

```bash
curl https://teeunit-server.northflank.app/health
```

## Cost Optimization

- Use spot/preemptible GPUs for training when available
- Scale server to 0 when not in use
- Use checkpointing to resume interrupted training

## Troubleshooting

### GPU Not Detected

Ensure the NVIDIA driver is working:

```bash
northflank exec teeunit-training -- nvidia-smi
```

### Connection Timeout

The Teeworlds server needs a few seconds to start. Check logs:

```bash
northflank logs teeunit-server --tail 100
```

### Out of Memory

Reduce batch size or use gradient accumulation:

```bash
python -m teeunit.train train --batch-size 32 --device cuda
```
