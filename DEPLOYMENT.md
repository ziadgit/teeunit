# TeeUnit Deployment Guide

Deploy TeeUnit to Northflank with GPU training and automatic HuggingFace model upload.

## Prerequisites

- [Northflank account](https://northflank.com/) with GPU access
- [HuggingFace account](https://huggingface.co/) with write token
- [GitHub repository](https://github.com/ziadgit/teeunit) connected to Northflank

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Northflank Project                       │
│                      "Hackathon"                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐     ┌─────────────────────────┐   │
│  │   teeunit-env       │     │   teeunit-train         │   │
│  │   (Internal)        │◄────│   (GPU Job)             │   │
│  │                     │     │                         │   │
│  │  - Teeworlds 0.7.5  │     │  - PPO Training         │   │
│  │  - FastAPI Server   │     │  - NVIDIA T4 GPU        │   │
│  │  - Port 7860        │     │  - 500K steps           │   │
│  └─────────────────────┘     └──────────┬──────────────┘   │
│                                         │                   │
└─────────────────────────────────────────┼───────────────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │   HuggingFace Hub     │
                              │   ziadbc/teeunit-agent│
                              └───────────────────────┘
```

## Step 1: Set Up Northflank Project

```bash
# Install Northflank CLI
npm install -g @northflank/cli

# Login
northflank login

# Set project context
northflank context use project --id hackathon
```

## Step 2: Create HuggingFace Token Secret

1. Get your HuggingFace token from https://huggingface.co/settings/tokens
2. Create a secret in Northflank:

```bash
# Via CLI (create JSON file first)
echo '{"HF_TOKEN": "hf_YOUR_TOKEN_HERE"}' > /tmp/hf-secret.json
northflank create secret --project hackathon -f /tmp/hf-secret.json
rm /tmp/hf-secret.json
```

Or via Northflank UI:
1. Go to Project > Secrets
2. Create new secret named `hf-token`
3. Add key `HF_TOKEN` with your token value

## Step 3: Deploy Environment Server (Internal)

The environment server runs Teeworlds + FastAPI and is **internal only** (not publicly accessible).

```bash
# Create the combined service from spec file
northflank create service combined --project hackathon -f northflank/server-service.json
```

Or using the spec file:

```bash
northflank create service combined --project hackathon -f northflank/server-service.json
```

### Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment |
| `/step` | POST | Execute single agent action |
| `/step_all` | POST | Execute all agent actions |
| `/step_discrete` | POST | Execute discrete action |
| `/observation/{agent_id}` | GET | Get agent observation |
| `/observation_tensor/{agent_id}` | GET | Get observation as tensor |

## Step 4: Deploy GPU Training Job

```bash
# Create the GPU job from spec file
northflank create job manual --project hackathon -f northflank/training-job.json
```

## Step 5: Run Training

```bash
# Start the training job
northflank start job run --project hackathon --job teeunit-train

# Monitor logs (via UI or API)
# Go to: Northflank UI > Project > Jobs > teeunit-train > Logs
```

Training will:
1. Connect to the internal `teeunit-env` server
2. Train PPO agent for 500K steps (~1 hour on T4)
3. Save checkpoints every 10K steps
4. Upload final model to HuggingFace Hub

## Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--steps` | 500000 | Total training timesteps |
| `--device` | cuda | Use GPU |
| `--single-agent` | - | Train single agent, others random |
| `--upload-hf` | - | Auto-upload to HuggingFace |
| `--hf-repo` | ziadbc/teeunit-agent | HuggingFace repo |

### Hyperparameters (defaults)

| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 |
| Batch size | 64 |
| n_epochs | 10 |
| gamma | 0.99 |
| GAE lambda | 0.95 |
| Clip range | 0.2 |
| Entropy coef | 0.01 |

## Local Development

### Run Server Locally

```bash
# Build and run
docker build -t teeunit .
docker run -p 7860:7860 teeunit
```

### Run Training Locally (CPU)

```bash
# Install dependencies
pip install -e ".[rl]"

# Train with local server
python -m teeunit.train train --steps 10000 --device cpu

# Train with remote server
python -m teeunit.train train --remote http://localhost:7860 --steps 10000
```

### Run Training Locally (GPU)

```bash
# Build GPU image
docker build -f Dockerfile.gpu -t teeunit-gpu .

# Run training
docker run --gpus all teeunit-gpu \
  python -m teeunit.train train \
  --steps 100000 \
  --device cuda \
  --upload-hf \
  --hf-repo ziadbc/teeunit-agent
```

## Downloading Trained Model

```python
from huggingface_hub import snapshot_download

# Download model
model_path = snapshot_download(repo_id="ziadbc/teeunit-agent")

# Load with SB3
from stable_baselines3 import PPO
model = PPO.load(f"{model_path}/final_model")
```

## Evaluation

```bash
# Evaluate a trained model
python -m teeunit.train eval models/teeunit_ppo/final_model --episodes 10

# Evaluate with remote server
python -m teeunit.train eval models/teeunit_ppo/final_model \
  --remote http://localhost:7860 \
  --episodes 10
```

## Troubleshooting

### Training job fails to connect to server

Ensure the server service is running and healthy:

```bash
northflank services status teeunit-env
```

The training job should use `--remote http://teeunit-env:7860` to connect to the internal server.

### HuggingFace upload fails

1. Verify the secret is correctly set:
   ```bash
   northflank secrets list
   ```

2. Check the token has write permissions on HuggingFace

3. Check training job logs for specific error:
   ```bash
   northflank jobs logs teeunit-train
   ```

### Out of GPU memory

Reduce batch size or n_steps:

```bash
python -m teeunit.train train --batch-size 32 --steps 500000
```

## Cost Estimation

| Resource | Northflank Estimate |
|----------|---------------------|
| teeunit-env (CPU) | ~$0.05/hour |
| teeunit-train (T4 GPU) | ~$0.50/hour |
| **500K steps training** | **~$0.50-1.00 total** |

## Files Reference

```
teeunit/
├── Dockerfile              # Server container
├── Dockerfile.gpu          # GPU training container
├── DEPLOYMENT.md           # This file
├── northflank/
│   ├── server-service.json # Server deployment spec
│   └── training-job.json   # GPU job spec
└── teeunit/
    ├── train.py            # Training script
    ├── openenv_models.py   # Pydantic models
    ├── openenv_environment.py
    ├── openenv_client.py
    └── server/
        └── app.py          # FastAPI server
```

## Next Steps

After training completes:

1. Download the model from HuggingFace
2. Evaluate performance locally
3. Iterate on reward shaping in `RewardConfig`
4. Consider multi-agent training (remove `--single-agent`)
5. Explore LLM agent fine-tuning with Unsloth (future)
