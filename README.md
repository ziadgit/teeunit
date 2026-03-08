# TeeUnit: Multi-Agent Arena Environment

A multi-agent deathmatch environment for LLM reinforcement learning, powered by the real [Teeworlds](https://www.teeworlds.com/) game engine. Built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Teeworlds 0.7.5](https://img.shields.io/badge/Teeworlds-0.7.5-orange)](https://github.com/teeworlds/teeworlds)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

TeeUnit wraps the **real Teeworlds 0.7.5 game server** with Python bot clients, exposing it as an OpenEnv-compatible environment for **multi-agent reinforcement learning research**.

### Why Real Teeworlds?

- **Authentic physics**: Proper momentum, hook mechanics, projectile behavior
- **Battle-tested netcode**: Robust UDP protocol with delta compression
- **Rich game state**: Full snapshot data (positions, velocities, weapons, projectiles)
- **Extensible**: Access to all Teeworlds game modes and maps

### Key Features

- **Turn-based mode**: Configurable ticks-per-step for LLM compatibility
- **4 bot agents**: Connect as real game clients via Teeworlds protocol
- **Full observability from snapshots**: Positions, health, weapons, projectiles
- **Simple K/D rewards**: Clear reward signal for training
- **OpenEnv compatible**: Works with TRL, torchforge, and more
- **Docker deployable**: Ready for HuggingFace Spaces and Northflank

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Docker Container                           │
│                                                                 │
│  ┌──────────────────┐         ┌─────────────────────────────┐  │
│  │ Teeworlds Server │◄──UDP──►│  TeeUnit Bot Manager        │  │
│  │ (teeworlds_srv)  │         │  ┌─────────────────────┐    │  │
│  │                  │         │  │ Bot Client 0        │    │  │
│  │ - DM mode        │         │  │ Bot Client 1        │    │  │
│  │ - 4 players      │         │  │ Bot Client 2        │    │  │
│  │ - dm1 map        │         │  │ Bot Client 3        │    │  │
│  └──────────────────┘         │  └─────────────────────┘    │  │
│                               │                             │  │
│                               │  ┌─────────────────────┐    │  │
│                               │  │ FastAPI Server      │    │  │
│                               │  │ (OpenEnv API)       │    │  │
│                               │  └─────────────────────┘    │  │
│                               └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                           │
                                           │ HTTP/WebSocket
                                           ▼
                                  ┌─────────────────────┐
                                  │ External Client     │
                                  │ (LLM Agent)         │
                                  └─────────────────────┘
```

---

## Quick Start

### Docker (Recommended)

```bash
# Pull and run
docker pull ghcr.io/ziadgit/teeunit:latest
docker run -p 7860:7860 ghcr.io/ziadgit/teeunit:latest

# Or build locally
git clone https://github.com/ziadgit/teeunit.git
cd teeunit
docker build -t teeunit:latest .
docker run -p 7860:7860 teeunit:latest
```

### Python Client

```bash
pip install git+https://github.com/ziadgit/teeunit.git
```

```python
from teeunit import TeeEnv, TeeInput

# Connect to running server
with TeeEnv(base_url="http://localhost:7860").sync() as env:
    # Reset starts a new match
    observations = env.reset()
    
    while True:
        # Collect actions for all agents
        actions = {}
        for agent_id, obs in observations.items():
            # Your LLM policy here
            actions[agent_id] = TeeInput(
                direction=1,      # Move right
                target_x=100,     # Aim right
                target_y=0,
                fire=True         # Shoot!
            )
        
        # Step all agents simultaneously
        results = env.step_all(actions)
        
        # Check if episode ended
        state = env.state()
        if state.done:
            print(f"Match ended! Scores: {state.scores}")
            break
        
        # Update observations for next step
        observations = {agent_id: r[0] for agent_id, r in results.items()}
```

---

## Input Format

Actions map directly to Teeworlds `PlayerInput`:

```python
@dataclass
class TeeInput:
    direction: int = 0      # -1 (left), 0 (none), 1 (right)
    target_x: int = 0       # Aim X (relative to player position)
    target_y: int = 0       # Aim Y (relative to player position)
    jump: bool = False      # Jump this tick
    fire: bool = False      # Fire weapon this tick
    hook: bool = False      # Grappling hook this tick
    wanted_weapon: int = 0  # 0=keep, 1=hammer, 2=gun, 3=shotgun, 4=grenade, 5=laser
```

### Example Actions

```python
# Move right and jump
TeeInput(direction=1, jump=True)

# Aim up-right and fire
TeeInput(target_x=100, target_y=-100, fire=True)

# Switch to shotgun
TeeInput(wanted_weapon=3)

# Grappling hook to the right
TeeInput(target_x=200, target_y=0, hook=True)
```

---

## Observation Format

Observations are extracted from Teeworlds game snapshots:

```python
@dataclass
class TeeObservation:
    agent_id: int
    tick: int                          # Current game tick
    
    # Player state (from Character snapshot object)
    position: Tuple[int, int]          # (x, y) in game units
    velocity: Tuple[int, int]          # (vx, vy) 
    angle: int                         # Aim angle (0-360)
    health: int                        # 0-10
    armor: int                         # 0-10
    weapon: int                        # Current weapon ID
    ammo: int                          # Ammo for current weapon
    
    # Score (from PlayerInfo snapshot object)
    score: int
    
    # Visible entities (from snapshot)
    visible_players: List[PlayerState] # Other players in snapshot
    visible_projectiles: List[Projectile]
    visible_pickups: List[Pickup]
    
    # Game state
    game_state_flags: int              # Warmup, paused, etc.
```

### Coordinate System

- **Game units**: Teeworlds uses high-resolution coordinates (1 tile = 32 units)
- **Origin**: Top-left of map
- **Positive X**: Right
- **Positive Y**: Down

---

## Weapons

| ID | Weapon | Damage | Fire Rate | Notes |
|----|--------|--------|-----------|-------|
| 0 | Hammer | 50 | Slow | Melee, knockback |
| 1 | Gun (Pistol) | 15 | Fast | Default, infinite ammo |
| 2 | Shotgun | 30 | Medium | Spread shot |
| 3 | Grenade | 45 | Slow | Explosive, area damage |
| 4 | Laser | 25 | Fast | Instant hit, bounces |
| 5 | Ninja | 90 | Special | Powerup only |

---

## Turn-Based Mode

Teeworlds runs at 50 ticks/second. TeeUnit buffers ticks to create discrete "steps":

```python
# Configure ticks per step (default: 10 = 200ms per step)
env = TeeEnv(base_url="...", ticks_per_step=10)

# Each step() call:
# 1. Sends your input for N ticks
# 2. Waits for N ticks to elapse
# 3. Returns the final observation
```

This makes the environment compatible with LLM inference latency while preserving authentic game physics.

---

## Rewards

| Event | Reward |
|-------|--------|
| Kill enemy | +10.0 |
| Death | -5.0 |
| Damage dealt | +0.1 per HP |

Rewards are calculated from `Sv_KillMsg` network messages and health delta tracking.

---

## Configuration

```python
# At reset, configure the match
observations = env.reset(config={
    "map": "dm1",              # Map name
    "max_turns": 100,          # Episode length (in steps)
    "max_kills": 10,           # Episode ends if any agent reaches this
    "ticks_per_step": 10,      # Game ticks per API step
    "num_agents": 4,           # Number of bot agents (1-8)
})
```

### Server Configuration

The Teeworlds server is configured via `teeworlds.cfg`:

```
sv_name "TeeUnit Arena"
sv_gametype dm
sv_map dm1
sv_max_clients 8
sv_register 0
sv_rcon_password "teeunit"
```

---

## API Reference

### HTTP Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new match |
| `/step` | POST | Execute action for one agent |
| `/step_all` | POST | Execute actions for all agents |
| `/state` | GET | Get full game state |
| `/health` | GET | Health check |

### WebSocket

Connect to `/ws` for real-time streaming updates.

---

## Deployment

### HuggingFace Spaces

```yaml
# In your Space's README.md
sdk: docker
app_port: 7860
```

### Northflank

```bash
# Deploy as a Docker service
docker build -t teeunit .
# Push to your container registry and deploy
```

### Local Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest teeunit/tests/ -v

# The full environment requires Docker (for Teeworlds server)
docker build -t teeunit . && docker run -p 7860:7860 teeunit
```

---

## Protocol Implementation

TeeUnit implements the Teeworlds 0.7 network protocol in Python:

- **UDP client**: Connection handshake, keepalive, reconnection
- **Huffman compression**: For packet payload
- **Variable-int packing**: Teeworlds integer encoding
- **Snapshot parsing**: Delta-compressed game state
- **Input sending**: `NETMSG_INPUT` with `PlayerInput` structure

Based on protocol documentation and reference implementations.

---

## Roadmap

- [x] Core environment with Python simulation
- [x] OpenEnv API compatibility
- [ ] **Real Teeworlds integration** (in progress)
  - [ ] Protocol implementation
  - [ ] Bot client manager
  - [ ] Snapshot parsing
  - [ ] Turn-based tick buffering
- [ ] Team-based modes (TDM, CTF)
- [ ] Custom map support
- [ ] Replay recording/playback
- [ ] Multi-server scaling

---

## Documentation

- [OpenEnv Architecture](docs/openenv.md) - How TeeUnit integrates with OpenEnv and MCP for LLM training
- [Deployment Guide](DEPLOYMENT.md) - Deploy to Northflank with GPU training
- [Training Notebook](notebooks/teeunit_training.ipynb) - Fine-tune LLMs with Unsloth + GRPO

---

## Contributing

Contributions are welcome! Key areas:

1. **Protocol implementation**: Snapshot parsing, edge cases
2. **Bot AI**: Baseline policies for benchmarking
3. **Documentation**: Examples, tutorials
4. **Testing**: Integration tests, CI/CD

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Teeworlds](https://www.teeworlds.com/) - The game engine
- [teeworlds_py](https://github.com/Myr-13/teeworlds_py) - Protocol reference
- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) - Environment framework
- [DDNet](https://ddnet.org/) - Protocol documentation

---

## Citation

```bibtex
@software{teeunit2026,
  title = {TeeUnit: Multi-Agent Arena Environment for LLM Training},
  author = {TeeUnit Contributors},
  year = {2026},
  url = {https://github.com/ziadgit/teeunit}
}
```
