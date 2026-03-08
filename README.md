# TeeUnit: Multi-Agent Arena Environment

A turn-based multi-agent deathmatch environment for LLM reinforcement learning, inspired by [Teeworlds](https://www.teeworlds.com/). Built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-green.svg)](LICENSE)

---

## Overview

TeeUnit is designed for **multi-agent reinforcement learning research**, specifically targeting:

- **Theory-of-mind reasoning**: Agents must predict opponent behavior
- **Emergent strategic behavior**: Ambush tactics, resource denial, positioning
- **Competitive multi-agent dynamics**: 4 agents in free-for-all combat

### Key Features

- **Turn-based combat**: Discretized actions for LLM compatibility
- **Partial observability**: Agents only see within their vision radius
- **Simple K/D rewards**: Clear reward signal for training
- **OpenEnv compatible**: Works with TRL, torchforge, Unsloth, and more
- **Deployable**: Ready for HuggingFace Spaces and Northflank

---

## Quick Start

### Installation

```bash
# Install from GitHub
pip install git+https://github.com/ziadgit/teeunit.git

# Or clone and install locally
git clone https://github.com/ziadgit/teeunit.git
cd teeunit
pip install -e .
```

### Basic Usage

```python
import asyncio
from teeunit import TeeAction, TeeEnv

async def main():
    # Connect to a running TeeUnit server
    async with TeeEnv(base_url="https://your-deployment-url") as env:
        # Reset starts a new match
        result = await env.reset()
        print(f"Match started! Episode: {result.observation.episode_id}")
        
        # Game loop - each agent takes actions
        while not result.done:
            for agent_id in range(4):
                # Get observation for this agent
                obs = result.observation
                
                # Decide action based on observation
                action = TeeAction(
                    agent_id=agent_id,
                    action_type="move",
                    direction="north"
                )
                
                result = await env.step(action)
                print(f"Agent {agent_id}: {obs.text_description}")

asyncio.run(main())
```

### Synchronous Usage

```python
from teeunit import TeeAction, TeeEnv

with TeeEnv(base_url="https://your-deployment-url").sync() as env:
    result = env.reset()
    
    action = TeeAction(agent_id=0, action_type="shoot", target_x=5, target_y=5)
    result = env.step(action)
    print(result.observation.text_description)
```

---

## Game Mechanics

### Arena

- **Grid size**: 20x20 cells (configurable)
- **Terrain types**: 
  - `empty`: Walkable space
  - `wall`: Blocks movement and line-of-sight
  - `water`: Damaging terrain (5 HP/turn)
  - `platform`: Elevated terrain

### Agents

Each match has **4 agents** competing in free-for-all deathmatch.

| Attribute | Value | Description |
|-----------|-------|-------------|
| Health | 100 | Reduced by damage, 0 = death |
| Armor | 0-100 | Absorbs 50% of incoming damage |
| Vision | 6 cells | Radius of visibility |
| Spawn protection | 3 turns | Invulnerable after respawn |

### Actions

Agents choose one action per turn:

| Action | Parameters | Description |
|--------|------------|-------------|
| `move` | `direction` | Move 1 cell in direction (n/s/e/w/ne/nw/se/sw) |
| `shoot` | `target_x`, `target_y` | Fire current weapon at target |
| `switch_weapon` | `weapon` | Change to specified weapon |
| `use_item` | - | Use picked up item (health pack, etc.) |
| `wait` | - | Do nothing this turn |

### Weapons

| Weapon | Damage | Range | Ammo | Notes |
|--------|--------|-------|------|-------|
| Pistol | 15 | 8 cells | ∞ | Default weapon, reliable |
| Shotgun | 30 | 4 cells | 10 | High damage, close range |
| Laser | 25 | 12 cells | 5 | Instant hit, long range |
| Hammer | 50 | 1 cell | ∞ | Melee, devastating |

### Pickups

Spawn randomly on the arena and respawn after 10 turns:

| Pickup | Effect |
|--------|--------|
| Health Pack | +25 health |
| Armor | +50 armor |
| Shotgun Ammo | +5 shells |
| Laser Ammo | +3 charges |

### Vision & Partial Observability

- **Vision radius**: 6 cells from agent position
- **Line-of-sight**: Blocked by walls
- **Sound events**: Shots heard within 10 cells (direction only)
- **Memory**: Last known enemy positions tracked for 3 turns

---

## Observations

Each agent receives an observation tailored to their perspective:

```python
@dataclass
class TeeObservation:
    agent_id: int                      # Which agent (0-3)
    position: Tuple[int, int]          # Current (x, y) position
    health: int                        # Current health (0-100)
    armor: int                         # Current armor (0-100)
    current_weapon: str                # Equipped weapon name
    ammo: Dict[str, int]               # Ammo per weapon type
    visible_enemies: List[Dict]        # Enemies in line-of-sight
    visible_pickups: List[Dict]        # Pickups in line-of-sight
    nearby_obstacles: List[Tuple]      # Wall positions nearby
    recent_events: List[str]           # Combat log entries
    turn_number: int                   # Current turn
    your_kills: int                    # Total kills this match
    your_deaths: int                   # Total deaths this match
    text_description: str              # Natural language summary
```

### Example Text Description

```
Turn 15 | Position: (8, 12) | Health: 75/100 | Armor: 25
Weapon: Shotgun (8 ammo)

VISIBLE ENEMIES:
- Agent 2 at (10, 14), ~30 health, 3 cells away (northeast)
- Agent 0 at (5, 10), ~80 health, 4 cells away (southwest)

PICKUPS NEARBY:
- Health Pack at (9, 13), 2 cells away

RECENT EVENTS:
- You hit Agent 2 for 30 damage!
- Agent 3 was eliminated by Agent 1

Your score: 2 kills, 1 death
```

---

## Rewards

Simple kill/death reward structure:

| Event | Reward |
|-------|--------|
| Kill enemy | +10.0 |
| Assist (damaged enemy killed by another) | +3.0 |
| Death | -5.0 |
| Damage dealt | +0.1 per HP |
| Survival bonus (per turn alive) | +0.1 |
| Match victory (most kills at end) | +20.0 |

---

## Episode Structure

### Start Conditions
- 4 agents spawn at predefined spawn points
- Each agent starts with 100 HP, 0 armor, pistol equipped
- Pickups spawn at random locations

### Turn Resolution Order
1. All agents submit actions (with timeout)
2. Movement resolved (collision detection)
3. Weapon switches applied
4. Shots fired and damage calculated
5. Deaths registered, respawn queued
6. Pickups collected
7. Observations generated
8. Turn counter incremented

### End Conditions
- **Turn limit**: Match ends after 100 turns (configurable)
- **Dominance**: One agent reaches 10 kills
- Winner = agent with most kills (ties broken by deaths, then damage dealt)

---

## Training Integration

### With TRL (Hugging Face)

```python
from trl import GRPOTrainer
from teeunit import TeeEnv, TeeAction

# See examples/trl_training.py for full implementation
```

### With torchforge (PyTorch)

```python
from torchforge import GRPOTrainer
from teeunit import TeeEnv

# See examples/torchforge_training.py for full implementation
```

### Multi-Agent Training Pattern

```python
async def collect_trajectory(env, agents):
    """Collect a full episode trajectory for all agents."""
    result = await env.reset()
    trajectories = {i: [] for i in range(4)}
    
    while not result.done:
        for agent_id in range(4):
            # Get agent's observation
            obs = await env.get_observation(agent_id)
            
            # Agent decides action (your LLM policy here)
            action = agents[agent_id].act(obs.text_description)
            
            # Execute action
            result = await env.step(action)
            
            # Store transition
            trajectories[agent_id].append({
                'observation': obs,
                'action': action,
                'reward': result.reward,
                'done': result.done
            })
    
    return trajectories
```

---

## Deployment

### HuggingFace Spaces

```bash
# Login to Hugging Face
huggingface-cli login

# Deploy using OpenEnv CLI
cd teeunit
openenv push --repo-id your-username/teeunit
```

### Northflank (Docker)

```bash
# Build the Docker image
docker build -t teeunit:latest -f server/Dockerfile .

# Run locally
docker run -p 8000:8000 teeunit:latest

# Push to your container registry
docker tag teeunit:latest your-registry/teeunit:latest
docker push your-registry/teeunit:latest
```

### Local Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Run server locally
uvicorn teeunit.server.app:app --host 0.0.0.0 --port 8000 --reload

# Run tests
pytest tests/ -v
```

---

## Configuration

Environment behavior can be configured at reset:

```python
result = await env.reset(config={
    "arena_width": 20,
    "arena_height": 20,
    "max_turns": 100,
    "num_agents": 4,
    "vision_radius": 6,
    "spawn_protection_turns": 3,
    "pickup_respawn_turns": 10,
    "win_kill_threshold": 10,
})
```

---

## API Reference

### Environment Methods

| Method | Description |
|--------|-------------|
| `reset(config?)` | Start new match, returns initial observations |
| `step(action)` | Execute action, returns StepResult |
| `state()` | Get current episode state |
| `get_observation(agent_id)` | Get specific agent's observation |

### TeeAction Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `agent_id` | int | Yes | Agent taking action (0-3) |
| `action_type` | str | Yes | "move", "shoot", "switch_weapon", "use_item", "wait" |
| `direction` | str | For move | "n", "s", "e", "w", "ne", "nw", "se", "sw" |
| `target_x` | int | For shoot | Target X coordinate |
| `target_y` | int | For shoot | Target Y coordinate |
| `weapon` | str | For switch | "pistol", "shotgun", "laser", "hammer" |

### StepResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `observation` | TeeObservation | Agent's new observation |
| `reward` | float | Reward for this step |
| `done` | bool | Whether episode ended |
| `info` | dict | Additional metadata |

---

## Multi-Agent Scenarios

TeeUnit is designed as a showcase for multi-agent LLM training:

### Competition
- Agents compete for kills and survival
- Zero-sum dynamics encourage strategic play
- Emergent behaviors: camping, baiting, third-party attacks

### Theory of Mind
- Partial observability requires modeling opponent intentions
- "If I move here, what will Agent 2 do?"
- Predicting enemy positions based on last known location + time

### Strategic Depth
- Weapon selection based on engagement distance
- Resource control (denying health packs to damaged enemies)
- Positioning advantages (corners, sight lines)

---

## Roadmap

- [x] Core environment implementation
- [x] Partial observability
- [x] Simple K/D rewards
- [ ] Team-based modes (2v2)
- [ ] Capture the Flag mode
- [ ] Inter-agent communication (chat actions)
- [ ] Custom map support
- [ ] Replay visualization

---

## Contributing

Contributions are welcome! Please see our contribution guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

---

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Teeworlds](https://www.teeworlds.com/) - Inspiration for game mechanics
- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) - Environment framework
- [Gymnasium](https://gymnasium.farama.org/) - API design inspiration

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
