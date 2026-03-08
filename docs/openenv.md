# How OpenEnv is Part of the TeeUnit Architecture

This document explains how TeeUnit integrates with the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework to enable LLM-based reinforcement learning on the Teeworlds game.

## What is OpenEnv?

**OpenEnv** is a universal RL environment interface framework that standardizes how AI agents interact with environments. It provides:

1. **Pydantic-based models** for Actions, Observations, and States
2. **MCP (Model Context Protocol)** tool interface for LLM agents
3. **HTTP/WebSocket server** scaffolding for remote environments
4. **Client libraries** for connecting to environments

Unlike traditional RL interfaces (Gym/Gymnasium) that use numerical tensors, OpenEnv uses **natural language** and **tool calls** that LLMs can process natively.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  LLM (Llama/GPT/etc)                                    │
│  "Enemy is close to my right... I should use hammer"    │
└─────────────────────────┬───────────────────────────────┘
                          │ MCP Tool Calls
                          ▼
┌─────────────────────────────────────────────────────────┐
│  OpenEnv MCP Interface                                  │
│  call_tool("shoot", weapon=0)  →  "Swung hammer!"       │
└─────────────────────────┬───────────────────────────────┘
                          │ Pydantic Models
                          ▼
┌─────────────────────────────────────────────────────────┐
│  TeeUnit Environment                                    │
│  TeeAction(fire=True, weapon=0)                         │
└─────────────────────────┬───────────────────────────────┘
                          │ UDP Protocol
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Teeworlds Server                                       │
│  *character swings hammer in game*                      │
└─────────────────────────────────────────────────────────┘
```

## Layer 1: OpenEnv Manifest

The `openenv.yaml` file declares TeeUnit as an OpenEnv-compatible environment:

```yaml
name: teeunit
version: "0.1.0"
environment:
  type: multi-agent
  agents: 4
  observation_type: partial
  action_space: discrete
  
server:
  entrypoint: teeunit.server.app:app
  framework: fastapi
  port: 7860

api:
  http:
    - POST /reset
    - POST /step
    - GET /state
```

This manifest allows OpenEnv tooling to automatically discover and interact with TeeUnit.

## Layer 2: OpenEnv Models

TeeUnit defines Pydantic models in `teeunit/openenv_models.py` that extend OpenEnv's base classes:

```python
class Action(BaseModel):
    """OpenEnv-compatible base action."""
    metadata: Dict[str, Any] = {}

class Observation(BaseModel):
    """OpenEnv-compatible base observation."""
    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = {}

class TeeAction(Action):
    """TeeUnit-specific action."""
    direction: int = 0      # -1=left, 0=none, 1=right
    jump: bool = False
    fire: bool = False
    hook: bool = False
    weapon: int = 1
    target_x: int = 0
    target_y: int = 0

class TeeObservation(Observation):
    """TeeUnit-specific observation."""
    agent_id: int
    position: Tuple[float, float]
    health: int
    armor: int
    weapon: int
    visible_players: List[VisiblePlayer]
    visible_projectiles: List[VisibleProjectile]
    # ... more fields
```

These models ensure type safety and validation for all environment interactions.

## Layer 3: MCP Tool Interface

The key innovation is the **MCP (Model Context Protocol)** tool interface in `teeunit_env/server/tee_environment.py`. This lets LLMs interact via natural language tool calls:

```python
class TeeEnvironment(MCPEnvironment):
    def __init__(self, num_agents=4, max_steps=1000):
        mcp = FastMCP("teeunit_env")
        
        @mcp.tool
        def move(direction: str) -> str:
            """Move the tee horizontally.
            
            Args:
                direction: "left", "right", or "none"
            """
            if direction == "left":
                self._pending_input.direction = -1
                return "Moving left."
            elif direction == "right":
                self._pending_input.direction = 1
                return "Moving right."
            else:
                self._pending_input.direction = 0
                return "Stopped."
            
        @mcp.tool
        def shoot(weapon: int = 1) -> str:
            """Fire current or specified weapon.
            
            Args:
                weapon: 0=hammer, 1=pistol, 2=shotgun, 3=grenade, 4=laser
            """
            self._fire_counter += 1
            self._pending_input.fire = self._fire_counter
            self._pending_input.wanted_weapon = weapon
            return f"Fired {WEAPONS[weapon]['name']}!"
            
        @mcp.tool
        def get_status() -> str:
            """Get current game state as natural language."""
            agent = self._agents[self._current_agent_id]
            
            status = f"You are Player {agent.agent_id} at position ({agent.x:.0f}, {agent.y:.0f}).\n"
            status += f"Health: {agent.health}/10, Armor: {agent.armor}/10\n"
            status += f"Weapon: {WEAPONS[agent.weapon]['name']}\n"
            
            # Add visible enemies
            for other in self._get_visible_agents(agent):
                dist = self._distance(agent, other)
                direction = "right" if other.x > agent.x else "left"
                status += f"  - Player {other.agent_id}: {dist:.0f} units to your {direction}\n"
            
            return status
```

This means an LLM can play by calling tools like:

| Tool Call | Response |
|-----------|----------|
| `get_status()` | "You are at (432, 210). Health: 8/10. Enemy 150 units to your right." |
| `move("right")` | "Moving right." |
| `shoot(weapon=0)` | "Swung hammer!" |
| `jump()` | "Jumped!" |

## Layer 4: How the Notebook Uses OpenEnv

The training notebook (`notebooks/teeunit_training.ipynb`) leverages OpenEnv in four key ways:

### 1. Environment Initialization

```python
from teeunit_env.server.tee_environment import TeeEnvironment

env = TeeEnvironment(num_agents=4, max_steps=200)
obs = env.reset()
print(f"Episode ID: {obs.metadata['episode_id']}")
```

### 2. Generating Training Data via MCP Tools

```python
def generate_game_state() -> str:
    """Generate a random game state for training."""
    env = TeeEnvironment(num_agents=4, max_steps=50)
    env.reset()
    
    # Simulate a few random ticks
    for _ in range(random.randint(1, 10)):
        env._simulate_tick()
    
    # Get state via MCP tool (returns natural language)
    return env.call_tool_sync('get_status')
```

The `call_tool_sync('get_status')` returns text like:

```
You are Player 0 at position (432, 210).
Health: 8/10, Armor: 0/10
Weapon: pistol (10 ammo)
Enemies visible:
  - Player 1: 150 units to your right, health 6/10
  - Player 2: 300 units to your left, health 10/10
```

### 3. GRPO Training with Environment Rewards

```python
def valid_action_reward(completions, **kwargs) -> list[float]:
    """Reward for outputting a valid, parseable action.
    
    +0.5 for valid action
    -0.5 for invalid/unparseable action
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"]
        action = parse_action(text)
        if action is not None:
            rewards.append(0.5)
        else:
            rewards.append(-0.5)
    return rewards

def game_outcome_reward(completions, **kwargs) -> list[float]:
    """Execute action in environment, return actual game reward."""
    env = get_reward_env()
    rewards = []
    for completion in completions:
        action = parse_action(completion[0]["content"])
        if action:
            # Execute via MCP tool
            result = env.call_tool_sync(action['tool'], **action['args'])
            # Get reward from environment
            rewards.append(env.get_last_reward())
        else:
            rewards.append(-1.0)
    return rewards
```

### 4. Evaluation Loop

```python
def play_episode(model, tokenizer, max_steps=20):
    """Play one episode with the trained model."""
    env = TeeEnvironment(num_agents=4, max_steps=max_steps)
    env.reset()
    
    total_reward = 0
    
    for step in range(max_steps):
        # Get game state via OpenEnv MCP tool
        state = env.call_tool_sync('get_status')
        
        # Format prompt for LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_prompt(state)},
        ]
        
        # LLM generates action
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
        outputs = model.generate(inputs, max_new_tokens=50)
        action_text = tokenizer.decode(outputs[0])
        
        # Parse and execute via OpenEnv MCP tool
        action = parse_action(action_text)
        if action:
            result = env.call_tool_sync(action['tool'], **action['args'])
            total_reward += env.get_last_reward()
    
    return total_reward
```

## Why OpenEnv + MCP for LLM Training?

| Approach | Interface | LLM-Friendly |
|----------|-----------|--------------|
| Gym/Gymnasium | Numerical tensors `[0.5, 0.2, 1.0, ...]` | No |
| OpenEnv + MCP | Natural language tools | Yes |

Traditional RL environments return numerical observations that LLMs can't naturally process. OpenEnv's MCP interface returns **text descriptions** and accepts **tool calls**, which LLMs handle natively:

```
Observation: "Enemy is 150 units to your right at low health."
Action: call_tool("shoot", weapon=1)
Result: "Fired pistol! Hit enemy for 2 damage."
```

This enables training LLMs on game tasks using standard language model fine-tuning techniques (GRPO, DPO, PPO) rather than requiring special RL architectures.

## Key Files

| File | Purpose |
|------|---------|
| `openenv.yaml` | Environment manifest for OpenEnv discovery |
| `teeunit/openenv_models.py` | Pydantic models (Action, Observation, State) |
| `teeunit/openenv_environment.py` | Main environment class for RL training |
| `teeunit_env/server/tee_environment.py` | MCP tool interface for LLM interaction |
| `teeunit_env/server/app.py` | FastAPI server exposing OpenEnv API |
| `notebooks/teeunit_training.ipynb` | LLM fine-tuning with Unsloth + GRPO |

## Further Reading

- [OpenEnv GitHub](https://github.com/meta-pytorch/OpenEnv)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)
- [Unsloth](https://github.com/unslothai/unsloth) - Memory-efficient LLM training
