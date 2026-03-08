"""
TeeUnit FastAPI Server

HTTP/WebSocket server implementing the OpenEnv API.
Wraps a real Teeworlds server via BotManager.

This server exposes the TeeEnvironment over HTTP and WebSocket,
allowing remote RL training and multi-agent coordination.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..openenv_models import (
    TeeAction,
    TeeMultiAction,
    TeeObservation,
    TeeMultiObservation,
    TeeState,
    TeeStepResult,
    TeeMultiStepResult,
    RewardConfig,
)
from ..openenv_environment import TeeEnvironment, TeeConfig

logger = logging.getLogger(__name__)

# Global environment instance
env: Optional[TeeEnvironment] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global env
    # Create environment but don't auto-connect (wait for reset)
    env = TeeEnvironment(auto_connect=False)
    logger.info("TeeUnit environment created")
    yield
    # Cleanup
    if env:
        env.close()
        logger.info("TeeUnit environment disconnected")


app = FastAPI(
    title="TeeUnit Environment",
    description="OpenEnv-compatible multi-agent arena environment wrapping real Teeworlds",
    version="0.3.0",
    lifespan=lifespan,
)

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request/Response Models
# =============================================================================

class ResetRequest(BaseModel):
    """Request body for reset endpoint."""
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class StepRequest(BaseModel):
    """Request body for step endpoint (single agent)."""
    agent_id: int = Field(ge=0, le=7, description="Agent ID (0-7)")
    direction: int = Field(default=0, ge=-1, le=1, description="Movement direction")
    target_x: int = Field(default=0, description="Aim X position")
    target_y: int = Field(default=0, description="Aim Y position")
    jump: bool = Field(default=False, description="Whether to jump")
    fire: bool = Field(default=False, description="Whether to fire")
    hook: bool = Field(default=False, description="Whether to hook")
    weapon: int = Field(default=0, ge=0, le=5, description="Weapon to switch to")
    
    def to_action(self) -> TeeAction:
        """Convert request to TeeAction."""
        return TeeAction(
            agent_id=self.agent_id,
            direction=self.direction,
            target_x=self.target_x,
            target_y=self.target_y,
            jump=self.jump,
            fire=self.fire,
            hook=self.hook,
            weapon=self.weapon,
        )


class DiscreteStepRequest(BaseModel):
    """Request for step with discrete action index."""
    agent_id: int = Field(ge=0, le=7, description="Agent ID")
    action: int = Field(ge=0, le=17, description="Discrete action index (0-17)")
    target_x: int = Field(default=100, description="Aim X position")
    target_y: int = Field(default=0, description="Aim Y position")
    
    def to_action(self) -> TeeAction:
        """Convert discrete action to TeeAction."""
        return TeeAction.from_discrete_action(
            self.action,
            agent_id=self.agent_id,
            target_x=self.target_x,
            target_y=self.target_y,
        )


class StepAllRequest(BaseModel):
    """Request body for step_all endpoint (all agents)."""
    actions: Dict[str, Dict[str, Any]] = Field(
        description="Map of agent_id (str) to action dict"
    )
    
    def to_multi_action(self) -> TeeMultiAction:
        """Convert request to TeeMultiAction."""
        agent_actions = {}
        for agent_id_str, action_dict in self.actions.items():
            agent_id = int(agent_id_str)
            agent_actions[agent_id] = TeeAction(
                agent_id=agent_id,
                direction=action_dict.get("direction", 0),
                target_x=action_dict.get("target_x", 0),
                target_y=action_dict.get("target_y", 0),
                jump=action_dict.get("jump", False),
                fire=action_dict.get("fire", False),
                hook=action_dict.get("hook", False),
                weapon=action_dict.get("weapon", 0),
            )
        return TeeMultiAction(actions=agent_actions)


class DiscreteStepAllRequest(BaseModel):
    """Request for step_all with discrete action indices."""
    actions: Dict[str, int] = Field(
        description="Map of agent_id (str) to discrete action (0-17)"
    )
    target_x: int = Field(default=100, description="Default aim X")
    target_y: int = Field(default=0, description="Default aim Y")
    
    def to_multi_action(self) -> TeeMultiAction:
        """Convert discrete actions to TeeMultiAction."""
        agent_actions = {}
        for agent_id_str, action_idx in self.actions.items():
            agent_id = int(agent_id_str)
            agent_actions[agent_id] = TeeAction.from_discrete_action(
                action_idx,
                agent_id=agent_id,
                target_x=self.target_x,
                target_y=self.target_y,
            )
        return TeeMultiAction(actions=agent_actions)


class ObservationRequest(BaseModel):
    """Request body for observation endpoint."""
    agent_id: int = Field(ge=0, le=7, description="Agent ID")
    as_tensor: bool = Field(default=False, description="Return as tensor array")


# =============================================================================
# HTTP Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "TeeUnit Environment",
        "version": "0.3.0",
        "description": "OpenEnv-compatible multi-agent arena wrapping Teeworlds",
        "openenv_compatible": True,
        "endpoints": {
            "GET /health": "Health check",
            "POST /reset": "Reset environment (connects to Teeworlds server)",
            "POST /step": "Execute action for one agent",
            "POST /step_discrete": "Execute discrete action for one agent",
            "POST /step_all": "Execute actions for all agents",
            "POST /step_all_discrete": "Execute discrete actions for all agents",
            "GET /state": "Get episode state",
            "POST /observation": "Get agent observation",
            "GET /tensor_shape": "Get observation tensor shape",
            "GET /action_space": "Get discrete action space info",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    global env
    connected = env._connected if env else False
    return {
        "status": "healthy",
        "environment": "teeunit",
        "connected": connected,
        "num_agents": env.config.num_agents if env else 0,
        "openenv_version": "0.3.0",
    }


@app.get("/tensor_shape")
async def tensor_shape():
    """Get the observation tensor shape for RL."""
    return {
        "observation_shape": list(TeeObservation.tensor_shape()),
        "action_space_size": 18,
        "description": "Observation is 195-dim vector; action is discrete 0-17",
    }


@app.get("/action_space")
async def action_space():
    """Get discrete action space information."""
    actions = []
    for i in range(18):
        action = TeeAction.from_discrete_action(i)
        actions.append({
            "index": i,
            "direction": action.direction,
            "jump": action.jump,
            "fire": action.fire,
            "hook": action.hook,
        })
    return {
        "type": "discrete",
        "size": 18,
        "actions": actions,
    }


@app.post("/reset")
async def reset(request: ResetRequest = None):
    """
    Reset the environment to start a new match.
    
    Connects to Teeworlds server if not already connected.
    Returns initial observations for all agents.
    """
    global env
    if env is None:
        env = TeeEnvironment(auto_connect=False)
    
    seed = request.seed if request else None
    episode_id = request.episode_id if request else None
    
    multi_obs = env.reset(seed=seed, episode_id=episode_id)
    
    return {
        "observations": {
            str(agent_id): obs.model_dump()
            for agent_id, obs in multi_obs.observations.items()
        },
        "done": multi_obs.done,
        "metadata": multi_obs.metadata,
        "state": env.state.model_dump(),
    }


@app.post("/step")
async def step(request: StepRequest):
    """
    Execute an action for a single agent.
    
    Returns the agent's new observation and reward.
    """
    global env
    if env is None or not env._connected:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    action = request.to_action()
    observation = env.step(action)
    
    return {
        "observation": observation.model_dump(),
        "reward": observation.reward,
        "done": observation.done,
    }


@app.post("/step_discrete")
async def step_discrete(request: DiscreteStepRequest):
    """
    Execute a discrete action for a single agent.
    
    Discrete actions 0-17 map to movement/jump/fire/hook combinations.
    """
    global env
    if env is None or not env._connected:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    action = request.to_action()
    observation = env.step(action)
    
    return {
        "observation": observation.model_dump(),
        "reward": observation.reward,
        "done": observation.done,
        "action_decoded": {
            "direction": action.direction,
            "jump": action.jump,
            "fire": action.fire,
            "hook": action.hook,
        },
    }


@app.post("/step_all")
async def step_all(request: StepAllRequest):
    """
    Execute actions for all agents simultaneously.
    
    Returns observations and rewards for all agents.
    This is the main endpoint for self-play RL training.
    """
    global env
    if env is None or not env._connected:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    try:
        multi_action = request.to_multi_action()
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid action: {e}")
    
    results = env.step_all(multi_action)
    
    return {
        "results": {
            str(agent_id): {
                "observation": result.observation.model_dump(),
                "reward": result.reward,
                "done": result.done,
                "truncated": result.truncated,
                "info": result.info,
            }
            for agent_id, result in results.results.items()
        },
        "state": results.state.model_dump(),
    }


@app.post("/step_all_discrete")
async def step_all_discrete(request: DiscreteStepAllRequest):
    """
    Execute discrete actions for all agents simultaneously.
    
    Each agent's action is a discrete index 0-17.
    """
    global env
    if env is None or not env._connected:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    try:
        multi_action = request.to_multi_action()
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid action: {e}")
    
    results = env.step_all(multi_action)
    
    # Include decoded actions
    decoded_actions = {}
    for agent_id, action in multi_action.actions.items():
        decoded_actions[str(agent_id)] = {
            "direction": action.direction,
            "jump": action.jump,
            "fire": action.fire,
            "hook": action.hook,
        }
    
    return {
        "results": {
            str(agent_id): {
                "observation": result.observation.model_dump(),
                "reward": result.reward,
                "done": result.done,
                "truncated": result.truncated,
                "info": result.info,
            }
            for agent_id, result in results.results.items()
        },
        "state": results.state.model_dump(),
        "actions_decoded": decoded_actions,
    }


@app.get("/state")
async def state():
    """Get current episode state."""
    global env
    if env is None or not env._connected:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    return env.state.model_dump()


@app.post("/observation")
async def observation(request: ObservationRequest):
    """Get observation for a specific agent."""
    global env
    if env is None or not env._connected:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    obs = env._build_observation(request.agent_id)
    
    if request.as_tensor:
        tensor = obs.to_tensor()
        return {
            "tensor": tensor.tolist(),
            "shape": list(tensor.shape),
        }
    
    return obs.model_dump()


# =============================================================================
# Tensor Endpoints for RL Training
# =============================================================================

@app.post("/step_all_tensor")
async def step_all_tensor(request: DiscreteStepAllRequest):
    """
    Execute discrete actions and return tensor observations.
    
    Optimized endpoint for RL training - returns numpy-compatible arrays.
    """
    global env
    if env is None or not env._connected:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    try:
        multi_action = request.to_multi_action()
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid action: {e}")
    
    results = env.step_all(multi_action)
    arrays = results.to_arrays(normalize=True)
    
    return {
        "observations": arrays['observations'].tolist(),
        "rewards": arrays['rewards'].tolist(),
        "dones": arrays['dones'].tolist(),
        "truncateds": arrays['truncateds'].tolist(),
        "state": results.state.model_dump(),
    }


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time environment interaction.
    
    Protocol:
    - Send: {"type": "reset", "seed": N, "episode_id": "..."} -> Reset
    - Send: {"type": "step", "agent_id": N, "action": {...}} -> Execute action
    - Send: {"type": "step_discrete", "agent_id": N, "action": N} -> Discrete action
    - Send: {"type": "step_all", "actions": {...}} -> All agents
    - Send: {"type": "step_all_discrete", "actions": {...}} -> All discrete
    - Send: {"type": "state"} -> Get state
    - Send: {"type": "observation", "agent_id": N} -> Get observation
    - Receive: {"type": "result", "data": {...}}
    - Receive: {"type": "error", "message": "..."}
    """
    global env
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")
            
            try:
                if msg_type == "reset":
                    if env is None:
                        env = TeeEnvironment(auto_connect=False)
                    
                    multi_obs = env.reset(
                        seed=data.get("seed"),
                        episode_id=data.get("episode_id"),
                    )
                    
                    await websocket.send_json({
                        "type": "result",
                        "action": "reset",
                        "data": {
                            "observations": {
                                str(k): v.model_dump() 
                                for k, v in multi_obs.observations.items()
                            },
                            "done": multi_obs.done,
                            "state": env.state.model_dump(),
                        },
                    })
                
                elif msg_type == "step":
                    if env is None or not env._connected:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Environment not initialized. Call reset first.",
                        })
                        continue
                    
                    action_data = data.get("action", {})
                    action = TeeAction(
                        agent_id=data.get("agent_id", 0),
                        direction=action_data.get("direction", 0),
                        target_x=action_data.get("target_x", 0),
                        target_y=action_data.get("target_y", 0),
                        jump=action_data.get("jump", False),
                        fire=action_data.get("fire", False),
                        hook=action_data.get("hook", False),
                        weapon=action_data.get("weapon", 0),
                    )
                    
                    obs = env.step(action)
                    
                    await websocket.send_json({
                        "type": "result",
                        "action": "step",
                        "data": {
                            "observation": obs.model_dump(),
                            "reward": obs.reward,
                            "done": obs.done,
                        },
                    })
                
                elif msg_type == "step_discrete":
                    if env is None or not env._connected:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Environment not initialized. Call reset first.",
                        })
                        continue
                    
                    action = TeeAction.from_discrete_action(
                        data.get("action", 0),
                        agent_id=data.get("agent_id", 0),
                        target_x=data.get("target_x", 100),
                        target_y=data.get("target_y", 0),
                    )
                    
                    obs = env.step(action)
                    
                    await websocket.send_json({
                        "type": "result",
                        "action": "step_discrete",
                        "data": {
                            "observation": obs.model_dump(),
                            "reward": obs.reward,
                            "done": obs.done,
                        },
                    })
                
                elif msg_type == "step_all":
                    if env is None or not env._connected:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Environment not initialized. Call reset first.",
                        })
                        continue
                    
                    agent_actions = {}
                    for agent_id_str, action_dict in data.get("actions", {}).items():
                        agent_id = int(agent_id_str)
                        agent_actions[agent_id] = TeeAction(
                            agent_id=agent_id,
                            direction=action_dict.get("direction", 0),
                            target_x=action_dict.get("target_x", 0),
                            target_y=action_dict.get("target_y", 0),
                            jump=action_dict.get("jump", False),
                            fire=action_dict.get("fire", False),
                            hook=action_dict.get("hook", False),
                            weapon=action_dict.get("weapon", 0),
                        )
                    
                    results = env.step_all(TeeMultiAction(actions=agent_actions))
                    
                    await websocket.send_json({
                        "type": "result",
                        "action": "step_all",
                        "data": {
                            "results": {
                                str(k): {
                                    "observation": v.observation.model_dump(),
                                    "reward": v.reward,
                                    "done": v.done,
                                    "truncated": v.truncated,
                                    "info": v.info,
                                }
                                for k, v in results.results.items()
                            },
                            "state": results.state.model_dump(),
                        },
                    })
                
                elif msg_type == "step_all_discrete":
                    if env is None or not env._connected:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Environment not initialized. Call reset first.",
                        })
                        continue
                    
                    agent_actions = {}
                    for agent_id_str, action_idx in data.get("actions", {}).items():
                        agent_id = int(agent_id_str)
                        agent_actions[agent_id] = TeeAction.from_discrete_action(
                            action_idx,
                            agent_id=agent_id,
                            target_x=data.get("target_x", 100),
                            target_y=data.get("target_y", 0),
                        )
                    
                    results = env.step_all(TeeMultiAction(actions=agent_actions))
                    
                    await websocket.send_json({
                        "type": "result",
                        "action": "step_all_discrete",
                        "data": {
                            "results": {
                                str(k): {
                                    "observation": v.observation.model_dump(),
                                    "reward": v.reward,
                                    "done": v.done,
                                    "truncated": v.truncated,
                                    "info": v.info,
                                }
                                for k, v in results.results.items()
                            },
                            "state": results.state.model_dump(),
                        },
                    })
                
                elif msg_type == "state":
                    if env is None or not env._connected:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Environment not initialized. Call reset first.",
                        })
                        continue
                    
                    await websocket.send_json({
                        "type": "result",
                        "action": "state",
                        "data": env.state.model_dump(),
                    })
                
                elif msg_type == "observation":
                    if env is None or not env._connected:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Environment not initialized. Call reset first.",
                        })
                        continue
                    
                    agent_id = data.get("agent_id", 0)
                    obs = env._build_observation(agent_id)
                    
                    if data.get("as_tensor", False):
                        tensor = obs.to_tensor()
                        await websocket.send_json({
                            "type": "result",
                            "action": "observation",
                            "data": {
                                "tensor": tensor.tolist(),
                                "shape": list(tensor.shape),
                            },
                        })
                    else:
                        await websocket.send_json({
                            "type": "result",
                            "action": "observation",
                            "data": obs.model_dump(),
                        })
                
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                    })
            
            except Exception as e:
                logger.exception(f"Error handling WebSocket message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                })
    
    except WebSocketDisconnect:
        pass


# Run with: uvicorn teeunit.server.app:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
