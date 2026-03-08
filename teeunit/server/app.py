"""
TeeUnit FastAPI Server

HTTP/WebSocket server implementing the OpenEnv API.
Wraps a real Teeworlds server via BotManager.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .tee_environment import TeeEnvironment
from ..models import GameConfig, TeeInput

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
        env.disconnect()
        logger.info("TeeUnit environment disconnected")


app = FastAPI(
    title="TeeUnit Environment",
    description="Multi-agent arena environment wrapping real Teeworlds",
    version="0.2.0",
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


# Request/Response Models
class ResetRequest(BaseModel):
    """Request body for reset endpoint."""
    config: Optional[Dict] = None


class StepRequest(BaseModel):
    """Request body for step endpoint."""
    agent_id: int
    direction: int = 0
    target_x: int = 0
    target_y: int = 0
    jump: bool = False
    fire: bool = False
    hook: bool = False
    wanted_weapon: int = 0


class StepAllRequest(BaseModel):
    """Request body for step_all endpoint."""
    actions: Dict[str, Dict]  # agent_id (str) -> action dict


class ObservationRequest(BaseModel):
    """Request body for observation endpoint."""
    agent_id: int


# HTTP Endpoints
@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "TeeUnit Environment",
        "version": "0.2.0",
        "description": "Multi-agent arena environment wrapping real Teeworlds",
        "endpoints": {
            "GET /health": "Health check",
            "POST /reset": "Reset environment (connects to Teeworlds server)",
            "POST /step": "Execute action for one agent",
            "POST /step_all": "Execute actions for all agents simultaneously",
            "GET /state": "Get episode state",
            "POST /observation": "Get agent observation",
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
    }


@app.post("/reset")
async def reset(request: ResetRequest = None):
    """
    Reset the environment to start a new match.
    
    Connects to Teeworlds server if not already connected.
    Returns initial observation for agent 0.
    """
    global env
    if env is None:
        env = TeeEnvironment(auto_connect=False)
    
    config = request.config if request else None
    result = env.reset(config)
    
    return {
        "observation": result.observation.to_dict(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
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
    
    action = TeeInput(
        direction=request.direction,
        target_x=request.target_x,
        target_y=request.target_y,
        jump=request.jump,
        fire=request.fire,
        hook=request.hook,
        wanted_weapon=request.wanted_weapon,
    )
    
    result = env.step(request.agent_id, action)
    
    return {
        "observation": result.observation.to_dict(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


@app.post("/step_all")
async def step_all(request: StepAllRequest):
    """
    Execute actions for all agents simultaneously.
    
    Returns observations and rewards for all agents.
    """
    global env
    if env is None or not env._connected:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    try:
        actions = {}
        for agent_id_str, action_dict in request.actions.items():
            agent_id = int(agent_id_str)
            actions[agent_id] = TeeInput(
                direction=action_dict.get("direction", 0),
                target_x=action_dict.get("target_x", 0),
                target_y=action_dict.get("target_y", 0),
                jump=action_dict.get("jump", False),
                fire=action_dict.get("fire", False),
                hook=action_dict.get("hook", False),
                wanted_weapon=action_dict.get("wanted_weapon", 0),
            )
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid action: {e}")
    
    results = env.step_all(actions)
    
    return {
        str(agent_id): {
            "observation": result.observation.to_dict(),
            "reward": result.reward,
            "done": result.done,
            "info": result.info,
        }
        for agent_id, result in results.items()
    }


@app.get("/state")
async def state():
    """Get current episode state."""
    global env
    if env is None or not env._connected:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    return env.state().to_dict()


@app.post("/observation")
async def observation(request: ObservationRequest):
    """Get observation for a specific agent."""
    global env
    if env is None or not env._connected:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    obs = env.get_observation(request.agent_id)
    return obs.to_dict()


# WebSocket endpoint for real-time communication
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time environment interaction.
    
    Protocol:
    - Send: {"type": "reset", "config": {...}} -> Reset environment
    - Send: {"type": "step", "agent_id": N, "action": {...}} -> Execute action
    - Send: {"type": "step_all", "actions": {...}} -> Execute all actions
    - Send: {"type": "state"} -> Get state
    - Send: {"type": "observation", "agent_id": N} -> Get observation
    - Receive: {"type": "result", "data": {...}}
    """
    global env
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")
            
            if msg_type == "reset":
                if env is None:
                    env = TeeEnvironment(auto_connect=False)
                config = data.get("config")
                result = env.reset(config)
                await websocket.send_json({
                    "type": "result",
                    "action": "reset",
                    "data": {
                        "observation": result.observation.to_dict(),
                        "reward": result.reward,
                        "done": result.done,
                        "info": result.info,
                    },
                })
            
            elif msg_type == "step":
                if env is None or not env._connected:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Environment not initialized. Call reset first.",
                    })
                    continue
                
                agent_id = data.get("agent_id", 0)
                action_data = data.get("action", {})
                action = TeeInput(
                    direction=action_data.get("direction", 0),
                    target_x=action_data.get("target_x", 0),
                    target_y=action_data.get("target_y", 0),
                    jump=action_data.get("jump", False),
                    fire=action_data.get("fire", False),
                    hook=action_data.get("hook", False),
                    wanted_weapon=action_data.get("wanted_weapon", 0),
                )
                result = env.step(agent_id, action)
                await websocket.send_json({
                    "type": "result",
                    "action": "step",
                    "data": {
                        "observation": result.observation.to_dict(),
                        "reward": result.reward,
                        "done": result.done,
                        "info": result.info,
                    },
                })
            
            elif msg_type == "step_all":
                if env is None or not env._connected:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Environment not initialized. Call reset first.",
                    })
                    continue
                
                try:
                    actions = {}
                    for agent_id_str, action_dict in data.get("actions", {}).items():
                        agent_id = int(agent_id_str)
                        actions[agent_id] = TeeInput(
                            direction=action_dict.get("direction", 0),
                            target_x=action_dict.get("target_x", 0),
                            target_y=action_dict.get("target_y", 0),
                            jump=action_dict.get("jump", False),
                            fire=action_dict.get("fire", False),
                            hook=action_dict.get("hook", False),
                            wanted_weapon=action_dict.get("wanted_weapon", 0),
                        )
                    
                    results = env.step_all(actions)
                    await websocket.send_json({
                        "type": "result",
                        "action": "step_all",
                        "data": {
                            str(agent_id): {
                                "observation": result.observation.to_dict(),
                                "reward": result.reward,
                                "done": result.done,
                                "info": result.info,
                            }
                            for agent_id, result in results.items()
                        },
                    })
                except (ValueError, KeyError) as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
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
                    "data": env.state().to_dict(),
                })
            
            elif msg_type == "observation":
                if env is None or not env._connected:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Environment not initialized. Call reset first.",
                    })
                    continue
                
                agent_id = data.get("agent_id", 0)
                obs = env.get_observation(agent_id)
                await websocket.send_json({
                    "type": "result",
                    "action": "observation",
                    "data": obs.to_dict(),
                })
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                })
    
    except WebSocketDisconnect:
        pass


# Run with: uvicorn teeunit.server.app:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
