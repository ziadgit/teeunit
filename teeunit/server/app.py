"""
TeeUnit FastAPI Server

HTTP/WebSocket server implementing the OpenEnv API.
"""

from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .tee_environment import TeeEnvironment
from ..models import GameConfig, TeeAction


# Global environment instance
env: Optional[TeeEnvironment] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global env
    env = TeeEnvironment()
    yield
    # Cleanup if needed


app = FastAPI(
    title="TeeUnit Environment",
    description="Multi-agent arena environment for LLM training",
    version="0.1.0",
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
    action_type: str
    direction: Optional[str] = None
    target_x: Optional[int] = None
    target_y: Optional[int] = None
    weapon: Optional[str] = None


class StepAllRequest(BaseModel):
    """Request body for step_all endpoint."""
    actions: Dict[int, Dict]  # agent_id -> action dict


class ObservationRequest(BaseModel):
    """Request body for observation endpoint."""
    agent_id: int


# HTTP Endpoints
@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "TeeUnit Environment",
        "version": "0.1.0",
        "description": "Multi-agent arena environment for LLM training",
        "endpoints": {
            "GET /health": "Health check",
            "POST /reset": "Reset environment",
            "POST /step": "Execute action",
            "POST /step_all": "Execute all actions simultaneously",
            "GET /state": "Get episode state",
            "POST /observation": "Get agent observation",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "environment": "teeunit"}


@app.post("/reset")
async def reset(request: ResetRequest = None):
    """
    Reset the environment to start a new match.
    
    Returns initial observation for agent 0.
    """
    global env
    if env is None:
        env = TeeEnvironment()
    
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
    Execute an action for an agent.
    
    Returns the agent's new observation and reward.
    """
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    try:
        action = TeeAction(
            agent_id=request.agent_id,
            action_type=request.action_type,
            direction=request.direction,
            target_x=request.target_x,
            target_y=request.target_y,
            weapon=request.weapon,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    result = env.step(action)
    
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
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    try:
        actions = {}
        for agent_id_str, action_dict in request.actions.items():
            agent_id = int(agent_id_str)
            actions[agent_id] = TeeAction(
                agent_id=agent_id,
                action_type=action_dict["action_type"],
                direction=action_dict.get("direction"),
                target_x=action_dict.get("target_x"),
                target_y=action_dict.get("target_y"),
                weapon=action_dict.get("weapon"),
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
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    return env.state().to_dict()


@app.post("/observation")
async def observation(request: ObservationRequest):
    """Get observation for a specific agent."""
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    obs = env.get_observation(request.agent_id)
    return obs.to_dict()


@app.get("/arena")
async def arena():
    """Get ASCII representation of the arena."""
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    agent_positions = env.agents.get_agent_positions()
    ascii_map = env.arena.to_ascii(agent_positions)
    
    return {
        "map": ascii_map,
        "width": env.arena.width,
        "height": env.arena.height,
        "agents": {aid: list(pos) for aid, pos in agent_positions.items()},
    }


# WebSocket endpoint for real-time communication
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time environment interaction.
    
    Protocol:
    - Send: {"type": "reset", "config": {...}} -> Reset environment
    - Send: {"type": "step", "action": {...}} -> Execute action
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
                    env = TeeEnvironment()
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
                if env is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Environment not initialized",
                    })
                    continue
                
                action_data = data.get("action", {})
                try:
                    action = TeeAction(
                        agent_id=action_data.get("agent_id", 0),
                        action_type=action_data.get("action_type", "wait"),
                        direction=action_data.get("direction"),
                        target_x=action_data.get("target_x"),
                        target_y=action_data.get("target_y"),
                        weapon=action_data.get("weapon"),
                    )
                    result = env.step(action)
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
                except ValueError as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                    })
            
            elif msg_type == "step_all":
                if env is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Environment not initialized",
                    })
                    continue
                
                try:
                    actions = {}
                    for agent_id_str, action_dict in data.get("actions", {}).items():
                        agent_id = int(agent_id_str)
                        actions[agent_id] = TeeAction(
                            agent_id=agent_id,
                            action_type=action_dict.get("action_type", "wait"),
                            direction=action_dict.get("direction"),
                            target_x=action_dict.get("target_x"),
                            target_y=action_dict.get("target_y"),
                            weapon=action_dict.get("weapon"),
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
                if env is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Environment not initialized",
                    })
                    continue
                
                await websocket.send_json({
                    "type": "result",
                    "action": "state",
                    "data": env.state().to_dict(),
                })
            
            elif msg_type == "observation":
                if env is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Environment not initialized",
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
