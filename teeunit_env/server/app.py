# Copyright (c) 2024 TeeUnit Project
# SPDX-License-Identifier: MIT

"""
FastAPI application for the TeeUnit Environment.

This module creates an HTTP server that exposes the TeeEnvironment
over HTTP and WebSocket endpoints, compatible with MCPToolClient.

Usage:
    # Development (with auto-reload):
    uvicorn teeunit_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn teeunit_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m teeunit_env.server.app
"""

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    
    from .tee_environment import TeeEnvironment
    
    # Create the app with web interface
    # Pass the class (factory) instead of an instance for WebSocket session support
    # Use MCP types for action/observation since this is a pure MCP environment
    app = create_app(
        TeeEnvironment, CallToolAction, CallToolObservation, env_name="teeunit_env"
    )

except ImportError:
    # Fallback: Create a simple FastAPI app for development/testing
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import Optional, Dict, Any
    import json
    
    # Import our environment
    try:
        from .tee_environment import TeeEnvironment
    except ImportError:
        from tee_environment import TeeEnvironment
    
    app = FastAPI(
        title="TeeUnit OpenEnv",
        description="OpenEnv-compatible Teeworlds arena environment for LLM RL training",
        version="0.1.0",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Store environment instances per session
    _environments: Dict[str, TeeEnvironment] = {}
    
    class ResetRequest(BaseModel):
        seed: Optional[int] = None
        episode_id: Optional[str] = None
    
    class ResetResponse(BaseModel):
        status: str
        episode_id: str
        message: str
    
    class ToolCallRequest(BaseModel):
        tool_name: str
        arguments: Dict[str, Any] = {}
    
    class ToolCallResponse(BaseModel):
        result: Any
        reward: float
        done: bool
        metadata: Dict[str, Any] = {}
    
    class ToolInfo(BaseModel):
        name: str
        description: str
        parameters: Dict[str, Any] = {}
    
    @app.get("/")
    async def root():
        """Root endpoint with environment info."""
        return {
            "name": "TeeUnit OpenEnv",
            "version": "0.1.0",
            "description": "OpenEnv-compatible Teeworlds arena for LLM training",
            "endpoints": {
                "reset": "POST /reset",
                "tools": "GET /tools",
                "call_tool": "POST /call_tool",
                "websocket": "WS /ws",
            }
        }
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    @app.post("/reset", response_model=ResetResponse)
    async def reset(request: ResetRequest):
        """Reset the environment for a new episode."""
        env = TeeEnvironment()
        obs = env.reset(seed=request.seed, episode_id=request.episode_id)
        
        session_id = obs.metadata.get("episode_id", "default")
        _environments[session_id] = env
        
        return ResetResponse(
            status="ready",
            episode_id=session_id,
            message=obs.metadata.get("message", "Environment ready"),
        )
    
    @app.get("/tools")
    async def list_tools():
        """List available MCP tools."""
        return {
            "tools": [
                {
                    "name": "move",
                    "description": "Move the tee left, right, or none",
                    "parameters": {"direction": {"type": "string", "enum": ["left", "right", "none"]}},
                },
                {
                    "name": "jump",
                    "description": "Make the tee jump",
                    "parameters": {},
                },
                {
                    "name": "aim",
                    "description": "Aim at target coordinates",
                    "parameters": {
                        "x": {"type": "integer", "description": "Target X coordinate"},
                        "y": {"type": "integer", "description": "Target Y coordinate"},
                    },
                },
                {
                    "name": "shoot",
                    "description": "Fire the specified weapon",
                    "parameters": {
                        "weapon": {"type": "integer", "description": "Weapon ID (0-5) or -1 for current", "default": -1},
                    },
                },
                {
                    "name": "hook",
                    "description": "Use the grappling hook",
                    "parameters": {},
                },
                {
                    "name": "get_status",
                    "description": "Get current game state as text",
                    "parameters": {},
                },
            ]
        }
    
    @app.post("/call_tool", response_model=ToolCallResponse)
    async def call_tool(request: ToolCallRequest, session_id: str = "default"):
        """Call an MCP tool."""
        env = _environments.get(session_id)
        if env is None:
            env = TeeEnvironment()
            env.reset()
            _environments[session_id] = env
        
        # Get the MCP server from environment
        mcp = env._mcp
        
        # Call the tool
        tool_name = request.tool_name
        arguments = request.arguments
        
        try:
            # Use FastMCP's async call_tool method
            tool_result = await mcp.call_tool(tool_name, arguments)
            
            # Extract text result from ToolResult
            if tool_result and tool_result.content:
                result = tool_result.content[0].text if hasattr(tool_result.content[0], 'text') else str(tool_result.content[0])
            else:
                result = str(tool_result)
            
            # Simulate tick and get reward
            env._simulate_tick()
            reward = env._calculate_reward()
            
            # Check done
            done = env._state.step_count >= env._max_steps
            player = env._agents.get(env._current_agent_id)
            if player and not player.is_alive:
                done = True
            
            return ToolCallResponse(
                result=result,
                reward=reward,
                done=done,
                metadata={
                    "step": env._state.step_count,
                    "tick": env._tick,
                },
            )
        except Exception as e:
            return ToolCallResponse(
                result=f"Error: {str(e)}",
                reward=0.0,
                done=False,
                metadata={"error": str(e)},
            )
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time interaction."""
        await websocket.accept()
        
        env = TeeEnvironment()
        env.reset()
        
        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                action_type = message.get("type", "call_tool")
                
                if action_type == "reset":
                    obs = env.reset(
                        seed=message.get("seed"),
                        episode_id=message.get("episode_id"),
                    )
                    await websocket.send_json({
                        "type": "reset",
                        "status": "ready",
                        "episode_id": obs.metadata.get("episode_id"),
                        "message": obs.metadata.get("message"),
                    })
                
                elif action_type == "list_tools":
                    await websocket.send_json({
                        "type": "tools",
                        "tools": [
                            {"name": "move", "description": "Move left/right/none"},
                            {"name": "jump", "description": "Jump"},
                            {"name": "aim", "description": "Aim at x,y"},
                            {"name": "shoot", "description": "Fire weapon"},
                            {"name": "hook", "description": "Use hook"},
                            {"name": "get_status", "description": "Get game state"},
                        ],
                    })
                
                elif action_type == "call_tool":
                    tool_name = message.get("tool_name")
                    arguments = message.get("arguments", {})
                    
                    # Call tool using FastMCP's async call_tool method
                    mcp = env._mcp
                    try:
                        tool_result = await mcp.call_tool(tool_name, arguments)
                        if tool_result and tool_result.content:
                            result = tool_result.content[0].text if hasattr(tool_result.content[0], 'text') else str(tool_result.content[0])
                        else:
                            result = str(tool_result)
                    except Exception as e:
                        result = f"Error: {str(e)}"
                    
                    # Simulate and get reward
                    env._simulate_tick()
                    reward = env._calculate_reward()
                    done = env._state.step_count >= env._max_steps
                    
                    await websocket.send_json({
                        "type": "tool_result",
                        "tool_name": tool_name,
                        "result": result,
                        "reward": reward,
                        "done": done,
                        "step": env._state.step_count,
                        "tick": env._tick,
                    })
        
        except WebSocketDisconnect:
            pass


def main():
    """
    Entry point for direct execution.
    
    Usage:
        python -m teeunit_env.server.app
        uvicorn teeunit_env.server.app:app --host 0.0.0.0 --port 8000
    """
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
