# Copyright (c) 2024 TeeUnit Project
# SPDX-License-Identifier: MIT

"""
TeeUnit Environment Client.

This module provides the client for connecting to a TeeUnit Environment server.
TeeEnv extends MCPToolClient to provide tool-calling style interactions.

Example:
    >>> with TeeEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...
    ...     # Discover tools
    ...     tools = env.list_tools()
    ...     print([t.name for t in tools])  # ['move', 'jump', 'aim', 'shoot', 'hook', 'get_status']
    ...
    ...     # Get game state
    ...     status = env.call_tool("get_status")
    ...     print(status)
    ...
    ...     # Take actions
    ...     result = env.call_tool("move", direction="right")
    ...     result = env.call_tool("shoot", weapon=2)
"""

from dataclasses import dataclass
from typing import Optional, Any, Dict

# Support both in-repo and standalone imports
try:
    from openenv.core.mcp_client import MCPToolClient
except ImportError:
    # Fallback for development/testing without openenv installed
    class MCPToolClient:
        """Fallback MCPToolClient for development."""
        
        def __init__(self, base_url: str = "http://localhost:8000", **kwargs):
            self.base_url = base_url
            self._connected = False
        
        def __enter__(self):
            self._connected = True
            return self
        
        def __exit__(self, *args):
            self._connected = False
        
        def reset(self, **kwargs):
            pass
        
        def list_tools(self):
            return []
        
        def call_tool(self, name: str, **kwargs):
            return None
        
        def step(self, action):
            return None
        
        def close(self):
            self._connected = False
        
        @classmethod
        def from_docker_image(cls, image: str, **kwargs):
            return cls(**kwargs)
        
        @classmethod
        def from_env(cls, env_id: str, **kwargs):
            return cls(**kwargs)


@dataclass
class TeeAction:
    """
    Action for the TeeUnit environment.
    
    This is a convenience class for building actions to send to the environment.
    The actual actions are sent via MCP tools (move, jump, aim, shoot, hook).
    
    Attributes:
        tool_name: The MCP tool to call
        arguments: Arguments for the tool
    """
    tool_name: str
    arguments: Dict[str, Any]
    
    @classmethod
    def move(cls, direction: str = "none") -> "TeeAction":
        """Create a move action."""
        return cls(tool_name="move", arguments={"direction": direction})
    
    @classmethod
    def jump(cls) -> "TeeAction":
        """Create a jump action."""
        return cls(tool_name="jump", arguments={})
    
    @classmethod
    def aim(cls, x: int, y: int) -> "TeeAction":
        """Create an aim action."""
        return cls(tool_name="aim", arguments={"x": x, "y": y})
    
    @classmethod
    def shoot(cls, weapon: int = -1) -> "TeeAction":
        """Create a shoot action."""
        return cls(tool_name="shoot", arguments={"weapon": weapon})
    
    @classmethod
    def hook(cls) -> "TeeAction":
        """Create a hook action."""
        return cls(tool_name="hook", arguments={})
    
    @classmethod
    def get_status(cls) -> "TeeAction":
        """Create a get_status action."""
        return cls(tool_name="get_status", arguments={})


class TeeEnv(MCPToolClient):
    """
    Client for the TeeUnit Environment.
    
    This client provides a simple interface for interacting with the TeeUnit
    Environment via MCP tools. It inherits all functionality from MCPToolClient:
    - `list_tools()`: Discover available tools
    - `call_tool(name, **kwargs)`: Call a tool by name
    - `reset(**kwargs)`: Reset the environment
    - `step(action)`: Execute an action
    
    Available MCP Tools:
    - `move(direction)`: Move left, right, or none
    - `jump()`: Make the tee jump
    - `aim(x, y)`: Aim at coordinates
    - `shoot(weapon)`: Fire weapon (0-5 or -1 for current)
    - `hook()`: Use grappling hook
    - `get_status()`: Get game state as text
    
    Example:
        >>> # Connect to a running server
        >>> with TeeEnv(base_url="http://localhost:8000") as env:
        ...     env.reset()
        ...
        ...     # List available tools
        ...     tools = env.list_tools()
        ...     for tool in tools:
        ...         print(f"{tool.name}: {tool.description}")
        ...
        ...     # Get game state
        ...     status = env.call_tool("get_status")
        ...     print(status)
        ...
        ...     # Take actions
        ...     env.call_tool("move", direction="right")
        ...     env.call_tool("aim", x=500, y=300)
        ...     env.call_tool("shoot", weapon=2)
    
    Example with HuggingFace Space:
        >>> # Connect to HuggingFace Space
        >>> env = TeeEnv.from_env("ziadbc/teeunit-env")
        >>> try:
        ...     env.reset()
        ...     status = env.call_tool("get_status")
        ...     print(status)
        ... finally:
        ...     env.close()
    
    Example with Docker:
        >>> # Automatically start container and connect
        >>> env = TeeEnv.from_docker_image("teeunit-env:latest")
        >>> try:
        ...     env.reset()
        ...     env.call_tool("move", direction="right")
        ... finally:
        ...     env.close()
    """
    
    def execute_action(self, action: TeeAction) -> Any:
        """
        Execute a TeeAction.
        
        Args:
            action: TeeAction to execute
            
        Returns:
            Result from the MCP tool call
        """
        return self.call_tool(action.tool_name, **action.arguments)
    
    def get_status(self) -> str:
        """
        Get the current game state as text.
        
        Returns:
            Text description of game state
        """
        return self.call_tool("get_status")
    
    def move(self, direction: str = "none") -> str:
        """
        Move the tee.
        
        Args:
            direction: "left", "right", or "none"
            
        Returns:
            Result message
        """
        return self.call_tool("move", direction=direction)
    
    def jump(self) -> str:
        """
        Make the tee jump.
        
        Returns:
            Result message
        """
        return self.call_tool("jump")
    
    def aim(self, x: int, y: int) -> str:
        """
        Aim at coordinates.
        
        Args:
            x: Target X coordinate
            y: Target Y coordinate
            
        Returns:
            Result message
        """
        return self.call_tool("aim", x=x, y=y)
    
    def shoot(self, weapon: int = -1) -> str:
        """
        Fire weapon.
        
        Args:
            weapon: Weapon ID (0-5) or -1 for current weapon
            
        Returns:
            Result message
        """
        return self.call_tool("shoot", weapon=weapon)
    
    def hook(self) -> str:
        """
        Use grappling hook.
        
        Returns:
            Result message
        """
        return self.call_tool("hook")
