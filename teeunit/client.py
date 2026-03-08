"""
TeeUnit Client

Client implementation for connecting to a TeeUnit server.
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, Optional, Union

try:
    import httpx
except ImportError:
    httpx = None

try:
    import websockets
except ImportError:
    websockets = None

from .models import (
    GameConfig,
    StepResult,
    TeeAction,
    TeeObservation,
    TeeState,
)


class TeeEnvError(Exception):
    """Exception raised by TeeEnv client."""
    pass


class TeeEnv:
    """
    Client for connecting to a TeeUnit environment server.
    
    Supports both async and sync usage patterns.
    
    Example (async):
        async with TeeEnv(base_url="http://localhost:8000") as env:
            result = await env.reset()
            result = await env.step(TeeAction(...))
    
    Example (sync):
        with TeeEnv(base_url="http://localhost:8000").sync() as env:
            result = env.reset()
            result = env.step(TeeAction(...))
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.
        
        Args:
            base_url: URL of the TeeUnit server
        """
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None
        self._ws = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        if httpx is None:
            raise ImportError("httpx is required for async client. Install with: pip install httpx")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def sync(self) -> "SyncTeeEnv":
        """
        Get a synchronous wrapper for this client.
        
        Returns:
            SyncTeeEnv wrapper
        """
        return SyncTeeEnv(self)
    
    async def reset(self, config: Optional[Dict] = None) -> StepResult:
        """
        Reset the environment to start a new match.
        
        Args:
            config: Optional configuration overrides
        
        Returns:
            StepResult with initial observation
        """
        if self._client is None:
            raise TeeEnvError("Client not connected. Use 'async with' context manager.")
        
        response = await self._client.post("/reset", json={"config": config})
        response.raise_for_status()
        
        data = response.json()
        return StepResult.from_dict(data)
    
    async def step(self, action: TeeAction) -> StepResult:
        """
        Execute an action.
        
        Args:
            action: The action to execute
        
        Returns:
            StepResult with new observation
        """
        if self._client is None:
            raise TeeEnvError("Client not connected. Use 'async with' context manager.")
        
        response = await self._client.post("/step", json={
            "agent_id": action.agent_id,
            "action_type": action.action_type,
            "direction": action.direction,
            "target_x": action.target_x,
            "target_y": action.target_y,
            "weapon": action.weapon,
        })
        response.raise_for_status()
        
        data = response.json()
        return StepResult.from_dict(data)
    
    async def step_all(self, actions: Dict[int, TeeAction]) -> Dict[int, StepResult]:
        """
        Execute actions for all agents simultaneously.
        
        Args:
            actions: Dict mapping agent_id to action
        
        Returns:
            Dict mapping agent_id to StepResult
        """
        if self._client is None:
            raise TeeEnvError("Client not connected. Use 'async with' context manager.")
        
        actions_dict = {
            str(agent_id): action.to_dict()
            for agent_id, action in actions.items()
        }
        
        response = await self._client.post("/step_all", json={"actions": actions_dict})
        response.raise_for_status()
        
        data = response.json()
        return {
            int(agent_id): StepResult.from_dict(result)
            for agent_id, result in data.items()
        }
    
    async def state(self) -> TeeState:
        """
        Get current episode state.
        
        Returns:
            TeeState with episode metadata
        """
        if self._client is None:
            raise TeeEnvError("Client not connected. Use 'async with' context manager.")
        
        response = await self._client.get("/state")
        response.raise_for_status()
        
        data = response.json()
        return TeeState.from_dict(data)
    
    async def get_observation(self, agent_id: int) -> TeeObservation:
        """
        Get observation for a specific agent.
        
        Args:
            agent_id: Agent ID
        
        Returns:
            TeeObservation for the agent
        """
        if self._client is None:
            raise TeeEnvError("Client not connected. Use 'async with' context manager.")
        
        response = await self._client.post("/observation", json={"agent_id": agent_id})
        response.raise_for_status()
        
        data = response.json()
        return TeeObservation.from_dict(data)
    
    async def get_arena(self) -> Dict:
        """
        Get ASCII map of the arena.
        
        Returns:
            Dict with map string and metadata
        """
        if self._client is None:
            raise TeeEnvError("Client not connected. Use 'async with' context manager.")
        
        response = await self._client.get("/arena")
        response.raise_for_status()
        
        return response.json()
    
    async def health(self) -> Dict:
        """
        Check server health.
        
        Returns:
            Health status dict
        """
        if self._client is None:
            raise TeeEnvError("Client not connected. Use 'async with' context manager.")
        
        response = await self._client.get("/health")
        response.raise_for_status()
        
        return response.json()


class SyncTeeEnv:
    """
    Synchronous wrapper for TeeEnv.
    
    Provides blocking versions of all async methods.
    
    Example:
        with TeeEnv(base_url="http://localhost:8000").sync() as env:
            result = env.reset()
            result = env.step(TeeAction(...))
    """
    
    def __init__(self, async_env: TeeEnv):
        """
        Initialize sync wrapper.
        
        Args:
            async_env: The async TeeEnv to wrap
        """
        self._async_env = async_env
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    def __enter__(self):
        """Context manager entry."""
        # Create new event loop if needed
        try:
            self._loop = asyncio.get_event_loop()
            if self._loop.is_running():
                # Can't use running loop, create new one
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        
        # Enter async context
        self._loop.run_until_complete(self._async_env.__aenter__())
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._loop:
            self._loop.run_until_complete(self._async_env.__aexit__(exc_type, exc_val, exc_tb))
    
    def reset(self, config: Optional[Dict] = None) -> StepResult:
        """Reset the environment."""
        return self._loop.run_until_complete(self._async_env.reset(config))
    
    def step(self, action: TeeAction) -> StepResult:
        """Execute an action."""
        return self._loop.run_until_complete(self._async_env.step(action))
    
    def step_all(self, actions: Dict[int, TeeAction]) -> Dict[int, StepResult]:
        """Execute all actions simultaneously."""
        return self._loop.run_until_complete(self._async_env.step_all(actions))
    
    def state(self) -> TeeState:
        """Get episode state."""
        return self._loop.run_until_complete(self._async_env.state())
    
    def get_observation(self, agent_id: int) -> TeeObservation:
        """Get agent observation."""
        return self._loop.run_until_complete(self._async_env.get_observation(agent_id))
    
    def get_arena(self) -> Dict:
        """Get arena map."""
        return self._loop.run_until_complete(self._async_env.get_arena())
    
    def health(self) -> Dict:
        """Check server health."""
        return self._loop.run_until_complete(self._async_env.health())


class LocalTeeEnv:
    """
    Local environment that doesn't require a server.
    
    Useful for testing and single-machine usage.
    
    Example:
        env = LocalTeeEnv()
        result = env.reset()
        result = env.step(TeeAction(...))
    """
    
    def __init__(self, config: Optional[GameConfig] = None):
        """
        Initialize local environment.
        
        Args:
            config: Game configuration
        """
        from .server.tee_environment import TeeEnvironment
        self._env = TeeEnvironment(config)
    
    def reset(self, config: Optional[Dict] = None) -> StepResult:
        """Reset the environment."""
        return self._env.reset(config)
    
    def step(self, action: TeeAction) -> StepResult:
        """Execute an action."""
        return self._env.step(action)
    
    def step_all(self, actions: Dict[int, TeeAction]) -> Dict[int, StepResult]:
        """Execute all actions simultaneously."""
        return self._env.step_all(actions)
    
    def state(self) -> TeeState:
        """Get episode state."""
        return self._env.state()
    
    def get_observation(self, agent_id: int) -> TeeObservation:
        """Get agent observation."""
        return self._env.get_observation(agent_id)
    
    def get_arena_ascii(self) -> str:
        """Get ASCII map of the arena."""
        return self._env.arena.to_ascii(self._env.agents.get_agent_positions())


# Convenience functions
def make_env(base_url: Optional[str] = None, config: Optional[GameConfig] = None) -> Union[TeeEnv, LocalTeeEnv]:
    """
    Create a TeeUnit environment.
    
    Args:
        base_url: Server URL (if None, creates local environment)
        config: Game configuration (for local environment)
    
    Returns:
        TeeEnv or LocalTeeEnv
    """
    if base_url:
        return TeeEnv(base_url)
    else:
        return LocalTeeEnv(config)
