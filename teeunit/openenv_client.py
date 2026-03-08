"""
TeeUnit OpenEnv Client

Client for connecting to a remote TeeUnit environment server.
Supports both HTTP and WebSocket protocols.

Usage:
    # Async (recommended)
    async with TeeEnv(base_url="http://localhost:7860") as env:
        result = await env.reset()
        result = await env.step(TeeAction(direction=1, fire=True))
    
    # Sync
    with TeeEnv(base_url="http://localhost:7860").sync() as env:
        result = env.reset()
        result = env.step(TeeAction(direction=1, fire=True))
"""

import asyncio
import logging
from typing import Any, Dict, Generic, Optional, TypeVar, Union

import httpx

from .openenv_models import (
    TeeAction,
    TeeMultiAction,
    TeeObservation,
    TeeMultiObservation,
    TeeState,
    TeeStepResult,
    TeeMultiStepResult,
    VisiblePlayer,
    VisibleProjectile,
    VisiblePickup,
    KillEvent,
)

logger = logging.getLogger(__name__)


class TeeEnv:
    """
    Async client for TeeUnit environment.
    
    Connects to a TeeUnit server via HTTP/WebSocket.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:7860",
        timeout: float = 30.0,
    ):
        """
        Initialize the client.
        
        Args:
            base_url: URL of the TeeUnit server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None
        return False
    
    def sync(self) -> "SyncTeeEnv":
        """
        Get a synchronous wrapper for this client.
        
        Usage:
            with TeeEnv(base_url="...").sync() as env:
                result = env.reset()
        """
        return SyncTeeEnv(self)
    
    async def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TeeMultiObservation:
        """
        Reset the environment.
        
        Args:
            seed: Random seed (unused)
            episode_id: Custom episode ID
            **kwargs: Additional config options
        
        Returns:
            Initial observations for all agents
        """
        if not self._client:
            raise RuntimeError("Client not connected. Use 'async with TeeEnv() as env:'")
        
        payload = {"config": kwargs}
        if episode_id:
            payload["episode_id"] = episode_id
        
        response = await self._client.post("/reset", json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Parse response - could be single observation or multi
        obs = self._parse_observation(data.get("observation", data))
        
        return TeeMultiObservation(
            observations={obs.agent_id: obs},
            done=data.get("done", False),
            reward=data.get("reward", 0.0),
            metadata=data.get("info", {}),
        )
    
    async def step(
        self,
        action: TeeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TeeObservation:
        """
        Execute action for a single agent.
        
        Args:
            action: TeeAction to execute
            timeout_s: Request timeout override
            **kwargs: Additional options
        
        Returns:
            Observation for the acting agent
        """
        if not self._client:
            raise RuntimeError("Client not connected. Use 'async with TeeEnv() as env:'")
        
        payload = {
            "agent_id": action.agent_id,
            "direction": action.direction,
            "target_x": action.target_x,
            "target_y": action.target_y,
            "jump": action.jump,
            "fire": action.fire,
            "hook": action.hook,
            "wanted_weapon": action.weapon,
        }
        
        timeout = timeout_s or self.timeout
        response = await self._client.post("/step", json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        
        obs = self._parse_observation(data.get("observation", {}))
        obs.reward = data.get("reward", 0.0)
        obs.done = data.get("done", False)
        
        return obs
    
    async def step_all(
        self,
        actions: Union[TeeMultiAction, Dict[int, TeeAction]],
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TeeMultiStepResult:
        """
        Execute actions for all agents.
        
        Args:
            actions: TeeMultiAction or dict of agent_id -> TeeAction
            timeout_s: Request timeout override
            **kwargs: Additional options
        
        Returns:
            Results for all agents
        """
        if not self._client:
            raise RuntimeError("Client not connected. Use 'async with TeeEnv() as env:'")
        
        # Convert to dict format
        if isinstance(actions, TeeMultiAction):
            action_dict = actions.actions
        else:
            action_dict = actions
        
        payload = {
            "actions": {
                str(agent_id): {
                    "direction": action.direction,
                    "target_x": action.target_x,
                    "target_y": action.target_y,
                    "jump": action.jump,
                    "fire": action.fire,
                    "hook": action.hook,
                    "wanted_weapon": action.weapon,
                }
                for agent_id, action in action_dict.items()
            }
        }
        
        timeout = timeout_s or self.timeout
        response = await self._client.post("/step_all", json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        
        # Parse results for each agent
        results = {}
        game_over = False
        
        for agent_id_str, result_data in data.items():
            agent_id = int(agent_id_str)
            obs = self._parse_observation(result_data.get("observation", {}))
            obs.reward = result_data.get("reward", 0.0)
            obs.done = result_data.get("done", False)
            
            if obs.done:
                game_over = True
            
            results[agent_id] = TeeStepResult(
                observation=obs,
                reward=result_data.get("reward", 0.0),
                done=result_data.get("done", False),
                truncated=False,
                info=result_data.get("info", {}),
            )
        
        # Get state
        state = await self.state()
        
        return TeeMultiStepResult(results=results, state=state)
    
    async def state(self) -> TeeState:
        """Get current environment state."""
        if not self._client:
            raise RuntimeError("Client not connected. Use 'async with TeeEnv() as env:'")
        
        response = await self._client.get("/state")
        response.raise_for_status()
        data = response.json()
        
        return TeeState(
            episode_id=data.get("episode_id", ""),
            step_count=data.get("step_count", 0),
            tick=data.get("tick", 0),
            agents_alive=data.get("agents_alive", []),
            scores={int(k): v for k, v in data.get("scores", {}).items()},
            game_over=data.get("game_over", False),
            winner=data.get("winner"),
            ticks_per_step=data.get("ticks_per_step", 10),
            num_agents=data.get("config", {}).get("num_agents", 4),
            config=data.get("config", {}),
        )
    
    async def health(self) -> Dict[str, Any]:
        """Check server health."""
        if not self._client:
            raise RuntimeError("Client not connected. Use 'async with TeeEnv() as env:'")
        
        response = await self._client.get("/health")
        response.raise_for_status()
        return response.json()
    
    def _parse_observation(self, data: Dict[str, Any]) -> TeeObservation:
        """Parse observation from JSON response."""
        # Parse visible players
        visible_players = [
            VisiblePlayer(
                client_id=p.get("client_id", 0),
                x=p.get("x", 0),
                y=p.get("y", 0),
                vel_x=p.get("vel_x", 0),
                vel_y=p.get("vel_y", 0),
                health=p.get("health", 10),
                armor=p.get("armor", 0),
                weapon=p.get("weapon", 1),
                direction=p.get("direction", 0),
                score=p.get("score", 0),
                is_hooking=p.get("is_hooking", False),
            )
            for p in data.get("visible_players", [])
        ]
        
        # Parse projectiles
        projectiles = [
            VisibleProjectile(
                x=p.get("x", 0),
                y=p.get("y", 0),
                vel_x=p.get("vel_x", 0),
                vel_y=p.get("vel_y", 0),
                weapon_type=p.get("weapon_type", 0),
            )
            for p in data.get("projectiles", [])
        ]
        
        # Parse pickups
        pickups = [
            VisiblePickup(
                x=p.get("x", 0),
                y=p.get("y", 0),
                pickup_type=p.get("pickup_type", 0),
            )
            for p in data.get("pickups", [])
        ]
        
        # Parse kill events
        recent_kills = [
            KillEvent(
                killer_id=k.get("killer_id", 0),
                victim_id=k.get("victim_id", 0),
                weapon=k.get("weapon", 0),
                tick=k.get("tick", 0),
            )
            for k in data.get("recent_kills", [])
        ]
        
        return TeeObservation(
            agent_id=data.get("agent_id", 0),
            tick=data.get("tick", 0),
            x=data.get("x", 0),
            y=data.get("y", 0),
            vel_x=data.get("vel_x", 0),
            vel_y=data.get("vel_y", 0),
            health=data.get("health", 10),
            armor=data.get("armor", 0),
            weapon=data.get("weapon", 1),
            ammo=data.get("ammo", -1),
            direction=data.get("direction", 0),
            is_grounded=data.get("is_grounded", True),
            is_alive=data.get("is_alive", True),
            score=data.get("score", 0),
            visible_players=visible_players,
            projectiles=projectiles,
            pickups=pickups,
            recent_kills=recent_kills,
            episode_id=data.get("episode_id", ""),
            text_description=data.get("text_description", ""),
            done=data.get("done", False),
            reward=data.get("reward", 0.0),
        )


class SyncTeeEnv:
    """
    Synchronous wrapper for TeeEnv.
    
    Usage:
        with TeeEnv(base_url="...").sync() as env:
            result = env.reset()
            result = env.step(TeeAction(direction=1))
    """
    
    def __init__(self, async_env: TeeEnv):
        self._async_env = async_env
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    def __enter__(self):
        """Context manager entry."""
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._async_env.__aenter__())
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._loop:
            self._loop.run_until_complete(self._async_env.__aexit__(exc_type, exc_val, exc_tb))
            self._loop.close()
            self._loop = None
        return False
    
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TeeMultiObservation:
        """Reset environment (sync)."""
        if not self._loop:
            raise RuntimeError("Client not connected. Use 'with env.sync() as env:'")
        return self._loop.run_until_complete(
            self._async_env.reset(seed=seed, episode_id=episode_id, **kwargs)
        )
    
    def step(
        self,
        action: TeeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TeeObservation:
        """Execute action (sync)."""
        if not self._loop:
            raise RuntimeError("Client not connected. Use 'with env.sync() as env:'")
        return self._loop.run_until_complete(
            self._async_env.step(action, timeout_s=timeout_s, **kwargs)
        )
    
    def step_all(
        self,
        actions: Union[TeeMultiAction, Dict[int, TeeAction]],
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TeeMultiStepResult:
        """Execute actions for all agents (sync)."""
        if not self._loop:
            raise RuntimeError("Client not connected. Use 'with env.sync() as env:'")
        return self._loop.run_until_complete(
            self._async_env.step_all(actions, timeout_s=timeout_s, **kwargs)
        )
    
    def state(self) -> TeeState:
        """Get state (sync)."""
        if not self._loop:
            raise RuntimeError("Client not connected. Use 'with env.sync() as env:'")
        return self._loop.run_until_complete(self._async_env.state())
    
    def health(self) -> Dict[str, Any]:
        """Check health (sync)."""
        if not self._loop:
            raise RuntimeError("Client not connected. Use 'with env.sync() as env:'")
        return self._loop.run_until_complete(self._async_env.health())
    
    def close(self) -> None:
        """Close the client connection."""
        if self._loop:
            self._loop.run_until_complete(self._async_env.__aexit__(None, None, None))
            self._loop.close()
            self._loop = None


# Alias for training script compatibility
TeeEnvClient = SyncTeeEnv


def create_client(base_url: str = "http://localhost:7860") -> SyncTeeEnv:
    """
    Create a synchronous TeeUnit client.
    
    This is a convenience function that returns a ready-to-use client.
    Remember to call close() when done, or use as context manager.
    
    Args:
        base_url: URL of the TeeUnit server
    
    Returns:
        SyncTeeEnv instance (already connected)
    """
    async_env = TeeEnv(base_url=base_url)
    sync_env = SyncTeeEnv(async_env)
    sync_env._loop = asyncio.new_event_loop()
    sync_env._loop.run_until_complete(async_env.__aenter__())
    return sync_env
