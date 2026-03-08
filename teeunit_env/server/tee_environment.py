# Copyright (c) 2024 TeeUnit Project
# SPDX-License-Identifier: MIT

"""
TeeUnit Environment Implementation.

A MCP environment that wraps the Teeworlds game for LLM-based RL training.
All interactions happen through MCP tools that translate to game actions.

Supports two modes:
- Simulation mode (default): Uses built-in physics simulation
- Real server mode: Connects to actual Teeworlds 0.7.5 server

MCP Tools:
- `move(direction)`: Move the tee left, right, or none
- `jump()`: Make the tee jump
- `aim(x, y)`: Aim at target coordinates
- `shoot(weapon)`: Fire the specified weapon
- `hook()`: Use the grappling hook
- `get_status()`: Get current game state as text

Example:
    >>> from openenv.core.env_server.mcp_types import ListToolsAction, CallToolAction
    >>> env = TeeEnvironment()
    >>> env.reset()
    >>>
    >>> # List available tools
    >>> obs = env.step(ListToolsAction())
    >>> print([t.name for t in obs.tools])  # ["move", "jump", "aim", "shoot", "hook", "get_status"]
    >>>
    >>> # Get game state
    >>> obs = env.step(CallToolAction(tool_name="get_status", arguments={}))
    >>> print(obs.result)
    
    # With real Teeworlds server:
    >>> env = TeeEnvironment(use_real_server=True, server_host="127.0.0.1", server_port=8303)
    >>> env.reset()  # Connects to server
"""

from typing import Any, Optional, Dict, List
from uuid import uuid4
import random
import math
import logging

logger = logging.getLogger(__name__)

# Try to import real server components
try:
    import sys
    import os
    # Add parent path for teeunit package
    _parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    
    from teeunit.server.bot_manager import BotManager, GameState as RealGameState
    from teeunit.protocol.objects import PlayerInput, Character
    REAL_SERVER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Real server support not available: {e}")
    REAL_SERVER_AVAILABLE = False
    BotManager = None
    PlayerInput = None
    Character = None
    RealGameState = None

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Fallback for development/testing
    from dataclasses import dataclass
    
    @dataclass
    class State:
        episode_id: str = ""
        step_count: int = 0
    
    @dataclass
    class Observation:
        done: bool = False
        reward: float = 0.0
        metadata: dict = None
        
        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {}
    
    class Action:
        pass
    
    class MCPEnvironment:
        def __init__(self, mcp):
            self._mcp = mcp
        
        def step(self, action, **kwargs):
            return Observation()

from fastmcp import FastMCP


# Weapon definitions
WEAPONS = {
    0: {"name": "hammer", "ammo": -1, "damage": 3},
    1: {"name": "pistol", "ammo": 10, "damage": 1},
    2: {"name": "shotgun", "ammo": 10, "damage": 3},
    3: {"name": "grenade", "ammo": 10, "damage": 6},
    4: {"name": "laser", "ammo": 10, "damage": 5},
    5: {"name": "ninja", "ammo": -1, "damage": 9},
}


class GameAgent:
    """Represents a player/bot in the game."""
    
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.x = 400.0 + random.uniform(-200, 200)
        self.y = 300.0 + random.uniform(-100, 100)
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.health = 10
        self.armor = 0
        self.weapon = 1  # pistol
        self.ammo = {w: WEAPONS[w]["ammo"] for w in WEAPONS}
        self.direction = 1  # 1 = right, -1 = left
        self.is_alive = True
        self.score = 0
        self.aim_x = self.x + 100
        self.aim_y = self.y
        self.is_hooking = False
        self.is_grounded = True
        
    def respawn(self):
        """Respawn at random location."""
        self.x = 400.0 + random.uniform(-200, 200)
        self.y = 300.0 + random.uniform(-100, 100)
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.health = 10
        self.armor = 0
        self.weapon = 1
        self.ammo = {w: WEAPONS[w]["ammo"] for w in WEAPONS}
        self.is_alive = True
        self.is_hooking = False


class TeeEnvironment(MCPEnvironment):
    """
    OpenEnv-compatible Teeworlds environment with MCP tool interface.
    
    This environment provides a text-based interface for LLM agents to play
    Teeworlds. The LLM receives game state as natural language descriptions
    and issues commands through MCP tools.
    
    For hackathon demo, this uses a simplified game simulation. For production,
    it can be connected to the real Teeworlds server via bot_manager.
    
    Example:
        >>> with TeeEnv(base_url="http://localhost:8000") as env:
        ...     env.reset()
        ...     status = env.call_tool("get_status")
        ...     env.call_tool("move", direction="right")
        ...     env.call_tool("shoot", weapon=2)
    """
    
    def __init__(
        self,
        num_agents: int = 4,
        max_steps: int = 1000,
        use_real_server: bool = False,
        server_host: str = "127.0.0.1",
        server_port: int = 8303,
    ):
        """
        Initialize the TeeUnit environment.
        
        Args:
            num_agents: Number of agents in the arena
            max_steps: Maximum steps per episode
            use_real_server: If True, connect to real Teeworlds server
            server_host: Teeworlds server host
            server_port: Teeworlds server port
        """
        # Validate real server mode
        if use_real_server and not REAL_SERVER_AVAILABLE:
            raise RuntimeError(
                "Real server mode requested but teeunit package not available. "
                "Make sure teeunit is installed or in PYTHONPATH."
            )
        
        # Create MCP server and define tools inline
        mcp = FastMCP("teeunit_env")
        
        # Store config
        self._num_agents = num_agents
        self._max_steps = max_steps
        self._use_real_server = use_real_server
        self._server_host = server_host
        self._server_port = server_port
        
        # Game state (simulation mode)
        self._agents: Dict[int, GameAgent] = {}
        self._tick = 0
        self._kill_events: List[dict] = []
        self._current_agent_id = 0  # LLM controls agent 0
        
        # Episode state
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # Real server connection
        self._bot_manager: Optional[BotManager] = None
        self._pending_input: Optional[PlayerInput] = None if not REAL_SERVER_AVAILABLE else PlayerInput()
        self._fire_counter = 0  # Track fire presses for real server
        
        # Store tool functions for direct synchronous access
        self._tool_fns = {}
        
        # Define MCP tools
        @mcp.tool
        def move(direction: str) -> str:
            """
            Move the tee horizontally.
            
            Args:
                direction: "left", "right", or "none"
            
            Returns:
                Result message describing the action taken
            """
            if self._use_real_server:
                # Real server mode: update pending input
                if direction == "left":
                    self._pending_input.direction = -1
                    return "Moving left."
                elif direction == "right":
                    self._pending_input.direction = 1
                    return "Moving right."
                else:
                    self._pending_input.direction = 0
                    return "Stopped."
            else:
                # Simulation mode
                agent = self._agents.get(self._current_agent_id)
                if not agent or not agent.is_alive:
                    return "Cannot move: agent is dead"
                
                if direction == "left":
                    agent.direction = -1
                    agent.vel_x = max(agent.vel_x - 5, -15)
                    return f"Moving left. Velocity: ({agent.vel_x:.1f}, {agent.vel_y:.1f})"
                elif direction == "right":
                    agent.direction = 1
                    agent.vel_x = min(agent.vel_x + 5, 15)
                    return f"Moving right. Velocity: ({agent.vel_x:.1f}, {agent.vel_y:.1f})"
                else:
                    agent.vel_x *= 0.8  # friction
                    return f"Stopped. Velocity: ({agent.vel_x:.1f}, {agent.vel_y:.1f})"
        
        @mcp.tool
        def jump() -> str:
            """
            Make the tee jump. Can double-jump in the air.
            
            Returns:
                Result message describing the jump
            """
            if self._use_real_server:
                # Real server mode: set jump flag
                self._pending_input.jump = True
                return "Jumping!"
            else:
                # Simulation mode
                agent = self._agents.get(self._current_agent_id)
                if not agent or not agent.is_alive:
                    return "Cannot jump: agent is dead"
                
                if agent.is_grounded:
                    agent.vel_y = -12
                    agent.is_grounded = False
                    return f"Jumped! Velocity: ({agent.vel_x:.1f}, {agent.vel_y:.1f})"
                else:
                    # Air jump (weaker)
                    agent.vel_y = -8
                    return f"Air jumped! Velocity: ({agent.vel_x:.1f}, {agent.vel_y:.1f})"
        
        @mcp.tool
        def aim(x: int, y: int) -> str:
            """
            Aim at target coordinates.
            
            Args:
                x: Target X coordinate
                y: Target Y coordinate
            
            Returns:
                Result message confirming aim direction
            """
            if self._use_real_server:
                # Real server mode: set target position (relative to player)
                # In Teeworlds, target is relative to player position
                self._pending_input.target_x = x
                self._pending_input.target_y = y
                angle = math.atan2(y, x) * 180 / math.pi
                return f"Aiming at ({x}, {y}). Angle: {angle:.1f} deg"
            else:
                # Simulation mode
                agent = self._agents.get(self._current_agent_id)
                if not agent or not agent.is_alive:
                    return "Cannot aim: agent is dead"
                
                agent.aim_x = x
                agent.aim_y = y
                
                # Calculate angle for display
                dx = x - agent.x
                dy = y - agent.y
                angle = math.atan2(dy, dx) * 180 / math.pi
                distance = math.sqrt(dx*dx + dy*dy)
                
                return f"Aiming at ({x}, {y}). Angle: {angle:.1f} deg, Distance: {distance:.1f} units"
        
        @mcp.tool
        def shoot(weapon: int = -1) -> str:
            """
            Fire the current or specified weapon.
            
            Args:
                weapon: Weapon ID (0=hammer, 1=pistol, 2=shotgun, 3=grenade, 4=laser, 5=ninja).
                       Use -1 for current weapon.
            
            Returns:
                Result message describing the shot and any hits
            """
            if self._use_real_server:
                # Real server mode: increment fire counter and set weapon
                self._fire_counter += 1
                self._pending_input.fire = self._fire_counter
                
                # Set wanted weapon (Teeworlds uses 1-indexed: 1=hammer, 2=gun, etc.)
                if weapon >= 0 and weapon <= 5:
                    self._pending_input.wanted_weapon = weapon + 1  # Convert to 1-indexed
                    wpn_name = WEAPONS[weapon]["name"]
                else:
                    wpn_name = "current weapon"
                
                return f"Fired {wpn_name}! (fire counter: {self._fire_counter})"
            else:
                # Simulation mode
                agent = self._agents.get(self._current_agent_id)
                if not agent or not agent.is_alive:
                    return "Cannot shoot: agent is dead"
                
                # Switch weapon if specified
                if weapon >= 0 and weapon <= 5:
                    agent.weapon = weapon
                
                wpn = WEAPONS[agent.weapon]
                wpn_name = wpn["name"]
                
                # Check ammo
                if wpn["ammo"] > 0 and agent.ammo[agent.weapon] <= 0:
                    return f"Out of ammo for {wpn_name}!"
                
                # Use ammo
                if wpn["ammo"] > 0:
                    agent.ammo[agent.weapon] -= 1
                
                # Check for hits on other agents
                hits = []
                for other_id, other in self._agents.items():
                    if other_id == self._current_agent_id or not other.is_alive:
                        continue
                    
                    # Simple hit detection based on aim
                    dx = other.x - agent.x
                    dy = other.y - agent.y
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    aim_dx = agent.aim_x - agent.x
                    aim_dy = agent.aim_y - agent.y
                    aim_dist = math.sqrt(aim_dx*aim_dx + aim_dy*aim_dy)
                    
                    if aim_dist > 0:
                        # Check if enemy is roughly in line of fire
                        dot = (dx * aim_dx + dy * aim_dy) / (aim_dist * max(distance, 1))
                        
                        # Hit probability based on weapon and distance
                        hit_range = 400 if agent.weapon != 0 else 50  # hammer short range
                        if distance < hit_range and dot > 0.8:
                            # Hit!
                            damage = wpn["damage"]
                            other.health -= damage
                            other.armor = max(0, other.armor - damage // 2)
                            
                            if other.health <= 0:
                                other.is_alive = False
                                agent.score += 1
                                self._kill_events.append({
                                    "killer_id": self._current_agent_id,
                                    "victim_id": other_id,
                                    "weapon": agent.weapon,
                                    "tick": self._tick,
                                })
                                hits.append(f"KILLED Player {other_id} with {wpn_name}!")
                            else:
                                hits.append(f"Hit Player {other_id} for {damage} damage ({other.health}HP remaining)")
                
                ammo_str = f"({agent.ammo[agent.weapon]} ammo)" if wpn["ammo"] > 0 else ""
                if hits:
                    return f"Fired {wpn_name} {ammo_str}. " + " ".join(hits)
                else:
                    return f"Fired {wpn_name} {ammo_str}. No hits."
        
        @mcp.tool
        def hook() -> str:
            """
            Use the grappling hook in the aim direction.
            The hook can grab walls or enemies to pull yourself toward them.
            
            Returns:
                Result message describing hook action
            """
            if self._use_real_server:
                # Real server mode: toggle hook flag
                self._pending_input.hook = not self._pending_input.hook
                if self._pending_input.hook:
                    return "Hook deployed!"
                else:
                    return "Hook released."
            else:
                # Simulation mode
                agent = self._agents.get(self._current_agent_id)
                if not agent or not agent.is_alive:
                    return "Cannot hook: agent is dead"
                
                agent.is_hooking = not agent.is_hooking
                
                if agent.is_hooking:
                    # Pull toward aim point
                    dx = agent.aim_x - agent.x
                    dy = agent.aim_y - agent.y
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist > 0:
                        agent.vel_x += (dx / dist) * 3
                        agent.vel_y += (dy / dist) * 3
                    return f"Hook deployed! Pulling toward ({agent.aim_x}, {agent.aim_y})"
                else:
                    return "Hook released."
        
        @mcp.tool
        def get_status() -> str:
            """
            Get the current game state as a text description.
            
            Returns:
                Detailed text description of current game state including:
                - Your position, health, weapon, ammo
                - Visible enemies with positions and health
                - Recent events (kills, deaths)
            """
            if self._use_real_server:
                # Real server mode: read from BotManager.game_state
                if not self._bot_manager or not self._bot_manager.all_connected:
                    return "Not connected to server."
                
                gs = self._bot_manager.game_state
                my_char = gs.get_character(self._current_agent_id)
                my_info = gs.get_player_info(self._current_agent_id)
                
                lines = []
                lines.append(f"=== Teeworlds Game State (Tick {gs.tick}) ===")
                lines.append("")
                
                if my_char is None:
                    lines.append("STATUS: DEAD - Waiting for respawn...")
                    lines.append("")
                else:
                    # Position is in fixed-point (divide by 32 for world units)
                    x = my_char.x / 32.0
                    y = my_char.y / 32.0
                    vel_x = my_char.vel_x / 256.0
                    vel_y = my_char.vel_y / 256.0
                    
                    lines.append(f"Position: ({x:.0f}, {y:.0f}) | Velocity: ({vel_x:.1f}, {vel_y:.1f})")
                    lines.append(f"Health: {my_char.health}/10 | Armor: {my_char.armor}/10")
                    
                    # Weapon (0=hammer, 1=gun, etc.)
                    wpn_id = my_char.weapon
                    wpn_name = WEAPONS.get(wpn_id, {}).get("name", f"weapon_{wpn_id}")
                    lines.append(f"Weapon: {wpn_name} ({my_char.ammo_count} ammo)")
                    
                    if my_info:
                        lines.append(f"Score: {my_info.score} kills")
                    lines.append("")
                
                # Other players
                enemies = []
                for client_id, char in gs.characters.items():
                    if client_id == self._current_agent_id:
                        continue
                    
                    other_info = gs.get_player_info(client_id)
                    x = char.x / 32.0
                    y = char.y / 32.0
                    
                    if my_char:
                        dx = x - (my_char.x / 32.0)
                        dy = y - (my_char.y / 32.0)
                        dist = math.sqrt(dx*dx + dy*dy)
                    else:
                        dist = 0
                    
                    wpn_name = WEAPONS.get(char.weapon, {}).get("name", "unknown")
                    score = other_info.score if other_info else 0
                    
                    enemies.append(
                        f"  - Player {client_id}: pos({x:.0f}, {y:.0f}), "
                        f"{char.health}HP, {wpn_name}, {dist:.0f} units away, {score} kills"
                    )
                
                if enemies:
                    lines.append("OTHER PLAYERS:")
                    lines.extend(enemies)
                else:
                    lines.append("OTHER PLAYERS: None")
                lines.append("")
                
                # Recent kills
                recent = gs.kill_events[-5:] if gs.kill_events else []
                if recent:
                    lines.append("RECENT EVENTS:")
                    for event in recent:
                        killer = event.killer_id
                        victim = event.victim_id
                        wpn_name = WEAPONS.get(event.weapon, {}).get("name", "unknown")
                        if killer == self._current_agent_id:
                            lines.append(f"  - You killed Player {victim} with {wpn_name}")
                        elif victim == self._current_agent_id:
                            lines.append(f"  - Player {killer} killed you with {wpn_name}")
                        else:
                            lines.append(f"  - Player {killer} killed Player {victim} with {wpn_name}")
                    lines.append("")
                
                lines.append("AVAILABLE ACTIONS: move, jump, aim, shoot, hook, get_status")
                
                return "\n".join(lines)
            else:
                # Simulation mode
                agent = self._agents.get(self._current_agent_id)
                
                lines = []
                lines.append(f"=== Teeworlds Game State (Tick {self._tick}) ===")
                lines.append("")
                
                if not agent or not agent.is_alive:
                    lines.append("STATUS: DEAD - Waiting for respawn...")
                    lines.append("")
                else:
                    lines.append(f"Position: ({agent.x:.0f}, {agent.y:.0f}) | Velocity: ({agent.vel_x:.1f}, {agent.vel_y:.1f})")
                    lines.append(f"Health: {agent.health}/10 | Armor: {agent.armor}/10")
                    
                    wpn = WEAPONS[agent.weapon]
                    ammo_str = str(agent.ammo[agent.weapon]) if wpn["ammo"] > 0 else "infinite"
                    lines.append(f"Weapon: {wpn['name']} ({ammo_str} ammo)")
                    lines.append(f"Score: {agent.score} kills")
                    lines.append(f"Aim: ({agent.aim_x:.0f}, {agent.aim_y:.0f})")
                    lines.append("")
                
                # Other players
                enemies = []
                for other_id, other in self._agents.items():
                    if other_id == self._current_agent_id:
                        continue
                        
                    if other.is_alive:
                        dx = other.x - agent.x if agent else other.x
                        dy = other.y - agent.y if agent else other.y
                        dist = math.sqrt(dx*dx + dy*dy)
                        wpn_name = WEAPONS[other.weapon]["name"]
                        enemies.append(
                            f"  - Player {other_id}: pos({other.x:.0f}, {other.y:.0f}), "
                            f"{other.health}HP, {wpn_name}, {dist:.0f} units away"
                        )
                    else:
                        enemies.append(f"  - Player {other_id}: DEAD")
                
                if enemies:
                    lines.append("OTHER PLAYERS:")
                    lines.extend(enemies)
                else:
                    lines.append("OTHER PLAYERS: None")
                lines.append("")
                
                # Recent kills
                recent = self._kill_events[-5:] if self._kill_events else []
                if recent:
                    lines.append("RECENT EVENTS:")
                    for event in recent:
                        killer = event["killer_id"]
                        victim = event["victim_id"]
                        wpn_name = WEAPONS[event["weapon"]]["name"]
                        if killer == self._current_agent_id:
                            lines.append(f"  - You killed Player {victim} with {wpn_name}")
                        elif victim == self._current_agent_id:
                            lines.append(f"  - Player {killer} killed you with {wpn_name}")
                        else:
                            lines.append(f"  - Player {killer} killed Player {victim} with {wpn_name}")
                    lines.append("")
                
                lines.append("AVAILABLE ACTIONS: move, jump, aim, shoot, hook, get_status")
                
                return "\n".join(lines)
        
        # Store tool functions for direct synchronous access (for Colab/notebooks)
        self._tool_fns = {
            "move": move,
            "jump": jump,
            "aim": aim,
            "shoot": shoot,
            "hook": hook,
            "get_status": get_status,
        }
        
        # Store MCP reference and pass to base class
        self._mcp = mcp
        super().__init__(mcp)
    
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Optional random seed
            episode_id: Optional episode ID
            **kwargs: Additional reset options
        
        Returns:
            Observation indicating the environment is ready
        """
        if seed is not None:
            random.seed(seed)
        
        # Reset episode state
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._tick = 0
        self._kill_events = []
        self._fire_counter = 0
        
        if self._use_real_server:
            # Real server mode: connect via BotManager
            if self._bot_manager:
                # Disconnect existing connection
                self._bot_manager.disconnect()
            
            # Create new BotManager
            self._bot_manager = BotManager(
                host=self._server_host,
                port=self._server_port,
                num_bots=self._num_agents,
                ticks_per_step=10,  # 200ms per step at 50 ticks/sec
                bot_name_prefix="TeeUnit",
            )
            
            # Connect to server
            connected = self._bot_manager.connect(timeout=10.0)
            if not connected:
                return Observation(
                    done=True,
                    reward=0.0,
                    metadata={
                        "status": "error",
                        "message": f"Failed to connect to Teeworlds server at {self._server_host}:{self._server_port}",
                        "episode_id": self._state.episode_id,
                    },
                )
            
            # Initialize pending input
            self._pending_input = PlayerInput()
            
            # Wait for initial game state
            self._bot_manager.step()  # Get initial snapshot
            
            status = self._get_status_text_real()
            
            return Observation(
                done=False,
                reward=0.0,
                metadata={
                    "status": "ready",
                    "message": status,
                    "episode_id": self._state.episode_id,
                    "mode": "real_server",
                    "server": f"{self._server_host}:{self._server_port}",
                },
            )
        else:
            # Simulation mode
            self._agents = {}
            for i in range(self._num_agents):
                self._agents[i] = GameAgent(i)
            
            status = self._get_status_text()
            
            return Observation(
                done=False,
                reward=0.0,
                metadata={
                    "status": "ready",
                    "message": status,
                    "episode_id": self._state.episode_id,
                    "mode": "simulation",
                },
            )
    
    def _get_status_text(self) -> str:
        """Generate current game status text (simulation mode)."""
        agent = self._agents.get(self._current_agent_id)
        
        lines = []
        lines.append(f"=== Teeworlds Game State (Tick {self._tick}) ===")
        
        if agent and agent.is_alive:
            lines.append(f"Position: ({agent.x:.0f}, {agent.y:.0f})")
            lines.append(f"Health: {agent.health}/10 | Armor: {agent.armor}/10")
            wpn = WEAPONS[agent.weapon]
            lines.append(f"Weapon: {wpn['name']}")
            lines.append(f"Score: {agent.score} kills")
        else:
            lines.append("STATUS: DEAD")
        
        return "\n".join(lines)
    
    def _get_status_text_real(self) -> str:
        """Generate current game status text (real server mode)."""
        if not self._bot_manager:
            return "Not connected to server."
        
        gs = self._bot_manager.game_state
        my_char = gs.get_character(self._current_agent_id)
        
        lines = []
        lines.append(f"=== Teeworlds Game State (Tick {gs.tick}) ===")
        
        if my_char:
            x = my_char.x / 32.0
            y = my_char.y / 32.0
            lines.append(f"Position: ({x:.0f}, {y:.0f})")
            lines.append(f"Health: {my_char.health}/10 | Armor: {my_char.armor}/10")
            wpn_name = WEAPONS.get(my_char.weapon, {}).get("name", "unknown")
            lines.append(f"Weapon: {wpn_name}")
            
            my_info = gs.get_player_info(self._current_agent_id)
            if my_info:
                lines.append(f"Score: {my_info.score} kills")
        else:
            lines.append("STATUS: DEAD")
        
        return "\n".join(lines)
    
    def _execute_real_step(self):
        """Execute one step on the real server."""
        if not self._bot_manager:
            return
        
        # Send pending input for our controlled bot
        inputs = {self._current_agent_id: self._pending_input}
        
        # Execute the step (waits for ticks_per_step game ticks)
        self._bot_manager.step(inputs)
        
        # Update tick from game state
        self._tick = self._bot_manager.game_state.tick
        
        # Reset one-shot inputs (jump resets automatically in Teeworlds)
        self._pending_input.jump = False
    
    def _simulate_tick(self):
        """Simulate one game tick (physics, AI, etc.)."""
        self._tick += 1
        
        for agent in self._agents.values():
            if not agent.is_alive:
                continue
            
            # Apply gravity
            agent.vel_y += 0.5
            
            # Apply velocity
            agent.x += agent.vel_x
            agent.y += agent.vel_y
            
            # Ground collision (simple)
            if agent.y > 500:
                agent.y = 500
                agent.vel_y = 0
                agent.is_grounded = True
            
            # Wall collision
            agent.x = max(50, min(750, agent.x))
            
            # Friction
            agent.vel_x *= 0.95
            
            # Simple AI for non-player agents
            if agent.agent_id != self._current_agent_id:
                self._simple_ai(agent)
    
    def _simple_ai(self, agent: GameAgent):
        """Simple AI behavior for non-player agents."""
        # Random movement
        if random.random() < 0.1:
            agent.vel_x += random.uniform(-3, 3)
        
        # Random jump
        if agent.is_grounded and random.random() < 0.05:
            agent.vel_y = -10
            agent.is_grounded = False
        
        # Aim at player
        player = self._agents.get(self._current_agent_id)
        if player and player.is_alive:
            agent.aim_x = player.x
            agent.aim_y = player.y
            
            # Occasionally shoot
            if random.random() < 0.02:
                dx = player.x - agent.x
                dy = player.y - agent.y
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist < 300:
                    # Attack player
                    wpn = WEAPONS[agent.weapon]
                    if wpn["ammo"] < 0 or agent.ammo[agent.weapon] > 0:
                        if wpn["ammo"] > 0:
                            agent.ammo[agent.weapon] -= 1
                        
                        # Check hit (simplified)
                        if dist < 200 and random.random() < 0.3:
                            damage = wpn["damage"]
                            player.health -= damage
                            
                            if player.health <= 0:
                                player.is_alive = False
                                agent.score += 1
                                self._kill_events.append({
                                    "killer_id": agent.agent_id,
                                    "victim_id": self._current_agent_id,
                                    "weapon": agent.weapon,
                                    "tick": self._tick,
                                })
    
    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Handle non-MCP actions.
        
        Args:
            action: The action to execute
            timeout_s: Optional timeout
            **kwargs: Additional arguments
        
        Returns:
            Observation with error for unknown action types
        """
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use ListToolsAction or CallToolAction for MCP interactions."
            },
        )
    
    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Execute a step in the environment.
        
        Args:
            action: The MCP action to execute
            timeout_s: Optional timeout
            **kwargs: Additional arguments
        
        Returns:
            Observation from the action execution
        """
        # Increment step count
        self._state.step_count += 1
        
        if self._use_real_server:
            # Real server mode: execute step on actual Teeworlds server
            self._execute_real_step()
            
            # Calculate reward from real game state
            reward = self._calculate_reward_real()
            
            # Check done conditions
            done = self._state.step_count >= self._max_steps
            
            # Check if our bot is dead
            if self._bot_manager:
                my_char = self._bot_manager.game_state.get_character(self._current_agent_id)
                if my_char is None:
                    # Character not in snapshot = dead
                    done = True
                    reward -= 5.0
                
                # Check for kill events this step
                for event in self._bot_manager.game_state.kill_events:
                    if event.killer_id == self._current_agent_id:
                        reward += 1.0  # We killed someone
        else:
            # Simulation mode
            self._simulate_tick()
            
            # Calculate reward
            reward = self._calculate_reward()
            
            # Check done
            done = self._state.step_count >= self._max_steps
            
            # Check if all enemies dead (win condition)
            enemies_alive = sum(1 for a in self._agents.values() 
                              if a.agent_id != self._current_agent_id and a.is_alive)
            if enemies_alive == 0:
                done = True
                reward += 10.0  # Win bonus
            
            # Check if player dead
            player = self._agents.get(self._current_agent_id)
            if player and not player.is_alive:
                done = True
                reward -= 5.0  # Death penalty
        
        # Let the base class handle MCP actions
        obs = super().step(action, timeout_s=timeout_s, **kwargs)
        
        # Update observation with reward and done
        obs.reward = reward
        obs.done = done
        obs.metadata["step"] = self._state.step_count
        obs.metadata["tick"] = self._tick
        obs.metadata["mode"] = "real_server" if self._use_real_server else "simulation"
        
        return obs
    
    def _calculate_reward(self) -> float:
        """Calculate reward for current step (simulation mode)."""
        reward = 0.0
        
        player = self._agents.get(self._current_agent_id)
        if not player:
            return reward
        
        # Survival bonus
        if player.is_alive:
            reward += 0.01
        
        # Kill bonus (from recent events)
        for event in self._kill_events:
            if event["tick"] == self._tick:
                if event["killer_id"] == self._current_agent_id:
                    reward += 1.0
                elif event["victim_id"] == self._current_agent_id:
                    reward -= 0.5
        
        return reward
    
    def _calculate_reward_real(self) -> float:
        """Calculate reward for current step (real server mode)."""
        reward = 0.0
        
        if not self._bot_manager:
            return reward
        
        gs = self._bot_manager.game_state
        my_char = gs.get_character(self._current_agent_id)
        
        # Survival bonus
        if my_char is not None:
            reward += 0.01
        
        # Kill/death events are already handled in step() method
        # The step() adds +1.0 for kills and -5.0 for death
        
        return reward
    
    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
    
    def call_tool_sync(self, name: str, **kwargs) -> str:
        """
        Call a tool synchronously (for notebooks/Colab).
        
        Args:
            name: Tool name (move, jump, aim, shoot, hook, get_status)
            **kwargs: Arguments for the tool
            
        Returns:
            Tool result as string
        """
        if name not in self._tool_fns:
            return f"Unknown tool: {name}"
        return self._tool_fns[name](**kwargs)
    
    def close(self):
        """Clean up resources and disconnect from server."""
        if self._bot_manager:
            self._bot_manager.disconnect()
            self._bot_manager = None
    
    def __del__(self):
        """Destructor - ensure cleanup."""
        self.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
