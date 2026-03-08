"""
TeeUnit Data Models

Defines the TeeInput, TeeObservation, and TeeState dataclasses for the
multi-agent arena environment wrapping real Teeworlds.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import IntEnum


class WeaponType(IntEnum):
    """Teeworlds weapon types (matches protocol constants)."""
    HAMMER = 0
    GUN = 1      # Pistol
    SHOTGUN = 2
    GRENADE = 3
    LASER = 4    # Rifle
    NINJA = 5


# Weapon names for display
WEAPON_NAMES = {
    WeaponType.HAMMER: "hammer",
    WeaponType.GUN: "pistol",
    WeaponType.SHOTGUN: "shotgun",
    WeaponType.GRENADE: "grenade",
    WeaponType.LASER: "laser",
    WeaponType.NINJA: "ninja",
}


@dataclass
class TeeInput:
    """
    Input for controlling a Tee character.
    
    Maps directly to Teeworlds PlayerInput with LLM-friendly interface.
    
    Attributes:
        direction: Movement direction (-1=left, 0=none, 1=right)
        target_x: Aim X position (world coordinates relative to player)
        target_y: Aim Y position (world coordinates relative to player)
        jump: Whether to jump this tick
        fire: Whether to fire weapon (internally tracked as counter)
        hook: Whether to use grappling hook
        wanted_weapon: Weapon to switch to (0-5, see WeaponType)
    """
    direction: int = 0  # -1 (left), 0 (none), 1 (right)
    target_x: int = 0   # Aim X (relative to player position)
    target_y: int = 0   # Aim Y (relative to player position)
    jump: bool = False
    fire: bool = False
    hook: bool = False
    wanted_weapon: int = 0  # WeaponType value
    
    def __post_init__(self):
        """Clamp direction to valid range."""
        self.direction = max(-1, min(1, self.direction))
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "direction": self.direction,
            "target_x": self.target_x,
            "target_y": self.target_y,
            "jump": self.jump,
            "fire": self.fire,
            "hook": self.hook,
            "wanted_weapon": self.wanted_weapon,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TeeInput":
        """Create from dictionary."""
        return cls(
            direction=data.get("direction", 0),
            target_x=data.get("target_x", 0),
            target_y=data.get("target_y", 0),
            jump=data.get("jump", False),
            fire=data.get("fire", False),
            hook=data.get("hook", False),
            wanted_weapon=data.get("wanted_weapon", 0),
        )
    
    @classmethod
    def move_left(cls) -> "TeeInput":
        """Create input to move left."""
        return cls(direction=-1)
    
    @classmethod
    def move_right(cls) -> "TeeInput":
        """Create input to move right."""
        return cls(direction=1)
    
    @classmethod
    def jump_left(cls) -> "TeeInput":
        """Create input to jump left."""
        return cls(direction=-1, jump=True)
    
    @classmethod
    def jump_right(cls) -> "TeeInput":
        """Create input to jump right."""
        return cls(direction=1, jump=True)
    
    @classmethod
    def fire_at(cls, target_x: int, target_y: int) -> "TeeInput":
        """Create input to fire at a target."""
        return cls(target_x=target_x, target_y=target_y, fire=True)
    
    @classmethod
    def hook_at(cls, target_x: int, target_y: int) -> "TeeInput":
        """Create input to hook at a target."""
        return cls(target_x=target_x, target_y=target_y, hook=True)


@dataclass
class VisiblePlayer:
    """Information about a visible player in the game."""
    client_id: int          # Player's client ID
    x: int                  # World X position
    y: int                  # World Y position
    vel_x: int              # Velocity X (fixed-point)
    vel_y: int              # Velocity Y (fixed-point)
    health: int             # Health (0-10)
    armor: int              # Armor (0-10)
    weapon: int             # Current weapon (WeaponType)
    direction: int          # Movement direction (-1, 0, 1)
    score: int              # Player's score (kills)
    is_hooking: bool        # Whether player is using hook
    
    def to_dict(self) -> dict:
        return {
            "client_id": self.client_id,
            "x": self.x,
            "y": self.y,
            "vel_x": self.vel_x,
            "vel_y": self.vel_y,
            "health": self.health,
            "armor": self.armor,
            "weapon": self.weapon,
            "weapon_name": WEAPON_NAMES.get(self.weapon, "unknown"),
            "direction": self.direction,
            "score": self.score,
            "is_hooking": self.is_hooking,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "VisiblePlayer":
        return cls(
            client_id=data["client_id"],
            x=data["x"],
            y=data["y"],
            vel_x=data.get("vel_x", 0),
            vel_y=data.get("vel_y", 0),
            health=data.get("health", 10),
            armor=data.get("armor", 0),
            weapon=data.get("weapon", 1),
            direction=data.get("direction", 0),
            score=data.get("score", 0),
            is_hooking=data.get("is_hooking", False),
        )
    
    def distance_to(self, x: int, y: int) -> float:
        """Calculate distance to a point."""
        dx = self.x - x
        dy = self.y - y
        return (dx * dx + dy * dy) ** 0.5


@dataclass
class VisibleProjectile:
    """Information about a projectile in flight."""
    x: int              # World X position
    y: int              # World Y position
    vel_x: int          # Velocity X
    vel_y: int          # Velocity Y
    weapon_type: int    # Weapon that fired it
    
    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "vel_x": self.vel_x,
            "vel_y": self.vel_y,
            "weapon_type": self.weapon_type,
        }


@dataclass
class VisiblePickup:
    """Information about a pickup on the map."""
    x: int          # World X position
    y: int          # World Y position
    pickup_type: int  # Pickup type (health, armor, weapon)
    
    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "pickup_type": self.pickup_type,
        }


@dataclass 
class KillEvent:
    """A kill event that occurred."""
    killer_id: int
    victim_id: int
    weapon: int
    tick: int
    
    def to_dict(self) -> dict:
        return {
            "killer_id": self.killer_id,
            "victim_id": self.victim_id,
            "weapon": self.weapon,
            "weapon_name": WEAPON_NAMES.get(self.weapon, "unknown"),
            "tick": self.tick,
        }


@dataclass
class TeeObservation:
    """
    Observation returned to an agent after each step.
    
    Contains game state from the agent's perspective based on
    Teeworlds snapshots.
    
    Attributes:
        agent_id: Which agent/bot this observation is for
        tick: Current game tick
        x: Agent's X position (world coordinates)
        y: Agent's Y position (world coordinates)
        vel_x: Agent's X velocity (fixed-point)
        vel_y: Agent's Y velocity (fixed-point)
        health: Current health (0-10)
        armor: Current armor (0-10)
        weapon: Currently equipped weapon (WeaponType)
        ammo: Ammo for current weapon (-1 = unlimited)
        direction: Movement direction (-1, 0, 1)
        is_grounded: Whether agent is on ground
        is_alive: Whether agent is alive
        score: Agent's score (kills)
        visible_players: Other players visible in snapshot
        projectiles: Projectiles in the game
        pickups: Pickups visible on map
        recent_kills: Kill events this step
        episode_id: Unique episode identifier
        text_description: Natural language summary for LLM
    """
    agent_id: int
    tick: int
    x: int
    y: int
    vel_x: int
    vel_y: int
    health: int
    armor: int
    weapon: int
    ammo: int
    direction: int
    is_grounded: bool
    is_alive: bool
    score: int
    visible_players: List[VisiblePlayer]
    projectiles: List[VisibleProjectile]
    pickups: List[VisiblePickup]
    recent_kills: List[KillEvent]
    episode_id: str = ""
    text_description: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "agent_id": self.agent_id,
            "tick": self.tick,
            "x": self.x,
            "y": self.y,
            "vel_x": self.vel_x,
            "vel_y": self.vel_y,
            "health": self.health,
            "armor": self.armor,
            "weapon": self.weapon,
            "weapon_name": WEAPON_NAMES.get(self.weapon, "unknown"),
            "ammo": self.ammo,
            "direction": self.direction,
            "is_grounded": self.is_grounded,
            "is_alive": self.is_alive,
            "score": self.score,
            "visible_players": [p.to_dict() for p in self.visible_players],
            "projectiles": [p.to_dict() for p in self.projectiles],
            "pickups": [p.to_dict() for p in self.pickups],
            "recent_kills": [k.to_dict() for k in self.recent_kills],
            "episode_id": self.episode_id,
            "text_description": self.text_description,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TeeObservation":
        """Create from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            tick=data["tick"],
            x=data["x"],
            y=data["y"],
            vel_x=data.get("vel_x", 0),
            vel_y=data.get("vel_y", 0),
            health=data["health"],
            armor=data.get("armor", 0),
            weapon=data.get("weapon", 1),
            ammo=data.get("ammo", -1),
            direction=data.get("direction", 0),
            is_grounded=data.get("is_grounded", True),
            is_alive=data.get("is_alive", True),
            score=data.get("score", 0),
            visible_players=[VisiblePlayer.from_dict(p) for p in data.get("visible_players", [])],
            projectiles=[],  # Skip for simplicity
            pickups=[],  # Skip for simplicity
            recent_kills=[],  # Skip for simplicity
            episode_id=data.get("episode_id", ""),
            text_description=data.get("text_description", ""),
        )
    
    @classmethod
    def dead(cls, agent_id: int, tick: int = 0, episode_id: str = "") -> "TeeObservation":
        """Create a dead observation."""
        return cls(
            agent_id=agent_id,
            tick=tick,
            x=0,
            y=0,
            vel_x=0,
            vel_y=0,
            health=0,
            armor=0,
            weapon=0,
            ammo=0,
            direction=0,
            is_grounded=False,
            is_alive=False,
            score=0,
            visible_players=[],
            projectiles=[],
            pickups=[],
            recent_kills=[],
            episode_id=episode_id,
            text_description="You are dead. Waiting to respawn...",
        )


@dataclass
class AgentScore:
    """Score tracking for a single agent."""
    kills: int = 0
    deaths: int = 0
    damage_dealt: int = 0
    
    def to_dict(self) -> dict:
        return {
            "kills": self.kills,
            "deaths": self.deaths,
            "damage_dealt": self.damage_dealt,
        }


@dataclass
class TeeState:
    """
    Episode state metadata.
    
    Attributes:
        episode_id: Unique identifier for this match
        tick: Current game tick
        step_count: Number of environment steps taken
        agents_alive: List of agent IDs currently alive
        scores: Score tracking per agent
        game_over: Whether the match has ended
        winner: Winning agent ID (if game_over)
        ticks_per_step: Game ticks per environment step
        config: Configuration used for this match
    """
    episode_id: str
    tick: int
    step_count: int
    agents_alive: List[int]
    scores: Dict[int, int]  # agent_id -> kills
    game_over: bool
    winner: Optional[int]
    ticks_per_step: int = 10
    config: Dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "episode_id": self.episode_id,
            "tick": self.tick,
            "step_count": self.step_count,
            "agents_alive": self.agents_alive,
            "scores": self.scores,
            "game_over": self.game_over,
            "winner": self.winner,
            "ticks_per_step": self.ticks_per_step,
            "config": self.config,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TeeState":
        """Create from dictionary."""
        return cls(
            episode_id=data["episode_id"],
            tick=data["tick"],
            step_count=data["step_count"],
            agents_alive=data["agents_alive"],
            scores={int(k): v for k, v in data["scores"].items()},
            game_over=data["game_over"],
            winner=data.get("winner"),
            ticks_per_step=data.get("ticks_per_step", 10),
            config=data.get("config", {}),
        )


@dataclass
class StepResult:
    """
    Result of a step action.
    
    Attributes:
        observation: The agent's new observation
        reward: Reward earned this step
        done: Whether the episode has ended
        info: Additional metadata
    """
    observation: TeeObservation
    reward: float
    done: bool
    info: Dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "observation": self.observation.to_dict(),
            "reward": self.reward,
            "done": self.done,
            "info": self.info,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "StepResult":
        return cls(
            observation=TeeObservation.from_dict(data["observation"]),
            reward=data["reward"],
            done=data["done"],
            info=data.get("info", {}),
        )


@dataclass
class GameConfig:
    """
    Configuration for a TeeUnit match.
    
    Attributes:
        num_agents: Number of agents/bots (1-8, default 4)
        ticks_per_step: Game ticks per environment step (default 10 = 200ms)
        max_steps: Maximum steps before match ends (0 = no limit)
        win_score: Score to win early (0 = disabled, uses server config)
        server_host: Teeworlds server host
        server_port: Teeworlds server port
    """
    num_agents: int = 4
    ticks_per_step: int = 10
    max_steps: int = 0  # 0 = no limit (use server timelimit)
    win_score: int = 0  # 0 = use server sv_scorelimit
    server_host: str = "127.0.0.1"
    server_port: int = 8303
    
    def to_dict(self) -> dict:
        return {
            "num_agents": self.num_agents,
            "ticks_per_step": self.ticks_per_step,
            "max_steps": self.max_steps,
            "win_score": self.win_score,
            "server_host": self.server_host,
            "server_port": self.server_port,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "GameConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Legacy compatibility aliases (deprecated, will be removed)
# =============================================================================

# Keep TeeAction as alias for backwards compatibility during transition
TeeAction = TeeInput
