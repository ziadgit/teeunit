"""
TeeUnit Data Models

Defines the Action, Observation, and State dataclasses for the
multi-agent arena environment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ActionType(str, Enum):
    """Valid action types for agents."""
    MOVE = "move"
    SHOOT = "shoot"
    SWITCH_WEAPON = "switch_weapon"
    USE_ITEM = "use_item"
    WAIT = "wait"


class Direction(str, Enum):
    """Movement directions."""
    NORTH = "n"
    SOUTH = "s"
    EAST = "e"
    WEST = "w"
    NORTHEAST = "ne"
    NORTHWEST = "nw"
    SOUTHEAST = "se"
    SOUTHWEST = "sw"


class WeaponType(str, Enum):
    """Available weapons."""
    PISTOL = "pistol"
    SHOTGUN = "shotgun"
    LASER = "laser"
    HAMMER = "hammer"


class TerrainType(str, Enum):
    """Arena terrain types."""
    EMPTY = "empty"
    WALL = "wall"
    WATER = "water"
    PLATFORM = "platform"


class PickupType(str, Enum):
    """Collectible pickup types."""
    HEALTH = "health"
    ARMOR = "armor"
    SHOTGUN_AMMO = "shotgun_ammo"
    LASER_AMMO = "laser_ammo"


@dataclass
class TeeAction:
    """
    Action submitted by an agent each turn.
    
    Attributes:
        agent_id: Which agent is taking this action (0-3)
        action_type: Type of action to perform
        direction: Movement direction (required for 'move' action)
        target_x: X coordinate for shooting
        target_y: Y coordinate for shooting
        weapon: Weapon to switch to (required for 'switch_weapon' action)
    """
    agent_id: int
    action_type: str  # ActionType value
    direction: Optional[str] = None  # Direction value
    target_x: Optional[int] = None
    target_y: Optional[int] = None
    weapon: Optional[str] = None  # WeaponType value
    
    def __post_init__(self):
        """Validate action parameters."""
        if self.agent_id < 0 or self.agent_id > 3:
            raise ValueError(f"agent_id must be 0-3, got {self.agent_id}")
        
        valid_actions = [a.value for a in ActionType]
        if self.action_type not in valid_actions:
            raise ValueError(f"action_type must be one of {valid_actions}, got {self.action_type}")
        
        if self.action_type == ActionType.MOVE.value and self.direction is None:
            raise ValueError("direction is required for 'move' action")
        
        if self.action_type == ActionType.SHOOT.value:
            if self.target_x is None or self.target_y is None:
                raise ValueError("target_x and target_y are required for 'shoot' action")
        
        if self.action_type == ActionType.SWITCH_WEAPON.value and self.weapon is None:
            raise ValueError("weapon is required for 'switch_weapon' action")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "agent_id": self.agent_id,
            "action_type": self.action_type,
            "direction": self.direction,
            "target_x": self.target_x,
            "target_y": self.target_y,
            "weapon": self.weapon,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TeeAction":
        """Create from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            action_type=data["action_type"],
            direction=data.get("direction"),
            target_x=data.get("target_x"),
            target_y=data.get("target_y"),
            weapon=data.get("weapon"),
        )


@dataclass
class VisibleEnemy:
    """Information about a visible enemy."""
    agent_id: int
    position: Tuple[int, int]
    health_estimate: int  # Approximate, may not be exact
    distance: float
    direction: str  # Relative direction from observer
    
    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "position": list(self.position),
            "health_estimate": self.health_estimate,
            "distance": self.distance,
            "direction": self.direction,
        }


@dataclass
class VisiblePickup:
    """Information about a visible pickup."""
    pickup_type: str
    position: Tuple[int, int]
    distance: float
    
    def to_dict(self) -> dict:
        return {
            "pickup_type": self.pickup_type,
            "position": list(self.position),
            "distance": self.distance,
        }


@dataclass
class TeeObservation:
    """
    Observation returned to an agent after each step.
    
    Contains partial information based on the agent's vision radius
    and line-of-sight.
    
    Attributes:
        agent_id: Which agent this observation is for
        position: Agent's current (x, y) position
        health: Current health (0-100)
        armor: Current armor (0-100)
        current_weapon: Currently equipped weapon
        ammo: Ammo count per weapon type
        visible_enemies: List of enemies in line-of-sight
        visible_pickups: List of pickups in line-of-sight
        nearby_obstacles: Wall positions within vision
        recent_events: Combat log entries from this turn
        turn_number: Current game turn
        your_kills: Agent's total kills this match
        your_deaths: Agent's total deaths this match
        is_alive: Whether agent is currently alive
        spawn_protection: Turns of spawn protection remaining
        text_description: Natural language summary for LLM
    """
    agent_id: int
    position: Tuple[int, int]
    health: int
    armor: int
    current_weapon: str
    ammo: Dict[str, int]
    visible_enemies: List[VisibleEnemy]
    visible_pickups: List[VisiblePickup]
    nearby_obstacles: List[Tuple[int, int]]
    recent_events: List[str]
    turn_number: int
    your_kills: int
    your_deaths: int
    is_alive: bool
    spawn_protection: int
    text_description: str
    episode_id: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "agent_id": self.agent_id,
            "position": list(self.position),
            "health": self.health,
            "armor": self.armor,
            "current_weapon": self.current_weapon,
            "ammo": self.ammo,
            "visible_enemies": [e.to_dict() for e in self.visible_enemies],
            "visible_pickups": [p.to_dict() for p in self.visible_pickups],
            "nearby_obstacles": [list(o) for o in self.nearby_obstacles],
            "recent_events": self.recent_events,
            "turn_number": self.turn_number,
            "your_kills": self.your_kills,
            "your_deaths": self.your_deaths,
            "is_alive": self.is_alive,
            "spawn_protection": self.spawn_protection,
            "text_description": self.text_description,
            "episode_id": self.episode_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TeeObservation":
        """Create from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            position=tuple(data["position"]),
            health=data["health"],
            armor=data["armor"],
            current_weapon=data["current_weapon"],
            ammo=data["ammo"],
            visible_enemies=[
                VisibleEnemy(
                    agent_id=e["agent_id"],
                    position=tuple(e["position"]),
                    health_estimate=e["health_estimate"],
                    distance=e["distance"],
                    direction=e["direction"],
                )
                for e in data["visible_enemies"]
            ],
            visible_pickups=[
                VisiblePickup(
                    pickup_type=p["pickup_type"],
                    position=tuple(p["position"]),
                    distance=p["distance"],
                )
                for p in data["visible_pickups"]
            ],
            nearby_obstacles=[tuple(o) for o in data["nearby_obstacles"]],
            recent_events=data["recent_events"],
            turn_number=data["turn_number"],
            your_kills=data["your_kills"],
            your_deaths=data["your_deaths"],
            is_alive=data["is_alive"],
            spawn_protection=data["spawn_protection"],
            text_description=data["text_description"],
            episode_id=data.get("episode_id", ""),
        )


@dataclass
class AgentScore:
    """Score tracking for a single agent."""
    kills: int = 0
    deaths: int = 0
    damage_dealt: int = 0
    damage_taken: int = 0
    pickups_collected: int = 0
    
    def to_dict(self) -> dict:
        return {
            "kills": self.kills,
            "deaths": self.deaths,
            "damage_dealt": self.damage_dealt,
            "damage_taken": self.damage_taken,
            "pickups_collected": self.pickups_collected,
        }


@dataclass
class TeeState:
    """
    Episode state metadata.
    
    Attributes:
        episode_id: Unique identifier for this match
        step_count: Total steps taken across all agents
        current_turn: Current game turn number
        agents_alive: List of agent IDs currently alive
        scores: Score tracking per agent
        game_over: Whether the match has ended
        winner: Winning agent ID (if game_over)
        max_turns: Maximum turns before match ends
        config: Configuration used for this match
    """
    episode_id: str
    step_count: int
    current_turn: int
    agents_alive: List[int]
    scores: Dict[int, AgentScore]
    game_over: bool
    winner: Optional[int]
    max_turns: int = 100
    config: Dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "current_turn": self.current_turn,
            "agents_alive": self.agents_alive,
            "scores": {k: v.to_dict() for k, v in self.scores.items()},
            "game_over": self.game_over,
            "winner": self.winner,
            "max_turns": self.max_turns,
            "config": self.config,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TeeState":
        """Create from dictionary."""
        return cls(
            episode_id=data["episode_id"],
            step_count=data["step_count"],
            current_turn=data["current_turn"],
            agents_alive=data["agents_alive"],
            scores={
                int(k): AgentScore(**v) for k, v in data["scores"].items()
            },
            game_over=data["game_over"],
            winner=data.get("winner"),
            max_turns=data.get("max_turns", 100),
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
        arena_width: Width of the arena in cells
        arena_height: Height of the arena in cells
        max_turns: Maximum turns before match ends
        num_agents: Number of agents (default 4)
        vision_radius: How far agents can see
        spawn_protection_turns: Invulnerability after respawn
        pickup_respawn_turns: Turns before pickups respawn
        win_kill_threshold: Kills to win early (0 = disabled)
        water_damage: Damage per turn in water
    """
    arena_width: int = 20
    arena_height: int = 20
    max_turns: int = 100
    num_agents: int = 4
    vision_radius: int = 6
    spawn_protection_turns: int = 3
    pickup_respawn_turns: int = 10
    win_kill_threshold: int = 10
    water_damage: int = 5
    
    def to_dict(self) -> dict:
        return {
            "arena_width": self.arena_width,
            "arena_height": self.arena_height,
            "max_turns": self.max_turns,
            "num_agents": self.num_agents,
            "vision_radius": self.vision_radius,
            "spawn_protection_turns": self.spawn_protection_turns,
            "pickup_respawn_turns": self.pickup_respawn_turns,
            "win_kill_threshold": self.win_kill_threshold,
            "water_damage": self.water_damage,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "GameConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
