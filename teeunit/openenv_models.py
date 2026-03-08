"""
TeeUnit OpenEnv Models

Pydantic-based models compatible with the OpenEnv framework.
These replace the dataclass-based models for OpenEnv compatibility.

For RL training, we provide tensor conversion utilities.
"""

from typing import Any, Dict, List, Optional
from enum import IntEnum

import numpy as np
from pydantic import BaseModel, Field, ConfigDict, model_validator


# =============================================================================
# Constants
# =============================================================================

class WeaponType(IntEnum):
    """Teeworlds weapon types (matches protocol constants)."""
    HAMMER = 0
    GUN = 1      # Pistol
    SHOTGUN = 2
    GRENADE = 3
    LASER = 4    # Rifle
    NINJA = 5


WEAPON_NAMES = {
    WeaponType.HAMMER: "hammer",
    WeaponType.GUN: "pistol",
    WeaponType.SHOTGUN: "shotgun",
    WeaponType.GRENADE: "grenade",
    WeaponType.LASER: "laser",
    WeaponType.NINJA: "ninja",
}

# Maximum number of other players we track in observations
MAX_OTHER_PLAYERS = 7  # For 8-player max

# Maximum pickups/projectiles to track
MAX_PICKUPS = 16
MAX_PROJECTILES = 16


# =============================================================================
# OpenEnv Base Classes (inline to avoid import issues)
# We define our own compatible versions that match OpenEnv's interface
# =============================================================================

class Action(BaseModel):
    """Base class for all environment actions (OpenEnv compatible)."""
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the action"
    )


class Observation(BaseModel):
    """Base class for all environment observations (OpenEnv compatible)."""
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    
    done: bool = Field(default=False, description="Whether the episode has terminated")
    reward: Optional[float] = Field(default=None, description="Reward signal from the last action")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the observation"
    )


class State(BaseModel):
    """Base class for environment state (OpenEnv compatible)."""
    
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    
    episode_id: Optional[str] = Field(default=None, description="Unique identifier for the current episode")
    step_count: int = Field(default=0, ge=0, description="Number of steps taken in the current episode")


# =============================================================================
# TeeUnit Action
# =============================================================================

class TeeAction(Action):
    """
    Action for controlling a Tee character in TeeUnit.
    
    This is a multi-agent environment, so actions include agent_id.
    For step_all(), multiple TeeActions are batched together.
    
    Attributes:
        agent_id: Which agent this action is for (0 to num_agents-1)
        direction: Movement direction (-1=left, 0=none, 1=right)
        target_x: Aim X position (relative to player, -1000 to 1000 typical)
        target_y: Aim Y position (relative to player, -1000 to 1000 typical)
        jump: Whether to jump this tick
        fire: Whether to fire weapon
        hook: Whether to use grappling hook
        weapon: Weapon to switch to (0-5, see WeaponType)
    """
    
    agent_id: int = Field(default=0, ge=0, le=7, description="Agent/bot ID (0-7)")
    direction: int = Field(default=0, ge=-1, le=1, description="Movement direction (-1=left, 0=none, 1=right)")
    target_x: int = Field(default=0, description="Aim X position relative to player")
    target_y: int = Field(default=0, description="Aim Y position relative to player")
    jump: bool = Field(default=False, description="Whether to jump")
    fire: bool = Field(default=False, description="Whether to fire weapon")
    hook: bool = Field(default=False, description="Whether to use grappling hook")
    weapon: int = Field(default=0, ge=0, le=5, description="Weapon to switch to (0-5)")
    
    def __init__(self, **data):
        """Initialize with clamped direction."""
        if "direction" in data:
            data["direction"] = max(-1, min(1, data["direction"]))
        super().__init__(**data)
    
    def to_discrete_action(self) -> int:
        """
        Convert to a discrete action index for discrete action space RL.
        
        Action space (18 actions):
        - 0: No action (idle)
        - 1-3: Move (left, none, right)
        - 4-6: Move + Jump
        - 7-9: Move + Fire
        - 10-12: Move + Jump + Fire
        - 13-15: Move + Hook
        - 16-17: Move + Fire + Jump + Hook (combo)
        
        Returns:
            Integer action index (0-17)
        """
        # Base: direction (-1, 0, 1) -> (0, 1, 2)
        base = self.direction + 1
        
        if self.hook:
            return 13 + base
        elif self.jump and self.fire:
            return 10 + base
        elif self.fire:
            return 7 + base
        elif self.jump:
            return 4 + base
        else:
            return base
    
    @classmethod
    def from_discrete_action(cls, action_idx: int, agent_id: int = 0, target_x: int = 100, target_y: int = 0) -> "TeeAction":
        """
        Create TeeAction from discrete action index.
        
        Args:
            action_idx: Discrete action index (0-17)
            agent_id: Which agent this action is for
            target_x: Aim X (default: slightly ahead)
            target_y: Aim Y (default: level)
        
        Returns:
            TeeAction instance
        """
        action_idx = action_idx % 18  # Wrap around
        
        # Decode action
        if action_idx < 3:
            direction = action_idx - 1
            jump, fire, hook = False, False, False
        elif action_idx < 6:
            direction = (action_idx - 3) - 1
            jump, fire, hook = True, False, False
        elif action_idx < 9:
            direction = (action_idx - 6) - 1
            jump, fire, hook = False, True, False
        elif action_idx < 12:
            direction = (action_idx - 9) - 1
            jump, fire, hook = True, True, False
        elif action_idx < 15:
            direction = (action_idx - 12) - 1
            jump, fire, hook = False, False, True
        else:
            direction = (action_idx - 15) - 1
            jump, fire, hook = True, True, True
        
        # Adjust target based on direction
        if direction != 0:
            target_x = direction * abs(target_x) if target_x != 0 else direction * 100
        
        return cls(
            agent_id=agent_id,
            direction=direction,
            target_x=target_x,
            target_y=target_y,
            jump=jump,
            fire=fire,
            hook=hook,
            weapon=0,  # Default weapon
        )


class TeeMultiAction(Action):
    """
    Batched action for all agents in a step.
    
    Used for step_all() which executes actions for all agents simultaneously.
    """
    
    actions: Dict[int, TeeAction] = Field(
        default_factory=dict,
        description="Map of agent_id -> TeeAction"
    )
    
    @classmethod
    def from_list(cls, actions: List[TeeAction]) -> "TeeMultiAction":
        """Create from list of TeeActions."""
        return cls(actions={a.agent_id: a for a in actions})


# =============================================================================
# TeeUnit Observation
# =============================================================================

class VisiblePlayer(BaseModel):
    """Information about another visible player."""
    
    model_config = ConfigDict(extra="forbid")
    
    client_id: int = Field(description="Player's client ID")
    x: int = Field(description="World X position")
    y: int = Field(description="World Y position")
    vel_x: int = Field(default=0, description="Velocity X")
    vel_y: int = Field(default=0, description="Velocity Y")
    health: int = Field(default=10, ge=0, le=10, description="Health (0-10)")
    armor: int = Field(default=0, ge=0, le=10, description="Armor (0-10)")
    weapon: int = Field(default=1, ge=0, le=5, description="Current weapon")
    direction: int = Field(default=0, description="Movement direction")
    score: int = Field(default=0, description="Player's score (kills)")
    is_hooking: bool = Field(default=False, description="Whether player is using hook")
    
    def distance_to(self, x: int, y: int) -> float:
        """Calculate distance to a point."""
        dx = self.x - x
        dy = self.y - y
        return (dx * dx + dy * dy) ** 0.5


class VisibleProjectile(BaseModel):
    """Information about a projectile in flight."""
    
    model_config = ConfigDict(extra="forbid")
    
    x: int = Field(description="World X position")
    y: int = Field(description="World Y position")
    vel_x: int = Field(default=0, description="Velocity X")
    vel_y: int = Field(default=0, description="Velocity Y")
    weapon_type: int = Field(default=0, description="Weapon that fired it")


class VisiblePickup(BaseModel):
    """Information about a pickup on the map."""
    
    model_config = ConfigDict(extra="forbid")
    
    x: int = Field(description="World X position")
    y: int = Field(description="World Y position")
    pickup_type: int = Field(default=0, description="Pickup type")


class KillEvent(BaseModel):
    """A kill event that occurred this step."""
    
    model_config = ConfigDict(extra="forbid")
    
    killer_id: int = Field(description="Killer's client ID")
    victim_id: int = Field(description="Victim's client ID")
    weapon: int = Field(description="Weapon used")
    tick: int = Field(description="Game tick when kill occurred")


class TeeObservation(Observation):
    """
    Observation for a single agent in TeeUnit.
    
    Contains the game state from this agent's perspective.
    Includes methods for converting to fixed-size tensors for RL.
    
    Attributes:
        agent_id: Which agent this observation is for
        tick: Current game tick
        x, y: Agent's position
        vel_x, vel_y: Agent's velocity
        health, armor: Current health/armor (0-10)
        weapon: Currently equipped weapon
        ammo: Ammo for current weapon
        direction: Movement direction
        is_grounded: Whether on ground
        is_alive: Whether alive
        score: Agent's kill count
        visible_players: Other players in the game
        projectiles: Projectiles in flight
        pickups: Pickups on the map
        recent_kills: Kill events this step
    """
    
    agent_id: int = Field(default=0, description="Agent/bot ID")
    tick: int = Field(default=0, ge=0, description="Current game tick")
    
    # Position and velocity
    x: int = Field(default=0, description="World X position")
    y: int = Field(default=0, description="World Y position")
    vel_x: int = Field(default=0, description="Velocity X")
    vel_y: int = Field(default=0, description="Velocity Y")
    
    # Status
    health: int = Field(default=10, ge=0, le=10, description="Health (0-10)")
    armor: int = Field(default=0, ge=0, le=10, description="Armor (0-10)")
    weapon: int = Field(default=1, ge=0, le=5, description="Current weapon")
    ammo: int = Field(default=-1, description="Ammo for current weapon (-1=unlimited)")
    direction: int = Field(default=0, description="Movement direction (-1, 0, 1)")
    is_grounded: bool = Field(default=True, description="Whether on ground")
    is_alive: bool = Field(default=True, description="Whether alive")
    score: int = Field(default=0, ge=0, description="Kill count")
    
    # Visible entities
    visible_players: List[VisiblePlayer] = Field(default_factory=list)
    projectiles: List[VisibleProjectile] = Field(default_factory=list)
    pickups: List[VisiblePickup] = Field(default_factory=list)
    recent_kills: List[KillEvent] = Field(default_factory=list)
    
    # Episode info
    episode_id: str = Field(default="", description="Episode identifier")
    text_description: str = Field(default="", description="Natural language summary for LLM")
    
    def to_tensor(self, normalize: bool = True) -> np.ndarray:
        """
        Convert observation to a fixed-size tensor for RL training.
        
        Tensor layout (total: 13 + 7*10 + 16*4 + 16*3 = 195 floats):
        - Self state: 13 values
        - Other players: MAX_OTHER_PLAYERS * 10 values = 70
        - Projectiles: MAX_PROJECTILES * 4 values = 64
        - Pickups: MAX_PICKUPS * 3 values = 48
        
        Args:
            normalize: Whether to normalize values to reasonable ranges
        
        Returns:
            numpy array of shape (195,)
        """
        # Normalization constants
        POS_SCALE = 1000.0 if normalize else 1.0
        VEL_SCALE = 100.0 if normalize else 1.0
        HP_SCALE = 10.0 if normalize else 1.0
        
        obs = []
        
        # Self state (13 values)
        obs.extend([
            self.x / POS_SCALE,
            self.y / POS_SCALE,
            self.vel_x / VEL_SCALE,
            self.vel_y / VEL_SCALE,
            self.health / HP_SCALE,
            self.armor / HP_SCALE,
            self.weapon / 5.0 if normalize else self.weapon,
            self.ammo / 10.0 if normalize and self.ammo > 0 else (1.0 if self.ammo < 0 else 0.0),
            float(self.direction),
            float(self.is_grounded),
            float(self.is_alive),
            self.score / 10.0 if normalize else self.score,
            float(self.agent_id) / 7.0 if normalize else self.agent_id,
        ])
        
        # Other players (MAX_OTHER_PLAYERS * 10 values)
        for i in range(MAX_OTHER_PLAYERS):
            if i < len(self.visible_players):
                p = self.visible_players[i]
                # Relative position to self
                rel_x = (p.x - self.x) / POS_SCALE
                rel_y = (p.y - self.y) / POS_SCALE
                obs.extend([
                    1.0,  # Valid flag
                    rel_x,
                    rel_y,
                    p.vel_x / VEL_SCALE,
                    p.vel_y / VEL_SCALE,
                    p.health / HP_SCALE,
                    p.armor / HP_SCALE,
                    p.weapon / 5.0 if normalize else p.weapon,
                    float(p.direction),
                    float(p.is_hooking),
                ])
            else:
                obs.extend([0.0] * 10)  # Padding
        
        # Projectiles (MAX_PROJECTILES * 4 values)
        for i in range(MAX_PROJECTILES):
            if i < len(self.projectiles):
                p = self.projectiles[i]
                rel_x = (p.x - self.x) / POS_SCALE
                rel_y = (p.y - self.y) / POS_SCALE
                obs.extend([
                    1.0,  # Valid flag
                    rel_x,
                    rel_y,
                    p.weapon_type / 5.0 if normalize else p.weapon_type,
                ])
            else:
                obs.extend([0.0] * 4)  # Padding
        
        # Pickups (MAX_PICKUPS * 3 values)
        for i in range(MAX_PICKUPS):
            if i < len(self.pickups):
                p = self.pickups[i]
                rel_x = (p.x - self.x) / POS_SCALE
                rel_y = (p.y - self.y) / POS_SCALE
                obs.extend([
                    1.0,  # Valid flag
                    rel_x,
                    rel_y,
                ])
            else:
                obs.extend([0.0] * 3)  # Padding
        
        return np.array(obs, dtype=np.float32)
    
    @classmethod
    def dead(cls, agent_id: int, tick: int = 0, episode_id: str = "") -> "TeeObservation":
        """Create observation for a dead agent."""
        return cls(
            agent_id=agent_id,
            tick=tick,
            x=0, y=0,
            vel_x=0, vel_y=0,
            health=0, armor=0,
            weapon=0, ammo=0,
            direction=0,
            is_grounded=False,
            is_alive=False,
            score=0,
            done=False,
            reward=0.0,
            episode_id=episode_id,
            text_description="You are dead. Waiting to respawn...",
        )
    
    @classmethod
    def tensor_shape(cls) -> tuple:
        """Get the shape of the tensor representation."""
        # 13 self + 70 players + 64 projectiles + 48 pickups = 195
        return (195,)


class TeeMultiObservation(Observation):
    """
    Batched observations for all agents.
    
    Used as return type for step_all() which returns observations
    for all agents simultaneously.
    """
    
    observations: Dict[int, TeeObservation] = Field(
        default_factory=dict,
        description="Map of agent_id -> TeeObservation"
    )
    
    def to_tensor_batch(self, normalize: bool = True) -> np.ndarray:
        """
        Convert all observations to a batched tensor.
        
        Args:
            normalize: Whether to normalize values
        
        Returns:
            numpy array of shape (num_agents, 195)
        """
        if not self.observations:
            return np.zeros((0, 195), dtype=np.float32)
        
        tensors = []
        for agent_id in sorted(self.observations.keys()):
            tensors.append(self.observations[agent_id].to_tensor(normalize))
        
        return np.stack(tensors, axis=0)


# =============================================================================
# TeeUnit State
# =============================================================================

class TeeState(State):
    """
    Episode state for TeeUnit.
    
    Tracks overall match state across all agents.
    """
    
    tick: int = Field(default=0, ge=0, description="Current game tick")
    agents_alive: List[int] = Field(default_factory=list, description="List of alive agent IDs")
    scores: Dict[int, int] = Field(default_factory=dict, description="Map of agent_id -> kills")
    game_over: bool = Field(default=False, description="Whether match has ended")
    winner: Optional[int] = Field(default=None, description="Winning agent ID (if game over)")
    ticks_per_step: int = Field(default=10, ge=1, description="Game ticks per env step")
    num_agents: int = Field(default=4, ge=1, le=8, description="Number of agents in match")
    
    # Config
    config: Dict[str, Any] = Field(default_factory=dict, description="Game configuration")


# =============================================================================
# Step Result (combines observation + reward + done)
# =============================================================================

class TeeStepResult(BaseModel):
    """
    Result of stepping the environment for a single agent.
    
    Combines observation, reward, done flag, and info dict.
    """
    
    model_config = ConfigDict(extra="forbid")
    
    observation: TeeObservation
    reward: float = Field(default=0.0, description="Reward for this step")
    done: bool = Field(default=False, description="Whether episode is done")
    truncated: bool = Field(default=False, description="Whether episode was truncated (time limit)")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional info")


class TeeMultiStepResult(BaseModel):
    """
    Result of step_all() for all agents.
    """
    
    model_config = ConfigDict(extra="forbid")
    
    results: Dict[int, TeeStepResult] = Field(
        default_factory=dict,
        description="Map of agent_id -> TeeStepResult"
    )
    state: TeeState = Field(description="Current episode state")
    
    def to_arrays(self, normalize: bool = True) -> Dict[str, np.ndarray]:
        """
        Convert to numpy arrays for batch RL training.
        
        Returns:
            Dict with keys: 'observations', 'rewards', 'dones', 'truncateds'
        """
        if not self.results:
            return {
                'observations': np.zeros((0, 195), dtype=np.float32),
                'rewards': np.zeros((0,), dtype=np.float32),
                'dones': np.zeros((0,), dtype=bool),
                'truncateds': np.zeros((0,), dtype=bool),
            }
        
        obs_list = []
        rewards = []
        dones = []
        truncateds = []
        
        for agent_id in sorted(self.results.keys()):
            r = self.results[agent_id]
            obs_list.append(r.observation.to_tensor(normalize))
            rewards.append(r.reward)
            dones.append(r.done)
            truncateds.append(r.truncated)
        
        return {
            'observations': np.stack(obs_list, axis=0),
            'rewards': np.array(rewards, dtype=np.float32),
            'dones': np.array(dones, dtype=bool),
            'truncateds': np.array(truncateds, dtype=bool),
        }


# =============================================================================
# Reward Configuration
# =============================================================================

class RewardConfig(BaseModel):
    """
    Configuration for reward shaping in self-play RL.
    """
    
    model_config = ConfigDict(extra="forbid")
    
    kill_reward: float = Field(default=10.0, description="Reward for killing an enemy")
    death_penalty: float = Field(default=-5.0, description="Penalty for dying")
    damage_reward: float = Field(default=0.1, description="Reward per damage dealt")
    survival_bonus: float = Field(default=0.01, description="Per-step bonus for staying alive")
    health_pickup_reward: float = Field(default=0.5, description="Reward for health pickup")
    armor_pickup_reward: float = Field(default=0.3, description="Reward for armor pickup")
    weapon_pickup_reward: float = Field(default=0.2, description="Reward for weapon pickup")
    win_bonus: float = Field(default=50.0, description="Bonus for winning the match")
    lose_penalty: float = Field(default=-25.0, description="Penalty for losing the match")
