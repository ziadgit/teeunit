"""
TeeUnit Agent State Module

Manages per-agent state including position, health, weapons, and stats.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..models import AgentScore, WeaponType


@dataclass
class AgentState:
    """
    State for a single agent in the arena.
    
    Attributes:
        agent_id: Unique identifier (0-3)
        position: Current (x, y) position
        health: Current health (0-100)
        armor: Current armor (0-100)
        current_weapon: Currently equipped weapon
        ammo: Ammo count per weapon type
        is_alive: Whether agent is currently alive
        spawn_protection: Turns of spawn protection remaining
        score: Kill/death/damage tracking
        last_known_positions: Memory of where enemies were last seen
        respawn_turn: Turn when agent will respawn (if dead)
    """
    agent_id: int
    position: Tuple[int, int] = (0, 0)
    health: int = 100
    armor: int = 0
    current_weapon: str = WeaponType.PISTOL.value
    ammo: Dict[str, int] = field(default_factory=dict)
    is_alive: bool = True
    spawn_protection: int = 0
    score: AgentScore = field(default_factory=AgentScore)
    last_known_positions: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)  # agent_id -> (x, y, turn_seen)
    respawn_turn: int = -1
    
    def __post_init__(self):
        """Initialize default ammo if not provided."""
        if not self.ammo:
            self.ammo = {
                WeaponType.PISTOL.value: -1,  # -1 = unlimited
                WeaponType.SHOTGUN.value: 10,
                WeaponType.LASER.value: 5,
                WeaponType.HAMMER.value: -1,  # -1 = unlimited
            }
    
    def reset(self, spawn_position: Tuple[int, int], spawn_protection_turns: int = 3):
        """Reset agent to initial state at spawn position."""
        self.position = spawn_position
        self.health = 100
        self.armor = 0
        self.current_weapon = WeaponType.PISTOL.value
        self.ammo = {
            WeaponType.PISTOL.value: -1,
            WeaponType.SHOTGUN.value: 10,
            WeaponType.LASER.value: 5,
            WeaponType.HAMMER.value: -1,
        }
        self.is_alive = True
        self.spawn_protection = spawn_protection_turns
        self.respawn_turn = -1
        self.last_known_positions = {}
    
    def take_damage(self, damage: int) -> int:
        """
        Apply damage to the agent.
        
        Armor absorbs 50% of incoming damage.
        
        Args:
            damage: Raw damage amount
        
        Returns:
            Actual damage dealt after armor reduction
        """
        if self.spawn_protection > 0:
            return 0  # Invulnerable during spawn protection
        
        if not self.is_alive:
            return 0
        
        # Armor absorbs 50% of damage
        if self.armor > 0:
            armor_absorbed = min(self.armor, damage // 2)
            self.armor -= armor_absorbed
            damage -= armor_absorbed
        
        actual_damage = min(damage, self.health)
        self.health -= actual_damage
        self.score.damage_taken += actual_damage
        
        if self.health <= 0:
            self.health = 0
            self.is_alive = False
        
        return actual_damage
    
    def heal(self, amount: int) -> int:
        """
        Heal the agent.
        
        Args:
            amount: Amount to heal
        
        Returns:
            Actual amount healed
        """
        if not self.is_alive:
            return 0
        
        old_health = self.health
        self.health = min(100, self.health + amount)
        return self.health - old_health
    
    def add_armor(self, amount: int) -> int:
        """
        Add armor to the agent.
        
        Args:
            amount: Amount of armor to add
        
        Returns:
            Actual amount added
        """
        if not self.is_alive:
            return 0
        
        old_armor = self.armor
        self.armor = min(100, self.armor + amount)
        return self.armor - old_armor
    
    def add_ammo(self, weapon: str, amount: int) -> int:
        """
        Add ammo for a weapon.
        
        Args:
            weapon: Weapon type
            amount: Amount to add
        
        Returns:
            Amount added
        """
        if weapon not in self.ammo:
            return 0
        
        if self.ammo[weapon] == -1:  # Unlimited
            return 0
        
        self.ammo[weapon] += amount
        return amount
    
    def can_fire(self) -> bool:
        """Check if agent can fire current weapon."""
        if not self.is_alive:
            return False
        
        ammo = self.ammo.get(self.current_weapon, 0)
        return ammo == -1 or ammo > 0
    
    def consume_ammo(self) -> bool:
        """
        Consume one ammo for current weapon.
        
        Returns:
            True if ammo was consumed, False if no ammo
        """
        if not self.can_fire():
            return False
        
        ammo = self.ammo.get(self.current_weapon, 0)
        if ammo > 0:
            self.ammo[self.current_weapon] -= 1
        
        return True
    
    def switch_weapon(self, weapon: str) -> bool:
        """
        Switch to a different weapon.
        
        Args:
            weapon: Weapon to switch to
        
        Returns:
            True if switch was successful
        """
        valid_weapons = [w.value for w in WeaponType]
        if weapon not in valid_weapons:
            return False
        
        self.current_weapon = weapon
        return True
    
    def move_to(self, new_position: Tuple[int, int]):
        """Move agent to new position."""
        self.position = new_position
    
    def record_enemy_sighting(self, enemy_id: int, position: Tuple[int, int], turn: int):
        """Record last known position of an enemy."""
        self.last_known_positions[enemy_id] = (position[0], position[1], turn)
    
    def get_last_known_position(self, enemy_id: int, current_turn: int, memory_turns: int = 3) -> Optional[Tuple[int, int]]:
        """
        Get last known position of an enemy if within memory.
        
        Args:
            enemy_id: Enemy agent ID
            current_turn: Current game turn
            memory_turns: How many turns to remember
        
        Returns:
            (x, y) position or None if not in memory
        """
        if enemy_id not in self.last_known_positions:
            return None
        
        x, y, turn_seen = self.last_known_positions[enemy_id]
        if current_turn - turn_seen <= memory_turns:
            return (x, y)
        return None
    
    def update_turn(self):
        """Update agent state for new turn."""
        if self.spawn_protection > 0:
            self.spawn_protection -= 1
    
    def die(self, killed_by: Optional[int] = None):
        """Mark agent as dead."""
        self.is_alive = False
        self.health = 0
        self.score.deaths += 1
    
    def respawn(self, spawn_position: Tuple[int, int], spawn_protection_turns: int = 3):
        """Respawn agent at given position."""
        self.position = spawn_position
        self.health = 100
        self.armor = 0
        self.current_weapon = WeaponType.PISTOL.value
        self.ammo = {
            WeaponType.PISTOL.value: -1,
            WeaponType.SHOTGUN.value: 10,
            WeaponType.LASER.value: 5,
            WeaponType.HAMMER.value: -1,
        }
        self.is_alive = True
        self.spawn_protection = spawn_protection_turns
        self.respawn_turn = -1
    
    def register_kill(self):
        """Register a kill for this agent."""
        self.score.kills += 1
    
    def register_damage_dealt(self, damage: int):
        """Register damage dealt to another agent."""
        self.score.damage_dealt += damage
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "position": list(self.position),
            "health": self.health,
            "armor": self.armor,
            "current_weapon": self.current_weapon,
            "ammo": self.ammo,
            "is_alive": self.is_alive,
            "spawn_protection": self.spawn_protection,
            "score": self.score.to_dict(),
        }


class AgentManager:
    """Manages all agents in the game."""
    
    def __init__(self, num_agents: int = 4):
        """Initialize agent manager."""
        self.num_agents = num_agents
        self.agents: Dict[int, AgentState] = {}
        
        for i in range(num_agents):
            self.agents[i] = AgentState(agent_id=i)
    
    def get_agent(self, agent_id: int) -> Optional[AgentState]:
        """Get agent by ID."""
        return self.agents.get(agent_id)
    
    def get_alive_agents(self) -> List[AgentState]:
        """Get all alive agents."""
        return [a for a in self.agents.values() if a.is_alive]
    
    def get_alive_agent_ids(self) -> List[int]:
        """Get IDs of all alive agents."""
        return [a.agent_id for a in self.get_alive_agents()]
    
    def get_agent_positions(self) -> Dict[int, Tuple[int, int]]:
        """Get positions of all alive agents."""
        return {
            a.agent_id: a.position
            for a in self.agents.values()
            if a.is_alive
        }
    
    def get_agent_at_position(self, x: int, y: int) -> Optional[AgentState]:
        """Get alive agent at position, if any."""
        for agent in self.agents.values():
            if agent.is_alive and agent.position == (x, y):
                return agent
        return None
    
    def reset_all(self, spawn_positions: Dict[int, Tuple[int, int]], spawn_protection_turns: int = 3):
        """Reset all agents to initial state."""
        for agent_id, agent in self.agents.items():
            spawn_pos = spawn_positions.get(agent_id, (0, 0))
            agent.reset(spawn_pos, spawn_protection_turns)
            agent.score = AgentScore()  # Reset scores
    
    def update_turn(self):
        """Update all agents for new turn."""
        for agent in self.agents.values():
            agent.update_turn()
    
    def get_scores(self) -> Dict[int, AgentScore]:
        """Get scores for all agents."""
        return {a.agent_id: a.score for a in self.agents.values()}
    
    def check_respawns(self, current_turn: int) -> List[int]:
        """
        Check which agents need to respawn.
        
        Returns:
            List of agent IDs that should respawn
        """
        respawning = []
        for agent in self.agents.values():
            if not agent.is_alive and agent.respawn_turn <= current_turn and agent.respawn_turn != -1:
                respawning.append(agent.agent_id)
        return respawning
    
    def schedule_respawn(self, agent_id: int, respawn_turn: int):
        """Schedule an agent to respawn at given turn."""
        agent = self.get_agent(agent_id)
        if agent:
            agent.respawn_turn = respawn_turn
