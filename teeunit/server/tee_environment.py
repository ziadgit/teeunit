"""
TeeUnit Environment

The main game environment implementing reset(), step(), and state().
"""

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..models import (
    ActionType,
    AgentScore,
    Direction,
    GameConfig,
    PickupType,
    StepResult,
    TeeAction,
    TeeObservation,
    TeeState,
    VisibleEnemy,
    VisiblePickup,
    WeaponType,
)
from .arena import Arena
from .agent_state import AgentManager, AgentState
from .weapons import (
    calculate_damage,
    calculate_distance,
    estimate_health,
    get_direction_to,
    get_movement_delta,
    get_weapon_stats,
    is_in_range,
)
from .line_of_sight import (
    get_visible_agents,
    get_visible_cells,
    has_line_of_sight,
    get_cells_along_shot,
)


@dataclass
class CombatEvent:
    """A combat event that occurred this turn."""
    event_type: str  # "damage", "kill", "pickup", "respawn", "shot_missed"
    agent_id: int  # Agent involved
    target_id: Optional[int] = None  # Target agent if applicable
    damage: int = 0
    weapon: str = ""
    position: Optional[Tuple[int, int]] = None
    pickup_type: str = ""
    message: str = ""


class TeeEnvironment:
    """
    The TeeUnit multi-agent arena environment.
    
    Implements the OpenEnv interface:
    - reset(): Start a new match
    - step(action): Execute an agent's action
    - state(): Get current episode state
    """
    
    def __init__(self, config: Optional[GameConfig] = None):
        """
        Initialize the environment.
        
        Args:
            config: Game configuration (uses defaults if not provided)
        """
        self.config = config or GameConfig()
        self.arena = Arena(self.config)
        self.agents = AgentManager(self.config.num_agents)
        
        # Episode state
        self.episode_id = ""
        self.current_turn = 0
        self.step_count = 0
        self.game_over = False
        self.winner: Optional[int] = None
        
        # Turn tracking
        self.actions_this_turn: Dict[int, TeeAction] = {}
        self.events_this_turn: List[CombatEvent] = []
        self.pending_respawns: Dict[int, int] = {}  # agent_id -> respawn_turn
    
    def reset(self, config: Optional[Dict] = None) -> StepResult:
        """
        Reset the environment to start a new match.
        
        Args:
            config: Optional configuration overrides
        
        Returns:
            StepResult with initial observation for agent 0
        """
        # Apply config overrides
        if config:
            self.config = GameConfig.from_dict({**self.config.to_dict(), **config})
            self.arena = Arena(self.config)
            self.agents = AgentManager(self.config.num_agents)
        
        # Reset episode state
        self.episode_id = str(uuid.uuid4())
        self.current_turn = 0
        self.step_count = 0
        self.game_over = False
        self.winner = None
        self.actions_this_turn = {}
        self.events_this_turn = []
        self.pending_respawns = {}
        
        # Reset arena
        self.arena.reset()
        
        # Get spawn positions for all agents
        spawn_positions = {}
        occupied = []
        for agent_id in range(self.config.num_agents):
            spawn_pos = self.arena.get_spawn_point(self.current_turn, occupied)
            spawn_positions[agent_id] = spawn_pos
            occupied.append(spawn_pos)
        
        # Reset all agents at spawn positions
        self.agents.reset_all(spawn_positions, self.config.spawn_protection_turns)
        
        # Return initial observation for agent 0
        observation = self._build_observation(0)
        return StepResult(
            observation=observation,
            reward=0.0,
            done=False,
            info={"episode_id": self.episode_id, "turn": 0},
        )
    
    def step(self, action: TeeAction) -> StepResult:
        """
        Execute an action for an agent.
        
        In this implementation, actions are processed immediately.
        For true simultaneous actions, use step_all().
        
        Args:
            action: The action to execute
        
        Returns:
            StepResult with new observation, reward, done flag
        """
        if self.game_over:
            return StepResult(
                observation=self._build_observation(action.agent_id),
                reward=0.0,
                done=True,
                info={"error": "Game is over", "winner": self.winner},
            )
        
        agent = self.agents.get_agent(action.agent_id)
        if not agent:
            return StepResult(
                observation=self._build_observation(action.agent_id),
                reward=0.0,
                done=False,
                info={"error": f"Invalid agent_id: {action.agent_id}"},
            )
        
        # Clear events for this step
        self.events_this_turn = []
        
        # Process the action
        reward = self._process_action(agent, action)
        
        # Increment step count
        self.step_count += 1
        
        # Check if turn should advance (all agents have acted or just sequential)
        # For simplicity, advance turn after every 4 steps
        if self.step_count % self.config.num_agents == 0:
            self._advance_turn()
        
        # Check game over conditions
        self._check_game_over()
        
        # Build observation
        observation = self._build_observation(action.agent_id)
        
        return StepResult(
            observation=observation,
            reward=reward,
            done=self.game_over,
            info={
                "turn": self.current_turn,
                "events": [e.message for e in self.events_this_turn],
                "winner": self.winner if self.game_over else None,
            },
        )
    
    def step_all(self, actions: Dict[int, TeeAction]) -> Dict[int, StepResult]:
        """
        Execute actions for all agents simultaneously.
        
        Args:
            actions: Dict mapping agent_id to their action
        
        Returns:
            Dict mapping agent_id to their StepResult
        """
        if self.game_over:
            return {
                agent_id: StepResult(
                    observation=self._build_observation(agent_id),
                    reward=0.0,
                    done=True,
                    info={"error": "Game is over", "winner": self.winner},
                )
                for agent_id in range(self.config.num_agents)
            }
        
        # Clear events
        self.events_this_turn = []
        
        # Process all actions in order: movement -> weapons -> shooting -> pickups
        rewards = {agent_id: 0.0 for agent_id in range(self.config.num_agents)}
        
        # 1. Process movement actions
        for agent_id, action in actions.items():
            agent = self.agents.get_agent(agent_id)
            if agent and agent.is_alive and action.action_type == ActionType.MOVE.value:
                self._process_move(agent, action.direction)
        
        # 2. Process weapon switches
        for agent_id, action in actions.items():
            agent = self.agents.get_agent(agent_id)
            if agent and agent.is_alive and action.action_type == ActionType.SWITCH_WEAPON.value:
                agent.switch_weapon(action.weapon)
        
        # 3. Process shooting
        for agent_id, action in actions.items():
            agent = self.agents.get_agent(agent_id)
            if agent and agent.is_alive and action.action_type == ActionType.SHOOT.value:
                damage_reward = self._process_shoot(agent, action.target_x, action.target_y)
                rewards[agent_id] += damage_reward
        
        # 4. Process item usage and pickups
        for agent_id in range(self.config.num_agents):
            agent = self.agents.get_agent(agent_id)
            if agent and agent.is_alive:
                pickup = self.arena.collect_pickup(agent.position[0], agent.position[1])
                if pickup:
                    self._apply_pickup(agent, pickup.pickup_type)
        
        # Increment step count
        self.step_count += len(actions)
        
        # Advance turn
        self._advance_turn()
        
        # Check game over
        self._check_game_over()
        
        # Build results for all agents
        results = {}
        for agent_id in range(self.config.num_agents):
            agent = self.agents.get_agent(agent_id)
            
            # Add survival bonus
            if agent and agent.is_alive:
                rewards[agent_id] += 0.1
            
            results[agent_id] = StepResult(
                observation=self._build_observation(agent_id),
                reward=rewards[agent_id],
                done=self.game_over,
                info={
                    "turn": self.current_turn,
                    "events": [e.message for e in self.events_this_turn if e.agent_id == agent_id or e.target_id == agent_id],
                    "winner": self.winner if self.game_over else None,
                },
            )
        
        return results
    
    def state(self) -> TeeState:
        """Get current episode state."""
        return TeeState(
            episode_id=self.episode_id,
            step_count=self.step_count,
            current_turn=self.current_turn,
            agents_alive=self.agents.get_alive_agent_ids(),
            scores=self.agents.get_scores(),
            game_over=self.game_over,
            winner=self.winner,
            max_turns=self.config.max_turns,
            config=self.config.to_dict(),
        )
    
    def get_observation(self, agent_id: int) -> TeeObservation:
        """Get observation for a specific agent."""
        return self._build_observation(agent_id)
    
    def _process_action(self, agent: AgentState, action: TeeAction) -> float:
        """
        Process a single action and return reward.
        
        Args:
            agent: The agent taking the action
            action: The action to process
        
        Returns:
            Reward for this action
        """
        reward = 0.0
        
        if not agent.is_alive:
            # Dead agents can't act
            return reward
        
        action_type = action.action_type
        
        if action_type == ActionType.MOVE.value:
            self._process_move(agent, action.direction)
        
        elif action_type == ActionType.SHOOT.value:
            reward += self._process_shoot(agent, action.target_x, action.target_y)
        
        elif action_type == ActionType.SWITCH_WEAPON.value:
            agent.switch_weapon(action.weapon)
        
        elif action_type == ActionType.USE_ITEM.value:
            # Check for pickup at current position
            pickup = self.arena.collect_pickup(agent.position[0], agent.position[1])
            if pickup:
                self._apply_pickup(agent, pickup.pickup_type)
        
        elif action_type == ActionType.WAIT.value:
            pass  # Do nothing
        
        # Survival bonus
        if agent.is_alive:
            reward += 0.1
        
        return reward
    
    def _process_move(self, agent: AgentState, direction: str):
        """Process a movement action."""
        if not direction:
            return
        
        dx, dy = get_movement_delta(direction)
        new_x = agent.position[0] + dx
        new_y = agent.position[1] + dy
        
        # Check if new position is walkable
        if not self.arena.is_walkable(new_x, new_y):
            return  # Can't move there
        
        # Check for collision with other agents
        other_agent = self.agents.get_agent_at_position(new_x, new_y)
        if other_agent and other_agent.agent_id != agent.agent_id:
            return  # Can't move into another agent
        
        # Move the agent
        agent.move_to((new_x, new_y))
        
        # Check for water damage
        if self.arena.get_terrain(new_x, new_y) == "water":
            damage = self.config.water_damage
            agent.take_damage(damage)
            self.events_this_turn.append(CombatEvent(
                event_type="damage",
                agent_id=agent.agent_id,
                damage=damage,
                message=f"Agent {agent.agent_id} took {damage} damage from water",
            ))
            
            if not agent.is_alive:
                self._handle_death(agent, killer_id=None)
    
    def _process_shoot(self, agent: AgentState, target_x: int, target_y: int) -> float:
        """
        Process a shooting action.
        
        Returns reward from damage dealt and kills.
        """
        reward = 0.0
        
        if not agent.can_fire():
            self.events_this_turn.append(CombatEvent(
                event_type="shot_missed",
                agent_id=agent.agent_id,
                weapon=agent.current_weapon,
                message=f"Agent {agent.agent_id} is out of ammo for {agent.current_weapon}",
            ))
            return reward
        
        # Consume ammo
        agent.consume_ammo()
        
        weapon = agent.current_weapon
        weapon_stats = get_weapon_stats(weapon)
        
        if not weapon_stats:
            return reward
        
        # Check line of sight to target
        if not has_line_of_sight(
            agent.position[0], agent.position[1],
            target_x, target_y,
            lambda x, y: self.arena.is_blocking(x, y),
        ):
            self.events_this_turn.append(CombatEvent(
                event_type="shot_missed",
                agent_id=agent.agent_id,
                weapon=weapon,
                position=(target_x, target_y),
                message=f"Agent {agent.agent_id}'s shot was blocked",
            ))
            return reward
        
        # Check if any agent is at or near the target
        target_agent = self.agents.get_agent_at_position(target_x, target_y)
        
        if target_agent and target_agent.agent_id != agent.agent_id and target_agent.is_alive:
            # Direct hit
            damage = calculate_damage(agent.position, target_agent.position, weapon)
            
            if damage > 0:
                actual_damage = target_agent.take_damage(damage)
                agent.register_damage_dealt(actual_damage)
                reward += actual_damage * 0.1  # Reward per damage
                
                self.events_this_turn.append(CombatEvent(
                    event_type="damage",
                    agent_id=agent.agent_id,
                    target_id=target_agent.agent_id,
                    damage=actual_damage,
                    weapon=weapon,
                    message=f"Agent {agent.agent_id} hit Agent {target_agent.agent_id} for {actual_damage} damage",
                ))
                
                # Check for kill
                if not target_agent.is_alive:
                    agent.register_kill()
                    reward += 10.0  # Kill reward
                    self._handle_death(target_agent, killer_id=agent.agent_id)
        else:
            # Shot missed
            self.events_this_turn.append(CombatEvent(
                event_type="shot_missed",
                agent_id=agent.agent_id,
                weapon=weapon,
                position=(target_x, target_y),
                message=f"Agent {agent.agent_id}'s shot missed",
            ))
        
        return reward
    
    def _apply_pickup(self, agent: AgentState, pickup_type: str):
        """Apply a pickup's effect to an agent."""
        if pickup_type == PickupType.HEALTH.value:
            healed = agent.heal(25)
            if healed > 0:
                agent.score.pickups_collected += 1
                self.events_this_turn.append(CombatEvent(
                    event_type="pickup",
                    agent_id=agent.agent_id,
                    pickup_type=pickup_type,
                    message=f"Agent {agent.agent_id} picked up health (+{healed})",
                ))
        
        elif pickup_type == PickupType.ARMOR.value:
            added = agent.add_armor(50)
            if added > 0:
                agent.score.pickups_collected += 1
                self.events_this_turn.append(CombatEvent(
                    event_type="pickup",
                    agent_id=agent.agent_id,
                    pickup_type=pickup_type,
                    message=f"Agent {agent.agent_id} picked up armor (+{added})",
                ))
        
        elif pickup_type == PickupType.SHOTGUN_AMMO.value:
            stats = get_weapon_stats(WeaponType.SHOTGUN.value)
            added = agent.add_ammo(WeaponType.SHOTGUN.value, stats.ammo_per_pickup if stats else 5)
            if added > 0:
                agent.score.pickups_collected += 1
                self.events_this_turn.append(CombatEvent(
                    event_type="pickup",
                    agent_id=agent.agent_id,
                    pickup_type=pickup_type,
                    message=f"Agent {agent.agent_id} picked up shotgun ammo (+{added})",
                ))
        
        elif pickup_type == PickupType.LASER_AMMO.value:
            stats = get_weapon_stats(WeaponType.LASER.value)
            added = agent.add_ammo(WeaponType.LASER.value, stats.ammo_per_pickup if stats else 3)
            if added > 0:
                agent.score.pickups_collected += 1
                self.events_this_turn.append(CombatEvent(
                    event_type="pickup",
                    agent_id=agent.agent_id,
                    pickup_type=pickup_type,
                    message=f"Agent {agent.agent_id} picked up laser ammo (+{added})",
                ))
    
    def _handle_death(self, agent: AgentState, killer_id: Optional[int]):
        """Handle an agent's death."""
        agent.die(killer_id)
        
        killer_msg = f" by Agent {killer_id}" if killer_id is not None else ""
        self.events_this_turn.append(CombatEvent(
            event_type="kill",
            agent_id=agent.agent_id,
            target_id=killer_id,
            message=f"Agent {agent.agent_id} was eliminated{killer_msg}",
        ))
        
        # Schedule respawn (instant in this version)
        # For delayed respawn, set respawn_turn in the future
        respawn_turn = self.current_turn + 1  # Respawn next turn
        self.pending_respawns[agent.agent_id] = respawn_turn
        
        # Death penalty
        # (Handled by reward calculation)
    
    def _advance_turn(self):
        """Advance to the next turn."""
        self.current_turn += 1
        
        # Update all agents (spawn protection countdown)
        self.agents.update_turn()
        
        # Update pickup respawn timers
        self.arena.update_pickups()
        
        # Handle respawns
        respawning = []
        for agent_id, respawn_turn in list(self.pending_respawns.items()):
            if self.current_turn >= respawn_turn:
                respawning.append(agent_id)
                del self.pending_respawns[agent_id]
        
        for agent_id in respawning:
            agent = self.agents.get_agent(agent_id)
            if agent:
                # Get spawn position away from other agents
                other_positions = [
                    a.position for a in self.agents.get_alive_agents()
                ]
                spawn_pos = self.arena.get_spawn_point(self.current_turn, other_positions)
                agent.respawn(spawn_pos, self.config.spawn_protection_turns)
                
                self.events_this_turn.append(CombatEvent(
                    event_type="respawn",
                    agent_id=agent_id,
                    position=spawn_pos,
                    message=f"Agent {agent_id} respawned at {spawn_pos}",
                ))
    
    def _check_game_over(self):
        """Check if the game should end."""
        # Turn limit reached
        if self.current_turn >= self.config.max_turns:
            self.game_over = True
            self.winner = self._determine_winner()
            return
        
        # Kill threshold reached
        if self.config.win_kill_threshold > 0:
            for agent in self.agents.agents.values():
                if agent.score.kills >= self.config.win_kill_threshold:
                    self.game_over = True
                    self.winner = agent.agent_id
                    return
    
    def _determine_winner(self) -> Optional[int]:
        """Determine the winner based on scores."""
        scores = []
        for agent in self.agents.agents.values():
            scores.append((
                agent.agent_id,
                agent.score.kills,
                -agent.score.deaths,  # Negative so fewer deaths is better
                agent.score.damage_dealt,
            ))
        
        # Sort by kills (desc), deaths (asc), damage (desc)
        scores.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
        
        return scores[0][0] if scores else None
    
    def _build_observation(self, agent_id: int) -> TeeObservation:
        """Build observation for a specific agent."""
        agent = self.agents.get_agent(agent_id)
        
        if not agent:
            # Return a dead observation
            return TeeObservation(
                agent_id=agent_id,
                position=(0, 0),
                health=0,
                armor=0,
                current_weapon=WeaponType.PISTOL.value,
                ammo={},
                visible_enemies=[],
                visible_pickups=[],
                nearby_obstacles=[],
                recent_events=[],
                turn_number=self.current_turn,
                your_kills=0,
                your_deaths=0,
                is_alive=False,
                spawn_protection=0,
                text_description="Invalid agent",
                episode_id=self.episode_id,
            )
        
        # Get visible enemies
        visible_enemies = []
        if agent.is_alive:
            alive_agent_positions = self.agents.get_agent_positions()
            visible_agent_ids = get_visible_agents(
                agent.position,
                alive_agent_positions,
                agent_id,
                self.config.vision_radius,
                lambda x, y: self.arena.is_blocking(x, y),
            )
            
            for enemy_id in visible_agent_ids:
                enemy = self.agents.get_agent(enemy_id)
                if enemy and enemy.is_alive:
                    distance = calculate_distance(agent.position, enemy.position)
                    direction = get_direction_to(agent.position, enemy.position)
                    
                    # Record sighting for memory
                    agent.record_enemy_sighting(enemy_id, enemy.position, self.current_turn)
                    
                    visible_enemies.append(VisibleEnemy(
                        agent_id=enemy_id,
                        position=enemy.position,
                        health_estimate=estimate_health(enemy.health),
                        distance=round(distance, 1),
                        direction=direction,
                    ))
        
        # Get visible pickups
        visible_pickups = []
        if agent.is_alive:
            visible_cells = get_visible_cells(
                agent.position[0], agent.position[1],
                self.config.vision_radius,
                lambda x, y: self.arena.is_blocking(x, y),
                lambda x, y: self.arena._in_bounds(x, y),
            )
            
            for pickup in self.arena.get_available_pickups():
                if pickup.position in visible_cells:
                    distance = calculate_distance(agent.position, pickup.position)
                    visible_pickups.append(VisiblePickup(
                        pickup_type=pickup.pickup_type,
                        position=pickup.position,
                        distance=round(distance, 1),
                    ))
        
        # Get nearby obstacles
        nearby_obstacles = []
        if agent.is_alive:
            nearby_obstacles = self.arena.get_walls_in_area(
                agent.position[0], agent.position[1],
                self.config.vision_radius,
            )
        
        # Get recent events for this agent
        recent_events = [
            e.message for e in self.events_this_turn
            if e.agent_id == agent_id or e.target_id == agent_id
        ]
        
        # Build text description
        text_description = self._build_text_description(agent, visible_enemies, visible_pickups)
        
        return TeeObservation(
            agent_id=agent_id,
            position=agent.position,
            health=agent.health,
            armor=agent.armor,
            current_weapon=agent.current_weapon,
            ammo=dict(agent.ammo),
            visible_enemies=visible_enemies,
            visible_pickups=visible_pickups,
            nearby_obstacles=nearby_obstacles,
            recent_events=recent_events,
            turn_number=self.current_turn,
            your_kills=agent.score.kills,
            your_deaths=agent.score.deaths,
            is_alive=agent.is_alive,
            spawn_protection=agent.spawn_protection,
            text_description=text_description,
            episode_id=self.episode_id,
        )
    
    def _build_text_description(
        self,
        agent: AgentState,
        visible_enemies: List[VisibleEnemy],
        visible_pickups: List[VisiblePickup],
    ) -> str:
        """Build a natural language description for LLM agents."""
        lines = []
        
        # Status line
        status = "DEAD - Respawning soon" if not agent.is_alive else ""
        if agent.spawn_protection > 0:
            status = f"SPAWN PROTECTED ({agent.spawn_protection} turns)"
        
        lines.append(f"Turn {self.current_turn} | Position: {agent.position} | Health: {agent.health}/100 | Armor: {agent.armor}")
        
        # Weapon info
        ammo_str = "unlimited" if agent.ammo.get(agent.current_weapon, 0) == -1 else str(agent.ammo.get(agent.current_weapon, 0))
        lines.append(f"Weapon: {agent.current_weapon.title()} ({ammo_str} ammo)")
        
        if status:
            lines.append(f"Status: {status}")
        
        lines.append("")
        
        # Enemies
        if visible_enemies:
            lines.append("VISIBLE ENEMIES:")
            for enemy in sorted(visible_enemies, key=lambda e: e.distance):
                lines.append(f"  - Agent {enemy.agent_id} at {enemy.position}, ~{enemy.health_estimate} HP, {enemy.distance} cells ({enemy.direction})")
        else:
            lines.append("VISIBLE ENEMIES: None")
        
        lines.append("")
        
        # Pickups
        if visible_pickups:
            lines.append("PICKUPS NEARBY:")
            for pickup in sorted(visible_pickups, key=lambda p: p.distance):
                lines.append(f"  - {pickup.pickup_type.replace('_', ' ').title()} at {pickup.position}, {pickup.distance} cells away")
        else:
            lines.append("PICKUPS NEARBY: None")
        
        lines.append("")
        
        # Recent events
        recent = [e.message for e in self.events_this_turn[-3:]]  # Last 3 events
        if recent:
            lines.append("RECENT EVENTS:")
            for event in recent:
                lines.append(f"  - {event}")
        
        lines.append("")
        
        # Score
        lines.append(f"Your score: {agent.score.kills} kills, {agent.score.deaths} deaths")
        
        return "\n".join(lines)
