"""
TeeUnit Environment

The main game environment wrapping real Teeworlds server.
Implements reset(), step(), and state() using bot_manager.
"""

import uuid
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..models import (
    GameConfig,
    StepResult,
    TeeInput,
    TeeObservation,
    TeeState,
    VisiblePlayer,
    VisibleProjectile,
    VisiblePickup,
    KillEvent,
    WeaponType,
    WEAPON_NAMES,
)
from ..protocol.objects import PlayerInput
from .bot_manager import BotManager, GameState

logger = logging.getLogger(__name__)


def tee_input_to_player_input(tee_input: TeeInput, fire_count: int = 0) -> PlayerInput:
    """Convert TeeInput to protocol PlayerInput."""
    return PlayerInput(
        direction=tee_input.direction,
        target_x=tee_input.target_x,
        target_y=tee_input.target_y,
        jump=tee_input.jump,
        fire=fire_count if tee_input.fire else 0,
        hook=tee_input.hook,
        wanted_weapon=tee_input.wanted_weapon,
    )


class TeeEnvironment:
    """
    The TeeUnit multi-agent arena environment.
    
    Wraps a real Teeworlds server via BotManager.
    Implements the OpenEnv interface:
    - reset(): Start a new match (reconnect bots)
    - step(agent_id, action): Execute an agent's action  
    - step_all(actions): Execute actions for all agents
    - state(): Get current episode state
    """
    
    def __init__(self, config: Optional[GameConfig] = None, auto_connect: bool = True):
        """
        Initialize the environment.
        
        Args:
            config: Game configuration (uses defaults if not provided)
            auto_connect: Whether to connect to server immediately
        """
        self.config = config or GameConfig()
        
        # Bot manager handles connections to Teeworlds server
        self.bot_manager = BotManager(
            host=self.config.server_host,
            port=self.config.server_port,
            num_bots=self.config.num_agents,
            ticks_per_step=self.config.ticks_per_step,
        )
        
        # Episode state
        self.episode_id = ""
        self.step_count = 0
        self.game_over = False
        self.winner: Optional[int] = None
        
        # Fire counters per bot (increment to fire)
        self._fire_counts: Dict[int, int] = {i: 0 for i in range(self.config.num_agents)}
        
        # Track kills/deaths per episode
        self._episode_kills: Dict[int, int] = {}
        self._episode_deaths: Dict[int, int] = {}
        
        # Connected state
        self._connected = False
        
        if auto_connect:
            self.connect()
    
    def connect(self, timeout: float = 10.0) -> bool:
        """
        Connect to the Teeworlds server.
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            True if all bots connected successfully
        """
        if self._connected:
            return True
        
        logger.info(f"Connecting to Teeworlds server at {self.config.server_host}:{self.config.server_port}")
        
        self._connected = self.bot_manager.connect(timeout=timeout)
        
        if self._connected:
            logger.info("All bots connected successfully")
        else:
            logger.error("Failed to connect all bots")
        
        return self._connected
    
    def disconnect(self):
        """Disconnect from the server."""
        self.bot_manager.disconnect()
        self._connected = False
    
    def reset(self, config: Optional[Dict] = None) -> StepResult:
        """
        Reset the environment to start a new match.
        
        For a real Teeworlds server, this means:
        - Reconnecting bots if needed
        - Resetting episode tracking
        - The server handles respawning
        
        Args:
            config: Optional configuration overrides
        
        Returns:
            StepResult with initial observation for agent 0
        """
        # Apply config overrides
        if config:
            new_config = GameConfig.from_dict({**self.config.to_dict(), **config})
            
            # If server changed, need to reconnect
            if (new_config.server_host != self.config.server_host or 
                new_config.server_port != self.config.server_port or
                new_config.num_agents != self.config.num_agents):
                self.disconnect()
                self.config = new_config
                self.bot_manager = BotManager(
                    host=self.config.server_host,
                    port=self.config.server_port,
                    num_bots=self.config.num_agents,
                    ticks_per_step=self.config.ticks_per_step,
                )
            else:
                self.config = new_config
                self.bot_manager.ticks_per_step = self.config.ticks_per_step
        
        # Ensure connected
        if not self._connected:
            self.connect()
        
        # Reset episode state
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.game_over = False
        self.winner = None
        self._fire_counts = {i: 0 for i in range(self.config.num_agents)}
        self._episode_kills = {i: 0 for i in range(self.config.num_agents)}
        self._episode_deaths = {i: 0 for i in range(self.config.num_agents)}
        
        # Pump to get initial state
        self.bot_manager.pump()
        
        # Return initial observation for agent 0
        observation = self._build_observation(0)
        return StepResult(
            observation=observation,
            reward=0.0,
            done=False,
            info={"episode_id": self.episode_id, "tick": self.bot_manager.current_tick},
        )
    
    def step(self, agent_id: int, action: TeeInput) -> StepResult:
        """
        Execute an action for a single agent.
        
        Note: In the real game, this waits for ticks_per_step ticks.
        For better performance with multiple agents, use step_all().
        
        Args:
            agent_id: Which agent is acting
            action: The action to execute
        
        Returns:
            StepResult with new observation, reward, done flag
        """
        return self.step_all({agent_id: action})[agent_id]
    
    def step_all(self, actions: Dict[int, TeeInput]) -> Dict[int, StepResult]:
        """
        Execute actions for all agents simultaneously.
        
        Waits for ticks_per_step game ticks while sending inputs.
        
        Args:
            actions: Dict mapping agent_id to their TeeInput
        
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
        
        # Convert TeeInput to PlayerInput
        player_inputs: Dict[int, PlayerInput] = {}
        for agent_id, action in actions.items():
            if action.fire:
                self._fire_counts[agent_id] += 1
            player_inputs[agent_id] = tee_input_to_player_input(
                action, 
                self._fire_counts[agent_id]
            )
        
        # Execute step (waits for ticks_per_step ticks)
        game_state = self.bot_manager.step(player_inputs)
        self.step_count += 1
        
        # Track kills from this step
        for kill in game_state.kill_events:
            if kill.killer_id in self._episode_kills:
                self._episode_kills[kill.killer_id] += 1
            if kill.victim_id in self._episode_deaths:
                self._episode_deaths[kill.victim_id] += 1
        
        # Check game over conditions
        self._check_game_over()
        
        # Build results for all agents
        results = {}
        for agent_id in range(self.config.num_agents):
            # Calculate reward
            reward = self._calculate_reward(agent_id, game_state)
            
            results[agent_id] = StepResult(
                observation=self._build_observation(agent_id),
                reward=reward,
                done=self.game_over,
                info={
                    "tick": game_state.tick,
                    "step": self.step_count,
                    "kills_this_step": len([k for k in game_state.kill_events if k.killer_id == agent_id]),
                    "deaths_this_step": len([k for k in game_state.kill_events if k.victim_id == agent_id]),
                    "winner": self.winner if self.game_over else None,
                },
            )
        
        return results
    
    def state(self) -> TeeState:
        """Get current episode state."""
        return TeeState(
            episode_id=self.episode_id,
            tick=self.bot_manager.current_tick,
            step_count=self.step_count,
            agents_alive=self.bot_manager.get_alive_bots(),
            scores=self.bot_manager.get_scores(),
            game_over=self.game_over,
            winner=self.winner,
            ticks_per_step=self.config.ticks_per_step,
            config=self.config.to_dict(),
        )
    
    def get_observation(self, agent_id: int) -> TeeObservation:
        """Get observation for a specific agent."""
        return self._build_observation(agent_id)
    
    def _build_observation(self, agent_id: int) -> TeeObservation:
        """Build observation for a specific agent."""
        bot = self.bot_manager.get_bot_state(agent_id)
        game_state = self.bot_manager.game_state
        
        # Check if bot exists and has character data
        if not bot or not bot.character:
            return TeeObservation.dead(
                agent_id=agent_id,
                tick=game_state.tick,
                episode_id=self.episode_id,
            )
        
        char = bot.character
        info = bot.player_info
        
        # Check if alive
        is_alive = info is None or not info.is_dead
        
        if not is_alive:
            return TeeObservation.dead(
                agent_id=agent_id,
                tick=game_state.tick,
                episode_id=self.episode_id,
            )
        
        # Build visible players list (other players)
        visible_players = []
        for client_id, other_char in game_state.characters.items():
            if client_id == agent_id:
                continue
            
            other_info = game_state.player_infos.get(client_id)
            score = other_info.score if other_info else 0
            is_hooking = other_char.hook_state > 0
            
            visible_players.append(VisiblePlayer(
                client_id=client_id,
                x=other_char.x,
                y=other_char.y,
                vel_x=other_char.vel_x,
                vel_y=other_char.vel_y,
                health=other_char.health,
                armor=other_char.armor,
                weapon=other_char.weapon,
                direction=other_char.direction,
                score=score,
                is_hooking=is_hooking,
            ))
        
        # Build projectiles list
        projectiles = [
            VisibleProjectile(
                x=p.x,
                y=p.y,
                vel_x=p.vel_x,
                vel_y=p.vel_y,
                weapon_type=p.type,
            )
            for p in game_state.projectiles
        ]
        
        # Build pickups list
        pickups = [
            VisiblePickup(
                x=p.x,
                y=p.y,
                pickup_type=p.type,
            )
            for p in game_state.pickups
        ]
        
        # Build kill events
        recent_kills = [
            KillEvent(
                killer_id=k.killer_id,
                victim_id=k.victim_id,
                weapon=k.weapon,
                tick=k.tick,
            )
            for k in game_state.kill_events
        ]
        
        # Determine if grounded (vel_y near 0 and not recently changed)
        is_grounded = abs(char.vel_y) < 10
        
        # Build text description for LLM
        text_description = self._build_text_description(
            agent_id, char, info, visible_players, recent_kills
        )
        
        return TeeObservation(
            agent_id=agent_id,
            tick=game_state.tick,
            x=char.x,
            y=char.y,
            vel_x=char.vel_x,
            vel_y=char.vel_y,
            health=char.health,
            armor=char.armor,
            weapon=char.weapon,
            ammo=char.ammo_count,
            direction=char.direction,
            is_grounded=is_grounded,
            is_alive=True,
            score=info.score if info else 0,
            visible_players=visible_players,
            projectiles=projectiles,
            pickups=pickups,
            recent_kills=recent_kills,
            episode_id=self.episode_id,
            text_description=text_description,
        )
    
    def _build_text_description(
        self,
        agent_id: int,
        char,
        info,
        visible_players: List[VisiblePlayer],
        recent_kills: List[KillEvent],
    ) -> str:
        """Build natural language description for LLM agents."""
        lines = []
        
        # Status line
        weapon_name = WEAPON_NAMES.get(char.weapon, "unknown")
        score = info.score if info else 0
        
        lines.append(f"Tick {self.bot_manager.current_tick} | Position: ({char.x}, {char.y})")
        lines.append(f"Health: {char.health}/10 | Armor: {char.armor}/10")
        lines.append(f"Weapon: {weapon_name} | Ammo: {char.ammo_count}")
        lines.append(f"Velocity: ({char.vel_x}, {char.vel_y}) | Direction: {char.direction}")
        lines.append(f"Score: {score} kills")
        lines.append("")
        
        # Other players
        if visible_players:
            lines.append("OTHER PLAYERS:")
            for p in sorted(visible_players, key=lambda x: x.distance_to(char.x, char.y)):
                dist = int(p.distance_to(char.x, char.y))
                weapon = WEAPON_NAMES.get(p.weapon, "unknown")
                lines.append(f"  - Player {p.client_id}: pos({p.x}, {p.y}), {p.health}HP, {weapon}, {dist} units away")
        else:
            lines.append("OTHER PLAYERS: None visible")
        
        lines.append("")
        
        # Recent kills
        if recent_kills:
            lines.append("RECENT KILLS:")
            for k in recent_kills[-3:]:
                weapon = WEAPON_NAMES.get(k.weapon, "unknown")
                if k.killer_id == agent_id:
                    lines.append(f"  - You killed Player {k.victim_id} with {weapon}")
                elif k.victim_id == agent_id:
                    lines.append(f"  - Player {k.killer_id} killed you with {weapon}")
                else:
                    lines.append(f"  - Player {k.killer_id} killed Player {k.victim_id} with {weapon}")
        
        return "\n".join(lines)
    
    def _calculate_reward(self, agent_id: int, game_state: GameState) -> float:
        """Calculate reward for an agent this step."""
        reward = 0.0
        
        for kill in game_state.kill_events:
            if kill.killer_id == agent_id:
                reward += 10.0  # Kill reward
            if kill.victim_id == agent_id:
                reward -= 5.0   # Death penalty
        
        # Small survival bonus
        if self.bot_manager.is_alive(agent_id):
            reward += 0.1
        
        return reward
    
    def _check_game_over(self):
        """Check if the game should end."""
        # Max steps reached
        if self.config.max_steps > 0 and self.step_count >= self.config.max_steps:
            self.game_over = True
            self.winner = self._determine_winner()
            return
        
        # Win score reached
        if self.config.win_score > 0:
            scores = self.bot_manager.get_scores()
            for agent_id, score in scores.items():
                if score >= self.config.win_score:
                    self.game_over = True
                    self.winner = agent_id
                    return
    
    def _determine_winner(self) -> Optional[int]:
        """Determine winner based on scores."""
        scores = self.bot_manager.get_scores()
        if not scores:
            return None
        
        # Highest score wins
        winner = max(scores, key=scores.get)
        return winner
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
