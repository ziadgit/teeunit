"""
TeeUnit OpenEnv Environment

OpenEnv-compatible environment wrapping the real Teeworlds server.
This is the main class that RL training frameworks interact with.

Supports both:
- Single-agent mode: step(action) for one agent
- Multi-agent mode: step_all(actions) for all agents (self-play)
"""

import uuid
import logging
from typing import Any, Dict, List, Optional

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
    RewardConfig,
    WEAPON_NAMES,
)
from .protocol.objects import PlayerInput
from .server.bot_manager import BotManager, GameState

logger = logging.getLogger(__name__)


class TeeConfig:
    """Configuration for TeeUnit environment."""
    
    def __init__(
        self,
        num_agents: int = 4,
        ticks_per_step: int = 10,
        max_steps: int = 0,
        win_score: int = 0,
        server_host: str = "127.0.0.1",
        server_port: int = 8303,
        reward_config: Optional[RewardConfig] = None,
    ):
        self.num_agents = num_agents
        self.ticks_per_step = ticks_per_step
        self.max_steps = max_steps
        self.win_score = win_score
        self.server_host = server_host
        self.server_port = server_port
        self.reward_config = reward_config or RewardConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_agents": self.num_agents,
            "ticks_per_step": self.ticks_per_step,
            "max_steps": self.max_steps,
            "win_score": self.win_score,
            "server_host": self.server_host,
            "server_port": self.server_port,
        }


def _tee_action_to_player_input(action: TeeAction, fire_count: int = 0) -> PlayerInput:
    """Convert TeeAction to protocol PlayerInput."""
    return PlayerInput(
        direction=action.direction,
        target_x=action.target_x,
        target_y=action.target_y,
        jump=action.jump,
        fire=fire_count if action.fire else 0,
        hook=action.hook,
        wanted_weapon=action.weapon,
    )


class TeeEnvironment:
    """
    OpenEnv-compatible TeeUnit multi-agent environment.
    
    Wraps a real Teeworlds server via BotManager and implements
    the OpenEnv interface for RL training.
    
    Core methods:
    - reset(): Start new episode, return initial observations
    - step(action): Execute one agent's action (single-agent mode)
    - step_all(actions): Execute all agents' actions (multi-agent mode)
    - state: Property returning current TeeState
    
    For self-play RL training, use step_all() to get observations
    and rewards for all agents simultaneously.
    """
    
    def __init__(
        self,
        config: Optional[TeeConfig] = None,
        auto_connect: bool = True,
    ):
        """
        Initialize the TeeUnit environment.
        
        Args:
            config: Environment configuration
            auto_connect: Whether to connect to server immediately
        """
        self.config = config or TeeConfig()
        
        # Bot manager handles Teeworlds protocol
        self.bot_manager = BotManager(
            host=self.config.server_host,
            port=self.config.server_port,
            num_bots=self.config.num_agents,
            ticks_per_step=self.config.ticks_per_step,
        )
        
        # Episode state
        self._episode_id = ""
        self._step_count = 0
        self._game_over = False
        self._winner: Optional[int] = None
        
        # Fire counters (increment to fire)
        self._fire_counts: Dict[int, int] = {}
        
        # Kill/death tracking per episode
        self._episode_kills: Dict[int, int] = {}
        self._episode_deaths: Dict[int, int] = {}
        self._prev_health: Dict[int, int] = {}
        
        # Connection state
        self._connected = False
        
        if auto_connect:
            self._connect()
    
    def _connect(self, timeout: float = 10.0) -> bool:
        """Connect to Teeworlds server."""
        if self._connected:
            return True
        
        logger.info(f"Connecting to Teeworlds at {self.config.server_host}:{self.config.server_port}")
        self._connected = self.bot_manager.connect(timeout=timeout)
        
        if self._connected:
            logger.info("All bots connected")
        else:
            logger.error("Failed to connect bots")
        
        return self._connected
    
    def _disconnect(self):
        """Disconnect from server."""
        self.bot_manager.disconnect()
        self._connected = False
    
    # =========================================================================
    # OpenEnv Interface
    # =========================================================================
    
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TeeMultiObservation:
        """
        Reset the environment to start a new episode.
        
        Args:
            seed: Random seed (unused - Teeworlds handles spawning)
            episode_id: Custom episode ID (generated if not provided)
            **kwargs: Additional reset options
        
        Returns:
            TeeMultiObservation with initial observations for all agents
        """
        # Ensure connected
        if not self._connected:
            self._connect()
        
        # Reset episode state
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._game_over = False
        self._winner = None
        self._fire_counts = {i: 0 for i in range(self.config.num_agents)}
        self._episode_kills = {i: 0 for i in range(self.config.num_agents)}
        self._episode_deaths = {i: 0 for i in range(self.config.num_agents)}
        self._prev_health = {i: 10 for i in range(self.config.num_agents)}
        
        # Pump to get initial state
        self.bot_manager.pump()
        
        # Build observations for all agents
        observations = {}
        for agent_id in range(self.config.num_agents):
            observations[agent_id] = self._build_observation(agent_id, reward=0.0)
        
        return TeeMultiObservation(
            observations=observations,
            done=False,
            reward=0.0,
            metadata={"episode_id": self._episode_id, "tick": self.bot_manager.current_tick},
        )
    
    def step(
        self,
        action: TeeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TeeObservation:
        """
        Execute action for a single agent (single-agent mode).
        
        For multi-agent self-play, use step_all() instead.
        
        Args:
            action: TeeAction to execute (includes agent_id)
            timeout_s: Timeout (unused)
            **kwargs: Additional options
        
        Returns:
            TeeObservation for the acting agent
        """
        results = self.step_all(
            TeeMultiAction(actions={action.agent_id: action}),
            timeout_s=timeout_s,
            **kwargs,
        )
        
        return results.results[action.agent_id].observation
    
    def step_all(
        self,
        actions: TeeMultiAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TeeMultiStepResult:
        """
        Execute actions for all agents simultaneously (multi-agent mode).
        
        This is the main method for self-play RL training.
        
        Args:
            actions: TeeMultiAction containing actions for all agents
            timeout_s: Timeout (unused)
            **kwargs: Additional options
        
        Returns:
            TeeMultiStepResult with observations, rewards, dones for all agents
        """
        if self._game_over:
            # Return terminal observations
            results = {}
            for agent_id in range(self.config.num_agents):
                obs = self._build_observation(agent_id, reward=0.0)
                obs.done = True
                results[agent_id] = TeeStepResult(
                    observation=obs,
                    reward=0.0,
                    done=True,
                    truncated=False,
                    info={"winner": self._winner},
                )
            return TeeMultiStepResult(results=results, state=self.state)
        
        # Convert TeeActions to PlayerInputs
        player_inputs: Dict[int, PlayerInput] = {}
        for agent_id, action in actions.actions.items():
            if action.fire:
                self._fire_counts[agent_id] = self._fire_counts.get(agent_id, 0) + 1
            player_inputs[agent_id] = _tee_action_to_player_input(
                action,
                self._fire_counts.get(agent_id, 0),
            )
        
        # Store previous health for damage calculation
        for agent_id in range(self.config.num_agents):
            bot = self.bot_manager.get_bot_state(agent_id)
            if bot and bot.character:
                self._prev_health[agent_id] = bot.character.health
        
        # Execute step (waits for ticks_per_step ticks)
        game_state = self.bot_manager.step(player_inputs)
        self._step_count += 1
        
        # Track kills
        for kill in game_state.kill_events:
            if kill.killer_id in self._episode_kills:
                self._episode_kills[kill.killer_id] += 1
            if kill.victim_id in self._episode_deaths:
                self._episode_deaths[kill.victim_id] += 1
        
        # Check game over
        self._check_game_over()
        
        # Build results for all agents
        results = {}
        for agent_id in range(self.config.num_agents):
            reward = self._calculate_reward(agent_id, game_state)
            obs = self._build_observation(agent_id, reward=reward)
            obs.done = self._game_over
            
            results[agent_id] = TeeStepResult(
                observation=obs,
                reward=reward,
                done=self._game_over,
                truncated=self._step_count >= self.config.max_steps if self.config.max_steps > 0 else False,
                info={
                    "tick": game_state.tick,
                    "step": self._step_count,
                    "kills_this_step": len([k for k in game_state.kill_events if k.killer_id == agent_id]),
                    "deaths_this_step": len([k for k in game_state.kill_events if k.victim_id == agent_id]),
                    "winner": self._winner if self._game_over else None,
                },
            )
        
        return TeeMultiStepResult(results=results, state=self.state)
    
    @property
    def state(self) -> TeeState:
        """Get current episode state."""
        return TeeState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            tick=self.bot_manager.current_tick,
            agents_alive=self.bot_manager.get_alive_bots(),
            scores=self.bot_manager.get_scores(),
            game_over=self._game_over,
            winner=self._winner,
            ticks_per_step=self.config.ticks_per_step,
            num_agents=self.config.num_agents,
            config=self.config.to_dict(),
        )
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _build_observation(self, agent_id: int, reward: float = 0.0) -> TeeObservation:
        """Build observation for a specific agent."""
        bot = self.bot_manager.get_bot_state(agent_id)
        game_state = self.bot_manager.game_state
        
        # Check if bot has valid character data
        if not bot or not bot.character:
            return TeeObservation.dead(
                agent_id=agent_id,
                tick=game_state.tick,
                episode_id=self._episode_id,
            )
        
        char = bot.character
        info = bot.player_info
        
        # Check if alive
        is_alive = info is None or not info.is_dead
        
        if not is_alive:
            obs = TeeObservation.dead(
                agent_id=agent_id,
                tick=game_state.tick,
                episode_id=self._episode_id,
            )
            obs.reward = reward
            return obs
        
        # Build visible players
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
        
        # Build projectiles
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
        
        # Build pickups
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
        
        # Text description for LLM agents
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
            is_grounded=abs(char.vel_y) < 10,
            is_alive=True,
            score=info.score if info else 0,
            visible_players=visible_players,
            projectiles=projectiles,
            pickups=pickups,
            recent_kills=recent_kills,
            episode_id=self._episode_id,
            text_description=text_description,
            done=False,
            reward=reward,
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
        
        weapon_name = WEAPON_NAMES.get(char.weapon, "unknown")
        score = info.score if info else 0
        
        lines.append(f"Tick {self.bot_manager.current_tick} | Position: ({char.x}, {char.y})")
        lines.append(f"Health: {char.health}/10 | Armor: {char.armor}/10")
        lines.append(f"Weapon: {weapon_name} | Ammo: {char.ammo_count}")
        lines.append(f"Velocity: ({char.vel_x}, {char.vel_y}) | Direction: {char.direction}")
        lines.append(f"Score: {score} kills")
        lines.append("")
        
        if visible_players:
            lines.append("OTHER PLAYERS:")
            for p in sorted(visible_players, key=lambda x: x.distance_to(char.x, char.y)):
                dist = int(p.distance_to(char.x, char.y))
                weapon = WEAPON_NAMES.get(p.weapon, "unknown")
                lines.append(f"  - Player {p.client_id}: pos({p.x}, {p.y}), {p.health}HP, {weapon}, {dist} units away")
        else:
            lines.append("OTHER PLAYERS: None visible")
        
        lines.append("")
        
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
        rc = self.config.reward_config
        reward = 0.0
        
        # Kill rewards
        for kill in game_state.kill_events:
            if kill.killer_id == agent_id:
                reward += rc.kill_reward
            if kill.victim_id == agent_id:
                reward += rc.death_penalty
        
        # Damage dealt (estimated from health changes)
        for client_id, char in game_state.characters.items():
            if client_id != agent_id:
                prev_hp = self._prev_health.get(client_id, 10)
                damage = prev_hp - char.health
                if damage > 0:
                    # Assume this agent dealt the damage if nearby and fired
                    # This is approximate - real damage tracking needs server mods
                    pass  # TODO: Better damage attribution
        
        # Survival bonus
        if self.bot_manager.is_alive(agent_id):
            reward += rc.survival_bonus
        
        # Win/lose bonus at game over
        if self._game_over:
            if self._winner == agent_id:
                reward += rc.win_bonus
            elif self._winner is not None:
                reward += rc.lose_penalty
        
        return reward
    
    def _check_game_over(self):
        """Check if game should end."""
        # Max steps
        if self.config.max_steps > 0 and self._step_count >= self.config.max_steps:
            self._game_over = True
            self._winner = self._determine_winner()
            return
        
        # Win score
        if self.config.win_score > 0:
            scores = self.bot_manager.get_scores()
            for agent_id, score in scores.items():
                if score >= self.config.win_score:
                    self._game_over = True
                    self._winner = agent_id
                    return
    
    def _determine_winner(self) -> Optional[int]:
        """Determine winner based on scores."""
        scores = self.bot_manager.get_scores()
        if not scores:
            return None
        return max(scores, key=scores.get)
    
    # =========================================================================
    # Context Manager
    # =========================================================================
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._disconnect()
        return False
    
    def close(self):
        """Cleanup resources."""
        self._disconnect()
