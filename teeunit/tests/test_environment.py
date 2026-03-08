"""Tests for TeeUnit environment.

These tests mock the BotManager to test the TeeEnvironment logic
without requiring a real Teeworlds server.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock

from teeunit.models import (
    TeeInput,
    TeeObservation,
    TeeState,
    StepResult,
    GameConfig,
    VisiblePlayer,
    WeaponType,
)
from teeunit.server.tee_environment import TeeEnvironment
from teeunit.server.bot_manager import BotManager, BotState, GameState
from teeunit.protocol.objects import Character, PlayerInfo, PlayerInput
from teeunit.protocol.client import KillEvent


def create_mock_character(
    x: int = 100,
    y: int = 200,
    vel_x: int = 0,
    vel_y: int = 0,
    health: int = 10,
    armor: int = 0,
    weapon: int = 1,
    ammo_count: int = 10,
    direction: int = 1,
    hook_state: int = 0,
) -> Mock:
    """Create a mock Character object."""
    char = Mock(spec=Character)
    char.x = x
    char.y = y
    char.vel_x = vel_x
    char.vel_y = vel_y
    char.health = health
    char.armor = armor
    char.weapon = weapon
    char.ammo_count = ammo_count
    char.direction = direction
    char.hook_state = hook_state
    return char


def create_mock_player_info(
    score: int = 0,
    is_dead: bool = False,
) -> Mock:
    """Create a mock PlayerInfo object."""
    info = Mock(spec=PlayerInfo)
    info.score = score
    info.is_dead = is_dead
    return info


def create_mock_bot_state(
    client_id: int,
    character: Mock = None,
    player_info: Mock = None,
    connected: bool = True,
) -> Mock:
    """Create a mock BotState."""
    bot = Mock(spec=BotState)
    bot.client_id = client_id
    bot.character = character or create_mock_character()
    bot.player_info = player_info or create_mock_player_info()
    bot.connected = connected
    return bot


def setup_mock_bot_manager(env: TeeEnvironment, num_agents: int = 4, tick: int = 100):
    """Set up mock bot manager with common configuration."""
    game_state = GameState(tick=tick)
    game_state.characters = {i: create_mock_character() for i in range(num_agents)}
    game_state.player_infos = {i: create_mock_player_info() for i in range(num_agents)}
    
    env.bot_manager.game_state = game_state
    env.bot_manager.bots = {
        i: create_mock_bot_state(i, game_state.characters[i], game_state.player_infos[i])
        for i in range(num_agents)
    }
    
    def get_bot_state(agent_id):
        return env.bot_manager.bots.get(agent_id)
    env.bot_manager.get_bot_state = get_bot_state
    env.bot_manager.is_alive = Mock(return_value=True)
    env.bot_manager.get_scores = Mock(return_value={i: 0 for i in range(num_agents)})
    env.bot_manager.get_alive_bots = Mock(return_value=list(range(num_agents)))
    
    return game_state


class TestTeeEnvironment:
    """Test TeeEnvironment class."""

    @patch.object(TeeEnvironment, 'connect', return_value=True)
    def test_create_environment(self, mock_connect):
        """Test environment creation."""
        env = TeeEnvironment(auto_connect=False)
        assert env is not None
        assert env.config.num_agents == 4

    @patch.object(TeeEnvironment, 'connect', return_value=True)
    def test_create_with_custom_config(self, mock_connect):
        """Test environment with custom config."""
        config = GameConfig(num_agents=2, ticks_per_step=5, max_steps=100)
        env = TeeEnvironment(config=config, auto_connect=False)
        
        assert env.config.num_agents == 2
        assert env.config.ticks_per_step == 5
        assert env.config.max_steps == 100

    @patch.object(BotManager, 'connect', return_value=True)
    @patch.object(BotManager, 'pump')
    def test_reset(self, mock_pump, mock_connect):
        """Test environment reset."""
        env = TeeEnvironment(auto_connect=False)
        env._connected = True
        
        setup_mock_bot_manager(env, tick=100)
        
        result = env.reset()
        
        assert isinstance(result, StepResult)
        assert result.observation is not None
        assert result.reward == 0.0
        assert result.done is False
        assert env.episode_id != ""
        assert env.step_count == 0

    @patch.object(BotManager, 'connect', return_value=True)
    @patch.object(BotManager, 'pump')
    @patch.object(BotManager, 'step')
    def test_step(self, mock_step, mock_pump, mock_connect):
        """Test stepping with an action."""
        env = TeeEnvironment(auto_connect=False)
        env._connected = True
        
        game_state = setup_mock_bot_manager(env, tick=110)
        mock_step.return_value = game_state
        
        # Reset first
        env.reset()
        
        # Take a step
        action = TeeInput(direction=1, target_x=100, target_y=0)
        result = env.step(0, action)
        
        assert isinstance(result, StepResult)
        assert result.observation is not None
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)

    @patch.object(BotManager, 'connect', return_value=True)
    @patch.object(BotManager, 'pump')
    @patch.object(BotManager, 'step')
    def test_step_all(self, mock_step, mock_pump, mock_connect):
        """Test stepping all agents simultaneously."""
        env = TeeEnvironment(auto_connect=False)
        env._connected = True
        
        game_state = setup_mock_bot_manager(env, tick=110)
        mock_step.return_value = game_state
        
        env.reset()
        
        actions = {
            0: TeeInput(direction=-1),
            1: TeeInput(direction=1),
            2: TeeInput(jump=True),
            3: TeeInput(fire=True, target_x=50),
        }
        
        results = env.step_all(actions)
        
        assert len(results) == 4
        for agent_id, result in results.items():
            assert isinstance(result, StepResult)
            assert result.observation is not None

    @patch.object(BotManager, 'connect', return_value=True)
    @patch.object(BotManager, 'pump')
    def test_state(self, mock_pump, mock_connect):
        """Test getting game state."""
        env = TeeEnvironment(auto_connect=False)
        env._connected = True
        
        setup_mock_bot_manager(env, tick=100)
        env.bot_manager.get_scores = Mock(return_value={0: 5, 1: 3, 2: 2, 3: 1})
        
        env.reset()
        
        state = env.state()
        
        assert isinstance(state, TeeState)
        assert state.episode_id == env.episode_id
        assert state.tick == 100
        assert state.game_over is False

    @patch.object(BotManager, 'connect', return_value=True)
    @patch.object(BotManager, 'pump')
    @patch.object(BotManager, 'step')
    def test_game_over_on_max_steps(self, mock_step, mock_pump, mock_connect):
        """Test that game ends when max_steps is reached."""
        config = GameConfig(max_steps=3)
        env = TeeEnvironment(config=config, auto_connect=False)
        env._connected = True
        
        game_state = setup_mock_bot_manager(env, tick=100)
        mock_step.return_value = game_state
        
        env.reset()
        
        # Step 3 times
        actions = {i: TeeInput() for i in range(4)}
        for _ in range(3):
            results = env.step_all(actions)
        
        # Game should be over
        assert env.game_over is True
        assert results[0].done is True

    @patch.object(BotManager, 'connect', return_value=True)
    @patch.object(BotManager, 'pump')
    @patch.object(BotManager, 'step')
    def test_game_over_on_win_score(self, mock_step, mock_pump, mock_connect):
        """Test that game ends when win_score is reached."""
        config = GameConfig(win_score=5)
        env = TeeEnvironment(config=config, auto_connect=False)
        env._connected = True
        
        game_state = setup_mock_bot_manager(env, tick=100)
        mock_step.return_value = game_state
        # Agent 0 has 5 kills (wins)
        env.bot_manager.get_scores = Mock(return_value={0: 5, 1: 2, 2: 1, 3: 0})
        
        env.reset()
        
        actions = {i: TeeInput() for i in range(4)}
        results = env.step_all(actions)
        
        assert env.game_over is True
        assert env.winner == 0


class TestTeeInputConversion:
    """Test TeeInput to PlayerInput conversion."""

    @patch.object(TeeEnvironment, 'connect', return_value=True)
    def test_tee_input_to_player_input(self, mock_connect):
        """Test conversion from TeeInput to PlayerInput."""
        from teeunit.server.tee_environment import tee_input_to_player_input
        
        tee_input = TeeInput(
            direction=1,
            target_x=100,
            target_y=-50,
            jump=True,
            fire=True,
            hook=False,
            wanted_weapon=2,
        )
        
        player_input = tee_input_to_player_input(tee_input, fire_count=5)
        
        assert player_input.direction == 1
        assert player_input.target_x == 100
        assert player_input.target_y == -50
        assert player_input.jump == True
        assert player_input.fire == 5  # Uses fire_count when fire=True
        assert player_input.hook == False
        assert player_input.wanted_weapon == 2

    @patch.object(TeeEnvironment, 'connect', return_value=True)
    def test_tee_input_no_fire(self, mock_connect):
        """Test conversion when fire=False."""
        from teeunit.server.tee_environment import tee_input_to_player_input
        
        tee_input = TeeInput(fire=False)
        player_input = tee_input_to_player_input(tee_input, fire_count=10)
        
        # Fire should be 0 when not firing
        assert player_input.fire == 0


class TestRewardCalculation:
    """Test reward calculation in environment."""

    @patch.object(BotManager, 'connect', return_value=True)
    @patch.object(BotManager, 'pump')
    @patch.object(BotManager, 'step')
    def test_kill_reward(self, mock_step, mock_pump, mock_connect):
        """Test that kills give positive reward."""
        env = TeeEnvironment(auto_connect=False)
        env._connected = True
        
        # Create a kill event where agent 0 kills agent 1
        kill_event = KillEvent(killer_id=0, victim_id=1, weapon=1, tick=100)
        
        game_state = setup_mock_bot_manager(env, tick=110)
        game_state.kill_events = [kill_event]
        mock_step.return_value = game_state
        
        env.reset()
        
        actions = {i: TeeInput() for i in range(4)}
        results = env.step_all(actions)
        
        # Agent 0 should get kill reward (10.0) + survival bonus (0.1)
        assert results[0].reward > 0
        # Agent 1 should get death penalty (-5.0) + survival bonus (0.1)
        assert results[1].reward < 0


class TestObservationBuilding:
    """Test observation building."""

    @patch.object(BotManager, 'connect', return_value=True)
    @patch.object(BotManager, 'pump')
    def test_observation_contains_correct_data(self, mock_pump, mock_connect):
        """Test that observation contains correct character data."""
        env = TeeEnvironment(auto_connect=False)
        env._connected = True
        env.episode_id = "test-episode"
        
        # Create mock character with specific values
        char = create_mock_character(
            x=500, y=300, vel_x=10, vel_y=-5,
            health=7, armor=3, weapon=2, ammo_count=5
        )
        
        game_state = GameState(tick=100)
        game_state.characters = {0: char, 1: create_mock_character()}
        game_state.player_infos = {
            0: create_mock_player_info(score=3),
            1: create_mock_player_info(score=1),
        }
        game_state.projectiles = []
        game_state.pickups = []
        
        env.bot_manager.game_state = game_state
        env.bot_manager.bots = {
            0: create_mock_bot_state(0, character=char, player_info=game_state.player_infos[0]),
            1: create_mock_bot_state(1),
        }
        
        def get_bot_state(agent_id):
            return env.bot_manager.bots.get(agent_id)
        env.bot_manager.get_bot_state = get_bot_state
        
        obs = env.get_observation(0)
        
        assert obs.agent_id == 0
        assert obs.tick == 100
        assert obs.x == 500
        assert obs.y == 300
        assert obs.vel_x == 10
        assert obs.vel_y == -5
        assert obs.health == 7
        assert obs.armor == 3
        assert obs.weapon == 2
        assert obs.ammo == 5
        assert obs.is_alive is True
        assert obs.episode_id == "test-episode"

    @patch.object(BotManager, 'connect', return_value=True)
    @patch.object(BotManager, 'pump')
    def test_dead_observation(self, mock_pump, mock_connect):
        """Test observation for dead agent."""
        env = TeeEnvironment(auto_connect=False)
        env._connected = True
        env.episode_id = "test-episode"
        
        game_state = GameState(tick=100)
        game_state.projectiles = []
        game_state.pickups = []
        
        # Create dead player
        dead_info = create_mock_player_info(is_dead=True)
        bot = create_mock_bot_state(0, player_info=dead_info)
        
        env.bot_manager.game_state = game_state
        env.bot_manager.bots = {0: bot}
        
        def get_bot_state(agent_id):
            return env.bot_manager.bots.get(agent_id)
        env.bot_manager.get_bot_state = get_bot_state
        
        obs = env.get_observation(0)
        
        assert obs.is_alive is False
        assert obs.health == 0

    @patch.object(BotManager, 'connect', return_value=True)
    @patch.object(BotManager, 'pump')
    def test_visible_players(self, mock_pump, mock_connect):
        """Test that visible players are included in observation."""
        env = TeeEnvironment(auto_connect=False)
        env._connected = True
        env.episode_id = "test-episode"
        
        # Create characters for multiple players
        char0 = create_mock_character(x=100, y=100)
        char1 = create_mock_character(x=200, y=100)
        char2 = create_mock_character(x=300, y=100)
        
        game_state = GameState(tick=100)
        game_state.characters = {0: char0, 1: char1, 2: char2}
        game_state.player_infos = {
            0: create_mock_player_info(score=0),
            1: create_mock_player_info(score=2),
            2: create_mock_player_info(score=1),
        }
        game_state.projectiles = []
        game_state.pickups = []
        
        env.bot_manager.game_state = game_state
        env.bot_manager.bots = {
            0: create_mock_bot_state(0, character=char0, player_info=game_state.player_infos[0]),
            1: create_mock_bot_state(1, character=char1, player_info=game_state.player_infos[1]),
            2: create_mock_bot_state(2, character=char2, player_info=game_state.player_infos[2]),
        }
        
        def get_bot_state(agent_id):
            return env.bot_manager.bots.get(agent_id)
        env.bot_manager.get_bot_state = get_bot_state
        
        obs = env.get_observation(0)
        
        # Should see 2 other players (not self)
        assert len(obs.visible_players) == 2
        client_ids = [p.client_id for p in obs.visible_players]
        assert 0 not in client_ids  # Self not included
        assert 1 in client_ids
        assert 2 in client_ids


class TestContextManager:
    """Test context manager functionality."""

    @patch.object(BotManager, 'connect', return_value=True)
    @patch.object(BotManager, 'disconnect')
    @patch.object(BotManager, 'pump')
    def test_context_manager(self, mock_pump, mock_disconnect, mock_connect):
        """Test environment as context manager."""
        with TeeEnvironment(auto_connect=False) as env:
            assert env is not None
        
        # disconnect should be called on exit
        mock_disconnect.assert_called_once()
