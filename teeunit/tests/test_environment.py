"""Tests for TeeUnit environment."""

import pytest
from teeunit.models import ActionType, Direction, WeaponType, TeeAction, GameConfig
from teeunit.server.tee_environment import TeeEnvironment


class TestTeeEnvironment:
    """Test TeeEnvironment class."""

    def test_create_environment(self):
        """Test environment creation."""
        env = TeeEnvironment(seed=42)
        assert env is not None

    def test_reset(self):
        """Test environment reset."""
        env = TeeEnvironment(seed=42)
        observations = env.reset()
        
        assert len(observations) == 4  # 4 agents
        
        for agent_id, obs in observations.items():
            assert obs.agent_id == agent_id
            assert obs.health == 100
            assert obs.kills == 0
            assert obs.deaths == 0
            assert obs.current_turn == 0

    def test_step_wait_action(self):
        """Test stepping with wait action."""
        env = TeeEnvironment(seed=42)
        env.reset()
        
        action = TeeAction(action_type=ActionType.WAIT)
        obs, reward, done, info = env.step(0, action)
        
        assert obs is not None
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_step_move_action(self):
        """Test stepping with move action."""
        env = TeeEnvironment(seed=42)
        env.reset()
        
        # Get initial position
        initial_obs = env.get_observation(0)
        initial_pos = initial_obs.position
        
        # Try to move
        action = TeeAction(action_type=ActionType.MOVE, direction=Direction.RIGHT)
        obs, reward, done, info = env.step(0, action)
        
        # Position may or may not change depending on obstacles
        assert obs is not None

    def test_step_all(self):
        """Test stepping all agents simultaneously."""
        env = TeeEnvironment(seed=42)
        env.reset()
        
        actions = {
            0: TeeAction(action_type=ActionType.WAIT),
            1: TeeAction(action_type=ActionType.WAIT),
            2: TeeAction(action_type=ActionType.WAIT),
            3: TeeAction(action_type=ActionType.WAIT),
        }
        
        results = env.step_all(actions)
        
        assert len(results) == 4
        for agent_id, (obs, reward, done, info) in results.items():
            assert obs is not None
            assert isinstance(reward, float)

    def test_state(self):
        """Test getting full state."""
        env = TeeEnvironment(seed=42)
        env.reset()
        
        state = env.state()
        
        assert state.current_turn == 0
        assert state.done is False
        assert len(state.observations) == 4
        assert len(state.scores) == 4

    def test_spawn_protection(self):
        """Test that agents have spawn protection after reset."""
        env = TeeEnvironment(seed=42)
        observations = env.reset()
        
        for agent_id, obs in observations.items():
            assert obs.spawn_protection == 3  # Default spawn protection turns

    def test_game_ends_at_max_turns(self):
        """Test that game ends at max turns."""
        config = GameConfig(max_turns=5)
        env = TeeEnvironment(config=config, seed=42)
        env.reset()
        
        # Step until max turns
        for turn in range(5):
            actions = {i: TeeAction(action_type=ActionType.WAIT) for i in range(4)}
            env.step_all(actions)
        
        state = env.state()
        assert state.done is True

    def test_custom_config(self):
        """Test environment with custom config."""
        config = GameConfig(
            grid_size=15,
            num_agents=2,
            max_turns=50,
        )
        env = TeeEnvironment(config=config, seed=42)
        observations = env.reset()
        
        assert len(observations) == 2


class TestWeaponMechanics:
    """Test weapon mechanics in environment."""

    def test_switch_weapon(self):
        """Test switching weapons."""
        env = TeeEnvironment(seed=42)
        env.reset()
        
        # Initially should have pistol
        obs = env.get_observation(0)
        assert obs.current_weapon == WeaponType.PISTOL
        
        # Switch to hammer (always available)
        action = TeeAction(
            action_type=ActionType.SWITCH_WEAPON,
            weapon=WeaponType.HAMMER,
        )
        obs, _, _, _ = env.step(0, action)
        assert obs.current_weapon == WeaponType.HAMMER

    def test_shoot_action(self):
        """Test shooting creates expected results."""
        env = TeeEnvironment(seed=42)
        env.reset()
        
        action = TeeAction(
            action_type=ActionType.SHOOT,
            direction=Direction.RIGHT,
        )
        obs, reward, done, info = env.step(0, action)
        
        # Action should complete without error
        assert obs is not None
