"""Tests for TeeUnit client.

Note: These tests require a running Teeworlds server.
Mark them with @pytest.mark.integration to skip in CI without a server.
"""

import pytest
from teeunit.client import TeeEnv, SyncTeeEnv, LocalTeeEnv, TeeEnvError, make_env
from teeunit.models import TeeInput, GameConfig


class TestMakeEnv:
    """Test make_env factory function."""

    def test_make_local_env(self):
        """Test creating local environment without URL."""
        env = make_env(auto_connect=False)
        assert isinstance(env, LocalTeeEnv)

    def test_make_remote_env(self):
        """Test creating remote environment with URL."""
        env = make_env(base_url="http://localhost:8000")
        assert isinstance(env, TeeEnv)

    def test_make_local_with_config(self):
        """Test creating local environment with custom config."""
        config = GameConfig(num_agents=2, ticks_per_step=5)
        env = make_env(config=config, auto_connect=False)
        assert isinstance(env, LocalTeeEnv)


class TestTeeEnv:
    """Test TeeEnv async client (without actual server)."""

    def test_create_client(self):
        """Test creating async client."""
        env = TeeEnv(base_url="http://localhost:8000")
        assert env.base_url == "http://localhost:8000"
        assert env._client is None  # Not connected yet

    def test_sync_wrapper(self):
        """Test creating sync wrapper."""
        env = TeeEnv(base_url="http://localhost:8000")
        sync_env = env.sync()
        assert isinstance(sync_env, SyncTeeEnv)


class TestLocalTeeEnv:
    """Test LocalTeeEnv (direct connection to Teeworlds).
    
    Note: Most tests here don't actually connect to avoid needing a server.
    """

    def test_create_without_connect(self):
        """Test creating local env without auto-connect."""
        env = LocalTeeEnv(auto_connect=False)
        assert env is not None

    def test_create_with_config(self):
        """Test creating local env with custom config."""
        config = GameConfig(
            num_agents=2,
            ticks_per_step=5,
            server_host="127.0.0.1",
            server_port=8303,
        )
        env = LocalTeeEnv(config=config, auto_connect=False)
        assert env._env.config.num_agents == 2
        assert env._env.config.ticks_per_step == 5

    def test_context_manager(self):
        """Test using as context manager."""
        with LocalTeeEnv(auto_connect=False) as env:
            assert env is not None
        # Should disconnect on exit (no error even without connection)


class TestTeeInput:
    """Test TeeInput usage patterns for client."""

    def test_input_factories(self):
        """Test all input factory methods."""
        # These should all create valid inputs
        assert TeeInput.move_left().direction == -1
        assert TeeInput.move_right().direction == 1
        assert TeeInput.jump_left().jump is True
        assert TeeInput.jump_right().jump is True
        assert TeeInput.fire_at(100, 50).fire is True
        assert TeeInput.hook_at(100, 50).hook is True

    def test_input_serialization(self):
        """Test that inputs serialize/deserialize correctly."""
        original = TeeInput(
            direction=1,
            target_x=100,
            target_y=-50,
            jump=True,
            fire=True,
            hook=False,
            wanted_weapon=2,
        )
        
        d = original.to_dict()
        restored = TeeInput.from_dict(d)
        
        assert restored.direction == original.direction
        assert restored.target_x == original.target_x
        assert restored.target_y == original.target_y
        assert restored.jump == original.jump
        assert restored.fire == original.fire
        assert restored.hook == original.hook
        assert restored.wanted_weapon == original.wanted_weapon


# Integration tests that require a running Teeworlds server
# Run with: pytest -m integration
# Skip by default when no server is running
@pytest.mark.integration
@pytest.mark.skip(reason="Requires running Teeworlds server - run with pytest -m integration")
class TestLocalTeeEnvIntegration:
    """Integration tests for LocalTeeEnv with real Teeworlds server."""

    def test_connect_and_reset(self):
        """Test connecting and resetting."""
        config = GameConfig(num_agents=2)
        with LocalTeeEnv(config=config) as env:
            result = env.reset()
            assert result.observation is not None
            assert result.observation.is_alive is True

    def test_step(self):
        """Test stepping."""
        with LocalTeeEnv() as env:
            env.reset()
            
            # Take a step with agent 0
            result = env.step(0, TeeInput.move_right())
            
            assert result.observation is not None
            assert isinstance(result.reward, float)
            assert isinstance(result.done, bool)

    def test_step_all(self):
        """Test stepping all agents."""
        config = GameConfig(num_agents=2)
        with LocalTeeEnv(config=config) as env:
            env.reset()
            
            actions = {
                0: TeeInput.move_right(),
                1: TeeInput.move_left(),
            }
            results = env.step_all(actions)
            
            assert len(results) == 2
            assert 0 in results
            assert 1 in results

    def test_state(self):
        """Test getting state."""
        with LocalTeeEnv() as env:
            env.reset()
            
            state = env.state()
            
            assert state.episode_id != ""
            assert state.tick >= 0
            assert state.step_count == 0
            assert state.game_over is False

    def test_observation(self):
        """Test getting observation."""
        with LocalTeeEnv() as env:
            env.reset()
            
            obs = env.get_observation(0)
            
            assert obs.agent_id == 0
            assert obs.tick >= 0
