"""Tests for TeeUnit client."""

import pytest
from teeunit.client import LocalTeeEnv
from teeunit.models import ActionType, Direction, TeeAction


class TestLocalTeeEnv:
    """Test LocalTeeEnv (in-process client)."""

    def test_create_client(self):
        """Test creating local client."""
        env = LocalTeeEnv(seed=42)
        assert env is not None

    def test_reset(self):
        """Test resetting via client."""
        env = LocalTeeEnv(seed=42)
        observations = env.reset()
        
        assert len(observations) == 4
        for agent_id, obs in observations.items():
            assert obs.health == 100

    def test_step(self):
        """Test stepping via client."""
        env = LocalTeeEnv(seed=42)
        env.reset()
        
        action = TeeAction(action_type=ActionType.WAIT)
        obs, reward, done, info = env.step(0, action)
        
        assert obs is not None
        assert isinstance(reward, float)

    def test_step_all(self):
        """Test stepping all agents via client."""
        env = LocalTeeEnv(seed=42)
        env.reset()
        
        actions = {
            i: TeeAction(action_type=ActionType.WAIT) 
            for i in range(4)
        }
        results = env.step_all(actions)
        
        assert len(results) == 4

    def test_state(self):
        """Test getting state via client."""
        env = LocalTeeEnv(seed=42)
        env.reset()
        
        state = env.state()
        
        assert state.current_turn == 0
        assert state.done is False

    def test_close(self):
        """Test closing client."""
        env = LocalTeeEnv(seed=42)
        env.reset()
        env.close()  # Should not raise
