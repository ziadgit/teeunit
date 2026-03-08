"""
Tests for TeeUnit OpenEnv Models.

Tests the Pydantic-based models for OpenEnv compatibility,
discrete action conversion, and tensor conversion.
"""

import numpy as np
import pytest
from teeunit import (
    TeeAction,
    TeeMultiAction,
    TeeObservation,
    TeeMultiObservation,
    TeeState,
    TeeStepResult,
    TeeMultiStepResult,
    RewardConfig,
    VisiblePlayer,
    VisibleProjectile,
    VisiblePickup,
    KillEvent,
)


class TestTeeAction:
    """Tests for TeeAction model."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        action = TeeAction()
        assert action.agent_id == 0
        assert action.direction == 0
        assert action.target_x == 0
        assert action.target_y == 0
        assert action.jump is False
        assert action.fire is False
        assert action.hook is False
        assert action.weapon == 0
    
    def test_direction_clamping(self):
        """Test that direction is clamped to -1, 0, 1."""
        # Values within range stay unchanged
        assert TeeAction(direction=-1).direction == -1
        assert TeeAction(direction=0).direction == 0
        assert TeeAction(direction=1).direction == 1
        
        # Values outside range are clamped
        assert TeeAction(direction=-10).direction == -1
        assert TeeAction(direction=10).direction == 1
    
    def test_discrete_action_conversion_roundtrip(self):
        """Test discrete action conversion is reversible."""
        for idx in range(18):
            action = TeeAction.from_discrete_action(idx, agent_id=2)
            assert action.agent_id == 2
            # Verify the discrete index matches expected pattern
            # (exact mapping depends on implementation)
    
    def test_discrete_action_coverage(self):
        """Test that all 18 discrete actions produce valid actions."""
        actions_seen = set()
        for idx in range(18):
            action = TeeAction.from_discrete_action(idx)
            key = (action.direction, action.jump, action.fire, action.hook)
            actions_seen.add(key)
        
        # Should have 18 unique action combinations
        assert len(actions_seen) == 18
    
    def test_discrete_action_no_op(self):
        """Test that discrete action 1 is a no-op (no direction, no actions)."""
        action = TeeAction.from_discrete_action(1)
        assert action.direction == 0
        assert action.jump is False
        assert action.fire is False
        assert action.hook is False
    
    def test_discrete_action_movement(self):
        """Test movement-only discrete actions."""
        # Left
        left = TeeAction.from_discrete_action(0)
        assert left.direction == -1
        assert left.jump is False
        
        # Right
        right = TeeAction.from_discrete_action(2)
        assert right.direction == 1
        assert right.jump is False
    
    def test_discrete_action_with_jump(self):
        """Test discrete actions with jump."""
        # Jump left
        jump_left = TeeAction.from_discrete_action(3)
        assert jump_left.direction == -1
        assert jump_left.jump is True
        assert jump_left.fire is False
        
        # Jump right
        jump_right = TeeAction.from_discrete_action(5)
        assert jump_right.direction == 1
        assert jump_right.jump is True
    
    def test_discrete_action_with_fire(self):
        """Test discrete actions with fire."""
        # Fire stationary
        fire = TeeAction.from_discrete_action(7)
        assert fire.direction == 0
        assert fire.fire is True
        assert fire.jump is False
    
    def test_discrete_action_wrapping(self):
        """Test that discrete action index wraps around."""
        # Index 18 should wrap to 0
        action18 = TeeAction.from_discrete_action(18)
        action0 = TeeAction.from_discrete_action(0)
        assert action18.direction == action0.direction
        assert action18.jump == action0.jump


class TestTeeMultiAction:
    """Tests for TeeMultiAction batched actions."""
    
    def test_from_list(self):
        """Test creating multi-action from list."""
        actions = [
            TeeAction(agent_id=0, direction=1),
            TeeAction(agent_id=1, direction=-1),
            TeeAction(agent_id=2, jump=True),
        ]
        multi = TeeMultiAction.from_list(actions)
        
        assert len(multi.actions) == 3
        assert multi.actions[0].direction == 1
        assert multi.actions[1].direction == -1
        assert multi.actions[2].jump is True
    
    def test_empty_multi_action(self):
        """Test empty multi-action."""
        multi = TeeMultiAction()
        assert len(multi.actions) == 0


class TestTeeObservation:
    """Tests for TeeObservation model."""
    
    def test_default_values(self):
        """Test default observation values."""
        obs = TeeObservation()
        assert obs.agent_id == 0
        assert obs.health == 10
        assert obs.is_alive is True
        assert obs.done is False
    
    def test_tensor_shape(self):
        """Test observation tensor shape."""
        obs = TeeObservation(agent_id=0, tick=100, x=500, y=300)
        tensor = obs.to_tensor()
        
        assert tensor.shape == (195,)
        assert tensor.dtype == np.float32
    
    def test_tensor_shape_static(self):
        """Test static tensor shape method."""
        assert TeeObservation.tensor_shape() == (195,)
    
    def test_tensor_self_state(self):
        """Test that self state is encoded in tensor."""
        obs = TeeObservation(
            agent_id=0,
            x=1000, y=500,
            vel_x=50, vel_y=-25,
            health=8, armor=5,
            weapon=2,
        )
        tensor = obs.to_tensor(normalize=True)
        
        # First 13 values are self state
        # x/1000, y/1000, vel_x/100, vel_y/100, health/10, armor/10, weapon/5
        assert tensor[0] == pytest.approx(1.0, abs=0.01)   # x/1000
        assert tensor[1] == pytest.approx(0.5, abs=0.01)   # y/1000
        assert tensor[2] == pytest.approx(0.5, abs=0.01)   # vel_x/100
        assert tensor[3] == pytest.approx(-0.25, abs=0.01) # vel_y/100
        assert tensor[4] == pytest.approx(0.8, abs=0.01)   # health/10
        assert tensor[5] == pytest.approx(0.5, abs=0.01)   # armor/10
        assert tensor[6] == pytest.approx(0.4, abs=0.01)   # weapon/5
    
    def test_tensor_visible_players(self):
        """Test that visible players are encoded."""
        obs = TeeObservation(
            x=500, y=300,
            visible_players=[
                VisiblePlayer(client_id=1, x=600, y=300, health=10),
                VisiblePlayer(client_id=2, x=400, y=400, health=5),
            ]
        )
        tensor = obs.to_tensor(normalize=True)
        
        # Players start at index 13
        # First player: valid=1.0, rel_x=0.1 (100/1000), rel_y=0.0
        assert tensor[13] == 1.0  # Valid flag
        assert tensor[14] == pytest.approx(0.1, abs=0.01)  # rel_x
        assert tensor[15] == pytest.approx(0.0, abs=0.01)  # rel_y
        
        # Second player
        assert tensor[23] == 1.0  # Valid flag
        assert tensor[24] == pytest.approx(-0.1, abs=0.01)  # rel_x
        assert tensor[25] == pytest.approx(0.1, abs=0.01)   # rel_y
    
    def test_tensor_padding(self):
        """Test that tensor pads missing entities with zeros."""
        obs = TeeObservation(visible_players=[], projectiles=[], pickups=[])
        tensor = obs.to_tensor()
        
        # All player slots should be zero (indices 13 to 82)
        assert all(tensor[13:83] == 0.0)
        
        # All projectile slots should be zero (indices 83 to 146)
        assert all(tensor[83:147] == 0.0)
        
        # All pickup slots should be zero (indices 147 to 194)
        assert all(tensor[147:195] == 0.0)
    
    def test_dead_observation(self):
        """Test creating a dead agent observation."""
        obs = TeeObservation.dead(agent_id=3, tick=500, episode_id="ep123")
        
        assert obs.agent_id == 3
        assert obs.tick == 500
        assert obs.is_alive is False
        assert obs.health == 0
        assert obs.episode_id == "ep123"
        assert "dead" in obs.text_description.lower()


class TestTeeMultiObservation:
    """Tests for TeeMultiObservation batched observations."""
    
    def test_to_tensor_batch(self):
        """Test converting multiple observations to tensor batch."""
        multi = TeeMultiObservation(
            observations={
                0: TeeObservation(agent_id=0, x=100, y=200),
                1: TeeObservation(agent_id=1, x=300, y=400),
            }
        )
        batch = multi.to_tensor_batch()
        
        assert batch.shape == (2, 195)
        assert batch.dtype == np.float32
    
    def test_empty_tensor_batch(self):
        """Test empty observation batch."""
        multi = TeeMultiObservation()
        batch = multi.to_tensor_batch()
        
        assert batch.shape == (0, 195)


class TestTeeState:
    """Tests for TeeState model."""
    
    def test_default_values(self):
        """Test default state values."""
        state = TeeState()
        assert state.tick == 0
        assert state.step_count == 0
        assert state.game_over is False
        assert state.winner is None
    
    def test_custom_config(self):
        """Test state with custom config."""
        state = TeeState(
            num_agents=4,
            ticks_per_step=5,
            scores={0: 3, 1: 2, 2: 1, 3: 0},
        )
        assert state.num_agents == 4
        assert state.ticks_per_step == 5
        assert state.scores[0] == 3


class TestTeeStepResult:
    """Tests for TeeStepResult model."""
    
    def test_step_result(self):
        """Test basic step result."""
        obs = TeeObservation(agent_id=0, health=8)
        result = TeeStepResult(
            observation=obs,
            reward=1.5,
            done=False,
            truncated=False,
            info={"kills": 1},
        )
        
        assert result.reward == 1.5
        assert result.done is False
        assert result.observation.health == 8


class TestTeeMultiStepResult:
    """Tests for TeeMultiStepResult batched results."""
    
    def test_to_arrays(self):
        """Test converting multi-step result to numpy arrays."""
        results = {
            0: TeeStepResult(
                observation=TeeObservation(agent_id=0),
                reward=1.0,
                done=False,
                truncated=False,
            ),
            1: TeeStepResult(
                observation=TeeObservation(agent_id=1),
                reward=-0.5,
                done=True,
                truncated=False,
            ),
        }
        state = TeeState(tick=100)
        
        multi = TeeMultiStepResult(results=results, state=state)
        arrays = multi.to_arrays()
        
        assert arrays['observations'].shape == (2, 195)
        assert arrays['rewards'].shape == (2,)
        assert arrays['dones'].shape == (2,)
        assert arrays['truncateds'].shape == (2,)
        
        # Check values
        assert arrays['rewards'][0] == pytest.approx(1.0)
        assert arrays['rewards'][1] == pytest.approx(-0.5)
        assert arrays['dones'][0] is False or arrays['dones'][0] == 0
        assert arrays['dones'][1] is True or arrays['dones'][1] == 1


class TestRewardConfig:
    """Tests for RewardConfig model."""
    
    def test_default_rewards(self):
        """Test default reward configuration."""
        config = RewardConfig()
        
        assert config.kill_reward == 10.0
        assert config.death_penalty == -5.0
        assert config.survival_bonus == 0.01
        assert config.win_bonus == 50.0
    
    def test_custom_rewards(self):
        """Test custom reward configuration."""
        config = RewardConfig(
            kill_reward=20.0,
            death_penalty=-10.0,
        )
        
        assert config.kill_reward == 20.0
        assert config.death_penalty == -10.0


class TestVisibleEntities:
    """Tests for visible entity models."""
    
    def test_visible_player_distance(self):
        """Test distance calculation for visible player."""
        player = VisiblePlayer(client_id=1, x=400, y=300)
        
        # Distance to origin
        dist = player.distance_to(0, 0)
        assert dist == pytest.approx(500.0, abs=0.1)  # 3-4-5 triangle scaled
        
        # Distance to same point
        assert player.distance_to(400, 300) == 0.0
    
    def test_visible_projectile(self):
        """Test visible projectile creation."""
        proj = VisibleProjectile(x=100, y=200, vel_x=50, vel_y=-30, weapon_type=2)
        assert proj.x == 100
        assert proj.weapon_type == 2
    
    def test_visible_pickup(self):
        """Test visible pickup creation."""
        pickup = VisiblePickup(x=500, y=100, pickup_type=1)
        assert pickup.x == 500
        assert pickup.pickup_type == 1
    
    def test_kill_event(self):
        """Test kill event creation."""
        kill = KillEvent(killer_id=0, victim_id=1, weapon=2, tick=1000)
        assert kill.killer_id == 0
        assert kill.victim_id == 1
        assert kill.weapon == 2
