"""Tests for TeeUnit models."""

import pytest
from teeunit.models import (
    ActionType,
    Direction,
    WeaponType,
    TerrainType,
    TeeAction,
    TeeObservation,
    TeeState,
    GameConfig,
)


class TestEnums:
    """Test enum definitions."""

    def test_action_types(self):
        """Verify all action types exist."""
        assert ActionType.MOVE is not None
        assert ActionType.SHOOT is not None
        assert ActionType.SWITCH_WEAPON is not None
        assert ActionType.PICKUP is not None
        assert ActionType.WAIT is not None

    def test_directions(self):
        """Verify all directions exist."""
        assert Direction.UP is not None
        assert Direction.DOWN is not None
        assert Direction.LEFT is not None
        assert Direction.RIGHT is not None
        assert Direction.UP_LEFT is not None
        assert Direction.UP_RIGHT is not None
        assert Direction.DOWN_LEFT is not None
        assert Direction.DOWN_RIGHT is not None

    def test_weapon_types(self):
        """Verify all weapon types exist."""
        assert WeaponType.PISTOL is not None
        assert WeaponType.SHOTGUN is not None
        assert WeaponType.LASER is not None
        assert WeaponType.HAMMER is not None

    def test_terrain_types(self):
        """Verify all terrain types exist."""
        assert TerrainType.FLOOR is not None
        assert TerrainType.WALL is not None
        assert TerrainType.WATER is not None
        assert TerrainType.SPAWN is not None


class TestTeeAction:
    """Test TeeAction dataclass."""

    def test_create_move_action(self):
        """Test creating a move action."""
        action = TeeAction(
            action_type=ActionType.MOVE,
            direction=Direction.UP,
        )
        assert action.action_type == ActionType.MOVE
        assert action.direction == Direction.UP

    def test_create_shoot_action(self):
        """Test creating a shoot action."""
        action = TeeAction(
            action_type=ActionType.SHOOT,
            direction=Direction.RIGHT,
            weapon=WeaponType.PISTOL,
        )
        assert action.action_type == ActionType.SHOOT
        assert action.direction == Direction.RIGHT
        assert action.weapon == WeaponType.PISTOL

    def test_create_wait_action(self):
        """Test creating a wait action."""
        action = TeeAction(action_type=ActionType.WAIT)
        assert action.action_type == ActionType.WAIT


class TestTeeObservation:
    """Test TeeObservation dataclass."""

    def test_create_observation(self):
        """Test creating an observation."""
        obs = TeeObservation(
            agent_id=0,
            position=(5, 5),
            health=100,
            armor=0,
            current_weapon=WeaponType.PISTOL,
            ammo={WeaponType.PISTOL: -1, WeaponType.SHOTGUN: 0},
            visible_terrain={},
            visible_agents=[],
            visible_pickups=[],
            kills=0,
            deaths=0,
            spawn_protection=0,
            current_turn=1,
        )
        assert obs.agent_id == 0
        assert obs.position == (5, 5)
        assert obs.health == 100


class TestGameConfig:
    """Test GameConfig dataclass."""

    def test_default_config(self):
        """Test default game configuration."""
        config = GameConfig()
        assert config.grid_size == 20
        assert config.num_agents == 4
        assert config.max_turns == 100
        assert config.max_kills == 10
        assert config.vision_radius == 6

    def test_custom_config(self):
        """Test custom game configuration."""
        config = GameConfig(
            grid_size=30,
            num_agents=6,
            max_turns=200,
        )
        assert config.grid_size == 30
        assert config.num_agents == 6
        assert config.max_turns == 200
