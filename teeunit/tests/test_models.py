"""Tests for TeeUnit models."""

import pytest
from teeunit.models import (
    WeaponType,
    WEAPON_NAMES,
    TeeInput,
    TeeObservation,
    TeeState,
    GameConfig,
    StepResult,
    VisiblePlayer,
    KillEvent,
)


class TestWeaponType:
    """Test WeaponType enum."""

    def test_weapon_values(self):
        """Verify weapon integer values match Teeworlds protocol."""
        assert WeaponType.HAMMER == 0
        assert WeaponType.GUN == 1
        assert WeaponType.SHOTGUN == 2
        assert WeaponType.GRENADE == 3
        assert WeaponType.LASER == 4
        assert WeaponType.NINJA == 5
    
    def test_weapon_names(self):
        """Verify weapon display names."""
        assert WEAPON_NAMES[WeaponType.HAMMER] == "hammer"
        assert WEAPON_NAMES[WeaponType.GUN] == "pistol"
        assert WEAPON_NAMES[WeaponType.SHOTGUN] == "shotgun"
        assert WEAPON_NAMES[WeaponType.LASER] == "laser"


class TestTeeInput:
    """Test TeeInput dataclass."""

    def test_default_input(self):
        """Test default input values."""
        inp = TeeInput()
        assert inp.direction == 0
        assert inp.target_x == 0
        assert inp.target_y == 0
        assert inp.jump is False
        assert inp.fire is False
        assert inp.hook is False
        assert inp.wanted_weapon == 0

    def test_move_left(self):
        """Test move left factory."""
        inp = TeeInput.move_left()
        assert inp.direction == -1
        assert inp.jump is False

    def test_move_right(self):
        """Test move right factory."""
        inp = TeeInput.move_right()
        assert inp.direction == 1
        assert inp.jump is False

    def test_jump_left(self):
        """Test jump left factory."""
        inp = TeeInput.jump_left()
        assert inp.direction == -1
        assert inp.jump is True

    def test_jump_right(self):
        """Test jump right factory."""
        inp = TeeInput.jump_right()
        assert inp.direction == 1
        assert inp.jump is True

    def test_fire_at(self):
        """Test fire at target factory."""
        inp = TeeInput.fire_at(100, -50)
        assert inp.target_x == 100
        assert inp.target_y == -50
        assert inp.fire is True

    def test_hook_at(self):
        """Test hook at target factory."""
        inp = TeeInput.hook_at(200, 100)
        assert inp.target_x == 200
        assert inp.target_y == 100
        assert inp.hook is True

    def test_direction_clamping(self):
        """Test that direction is clamped to -1, 0, 1."""
        inp = TeeInput(direction=5)
        assert inp.direction == 1
        
        inp = TeeInput(direction=-10)
        assert inp.direction == -1

    def test_to_dict(self):
        """Test serialization to dict."""
        inp = TeeInput(direction=1, target_x=100, jump=True, fire=True)
        d = inp.to_dict()
        
        assert d["direction"] == 1
        assert d["target_x"] == 100
        assert d["jump"] is True
        assert d["fire"] is True

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "direction": -1,
            "target_x": 50,
            "target_y": -25,
            "jump": True,
            "fire": False,
            "hook": True,
            "wanted_weapon": 2,
        }
        inp = TeeInput.from_dict(d)
        
        assert inp.direction == -1
        assert inp.target_x == 50
        assert inp.target_y == -25
        assert inp.jump is True
        assert inp.fire is False
        assert inp.hook is True
        assert inp.wanted_weapon == 2


class TestTeeObservation:
    """Test TeeObservation dataclass."""

    def test_create_observation(self):
        """Test creating an observation."""
        obs = TeeObservation(
            agent_id=0,
            tick=100,
            x=1000,
            y=2000,
            vel_x=50,
            vel_y=-100,
            health=8,
            armor=5,
            weapon=WeaponType.SHOTGUN,
            ammo=5,
            direction=1,
            is_grounded=True,
            is_alive=True,
            score=3,
            visible_players=[],
            projectiles=[],
            pickups=[],
            recent_kills=[],
            episode_id="test-123",
        )
        
        assert obs.agent_id == 0
        assert obs.tick == 100
        assert obs.x == 1000
        assert obs.y == 2000
        assert obs.health == 8
        assert obs.weapon == WeaponType.SHOTGUN

    def test_dead_observation(self):
        """Test creating a dead observation."""
        obs = TeeObservation.dead(agent_id=1, tick=50, episode_id="ep-456")
        
        assert obs.agent_id == 1
        assert obs.tick == 50
        assert obs.is_alive is False
        assert obs.health == 0
        assert obs.episode_id == "ep-456"
        assert "dead" in obs.text_description.lower()

    def test_to_dict(self):
        """Test serialization to dict."""
        obs = TeeObservation(
            agent_id=0,
            tick=100,
            x=1000,
            y=2000,
            vel_x=0,
            vel_y=0,
            health=10,
            armor=0,
            weapon=1,
            ammo=10,
            direction=0,
            is_grounded=True,
            is_alive=True,
            score=0,
            visible_players=[],
            projectiles=[],
            pickups=[],
            recent_kills=[],
        )
        
        d = obs.to_dict()
        assert d["agent_id"] == 0
        assert d["tick"] == 100
        assert d["x"] == 1000
        assert d["y"] == 2000
        assert d["weapon_name"] == "pistol"


class TestVisiblePlayer:
    """Test VisiblePlayer dataclass."""

    def test_create_visible_player(self):
        """Test creating a visible player."""
        player = VisiblePlayer(
            client_id=2,
            x=500,
            y=600,
            vel_x=100,
            vel_y=-50,
            health=7,
            armor=3,
            weapon=WeaponType.LASER,
            direction=-1,
            score=5,
            is_hooking=False,
        )
        
        assert player.client_id == 2
        assert player.x == 500
        assert player.health == 7
        assert player.weapon == WeaponType.LASER

    def test_distance_to(self):
        """Test distance calculation."""
        player = VisiblePlayer(
            client_id=0,
            x=100,
            y=100,
            vel_x=0,
            vel_y=0,
            health=10,
            armor=0,
            weapon=1,
            direction=0,
            score=0,
            is_hooking=False,
        )
        
        # Distance to (100, 100) should be 0
        assert player.distance_to(100, 100) == 0
        
        # Distance to (103, 104) should be 5 (3-4-5 triangle)
        assert player.distance_to(103, 104) == 5.0


class TestKillEvent:
    """Test KillEvent dataclass."""

    def test_create_kill_event(self):
        """Test creating a kill event."""
        event = KillEvent(
            killer_id=0,
            victim_id=1,
            weapon=WeaponType.SHOTGUN,
            tick=500,
        )
        
        assert event.killer_id == 0
        assert event.victim_id == 1
        assert event.weapon == WeaponType.SHOTGUN
        assert event.tick == 500

    def test_to_dict(self):
        """Test serialization to dict."""
        event = KillEvent(
            killer_id=2,
            victim_id=3,
            weapon=WeaponType.HAMMER,
            tick=100,
        )
        
        d = event.to_dict()
        assert d["killer_id"] == 2
        assert d["victim_id"] == 3
        assert d["weapon_name"] == "hammer"


class TestGameConfig:
    """Test GameConfig dataclass."""

    def test_default_config(self):
        """Test default game configuration."""
        config = GameConfig()
        assert config.num_agents == 4
        assert config.ticks_per_step == 10
        assert config.max_steps == 0
        assert config.win_score == 0
        assert config.server_host == "127.0.0.1"
        assert config.server_port == 8303

    def test_custom_config(self):
        """Test custom game configuration."""
        config = GameConfig(
            num_agents=2,
            ticks_per_step=20,
            max_steps=1000,
            win_score=10,
            server_host="localhost",
            server_port=8304,
        )
        assert config.num_agents == 2
        assert config.ticks_per_step == 20
        assert config.max_steps == 1000
        assert config.win_score == 10
        assert config.server_host == "localhost"
        assert config.server_port == 8304

    def test_to_dict(self):
        """Test serialization to dict."""
        config = GameConfig(num_agents=3, ticks_per_step=15)
        d = config.to_dict()
        
        assert d["num_agents"] == 3
        assert d["ticks_per_step"] == 15

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "num_agents": 6,
            "ticks_per_step": 5,
            "server_port": 9000,
        }
        config = GameConfig.from_dict(d)
        
        assert config.num_agents == 6
        assert config.ticks_per_step == 5
        assert config.server_port == 9000


class TestTeeState:
    """Test TeeState dataclass."""

    def test_create_state(self):
        """Test creating episode state."""
        state = TeeState(
            episode_id="ep-123",
            tick=500,
            step_count=50,
            agents_alive=[0, 1, 2],
            scores={0: 3, 1: 2, 2: 1, 3: 0},
            game_over=False,
            winner=None,
            ticks_per_step=10,
        )
        
        assert state.episode_id == "ep-123"
        assert state.tick == 500
        assert state.step_count == 50
        assert len(state.agents_alive) == 3
        assert state.scores[0] == 3
        assert state.game_over is False

    def test_to_dict(self):
        """Test serialization to dict."""
        state = TeeState(
            episode_id="test",
            tick=100,
            step_count=10,
            agents_alive=[0, 1],
            scores={0: 1, 1: 2},
            game_over=True,
            winner=1,
        )
        
        d = state.to_dict()
        assert d["episode_id"] == "test"
        assert d["winner"] == 1
        assert d["game_over"] is True


class TestStepResult:
    """Test StepResult dataclass."""

    def test_create_step_result(self):
        """Test creating a step result."""
        obs = TeeObservation.dead(agent_id=0)
        result = StepResult(
            observation=obs,
            reward=10.0,
            done=False,
            info={"test": "value"},
        )
        
        assert result.observation.agent_id == 0
        assert result.reward == 10.0
        assert result.done is False
        assert result.info["test"] == "value"

    def test_to_dict(self):
        """Test serialization to dict."""
        obs = TeeObservation.dead(agent_id=0)
        result = StepResult(observation=obs, reward=5.0, done=True)
        
        d = result.to_dict()
        assert d["reward"] == 5.0
        assert d["done"] is True
        assert "observation" in d
