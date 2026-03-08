"""Tests for TeeUnit arena."""

import pytest
from teeunit.server.arena import Arena, TerrainType


class TestArena:
    """Test Arena class."""

    def test_create_arena(self):
        """Test arena creation."""
        arena = Arena(size=20, seed=42)
        assert arena.size == 20
        assert len(arena.grid) == 20
        assert len(arena.grid[0]) == 20

    def test_spawn_points_created(self):
        """Test that spawn points are created."""
        arena = Arena(size=20, seed=42)
        assert len(arena.spawn_points) >= 4  # At least 4 for agents
        
        # All spawn points should be on floor/spawn terrain
        for x, y in arena.spawn_points:
            assert arena.grid[y][x] in (TerrainType.FLOOR, TerrainType.SPAWN)

    def test_is_walkable(self):
        """Test walkability checks."""
        arena = Arena(size=20, seed=42)
        
        # Find a floor tile (should be walkable)
        for y in range(arena.size):
            for x in range(arena.size):
                if arena.grid[y][x] == TerrainType.FLOOR:
                    assert arena.is_walkable(x, y) is True
                    break
        
        # Walls should not be walkable
        for y in range(arena.size):
            for x in range(arena.size):
                if arena.grid[y][x] == TerrainType.WALL:
                    assert arena.is_walkable(x, y) is False
                    break

    def test_out_of_bounds(self):
        """Test out of bounds checks."""
        arena = Arena(size=20, seed=42)
        
        assert arena.is_walkable(-1, 0) is False
        assert arena.is_walkable(0, -1) is False
        assert arena.is_walkable(20, 0) is False
        assert arena.is_walkable(0, 20) is False

    def test_blocks_vision(self):
        """Test vision blocking."""
        arena = Arena(size=20, seed=42)
        
        # Walls should block vision
        for y in range(arena.size):
            for x in range(arena.size):
                if arena.grid[y][x] == TerrainType.WALL:
                    assert arena.blocks_vision(x, y) is True
                    break

    def test_deterministic_generation(self):
        """Test that same seed produces same arena."""
        arena1 = Arena(size=20, seed=42)
        arena2 = Arena(size=20, seed=42)
        
        for y in range(20):
            for x in range(20):
                assert arena1.grid[y][x] == arena2.grid[y][x]

    def test_pickups_placed(self):
        """Test that pickups are placed."""
        arena = Arena(size=20, seed=42)
        assert len(arena.pickups) > 0
