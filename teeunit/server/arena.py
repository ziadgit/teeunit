"""
TeeUnit Arena Module

Handles the 2D grid map, terrain, spawn points, and pickups.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..models import GameConfig, PickupType, TerrainType


@dataclass
class Pickup:
    """A pickup item on the arena."""
    pickup_type: str
    position: Tuple[int, int]
    respawn_timer: int = 0  # 0 = available, >0 = respawning
    
    def to_dict(self) -> dict:
        return {
            "pickup_type": self.pickup_type,
            "position": list(self.position),
            "respawn_timer": self.respawn_timer,
        }


@dataclass
class SpawnPoint:
    """A spawn point for agents."""
    position: Tuple[int, int]
    last_used_turn: int = -100  # Turn when last used


class Arena:
    """
    The game arena - a 2D grid with terrain, spawn points, and pickups.
    
    Attributes:
        width: Arena width in cells
        height: Arena height in cells
        grid: 2D array of terrain types
        spawn_points: List of spawn point locations
        pickups: List of pickup items
    """
    
    def __init__(self, config: GameConfig):
        """Initialize the arena with given configuration."""
        self.config = config
        self.width = config.arena_width
        self.height = config.arena_height
        
        # Initialize grid with empty terrain
        self.grid: List[List[str]] = [
            [TerrainType.EMPTY.value for _ in range(self.width)]
            for _ in range(self.height)
        ]
        
        # Initialize spawn points, pickups
        self.spawn_points: List[SpawnPoint] = []
        self.pickups: List[Pickup] = []
        
        # Build the default arena layout
        self._build_default_layout()
    
    def _build_default_layout(self):
        """Build the default arena with walls, spawn points, and pickup locations."""
        # Add border walls
        for x in range(self.width):
            self.grid[0][x] = TerrainType.WALL.value
            self.grid[self.height - 1][x] = TerrainType.WALL.value
        for y in range(self.height):
            self.grid[y][0] = TerrainType.WALL.value
            self.grid[y][self.width - 1] = TerrainType.WALL.value
        
        # Add some interior walls for cover
        # Central cross structure
        center_x, center_y = self.width // 2, self.height // 2
        
        # Horizontal walls near center
        for dx in range(-2, 3):
            if 0 < center_x + dx < self.width - 1:
                self.grid[center_y - 3][center_x + dx] = TerrainType.WALL.value
                self.grid[center_y + 3][center_x + dx] = TerrainType.WALL.value
        
        # Vertical walls near center
        for dy in range(-2, 3):
            if 0 < center_y + dy < self.height - 1:
                self.grid[center_y + dy][center_x - 3] = TerrainType.WALL.value
                self.grid[center_y + dy][center_x + 3] = TerrainType.WALL.value
        
        # Corner cover blocks
        corner_offset = 4
        corners = [
            (corner_offset, corner_offset),
            (corner_offset, self.height - corner_offset - 1),
            (self.width - corner_offset - 1, corner_offset),
            (self.width - corner_offset - 1, self.height - corner_offset - 1),
        ]
        for cx, cy in corners:
            # Small L-shaped walls
            if self._in_bounds(cx, cy):
                self.grid[cy][cx] = TerrainType.WALL.value
            if self._in_bounds(cx + 1, cy):
                self.grid[cy][cx + 1] = TerrainType.WALL.value
            if self._in_bounds(cx, cy + 1):
                self.grid[cy + 1][cx] = TerrainType.WALL.value
        
        # Add a small water hazard in the center
        self.grid[center_y][center_x] = TerrainType.WATER.value
        
        # Setup spawn points (corners, away from walls)
        spawn_offset = 2
        self.spawn_points = [
            SpawnPoint(position=(spawn_offset, spawn_offset)),
            SpawnPoint(position=(self.width - spawn_offset - 1, spawn_offset)),
            SpawnPoint(position=(spawn_offset, self.height - spawn_offset - 1)),
            SpawnPoint(position=(self.width - spawn_offset - 1, self.height - spawn_offset - 1)),
        ]
        
        # Setup pickup spawn locations
        self._spawn_initial_pickups()
    
    def _spawn_initial_pickups(self):
        """Spawn initial pickups on the arena."""
        self.pickups = []
        
        # Health packs - placed strategically
        health_positions = [
            (self.width // 4, self.height // 2),
            (3 * self.width // 4, self.height // 2),
            (self.width // 2, self.height // 4),
            (self.width // 2, 3 * self.height // 4),
        ]
        
        for pos in health_positions:
            if self._is_valid_pickup_position(pos):
                self.pickups.append(Pickup(
                    pickup_type=PickupType.HEALTH.value,
                    position=pos,
                ))
        
        # Armor - center adjacent
        armor_positions = [
            (self.width // 2 - 1, self.height // 2 - 1),
            (self.width // 2 + 1, self.height // 2 + 1),
        ]
        for pos in armor_positions:
            if self._is_valid_pickup_position(pos):
                self.pickups.append(Pickup(
                    pickup_type=PickupType.ARMOR.value,
                    position=pos,
                ))
        
        # Ammo pickups - scattered
        ammo_positions = [
            (self.width // 4, self.height // 4),
            (3 * self.width // 4, self.height // 4),
            (self.width // 4, 3 * self.height // 4),
            (3 * self.width // 4, 3 * self.height // 4),
        ]
        
        ammo_types = [PickupType.SHOTGUN_AMMO.value, PickupType.LASER_AMMO.value]
        for i, pos in enumerate(ammo_positions):
            if self._is_valid_pickup_position(pos):
                self.pickups.append(Pickup(
                    pickup_type=ammo_types[i % 2],
                    position=pos,
                ))
    
    def _is_valid_pickup_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is valid for placing a pickup."""
        x, y = pos
        if not self._in_bounds(x, y):
            return False
        return self.grid[y][x] == TerrainType.EMPTY.value
    
    def _in_bounds(self, x: int, y: int) -> bool:
        """Check if coordinates are within arena bounds."""
        return 0 <= x < self.width and 0 <= y < self.height
    
    def get_terrain(self, x: int, y: int) -> str:
        """Get terrain type at position."""
        if not self._in_bounds(x, y):
            return TerrainType.WALL.value  # Out of bounds = wall
        return self.grid[y][x]
    
    def is_walkable(self, x: int, y: int) -> bool:
        """Check if a cell can be walked on."""
        terrain = self.get_terrain(x, y)
        return terrain in [TerrainType.EMPTY.value, TerrainType.WATER.value, TerrainType.PLATFORM.value]
    
    def is_blocking(self, x: int, y: int) -> bool:
        """Check if a cell blocks line of sight."""
        terrain = self.get_terrain(x, y)
        return terrain == TerrainType.WALL.value
    
    def get_spawn_point(self, current_turn: int, exclude_positions: List[Tuple[int, int]] = None) -> Tuple[int, int]:
        """
        Get a spawn point for an agent.
        
        Tries to find a spawn point that wasn't recently used and isn't
        too close to existing agents.
        """
        exclude_positions = exclude_positions or []
        
        # Sort spawn points by how long ago they were used
        sorted_spawns = sorted(
            self.spawn_points,
            key=lambda sp: current_turn - sp.last_used_turn,
            reverse=True,
        )
        
        for spawn in sorted_spawns:
            # Check if any agent is too close
            too_close = False
            for pos in exclude_positions:
                dist = abs(spawn.position[0] - pos[0]) + abs(spawn.position[1] - pos[1])
                if dist < 3:  # Minimum spawn distance
                    too_close = True
                    break
            
            if not too_close:
                spawn.last_used_turn = current_turn
                return spawn.position
        
        # Fallback: return least recently used
        spawn = sorted_spawns[0]
        spawn.last_used_turn = current_turn
        return spawn.position
    
    def get_pickup_at(self, x: int, y: int) -> Optional[Pickup]:
        """Get available pickup at position, if any."""
        for pickup in self.pickups:
            if pickup.position == (x, y) and pickup.respawn_timer == 0:
                return pickup
        return None
    
    def collect_pickup(self, x: int, y: int) -> Optional[Pickup]:
        """
        Collect a pickup at position.
        
        Returns the collected pickup and starts its respawn timer.
        """
        pickup = self.get_pickup_at(x, y)
        if pickup:
            pickup.respawn_timer = self.config.pickup_respawn_turns
            return pickup
        return None
    
    def update_pickups(self):
        """Update pickup respawn timers (call once per turn)."""
        for pickup in self.pickups:
            if pickup.respawn_timer > 0:
                pickup.respawn_timer -= 1
    
    def get_available_pickups(self) -> List[Pickup]:
        """Get all currently available pickups."""
        return [p for p in self.pickups if p.respawn_timer == 0]
    
    def get_cells_in_radius(self, x: int, y: int, radius: int) -> List[Tuple[int, int]]:
        """Get all cell coordinates within radius of a position."""
        cells = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if self._in_bounds(nx, ny):
                    # Use Chebyshev distance (max of dx, dy) for square vision
                    if max(abs(dx), abs(dy)) <= radius:
                        cells.append((nx, ny))
        return cells
    
    def get_walls_in_area(self, x: int, y: int, radius: int) -> List[Tuple[int, int]]:
        """Get all wall positions within radius."""
        walls = []
        for cell in self.get_cells_in_radius(x, y, radius):
            if self.is_blocking(cell[0], cell[1]):
                walls.append(cell)
        return walls
    
    def to_ascii(self, agent_positions: Dict[int, Tuple[int, int]] = None) -> str:
        """
        Generate ASCII representation of the arena.
        
        Args:
            agent_positions: Dict mapping agent_id to (x, y) position
        
        Returns:
            Multi-line string showing the arena
        """
        agent_positions = agent_positions or {}
        
        # Create position to agent mapping
        pos_to_agent = {pos: aid for aid, pos in agent_positions.items()}
        
        lines = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                pos = (x, y)
                
                # Check for agent
                if pos in pos_to_agent:
                    row.append(str(pos_to_agent[pos]))
                    continue
                
                # Check for pickup
                pickup = self.get_pickup_at(x, y)
                if pickup:
                    if pickup.pickup_type == PickupType.HEALTH.value:
                        row.append("+")
                    elif pickup.pickup_type == PickupType.ARMOR.value:
                        row.append("A")
                    else:
                        row.append("*")
                    continue
                
                # Terrain
                terrain = self.grid[y][x]
                if terrain == TerrainType.WALL.value:
                    row.append("#")
                elif terrain == TerrainType.WATER.value:
                    row.append("~")
                elif terrain == TerrainType.PLATFORM.value:
                    row.append("=")
                else:
                    row.append(".")
            
            lines.append("".join(row))
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset the arena to initial state."""
        # Rebuild layout
        self.grid = [
            [TerrainType.EMPTY.value for _ in range(self.width)]
            for _ in range(self.height)
        ]
        self._build_default_layout()
        
        # Reset spawn point timers
        for spawn in self.spawn_points:
            spawn.last_used_turn = -100
