"""
TeeUnit Weapons Module

Defines weapon properties and damage calculation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math

from ..models import WeaponType


@dataclass
class WeaponStats:
    """Statistics for a weapon type."""
    name: str
    damage: int
    range: int  # Maximum range in cells
    ammo_per_pickup: int
    spread: float = 0.0  # Spread angle in degrees (for shotgun)
    is_melee: bool = False
    is_instant: bool = True  # True = hitscan, False = projectile
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "damage": self.damage,
            "range": self.range,
            "ammo_per_pickup": self.ammo_per_pickup,
            "spread": self.spread,
            "is_melee": self.is_melee,
            "is_instant": self.is_instant,
        }


# Weapon definitions
WEAPONS = {
    WeaponType.PISTOL.value: WeaponStats(
        name="Pistol",
        damage=15,
        range=8,
        ammo_per_pickup=0,  # Unlimited
    ),
    WeaponType.SHOTGUN.value: WeaponStats(
        name="Shotgun",
        damage=30,
        range=4,
        ammo_per_pickup=5,
        spread=15.0,  # 15 degree spread
    ),
    WeaponType.LASER.value: WeaponStats(
        name="Laser",
        damage=25,
        range=12,
        ammo_per_pickup=3,
    ),
    WeaponType.HAMMER.value: WeaponStats(
        name="Hammer",
        damage=50,
        range=1,
        ammo_per_pickup=0,  # Unlimited
        is_melee=True,
    ),
}


def get_weapon_stats(weapon: str) -> Optional[WeaponStats]:
    """Get stats for a weapon type."""
    return WEAPONS.get(weapon)


def calculate_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two positions."""
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    return math.sqrt(dx * dx + dy * dy)


def calculate_manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions."""
    return abs(pos2[0] - pos1[0]) + abs(pos2[1] - pos1[1])


def is_in_range(shooter_pos: Tuple[int, int], target_pos: Tuple[int, int], weapon: str) -> bool:
    """
    Check if target is within weapon range.
    
    Args:
        shooter_pos: Shooter's position
        target_pos: Target's position
        weapon: Weapon type
    
    Returns:
        True if target is in range
    """
    stats = get_weapon_stats(weapon)
    if not stats:
        return False
    
    distance = calculate_distance(shooter_pos, target_pos)
    return distance <= stats.range


def calculate_damage(
    shooter_pos: Tuple[int, int],
    target_pos: Tuple[int, int],
    weapon: str,
) -> int:
    """
    Calculate damage for a shot.
    
    Factors:
    - Base weapon damage
    - Distance falloff (for non-melee weapons)
    - Shotgun bonus at close range
    
    Args:
        shooter_pos: Shooter's position
        target_pos: Target's position
        weapon: Weapon type
    
    Returns:
        Damage amount (0 if out of range)
    """
    stats = get_weapon_stats(weapon)
    if not stats:
        return 0
    
    distance = calculate_distance(shooter_pos, target_pos)
    
    # Out of range
    if distance > stats.range:
        return 0
    
    base_damage = stats.damage
    
    if stats.is_melee:
        # Melee - full damage at range 1, 0 otherwise
        if distance <= 1.5:  # Allow diagonal melee
            return base_damage
        return 0
    
    # Distance falloff for ranged weapons
    # Damage decreases linearly with distance
    # At max range, damage is 50% of base
    falloff = 1.0 - (distance / stats.range) * 0.5
    
    # Shotgun close-range bonus
    if weapon == WeaponType.SHOTGUN.value:
        if distance <= 2:
            falloff = 1.2  # 20% bonus at point blank
        elif distance <= 3:
            falloff = 1.0  # Full damage at close range
    
    return int(base_damage * falloff)


def get_direction_to(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
    """
    Get cardinal/ordinal direction from one position to another.
    
    Args:
        from_pos: Starting position
        to_pos: Target position
    
    Returns:
        Direction string (e.g., "north", "northeast")
    """
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    
    # Normalize to get primary direction
    if dx == 0 and dy == 0:
        return "here"
    
    # Determine primary directions
    h_dir = ""
    v_dir = ""
    
    if dx > 0:
        h_dir = "east"
    elif dx < 0:
        h_dir = "west"
    
    if dy > 0:
        v_dir = "south"  # Y increases downward
    elif dy < 0:
        v_dir = "north"
    
    # Combine directions
    if abs(dx) > abs(dy) * 2:
        return h_dir
    elif abs(dy) > abs(dx) * 2:
        return v_dir
    elif h_dir and v_dir:
        return f"{v_dir}{h_dir}"  # e.g., "northeast"
    else:
        return h_dir or v_dir


def get_movement_delta(direction: str) -> Tuple[int, int]:
    """
    Get (dx, dy) movement delta for a direction.
    
    Args:
        direction: Direction string
    
    Returns:
        (dx, dy) tuple
    """
    direction = direction.lower()
    
    deltas = {
        "n": (0, -1),
        "north": (0, -1),
        "s": (0, 1),
        "south": (0, 1),
        "e": (1, 0),
        "east": (1, 0),
        "w": (-1, 0),
        "west": (-1, 0),
        "ne": (1, -1),
        "northeast": (1, -1),
        "nw": (-1, -1),
        "northwest": (-1, -1),
        "se": (1, 1),
        "southeast": (1, 1),
        "sw": (-1, 1),
        "southwest": (-1, 1),
    }
    
    return deltas.get(direction, (0, 0))


def estimate_health(actual_health: int, accuracy: float = 0.8) -> int:
    """
    Estimate health with some inaccuracy (for partial observability).
    
    Args:
        actual_health: The true health value
        accuracy: How accurate the estimate is (0-1)
    
    Returns:
        Estimated health (rounded to nearest 10)
    """
    import random
    
    # Add some noise based on accuracy
    noise_range = int((1 - accuracy) * 30)
    noise = random.randint(-noise_range, noise_range)
    
    estimated = actual_health + noise
    estimated = max(0, min(100, estimated))
    
    # Round to nearest 10 for "rough estimate" feel
    return round(estimated / 10) * 10
