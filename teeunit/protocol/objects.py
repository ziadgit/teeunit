"""
Teeworlds Game Objects

Dataclasses representing the game objects that appear in snapshots.
Based on: https://github.com/teeworlds/teeworlds/blob/master/datasrc/network.py
"""

from dataclasses import dataclass
from typing import List, Optional

from .const import (
    NETOBJTYPE_PLAYERINPUT,
    NETOBJTYPE_PROJECTILE,
    NETOBJTYPE_LASER,
    NETOBJTYPE_PICKUP,
    NETOBJTYPE_FLAG,
    NETOBJTYPE_GAMEDATA,
    NETOBJTYPE_GAMEDATATEAM,
    NETOBJTYPE_GAMEDATAFLAG,
    NETOBJTYPE_CHARACTERCORE,
    NETOBJTYPE_CHARACTER,
    NETOBJTYPE_PLAYERINFO,
    WEAPON_GUN,
)


@dataclass
class PlayerInput:
    """Player input state - what we send to control the character.
    
    This maps directly to Teeworlds' CNetObj_PlayerInput.
    """

    direction: int = 0  # -1 (left), 0 (none), 1 (right)
    target_x: int = 0  # Aim X (relative to player)
    target_y: int = 0  # Aim Y (relative to player)
    jump: bool = False
    fire: int = 0  # Fire counter (increment to fire)
    hook: bool = False
    player_flags: int = 0
    wanted_weapon: int = 0  # 0=none, 1=hammer, 2=gun, etc.
    next_weapon: int = 0
    prev_weapon: int = 0

    def to_ints(self) -> List[int]:
        """Convert to list of integers for network transmission."""
        return [
            max(-1, min(1, self.direction)),
            self.target_x,
            self.target_y,
            1 if self.jump else 0,
            self.fire,
            1 if self.hook else 0,
            self.player_flags,
            self.wanted_weapon,
            self.next_weapon,
            self.prev_weapon,
        ]

    @classmethod
    def from_ints(cls, data: List[int]) -> "PlayerInput":
        """Create from list of integers."""
        if len(data) < 10:
            data = data + [0] * (10 - len(data))
        return cls(
            direction=data[0],
            target_x=data[1],
            target_y=data[2],
            jump=bool(data[3]),
            fire=data[4],
            hook=bool(data[5]),
            player_flags=data[6],
            wanted_weapon=data[7],
            next_weapon=data[8],
            prev_weapon=data[9],
        )


@dataclass
class CharacterCore:
    """Core character physics state.
    
    Maps to CNetObj_CharacterCore.
    """

    tick: int = 0
    x: int = 0
    y: int = 0
    vel_x: int = 0
    vel_y: int = 0
    angle: int = 0
    direction: int = 0
    jumped: int = 0
    hooked_player: int = -1
    hook_state: int = -1
    hook_tick: int = 0
    hook_x: int = 0
    hook_y: int = 0
    hook_dx: int = 0
    hook_dy: int = 0

    @classmethod
    def from_ints(cls, data: List[int]) -> "CharacterCore":
        """Create from list of integers."""
        if len(data) < 15:
            data = data + [0] * (15 - len(data))
        return cls(
            tick=data[0],
            x=data[1],
            y=data[2],
            vel_x=data[3],
            vel_y=data[4],
            angle=data[5],
            direction=data[6],
            jumped=data[7],
            hooked_player=data[8],
            hook_state=data[9],
            hook_tick=data[10],
            hook_x=data[11],
            hook_y=data[12],
            hook_dx=data[13],
            hook_dy=data[14],
        )


@dataclass
class Character(CharacterCore):
    """Full character state including health/weapons.
    
    Maps to CNetObj_Character (extends CharacterCore).
    """

    health: int = 0
    armor: int = 0
    ammo_count: int = 0
    weapon: int = WEAPON_GUN
    emote: int = 0
    attack_tick: int = 0
    triggered_events: int = 0

    @classmethod
    def from_ints(cls, data: List[int]) -> "Character":
        """Create from list of integers."""
        if len(data) < 22:
            data = data + [0] * (22 - len(data))

        # First 15 ints are CharacterCore
        core = CharacterCore.from_ints(data[:15])

        return cls(
            # Core fields
            tick=core.tick,
            x=core.x,
            y=core.y,
            vel_x=core.vel_x,
            vel_y=core.vel_y,
            angle=core.angle,
            direction=core.direction,
            jumped=core.jumped,
            hooked_player=core.hooked_player,
            hook_state=core.hook_state,
            hook_tick=core.hook_tick,
            hook_x=core.hook_x,
            hook_y=core.hook_y,
            hook_dx=core.hook_dx,
            hook_dy=core.hook_dy,
            # Character extension
            health=data[15],
            armor=data[16],
            ammo_count=data[17],
            weapon=data[18],
            emote=data[19],
            attack_tick=data[20],
            triggered_events=data[21],
        )


@dataclass
class PlayerInfo:
    """Player information (score, latency, flags).
    
    Maps to CNetObj_PlayerInfo.
    """

    player_flags: int = 0
    score: int = 0
    latency: int = 0

    @property
    def is_dead(self) -> bool:
        """Check if player is dead."""
        from .const import PLAYERFLAG_DEAD
        return bool(self.player_flags & PLAYERFLAG_DEAD)

    @property
    def is_bot(self) -> bool:
        """Check if player is a bot."""
        from .const import PLAYERFLAG_BOT
        return bool(self.player_flags & PLAYERFLAG_BOT)

    @classmethod
    def from_ints(cls, data: List[int]) -> "PlayerInfo":
        """Create from list of integers."""
        if len(data) < 3:
            data = data + [0] * (3 - len(data))
        return cls(
            player_flags=data[0],
            score=data[1],
            latency=data[2],
        )


@dataclass
class Projectile:
    """A projectile in flight.
    
    Maps to CNetObj_Projectile.
    """

    x: int = 0
    y: int = 0
    vel_x: int = 0
    vel_y: int = 0
    type: int = 0  # Weapon type
    start_tick: int = 0

    @classmethod
    def from_ints(cls, data: List[int]) -> "Projectile":
        """Create from list of integers."""
        if len(data) < 6:
            data = data + [0] * (6 - len(data))
        return cls(
            x=data[0],
            y=data[1],
            vel_x=data[2],
            vel_y=data[3],
            type=data[4],
            start_tick=data[5],
        )


@dataclass
class Laser:
    """A laser beam.
    
    Maps to CNetObj_Laser.
    """

    x: int = 0
    y: int = 0
    from_x: int = 0
    from_y: int = 0
    start_tick: int = 0

    @classmethod
    def from_ints(cls, data: List[int]) -> "Laser":
        """Create from list of integers."""
        if len(data) < 5:
            data = data + [0] * (5 - len(data))
        return cls(
            x=data[0],
            y=data[1],
            from_x=data[2],
            from_y=data[3],
            start_tick=data[4],
        )


@dataclass
class Pickup:
    """A pickup item on the map.
    
    Maps to CNetObj_Pickup.
    """

    x: int = 0
    y: int = 0
    type: int = 0  # Pickup type (health, armor, weapon)

    @classmethod
    def from_ints(cls, data: List[int]) -> "Pickup":
        """Create from list of integers."""
        if len(data) < 3:
            data = data + [0] * (3 - len(data))
        return cls(
            x=data[0],
            y=data[1],
            type=data[2],
        )


@dataclass
class Flag:
    """A flag (for CTF mode).
    
    Maps to CNetObj_Flag.
    """

    x: int = 0
    y: int = 0
    team: int = 0

    @classmethod
    def from_ints(cls, data: List[int]) -> "Flag":
        """Create from list of integers."""
        if len(data) < 3:
            data = data + [0] * (3 - len(data))
        return cls(
            x=data[0],
            y=data[1],
            team=data[2],
        )


@dataclass
class GameData:
    """Game state data.
    
    Maps to CNetObj_GameData.
    """

    game_start_tick: int = 0
    game_state_flags: int = 0
    game_state_end_tick: int = 0

    @property
    def is_warmup(self) -> bool:
        from .const import GAMESTATEFLAG_WARMUP
        return bool(self.game_state_flags & GAMESTATEFLAG_WARMUP)

    @property
    def is_paused(self) -> bool:
        from .const import GAMESTATEFLAG_PAUSED
        return bool(self.game_state_flags & GAMESTATEFLAG_PAUSED)

    @property
    def is_game_over(self) -> bool:
        from .const import GAMESTATEFLAG_GAMEOVER
        return bool(self.game_state_flags & GAMESTATEFLAG_GAMEOVER)

    @classmethod
    def from_ints(cls, data: List[int]) -> "GameData":
        """Create from list of integers."""
        if len(data) < 3:
            data = data + [0] * (3 - len(data))
        return cls(
            game_start_tick=data[0],
            game_state_flags=data[1],
            game_state_end_tick=data[2],
        )


@dataclass
class GameDataTeam:
    """Team scores (for team modes).
    
    Maps to CNetObj_GameDataTeam.
    """

    teamscore_red: int = 0
    teamscore_blue: int = 0

    @classmethod
    def from_ints(cls, data: List[int]) -> "GameDataTeam":
        """Create from list of integers."""
        if len(data) < 2:
            data = data + [0] * (2 - len(data))
        return cls(
            teamscore_red=data[0],
            teamscore_blue=data[1],
        )


# Object type to class mapping
OBJECT_CLASSES = {
    NETOBJTYPE_PLAYERINPUT: PlayerInput,
    NETOBJTYPE_PROJECTILE: Projectile,
    NETOBJTYPE_LASER: Laser,
    NETOBJTYPE_PICKUP: Pickup,
    NETOBJTYPE_FLAG: Flag,
    NETOBJTYPE_GAMEDATA: GameData,
    NETOBJTYPE_GAMEDATATEAM: GameDataTeam,
    NETOBJTYPE_CHARACTERCORE: CharacterCore,
    NETOBJTYPE_CHARACTER: Character,
    NETOBJTYPE_PLAYERINFO: PlayerInfo,
}


def parse_object(obj_type: int, data: List[int]):
    """Parse a snapshot object by type."""
    cls = OBJECT_CLASSES.get(obj_type)
    if cls:
        return cls.from_ints(data)
    return None
