"""
Teeworlds Snapshot Parsing

Snapshots contain the complete game state at a point in time.
They are delta-compressed and contain multiple game objects.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .const import (
    NETOBJTYPE_CHARACTER,
    NETOBJTYPE_PLAYERINFO,
    NETOBJTYPE_PROJECTILE,
    NETOBJTYPE_LASER,
    NETOBJTYPE_PICKUP,
    NETOBJTYPE_GAMEDATA,
    NETOBJ_SIZES,
)
from .packer import Unpacker
from .objects import (
    Character,
    PlayerInfo,
    Projectile,
    Laser,
    Pickup,
    GameData,
    parse_object,
)


@dataclass
class SnapshotItem:
    """A single item in a snapshot."""

    type_id: int
    item_id: int
    data: List[int] = field(default_factory=list)

    @property
    def key(self) -> int:
        """Get the unique key for this item (type << 16 | id)."""
        return (self.type_id << 16) | (self.item_id & 0xFFFF)

    def parse(self) -> Optional[Any]:
        """Parse the item data into a typed object."""
        return parse_object(self.type_id, self.data)


@dataclass
class Snapshot:
    """A complete game state snapshot."""

    tick: int = 0
    delta_tick: int = -1  # -1 = full snapshot
    items: Dict[int, SnapshotItem] = field(default_factory=dict)  # key -> item

    # Parsed objects (cached)
    _characters: Optional[Dict[int, Character]] = field(default=None, repr=False)
    _player_infos: Optional[Dict[int, PlayerInfo]] = field(default=None, repr=False)
    _projectiles: Optional[List[Projectile]] = field(default=None, repr=False)
    _lasers: Optional[List[Laser]] = field(default=None, repr=False)
    _pickups: Optional[List[Pickup]] = field(default=None, repr=False)
    _game_data: Optional[GameData] = field(default=None, repr=False)

    def add_item(self, type_id: int, item_id: int, data: List[int]):
        """Add an item to the snapshot."""
        item = SnapshotItem(type_id=type_id, item_id=item_id, data=data)
        self.items[item.key] = item
        # Invalidate caches
        self._characters = None
        self._player_infos = None
        self._projectiles = None
        self._lasers = None
        self._pickups = None
        self._game_data = None

    def get_item(self, type_id: int, item_id: int) -> Optional[SnapshotItem]:
        """Get an item by type and ID."""
        key = (type_id << 16) | (item_id & 0xFFFF)
        return self.items.get(key)

    @property
    def characters(self) -> Dict[int, Character]:
        """Get all character objects, keyed by player ID."""
        if self._characters is None:
            self._characters = {}
            for item in self.items.values():
                if item.type_id == NETOBJTYPE_CHARACTER:
                    char = Character.from_ints(item.data)
                    self._characters[item.item_id] = char
        return self._characters

    @property
    def player_infos(self) -> Dict[int, PlayerInfo]:
        """Get all player info objects, keyed by player ID."""
        if self._player_infos is None:
            self._player_infos = {}
            for item in self.items.values():
                if item.type_id == NETOBJTYPE_PLAYERINFO:
                    info = PlayerInfo.from_ints(item.data)
                    self._player_infos[item.item_id] = info
        return self._player_infos

    @property
    def projectiles(self) -> List[Projectile]:
        """Get all projectiles."""
        if self._projectiles is None:
            self._projectiles = []
            for item in self.items.values():
                if item.type_id == NETOBJTYPE_PROJECTILE:
                    self._projectiles.append(Projectile.from_ints(item.data))
        return self._projectiles

    @property
    def lasers(self) -> List[Laser]:
        """Get all lasers."""
        if self._lasers is None:
            self._lasers = []
            for item in self.items.values():
                if item.type_id == NETOBJTYPE_LASER:
                    self._lasers.append(Laser.from_ints(item.data))
        return self._lasers

    @property
    def pickups(self) -> List[Pickup]:
        """Get all pickups."""
        if self._pickups is None:
            self._pickups = []
            for item in self.items.values():
                if item.type_id == NETOBJTYPE_PICKUP:
                    self._pickups.append(Pickup.from_ints(item.data))
        return self._pickups

    @property
    def game_data(self) -> Optional[GameData]:
        """Get game data object."""
        if self._game_data is None:
            for item in self.items.values():
                if item.type_id == NETOBJTYPE_GAMEDATA:
                    self._game_data = GameData.from_ints(item.data)
                    break
        return self._game_data


class SnapshotUnpacker:
    """Unpacks snapshot data from network packets."""

    def __init__(self):
        self.snapshots: Dict[int, Snapshot] = {}  # tick -> snapshot

    def unpack_snapshot(
        self,
        data: bytes,
        tick: int,
        delta_tick: int = -1,
    ) -> Optional[Snapshot]:
        """Unpack a snapshot from raw data.
        
        Teeworlds 0.7 SNAPSINGLE/SNAP format:
        1. CRC (packed int) - checksum of snapshot
        2. part_size (packed int) - size of data part
        3. num_deleted (packed int)
        4. deleted keys (packed ints)
        5. num_updated (packed int)
        6. items: type_id, item_id, data...
        
        Args:
            data: Raw snapshot data (after tick/delta_tick)
            tick: Current tick
            delta_tick: Reference tick for delta (-1 = full snapshot)
            
        Returns:
            Parsed Snapshot or None on error
        """
        snapshot = Snapshot(tick=tick, delta_tick=delta_tick)

        # Get base snapshot for delta
        base_snapshot = None
        if delta_tick >= 0:
            base_snapshot = self.snapshots.get(delta_tick)
            if base_snapshot:
                # Copy items from base
                snapshot.items = dict(base_snapshot.items)

        unpacker = Unpacker(data)

        try:
            # Read CRC (we don't validate it currently)
            crc = unpacker.get_int()
            
            # Read part size (tells us how much data follows)
            part_size = unpacker.get_int()
            
            # Read number of deleted items
            num_deleted = unpacker.get_int()

            # Read deleted item keys
            for _ in range(num_deleted):
                key = unpacker.get_int()
                # Remove from snapshot
                snapshot.items.pop(key, None)

            # Read number of updated items
            num_updated = unpacker.get_int()

            # Read updated items
            for _ in range(num_updated):
                type_id = unpacker.get_int()
                item_id = unpacker.get_int()

                # Get expected size from our known object sizes
                size = NETOBJ_SIZES.get(type_id, 0)
                if size == 0:
                    # Unknown type - skip this item
                    # We can't reliably parse without knowing size
                    continue

                # Read data ints
                item_data = []
                for _ in range(size):
                    item_data.append(unpacker.get_int())

                snapshot.add_item(type_id, item_id, item_data)

        except (ValueError, IndexError) as e:
            # Parsing error - return partial snapshot
            pass

        # Store for future deltas
        self.snapshots[tick] = snapshot

        # Clean old snapshots (keep last 100)
        if len(self.snapshots) > 100:
            old_ticks = sorted(self.snapshots.keys())[:-100]
            for old_tick in old_ticks:
                del self.snapshots[old_tick]

        return snapshot

    def unpack_snapshot_single(self, data: bytes, tick: int) -> Optional[Snapshot]:
        """Unpack a single (non-delta) snapshot."""
        return self.unpack_snapshot(data, tick, delta_tick=-1)


def unpack_snapshot_simple(data: bytes, tick: int = 0) -> Snapshot:
    """Simple snapshot unpacking without delta support.
    
    This is a simplified version that assumes the data contains
    a list of items without delta encoding.
    """
    snapshot = Snapshot(tick=tick, delta_tick=-1)
    unpacker = Unpacker(data)

    try:
        while unpacker.remaining() > 0:
            type_id = unpacker.get_int()
            item_id = unpacker.get_int()

            # Get expected size
            size = NETOBJ_SIZES.get(type_id, 0)
            if size == 0:
                # Try to infer or skip
                break

            # Read data ints
            item_data = []
            for _ in range(size):
                if unpacker.remaining() <= 0:
                    break
                item_data.append(unpacker.get_int())

            if len(item_data) == size:
                snapshot.add_item(type_id, item_id, item_data)

    except (ValueError, IndexError):
        pass

    return snapshot
