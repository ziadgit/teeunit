"""
TeeUnit Protocol Package

Pure Python implementation of the Teeworlds 0.7 network protocol.
Enables bot clients to connect to Teeworlds servers, send inputs,
and receive game state snapshots.
"""

from .const import *
from .packer import Packer, Unpacker
from .huffman import Huffman
from .chunk import Chunk
from .packet import Packet
from .snapshot import Snapshot, SnapshotItem
from .objects import (
    PlayerInput,
    Character,
    CharacterCore,
    PlayerInfo,
    Projectile,
    Laser,
    Pickup,
    GameData,
)
from .client import TwClient, ConnectionState

__all__ = [
    # Constants
    "NET_VERSION",
    "NETMSG_INFO",
    "NETMSG_MAP_CHANGE",
    "NETMSG_SNAP",
    "NETMSG_INPUT",
    # Encoding
    "Packer",
    "Unpacker",
    "Huffman",
    "Chunk",
    "Packet",
    # Game state
    "Snapshot",
    "SnapshotItem",
    "PlayerInput",
    "Character",
    "CharacterCore",
    "PlayerInfo",
    "Projectile",
    "Laser",
    "Pickup",
    "GameData",
    # Client
    "TwClient",
    "ConnectionState",
]
