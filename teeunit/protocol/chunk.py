"""
Teeworlds Network Chunk

A chunk is a single message within a packet. Packets can contain
multiple chunks, each with its own flags and sequence number.
"""

from dataclasses import dataclass, field
from typing import Optional

from .packer import Packer, Unpacker
from .const import NET_CHUNKFLAG_VITAL


@dataclass
class Chunk:
    """A single message chunk within a packet."""

    msg_id: int = 0
    sys: bool = False  # True = system message, False = game message
    flags: int = 0
    sequence: int = -1
    data: bytes = field(default_factory=bytes)

    # Parsed state
    _unpacker: Optional[Unpacker] = field(default=None, repr=False)

    def __post_init__(self):
        if self.data and self._unpacker is None:
            self._unpacker = Unpacker(self.data)

    @property
    def is_vital(self) -> bool:
        """Check if this is a vital (reliable) chunk."""
        return bool(self.flags & NET_CHUNKFLAG_VITAL)

    @property
    def size(self) -> int:
        """Get the data size."""
        return len(self.data)

    def get_int(self) -> int:
        """Read an integer from chunk data."""
        if self._unpacker is None:
            self._unpacker = Unpacker(self.data)
        return self._unpacker.get_int()

    def get_string(self) -> str:
        """Read a string from chunk data."""
        if self._unpacker is None:
            self._unpacker = Unpacker(self.data)
        return self._unpacker.get_string()

    def get_raw(self, length: int) -> bytes:
        """Read raw bytes from chunk data."""
        if self._unpacker is None:
            self._unpacker = Unpacker(self.data)
        return self._unpacker.get_raw(length)

    def get_remaining(self) -> bytes:
        """Get remaining unparsed data."""
        if self._unpacker is None:
            return self.data
        return self._unpacker.get_remaining()

    def reset_reader(self):
        """Reset the data reader to the beginning."""
        self._unpacker = Unpacker(self.data)

    @classmethod
    def create(
        cls,
        msg_id: int,
        sys: bool = False,
        vital: bool = True,
        sequence: int = 0,
    ) -> "Chunk":
        """Create a new chunk for sending."""
        flags = NET_CHUNKFLAG_VITAL if vital else 0
        return cls(msg_id=msg_id, sys=sys, flags=flags, sequence=sequence)

    def pack_header(self, packer: Packer):
        """Pack the message header (msg_id + sys flag)."""
        # Message header: (msg_id << 1) | sys
        header = (self.msg_id << 1) | (1 if self.sys else 0)
        packer.add_int(header)


class ChunkBuilder:
    """Helper for building chunk data."""

    def __init__(self, msg_id: int, sys: bool = False, vital: bool = True):
        self.msg_id = msg_id
        self.sys = sys
        self.vital = vital
        self.packer = Packer()

        # Add message header
        header = (msg_id << 1) | (1 if sys else 0)
        self.packer.add_int(header)

    def add_int(self, value: int) -> "ChunkBuilder":
        """Add an integer."""
        self.packer.add_int(value)
        return self

    def add_string(self, s: str) -> "ChunkBuilder":
        """Add a string."""
        self.packer.add_string(s)
        return self

    def add_raw(self, data: bytes) -> "ChunkBuilder":
        """Add raw bytes."""
        self.packer.add_raw(data)
        return self

    def build(self, sequence: int = 0) -> Chunk:
        """Build the final chunk."""
        flags = NET_CHUNKFLAG_VITAL if self.vital else 0
        return Chunk(
            msg_id=self.msg_id,
            sys=self.sys,
            flags=flags,
            sequence=sequence,
            data=self.packer.data(),
        )

    @property
    def data(self) -> bytes:
        """Get the packed data (including header)."""
        return self.packer.data()
