"""
Teeworlds Network Packet

Packets are the top-level network unit, containing a header and
one or more message chunks. Packets can be compressed with Huffman
coding and may be control packets (for connection management).
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from .const import (
    NET_PACKETFLAG_CONTROL,
    NET_PACKETFLAG_COMPRESSION,
    NET_PACKETFLAG_CONNLESS,
    NET_CHUNKFLAG_VITAL,
    NET_SEQUENCE_MASK,
)
from .chunk import Chunk
from .huffman import Huffman
from .packer import Unpacker


@dataclass
class Packet:
    """A network packet containing chunks."""

    flags: int = 0
    ack: int = 0
    num_chunks: int = 0
    chunks: List[Chunk] = field(default_factory=list)
    token: bytes = field(default_factory=lambda: b"\xff\xff\xff\xff")

    # For control packets
    ctrl_msg: int = -1
    ctrl_data: bytes = field(default_factory=bytes)

    @property
    def is_control(self) -> bool:
        """Check if this is a control packet."""
        return bool(self.flags & NET_PACKETFLAG_CONTROL)

    @property
    def is_compressed(self) -> bool:
        """Check if this packet is compressed."""
        return bool(self.flags & NET_PACKETFLAG_COMPRESSION)

    @property
    def is_connless(self) -> bool:
        """Check if this is a connectionless packet."""
        return bool(self.flags & NET_PACKETFLAG_CONNLESS)

    def unpack(self, data: bytes, huffman: Huffman) -> bool:
        """Unpack a packet from raw bytes."""
        if len(data) < 3:
            return False

        # Parse header
        self.flags = (data[0] >> 4) & 0x0F
        self.ack = ((data[0] & 0x0F) << 8) | data[1]
        self.num_chunks = data[2]

        payload = data[3:]

        # Handle control packets
        if self.is_control:
            if len(payload) >= 1:
                self.ctrl_msg = payload[0]
                self.ctrl_data = payload[1:]
            return True

        # Decompress if needed
        if self.is_compressed:
            payload = huffman.decompress(payload)
            if not payload:
                return False

        # Parse chunks
        self.chunks = []
        offset = 0

        for _ in range(self.num_chunks):
            if offset >= len(payload):
                break

            # Parse chunk header (2-3 bytes)
            if offset + 2 > len(payload):
                break

            chunk_flags = (payload[offset] >> 6) & 0x03
            chunk_size = ((payload[offset] & 0x3F) << 4) | (payload[offset + 1] & 0x0F)
            offset += 2

            # Vital chunks have sequence number
            chunk_sequence = -1
            if chunk_flags & NET_CHUNKFLAG_VITAL:
                if offset >= len(payload):
                    break
                chunk_sequence = ((payload[offset - 1] & 0xF0) << 2) | payload[offset]
                offset += 1

            # Extract chunk data
            if offset + chunk_size > len(payload):
                chunk_size = len(payload) - offset

            chunk_data = payload[offset : offset + chunk_size]
            offset += chunk_size

            # Parse message header from chunk data
            if not chunk_data:
                continue

            unpacker = Unpacker(chunk_data)
            try:
                msg_header = unpacker.get_int()
            except ValueError:
                continue

            sys_flag = bool(msg_header & 1)
            msg_id = msg_header >> 1

            chunk = Chunk(
                msg_id=msg_id,
                sys=sys_flag,
                flags=chunk_flags,
                sequence=chunk_sequence,
                data=unpacker.get_remaining(),
            )
            self.chunks.append(chunk)

        return True

    def pack(
        self,
        chunks: List[Tuple[bytes, int]],  # List of (data, flags)
        token: bytes,
        ack: int = 0,
        compress: bool = False,
        huffman: Huffman = None,
    ) -> bytes:
        """Pack chunks into a packet."""
        self.token = token
        self.ack = ack
        self.num_chunks = len(chunks)

        # Build payload from chunks
        payload = bytearray()

        for chunk_data, chunk_flags in chunks:
            chunk_size = len(chunk_data)

            # Build chunk header
            header = bytearray()
            header.append(((chunk_flags & 0x03) << 6) | ((chunk_size >> 4) & 0x3F))
            header.append(chunk_size & 0x0F)

            # Add sequence for vital chunks
            if chunk_flags & NET_CHUNKFLAG_VITAL:
                seq = self.ack & NET_SEQUENCE_MASK
                header[-1] |= (seq >> 2) & 0xF0
                header.append(seq & 0xFF)

            payload.extend(header)
            payload.extend(chunk_data)

        # Compress if requested
        self.flags = 0
        if compress and huffman and payload:
            compressed = huffman.compress(bytes(payload))
            if len(compressed) < len(payload):
                payload = bytearray(compressed)
                self.flags |= NET_PACKETFLAG_COMPRESSION

        # Build packet header
        packet = bytearray()
        packet.append(((self.flags << 4) & 0xF0) | ((self.ack >> 8) & 0x0F))
        packet.append(self.ack & 0xFF)
        packet.append(self.num_chunks)

        packet.extend(payload)
        packet.extend(token)

        return bytes(packet)


def pack_control_packet(ctrl_msg: int, extra: bytes = b"", token: bytes = b"\xff\xff\xff\xff", ack: int = 0) -> bytes:
    """Create a control packet."""
    packet = bytearray()

    # Header with control flag
    flags = NET_PACKETFLAG_CONTROL
    packet.append(((flags << 4) & 0xF0) | ((ack >> 8) & 0x0F))
    packet.append(ack & 0xFF)
    packet.append(0)  # num_chunks = 0 for control

    # Control message
    packet.append(ctrl_msg)
    packet.extend(extra)
    packet.extend(token)

    return bytes(packet)
