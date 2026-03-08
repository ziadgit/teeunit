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
    
    # Highest sequence number received (for ack tracking)
    max_recv_sequence: int = -1

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
        """Unpack a packet from raw bytes.
        
        Teeworlds 0.7 packet format (7 bytes header):
        - byte 0: (flags << 2) | (ack >> 8)   -- FFFFFFaa
        - byte 1: ack & 0xff                   -- aaaaaaaa  
        - byte 2: num_chunks                   -- NNNNNNNN
        - bytes 3-6: token (big-endian)        -- TTTTTTTT x4
        - payload: control message or compressed chunks
        """
        if len(data) < 7:  # 3 header + 4 token minimum
            return False

        # Parse header (3 bytes)
        # Flags are in upper 6 bits, shifted left by 2
        self.flags = (data[0] >> 2) & 0x3F
        self.ack = ((data[0] & 0x03) << 8) | data[1]
        self.num_chunks = data[2]
        
        # Extract token (4 bytes after header, big-endian)
        self.token = bytes(data[3:7])

        payload = data[7:]

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
        max_sequence = -1  # Track highest sequence received

        for _ in range(self.num_chunks):
            if offset >= len(payload):
                break

            # Parse chunk header (2-3 bytes)
            # Teeworlds format:
            # byte 0: (flags << 6) | ((size >> 6) & 0x3F)
            # byte 1: (size & 0x3F) [| ((seq >> 2) & 0xC0) if vital]
            # byte 2: (seq & 0xFF) if vital
            if offset + 2 > len(payload):
                break

            chunk_flags = (payload[offset] >> 6) & 0x03
            chunk_size = ((payload[offset] & 0x3F) << 6) | (payload[offset + 1] & 0x3F)
            offset += 2

            # Vital chunks have sequence number
            chunk_sequence = -1
            if chunk_flags & NET_CHUNKFLAG_VITAL:
                if offset >= len(payload):
                    break
                chunk_sequence = ((payload[offset - 1] & 0xC0) << 2) | payload[offset]
                offset += 1
                # Track highest sequence for ack
                if chunk_sequence > max_sequence:
                    max_sequence = chunk_sequence

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
        
        # Store max sequence for ack tracking
        self.max_recv_sequence = max_sequence

        return True

    def pack(
        self,
        chunks: List[Tuple[bytes, int]],  # List of (data, flags)
        token: bytes,
        ack: int = 0,
        compress: bool = False,
        huffman: Huffman = None,
    ) -> bytes:
        """Pack chunks into a packet.
        
        Teeworlds 0.7 packet format (7 bytes header):
        - byte 0: (flags << 2) | (ack >> 8)   -- FFFFFFaa
        - byte 1: ack & 0xff                   -- aaaaaaaa
        - byte 2: num_chunks                   -- NNNNNNNN
        - bytes 3-6: token (big-endian)        -- TTTTTTTT x4
        - payload: chunks (optionally compressed)
        """
        self.token = token
        self.ack = ack
        self.num_chunks = len(chunks)

        # Build payload from chunks
        payload = bytearray()

        for chunk_data, chunk_flags in chunks:
            chunk_size = len(chunk_data)

            # Build chunk header according to Teeworlds format:
            # byte 0: (flags << 6) | ((size >> 6) & 0x3F)
            # byte 1: (size & 0x3F) [| ((seq >> 2) & 0xC0) if vital]
            # byte 2: (seq & 0xFF) if vital
            header = bytearray()
            header.append(((chunk_flags & 0x03) << 6) | ((chunk_size >> 6) & 0x3F))
            
            if chunk_flags & NET_CHUNKFLAG_VITAL:
                # Use global sequence counter for vital chunks
                seq = self.ack & NET_SEQUENCE_MASK
                header.append((chunk_size & 0x3F) | ((seq >> 2) & 0xC0))
                header.append(seq & 0xFF)
            else:
                header.append(chunk_size & 0x3F)

            payload.extend(header)
            payload.extend(chunk_data)

        # Compress if requested
        self.flags = 0
        if compress and huffman and payload:
            compressed = huffman.compress(bytes(payload))
            if len(compressed) < len(payload):
                payload = bytearray(compressed)
                self.flags |= NET_PACKETFLAG_COMPRESSION

        # Build packet header (7 bytes)
        packet = bytearray()
        # byte 0: (flags << 2) | (ack >> 8)
        packet.append(((self.flags << 2) & 0xFC) | ((self.ack >> 8) & 0x03))
        # byte 1: ack & 0xff
        packet.append(self.ack & 0xFF)
        # byte 2: num_chunks
        packet.append(self.num_chunks)

        # Token (4 bytes, big-endian) - use as-is since it's already bytes
        packet.extend(token)
        
        # Payload
        packet.extend(payload)

        return bytes(packet)
    
    def pack_with_sequence(
        self,
        chunks: List[Tuple[bytes, int, int]],  # List of (data, flags, sequence)
        token: bytes,
        ack: int = 0,
        compress: bool = False,
        huffman: Huffman = None,
    ) -> bytes:
        """Pack chunks into a packet with explicit sequence numbers.
        
        Teeworlds 0.7 packet format (7 bytes header):
        - byte 0: (flags << 2) | (ack >> 8)   -- FFFFFFaa
        - byte 1: ack & 0xff                   -- aaaaaaaa
        - byte 2: num_chunks                   -- NNNNNNNN
        - bytes 3-6: token (big-endian)        -- TTTTTTTT x4
        - payload: chunks (optionally compressed)
        """
        self.token = token
        self.ack = ack
        self.num_chunks = len(chunks)

        # Build payload from chunks
        payload = bytearray()

        for chunk_data, chunk_flags, chunk_seq in chunks:
            chunk_size = len(chunk_data)

            # Build chunk header
            header = bytearray()
            header.append(((chunk_flags & 0x03) << 6) | ((chunk_size >> 6) & 0x3F))
            
            if chunk_flags & NET_CHUNKFLAG_VITAL:
                seq = chunk_seq & NET_SEQUENCE_MASK
                header.append((chunk_size & 0x3F) | ((seq >> 2) & 0xC0))
                header.append(seq & 0xFF)
            else:
                header.append(chunk_size & 0x3F)

            payload.extend(header)
            payload.extend(chunk_data)

        # Compress if requested
        self.flags = 0
        if compress and huffman and payload:
            compressed = huffman.compress(bytes(payload))
            if len(compressed) < len(payload):
                payload = bytearray(compressed)
                self.flags |= NET_PACKETFLAG_COMPRESSION

        # Build packet header (7 bytes)
        packet = bytearray()
        packet.append(((self.flags << 2) & 0xFC) | ((self.ack >> 8) & 0x03))
        packet.append(self.ack & 0xFF)
        packet.append(self.num_chunks)
        packet.extend(token)
        packet.extend(payload)

        return bytes(packet)


def pack_control_packet(ctrl_msg: int, extra: bytes = b"", token: bytes = b"\xff\xff\xff\xff", ack: int = 0) -> bytes:
    """Create a control packet for Teeworlds 0.7.
    
    Teeworlds 0.7 packet format (7 bytes header):
    - byte 0: (flags << 2) | (ack >> 8)   -- FFFFFFaa (flags=0x01 for control)
    - byte 1: ack & 0xff                   -- aaaaaaaa
    - byte 2: num_chunks (0 for control)   -- NNNNNNNN
    - bytes 3-6: token (big-endian)        -- TTTTTTTT x4
    - payload: control message byte + extra data
    """
    packet = bytearray()

    # Header with control flag (3 bytes)
    flags = NET_PACKETFLAG_CONTROL  # = 1
    # byte 0: (flags << 2) | (ack >> 8)
    packet.append(((flags << 2) & 0xFC) | ((ack >> 8) & 0x03))
    # byte 1: ack & 0xff
    packet.append(ack & 0xFF)
    # byte 2: num_chunks = 0 for control
    packet.append(0)

    # Token (4 bytes, big-endian) - already bytes
    packet.extend(token)
    
    # Control message (1 byte)
    packet.append(ctrl_msg)
    
    # Extra data (e.g., MyToken for TOKEN/CONNECT messages)
    packet.extend(extra)

    return bytes(packet)
