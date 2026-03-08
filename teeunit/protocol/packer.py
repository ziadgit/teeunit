"""
Teeworlds Variable-Integer Packer/Unpacker

Teeworlds uses a variable-length integer encoding:
- First byte: 1 sign bit, 1 extend bit, 6 data bits
- Subsequent bytes: 1 extend bit, 7 data bits
"""

from typing import List, Tuple


class Packer:
    """Packs data into Teeworlds wire format."""

    def __init__(self):
        self.buffer: bytearray = bytearray()

    def reset(self):
        """Clear the buffer."""
        self.buffer = bytearray()

    def add_int(self, value: int) -> "Packer":
        """Add a variable-length integer."""
        # Handle sign
        sign = 0
        if value < 0:
            sign = 0x40
            value = ~value

        # First byte: sign bit + extend bit + 6 data bits
        first_byte = sign | (value & 0x3F)
        value >>= 6

        if value != 0:
            first_byte |= 0x80  # Set extend bit

        self.buffer.append(first_byte)

        # Subsequent bytes: extend bit + 7 data bits
        while value != 0:
            byte = value & 0x7F
            value >>= 7
            if value != 0:
                byte |= 0x80  # Set extend bit
            self.buffer.append(byte)

        return self

    def add_string(self, s: str) -> "Packer":
        """Add a null-terminated string."""
        self.buffer.extend(s.encode("utf-8"))
        self.buffer.append(0)  # Null terminator
        return self

    def add_raw(self, data: bytes) -> "Packer":
        """Add raw bytes."""
        self.buffer.extend(data)
        return self

    def data(self) -> bytes:
        """Get the packed data."""
        return bytes(self.buffer)

    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)


class Unpacker:
    """Unpacks data from Teeworlds wire format."""

    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def remaining(self) -> int:
        """Get number of remaining bytes."""
        return len(self.data) - self.offset

    def get_int(self) -> int:
        """Read a variable-length integer."""
        if self.offset >= len(self.data):
            raise ValueError("Unpacker: buffer underflow reading int")

        # First byte
        byte = self.data[self.offset]
        self.offset += 1

        sign = (byte >> 6) & 1
        result = byte & 0x3F

        # Check for more bytes
        shift = 6
        while byte & 0x80:
            if self.offset >= len(self.data):
                raise ValueError("Unpacker: buffer underflow in varint")
            if shift > 32:
                raise ValueError("Unpacker: varint too large")

            byte = self.data[self.offset]
            self.offset += 1

            result |= (byte & 0x7F) << shift
            shift += 7

        # Apply sign
        if sign:
            result = ~result

        return result

    def get_string(self) -> str:
        """Read a null-terminated string."""
        start = self.offset
        while self.offset < len(self.data) and self.data[self.offset] != 0:
            self.offset += 1

        result = self.data[start : self.offset].decode("utf-8", errors="replace")

        # Skip null terminator
        if self.offset < len(self.data):
            self.offset += 1

        return result

    def get_raw(self, length: int) -> bytes:
        """Read raw bytes."""
        if self.offset + length > len(self.data):
            raise ValueError(f"Unpacker: buffer underflow reading {length} bytes")

        result = self.data[self.offset : self.offset + length]
        self.offset += length
        return result

    def get_remaining(self) -> bytes:
        """Get all remaining data."""
        result = self.data[self.offset :]
        self.offset = len(self.data)
        return result


def pack_int(value: int) -> bytes:
    """Pack a single integer (convenience function)."""
    p = Packer()
    p.add_int(value)
    return p.data()


def unpack_int(data: bytes) -> Tuple[int, int]:
    """Unpack a single integer, return (value, bytes_consumed)."""
    u = Unpacker(data)
    value = u.get_int()
    return value, u.offset
