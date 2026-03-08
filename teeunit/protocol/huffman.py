"""
Teeworlds Huffman Compression

Implementation of the Huffman coding used in Teeworlds network protocol.
Based on the frequency table from the Teeworlds source code.
"""

from typing import List, Optional

# Huffman frequency table from Teeworlds source
# fmt: off
HUFFMAN_FREQ_TABLE = [
    1 << 30, 4545, 2657, 431, 1950, 919, 444, 482, 2244, 617, 838, 542, 715, 1814, 304, 240,
    754, 212, 647, 186, 283, 131, 146, 166, 543, 164, 167, 104, 117, 178, 135, 131,
    172, 139, 103, 131, 155, 151, 205, 218, 152, 140, 103, 186, 230, 82, 134, 134,
    192, 129, 120, 131, 145, 163, 127, 131, 157, 136, 116, 171, 147, 153, 141, 205,
    180, 149, 177, 144, 199, 165, 202, 155, 138, 135, 105, 172, 163, 143, 196, 152,
    197, 134, 166, 173, 144, 147, 147, 135, 145, 163, 179, 167, 144, 164, 150, 209,
    125, 114, 153, 153, 105, 114, 173, 143, 135, 115, 137, 159, 166, 132, 122, 149,
    175, 168, 143, 144, 170, 169, 131, 132, 146, 141, 174, 143, 197, 158, 195, 154,
    163, 161, 193, 152, 191, 116, 186, 143, 143, 141, 161, 131, 134, 144, 186, 152,
    166, 132, 169, 157, 139, 139, 148, 144, 205, 132, 199, 151, 136, 133, 171, 175,
    153, 161, 141, 193, 166, 162, 153, 143, 128, 132, 159, 150, 153, 140, 190, 139,
    179, 166, 172, 139, 189, 143, 156, 167, 147, 128, 138, 141, 173, 117, 132, 147,
    157, 165, 137, 186, 132, 193, 159, 146, 205, 139, 157, 138, 145, 177, 169, 157,
    151, 142, 167, 139, 152, 144, 176, 174, 125, 159, 123, 155, 199, 139, 175, 199,
    133, 152, 185, 183, 128, 117, 204, 134, 161, 135, 174, 155, 173, 127, 157, 160,
    113, 185, 151, 212, 219, 136, 108, 172, 200, 182, 151, 154, 132, 194, 123, 155,
    EOF_SYMBOL := 256,
]
# fmt: on

# Replace the inline EOF_SYMBOL with actual value
HUFFMAN_FREQ_TABLE[-1] = 256
EOF_SYMBOL = 256
HUFFMAN_MAX_SYMBOLS = 257  # 256 bytes + EOF
HUFFMAN_MAX_NODES = HUFFMAN_MAX_SYMBOLS * 2 - 1
HUFFMAN_LUTBITS = 10
HUFFMAN_LUTSIZE = 1 << HUFFMAN_LUTBITS
HUFFMAN_LUTMASK = HUFFMAN_LUTSIZE - 1


class HuffmanNode:
    """Node in the Huffman tree."""

    def __init__(self):
        self.bits: int = 0
        self.num_bits: int = 0
        self.leafs: List[int] = [0, 0]  # [left, right] - indices or symbols
        self.symbol: int = -1


class Huffman:
    """Huffman encoder/decoder for Teeworlds protocol."""

    def __init__(self):
        self.nodes: List[HuffmanNode] = []
        self.decode_lut: List[Optional[HuffmanNode]] = [None] * HUFFMAN_LUTSIZE
        self.start_node: Optional[HuffmanNode] = None
        self.num_nodes: int = 0

        self._build_tree()

    def _build_tree(self):
        """Build the Huffman tree from frequency table."""
        # Initialize nodes
        self.nodes = [HuffmanNode() for _ in range(HUFFMAN_MAX_NODES)]
        self.num_nodes = HUFFMAN_MAX_SYMBOLS

        # Initialize leaf nodes
        for i in range(HUFFMAN_MAX_SYMBOLS):
            self.nodes[i].symbol = i
            self.nodes[i].num_bits = 0xFFFFFFFF
            self.nodes[i].leafs = [-1, -1]

        # Build tree using frequency table
        # Use node_indices for tree building
        node_indices = list(range(HUFFMAN_MAX_SYMBOLS))

        while len(node_indices) > 1:
            # Find two nodes with lowest frequency
            # Sort by frequency (approximated by symbol index for initial nodes)
            node_indices.sort(
                key=lambda x: HUFFMAN_FREQ_TABLE[self.nodes[x].symbol]
                if self.nodes[x].symbol >= 0 and self.nodes[x].symbol < len(HUFFMAN_FREQ_TABLE)
                else self.nodes[x].num_bits
            )

            # Take two lowest
            left_idx = node_indices.pop(0)
            right_idx = node_indices.pop(0)

            # Create parent node
            parent_idx = self.num_nodes
            self.num_nodes += 1
            parent = self.nodes[parent_idx]
            parent.leafs = [left_idx, right_idx]
            parent.symbol = -1

            # Calculate combined frequency (using num_bits as temp storage)
            left_freq = (
                HUFFMAN_FREQ_TABLE[self.nodes[left_idx].symbol]
                if self.nodes[left_idx].symbol >= 0
                and self.nodes[left_idx].symbol < len(HUFFMAN_FREQ_TABLE)
                else 1
            )
            right_freq = (
                HUFFMAN_FREQ_TABLE[self.nodes[right_idx].symbol]
                if self.nodes[right_idx].symbol >= 0
                and self.nodes[right_idx].symbol < len(HUFFMAN_FREQ_TABLE)
                else 1
            )
            parent.num_bits = left_freq + right_freq

            node_indices.append(parent_idx)

        # Set start node
        if node_indices:
            self.start_node = self.nodes[node_indices[0]]

        # Generate codes by traversing tree
        self._generate_codes(node_indices[0] if node_indices else 0, 0, 0)

        # Build decode LUT
        self._build_decode_lut()

    def _generate_codes(self, node_idx: int, bits: int, depth: int):
        """Generate Huffman codes by traversing the tree."""
        node = self.nodes[node_idx]

        if node.symbol >= 0:
            # Leaf node
            node.bits = bits
            node.num_bits = depth
            return

        # Internal node - traverse children
        if node.leafs[0] >= 0:
            self._generate_codes(node.leafs[0], bits, depth + 1)
        if node.leafs[1] >= 0:
            self._generate_codes(node.leafs[1], bits | (1 << depth), depth + 1)

    def _build_decode_lut(self):
        """Build lookup table for fast decoding."""
        # Simple LUT - for each possible LUTBITS input, store the node
        for i in range(HUFFMAN_LUTSIZE):
            node = self.start_node
            for bit in range(HUFFMAN_LUTBITS):
                if node is None or node.symbol >= 0:
                    break
                direction = (i >> bit) & 1
                next_idx = node.leafs[direction]
                if next_idx >= 0:
                    node = self.nodes[next_idx]
                else:
                    break
            self.decode_lut[i] = node

    def compress(self, data: bytes) -> bytes:
        """Compress data using Huffman coding."""
        if not data:
            return b""

        result = bytearray()
        bits = 0
        bitcount = 0

        for byte in data:
            node = self.nodes[byte]
            bits |= node.bits << bitcount
            bitcount += node.num_bits

            while bitcount >= 8:
                result.append(bits & 0xFF)
                bits >>= 8
                bitcount -= 8

        # Add EOF
        eof_node = self.nodes[EOF_SYMBOL]
        bits |= eof_node.bits << bitcount
        bitcount += eof_node.num_bits

        # Flush remaining bits
        while bitcount > 0:
            result.append(bits & 0xFF)
            bits >>= 8
            bitcount -= 8

        return bytes(result)

    def decompress(self, data: bytes) -> bytes:
        """Decompress Huffman-encoded data."""
        if not data:
            return b""

        result = bytearray()
        bits = 0
        bitcount = 0
        src_idx = 0

        while True:
            # Refill bits buffer
            while bitcount < 24 and src_idx < len(data):
                bits |= data[src_idx] << bitcount
                bitcount += 8
                src_idx += 1

            # Decode using tree traversal
            node = self.start_node
            while node and node.symbol < 0:
                if bitcount <= 0:
                    # Ran out of bits
                    return bytes(result)
                direction = bits & 1
                bits >>= 1
                bitcount -= 1
                next_idx = node.leafs[direction]
                if next_idx >= 0:
                    node = self.nodes[next_idx]
                else:
                    break

            if node is None:
                break

            # Check for EOF
            if node.symbol == EOF_SYMBOL:
                break

            # Output symbol
            if node.symbol >= 0 and node.symbol < 256:
                result.append(node.symbol)

        return bytes(result)
