"""
Teeworlds Huffman Compression

Implementation of the Huffman coding used in Teeworlds 0.7 network protocol.
Based on the exact algorithm from huffman.cpp in Teeworlds source.
"""

from typing import List, Optional

# Huffman frequency table from Teeworlds 0.7.5 source (huffman.cpp gs_aFreqTable)
# This must match EXACTLY or decompression will fail
# fmt: off
HUFFMAN_FREQ_TABLE = [
    1 << 30, 4545, 2657, 431, 1950, 919, 444, 482, 2244, 617, 838, 542, 715, 1814, 304, 240, 754, 212, 647, 186,
    283, 131, 146, 166, 543, 164, 167, 136, 179, 859, 363, 113, 157, 154, 204, 108, 137, 180, 202, 176,
    872, 404, 168, 134, 151, 111, 113, 109, 120, 126, 129, 100, 41, 20, 16, 22, 18, 18, 17, 19,
    16, 37, 13, 21, 362, 166, 99, 78, 95, 88, 81, 70, 83, 284, 91, 187, 77, 68, 52, 68,
    59, 66, 61, 638, 71, 157, 50, 46, 69, 43, 11, 24, 13, 19, 10, 12, 12, 20, 14, 9,
    20, 20, 10, 10, 15, 15, 12, 12, 7, 19, 15, 14, 13, 18, 35, 19, 17, 14, 8, 5,
    15, 17, 9, 15, 14, 18, 8, 10, 2173, 134, 157, 68, 188, 60, 170, 60, 194, 62, 175, 71,
    148, 67, 167, 78, 211, 67, 156, 69, 1674, 90, 174, 53, 147, 89, 181, 51, 174, 63, 163, 80,
    167, 94, 128, 122, 223, 153, 218, 77, 200, 110, 190, 73, 174, 69, 145, 66, 277, 143, 141, 60,
    136, 53, 180, 57, 142, 57, 158, 61, 166, 112, 152, 92, 26, 22, 21, 28, 20, 26, 30, 21,
    32, 27, 20, 17, 23, 21, 30, 22, 22, 21, 27, 25, 17, 27, 23, 18, 39, 26, 15, 21,
    12, 18, 18, 27, 20, 18, 15, 19, 11, 17, 33, 12, 18, 15, 19, 18, 16, 26, 17, 18,
    9, 10, 25, 22, 22, 17, 20, 16, 6, 16, 15, 20, 14, 18, 24, 335, 1517,
]
# fmt: on

EOF_SYMBOL = 256
HUFFMAN_MAX_SYMBOLS = 257  # 256 bytes + EOF
HUFFMAN_MAX_NODES = HUFFMAN_MAX_SYMBOLS * 2 - 1
HUFFMAN_LUTBITS = 10
HUFFMAN_LUTSIZE = 1 << HUFFMAN_LUTBITS
HUFFMAN_LUTMASK = HUFFMAN_LUTSIZE - 1


class HuffmanNode:
    """Node in the Huffman tree."""

    __slots__ = ['bits', 'num_bits', 'leafs', 'symbol']

    def __init__(self):
        self.bits: int = 0
        self.num_bits: int = 0
        self.leafs: List[int] = [0xFFFF, 0xFFFF]  # Use 0xFFFF as sentinel like TW
        self.symbol: int = -1


class HuffmanConstructNode:
    """Temporary node for tree construction."""

    __slots__ = ['node_id', 'frequency']

    def __init__(self, node_id: int, frequency: int):
        self.node_id = node_id
        self.frequency = frequency


class Huffman:
    """Huffman encoder/decoder for Teeworlds protocol.
    
    This implementation matches the exact algorithm from Teeworlds 0.7.5 huffman.cpp
    to ensure bit-perfect compatibility.
    """

    def __init__(self):
        self.nodes: List[HuffmanNode] = []
        self.decode_lut: List[Optional[HuffmanNode]] = [None] * HUFFMAN_LUTSIZE
        self.start_node: Optional[HuffmanNode] = None
        self.num_nodes: int = 0

        self._build_tree()

    def _bubble_sort(self, nodes: List[HuffmanConstructNode]):
        """Bubble sort by descending frequency (highest first).
        
        Must use bubble sort for deterministic results matching TW's implementation.
        """
        size = len(nodes)
        changed = True
        while changed:
            changed = False
            for i in range(size - 1):
                # Sort descending (higher frequency first)
                if nodes[i].frequency < nodes[i + 1].frequency:
                    nodes[i], nodes[i + 1] = nodes[i + 1], nodes[i]
                    changed = True
            size -= 1

    def _setbits_r(self, node: HuffmanNode, bits: int, depth: int):
        """Recursively set bits for nodes (matches TW's Setbits_r)."""
        if node.leafs[1] != 0xFFFF:
            self._setbits_r(self.nodes[node.leafs[1]], bits | (1 << depth), depth + 1)
        if node.leafs[0] != 0xFFFF:
            self._setbits_r(self.nodes[node.leafs[0]], bits, depth + 1)

        if node.num_bits:
            node.bits = bits
            node.num_bits = depth

    def _build_tree(self):
        """Build the Huffman tree from frequency table.
        
        Matches TW's ConstructTree() exactly.
        """
        # Initialize nodes
        self.nodes = [HuffmanNode() for _ in range(HUFFMAN_MAX_NODES)]
        self.num_nodes = HUFFMAN_MAX_SYMBOLS

        # Create construction nodes list
        nodes_left: List[HuffmanConstructNode] = []

        # Add the symbols
        for i in range(HUFFMAN_MAX_SYMBOLS):
            self.nodes[i].num_bits = 0xFFFFFFFF
            self.nodes[i].symbol = i
            self.nodes[i].leafs = [0xFFFF, 0xFFFF]

            # EOF symbol gets frequency 1, others use table
            if i == EOF_SYMBOL:
                freq = 1
            else:
                freq = HUFFMAN_FREQ_TABLE[i]

            nodes_left.append(HuffmanConstructNode(i, freq))

        # Construct the tree
        while len(nodes_left) > 1:
            # Bubble sort descending by frequency
            self._bubble_sort(nodes_left)

            # Combine two lowest frequency nodes (at the end after descending sort)
            num_left = len(nodes_left)

            self.nodes[self.num_nodes].num_bits = 0
            self.nodes[self.num_nodes].leafs[0] = nodes_left[num_left - 1].node_id
            self.nodes[self.num_nodes].leafs[1] = nodes_left[num_left - 2].node_id

            # Update second-to-last node to be the new parent
            nodes_left[num_left - 2].node_id = self.num_nodes
            nodes_left[num_left - 2].frequency = (
                nodes_left[num_left - 1].frequency + nodes_left[num_left - 2].frequency
            )

            self.num_nodes += 1
            nodes_left.pop()  # Remove last node

        # Set start node
        self.start_node = self.nodes[self.num_nodes - 1]

        # Build symbol bits
        self._setbits_r(self.start_node, 0, 0)

        # Build decode LUT
        self._build_decode_lut()

    def _build_decode_lut(self):
        """Build lookup table for fast decoding.
        
        Matches TW's LUT building in Init().
        """
        for i in range(HUFFMAN_LUTSIZE):
            bits = i
            node = self.start_node

            for k in range(HUFFMAN_LUTBITS):
                node = self.nodes[node.leafs[bits & 1]]
                bits >>= 1

                if node is None:
                    break

                if node.num_bits:
                    self.decode_lut[i] = node
                    break
            else:
                # Finished all bits without finding a leaf
                self.decode_lut[i] = node

    def compress(self, data: bytes) -> bytes:
        """Compress data using Huffman coding.
        
        Matches TW's Compress() function.
        """
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

        while bitcount >= 8:
            result.append(bits & 0xFF)
            bits >>= 8
            bitcount -= 8

        # Write out the last bits
        if bitcount > 0:
            result.append(bits & 0xFF)

        return bytes(result)

    def decompress(self, data: bytes) -> bytes:
        """Decompress Huffman-encoded data.
        
        Matches TW's Decompress() function exactly.
        """
        if not data:
            return b""

        result = bytearray()
        src_idx = 0
        bits = 0
        bitcount = 0

        eof_node = self.nodes[EOF_SYMBOL]

        while True:
            # {A} try to load a node now, this will reduce dependency at location {D}
            node = None
            if bitcount >= HUFFMAN_LUTBITS:
                node = self.decode_lut[bits & HUFFMAN_LUTMASK]

            # {B} fill with new bits
            while bitcount < 24 and src_idx < len(data):
                bits |= data[src_idx] << bitcount
                bitcount += 8
                src_idx += 1

            # {C} load symbol now if we didn't earlier at location {A}
            if node is None:
                node = self.decode_lut[bits & HUFFMAN_LUTMASK]

            if node is None:
                return bytes(result)  # Error

            # {D} check if we hit a symbol already
            if node.num_bits:
                # Remove the bits for that symbol
                bits >>= node.num_bits
                bitcount -= node.num_bits
            else:
                # Remove the bits that the LUT checked up for us
                bits >>= HUFFMAN_LUTBITS
                bitcount -= HUFFMAN_LUTBITS

                # Walk the tree bit by bit
                while True:
                    # Traverse tree
                    if node.leafs[bits & 1] == 0xFFFF:
                        return bytes(result)  # Error
                    node = self.nodes[node.leafs[bits & 1]]

                    # Remove bit
                    bitcount -= 1
                    bits >>= 1

                    # Check if we hit a symbol
                    if node.num_bits:
                        break

                    # No more bits, decoding error
                    if bitcount == 0:
                        return bytes(result)

            # Check for EOF
            if node is eof_node:
                break

            # Output character
            if node.symbol >= 0 and node.symbol < 256:
                result.append(node.symbol)

        return bytes(result)
