"""
Teeworlds Network Client

A Python client that connects to Teeworlds servers using the native
UDP protocol. Handles connection, input sending, and snapshot receiving.
"""

import asyncio
import socket
import random
import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Tuple, Any

from .const import (
    NET_CTRLMSG_CONNECT,
    NET_CTRLMSG_CONNECTACCEPT,
    NET_CTRLMSG_ACCEPT,
    NET_CTRLMSG_CLOSE,
    NET_CTRLMSG_KEEPALIVE,
    NET_CTRLMSG_TOKEN,
    NET_CHUNKFLAG_VITAL,
    NET_SEQUENCE_MASK,
    NETMSG_INFO,
    NETMSG_MAP_CHANGE,
    NETMSG_MAP_DATA,
    NETMSG_CON_READY,
    NETMSG_SNAP,
    NETMSG_SNAPEMPTY,
    NETMSG_SNAPSINGLE,
    NETMSG_SNAPSMALL,
    NETMSG_READY,
    NETMSG_ENTERGAME,
    NETMSG_INPUT,
    NETMSG_PING,
    NETMSG_PING_REPLY,
    NETMSG_INPUTTIMING,
    NETMSGTYPE_SV_READYTOENTER,
    NETMSGTYPE_CL_STARTINFO,
    NETMSGTYPE_SV_KILLMSG,
    NETMSGTYPE_SV_CHAT,
    NET_VERSION_STR,
    NET_TOKEN_NONE,
    SERVER_TICK_SPEED,
)
from .huffman import Huffman
from .packet import Packet, pack_control_packet
from .chunk import Chunk, ChunkBuilder
from .packer import Packer
from .snapshot import Snapshot, SnapshotUnpacker
from .objects import PlayerInput, Character, PlayerInfo


logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Client connection state."""

    OFFLINE = auto()
    TOKEN = auto()       # Waiting for token response
    CONNECTING = auto()  # Sent CONNECT, waiting for ACCEPT
    LOADING = auto()     # Loading map / sending info
    ONLINE = auto()
    ERROR = auto()


@dataclass
class KillEvent:
    """A kill event from the server."""

    killer_id: int
    victim_id: int
    weapon: int
    tick: int


@dataclass
class ChatMessage:
    """A chat message from the server."""

    mode: int
    client_id: int
    target_id: int
    message: str


@dataclass
class TwClient:
    """Teeworlds network client.
    
    Connects to a Teeworlds server and provides methods to:
    - Send player input
    - Receive game state snapshots
    - Track kills and deaths
    """

    host: str = "127.0.0.1"
    port: int = 8303
    name: str = "TeeBot"
    clan: str = ""
    country: int = -1

    # Connection state
    state: ConnectionState = field(default=ConnectionState.OFFLINE)
    token: int = field(default=NET_TOKEN_NONE)  # Our token
    peer_token: int = field(default=NET_TOKEN_NONE)  # Server's token
    ack: int = 0
    sequence: int = 0
    client_id: int = -1

    # Network
    _socket: Optional[socket.socket] = field(default=None, repr=False)
    _huffman: Huffman = field(default_factory=Huffman, repr=False)

    # Game state
    current_tick: int = 0
    current_snapshot: Optional[Snapshot] = field(default=None, repr=False)
    _snapshot_unpacker: SnapshotUnpacker = field(
        default_factory=SnapshotUnpacker, repr=False
    )

    # Input state
    _input: PlayerInput = field(default_factory=PlayerInput, repr=False)
    _input_ack_tick: int = 0
    _fire_count: int = 0

    # Events
    kill_events: List[KillEvent] = field(default_factory=list)
    chat_messages: List[ChatMessage] = field(default_factory=list)

    # Callbacks
    on_snapshot: Optional[Callable[[Snapshot], None]] = field(default=None, repr=False)
    on_kill: Optional[Callable[[KillEvent], None]] = field(default=None, repr=False)
    on_connected: Optional[Callable[[], None]] = field(default=None, repr=False)
    on_disconnected: Optional[Callable[[str], None]] = field(default=None, repr=False)

    # Timing
    _last_recv_time: float = field(default=0, repr=False)
    _last_send_time: float = field(default=0, repr=False)
    _last_input_time: float = field(default=0, repr=False)
    
    # Keepalive interval (send INPUT at least every 200ms to prevent timeout)
    KEEPALIVE_INTERVAL: float = field(default=0.2, repr=False)

    def __post_init__(self):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        # Generate a random token for ourselves
        self.token = random.randint(0, NET_TOKEN_NONE - 1)
    
    def _token_to_bytes(self, token: int) -> bytes:
        """Convert token int to big-endian bytes."""
        return bytes([
            (token >> 24) & 0xFF,
            (token >> 16) & 0xFF,
            (token >> 8) & 0xFF,
            token & 0xFF,
        ])
    
    def _bytes_to_token(self, data: bytes) -> int:
        """Convert big-endian bytes to token int."""
        if len(data) < 4:
            return NET_TOKEN_NONE
        return (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3]

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self.state == ConnectionState.ONLINE

    @property
    def address(self) -> Tuple[str, int]:
        """Get server address tuple."""
        return (self.host, self.port)

    def _send_raw(self, data: bytes):
        """Send raw data to server."""
        if self._socket:
            try:
                self._socket.sendto(data, self.address)
                self._last_send_time = time.monotonic()
                logger.debug(f"SEND [{len(data)}]: {data.hex()}")
            except OSError as e:
                logger.error(f"Send error: {e}")

    def _send_control(self, msg: int, extra: bytes = b""):
        """Send a control message."""
        # Use peer_token in header (or 0xffffffff if unknown)
        token_bytes = self._token_to_bytes(self.peer_token)
        packet = pack_control_packet(msg, extra, token_bytes, self.ack)
        self._send_raw(packet)
    
    def _send_control_with_token(self, msg: int):
        """Send a control message that includes our token in the extra data.
        
        Used for TOKEN and CONNECT messages in Teeworlds 0.7.
        The extra data contains our token (4 bytes) followed by 508 bytes padding.
        """
        # Our token goes in the extra data (big-endian, 4 bytes)
        extra = self._token_to_bytes(self.token)
        # Extended version sends 512 bytes of extra data for TOKEN requests
        if msg == NET_CTRLMSG_TOKEN:
            extra = extra + b'\x00' * 508
        self._send_control(msg, extra)

    def _send_ack_packet(self):
        """Send an empty packet to acknowledge received vital chunks.
        
        This sends a packet with no chunks but with our current ack value,
        letting the server know we've received all vital messages up to that sequence.
        """
        # Build an empty packet header with just the ack
        # Format: flags(6) + ack_high(2), ack_low(8), num_chunks(8), token(32)
        token_bytes = self._token_to_bytes(self.peer_token)
        packet = bytearray()
        packet.append(((0 << 2) & 0xFC) | ((self.ack >> 8) & 0x03))  # flags=0, ack high bits
        packet.append(self.ack & 0xFF)  # ack low bits
        packet.append(0)  # num_chunks = 0
        packet.extend(token_bytes)
        self._send_raw(bytes(packet))

    def _next_sequence(self) -> int:
        """Get next sequence number."""
        self.sequence = (self.sequence + 1) & NET_SEQUENCE_MASK
        return self.sequence

    def _send_chunks(self, chunks: List[Tuple[bytes, int]], compress: bool = True):
        """Send chunks in a packet."""
        # For vital chunks, we need to set the sequence number
        # Increment sequence for each vital chunk
        processed_chunks = []
        for chunk_data, chunk_flags in chunks:
            if chunk_flags & NET_CHUNKFLAG_VITAL:
                seq = self._next_sequence()
                processed_chunks.append((chunk_data, chunk_flags, seq))
            else:
                processed_chunks.append((chunk_data, chunk_flags, -1))
        
        packet = Packet()
        # Use peer_token (server's token) in the header
        token_bytes = self._token_to_bytes(self.peer_token)
        data = packet.pack_with_sequence(
            processed_chunks,
            token_bytes,
            self.ack,
            compress=compress,
            huffman=self._huffman,
        )
        self._send_raw(data)

    def _send_msg(
        self,
        msg_id: int,
        sys: bool,
        vital: bool = True,
        *,
        ints: List[int] = None,
        strings: List[str] = None,
        raw: bytes = None,
    ):
        """Send a single message."""
        packer = Packer()

        # Message header
        header = (msg_id << 1) | (1 if sys else 0)
        packer.add_int(header)

        # Add data
        if ints:
            for i in ints:
                packer.add_int(i)
        if strings:
            for s in strings:
                packer.add_string(s)
        if raw:
            packer.add_raw(raw)

        flags = NET_CHUNKFLAG_VITAL if vital else 0
        self._send_chunks([(packer.data(), flags)])

    def connect(self):
        """Initiate connection to server.
        
        Teeworlds 0.7 connection sequence:
        1. Client sends TOKEN request with its own token
        2. Server responds with TOKEN containing server's token
        3. Client sends CONNECT with its token
        4. Server responds with ACCEPT
        5. Connection established, client sends INFO
        """
        logger.info(f"Connecting to {self.host}:{self.port}...")
        self.state = ConnectionState.TOKEN
        self.peer_token = NET_TOKEN_NONE  # Unknown initially
        
        # Generate new token for ourselves
        self.token = random.randint(0, NET_TOKEN_NONE - 1)

        # Send TOKEN request
        self._send_control_with_token(NET_CTRLMSG_TOKEN)

    def disconnect(self, reason: str = ""):
        """Disconnect from server."""
        if self.state != ConnectionState.OFFLINE:
            self._send_control(NET_CTRLMSG_CLOSE, reason.encode())
            self.state = ConnectionState.OFFLINE
            if self.on_disconnected:
                self.on_disconnected(reason)

    def send_input(self, inp: PlayerInput):
        """Send player input to server."""
        if self.state != ConnectionState.ONLINE:
            return

        # Debug log when fire counter changes
        if inp.fire > 0 and inp.fire != getattr(self, '_last_fire', 0):
            logger.info(f"FIRE! counter={inp.fire} weapon={inp.wanted_weapon} target=({inp.target_x}, {inp.target_y})")
            self._last_fire = inp.fire

        self._input = inp
        self._last_input_time = time.monotonic()

        # Build input message
        packer = Packer()
        header = (NETMSG_INPUT << 1) | 1  # sys=True
        packer.add_int(header)
        packer.add_int(self._input_ack_tick)
        packer.add_int(1)  # prediction margin

        # Input size (10 ints)
        packer.add_int(10)

        # Input data
        for val in inp.to_ints():
            packer.add_int(val)

        self._send_chunks([(packer.data(), NET_CHUNKFLAG_VITAL)])

    def send_fire(self, target_x: int = 0, target_y: int = 0):
        """Convenience method to fire weapon."""
        self._fire_count += 1
        inp = PlayerInput(
            target_x=target_x,
            target_y=target_y,
            fire=self._fire_count,
        )
        self.send_input(inp)

    def pump(self, timeout: float = 0.001) -> bool:
        """Process incoming packets.
        
        Returns True if a packet was processed.
        """
        if not self._socket:
            return False

        # Send keepalive INPUT if we're online and haven't sent input recently
        # This prevents the server from timing us out
        if self.state == ConnectionState.ONLINE:
            now = time.monotonic()
            if now - self._last_input_time >= self.KEEPALIVE_INTERVAL:
                self.send_input(self._input)

        try:
            data, addr = self._socket.recvfrom(2048)
        except BlockingIOError:
            return False
        except OSError as e:
            logger.error(f"Receive error: {e}")
            return False

        self._last_recv_time = time.monotonic()
        logger.debug(f"RECV [{len(data)}]: {data.hex()}")

        return self._handle_packet(data)

    def _handle_packet(self, data: bytes) -> bool:
        """Handle an incoming packet."""
        if len(data) < 7:
            return False

        packet = Packet()
        if not packet.unpack(data, self._huffman):
            return False
        
        # Verify token matches our token (except during initial handshake)
        received_token = self._bytes_to_token(packet.token)
        if self.state not in (ConnectionState.OFFLINE, ConnectionState.TOKEN):
            if received_token != self.token and received_token != NET_TOKEN_NONE:
                logger.debug(f"Token mismatch: got {received_token:08x}, expected {self.token:08x}")
                return False

        # Update ack from the highest vital sequence we received
        # This tells the server we've received all messages up to this sequence
        if packet.max_recv_sequence >= 0:
            old_ack = self.ack
            self.ack = packet.max_recv_sequence
            
            # If ack changed, we need to send it back to the server
            # The server needs ACK confirmation or it will timeout the client
            if self.ack != old_ack and self.state == ConnectionState.ONLINE:
                self._send_ack_packet()

        if packet.is_control:
            return self._handle_control(packet)

        # Handle chunks
        for chunk in packet.chunks:
            self._handle_chunk(chunk)

        return True

    def _handle_control(self, packet: Packet) -> bool:
        """Handle control packet."""
        msg = packet.ctrl_msg
        data = packet.ctrl_data

        if msg == NET_CTRLMSG_TOKEN:
            # Server is sending us its token
            # The ctrl_data contains the server's token (first 4 bytes, big-endian)
            if len(data) >= 4:
                self.peer_token = self._bytes_to_token(data[:4])
                logger.info(f"Got server token: {self.peer_token:08x}")
                
                if self.state == ConnectionState.TOKEN:
                    # Move to CONNECTING state and send CONNECT
                    self.state = ConnectionState.CONNECTING
                    self._send_control_with_token(NET_CTRLMSG_CONNECT)
            return True
        
        elif msg == NET_CTRLMSG_ACCEPT:
            # Connection accepted!
            logger.info("Connection accepted!")
            
            # Send client info
            self._send_client_info()
            
            self.state = ConnectionState.LOADING
            return True

        elif msg == NET_CTRLMSG_CLOSE:
            reason = data.decode("utf-8", errors="replace") if data else "Unknown"
            logger.info(f"Disconnected: {reason}")
            self.state = ConnectionState.OFFLINE
            if self.on_disconnected:
                self.on_disconnected(reason)
            return True

        elif msg == NET_CTRLMSG_KEEPALIVE:
            # Respond with keepalive
            self._send_control(NET_CTRLMSG_KEEPALIVE)
            return True

        return False

    def _send_client_info(self):
        """Send client version and info."""
        # Send NETMSG_INFO
        # TW 0.7 format: version_string, password_string, client_version_int
        # Need to construct manually since _send_msg puts ints before strings
        packer = Packer()
        header = (NETMSG_INFO << 1) | 1  # msg_id=1, sys=True
        packer.add_int(header)
        packer.add_string(NET_VERSION_STR)  # version
        packer.add_string("")  # password
        packer.add_int(0x0705)  # client version (0.7.5)
        
        self._send_chunks([(packer.data(), NET_CHUNKFLAG_VITAL)])

    def _handle_chunk(self, chunk: Chunk):
        """Handle a message chunk."""
        if chunk.sys:
            self._handle_system_msg(chunk)
        else:
            self._handle_game_msg(chunk)

    def _handle_system_msg(self, chunk: Chunk):
        """Handle system message."""
        msg_id = chunk.msg_id

        if msg_id == NETMSG_MAP_CHANGE:
            # Map change - we need to "download" the map
            # For simplicity, just send ready
            logger.info("Map change received, sending ready")
            self._send_msg(NETMSG_READY, sys=True)

        elif msg_id == NETMSG_CON_READY:
            # Server ready for us to enter
            logger.info("Server ready, sending start info")
            self._send_start_info()

        elif msg_id in (NETMSG_SNAP, NETMSG_SNAPSINGLE, NETMSG_SNAPEMPTY, NETMSG_SNAPSMALL):
            self._handle_snapshot(chunk)

        elif msg_id == NETMSG_INPUTTIMING:
            # Input timing info
            try:
                intended_tick = chunk.get_int()
                time_left = chunk.get_int()
                self._input_ack_tick = intended_tick
            except ValueError:
                pass

        elif msg_id == NETMSG_PING:
            # Respond to ping
            self._send_msg(NETMSG_PING_REPLY, sys=True)

    def _handle_game_msg(self, chunk: Chunk):
        """Handle game message."""
        msg_id = chunk.msg_id

        if msg_id == NETMSGTYPE_SV_READYTOENTER:
            # Ready to enter game
            logger.info("Ready to enter, sending enter game")
            self._send_msg(NETMSG_ENTERGAME, sys=True)
            self.state = ConnectionState.ONLINE
            if self.on_connected:
                self.on_connected()

        elif msg_id == NETMSGTYPE_SV_KILLMSG:
            # Kill notification
            try:
                killer = chunk.get_int()
                victim = chunk.get_int()
                weapon = chunk.get_int()
                event = KillEvent(
                    killer_id=killer,
                    victim_id=victim,
                    weapon=weapon,
                    tick=self.current_tick,
                )
                self.kill_events.append(event)
                logger.debug(f"Kill: {killer} killed {victim} with {weapon}")
                if self.on_kill:
                    self.on_kill(event)
            except ValueError:
                pass

        elif msg_id == NETMSGTYPE_SV_CHAT:
            # Chat message
            try:
                mode = chunk.get_int()
                client_id = chunk.get_int()
                target_id = chunk.get_int()
                message = chunk.get_string()
                chat = ChatMessage(
                    mode=mode,
                    client_id=client_id,
                    target_id=target_id,
                    message=message,
                )
                self.chat_messages.append(chat)
            except ValueError:
                pass

    def _send_start_info(self):
        """Send player start info (name, clan, skin)."""
        # Build NETMSGTYPE_CL_STARTINFO
        packer = Packer()
        header = (NETMSGTYPE_CL_STARTINFO << 1) | 0  # sys=False
        packer.add_int(header)
        packer.add_string(self.name)
        packer.add_string(self.clan)
        packer.add_int(self.country)

        # Skin parts (6 parts for 0.7)
        skin_parts = ["standard", "", "", "standard", "standard", "standard"]
        for part in skin_parts:
            packer.add_string(part)

        # Use custom colors (6 bools)
        for _ in range(6):
            packer.add_int(0)

        # Skin colors (6 ints)
        for _ in range(6):
            packer.add_int(0)

        self._send_chunks([(packer.data(), NET_CHUNKFLAG_VITAL)])

    def _handle_snapshot(self, chunk: Chunk):
        """Handle snapshot message."""
        try:
            tick = chunk.get_int()
            delta_tick = chunk.get_int()

            if chunk.msg_id == NETMSG_SNAPEMPTY:
                # Empty snapshot
                self.current_tick = tick
                return

            # Get remaining data as snapshot
            snap_data = chunk.get_remaining()

            snapshot = self._snapshot_unpacker.unpack_snapshot(
                snap_data, tick, delta_tick
            )

            if snapshot:
                self.current_tick = tick
                self.current_snapshot = snapshot

                if self.on_snapshot:
                    self.on_snapshot(snapshot)

        except ValueError as e:
            logger.debug(f"Snapshot parse error: {e}")

    def get_character(self, client_id: int) -> Optional[Character]:
        """Get character state for a client."""
        if self.current_snapshot:
            return self.current_snapshot.characters.get(client_id)
        return None

    def get_player_info(self, client_id: int) -> Optional[PlayerInfo]:
        """Get player info for a client."""
        if self.current_snapshot:
            return self.current_snapshot.player_infos.get(client_id)
        return None

    def close(self):
        """Close the client."""
        self.disconnect()
        if self._socket:
            self._socket.close()
            self._socket = None


class AsyncTwClient(TwClient):
    """Async version of TwClient."""

    async def connect_async(self, timeout: float = 5.0):
        """Connect to server asynchronously."""
        self.connect()

        start = time.monotonic()
        while self.state in (ConnectionState.TOKEN, ConnectionState.CONNECTING):
            if time.monotonic() - start > timeout:
                raise TimeoutError("Connection timeout")
            self.pump()
            await asyncio.sleep(0.01)

        # Wait for full connection
        while self.state == ConnectionState.LOADING:
            if time.monotonic() - start > timeout:
                raise TimeoutError("Loading timeout")
            self.pump()
            await asyncio.sleep(0.01)

    async def pump_async(self):
        """Async version of pump."""
        loop = asyncio.get_event_loop()
        try:
            data = await asyncio.wait_for(
                loop.sock_recv(self._socket, 2048),
                timeout=0.001,
            )
            self._handle_packet(data)
            return True
        except asyncio.TimeoutError:
            return False
        except OSError:
            return False

    async def run_loop(self, tick_callback: Callable[[int], None] = None):
        """Run the client loop."""
        while self.state in (ConnectionState.LOADING, ConnectionState.ONLINE):
            self.pump()

            if tick_callback and self.is_connected:
                tick_callback(self.current_tick)

            await asyncio.sleep(1 / SERVER_TICK_SPEED)
