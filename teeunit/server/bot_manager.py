"""
Bot Manager

Manages multiple TwClient instances that connect to a Teeworlds server.
Provides turn-based stepping by buffering game ticks.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any

from ..protocol.client import TwClient, AsyncTwClient, ConnectionState, KillEvent
from ..protocol.objects import PlayerInput, Character, PlayerInfo, Projectile, Pickup
from ..protocol.snapshot import Snapshot
from ..protocol.const import (
    WEAPON_HAMMER,
    WEAPON_GUN,
    WEAPON_SHOTGUN,
    WEAPON_GRENADE,
    WEAPON_LASER,
    WEAPON_NINJA,
    SERVER_TICK_SPEED,
)

logger = logging.getLogger(__name__)

# Default ticks per step (200ms at 50 ticks/sec)
DEFAULT_TICKS_PER_STEP = 10


@dataclass
class BotState:
    """State for a single bot client."""
    
    client_id: int  # Our client ID (0-7)
    client: TwClient = field(repr=False)
    
    # Last known state from snapshots
    character: Optional[Character] = None
    player_info: Optional[PlayerInfo] = None
    
    # Pending input to send
    pending_input: Optional[PlayerInput] = None
    
    # Events since last step
    kills: List[KillEvent] = field(default_factory=list)
    deaths: List[KillEvent] = field(default_factory=list)
    
    # Tracking
    last_snapshot_tick: int = 0
    connected: bool = False


@dataclass 
class GameState:
    """Aggregated game state from all bot perspectives."""
    
    tick: int = 0
    characters: Dict[int, Character] = field(default_factory=dict)
    player_infos: Dict[int, PlayerInfo] = field(default_factory=dict)
    projectiles: List[Projectile] = field(default_factory=list)
    pickups: List[Pickup] = field(default_factory=list)
    kill_events: List[KillEvent] = field(default_factory=list)
    
    def get_character(self, client_id: int) -> Optional[Character]:
        """Get character for a client."""
        return self.characters.get(client_id)
    
    def get_player_info(self, client_id: int) -> Optional[PlayerInfo]:
        """Get player info for a client."""
        return self.player_infos.get(client_id)


class BotManager:
    """
    Manages multiple bot clients connecting to a Teeworlds server.
    
    Provides:
    - Connection management for N bots
    - Turn-based stepping (buffers ticks)
    - Aggregated game state from snapshots
    - Input submission for each bot
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8303,
        num_bots: int = 4,
        ticks_per_step: int = DEFAULT_TICKS_PER_STEP,
        bot_name_prefix: str = "Bot",
    ):
        """
        Initialize the bot manager.
        
        Args:
            host: Teeworlds server host
            port: Teeworlds server port
            num_bots: Number of bots to connect (1-8)
            ticks_per_step: Game ticks per environment step (default 10 = 200ms)
            bot_name_prefix: Prefix for bot names (e.g., "Bot" -> "Bot0", "Bot1", ...)
        """
        self.host = host
        self.port = port
        self.num_bots = min(max(num_bots, 1), 8)
        self.ticks_per_step = ticks_per_step
        self.bot_name_prefix = bot_name_prefix
        
        # Bot clients
        self.bots: Dict[int, BotState] = {}
        
        # Aggregated game state
        self.game_state = GameState()
        
        # Step tracking
        self.step_count = 0
        self.last_step_tick = 0
        
        # Callbacks
        self.on_kill: Optional[Callable[[KillEvent], None]] = None
        self.on_state_update: Optional[Callable[[GameState], None]] = None
        
        # Running state
        self._running = False
        self._pump_task: Optional[asyncio.Task] = None
    
    def _create_bot(self, bot_id: int) -> BotState:
        """Create a bot client."""
        name = f"{self.bot_name_prefix}{bot_id}"
        client = TwClient(
            host=self.host,
            port=self.port,
            name=name,
            clan="TeeUnit",
        )
        
        bot = BotState(
            client_id=bot_id,
            client=client,
        )
        
        # Set up callbacks
        def on_snapshot(snapshot: Snapshot):
            self._handle_snapshot(bot, snapshot)
        
        def on_kill(event: KillEvent):
            self._handle_kill(bot, event)
        
        def on_connected():
            bot.connected = True
            logger.info(f"Bot {bot_id} ({name}) connected")
        
        def on_disconnected(reason: str):
            bot.connected = False
            logger.info(f"Bot {bot_id} ({name}) disconnected: {reason}")
        
        client.on_snapshot = on_snapshot
        client.on_kill = on_kill
        client.on_connected = on_connected
        client.on_disconnected = on_disconnected
        
        return bot
    
    def _handle_snapshot(self, bot: BotState, snapshot: Snapshot):
        """Handle a snapshot from a bot's perspective."""
        bot.last_snapshot_tick = snapshot.tick
        
        # Auto-detect real client ID if not found yet
        # The server assigns IDs that may differ from our bot_id
        if bot.character is None and bot.client_id not in snapshot.characters:
            # Find a character ID that isn't claimed by another bot
            claimed_ids = {b.client_id for b in self.bots.values() if b.character is not None}
            for char_id in snapshot.characters:
                if char_id not in claimed_ids:
                    # Found an unclaimed character - this is likely ours
                    logger.debug(f"Auto-detected client_id {char_id} for bot (was {bot.client_id})")
                    bot.client_id = char_id
                    break
        
        # Update bot's character state
        if bot.client_id in snapshot.characters:
            bot.character = snapshot.characters[bot.client_id]
        
        if bot.client_id in snapshot.player_infos:
            bot.player_info = snapshot.player_infos[bot.client_id]
        
        # Update aggregated game state
        self.game_state.tick = max(self.game_state.tick, snapshot.tick)
        self.game_state.characters.update(snapshot.characters)
        self.game_state.player_infos.update(snapshot.player_infos)
        self.game_state.projectiles = snapshot.projectiles
        self.game_state.pickups = snapshot.pickups
        
        if self.on_state_update:
            self.on_state_update(self.game_state)
    
    def _handle_kill(self, bot: BotState, event: KillEvent):
        """Handle a kill event."""
        # Track kills and deaths for this bot
        if event.killer_id == bot.client_id:
            bot.kills.append(event)
        if event.victim_id == bot.client_id:
            bot.deaths.append(event)
        
        # Add to game state
        self.game_state.kill_events.append(event)
        
        if self.on_kill:
            self.on_kill(event)
        
        logger.debug(f"Kill: {event.killer_id} -> {event.victim_id} (weapon {event.weapon})")
    
    def connect(self, timeout: float = 10.0) -> bool:
        """
        Connect all bots to the server synchronously.
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            True if all bots connected successfully
        """
        logger.info(f"Connecting {self.num_bots} bots to {self.host}:{self.port}...")
        
        # Create and connect bots
        for bot_id in range(self.num_bots):
            bot = self._create_bot(bot_id)
            self.bots[bot_id] = bot
            bot.client.connect()
        
        # Wait for all bots to connect
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            # Pump all clients
            for bot in self.bots.values():
                bot.client.pump()
            
            # Check if all connected
            all_connected = all(
                bot.client.state == ConnectionState.ONLINE
                for bot in self.bots.values()
            )
            
            if all_connected:
                logger.info(f"All {self.num_bots} bots connected!")
                for bot in self.bots.values():
                    bot.connected = True
                    # Get client ID from server
                    # Note: In TW, client IDs are assigned by server order
                    # For now, we assume bot_id matches client_id
                return True
            
            # Check for errors
            any_error = any(
                bot.client.state == ConnectionState.ERROR
                for bot in self.bots.values()
            )
            if any_error:
                logger.warning("Bot connection error")
                break
            
            time.sleep(0.01)
        
        # Timeout - check which bots connected
        connected = sum(1 for b in self.bots.values() if b.client.is_connected)
        logger.warning(f"Connection timeout: {connected}/{self.num_bots} bots connected")
        return False
    
    async def connect_async(self, timeout: float = 10.0) -> bool:
        """
        Connect all bots to the server asynchronously.
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            True if all bots connected successfully
        """
        logger.info(f"Connecting {self.num_bots} bots to {self.host}:{self.port}...")
        
        # Create and connect bots
        for bot_id in range(self.num_bots):
            bot = self._create_bot(bot_id)
            self.bots[bot_id] = bot
            bot.client.connect()
        
        # Wait for all bots to connect
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            # Pump all clients
            for bot in self.bots.values():
                bot.client.pump()
            
            # Check if all connected
            all_connected = all(
                bot.client.state == ConnectionState.ONLINE
                for bot in self.bots.values()
            )
            
            if all_connected:
                logger.info(f"All {self.num_bots} bots connected!")
                for bot in self.bots.values():
                    bot.connected = True
                return True
            
            # Check for errors
            any_error = any(
                bot.client.state == ConnectionState.ERROR
                for bot in self.bots.values()
            )
            if any_error:
                logger.warning("Bot connection error (async)")
                break
            
            await asyncio.sleep(0.01)
        
        connected = sum(1 for b in self.bots.values() if b.client.is_connected)
        logger.warning(f"Connection timeout: {connected}/{self.num_bots} bots connected")
        return False
    
    def disconnect(self):
        """Disconnect all bots."""
        logger.info("Disconnecting all bots...")
        self._running = False
        
        for bot in self.bots.values():
            try:
                bot.client.disconnect()
                bot.client.close()
            except Exception as e:
                logger.error(f"Error disconnecting bot {bot.client_id}: {e}")
        
        self.bots.clear()
    
    def set_input(self, bot_id: int, input: PlayerInput):
        """
        Set pending input for a bot.
        
        The input will be sent on the next pump cycle.
        
        Args:
            bot_id: Bot ID (0 to num_bots-1)
            input: Player input to send
        """
        if bot_id not in self.bots:
            logger.warning(f"Invalid bot_id: {bot_id}")
            return
        
        self.bots[bot_id].pending_input = input
    
    def set_inputs(self, inputs: Dict[int, PlayerInput]):
        """
        Set pending inputs for multiple bots.
        
        Args:
            inputs: Dict mapping bot_id to PlayerInput
        """
        for bot_id, inp in inputs.items():
            self.set_input(bot_id, inp)
    
    def pump(self) -> int:
        """
        Process network for all bots.
        
        Receives snapshots and sends pending inputs.
        
        Returns:
            Number of packets processed
        """
        packets = 0
        
        for bot in self.bots.values():
            # Receive packets
            while bot.client.pump():
                packets += 1
            
            # Send pending input
            if bot.pending_input and bot.client.is_connected:
                bot.client.send_input(bot.pending_input)
                bot.pending_input = None
        
        return packets
    
    def step(self, inputs: Optional[Dict[int, PlayerInput]] = None) -> GameState:
        """
        Execute one environment step.
        
        Waits for `ticks_per_step` game ticks, sending inputs each tick.
        
        Args:
            inputs: Optional inputs to set before stepping
            
        Returns:
            Current game state after the step
        """
        if inputs:
            self.set_inputs(inputs)
        
        # Clear kill events for this step
        for bot in self.bots.values():
            bot.kills.clear()
            bot.deaths.clear()
        self.game_state.kill_events.clear()
        
        # Wait for ticks_per_step ticks
        target_tick = self.game_state.tick + self.ticks_per_step
        step_start = time.monotonic()
        timeout = (self.ticks_per_step / SERVER_TICK_SPEED) * 2  # 2x expected time
        
        while self.game_state.tick < target_tick:
            self.pump()
            
            # Send inputs continuously
            for bot in self.bots.values():
                if bot.pending_input and bot.client.is_connected:
                    bot.client.send_input(bot.pending_input)
            
            # Check timeout
            if time.monotonic() - step_start > timeout:
                logger.warning(f"Step timeout at tick {self.game_state.tick}, target was {target_tick}")
                break
            
            time.sleep(1 / SERVER_TICK_SPEED / 2)  # Sleep half a tick
        
        self.step_count += 1
        self.last_step_tick = self.game_state.tick
        
        return self.game_state
    
    async def step_async(self, inputs: Optional[Dict[int, PlayerInput]] = None) -> GameState:
        """
        Execute one environment step asynchronously.
        
        Args:
            inputs: Optional inputs to set before stepping
            
        Returns:
            Current game state after the step
        """
        if inputs:
            self.set_inputs(inputs)
        
        # Clear kill events for this step
        for bot in self.bots.values():
            bot.kills.clear()
            bot.deaths.clear()
        self.game_state.kill_events.clear()
        
        # Wait for ticks_per_step ticks
        target_tick = self.game_state.tick + self.ticks_per_step
        step_start = time.monotonic()
        timeout = (self.ticks_per_step / SERVER_TICK_SPEED) * 2
        
        while self.game_state.tick < target_tick:
            self.pump()
            
            # Send inputs continuously
            for bot in self.bots.values():
                if bot.pending_input and bot.client.is_connected:
                    bot.client.send_input(bot.pending_input)
            
            if time.monotonic() - step_start > timeout:
                logger.warning(f"Step timeout at tick {self.game_state.tick}")
                break
            
            await asyncio.sleep(1 / SERVER_TICK_SPEED / 2)
        
        self.step_count += 1
        self.last_step_tick = self.game_state.tick
        
        return self.game_state
    
    def get_bot_state(self, bot_id: int) -> Optional[BotState]:
        """Get state for a specific bot."""
        return self.bots.get(bot_id)
    
    def get_character(self, bot_id: int) -> Optional[Character]:
        """Get character state for a bot."""
        bot = self.bots.get(bot_id)
        if bot:
            return bot.character
        return None
    
    def get_all_characters(self) -> Dict[int, Character]:
        """Get all character states."""
        return self.game_state.characters
    
    def get_score(self, bot_id: int) -> int:
        """Get score for a bot."""
        bot = self.bots.get(bot_id)
        if bot and bot.player_info:
            return bot.player_info.score
        return 0
    
    def get_scores(self) -> Dict[int, int]:
        """Get scores for all bots."""
        return {
            bot_id: self.get_score(bot_id)
            for bot_id in self.bots
        }
    
    def is_alive(self, bot_id: int) -> bool:
        """Check if a bot is alive."""
        bot = self.bots.get(bot_id)
        if bot and bot.player_info:
            return not bot.player_info.is_dead
        return False
    
    def get_alive_bots(self) -> List[int]:
        """Get list of alive bot IDs."""
        return [bot_id for bot_id in self.bots if self.is_alive(bot_id)]
    
    @property
    def all_connected(self) -> bool:
        """Check if all bots are connected."""
        return all(bot.connected for bot in self.bots.values())
    
    @property
    def current_tick(self) -> int:
        """Get current game tick."""
        return self.game_state.tick
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
