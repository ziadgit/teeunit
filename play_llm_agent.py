#!/usr/bin/env python3
"""
Play LLM Agent on Real Teeworlds Server

This script uses a Large Language Model to make real-time decisions in a 
Teeworlds game. The LLM receives game state as text and selects actions 
using natural language reasoning.

Supports two providers:
  - ollama: Local LLM via Ollama (free, unlimited, recommended)
  - huggingface: Cloud LLM via HuggingFace Inference API (requires credits)

Usage:
    # With Ollama (recommended - free and unlimited)
    ollama serve  # Start Ollama in another terminal
    ollama pull llama3.2:1b  # Pull a fast model
    python play_llm_agent.py --provider ollama --model llama3.2:1b

    # With HuggingFace (requires credits)
    python play_llm_agent.py --provider huggingface --hf-token YOUR_TOKEN

Requirements:
    - Docker container 'teeunit' running on localhost:8303
    - For Ollama: ollama installed and running (brew install ollama)
    - For HuggingFace: HF token with inference credits
"""

import argparse
import json
import logging
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import requests

try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from teeunit.server.bot_manager import BotManager, GameState
from teeunit.protocol.objects import PlayerInput

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Available actions the LLM can choose from
ACTIONS = [
    {"name": "move_left", "tool": "move", "args": {"direction": "left"}, "description": "Move left"},
    {"name": "move_right", "tool": "move", "args": {"direction": "right"}, "description": "Move right"},
    {"name": "stop", "tool": "move", "args": {"direction": "none"}, "description": "Stop moving"},
    {"name": "jump", "tool": "jump", "args": {}, "description": "Jump"},
    {"name": "hammer", "tool": "shoot", "args": {"weapon": 1}, "description": "Attack with hammer (close range)"},
    {"name": "shoot_gun", "tool": "shoot", "args": {"weapon": 2}, "description": "Shoot gun (long range)"},
    {"name": "hook", "tool": "hook", "args": {}, "description": "Use grappling hook"},
]

ACTION_NAMES = [a["name"] for a in ACTIONS]


class LLMAgent:
    """Agent that uses an LLM for decision making. Supports Ollama and HuggingFace."""
    
    def __init__(
        self, 
        provider: str = "ollama",
        model_id: str = "llama3.2:1b",
        ollama_url: str = "http://localhost:11434",
        hf_token: Optional[str] = None,
    ):
        self.provider = provider
        self.model_id = model_id
        self.ollama_url = ollama_url
        self.hf_token = hf_token
        self.client = None
        
        # Thread pool for non-blocking LLM calls
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.pending_futures: Dict[int, Any] = {}  # bot_id -> future
        self.last_actions: Dict[int, Dict] = {}  # bot_id -> last action (for repeat)
        
        if provider == "ollama":
            # Ollama uses direct HTTP calls (OpenAI-compatible API)
            self.api_url = f"{ollama_url}/api/chat"
            logger.info(f"Using Ollama at {ollama_url}")
        elif provider == "huggingface":
            if not HF_HUB_AVAILABLE:
                raise RuntimeError("huggingface_hub not installed. Run: pip install huggingface_hub")
            if not hf_token:
                raise RuntimeError("HuggingFace token required for huggingface provider")
            # Use scaleway provider which works with HF token routing
            self.client = InferenceClient(
                provider="scaleway",
                api_key=hf_token,
            )
            logger.info(f"Using HuggingFace with scaleway provider")
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'ollama' or 'huggingface'")
        
        # Build action list for prompt
        self.action_list = "\n".join([f"- {a['name']}: {a['description']}" for a in ACTIONS])
        
        logger.info(f"LLM Agent using model: {model_id}")
    
    def start_action_request(self, game_state: GameState, bot_id: int):
        """Start async LLM request for action selection."""
        # Don't start new request if one is pending
        if bot_id in self.pending_futures and not self.pending_futures[bot_id].done():
            return
        
        # Format game state as text
        state_text = self._format_game_state(game_state, bot_id)
        prompt = self._build_prompt(state_text)
        
        # Submit to thread pool
        if self.provider == "ollama":
            future = self.executor.submit(self._call_ollama, prompt)
        elif self.provider == "huggingface":
            future = self.executor.submit(self._call_huggingface, prompt)
        else:
            return
        
        self.pending_futures[bot_id] = future
    
    def get_action(self, bot_id: int) -> Dict[str, Any]:
        """
        Get action for bot. Returns immediately with best available action.
        If LLM response is ready, use it. Otherwise use last action or random.
        """
        import random
        
        # Check if we have a completed future
        if bot_id in self.pending_futures:
            future = self.pending_futures[bot_id]
            if future.done():
                try:
                    generated_text = future.result()
                    if generated_text:
                        action = self._parse_action(generated_text)
                        if action:
                            self.last_actions[bot_id] = action
                            return action
                except Exception as e:
                    logger.debug(f"LLM error for bot {bot_id}: {e}")
                finally:
                    del self.pending_futures[bot_id]
        
        # No result yet - use last action or random
        if bot_id in self.last_actions:
            return self.last_actions[bot_id]
        
        # First time - pick random action
        return random.choice(ACTIONS)
    
    def _parse_action(self, generated_text: str) -> Optional[Dict[str, Any]]:
        """Parse action from LLM response."""
        generated_text = generated_text.strip().lower()
        
        # Exact match first
        for action in ACTIONS:
            if action["name"].lower() in generated_text:
                return action
        
        # Fuzzy matching
        if "left" in generated_text:
            return ACTIONS[0]  # move_left
        elif "right" in generated_text:
            return ACTIONS[1]  # move_right
        elif "jump" in generated_text:
            return ACTIONS[3]  # jump
        elif "hammer" in generated_text:
            return ACTIONS[4]  # hammer
        elif "shoot" in generated_text or "gun" in generated_text:
            return ACTIONS[5]  # shoot_gun
        elif "hook" in generated_text:
            return ACTIONS[6]  # hook
        
        return None
    
    def _format_game_state(self, game_state: GameState, bot_id: int) -> str:
        """Convert game state to ULTRA SHORT description for fast LLM response."""
        character = game_state.get_character(bot_id)
        
        if not character:
            return "Dead. Respawning."
        
        # Find nearest enemy - keep it super short
        nearest = None
        nearest_dist = float('inf')
        nearest_dir = ""
        
        for other_id, other_char in game_state.characters.items():
            if other_id != bot_id:
                dx = other_char.x - character.x
                dy = other_char.y - character.y
                dist = int((dx**2 + dy**2) ** 0.5)
                
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = other_char
                    nearest_dir = "right" if dx > 0 else "left"
        
        # Ultra compact format
        if nearest:
            if nearest_dist < 200:
                range_word = "close"
            elif nearest_dist < 500:
                range_word = "medium"
            else:
                range_word = "far"
            return f"HP:{character.health} Enemy:{range_word},{nearest_dir}"
        else:
            return f"HP:{character.health} No enemy"
    
    def _build_prompt(self, game_state_text: str) -> str:
        """Build a SHORT prompt for fast LLM response."""
        # Ultra-short prompt for speed (< 100 tokens)
        return f"""Game: {game_state_text}
Actions: move_left, move_right, jump, hammer, shoot_gun, hook
Reply with ONE action name only:"""

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Call Ollama API with short timeout."""
        response = requests.post(
            self.api_url,
            json={
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 10,  # Very short response
                }
            },
            timeout=8.0,  # Allow time for queued requests
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("message", {}).get("content", "")
        else:
            logger.warning(f"Ollama error {response.status_code}: {response.text[:100]}")
            return None
    
    def _call_huggingface(self, prompt: str) -> Optional[str]:
        """Call HuggingFace Inference API."""
        if self.client is None:
            return None
        
        response = self.client.chat_completion(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.3,
        )
        
        if response and hasattr(response, 'choices') and len(response.choices) > 0:
            return response.choices[0].message.content
        return None


def action_to_input(action: Dict[str, Any], current_input: PlayerInput, fire_count: int) -> Tuple[PlayerInput, int]:
    """Convert action dict to PlayerInput."""
    tool = action["tool"]
    args = action.get("args", {})
    
    inp = PlayerInput(
        direction=current_input.direction,
        target_x=current_input.target_x,
        target_y=current_input.target_y,
        jump=False,
        fire=current_input.fire,
        hook=False,
        wanted_weapon=current_input.wanted_weapon,
    )
    
    if tool == "move":
        direction = args.get("direction", "none")
        if direction == "left":
            inp.direction = -1
        elif direction == "right":
            inp.direction = 1
        else:
            inp.direction = 0
    elif tool == "jump":
        inp.jump = True
    elif tool == "shoot":
        weapon = args.get("weapon", 1)
        fire_count += 1
        inp.fire = fire_count
        inp.wanted_weapon = weapon
    elif tool == "hook":
        inp.hook = True
    
    return inp, fire_count


def main():
    parser = argparse.ArgumentParser(
        description="Play LLM agent on Teeworlds server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using Ollama (recommended - free and unlimited)
  ollama serve                    # Start Ollama in another terminal
  ollama pull llama3.2:1b         # Pull a fast model  
  python play_llm_agent.py --provider ollama --model llama3.2:1b

  # Using HuggingFace (requires credits)
  python play_llm_agent.py --provider huggingface --hf-token YOUR_TOKEN
        """
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8303, help="Server port")
    parser.add_argument("--num-bots", type=int, default=2, help="Number of LLM bots (default: 2)")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "huggingface"],
                        help="LLM provider: 'ollama' (local, free) or 'huggingface' (cloud, requires credits)")
    parser.add_argument("--model", default=None,
                        help="Model ID (default: llama3.2:1b for ollama, meta-llama/Llama-3.1-8B-Instruct for huggingface)")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama server URL (default: http://localhost:11434)")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--episodes", type=int, default=0, help="Number of episodes (0 = infinite)")
    parser.add_argument("--steps-per-episode", type=int, default=100, help="Steps per episode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set default model based on provider
    if args.model is None:
        if args.provider == "ollama":
            args.model = "llama3.2:1b"
        else:
            args.model = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Validate provider requirements
    if args.provider == "huggingface":
        hf_token = args.hf_token or os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.error("HuggingFace token required. Set HF_TOKEN env var or use --hf-token")
            sys.exit(1)
    else:
        hf_token = None
    
    num_bots = max(1, min(3, args.num_bots))  # Limit to 3 (server max players)
    
    # Create LLM agent (shared between all bots)
    agent = LLMAgent(
        provider=args.provider,
        model_id=args.model,
        ollama_url=args.ollama_url,
        hf_token=hf_token,
    )
    
    # Create bot manager
    logger.info(f"Connecting {num_bots} LLM bots to {args.host}:{args.port}...")
    manager = BotManager(
        host=args.host,
        port=args.port,
        num_bots=num_bots,
        ticks_per_step=15,  # 300ms per step (more time for LLM)
        bot_name_prefix="Claude",  # Fun name for demo
    )
    
    try:
        if not manager.connect(timeout=15.0):
            logger.error("Failed to connect!")
            logger.info("Make sure Docker container is running: docker ps | grep teeunit")
            sys.exit(1)
        
        logger.info(f"Connected! {num_bots} LLM agents are playing.")
        logger.info("")
        logger.info("=" * 60)
        logger.info("LLM is making real-time decisions based on game state!")
        logger.info(f"Provider: {args.provider}")
        logger.info(f"Model: {args.model}")
        logger.info(f"To spectate: connect to localhost:{args.port}")
        logger.info("=" * 60)
        logger.info("")
        
        # Per-bot state
        current_inputs: Dict[int, PlayerInput] = {i: PlayerInput() for i in range(num_bots)}
        fire_counts: Dict[int, int] = {i: 0 for i in range(num_bots)}
        
        episode = 0
        total_steps = 0
        
        while True:
            episode += 1
            logger.info(f"Episode {episode} starting...")
            
            for step in range(args.steps_per_episode):
                game_state = manager.game_state
                inputs: Dict[int, PlayerInput] = {}
                
                for bot_id in range(num_bots):
                    # Stagger LLM requests: only 1 bot queries per step (round-robin)
                    # This prevents Ollama from getting overwhelmed with concurrent requests
                    if step % num_bots == bot_id:
                        agent.start_action_request(game_state, bot_id)
                    
                    # Get action immediately (uses result if ready, else fallback)
                    action = agent.get_action(bot_id)
                    
                    # Log what action was taken
                    if step % 5 == 0:
                        char = game_state.get_character(bot_id)
                        hp = char.health if char else 0
                        logger.info(f"  Bot{bot_id} (hp={hp}): '{action['name']}'")
                    
                    new_input, fire_counts[bot_id] = action_to_input(
                        action, current_inputs[bot_id], fire_counts[bot_id]
                    )
                    current_inputs[bot_id] = new_input
                    inputs[bot_id] = new_input
                
                # Step the game - this keeps connection alive
                manager.step(inputs)
                total_steps += 1
            
            logger.info(f"Episode {episode} complete ({args.steps_per_episode} steps)")
            
            if args.episodes > 0 and episode >= args.episodes:
                break
            
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    finally:
        manager.disconnect()
        logger.info(f"Total: {episode} episodes, {total_steps} steps")


if __name__ == "__main__":
    main()
