#!/usr/bin/env python3
"""
Play LLM Agent on Real Teeworlds Server

This script uses a Large Language Model (via HuggingFace Inference API)
to make real-time decisions in a Teeworlds game. The LLM receives game
state as text and selects actions using natural language reasoning.

Usage:
    python play_llm_agent.py [--num-bots 2] [--model MODEL_NAME]

Requirements:
    - Docker container 'teeunit' running on localhost:8303
    - HuggingFace token set in HF_TOKEN env var or --hf-token flag
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    import requests

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
    """Agent that uses HuggingFace Inference API for decision making."""
    
    def __init__(self, hf_token: str, model_id: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.hf_token = hf_token
        self.model_id = model_id
        
        # Use huggingface_hub InferenceClient (new recommended approach)
        if HF_HUB_AVAILABLE:
            # Use "scaleway" provider which works with HF token routing
            self.client = InferenceClient(
                provider="scaleway",
                api_key=hf_token,
            )
            logger.info(f"Using InferenceClient with scaleway provider")
        else:
            self.client = None
            logger.warning("huggingface_hub not available, LLM calls will use fallback")
        
        # Build action list for prompt
        self.action_list = "\n".join([f"- {a['name']}: {a['description']}" for a in ACTIONS])
        
        logger.info(f"LLM Agent using model: {model_id}")
    
    def _format_game_state(self, game_state: GameState, bot_id: int) -> str:
        """Convert game state to natural language description."""
        character = game_state.get_character(bot_id)
        
        if not character:
            return "You are dead and waiting to respawn."
        
        # My state
        my_state = f"Your position: ({character.x}, {character.y}), Health: {character.health}/10"
        
        # Find enemies
        enemies = []
        for other_id, other_char in game_state.characters.items():
            if other_id != bot_id:
                dx = other_char.x - character.x
                dy = other_char.y - character.y
                dist = int((dx**2 + dy**2) ** 0.5)
                
                # Relative direction
                if dx > 100:
                    h_dir = "to your right"
                elif dx < -100:
                    h_dir = "to your left"
                else:
                    h_dir = "directly ahead"
                
                if dist < 200:
                    range_desc = "very close (hammer range)"
                elif dist < 500:
                    range_desc = "close (gun range)"
                else:
                    range_desc = "far away"
                
                enemies.append(f"Enemy {other_id}: {range_desc}, {h_dir}, health={other_char.health}")
        
        if enemies:
            enemy_state = "Enemies:\n" + "\n".join(enemies)
        else:
            enemy_state = "No enemies visible."
        
        return f"{my_state}\n{enemy_state}"
    
    def _build_prompt(self, game_state_text: str) -> str:
        """Build the prompt for the LLM."""
        return f"""You are an AI agent playing a 2D combat game called Teeworlds. You must choose ONE action to take.

GAME STATE:
{game_state_text}

AVAILABLE ACTIONS:
{self.action_list}

STRATEGY TIPS:
- Use hammer when enemies are very close
- Use gun for medium/long range
- Jump to dodge attacks
- Move toward enemies to engage
- Hook can pull you toward walls or enemies

Choose the BEST action. Reply with ONLY the action name, nothing else.

ACTION:"""

    def select_action(self, game_state: GameState, bot_id: int) -> Dict[str, Any]:
        """Query LLM to select an action."""
        # Format game state as text
        state_text = self._format_game_state(game_state, bot_id)
        prompt = self._build_prompt(state_text)
        
        if self.client is None:
            # Fallback if no client
            import random
            return random.choice(ACTIONS)
        
        try:
            # Use InferenceClient chat_completion
            response = self.client.chat_completion(
                model=self.model_id,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=20,
                temperature=0.3,
            )
            
            if response and hasattr(response, 'choices') and len(response.choices) > 0:
                generated_text = response.choices[0].message.content.strip().lower()
                
                # Parse the action from response
                for action in ACTIONS:
                    if action["name"].lower() in generated_text:
                        logger.debug(f"LLM chose: {action['name']} (raw: {generated_text[:50]})")
                        return action
                
                # If no exact match, try fuzzy matching
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
                
                logger.debug(f"LLM response not recognized: {generated_text[:50]}")
        
        except Exception as e:
            error_msg = str(e)
            if "loading" in error_msg.lower() or "503" in error_msg:
                logger.warning("Model is loading, using fallback...")
            else:
                logger.warning(f"LLM error: {e}")
        
        # Fallback: random action biased toward combat
        import random
        return random.choice(ACTIONS)


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
    parser = argparse.ArgumentParser(description="Play LLM agent on Teeworlds server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8303, help="Server port")
    parser.add_argument("--num-bots", type=int, default=2, help="Number of LLM bots (default: 2)")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--episodes", type=int, default=0, help="Number of episodes (0 = infinite)")
    parser.add_argument("--steps-per-episode", type=int, default=100, help="Steps per episode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HuggingFace token required. Set HF_TOKEN env var or use --hf-token")
        sys.exit(1)
    
    num_bots = max(1, min(4, args.num_bots))  # Limit to 4 for LLM (API rate limits)
    
    # Create LLM agent (shared between all bots)
    agent = LLMAgent(hf_token, model_id=args.model)
    
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
                    # LLM selects action
                    action = agent.select_action(game_state, bot_id)
                    
                    # Log what LLM decided
                    if step % 10 == 0:
                        char = game_state.get_character(bot_id)
                        hp = char.health if char else 0
                        logger.info(f"  Bot{bot_id} (hp={hp}): LLM chose '{action['name']}'")
                    
                    new_input, fire_counts[bot_id] = action_to_input(
                        action, current_inputs[bot_id], fire_counts[bot_id]
                    )
                    current_inputs[bot_id] = new_input
                    inputs[bot_id] = new_input
                
                manager.step(inputs)
                total_steps += 1
                
                # Small delay to avoid API rate limits
                time.sleep(0.1)
            
            logger.info(f"Episode {episode} complete")
            
            if args.episodes > 0 and episode >= args.episodes:
                break
            
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    finally:
        manager.disconnect()
        logger.info(f"Total: {episode} episodes, {total_steps} steps")


if __name__ == "__main__":
    main()
