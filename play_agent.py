#!/usr/bin/env python3
"""
Play Trained Q-Learning Agents on Real Teeworlds Server

This script loads a trained Q-learning agent and runs multiple bots
on a real Teeworlds server running in Docker. You can spectate by
connecting to the same server with the Teeworlds client.

Usage:
    python play_agent.py [--num-bots 4] [--host HOST] [--port PORT]

Requirements:
    - Docker container 'teeunit' running on localhost:8303
    - Trained model at /tmp/teeunit-agent/teeunit_qlearning_agent.json
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

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


class QLearningAgent:
    """Q-Learning agent that loads from trained model."""
    
    def __init__(self, model_path: str):
        """Load trained model from JSON file."""
        with open(model_path, "r") as f:
            data = json.load(f)
        
        self.actions = data["actions"]
        self.q_table = data["q_table"]
        self.metadata = data.get("metadata", {})
        
        logger.info(f"Loaded Q-learning model: {len(self.q_table)} states, {len(self.actions)} actions")
        logger.info(f"Trained for {data.get('episodes', 0)} episodes, {data.get('steps', 0)} steps")
        logger.info(f"Total reward: {data.get('total_reward', 0):.2f}")
    
    def discretize_state(self, game_state: GameState, bot_id: int) -> str:
        """Convert game state to discretized state string (same as training)."""
        character = game_state.get_character(bot_id)
        player_info = game_state.get_player_info(bot_id)
        
        if not character:
            return "('unknown',)"
        
        # Get bot position (discretized to grid)
        grid_x = character.x // 1000  # Rough grid
        grid_y = character.y // 1000
        
        # Get health level
        if player_info and player_info.is_dead:
            hp_level = "hp_0"
        elif character.health >= 8:
            hp_level = "hp_3"
        elif character.health >= 4:
            hp_level = "hp_2"
        elif character.health >= 1:
            hp_level = "hp_1"
        else:
            hp_level = "hp_0"
        
        # Find distances to other characters (enemies)
        enemy_distances = []
        for other_id, other_char in game_state.characters.items():
            if other_id != bot_id:
                dx = other_char.x - character.x
                dy = other_char.y - character.y
                dist = (dx**2 + dy**2) ** 0.5
                # Discretize distance
                if dist < 300:
                    enemy_distances.append("enemy_close")
                elif dist < 800:
                    enemy_distances.append("enemy_mid")
                else:
                    enemy_distances.append("enemy_far")
        
        # Sort for consistency
        enemy_distances.sort()
        
        # Build state tuple string (matches training format)
        state_parts = enemy_distances + [hp_level, f"pos_{grid_x}_{grid_y}"]
        state_str = str(tuple(state_parts))
        
        return state_str
    
    def select_action(self, state: str, epsilon: float = 0.1) -> Dict[str, Any]:
        """Select action using epsilon-greedy policy."""
        # Epsilon-greedy exploration (lower epsilon since we're showing off)
        if random.random() < epsilon:
            return random.choice(self.actions)
        
        # Get Q-values for this state
        if state in self.q_table:
            q_values = self.q_table[state]
            # Find best action
            best_action_idx = max(q_values.keys(), key=lambda k: q_values[k])
            return self.actions[int(best_action_idx)]
        
        # Unknown state - random action
        return random.choice(self.actions)


def action_to_input(action: Dict[str, Any], current_input: PlayerInput, fire_count: int) -> Tuple[PlayerInput, int]:
    """Convert action dict to PlayerInput."""
    tool = action["tool"]
    args = action.get("args", {})
    
    # Start with current input state
    inp = PlayerInput(
        direction=current_input.direction,
        target_x=current_input.target_x,
        target_y=current_input.target_y,
        jump=False,  # Reset per-step actions
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
    
    elif tool == "aim":
        inp.target_x = args.get("x", 0)
        inp.target_y = args.get("y", 0)
    
    return inp, fire_count


def main():
    parser = argparse.ArgumentParser(description="Play trained Q-learning agents on Teeworlds server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8303, help="Server port (default: 8303)")
    parser.add_argument("--model", default="/tmp/teeunit-agent/teeunit_qlearning_agent.json",
                        help="Path to trained model JSON")
    parser.add_argument("--num-bots", type=int, default=4, help="Number of bots (1-8, default: 4)")
    parser.add_argument("--episodes", type=int, default=0, help="Number of episodes (0 = infinite)")
    parser.add_argument("--steps-per-episode", type=int, default=200, help="Steps per episode")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Base exploration rate")
    parser.add_argument("--vary-epsilon", action="store_true", 
                        help="Give each bot different epsilon (more diverse behavior)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate num_bots
    num_bots = max(1, min(8, args.num_bots))
    
    # Check model exists
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        logger.info("Make sure to clone the model repo first:")
        logger.info("  git clone https://huggingface.co/ziadbc/teeunit-agent /tmp/teeunit-agent")
        sys.exit(1)
    
    # Load agent (shared Q-table for all bots)
    agent = QLearningAgent(args.model)
    
    # Calculate epsilon for each bot
    if args.vary_epsilon:
        # Give each bot different exploration rates for diverse behavior
        # Bot 0: most greedy, Bot N: most exploratory
        epsilons = [args.epsilon + (i * 0.1) for i in range(num_bots)]
        logger.info(f"Varying epsilon per bot: {[f'{e:.2f}' for e in epsilons]}")
    else:
        epsilons = [args.epsilon] * num_bots
    
    # Create bot manager
    logger.info(f"Connecting {num_bots} bots to Teeworlds server at {args.host}:{args.port}...")
    manager = BotManager(
        host=args.host,
        port=args.port,
        num_bots=num_bots,
        ticks_per_step=10,  # 200ms per step
        bot_name_prefix="QBot",
    )
    
    try:
        # Connect
        if not manager.connect(timeout=15.0):
            logger.error("Failed to connect all bots to server!")
            logger.info("Make sure the Docker container is running:")
            logger.info("  docker ps | grep teeunit")
            sys.exit(1)
        
        logger.info(f"Connected! {num_bots} agents are now playing.")
        logger.info("")
        logger.info("=" * 60)
        logger.info("To spectate, open Teeworlds client and connect to:")
        logger.info(f"  localhost:{args.port}")
        logger.info("=" * 60)
        logger.info("")
        
        # Per-bot state tracking
        current_inputs: Dict[int, PlayerInput] = {i: PlayerInput() for i in range(num_bots)}
        fire_counts: Dict[int, int] = {i: 0 for i in range(num_bots)}
        
        # Game loop
        episode = 0
        total_steps = 0
        
        while True:
            episode += 1
            episode_steps = 0
            kills_this_episode: Dict[int, int] = {i: 0 for i in range(num_bots)}
            
            logger.info(f"Episode {episode} starting...")
            
            for step in range(args.steps_per_episode):
                # Get current game state
                game_state = manager.game_state
                
                # Each bot selects its action independently
                inputs: Dict[int, PlayerInput] = {}
                actions: Dict[int, Dict[str, Any]] = {}
                
                for bot_id in range(num_bots):
                    # Discretize state from this bot's perspective
                    state = agent.discretize_state(game_state, bot_id=bot_id)
                    
                    # Select action with this bot's epsilon
                    action = agent.select_action(state, epsilon=epsilons[bot_id])
                    actions[bot_id] = action
                    
                    # Convert to input
                    new_input, fire_counts[bot_id] = action_to_input(
                        action, current_inputs[bot_id], fire_counts[bot_id]
                    )
                    current_inputs[bot_id] = new_input
                    inputs[bot_id] = new_input
                
                # Execute step for all bots
                new_state = manager.step(inputs)
                
                # Track kills
                for event in new_state.kill_events:
                    if event.killer_id in kills_this_episode:
                        kills_this_episode[event.killer_id] += 1
                
                # Log periodically
                if step % 50 == 0:
                    status_parts = []
                    for bot_id in range(num_bots):
                        char = new_state.get_character(bot_id)
                        if char:
                            status_parts.append(f"Bot{bot_id}:hp={char.health}")
                    logger.info(f"  Step {step}: {', '.join(status_parts)}")
                
                episode_steps += 1
                total_steps += 1
            
            # Episode summary
            kills_summary = ", ".join([f"Bot{i}:{k}" for i, k in kills_this_episode.items()])
            logger.info(f"Episode {episode} complete: {episode_steps} steps, kills: {kills_summary}")
            
            # Check if we should stop
            if args.episodes > 0 and episode >= args.episodes:
                break
            
            # Small pause between episodes
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    
    finally:
        manager.disconnect()
        logger.info(f"Total: {episode} episodes, {total_steps} steps")


if __name__ == "__main__":
    main()
