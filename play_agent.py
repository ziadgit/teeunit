#!/usr/bin/env python3
"""
Play Trained Q-Learning Agent on Real Teeworlds Server

This script loads a trained Q-learning agent and plays it on a real
Teeworlds server running in Docker. You can spectate by connecting
to the same server with the Teeworlds client.

Usage:
    python play_agent.py [--host HOST] [--port PORT] [--model MODEL_PATH]

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
from typing import Dict, Any, Optional, Tuple

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
    parser = argparse.ArgumentParser(description="Play trained Q-learning agent on Teeworlds server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8303, help="Server port (default: 8303)")
    parser.add_argument("--model", default="/tmp/teeunit-agent/teeunit_qlearning_agent.json",
                        help="Path to trained model JSON")
    parser.add_argument("--episodes", type=int, default=0, help="Number of episodes (0 = infinite)")
    parser.add_argument("--steps-per-episode", type=int, default=200, help="Steps per episode")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Exploration rate")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check model exists
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        logger.info("Make sure to clone the model repo first:")
        logger.info("  git clone https://huggingface.co/ziadbc/teeunit-agent /tmp/teeunit-agent")
        sys.exit(1)
    
    # Load agent
    agent = QLearningAgent(args.model)
    
    # Create bot manager
    logger.info(f"Connecting to Teeworlds server at {args.host}:{args.port}...")
    manager = BotManager(
        host=args.host,
        port=args.port,
        num_bots=1,
        ticks_per_step=10,  # 200ms per step
        bot_name_prefix="QAgent",
    )
    
    try:
        # Connect
        if not manager.connect(timeout=10.0):
            logger.error("Failed to connect to server!")
            logger.info("Make sure the Docker container is running:")
            logger.info("  docker ps | grep teeunit")
            sys.exit(1)
        
        logger.info("Connected! Agent is now playing.")
        logger.info("")
        logger.info("=" * 60)
        logger.info("To spectate, open Teeworlds client and connect to:")
        logger.info(f"  localhost:{args.port}")
        logger.info("=" * 60)
        logger.info("")
        
        # Game loop
        episode = 0
        total_steps = 0
        current_input = PlayerInput()
        fire_count = 0
        
        while True:
            episode += 1
            episode_reward = 0.0
            episode_steps = 0
            
            logger.info(f"Episode {episode} starting...")
            
            for step in range(args.steps_per_episode):
                # Get current state
                game_state = manager.game_state
                
                # Discretize state
                state = agent.discretize_state(game_state, bot_id=0)
                
                # Select action
                action = agent.select_action(state, epsilon=args.epsilon)
                
                # Convert to input
                current_input, fire_count = action_to_input(action, current_input, fire_count)
                
                # Execute step
                inputs = {0: current_input}
                new_state = manager.step(inputs)
                
                # Log periodically
                if step % 50 == 0:
                    char = new_state.get_character(0)
                    if char:
                        logger.info(f"  Step {step}: pos=({char.x}, {char.y}), hp={char.health}, action={action['tool']}")
                
                episode_steps += 1
                total_steps += 1
            
            logger.info(f"Episode {episode} complete: {episode_steps} steps")
            
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
