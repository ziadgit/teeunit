"""
TeeUnit Self-Play RL Training Script

Trains agents using Stable-Baselines3 PPO with self-play.
All 4 agents share the same policy network, enabling emergent behavior
through competitive self-play.

Usage:
    # Local training (requires Teeworlds server running)
    python -m teeunit.train --steps 1000000

    # Remote training (connect to TeeUnit server)
    python -m teeunit.train --remote http://localhost:8000 --steps 1000000

    # With GPU
    python -m teeunit.train --device cuda --steps 1000000
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Check for optional RL dependencies
try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    gym = None
    spaces = None
    PPO = None
    BaseCallback = object
    CheckpointCallback = None
    EvalCallback = None
    DummyVecEnv = None

from .openenv_models import (
    TeeAction,
    TeeMultiAction,
    TeeObservation,
    RewardConfig,
)
from .openenv_environment import TeeEnvironment, TeeConfig
from .openenv_client import TeeEnvClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _check_sb3():
    """Check if SB3 is available and raise helpful error if not."""
    if not HAS_SB3:
        raise ImportError(
            "Stable-Baselines3 and Gymnasium required for training. "
            "Install with: pip install teeunit[rl]"
        )


# =============================================================================
# Gymnasium Wrapper for SB3
# =============================================================================

if HAS_SB3:
    
    class TeeUnitGymEnv(gym.Env):
        """
        Gymnasium wrapper for TeeUnit environment.
        
        Designed for self-play training where all agents share the same policy.
        Each step executes actions for all agents and returns aggregated observations.
        
        The observation is a stack of all agents' observations.
        The action is decoded into per-agent discrete actions.
        """
        
        metadata = {"render_modes": ["human"]}
        
        def __init__(
            self,
            env: Optional[TeeEnvironment] = None,
            client: Optional[TeeEnvClient] = None,
            num_agents: int = 4,
            single_agent_mode: bool = False,
            agent_id: int = 0,
        ):
            """
            Initialize the Gymnasium wrapper.
            
            Args:
                env: Local TeeEnvironment (if running locally)
                client: Remote TeeEnvClient (if connecting to server)
                num_agents: Number of agents in the environment
                single_agent_mode: If True, only control one agent (others use random actions)
                agent_id: Which agent to control in single-agent mode
            """
            super().__init__()
            
            self.env = env
            self.client = client
            self.num_agents = num_agents
            self.single_agent_mode = single_agent_mode
            self.agent_id = agent_id
            
            # Observation space: stacked observations for all agents
            obs_dim = 195  # Per-agent observation dimension
            if single_agent_mode:
                self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(obs_dim,),
                    dtype=np.float32,
                )
            else:
                self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(num_agents, obs_dim),
                    dtype=np.float32,
                )
            
            # Action space: discrete actions for all agents
            if single_agent_mode:
                self.action_space = spaces.Discrete(18)
            else:
                # Multi-agent: product space or multi-discrete
                self.action_space = spaces.MultiDiscrete([18] * num_agents)
            
            # Episode tracking
            self._episode_reward = 0.0
            self._episode_length = 0
        
        def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None,
        ) -> Tuple[np.ndarray, Dict[str, Any]]:
            """Reset the environment."""
            super().reset(seed=seed)
            
            if self.env:
                multi_obs = self.env.reset(seed=seed)
            elif self.client:
                multi_obs = self.client.reset()
            else:
                raise RuntimeError("No environment or client configured")
            
            self._episode_reward = 0.0
            self._episode_length = 0
            
            # Convert to numpy array
            obs = self._observations_to_numpy(multi_obs.observations)
            
            info = {
                "episode_id": multi_obs.metadata.get("episode_id", ""),
                "tick": multi_obs.metadata.get("tick", 0),
            }
            
            return obs, info
        
        def step(
            self,
            action: np.ndarray,
        ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
            """
            Execute actions and return results.
            
            Args:
                action: Discrete action(s) - single int or array for multi-agent
            
            Returns:
                observation, reward, terminated, truncated, info
            """
            # Convert action to TeeMultiAction
            multi_action = self._action_to_multi_action(action)
            
            # Execute step
            if self.env:
                results = self.env.step_all(multi_action)
            elif self.client:
                results = self.client.step_all(multi_action)
            else:
                raise RuntimeError("No environment or client configured")
            
            # Extract observations
            obs = self._observations_to_numpy({
                agent_id: r.observation
                for agent_id, r in results.results.items()
            })
            
            # Aggregate rewards
            if self.single_agent_mode:
                reward = results.results[self.agent_id].reward
            else:
                # Sum rewards across all agents (cooperative) or mean
                reward = sum(r.reward for r in results.results.values()) / self.num_agents
            
            # Check termination
            terminated = any(r.done for r in results.results.values())
            truncated = any(r.truncated for r in results.results.values())
            
            self._episode_reward += reward
            self._episode_length += 1
            
            info = {
                "tick": results.state.tick,
                "step": results.state.step_count,
                "scores": results.state.scores,
                "agents_alive": results.state.agents_alive,
                "winner": results.state.winner,
                "episode_reward": self._episode_reward,
                "episode_length": self._episode_length,
            }
            
            return obs, reward, terminated, truncated, info
        
        def _observations_to_numpy(
            self,
            observations: Dict[int, TeeObservation],
        ) -> np.ndarray:
            """Convert observations dict to numpy array."""
            if self.single_agent_mode:
                obs = observations.get(self.agent_id)
                if obs is None:
                    return np.zeros(195, dtype=np.float32)
                return obs.to_tensor()
            else:
                # Stack all agent observations
                obs_list = []
                for agent_id in range(self.num_agents):
                    obs = observations.get(agent_id)
                    if obs is None:
                        obs_list.append(np.zeros(195, dtype=np.float32))
                    else:
                        obs_list.append(obs.to_tensor())
                return np.stack(obs_list, axis=0)
        
        def _action_to_multi_action(self, action: np.ndarray) -> TeeMultiAction:
            """Convert numpy action to TeeMultiAction."""
            agent_actions = {}
            
            if self.single_agent_mode:
                # Single agent mode - only control one agent
                agent_actions[self.agent_id] = TeeAction.from_discrete_action(
                    int(action),
                    agent_id=self.agent_id,
                )
                # Other agents get random actions
                for i in range(self.num_agents):
                    if i != self.agent_id:
                        random_action = np.random.randint(0, 18)
                        agent_actions[i] = TeeAction.from_discrete_action(
                            random_action,
                            agent_id=i,
                        )
            else:
                # Multi-agent mode - action is array of discrete actions
                for agent_id in range(self.num_agents):
                    agent_actions[agent_id] = TeeAction.from_discrete_action(
                        int(action[agent_id]),
                        agent_id=agent_id,
                    )
            
            return TeeMultiAction(actions=agent_actions)
        
        def render(self) -> None:
            """Render is handled by Teeworlds server."""
            pass
        
        def close(self) -> None:
            """Cleanup resources."""
            if self.env:
                self.env.close()
            if self.client:
                self.client.close()


    class SelfPlayCallback(BaseCallback):
        """
        Callback for self-play training.
        
        Logs self-play metrics and optionally saves opponent snapshots.
        """
        
        def __init__(
            self,
            log_interval: int = 1000,
            save_opponent_interval: int = 10000,
            opponent_dir: str = "opponents",
            verbose: int = 1,
        ):
            super().__init__(verbose)
            self.log_interval = log_interval
            self.save_opponent_interval = save_opponent_interval
            self.opponent_dir = opponent_dir
            self.episode_rewards: List[float] = []
            self.episode_lengths: List[int] = []
            self.kill_counts: List[int] = []
        
        def _on_step(self) -> bool:
            # Track episode completions
            for info in self.locals.get("infos", []):
                if "episode_reward" in info and info.get("_final_info", False):
                    self.episode_rewards.append(info["episode_reward"])
                    self.episode_lengths.append(info.get("episode_length", 0))
            
            # Log progress
            if self.n_calls % self.log_interval == 0:
                if self.episode_rewards:
                    mean_reward = np.mean(self.episode_rewards[-100:])
                    mean_length = np.mean(self.episode_lengths[-100:])
                    logger.info(
                        f"Steps: {self.n_calls} | "
                        f"Episodes: {len(self.episode_rewards)} | "
                        f"Mean Reward (last 100): {mean_reward:.2f} | "
                        f"Mean Length: {mean_length:.1f}"
                    )
            
            # Save opponent snapshot for league training
            if self.save_opponent_interval > 0 and self.n_calls % self.save_opponent_interval == 0:
                os.makedirs(self.opponent_dir, exist_ok=True)
                path = os.path.join(self.opponent_dir, f"opponent_{self.n_calls}.zip")
                self.model.save(path)
                logger.info(f"Saved opponent snapshot: {path}")
            
            return True

else:
    # Stub classes when SB3 not available
    TeeUnitGymEnv = None
    SelfPlayCallback = None


# =============================================================================
# Environment Factory
# =============================================================================

def make_env(
    host: str = "127.0.0.1",
    port: int = 8303,
    remote_url: Optional[str] = None,
    num_agents: int = 4,
    single_agent_mode: bool = False,
    agent_id: int = 0,
):
    """
    Create a TeeUnit Gymnasium environment.
    
    Args:
        host: Teeworlds server host (for local mode)
        port: Teeworlds server port (for local mode)
        remote_url: URL of TeeUnit server (for remote mode)
        num_agents: Number of agents
        single_agent_mode: Control single agent only
        agent_id: Agent to control in single-agent mode
    
    Returns:
        TeeUnitGymEnv instance
    """
    _check_sb3()
    
    if remote_url:
        # Remote mode - connect to TeeUnit server
        client = TeeEnvClient(remote_url)
        return TeeUnitGymEnv(
            client=client,
            num_agents=num_agents,
            single_agent_mode=single_agent_mode,
            agent_id=agent_id,
        )
    else:
        # Local mode - run environment directly
        config = TeeConfig(
            num_agents=num_agents,
            server_host=host,
            server_port=port,
            ticks_per_step=10,
            max_steps=1000,  # Episode length
            reward_config=RewardConfig(
                kill_reward=10.0,
                death_penalty=-5.0,
                survival_bonus=0.01,
                win_bonus=50.0,
            ),
        )
        env = TeeEnvironment(config=config, auto_connect=False)
        return TeeUnitGymEnv(
            env=env,
            num_agents=num_agents,
            single_agent_mode=single_agent_mode,
            agent_id=agent_id,
        )


# =============================================================================
# Training Function
# =============================================================================

def train(
    total_timesteps: int = 1_000_000,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    n_steps: int = 2048,
    device: str = "auto",
    save_path: str = "models/teeunit_ppo",
    log_dir: str = "logs",
    remote_url: Optional[str] = None,
    single_agent_mode: bool = False,
    num_envs: int = 1,
    **kwargs,
):
    """
    Train a PPO agent on TeeUnit with self-play.
    
    Args:
        total_timesteps: Total training timesteps
        learning_rate: Learning rate
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Max gradient norm
        n_steps: Steps per rollout
        device: Device to use (auto, cpu, cuda)
        save_path: Path to save final model
        log_dir: TensorBoard log directory
        remote_url: URL for remote training
        single_agent_mode: Train single agent
        num_envs: Number of parallel environments
    
    Returns:
        Trained PPO model
    """
    _check_sb3()
    
    logger.info("Creating TeeUnit environment...")
    
    # Create environment(s)
    if num_envs > 1:
        # Parallel environments
        def make_single_env():
            return make_env(
                remote_url=remote_url,
                single_agent_mode=single_agent_mode,
            )
        env = DummyVecEnv([make_single_env for _ in range(num_envs)])
    else:
        env = make_env(
            remote_url=remote_url,
            single_agent_mode=single_agent_mode,
        )
    
    logger.info(f"Observation space: {env.observation_space}")
    logger.info(f"Action space: {env.action_space}")
    logger.info(f"Device: {device}")
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
        device=device,
        tensorboard_log=log_dir,
    )
    
    # Callbacks
    callbacks = [
        SelfPlayCallback(
            log_interval=1000,
            save_opponent_interval=50000,
            opponent_dir=os.path.join(save_path, "opponents"),
        ),
        CheckpointCallback(
            save_freq=10000,
            save_path=save_path,
            name_prefix="teeunit_ppo",
        ),
    ]
    
    logger.info(f"Starting training for {total_timesteps} timesteps...")
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    # Save final model
    final_path = os.path.join(save_path, "final_model")
    model.save(final_path)
    logger.info(f"Training complete! Model saved to {final_path}")
    
    return model


# =============================================================================
# Evaluation Function
# =============================================================================

def evaluate(
    model_path: str,
    num_episodes: int = 10,
    remote_url: Optional[str] = None,
    render: bool = False,
) -> Dict[str, float]:
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to saved model
        num_episodes: Number of evaluation episodes
        remote_url: URL for remote environment
        render: Whether to render (Teeworlds handles this)
    
    Returns:
        Dict with evaluation metrics
    """
    _check_sb3()
    
    logger.info(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    env = make_env(remote_url=remote_url, single_agent_mode=True)
    
    episode_rewards = []
    episode_lengths = []
    wins = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if info.get("winner") == 0:
            wins += 1
        
        logger.info(
            f"Episode {episode + 1}/{num_episodes} | "
            f"Reward: {episode_reward:.2f} | "
            f"Length: {episode_length} | "
            f"Winner: {info.get('winner')}"
        )
    
    env.close()
    
    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "win_rate": wins / num_episodes,
    }
    
    logger.info(f"\nEvaluation Results:")
    logger.info(f"  Mean Reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    logger.info(f"  Mean Length: {results['mean_length']:.1f}")
    logger.info(f"  Win Rate: {results['win_rate']:.1%}")
    
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point for training CLI."""
    parser = argparse.ArgumentParser(
        description="TeeUnit Self-Play RL Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument(
        "--steps", type=int, default=1_000_000,
        help="Total training timesteps"
    )
    train_parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Minibatch size"
    )
    train_parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to train on"
    )
    train_parser.add_argument(
        "--save-path", type=str, default="models/teeunit_ppo",
        help="Path to save models"
    )
    train_parser.add_argument(
        "--remote", type=str, default=None,
        help="Remote TeeUnit server URL"
    )
    train_parser.add_argument(
        "--single-agent", action="store_true",
        help="Train single agent (others random)"
    )
    train_parser.add_argument(
        "--num-envs", type=int, default=1,
        help="Number of parallel environments"
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model")
    eval_parser.add_argument(
        "model_path", type=str,
        help="Path to saved model"
    )
    eval_parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of evaluation episodes"
    )
    eval_parser.add_argument(
        "--remote", type=str, default=None,
        help="Remote TeeUnit server URL"
    )
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(
            total_timesteps=args.steps,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            save_path=args.save_path,
            remote_url=args.remote,
            single_agent_mode=args.single_agent,
            num_envs=args.num_envs,
        )
    elif args.command == "eval":
        evaluate(
            model_path=args.model_path,
            num_episodes=args.episodes,
            remote_url=args.remote,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
