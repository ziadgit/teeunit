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

# Check for HuggingFace Hub
try:
    from huggingface_hub import HfApi, upload_folder
    HAS_HF = True
except ImportError:
    HAS_HF = False
    HfApi = None
    upload_folder = None

from .openenv_models import (
    TeeAction,
    TeeMultiAction,
    TeeObservation,
    RewardConfig,
)
from .openenv_environment import TeeEnvironment, TeeConfig
from .openenv_client import create_client, SyncTeeEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _check_sb3():
    """Check if SB3 is available and raise helpful error if not."""
    if not HAS_SB3:
        raise ImportError(
            "Stable-Baselines3 and Gymnasium required for training. "
            "Install with: pip install teeunit[rl]"
        )


def _check_hf():
    """Check if HuggingFace Hub is available and raise helpful error if not."""
    if not HAS_HF:
        raise ImportError(
            "huggingface_hub required for uploading models. "
            "Install with: pip install huggingface_hub"
        )


def upload_to_huggingface(
    model_path: str,
    repo_id: str,
    commit_message: str = "Upload trained TeeUnit model",
) -> str:
    """
    Upload trained model to HuggingFace Hub.
    
    Args:
        model_path: Path to saved model directory
        repo_id: HuggingFace repo ID (e.g., "ziadbc/teeunit-agent")
        commit_message: Commit message for the upload
    
    Returns:
        URL of the uploaded model
    
    Requires HF_TOKEN environment variable to be set.
    """
    _check_hf()
    
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF_TOKEN environment variable required for upload. "
            "Get your token from https://huggingface.co/settings/tokens"
        )
    
    api = HfApi(token=token)
    
    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
        logger.info(f"Created/verified repo: {repo_id}")
    except Exception as e:
        logger.warning(f"Could not create repo (may already exist): {e}")
    
    # Upload the model folder
    logger.info(f"Uploading model from {model_path} to {repo_id}...")
    
    url = upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        commit_message=commit_message,
        token=token,
    )
    
    logger.info(f"Model uploaded successfully: https://huggingface.co/{repo_id}")
    return f"https://huggingface.co/{repo_id}"


# =============================================================================
# Gymnasium Wrapper for SB3
# =============================================================================

if HAS_SB3:
    
    # Image observation dimensions for CNN policy
    # We create a 4-channel 16x16 "image" from the state:
    # Channel 0: Self state (health, armor, ammo, position encoded)
    # Channel 1: Player positions (rendered as gaussians)
    # Channel 2: Projectile positions
    # Channel 3: Pickup positions
    IMG_SIZE = 16
    IMG_CHANNELS = 4
    
    def _tensor_to_image(tensor: np.ndarray) -> np.ndarray:
        """
        Convert 195-dim observation tensor to 4x16x16 image for CNN.
        
        This creates a spatial representation that CNN can process effectively,
        utilizing GPU for convolution operations.
        """
        img = np.zeros((IMG_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        
        # Channel 0: Self state encoded as pattern
        # 13 self features: x, y, vel_x, vel_y, health, armor, weapon, ammo, direction, grounded, alive, score, agent_id
        self_state = tensor[:13]
        # Normalize and tile into a pattern
        health_norm = self_state[4] / 10.0  # health normalized
        armor_norm = self_state[5] / 10.0   # armor normalized
        # Fill top-left quadrant with self state pattern
        img[0, :4, :4] = health_norm
        img[0, :4, 4:8] = armor_norm
        img[0, :4, 8:12] = self_state[6] / 5.0  # weapon
        img[0, :4, 12:] = (self_state[8] + 1) / 2.0  # direction normalized
        
        # Channel 1: Visible players (70 features = 10 players x 7)
        # Features per player: rel_x, rel_y, vel_x, vel_y, health, weapon, team
        players = tensor[13:83].reshape(10, 7)
        for i, player in enumerate(players):
            if player[4] > 0:  # has health = visible
                # Map relative position to image coords (assume +-500 range)
                px = int(np.clip((player[0] / 1000.0 + 0.5) * IMG_SIZE, 0, IMG_SIZE-1))
                py = int(np.clip((player[1] / 1000.0 + 0.5) * IMG_SIZE, 0, IMG_SIZE-1))
                img[1, py, px] = player[4] / 10.0  # health as intensity
        
        # Channel 2: Projectiles (64 features = 16 projectiles x 4)
        # Features: rel_x, rel_y, vel_x, vel_y
        projectiles = tensor[83:147].reshape(16, 4)
        for proj in projectiles:
            if proj[0] != 0 or proj[1] != 0:  # has position
                px = int(np.clip((proj[0] / 1000.0 + 0.5) * IMG_SIZE, 0, IMG_SIZE-1))
                py = int(np.clip((proj[1] / 1000.0 + 0.5) * IMG_SIZE, 0, IMG_SIZE-1))
                # Encode velocity as intensity
                vel = np.sqrt(proj[2]**2 + proj[3]**2)
                img[2, py, px] = min(1.0, vel / 500.0)
        
        # Channel 3: Pickups (48 features = 16 pickups x 3)
        # Features: rel_x, rel_y, type
        pickups = tensor[147:195].reshape(16, 3)
        for pickup in pickups:
            if pickup[0] != 0 or pickup[1] != 0:
                px = int(np.clip((pickup[0] / 1000.0 + 0.5) * IMG_SIZE, 0, IMG_SIZE-1))
                py = int(np.clip((pickup[1] / 1000.0 + 0.5) * IMG_SIZE, 0, IMG_SIZE-1))
                img[3, py, px] = (pickup[2] + 1) / 5.0  # type as intensity
        
        return img
    
    class TeeUnitGymEnv(gym.Env):
        """
        Gymnasium wrapper for TeeUnit environment.
        
        Designed for self-play training where all agents share the same policy.
        Uses CNN-friendly image observations to leverage GPU acceleration.
        
        The observation is converted to a 4x16x16 image format:
        - Channel 0: Self state (health, armor, weapon, direction)
        - Channel 1: Enemy player positions
        - Channel 2: Projectile positions and velocities
        - Channel 3: Pickup positions and types
        """
        
        metadata = {"render_modes": ["human"]}
        
        def __init__(
            self,
            env: Optional[TeeEnvironment] = None,
            client: Optional[SyncTeeEnv] = None,
            num_agents: int = 4,
            single_agent_mode: bool = False,
            agent_id: int = 0,
            use_cnn: bool = True,
        ):
            """
            Initialize the Gymnasium wrapper.
            
            Args:
                env: Local TeeEnvironment (if running locally)
                client: Remote SyncTeeEnv (if connecting to server)
                num_agents: Number of agents in the environment
                single_agent_mode: If True, only control one agent (others use random actions)
                agent_id: Which agent to control in single-agent mode
                use_cnn: If True, use image observations for CNN policy (GPU-accelerated)
            """
            super().__init__()
            
            self.env = env
            self.client = client
            self.num_agents = num_agents
            self.single_agent_mode = single_agent_mode
            self.agent_id = agent_id
            self.use_cnn = use_cnn
            
            # Observation space
            if use_cnn:
                # Image observation for CNN: 4 channels x 16x16
                if single_agent_mode:
                    self.observation_space = spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(IMG_CHANNELS, IMG_SIZE, IMG_SIZE),
                        dtype=np.float32,
                    )
                else:
                    # Multi-agent: batch of images
                    self.observation_space = spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(num_agents, IMG_CHANNELS, IMG_SIZE, IMG_SIZE),
                        dtype=np.float32,
                    )
            else:
                # Original flat observation
                obs_dim = 195
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
            """Convert observations dict to numpy array (flat or image format)."""
            if self.single_agent_mode:
                obs = observations.get(self.agent_id)
                if obs is None:
                    if self.use_cnn:
                        return np.zeros((IMG_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=np.float32)
                    return np.zeros(195, dtype=np.float32)
                tensor = obs.to_tensor()
                if self.use_cnn:
                    return _tensor_to_image(tensor)
                return tensor
            else:
                # Stack all agent observations
                obs_list = []
                for agent_id in range(self.num_agents):
                    obs = observations.get(agent_id)
                    if obs is None:
                        if self.use_cnn:
                            obs_list.append(np.zeros((IMG_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=np.float32))
                        else:
                            obs_list.append(np.zeros(195, dtype=np.float32))
                    else:
                        tensor = obs.to_tensor()
                        if self.use_cnn:
                            obs_list.append(_tensor_to_image(tensor))
                        else:
                            obs_list.append(tensor)
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
    use_cnn: bool = True,
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
        use_cnn: Use CNN-compatible image observations (GPU-accelerated)
    
    Returns:
        TeeUnitGymEnv instance
    """
    _check_sb3()
    
    if remote_url:
        # Remote mode - connect to TeeUnit server
        client = create_client(remote_url)
        return TeeUnitGymEnv(
            client=client,
            num_agents=num_agents,
            single_agent_mode=single_agent_mode,
            agent_id=agent_id,
            use_cnn=use_cnn,
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
            use_cnn=use_cnn,
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
    upload_hf: bool = False,
    hf_repo: str = "ziadbc/teeunit-agent",
    use_cnn: bool = True,
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
        upload_hf: Whether to upload model to HuggingFace Hub after training
        hf_repo: HuggingFace repo ID for upload
        use_cnn: Use CNN policy with image observations (GPU-accelerated)
    
    Returns:
        Trained PPO model
    """
    _check_sb3()
    
    policy_type = "CnnPolicy" if use_cnn else "MlpPolicy"
    logger.info(f"Creating TeeUnit environment (CNN mode: {use_cnn})...")
    
    # Create environment(s)
    if num_envs > 1:
        # Parallel environments
        def make_single_env():
            return make_env(
                remote_url=remote_url,
                single_agent_mode=single_agent_mode,
                use_cnn=use_cnn,
            )
        env = DummyVecEnv([make_single_env for _ in range(num_envs)])
    else:
        env = make_env(
            remote_url=remote_url,
            single_agent_mode=single_agent_mode,
            use_cnn=use_cnn,
        )
    
    logger.info(f"Observation space: {env.observation_space}")
    logger.info(f"Action space: {env.action_space}")
    logger.info(f"Policy: {policy_type}")
    logger.info(f"Device: {device}")
    
    # Create PPO model with CNN or MLP policy
    model = PPO(
        policy_type,
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
    
    # Upload to HuggingFace Hub if requested
    if upload_hf:
        try:
            url = upload_to_huggingface(
                model_path=save_path,
                repo_id=hf_repo,
                commit_message=f"Training complete: {total_timesteps} steps",
            )
            logger.info(f"Model uploaded to HuggingFace: {url}")
        except Exception as e:
            logger.error(f"Failed to upload to HuggingFace: {e}")
            raise
    
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
    train_parser.add_argument(
        "--upload-hf", action="store_true",
        help="Upload model to HuggingFace Hub after training"
    )
    train_parser.add_argument(
        "--hf-repo", type=str, default="ziadbc/teeunit-agent",
        help="HuggingFace repo ID for upload"
    )
    train_parser.add_argument(
        "--use-cnn", action="store_true", default=True,
        help="Use CNN policy with image observations (GPU-accelerated, default: True)"
    )
    train_parser.add_argument(
        "--no-cnn", action="store_true",
        help="Use MLP policy instead of CNN"
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
        # Determine CNN mode
        use_cnn = args.use_cnn and not args.no_cnn
        train(
            total_timesteps=args.steps,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            save_path=args.save_path,
            remote_url=args.remote,
            single_agent_mode=args.single_agent,
            num_envs=args.num_envs,
            upload_hf=args.upload_hf,
            hf_repo=args.hf_repo,
            use_cnn=use_cnn,
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
