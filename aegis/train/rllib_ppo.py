"""RLlib PPO multi-agent training script for AEGIS."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, *args, **kwargs):
        return iterable


_NP_COMPAT_ALIASES = {
    'float_': np.float64,
    'int_': np.int64,
    'bool_': np.bool_,
    'object_': np.object_,
}
for _alias, _dtype in _NP_COMPAT_ALIASES.items():
    if not hasattr(np, _alias):
        setattr(np, _alias, _dtype)

import torch

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.tune.registry import register_env

from aegis.core.rules import GameConfig
from aegis.core.world import Role
from aegis.env.pettingzoo_env import AegisEnv


class AegisRLlibEnv(MultiAgentEnv):
    """
    RLlib-compatible MultiAgentEnv wrapper for AegisEnv.
    
    This wrapper converts the PettingZoo ParallelEnv to RLlib's MultiAgentEnv
    format, properly handling observation spaces and action masking.
    """
    
    def __init__(self, config: dict = None):
        super().__init__()
        config = config or {}
        
        # Create the underlying AegisEnv
        env_config = config.get("env_config", {})
        rewards_config = config.get("rewards_config", {})
        
        self.aegis_env = create_env_from_dicts(env_config, rewards_config)
        
        # Store agent IDs
        self._agent_ids = set(self.aegis_env.possible_agents)
        
        # Get observation and action spaces
        sample_agent = self.aegis_env.possible_agents[0]
        self._obs_space = self.aegis_env.observation_space(sample_agent)
        self._action_space = self.aegis_env.action_space(sample_agent)
    
    @property
    def observation_space(self):
        """Return the observation space."""
        return self._obs_space
    
    @property
    def action_space(self):
        """Return the action space."""
        return self._action_space
    
    def get_agent_ids(self):
        """Return the set of agent IDs."""
        return self._agent_ids
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment."""
        obs, infos = self.aegis_env.reset(seed=seed, options=options)
        return obs, infos
    
    def step(self, action_dict):
        """Take a step in the environment."""
        obs, rewards, terminations, truncations, infos = self.aegis_env.step(action_dict)
        
        # RLlib expects "__all__" key in terminations and truncations
        all_done = all(terminations.get(a, False) for a in self._agent_ids)
        all_truncated = all(truncations.get(a, False) for a in self._agent_ids)
        
        terminations["__all__"] = all_done
        truncations["__all__"] = all_truncated
        
        return obs, rewards, terminations, truncations, infos
    
    def render(self):
        """Render the environment."""
        return self.aegis_env.render()
    
    def close(self):
        """Close the environment."""
        self.aegis_env.close()


def create_env_from_dicts(env_config: dict, rewards_config: dict) -> AegisEnv:
    """Create an AEGIS environment from config dicts."""
    rewards = rewards_config or {}
    game_config = GameConfig(
        num_survivors=env_config.get("num_survivors", 4),
        num_impostors=env_config.get("num_impostors", 1),
        num_rooms=env_config.get("num_rooms", 9),
        evac_room=env_config.get("evac_room", 4),
        vision_radius=env_config.get("vision_radius", 2),
        tasks_per_survivor=env_config.get("tasks_per_survivor", 4),
        task_ticks_single=env_config.get("task_ticks_single", 5),
        task_ticks_two_step=env_config.get("task_ticks_two_step", 3),
        two_step_task_probability=env_config.get("two_step_task_probability", 0.3),
        kill_cooldown=env_config.get("kill_cooldown", 30),
        door_close_duration=env_config.get("door_close_duration", 15),
        door_cooldown=env_config.get("door_cooldown", 40),
        protect_evac_door=env_config.get("protect_evac_door", True),
        meeting_discussion_ticks=env_config.get("meeting_discussion_ticks", 60),
        meeting_voting_ticks=env_config.get("meeting_voting_ticks", 30),
        evac_ticks_required=env_config.get("evac_ticks_required", 5),
        max_ticks=env_config.get("max_ticks", 1000),
        enable_knower=env_config.get("enable_knower", False),
        max_messages_per_meeting=env_config.get("max_messages_per_meeting", 10),
        comm_mode=env_config.get("comm_mode", "broadcast"),
        memory_mode=env_config.get("memory_mode", "none"),
        # Reward parameters
        reward_win=rewards.get("win", 1.0),
        reward_loss=rewards.get("loss", -1.0),
        reward_task_step=rewards.get("task_step", 0.05),
        reward_correct_report=rewards.get("correct_report", 0.2),
        reward_impostor_ejected_survivor=rewards.get("impostor_ejected_survivor", 0.3),
        reward_survivor_ejected_survivor=rewards.get("survivor_ejected_survivor", -0.3),
        reward_kill=rewards.get("kill", 0.3),
        reward_impostor_ejected_impostor=rewards.get("impostor_ejected_impostor", -0.4),
        reward_survivor_ejected_impostor=rewards.get("survivor_ejected_impostor", 0.2),
        reward_time_penalty=rewards.get("time_penalty", -0.001),
        reward_proximity_penalty=rewards.get("proximity_penalty", -0.01),
        proximity_penalty_ticks=rewards.get("proximity_penalty_ticks", 1),
    )
    return AegisEnv(config=game_config)


class ActionMaskModel(TorchModelV2, torch.nn.Module):
    """
    Custom model that handles action masking for invalid actions.
    
    Works with both flattened (Box) and structured (Dict) observation spaces.
    When using Dict obs, the action_mask key is used to mask invalid actions.
    """
    
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        torch.nn.Module.__init__(self)
        
        # Store original obs space for reference
        self.original_obs_space = model_config.get("custom_model_config", {}).get(
            "original_obs_space", obs_space
        )
        
        # Calculate observation size
        # Handle both Dict and Box observation spaces
        from gymnasium import spaces as gym_spaces
        
        # Action mask size is always the action space size
        self.action_mask_size = action_space.n
        
        if isinstance(obs_space, gym_spaces.Dict):
            obs_size = 0
            for key, space in obs_space.spaces.items():
                if key != "action_mask":
                    if isinstance(space, gym_spaces.Discrete):
                        obs_size += 1  # Discrete is a single integer
                    else:
                        obs_size += int(np.prod(space.shape))
            self.is_dict_obs = True
        else:
            total_size = int(np.prod(obs_space.shape))
            obs_size = total_size - self.action_mask_size
            self.is_dict_obs = False
        
        # Get hidden layer sizes from config
        hiddens = model_config.get("fcnet_hiddens", [256, 256])
        activation = model_config.get("fcnet_activation", "relu")
        
        # Build network layers
        layers = []
        prev_size = obs_size
        
        for hidden_size in hiddens:
            layers.append(torch.nn.Linear(prev_size, hidden_size))
            if activation == "relu":
                layers.append(torch.nn.ReLU())
            elif activation == "tanh":
                layers.append(torch.nn.Tanh())
            prev_size = hidden_size
        
        self.feature_net = torch.nn.Sequential(*layers)
        
        # Policy head
        self.policy_head = torch.nn.Linear(prev_size, num_outputs)
        
        # Value head
        self.value_head = torch.nn.Linear(prev_size, 1)
        
        self._features = None
    
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        device = next(self.parameters()).device
        
        if isinstance(obs, dict):
            flat_obs = []
            for key, value in sorted(obs.items()):
                if key != "action_mask":
                    if isinstance(value, torch.Tensor):
                        flat_obs.append(value.float().to(device).view(value.shape[0], -1))
                    else:
                        tensor = torch.tensor(value, dtype=torch.float32, device=device)
                        flat_obs.append(tensor.view(tensor.shape[0], -1))
            
            x = torch.cat(flat_obs, dim=-1)
            action_mask = obs.get("action_mask")
            if action_mask is not None:
                if isinstance(action_mask, torch.Tensor):
                    action_mask = action_mask.to(device)
                else:
                    action_mask = torch.tensor(action_mask, dtype=torch.float32, device=device)
        else:
            if isinstance(obs, torch.Tensor):
                obs_tensor = obs.float().to(device)
            else:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            
            action_mask = obs_tensor[..., :self.action_mask_size]
            x = obs_tensor[..., self.action_mask_size:]
        
        self._features = self.feature_net(x)
        
        logits = self.policy_head(self._features)
        
        if action_mask is not None:
            if not isinstance(action_mask, torch.Tensor):
                action_mask = torch.tensor(action_mask, dtype=torch.float32, device=device)
            
            inf_mask = torch.clamp(
                torch.log(action_mask.float() + 1e-10),
                min=FLOAT_MIN,
            )
            logits = logits + inf_mask
        
        return logits, state
    
    def value_function(self):
        return self.value_head(self._features).squeeze(-1)


def create_env(config: dict, rewards_config: Optional[dict] = None) -> AegisEnv:
    """Create an AEGIS environment from config dict."""
    return create_env_from_dicts(config, rewards_config or {})


def env_creator(config: dict):
    """Environment creator for RLlib registration."""
    # RLlib passes env_config, but we need to merge with rewards
    # If config has "env" key, it's the full config; otherwise it's just env_config
    if "env" in config:
        env_config = config["env"]
        rewards_config = config.get("rewards", {})
    else:
        # RLlib passed just env_config, use defaults for rewards
        env_config = config
        rewards_config = {}
    
    return AegisRLlibEnv({
        "env_config": env_config,
        "rewards_config": rewards_config,
    })


def policy_mapping_fn(agent_id: str, episode, worker, **kwargs) -> str:
    """Map agent IDs to policy names."""
    # Extract agent index from name like "agent_0"
    agent_idx = int(agent_id.split("_")[1])
    
    # Get agent role from environment (this is a simplification)
    # In practice, we determine this from the environment state
    # For now, we use the first N agents as impostors based on config
    # This is set during training setup
    return "survivor_policy"  # Default, will be overridden by actual mapping


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def is_gpu_compatible() -> bool:
    """
    Check if GPU is available and compatible with PyTorch.
    
    Returns:
        True if GPU is available and can be used, False otherwise.
    """
    # Check environment variable to force CPU
    if os.environ.get("FORCE_CPU", "0").lower() in ("1", "true", "yes"):
        return False
    
    # If CUDA_VISIBLE_DEVICES is explicitly set to empty, force CPU
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible == "" and "CUDA_VISIBLE_DEVICES" in os.environ:
        return False
    
    if not torch.cuda.is_available():
        return False
    
    try:
        # Try to create a small tensor on GPU to verify compatibility
        # Use CUDA_LAUNCH_BLOCKING to get immediate errors
        with torch.cuda.device(0):
            test_tensor = torch.tensor([1.0], device="cuda:0")
            # Try a simple operation that requires kernel execution
            result = torch.relu(test_tensor)
            # Clean up
            del test_tensor, result
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return True
    except (RuntimeError, torch.cuda.DeviceError, Exception) as e:
        # GPU is not compatible or not working
        error_msg = str(e)
        if "no kernel image" in error_msg.lower() or "cuda capability" in error_msg.lower():
            print(f"GPU compute capability not supported by PyTorch: {error_msg}")
        else:
            print(f"GPU not compatible or not available: {error_msg}")
        print("Falling back to CPU training.")
        return False


def select_torch_device(device_pref: str) -> str:
    """
    Select torch device based on user preference and availability.
    
    Args:
        device_pref: "auto", "cpu", or "cuda"
    
    Returns:
        Selected device string: "cpu" or "cuda"
    """
    pref = (device_pref or "auto").lower()
    if pref not in ("auto", "cpu", "cuda"):
        print(f"Unknown device '{device_pref}', falling back to auto.")
        pref = "auto"
    
    if pref == "cpu":
        return "cpu"
    
    if pref == "cuda":
        return "cuda" if is_gpu_compatible() else "cpu"
    
    return "cuda" if is_gpu_compatible() else "cpu"


def train(
    config_path: Optional[str] = None,
    num_iterations: int = 1000,
    checkpoint_dir: str = "./checkpoints",
    use_wandb: bool = False,
    wandb_project: str = "aegis",
    device: str = "auto",
):
    """
    Train AEGIS agents using PPO.
    
    Args:
        config_path: Path to YAML config file
        num_iterations: Number of training iterations
        checkpoint_dir: Directory for checkpoints
        use_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name
    """
    # Load config
    if config_path:
        config = load_config(config_path)
    else:
        config = {}
    
    env_config = config.get("env", {})
    rewards_config = config.get("rewards", {})
    training_config = config.get("training", {})
    logging_config = config.get("logging", {})
    
    # Select torch device based on user preference
    selected_device = select_torch_device(device)
    if selected_device == "cpu":
        if torch.cuda.is_available() and device in ("cuda", "auto"):
            print("GPU detected but not compatible. Forcing CPU training.")
        torch.set_default_device("cpu")
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Register custom model
    ModelCatalog.register_custom_model("action_mask_model", ActionMaskModel)
    
    # Register environment - pass full config structure
    full_config = {"env": env_config, "rewards": rewards_config}
    register_env("aegis", lambda cfg: env_creator(full_config))
    
    # Determine number of agents and roles
    num_survivors = env_config.get("num_survivors", 4)
    num_impostors = env_config.get("num_impostors", 1)
    num_agents = num_survivors + num_impostors
    
    # Create sample environment to get spaces
    sample_env = create_env(env_config, rewards_config)
    obs_space = sample_env.observation_space("agent_0")
    act_space = sample_env.action_space("agent_0")
    
    # Define policies
    policies = {
        "survivor_policy": (None, obs_space, act_space, {}),
        "impostor_policy": (None, obs_space, act_space, {}),
    }
    
    # Create policy mapping that will be set per episode
    # We need to know which agents are impostors
    def dynamic_policy_mapping(agent_id, episode, worker, **kwargs):
        # Access the underlying environment to get roles
        if hasattr(episode, "_agent_to_policy"):
            return episode._agent_to_policy.get(agent_id, "survivor_policy")
        return "survivor_policy"
    
    # Build PPO config
    model_config = training_config.get("model", {})
    
    ppo_config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env="aegis",
            env_config=env_config,
        )
        .framework("torch")
        .env_runners(
            num_env_runners=training_config.get("num_workers", 4),
            num_envs_per_env_runner=training_config.get("num_envs_per_worker", 2),
            rollout_fragment_length="auto",
        )
        .training(
            train_batch_size=training_config.get("train_batch_size", 4000),
            minibatch_size=training_config.get("sgd_minibatch_size", 128),
            num_sgd_iter=training_config.get("num_sgd_iter", 10),
            lr=training_config.get("lr", 0.0003),
            entropy_coeff=training_config.get("entropy_coeff", 0.01),
            clip_param=training_config.get("clip_param", 0.2),
            vf_clip_param=training_config.get("vf_clip_param", 10.0),
            gamma=training_config.get("gamma", 0.99),
            lambda_=training_config.get("lambda_", 0.95),
            model={
                "custom_model": "action_mask_model",
                "fcnet_hiddens": model_config.get("fcnet_hiddens", [256, 256]),
                "fcnet_activation": model_config.get("fcnet_activation", "relu"),
                # Keep Dict observation structure for action masking
                "_disable_preprocessor_api": True,
            },
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=lambda agent_id, *args, **kwargs: (
                "impostor_policy" if int(agent_id.split("_")[1]) < num_impostors
                else "survivor_policy"
            ),
        )
        .evaluation(
            evaluation_interval=training_config.get("evaluation_interval", 25),
            evaluation_num_env_runners=1,
            evaluation_duration=training_config.get("evaluation_num_episodes", 10),
            evaluation_duration_unit="episodes",
        )
        .resources(
            num_gpus=1 if selected_device == "cuda" else 0,
        )
    )
    
    # Build algorithm
    algo = ppo_config.build_algo()
    
    # Setup wandb logging if requested
    if use_wandb or logging_config.get("use_wandb", False):
        try:
            import wandb
            wandb.init(
                project=wandb_project or logging_config.get("wandb_project", "aegis"),
                entity=logging_config.get("wandb_entity"),
                config=config,
            )
        except ImportError:
            print("wandb not installed, skipping W&B logging")
            use_wandb = False
    
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Training loop with progress bar
    print(f"Starting training for {num_iterations} iterations...")
    
    # Create progress bar
    pbar = tqdm(
        range(num_iterations),
        desc="Training",
        unit="iter",
        disable=not HAS_TQDM,
        ncols=100,
    )
    
    def _get_metric(result: Dict[str, Any], key: str, default: float = 0.0) -> float:
        """Fetch metrics from RLlib result across API versions."""
        if key in result:
            return result.get(key, default)
        for container in ("env_runners", "sampler_results", "evaluation"):
            if container in result and key in result[container]:
                return result[container].get(key, default)
        return default

    for i in pbar:
        result = algo.train()
        
        episode_reward_mean = _get_metric(result, "episode_reward_mean", 0.0)
        episode_len_mean = _get_metric(result, "episode_len_mean", 0.0)
        
        if HAS_TQDM:
            pbar.set_postfix({
                "reward": f"{episode_reward_mean:.3f}",
                "len": f"{episode_len_mean:.1f}",
                "iter": f"{i + 1}/{num_iterations}",
            })
        
        # Also print detailed info (less frequently or when checkpointing)
        checkpoint_freq = training_config.get("checkpoint_freq", 50)
        if (i + 1) % checkpoint_freq == 0 or (i + 1) % 10 == 0:
            print(f"\nIteration {i + 1}/{num_iterations}: "
                  f"reward={episode_reward_mean:.3f}, "
                  f"len={episode_len_mean:.1f}")
        
        if use_wandb:
            wandb.log({
                "iteration": i + 1,
                "episode_reward_mean": episode_reward_mean,
                "episode_len_mean": episode_len_mean,
                **{f"policy/{k}": v for k, v in result.get("policy_reward_mean", {}).items()},
            })
        
        # Save checkpoint
        if (i + 1) % checkpoint_freq == 0:
            checkpoint_path = algo.save(checkpoint_dir)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Close progress bar
    pbar.close()
    
    # Final checkpoint
    final_path = algo.save(checkpoint_dir)
    print(f"Training complete. Final checkpoint: {final_path}")
    
    # Cleanup
    algo.stop()
    ray.shutdown()
    
    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train AEGIS agents with PPO")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="aegis",
        help="W&B project name",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Torch device selection (auto, cpu, cuda)",
    )
    
    args = parser.parse_args()
    
    train(
        config_path=args.config,
        num_iterations=args.iterations,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        device=args.device,
    )


if __name__ == "__main__":
    main()

