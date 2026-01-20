from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field, fields, replace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.distributions import Categorical

from aegis.core.rules import GameConfig
from aegis.env.pettingzoo_env import AegisEnv


@dataclass
class Args:
    """Training arguments."""
    # Training
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    num_steps: int = 128
    num_minibatches: int = 4
    update_epochs: int = 4
    
    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    
    # Architecture
    hidden_dim: int = 256
    
    # Misc
    seed: int = 1
    device: str = "cpu"
    log_interval: int = 10
    save_interval: int = 100
    checkpoint_timesteps: Optional[int] = None
    checkpoint_dir: str = "./checkpoints"
    
    game_config: Optional[dict] = None

    @classmethod
    def from_yaml(cls, path: str) -> "Args":
        """Load Args from YAML config file."""
        with open(path) as f:
            cfg = yaml.safe_load(f)
        
        env = cfg.get("env", {})
        rewards = cfg.get("rewards", {})
        training = cfg.get("training", {})
        logging = cfg.get("logging", {})
        model = training.get("model", {})
        
        hidden = model.get("fcnet_hiddens", [256])
        device = training.get("device", "cpu")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Build GameConfig dict from env + rewards
        game_config = {
            "num_survivors": env.get("num_survivors", 2),
            "num_impostors": env.get("num_impostors", 1),
            "num_rooms": env.get("num_rooms", 9),
            "evac_room": env.get("evac_room", 4),
            "vision_radius": env.get("vision_radius", 2),
            "tasks_per_survivor": env.get("tasks_per_survivor", 4),
            "task_ticks_single": env.get("task_ticks_single", 5),
            "task_ticks_two_step": env.get("task_ticks_two_step", 3),
            "two_step_task_probability": env.get("two_step_task_probability", 0.3),
            "kill_cooldown": env.get("kill_cooldown", 30),
            "door_close_duration": env.get("door_close_duration", 15),
            "door_cooldown": env.get("door_cooldown", 40),
            "protect_evac_door": env.get("protect_evac_door", True),
            "meeting_discussion_ticks": env.get("meeting_discussion_ticks", 60),
            "meeting_voting_ticks": env.get("meeting_voting_ticks", 30),
            "evac_ticks_required": env.get("evac_ticks_required", 5),
            "max_ticks": env.get("max_ticks", 500),
            "enable_knower": env.get("enable_knower", False),
            "max_messages_per_meeting": env.get("max_messages_per_meeting", 10),
            "comm_mode": env.get("comm_mode", "broadcast"),
            "memory_mode": env.get("memory_mode", "none"),
            # Rewards
            "reward_win": rewards.get("win", 1.0),
            "reward_loss": rewards.get("loss", -1.0),
            "reward_task_step": rewards.get("task_step", 0.05),
            "reward_correct_report": rewards.get("correct_report", 0.2),
            "reward_impostor_ejected_survivor": rewards.get("impostor_ejected_survivor", 0.3),
            "reward_survivor_ejected_survivor": rewards.get("survivor_ejected_survivor", -0.3),
            "reward_kill": rewards.get("kill", 0.3),
            "reward_impostor_ejected_impostor": rewards.get("impostor_ejected_impostor", -0.4),
            "reward_survivor_ejected_impostor": rewards.get("survivor_ejected_impostor", 0.2),
            "reward_time_penalty": rewards.get("time_penalty", -0.001),
            "reward_proximity_penalty": rewards.get("proximity_penalty", -0.01),
            "proximity_penalty_ticks": rewards.get("proximity_penalty_ticks", 1),
        }
        
        num_steps = training.get("rollout_steps", training.get("num_steps", 128))
        
        return cls(
            total_timesteps=training.get("total_timesteps", 1_000_000),
            learning_rate=training.get("lr", 3e-4),
            num_steps=num_steps,
            num_minibatches=training.get("num_minibatches", 4),
            update_epochs=training.get("num_sgd_iter", 4),
            gamma=training.get("gamma", 0.99),
            gae_lambda=training.get("lambda_", 0.95),
            clip_coef=training.get("clip_param", 0.2),
            clip_vloss=training.get("clip_vloss", True),
            vf_coef=training.get("vf_coef", 0.5),
            ent_coef=training.get("entropy_coeff", 0.01),
            max_grad_norm=training.get("max_grad_norm", 0.5),
            target_kl=training.get("target_kl"),
            hidden_dim=hidden[0] if hidden else 256,
            seed=env.get("seed") or 1,
            device=device,
            log_interval=10,
            save_interval=training.get("checkpoint_freq", 100),
            checkpoint_timesteps=training.get("checkpoint_timesteps"),
            checkpoint_dir=logging.get("checkpoint_dir", "./checkpoints"),
            game_config=game_config,
        )

    def override(self, **kwargs) -> "Args":
        """Return new Args with specified fields overridden (ignores None)."""
        updates = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **updates)


def flatten_obs(obs_dict: dict) -> np.ndarray:
    """Flatten a Dict observation into a 1D numpy array."""
    parts = []
    
    # Phase (one-hot)
    phase_oh = np.zeros(3, dtype=np.float32)
    phase_oh[obs_dict["phase"]] = 1.0
    parts.append(phase_oh)
    
    # Self state (one-hot encoded)
    self_room_oh = np.zeros(9, dtype=np.float32)  # Max 9 rooms
    self_room_oh[obs_dict["self_room"]] = 1.0
    parts.append(self_room_oh)
    
    self_role_oh = np.zeros(2, dtype=np.float32)
    self_role_oh[obs_dict["self_role"]] = 1.0
    parts.append(self_role_oh)
    
    parts.append(np.array([obs_dict["self_alive"]], dtype=np.float32))
    parts.append(np.array([obs_dict["evac_active"]], dtype=np.float32))
    
    # Visible agents (flattened)
    parts.append(obs_dict["visible_agents"].astype(np.float32).flatten())
    
    # Visible bodies (flattened)
    parts.append(obs_dict["visible_bodies"].astype(np.float32).flatten())
    
    # Door states (flattened)
    parts.append(obs_dict["door_states"].astype(np.float32).flatten())
    
    # Task state
    parts.append(obs_dict["task_state"].astype(np.float32))
    
    # Continuous features
    parts.append(obs_dict["team_task_progress"])
    parts.append(obs_dict["kill_cooldown"])
    parts.append(obs_dict["door_cooldown"])
    
    # Meeting messages (flattened)
    parts.append(obs_dict["meeting_messages"].astype(np.float32).flatten())
    
    # Meeting info
    parts.append(np.array([obs_dict["meeting_reporter"]], dtype=np.float32))
    parts.append(np.array([obs_dict["meeting_body_room"]], dtype=np.float32))
    
    # Knower info
    parts.append(np.array([obs_dict["knows_role_of"]], dtype=np.float32))
    parts.append(np.array([obs_dict["known_role"]], dtype=np.float32))
    
    # Tick (normalized)
    parts.append(obs_dict["tick"].astype(np.float32) / 500.0)
    
    return np.concatenate(parts)


def get_flat_obs_dim(env: AegisEnv) -> int:
    """Get the dimension of the flattened observation."""
    obs, _ = env.reset()
    sample_obs = obs[env.possible_agents[0]]
    flat = flatten_obs(sample_obs)
    return flat.shape[0]


def get_action_dim(env: AegisEnv) -> int:
    """Get action space size."""
    return env.action_space(env.possible_agents[0]).n


class CategoricalMasked(Categorical):
    """Categorical distribution with action masking."""
    
    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        # Apply mask: set invalid actions to large negative value
        self.mask = mask
        masked_logits = logits.clone()
        masked_logits[~mask.bool()] = -1e8
        super().__init__(logits=masked_logits)
    
    def entropy(self) -> torch.Tensor:
        # Entropy only over valid actions
        p_log_p = self.logits * self.probs
        p_log_p[~self.mask.bool()] = 0.0
        return -p_log_p.sum(dim=-1)


class ActorCritic(nn.Module):
    """
    Actor-Critic network with shared feature extractor.
    Supports action masking.
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Smaller init for output heads
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.shared(obs)
        return self.critic(features)
    
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            action: Selected action
            log_prob: Log probability of action
            entropy: Distribution entropy
            value: Value estimate
        """
        features = self.shared(obs)
        logits = self.actor(features)
        value = self.critic(features)
        
        dist = CategoricalMasked(logits=logits, mask=action_mask)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)


class RolloutBuffer:
    """Storage for rollout data."""
    
    def __init__(
        self,
        num_steps: int,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        device: str = "cpu",
    ):
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        
        # Pre-allocate tensors
        self.obs = torch.zeros((num_steps, num_agents, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((num_steps, num_agents), dtype=torch.long, device=device)
        self.logprobs = torch.zeros((num_steps, num_agents), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((num_steps, num_agents), dtype=torch.float32, device=device)
        self.dones = torch.zeros((num_steps, num_agents), dtype=torch.float32, device=device)
        self.values = torch.zeros((num_steps, num_agents), dtype=torch.float32, device=device)
        self.action_masks = torch.zeros((num_steps, num_agents, action_dim), dtype=torch.float32, device=device)
        
        self.step = 0
    
    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        action_masks: torch.Tensor,
    ):
        """Add one step of data."""
        self.obs[self.step] = obs
        self.actions[self.step] = actions
        self.logprobs[self.step] = logprobs
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.values[self.step] = values
        self.action_masks[self.step] = action_masks
        self.step += 1
    
    def reset(self):
        self.step = 0
    
    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        last_done: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        advantages = torch.zeros_like(self.rewards)
        lastgaelam = torch.zeros(self.num_agents, dtype=torch.float32, device=self.device)
        
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - last_done
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        
        returns = advantages + self.values
        return returns, advantages
    
    def flatten(self) -> tuple:
        """Flatten buffer for minibatch sampling."""
        batch_size = self.num_steps * self.num_agents
        
        b_obs = self.obs.reshape(batch_size, -1)
        b_actions = self.actions.reshape(batch_size)
        b_logprobs = self.logprobs.reshape(batch_size)
        b_action_masks = self.action_masks.reshape(batch_size, -1)
        b_values = self.values.reshape(batch_size)  # For value clipping
        
        return b_obs, b_actions, b_logprobs, b_action_masks, b_values


def collect_rollout(
    env: AegisEnv,
    agent: ActorCritic,
    buffer: RolloutBuffer,
    obs_dict: dict,
    device: str,
) -> tuple[dict, list, torch.Tensor]:
    """
    Collect one rollout of num_steps.
    
    Returns:
        obs_dict: Final observation dict
        completed_episodes: List of completed episode stats
        last_dones: Done flags from the last step (for correct GAE)
    """
    agent_names = env.possible_agents
    num_agents = len(agent_names)
    
    episode_rewards = {name: 0.0 for name in agent_names}
    episode_lengths = 0
    completed_episodes = []
    
    # Track dones from last step for GAE bootstrap
    last_dones = torch.zeros(num_agents, dtype=torch.float32, device=device)
    
    for step in range(buffer.num_steps):
        # Convert observations to tensors
        obs_flat = []
        masks = []
        for name in agent_names:
            obs_flat.append(flatten_obs(obs_dict[name]))
            masks.append(obs_dict[name]["action_mask"])
        
        obs_tensor = torch.tensor(np.array(obs_flat), dtype=torch.float32, device=device)
        mask_tensor = torch.tensor(np.array(masks), dtype=torch.float32, device=device)
        
        # Get actions
        with torch.no_grad():
            actions, logprobs, _, values = agent.get_action_and_value(
                obs_tensor, mask_tensor
            )
        
        # Step environment
        actions_dict = {name: actions[i].item() for i, name in enumerate(agent_names)}
        next_obs_dict, rewards_dict, terminations, truncations, infos = env.step(actions_dict)
        
        # Store in buffer
        rewards = torch.tensor([rewards_dict[name] for name in agent_names], dtype=torch.float32, device=device)
        dones = torch.tensor(
            [float(terminations[name] or truncations[name]) for name in agent_names],
            dtype=torch.float32,
            device=device,
        )
        
        buffer.add(
            obs=obs_tensor,
            actions=actions,
            logprobs=logprobs,
            rewards=rewards,
            dones=dones,
            values=values,
            action_masks=mask_tensor,
        )
        
        # Track episode stats
        for name in agent_names:
            episode_rewards[name] += rewards_dict[name]
        episode_lengths += 1
        
        # Check for episode end
        if any(terminations.values()) or any(truncations.values()):
            completed_episodes.append({
                "total_reward": sum(episode_rewards.values()),
                "mean_reward": np.mean(list(episode_rewards.values())),
                "length": episode_lengths,
                "winner": infos[agent_names[0]].get("winner"),
            })
            
            # Reset
            obs_dict, _ = env.reset()
            episode_rewards = {name: 0.0 for name in agent_names}
            episode_lengths = 0
            # After reset, dones are 0 for the new episode start
            last_dones = torch.zeros(num_agents, dtype=torch.float32, device=device)
        else:
            obs_dict = next_obs_dict
            # Update last_dones (will be 0 if episode didn't end)
            last_dones = dones
    
    return obs_dict, completed_episodes, last_dones


def ppo_update(
    agent: ActorCritic,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    args: Args,
) -> dict:
    """Perform PPO update on collected data."""
    b_obs, b_actions, b_old_logprobs, b_action_masks, b_old_values = buffer.flatten()
    b_returns = returns.reshape(-1)
    b_advantages = advantages.reshape(-1)    
    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
    
    batch_size = b_obs.shape[0]
    minibatch_size = batch_size // args.num_minibatches
    
    # Tracking
    total_pg_loss = 0.0
    total_v_loss = 0.0
    total_entropy = 0.0
    total_approx_kl = 0.0
    total_clipfrac = 0.0
    total_v_clipfrac = 0.0
    num_updates = 0
    
    for epoch in range(args.update_epochs):
        # Random permutation
        indices = torch.randperm(batch_size, device=args.device)
        
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_indices = indices[start:end]
            
            # Get minibatch
            mb_obs = b_obs[mb_indices]
            mb_actions = b_actions[mb_indices]
            mb_old_logprobs = b_old_logprobs[mb_indices]
            mb_old_values = b_old_values[mb_indices]
            mb_returns = b_returns[mb_indices]
            mb_advantages = b_advantages[mb_indices]
            mb_action_masks = b_action_masks[mb_indices]
            
            # Forward pass
            _, new_logprobs, entropy, new_values = agent.get_action_and_value(
                mb_obs, mb_action_masks, mb_actions
            )
            
            # Policy loss
            logratio = new_logprobs - mb_old_logprobs
            ratio = logratio.exp()
            
            # Approx KL for monitoring
            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
            
            # Clipped surrogate objective
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            
            # Value loss with proper clipping
            if args.clip_vloss:
                # Clipped value loss: clip new_values relative to old_values
                v_clipped = mb_old_values + torch.clamp(
                    new_values - mb_old_values, -args.clip_coef, args.clip_coef
                )
                v_loss_unclipped = (new_values - mb_returns) ** 2
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                
                # Track value clipping for monitoring
                with torch.no_grad():
                    v_clipfrac = ((new_values - mb_old_values).abs() > args.clip_coef).float().mean()
            else:
                v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()
                v_clipfrac = torch.tensor(0.0)
            
            # Entropy loss
            entropy_loss = entropy.mean()
            
            # Total loss
            loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
            
            # Track
            total_pg_loss += pg_loss.item()
            total_v_loss += v_loss.item()
            total_entropy += entropy_loss.item()
            total_approx_kl += approx_kl.item()
            total_clipfrac += clipfrac.item()
            total_v_clipfrac += v_clipfrac.item()
            num_updates += 1
        
        # Early stopping on KL
        if args.target_kl is not None and approx_kl > args.target_kl:
            break
    
    return {
        "pg_loss": total_pg_loss / num_updates,
        "v_loss": total_v_loss / num_updates,
        "entropy": total_entropy / num_updates,
        "approx_kl": total_approx_kl / num_updates,
        "clipfrac": total_clipfrac / num_updates,
        "v_clipfrac": total_v_clipfrac / num_updates,
    }


def make_env(args: Args) -> AegisEnv:
    """Create the environment with full config from YAML."""
    if args.game_config:
        config = GameConfig(**args.game_config, seed=args.seed)
    else:
        config = GameConfig(seed=args.seed)
    return AegisEnv(config=config)


def train(args: Args):
    """Main training loop."""
    print(f"Starting CleanRL PPO training")
    print(f"  Total timesteps: {args.total_timesteps:,}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Device: {args.device}")
    if args.checkpoint_timesteps:
        print(f"  Checkpoint mode: every {args.checkpoint_timesteps:,} timesteps")
    else:
        print(f"  Checkpoint mode: every {args.save_interval} updates")
    print()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environment
    env = make_env(args)
    num_agents = len(env.possible_agents)
    
    # Get dimensions
    obs_dim = get_flat_obs_dim(env)
    action_dim = get_action_dim(env)
    
    print(f"Environment: {num_agents} agents")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")
    print()
    
    # Create agent
    agent = ActorCritic(obs_dim, action_dim, args.hidden_dim).to(args.device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # Create rollout buffer
    buffer = RolloutBuffer(
        num_steps=args.num_steps,
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=args.device,
    )
    
    # Training state
    global_step = 0
    num_updates = 0
    start_time = time.time()
    
    # Episode tracking
    all_episode_rewards = []
    all_episode_lengths = []
    
    # Initial reset
    obs_dict, _ = env.reset()
    
    # Checkpoint dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Timestep-based checkpoint tracking
    next_checkpoint_step = args.checkpoint_timesteps if args.checkpoint_timesteps else None
    
    batch_size = args.num_steps * num_agents
    num_total_updates = args.total_timesteps // batch_size
    
    print(f"Batch size: {batch_size}")
    print(f"Total updates: {num_total_updates}")
    print()
    
    while global_step < args.total_timesteps:
        # Collect rollout
        buffer.reset()
        obs_dict, completed_episodes, last_dones = collect_rollout(
            env, agent, buffer, obs_dict, args.device
        )
        
        global_step += batch_size
        num_updates += 1
        
        # Track episodes
        for ep_info in completed_episodes:
            all_episode_rewards.append(ep_info["mean_reward"])
            all_episode_lengths.append(ep_info["length"])
        
        # Compute last value for GAE bootstrap
        obs_flat = []
        for name in env.possible_agents:
            obs_flat.append(flatten_obs(obs_dict[name]))
        obs_tensor = torch.tensor(np.array(obs_flat), dtype=torch.float32, device=args.device)
        
        with torch.no_grad():
            last_value = agent.get_value(obs_tensor).squeeze(-1)
        
        # Compute returns and advantages with correct last_dones
        returns, advantages = buffer.compute_returns_and_advantages(
            last_value, last_dones, args.gamma, args.gae_lambda
        )
        
        # PPO update
        update_stats = ppo_update(agent, optimizer, buffer, returns, advantages, args)
        
        # Logging
        if num_updates % args.log_interval == 0:
            elapsed = time.time() - start_time
            sps = global_step / elapsed
            
            print(f"Update {num_updates:5d} | Step {global_step:8,d} | SPS {sps:6.0f}")
            
            if all_episode_rewards:
                recent_rewards = all_episode_rewards[-100:]
                recent_lengths = all_episode_lengths[-100:]
                print(f"  Episodes: {len(all_episode_rewards):5d} | "
                      f"Mean reward: {np.mean(recent_rewards):7.3f} | "
                      f"Mean length: {np.mean(recent_lengths):6.1f}")
            
            print(f"  PG loss: {update_stats['pg_loss']:7.4f} | "
                  f"V loss: {update_stats['v_loss']:7.4f} | "
                  f"Entropy: {update_stats['entropy']:6.4f} | "
                  f"KL: {update_stats['approx_kl']:6.4f}")
            print()
        
        # Save checkpoint
        should_save = False
        checkpoint_label = ""
        
        if next_checkpoint_step is not None:
            if global_step >= next_checkpoint_step:
                should_save = True
                checkpoint_label = f"step_{global_step}"
                next_checkpoint_step += args.checkpoint_timesteps
        else:
            if num_updates % args.save_interval == 0:
                should_save = True
                checkpoint_label = str(num_updates)
        
        if should_save:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, f"ppo_checkpoint_{checkpoint_label}.pt"
            )
            torch.save({
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "num_updates": num_updates,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Final save
    final_path = os.path.join(args.checkpoint_dir, "ppo_final.pt")
    torch.save({
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "num_updates": num_updates,
    }, final_path)
    print(f"Training complete! Final model saved to {final_path}")
    
    env.close()
    return agent


def main():
    parser = argparse.ArgumentParser(description="CleanRL PPO for AEGIS")
    
    # Config file
    parser.add_argument("-c", "--config", type=str, help="Path to YAML config")
    
    # Environment
    parser.add_argument("--num-survivors", type=int)
    parser.add_argument("--num-impostors", type=int)
    
    # Training
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--num-minibatches", type=int)
    parser.add_argument("--update-epochs", type=int)
    
    # PPO
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--gae-lambda", type=float)
    parser.add_argument("--clip-coef", type=float)
    parser.add_argument("--no-clip-vloss", action="store_true")
    parser.add_argument("--vf-coef", type=float)
    parser.add_argument("--ent-coef", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--target-kl", type=float)
    
    # Architecture
    parser.add_argument("--hidden-dim", type=int)
    
    # Misc
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--checkpoint-timesteps", type=int, help="Save checkpoints by timesteps instead of updates")
    parser.add_argument("--checkpoint-dir", type=str)
    
    cli = parser.parse_args()
    
    # Load config or use defaults
    args = Args.from_yaml(cli.config) if cli.config else Args()
    
    # Override training args
    args = args.override(
        total_timesteps=cli.total_timesteps,
        learning_rate=cli.learning_rate,
        num_steps=cli.num_steps,
        num_minibatches=cli.num_minibatches,
        update_epochs=cli.update_epochs,
        gamma=cli.gamma,
        gae_lambda=cli.gae_lambda,
        clip_coef=cli.clip_coef,
        clip_vloss=False if cli.no_clip_vloss else None,
        vf_coef=cli.vf_coef,
        ent_coef=cli.ent_coef,
        max_grad_norm=cli.max_grad_norm,
        target_kl=cli.target_kl,
        hidden_dim=cli.hidden_dim,
        seed=cli.seed,
        device=cli.device,
        log_interval=cli.log_interval,
        save_interval=cli.save_interval,
        checkpoint_timesteps=cli.checkpoint_timesteps,
        checkpoint_dir=cli.checkpoint_dir,
    )
    
    # Override game_config params via CLI
    if args.game_config and (cli.num_survivors or cli.num_impostors):
        if cli.num_survivors:
            args.game_config["num_survivors"] = cli.num_survivors
        if cli.num_impostors:
            args.game_config["num_impostors"] = cli.num_impostors
    
    train(args)


if __name__ == "__main__":
    main()

