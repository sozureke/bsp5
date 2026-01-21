from __future__ import annotations

import argparse
import json
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
class RoleRewards:
    survivor_win: float = 3.0
    survivor_task_step: float = 0.1
    survivor_team_task_step: float = 0.0
    survivor_death: float = -1.0
    survivor_ejected: float = -1.0
    survivor_ejection_bonus: float = 0.3
    survivor_ejection_penalty: float = -0.3

    impostor_win: float = 0.5
    impostor_kill: float = 0.5
    impostor_kill_witness_penalty: float = -0.3
    impostor_kill_early_penalty: float = -0.2
    impostor_ejected: float = -2.0
    impostor_ejection_bonus: float = 0.3
    impostor_ejection_penalty: float = -0.5
    impostor_proximity_penalty: float = -0.05
    impostor_suspicious_penalty: float = -0.1


@dataclass
class RewardBreakdown:
    task_step: float = 0.0
    team_task_step: float = 0.0  # Optional team-wide task bonus
    kill: float = 0.0
    kill_witness_penalty: float = 0.0
    kill_early_penalty: float = 0.0
    proximity_penalty: float = 0.0
    suspicious_penalty: float = 0.0
    win: float = 0.0
    death: float = 0.0
    ejected: float = 0.0
    ejection_bonus: float = 0.0
    ejection_penalty: float = 0.0
    env_raw: float = 0.0
    
    @property
    def total(self) -> float:
        return (self.task_step + self.team_task_step + self.kill + self.kill_witness_penalty + 
                self.kill_early_penalty + self.proximity_penalty + self.suspicious_penalty +
                self.win + self.death + self.ejected + self.ejection_bonus + self.ejection_penalty)


@dataclass
class EpisodeRewardStats:
    survivor_breakdowns: list = field(default_factory=list)
    impostor_breakdowns: list = field(default_factory=list)
    env_positive_steps_survivor: int = 0
    env_positive_steps_impostor: int = 0
    total_steps: int = 0
    
    def add_step(self, role: int, breakdown: RewardBreakdown, env_reward: float):
        self.total_steps += 1
        if role == 0:
            self.survivor_breakdowns.append(breakdown)
            if env_reward > 0:
                self.env_positive_steps_survivor += 1
        else:
            self.impostor_breakdowns.append(breakdown)
            if env_reward > 0:
                self.env_positive_steps_impostor += 1
    
    def summary(self) -> dict:
        def aggregate(breakdowns: list) -> dict:
            if not breakdowns:
                return {}
            totals = [b.total for b in breakdowns]
            task_steps = [b.task_step for b in breakdowns]
            team_task_steps = [b.team_task_step for b in breakdowns]
            kills = [b.kill for b in breakdowns]
            kill_witness_penalties = [b.kill_witness_penalty for b in breakdowns]
            kill_early_penalties = [b.kill_early_penalty for b in breakdowns]
            proximity_penalties = [b.proximity_penalty for b in breakdowns]
            suspicious_penalties = [b.suspicious_penalty for b in breakdowns]
            wins = [b.win for b in breakdowns]
            deaths = [b.death for b in breakdowns]
            ejected_list = [b.ejected for b in breakdowns]
            ejection_bonus_list = [b.ejection_bonus for b in breakdowns]
            ejection_penalty_list = [b.ejection_penalty for b in breakdowns]
            env_raws = [b.env_raw for b in breakdowns]
            return {
                "total": {"sum": sum(totals), "mean": np.mean(totals)},
                "task_step": {"sum": sum(task_steps)},
                "team_task_step": {"sum": sum(team_task_steps)},
                "kill": {"sum": sum(kills)},
                "kill_witness_penalty": {"sum": sum(kill_witness_penalties)},
                "kill_early_penalty": {"sum": sum(kill_early_penalties)},
                "proximity_penalty": {"sum": sum(proximity_penalties)},
                "suspicious_penalty": {"sum": sum(suspicious_penalties)},
                "win": {"sum": sum(wins)},
                "death": {"sum": sum(deaths)},
                "ejected": {"sum": sum(ejected_list)},
                "ejection_bonus": {"sum": sum(ejection_bonus_list)},
                "ejection_penalty": {"sum": sum(ejection_penalty_list)},
                "env_raw": {"sum": sum(env_raws), "mean": np.mean(env_raws)},
            }
        
        return {
            "survivor": aggregate(self.survivor_breakdowns),
            "impostor": aggregate(self.impostor_breakdowns),
            "env_positive_steps_survivor": self.env_positive_steps_survivor,
            "env_positive_steps_impostor": self.env_positive_steps_impostor,
            "total_steps": self.total_steps,
        }


@dataclass
class Args:
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    num_steps: int = 128
    num_minibatches: int = 4
    update_epochs: int = 4
    
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    
    hidden_dim: int = 256
    
    seed: int = 1
    device: str = "cpu"
    log_interval: int = 10
    save_interval: int = 100
    checkpoint_timesteps: Optional[int] = None
    checkpoint_dir: str = "./checkpoints"
    
    debug_rewards: bool = False
    debug_rewards_interval: int = 10
    
    game_config: Optional[dict] = None
    role_rewards: Optional[RoleRewards] = None

    @classmethod
    def from_yaml(cls, path: str) -> "Args":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        
        env = cfg.get("env", {})
        rewards = cfg.get("rewards", {})
        training = cfg.get("training", {})
        logging_cfg = cfg.get("logging", {})
        model = training.get("model", {})        
        survivor_rewards = rewards.get("survivor", {})
        impostor_rewards = rewards.get("impostor", {})
        
        role_rewards = RoleRewards(
            survivor_win=survivor_rewards.get("win", 3.0),
            survivor_task_step=survivor_rewards.get("task_step", 0.1),
            survivor_team_task_step=survivor_rewards.get("team_task_step", 0.0),
            survivor_death=survivor_rewards.get("death", -1.0),
            survivor_ejected=survivor_rewards.get("ejected", -1.0),
            survivor_ejection_bonus=survivor_rewards.get("ejection_bonus", 0.3),
            survivor_ejection_penalty=survivor_rewards.get("ejection_penalty", -0.3),
            impostor_win=impostor_rewards.get("win", 0.5),  # Reduced default
            impostor_kill=impostor_rewards.get("kill", 0.5),
            impostor_kill_witness_penalty=impostor_rewards.get("kill_witness_penalty", -0.3),
            impostor_kill_early_penalty=impostor_rewards.get("kill_early_penalty", -0.2),
            impostor_ejected=impostor_rewards.get("ejected", -2.0),
            impostor_ejection_bonus=impostor_rewards.get("ejection_bonus", 0.3),
            impostor_ejection_penalty=impostor_rewards.get("ejection_penalty", -0.5),
            impostor_proximity_penalty=impostor_rewards.get("proximity_penalty", -0.05),
            impostor_suspicious_penalty=impostor_rewards.get("suspicious_penalty", -0.1),
        )
        
        hidden = model.get("fcnet_hiddens", [256])
        device = training.get("device", "cpu")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
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
            log_interval=logging_cfg.get("log_interval", 10),
            save_interval=training.get("checkpoint_freq", 100),
            checkpoint_timesteps=training.get("checkpoint_timesteps"),
            checkpoint_dir=logging_cfg.get("checkpoint_dir", "./checkpoints"),
            debug_rewards=logging_cfg.get("debug_rewards", False),
            debug_rewards_interval=logging_cfg.get("debug_rewards_interval", 10),
            game_config=game_config,
            role_rewards=role_rewards,
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
    Supports action masking and role-aware actor/critic heads.
    
    Architecture:
        - Shared encoder for all agents/roles
        - Separate actor heads per role (survivor vs impostor)
        - Separate critic heads per role (survivor vs impostor value functions)
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.action_dim = action_dim
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Separate actor heads per role
        self.actor_survivor = nn.Linear(hidden_dim, action_dim)  # role == 0
        self.actor_impostor = nn.Linear(hidden_dim, action_dim)  # role == 1
        
        # Separate critic heads per role (different MDPs)
        self.critic_survivor = nn.Linear(hidden_dim, 1)  # role == 0
        self.critic_impostor = nn.Linear(hidden_dim, 1)  # role == 1
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Smaller init for output heads
        nn.init.orthogonal_(self.actor_survivor.weight, gain=0.01)
        nn.init.orthogonal_(self.actor_impostor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_survivor.weight, gain=1.0)
        nn.init.orthogonal_(self.critic_impostor.weight, gain=1.0)
    
    def get_value(self, obs: torch.Tensor, roles: torch.Tensor) -> torch.Tensor:
        """
        Get value estimates for observations with role-specific critics.
        
        Args:
            obs: Observations tensor (batch, obs_dim)
            roles: Role for each sample (batch,) - 0=survivor, 1=impostor
            
        Returns:
            values: Value estimates (batch,)
        """
        features = self.shared(obs)
        values_survivor = self.critic_survivor(features).squeeze(-1)
        values_impostor = self.critic_impostor(features).squeeze(-1)
        
        # Select value based on role
        values = torch.where(roles == 0, values_survivor, values_impostor)
        return values
    
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        roles: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            action: Selected action
            log_prob: Log probability of action
            entropy: Distribution entropy
            value: Value estimate (role-specific)
            
        Args:
            obs: Observations tensor (batch, obs_dim)
            action_mask: Valid action mask (batch, action_dim)
            roles: Role for each sample (batch,) - 0=survivor, 1=impostor
            action: Optional pre-selected actions for log_prob computation
        """
        features = self.shared(obs)
        
        # Get role-specific value estimates
        values_survivor = self.critic_survivor(features).squeeze(-1)
        values_impostor = self.critic_impostor(features).squeeze(-1)
        value = torch.where(roles == 0, values_survivor, values_impostor)
        
        # Compute logits from both actor heads
        logits_survivor = self.actor_survivor(features)  # (batch, action_dim)
        logits_impostor = self.actor_impostor(features)  # (batch, action_dim)
        
        # Select logits based on role per sample
        # roles: (batch,) -> expand to (batch, action_dim) for gathering
        role_mask = roles.unsqueeze(-1).expand(-1, self.action_dim)  # (batch, action_dim)
        
        # role == 0 -> survivor, role == 1 -> impostor
        logits = torch.where(role_mask == 0, logits_survivor, logits_impostor)
        
        dist = CategoricalMasked(logits=logits, mask=action_mask)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value


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
        self.roles = torch.zeros((num_steps, num_agents), dtype=torch.long, device=device)
        
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
        roles: torch.Tensor,
    ):
        """Add one step of data."""
        self.obs[self.step] = obs
        self.actions[self.step] = actions
        self.logprobs[self.step] = logprobs
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.values[self.step] = values
        self.action_masks[self.step] = action_masks
        self.roles[self.step] = roles
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
        b_roles = self.roles.reshape(batch_size)
        
        return b_obs, b_actions, b_logprobs, b_action_masks, b_values, b_roles


def compute_role_shaped_rewards(
    agent_names: list[str],
    obs_dict: dict,
    next_obs_dict: dict,
    env_rewards: dict[str, float],
    infos: dict[str, dict],
    role_rewards: Optional[RoleRewards],
    terminated: bool,
) -> tuple[dict[str, float], dict[str, RewardBreakdown]]:
    if role_rewards is None:
        role_rewards = RoleRewards()
    
    shaped_rewards = {}
    breakdowns = {}
    
    first_agent_info = infos[agent_names[0]]
    authoritative_winner = first_agent_info.get("winner")
    
    for name in agent_names:
        role = obs_dict[name]["self_role"]
        was_alive = obs_dict[name]["self_alive"]
        is_alive = next_obs_dict[name]["self_alive"]
        env_reward = env_rewards[name]
        info = infos[name]
        
        breakdown = RewardBreakdown(env_raw=env_reward)
        reward = 0.0
        
        events_present = "events" in info and isinstance(info.get("events"), list)
        if not events_present:
            raise RuntimeError(
                f"Missing events in infos for agent {name}; reward shaping requires event tags for correctness. "
                "Check that environment is properly setting event tags in infos."
            )
        
        if role == 0:  # SURVIVOR
            # Individual task step bonus (for the agent who completed the task)
            if info.get("did_task_step", False):
                breakdown.task_step = role_rewards.survivor_task_step
                reward += breakdown.task_step
            
            # Optional team-wide task step bonus (default 0.0, disabled)
            if role_rewards.survivor_team_task_step != 0.0 and info.get("team_task_step", False):
                breakdown.team_task_step = role_rewards.survivor_team_task_step
                reward += breakdown.team_task_step
            
            if info.get("ejection_benefited", False):
                breakdown.ejection_bonus = role_rewards.survivor_ejection_bonus
                reward += breakdown.ejection_bonus
            
            if info.get("ejection_penalized", False):
                breakdown.ejection_penalty = role_rewards.survivor_ejection_penalty
                reward += breakdown.ejection_penalty
            
            # Survivor elimination penalties
            if was_alive and not is_alive:
                if info.get("was_killed", False):
                    # Survivor was killed by impostor
                    breakdown.death = role_rewards.survivor_death
                    reward += breakdown.death
                elif info.get("was_ejected", False):
                    # Survivor was voted out
                    breakdown.ejected = role_rewards.survivor_ejected
                    reward += breakdown.ejected
                else:
                    # Safety check: survivor died but neither was_killed nor was_ejected is set
                    raise RuntimeError(
                        f"Survivor {name} died but neither was_killed nor was_ejected is True. "
                        "Check environment event generation."
                    )
            
            if terminated:
                # B1: Winner type guard - ensure winner is int (belt + suspenders)
                winner = authoritative_winner if authoritative_winner is not None else info.get("winner")
                if winner is not None:
                    winner = int(winner)
                if winner == 0 and is_alive:
                    breakdown.win = role_rewards.survivor_win
                    reward += breakdown.win
        
        else:  # IMPOSTOR
            # Context-dependent kill rewards
            if info.get("did_kill", False):
                breakdown.kill = role_rewards.impostor_kill
                reward += breakdown.kill
                visible_agents = obs_dict[name]["visible_agents"]
                self_room = obs_dict[name]["self_room"]
                
                # Get agent_id from name (agent_0 -> 0, agent_1 -> 1, etc.)
                agent_id = int(name.split("_")[1])
                
                num_witnesses = 0
                for i in range(len(visible_agents)):
                    if i == agent_id:
                        continue  # Skip self
                    seen, room, alive = visible_agents[i]
                    if seen == 1 and room == self_room and alive == 1:
                        num_witnesses += 1
                
                # num_witnesses includes the victim, so subtract 1
                num_witnesses = max(0, num_witnesses - 1)
                
                if num_witnesses > 0:
                    witness_penalty = role_rewards.impostor_kill_witness_penalty * num_witnesses
                    breakdown.kill_witness_penalty = witness_penalty
                    reward += witness_penalty
                
                # Early kill penalty: penalize kills in first 20% of episode
                # Use tick from observation to determine episode progress
                tick = next_obs_dict[name]["tick"][0]
                max_ticks = 400  # Default max_ticks, could be made configurable
                episode_progress = tick / max_ticks if max_ticks > 0 else 0.0
                
                if episode_progress < 0.2:
                    breakdown.kill_early_penalty = role_rewards.impostor_kill_early_penalty
                    reward += breakdown.kill_early_penalty
            
            # Intermediate penalties: proximity to multiple survivors
            if is_alive:
                visible_agents = next_obs_dict[name]["visible_agents"]
                self_room = next_obs_dict[name]["self_room"]
                agent_id = int(name.split("_")[1])
                
                # Count other alive agents in same room (potential survivors)
                # We can't directly know roles, but being near multiple agents is risky
                agents_in_room = 0
                for i in range(len(visible_agents)):
                    if i == agent_id:
                        continue  # Skip self
                    seen, room, alive = visible_agents[i]
                    if seen == 1 and room == self_room and alive == 1:
                        agents_in_room += 1
                
                # Penalty if near 2+ other agents (high risk of being caught)
                if agents_in_room >= 2:
                    breakdown.proximity_penalty = role_rewards.impostor_proximity_penalty
                    reward += breakdown.proximity_penalty
                
                # Suspicious behavior: being near a fresh body (could indicate recent kill)
                visible_bodies = next_obs_dict[name]["visible_bodies"]
                for body in visible_bodies:
                    seen, room, age = body
                    if seen == 1 and room == self_room and 0 <= age < 10:
                        # Fresh body nearby - suspicious behavior
                        breakdown.suspicious_penalty = role_rewards.impostor_suspicious_penalty
                        reward += breakdown.suspicious_penalty
                        break  # Only penalize once per step
            
            if info.get("ejection_benefited", False):
                breakdown.ejection_bonus = role_rewards.impostor_ejection_bonus
                reward += breakdown.ejection_bonus
            
            if info.get("ejection_penalized", False):
                breakdown.ejection_penalty = role_rewards.impostor_ejection_penalty
                reward += breakdown.ejection_penalty
            
            # Impostor elimination: impostors can only leave via voting (EJECTION)
            if was_alive and not is_alive:
                # Safety assertion: impostors can only be ejected, never killed
                if not info.get("was_ejected", False):
                    raise RuntimeError(
                        f"Invariant violated: impostor {name} died without was_ejected=True. "
                        "Impostors can only be ejected, never killed. Check environment event generation."
                    )
                breakdown.ejected = role_rewards.impostor_ejected
                reward += breakdown.ejected
            
            if terminated:
                # B1: Winner type guard - ensure winner is int (belt + suspenders)
                winner = authoritative_winner if authoritative_winner is not None else info.get("winner")
                if winner is not None:
                    winner = int(winner)
                if winner == 1 and is_alive:
                    breakdown.win = role_rewards.impostor_win
                    reward += breakdown.win
        
        shaped_rewards[name] = reward
        breakdowns[name] = breakdown
    
    return shaped_rewards, breakdowns


def collect_rollout(
    env: AegisEnv,
    agent: ActorCritic,
    buffer: RolloutBuffer,
    obs_dict: dict,
    device: str,
    role_rewards: Optional[RoleRewards] = None,
    debug_rewards: bool = False,
) -> tuple[dict, list, torch.Tensor, list]:
    agent_names = env.possible_agents
    num_agents = len(agent_names)
    
    episode_rewards = {name: 0.0 for name in agent_names}
    episode_lengths = 0
    completed_episodes = []
    episode_stats = EpisodeRewardStats() if debug_rewards else None
    all_episode_stats = []
    
    last_dones = torch.zeros(num_agents, dtype=torch.float32, device=device)
    
    for step in range(buffer.num_steps):
        obs_flat = []
        masks = []
        roles = []
        for name in agent_names:
            obs_flat.append(flatten_obs(obs_dict[name]))
            masks.append(obs_dict[name]["action_mask"])
            roles.append(obs_dict[name]["self_role"])
        
        obs_tensor = torch.tensor(np.array(obs_flat), dtype=torch.float32, device=device)
        mask_tensor = torch.tensor(np.array(masks), dtype=torch.float32, device=device)
        roles_tensor = torch.tensor(np.array(roles), dtype=torch.long, device=device)
        
        with torch.no_grad():
            actions, logprobs, _, values = agent.get_action_and_value(
                obs_tensor, mask_tensor, roles_tensor
            )
        
        actions_dict = {name: actions[i].item() for i, name in enumerate(agent_names)}
        next_obs_dict, rewards_dict, terminations, truncations, infos = env.step(actions_dict)
        
        terminated = any(terminations.values()) or any(truncations.values())
        
        shaped_rewards, breakdowns = compute_role_shaped_rewards(
            agent_names=agent_names,
            obs_dict=obs_dict,
            next_obs_dict=next_obs_dict,
            env_rewards=rewards_dict,
            infos=infos,
            role_rewards=role_rewards,
            terminated=terminated,
        )
        
        if debug_rewards and episode_stats is not None:
            for name in agent_names:
                role = obs_dict[name]["self_role"]
                episode_stats.add_step(role, breakdowns[name], rewards_dict[name])
        
        rewards = torch.tensor(
            [shaped_rewards[name] for name in agent_names],
            dtype=torch.float32,
            device=device,
        )
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
            roles=roles_tensor,
        )
        
        for i, name in enumerate(agent_names):
            episode_rewards[name] += shaped_rewards[name]
        episode_lengths += 1
        
        if terminated:
            survivor_rewards = []
            impostor_rewards = []
            impostor_ejected = False
            for name in agent_names:
                role = obs_dict[name]["self_role"]
                if role == 0:
                    survivor_rewards.append(episode_rewards[name])
                else:
                    impostor_rewards.append(episode_rewards[name])
                    # Check if this impostor was ejected
                    if infos[name].get("was_ejected", False):
                        impostor_ejected = True
            
            completed_episodes.append({
                "total_reward": sum(episode_rewards.values()),
                "mean_reward": np.mean(list(episode_rewards.values())),
                "length": episode_lengths,
                "winner": infos[agent_names[0]].get("winner"),
                "survivor_mean_reward": np.mean(survivor_rewards) if survivor_rewards else 0.0,
                "impostor_mean_reward": np.mean(impostor_rewards) if impostor_rewards else 0.0,
                "impostor_ejected": impostor_ejected,
            })
            
            if debug_rewards and episode_stats is not None:
                all_episode_stats.append(episode_stats.summary())
                episode_stats = EpisodeRewardStats()
            
            obs_dict, _ = env.reset()
            episode_rewards = {name: 0.0 for name in agent_names}
            episode_lengths = 0
            last_dones = torch.zeros(num_agents, dtype=torch.float32, device=device)
        else:
            obs_dict = next_obs_dict
            last_dones = dones
    
    return obs_dict, completed_episodes, last_dones, all_episode_stats


def print_reward_debug(episode_stats: list, episode_num: int):
    if not episode_stats:
        return
    
    survivor_totals = []
    impostor_totals = []
    survivor_components = {"task_step": 0, "team_task_step": 0, "win": 0, "death": 0, "ejected": 0, "ejection_bonus": 0, "ejection_penalty": 0}
    impostor_components = {"kill": 0, "kill_witness_penalty": 0, "kill_early_penalty": 0, "proximity_penalty": 0, "suspicious_penalty": 0, "win": 0, "ejected": 0, "ejection_bonus": 0, "ejection_penalty": 0}
    env_pos_survivor = 0
    env_pos_impostor = 0
    
    for stats in episode_stats:
        if stats["survivor"]:
            survivor_totals.append(stats["survivor"]["total"]["sum"])
            for k in survivor_components:
                survivor_components[k] += stats["survivor"].get(k, {}).get("sum", 0)
        if stats["impostor"]:
            impostor_totals.append(stats["impostor"]["total"]["sum"])
            for k in impostor_components:
                impostor_components[k] += stats["impostor"].get(k, {}).get("sum", 0)
        env_pos_survivor += stats.get("env_positive_steps_survivor", 0)
        env_pos_impostor += stats.get("env_positive_steps_impostor", 0)
    
    print(f"\n=== REWARD DEBUG (last {len(episode_stats)} episodes) ===")
    
    if survivor_totals:
        print(f"SURVIVOR: total={sum(survivor_totals):.2f} mean={np.mean(survivor_totals):.3f} "
              f"min={min(survivor_totals):.2f} max={max(survivor_totals):.2f}")
        print(f"task_step={survivor_components['task_step']:.2f} "
              f"team_task_step={survivor_components['team_task_step']:.2f} "
              f"win={survivor_components['win']:.2f} "
              f"death={survivor_components['death']:.2f} "
              f"ejected={survivor_components['ejected']:.2f} "
              f"ej_bonus={survivor_components['ejection_bonus']:.2f} "
              f"ej_penalty={survivor_components['ejection_penalty']:.2f}")
        print(f"  env_reward>0 steps: {env_pos_survivor}")
    
    if impostor_totals:
        print(f"IMPOSTOR: total={sum(impostor_totals):.2f} mean={np.mean(impostor_totals):.3f} "
              f"min={min(impostor_totals):.2f} max={max(impostor_totals):.2f}")
        print(f"  kill={impostor_components['kill']:.2f} "
              f"kill_witness_pen={impostor_components['kill_witness_penalty']:.2f} "
              f"kill_early_pen={impostor_components['kill_early_penalty']:.2f} "
              f"proximity_pen={impostor_components['proximity_penalty']:.2f} "
              f"suspicious_pen={impostor_components['suspicious_penalty']:.2f} "
              f"win={impostor_components['win']:.2f} "
              f"ejected={impostor_components['ejected']:.2f} "
              f"ej_bonus={impostor_components['ejection_bonus']:.2f} "
              f"ej_penalty={impostor_components['ejection_penalty']:.2f}")
        print(f"  env_reward>0 steps: {env_pos_impostor}")
    print("=" * 50)


def ppo_update(
    agent: ActorCritic,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    args: Args,
) -> dict:
    """Perform PPO update on collected data."""
    b_obs, b_actions, b_old_logprobs, b_action_masks, b_old_values, b_roles = buffer.flatten()
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
    total_survivor_value = 0.0
    total_impostor_value = 0.0
    num_survivor_samples = 0
    num_impostor_samples = 0
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
            mb_roles = b_roles[mb_indices]
            
            # Forward pass (with roles for role-aware actor selection)
            _, new_logprobs, entropy, new_values = agent.get_action_and_value(
                mb_obs, mb_action_masks, mb_roles, mb_actions
            )
            
            # Track mean values per critic for logging
            with torch.no_grad():
                survivor_mask = (mb_roles == 0)
                impostor_mask = (mb_roles == 1)
                if survivor_mask.any():
                    total_survivor_value += new_values[survivor_mask].mean().item()
                    num_survivor_samples += 1
                if impostor_mask.any():
                    total_impostor_value += new_values[impostor_mask].mean().item()
                    num_impostor_samples += 1
            
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
        "mean_survivor_value": total_survivor_value / num_survivor_samples if num_survivor_samples > 0 else 0.0,
        "mean_impostor_value": total_impostor_value / num_impostor_samples if num_impostor_samples > 0 else 0.0,
    }


def make_env(args: Args) -> AegisEnv:
    """Create the environment with full config from YAML."""
    if args.game_config:
        config = GameConfig(**args.game_config, seed=args.seed)
    else:
        config = GameConfig(seed=args.seed)
    return AegisEnv(config=config)


def train(args: Args):
    print(f"Starting CleanRL PPO training")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {args.device}")
    if args.checkpoint_timesteps:
        print(f"Checkpoint mode: every {args.checkpoint_timesteps:,} timesteps")
    else:
        print(f"Checkpoint mode: every {args.save_interval} updates")
    
    if args.debug_rewards:
        print(f"Debug rewards: ENABLED (interval={args.debug_rewards_interval})")
    
    if args.role_rewards:
        rr = args.role_rewards
        print(f"Role-aware rewards enabled:")
        print(f"  Survivor: win={rr.survivor_win}, task_step={rr.survivor_task_step}, "
              f"team_task_step={rr.survivor_team_task_step}, "
              f"death={rr.survivor_death}, ejected={rr.survivor_ejected}, "
              f"ej_bonus={rr.survivor_ejection_bonus}, ej_pen={rr.survivor_ejection_penalty}")
        print(f"  Impostor: win={rr.impostor_win}, kill={rr.impostor_kill}, "
              f"kill_witness_penalty={rr.impostor_kill_witness_penalty}, "
              f"kill_early_penalty={rr.impostor_kill_early_penalty}, "
              f"proximity_penalty={rr.impostor_proximity_penalty}, "
              f"suspicious_penalty={rr.impostor_suspicious_penalty}, "
              f"ejected={rr.impostor_ejected}, ej_bonus={rr.impostor_ejection_bonus}, ej_pen={rr.impostor_ejection_penalty}")
    print()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    env = make_env(args)
    num_agents = len(env.possible_agents)
    
    obs_dim = get_flat_obs_dim(env)
    action_dim = get_action_dim(env)
    
    print(f"Environment: {num_agents} agents")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")
    print()
    
    agent = ActorCritic(obs_dim, action_dim, args.hidden_dim).to(args.device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    buffer = RolloutBuffer(
        num_steps=args.num_steps,
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=args.device,
    )
    
    global_step = 0
    num_updates = 0
    start_time = time.time()
    
    all_episode_rewards = []
    all_episode_lengths = []
    all_survivor_rewards = []
    all_impostor_rewards = []
    all_debug_stats = []
    
    # Track episode outcomes for win/ejection rates
    all_winners = []  # List of winner values (0=survivor, 1=impostor)
    all_impostor_ejected = []  # List of booleans indicating if impostor was ejected
    
    obs_dict, _ = env.reset()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Setup JSON logging file
    log_file_path = os.path.join(args.checkpoint_dir, "training_log.json")
    log_file = open(log_file_path, "w")
    print(f"Structured logging enabled: {log_file_path}")
    
    next_checkpoint_step = args.checkpoint_timesteps if args.checkpoint_timesteps else None
    
    batch_size = args.num_steps * num_agents
    num_total_updates = args.total_timesteps // batch_size
    
    print(f"Batch size: {batch_size}")
    print(f"Total updates: {num_total_updates}")
    print()
    
    try:
        while global_step < args.total_timesteps:
            buffer.reset()
            obs_dict, completed_episodes, last_dones, episode_debug_stats = collect_rollout(
                env, agent, buffer, obs_dict, args.device, args.role_rewards, args.debug_rewards
            )
            
            global_step += batch_size
            num_updates += 1
            
            for ep_info in completed_episodes:
                all_episode_rewards.append(ep_info["mean_reward"])
                all_episode_lengths.append(ep_info["length"])
                if "survivor_mean_reward" in ep_info:
                    all_survivor_rewards.append(ep_info["survivor_mean_reward"])
                if "impostor_mean_reward" in ep_info:
                    all_impostor_rewards.append(ep_info["impostor_mean_reward"])
                # Track winners and ejections
                winner = ep_info.get("winner")
                if winner is not None:
                    all_winners.append(int(winner))
                if "impostor_ejected" in ep_info:
                    all_impostor_ejected.append(ep_info["impostor_ejected"])
            
            if args.debug_rewards:
                all_debug_stats.extend(episode_debug_stats)
            
            obs_flat = []
            roles_list = []
            for name in env.possible_agents:
                obs_flat.append(flatten_obs(obs_dict[name]))
                roles_list.append(obs_dict[name]["self_role"])
            obs_tensor = torch.tensor(np.array(obs_flat), dtype=torch.float32, device=args.device)
            roles_tensor = torch.tensor(np.array(roles_list), dtype=torch.long, device=args.device)
            
            with torch.no_grad():
                last_value = agent.get_value(obs_tensor, roles_tensor)
            
            returns, advantages = buffer.compute_returns_and_advantages(
                last_value, last_dones, args.gamma, args.gae_lambda
            )
            
            update_stats = ppo_update(agent, optimizer, buffer, returns, advantages, args)
            
            if num_updates % args.log_interval == 0:
                elapsed = time.time() - start_time
                sps = global_step / elapsed
                
                print(f"Update {num_updates:5d} | Step {global_step:8,d} | SPS {sps:6.0f}")
                
                recent_rewards = all_episode_rewards[-100:] if all_episode_rewards else []
                recent_lengths = all_episode_lengths[-100:] if all_episode_lengths else []
                recent_survivor = all_survivor_rewards[-100:] if all_survivor_rewards else []
                recent_impostor = all_impostor_rewards[-100:] if all_impostor_rewards else []
                recent_winners = all_winners[-100:] if all_winners else []
                recent_ejected = all_impostor_ejected[-100:] if all_impostor_ejected else []
                
                mean_reward = np.mean(recent_rewards) if recent_rewards else 0.0
                mean_ep_len = np.mean(recent_lengths) if recent_lengths else 0.0
                survivor_reward_mean = np.mean(recent_survivor) if recent_survivor else 0.0
                impostor_reward_mean = np.mean(recent_impostor) if recent_impostor else 0.0
                
                # Compute win/ejection rates
                impostor_win_rate = 0.0
                if recent_winners:
                    impostor_wins = sum(1 for w in recent_winners if w == 1)
                    impostor_win_rate = impostor_wins / len(recent_winners)
                
                impostor_ejected_rate = 0.0
                if recent_ejected:
                    impostor_ejected_rate = sum(recent_ejected) / len(recent_ejected)
                
                if all_episode_rewards:
                    print(f"Episodes: {len(all_episode_rewards):5d} | "
                          f"Mean reward: {mean_reward:7.3f} | "
                          f"Mean length: {mean_ep_len:6.1f}")
                    
                    if recent_survivor or recent_impostor:
                        print(f"Role rewards: Survivor {survivor_reward_mean:7.3f} | Impostor {impostor_reward_mean:7.3f}")
                        if recent_impostor:
                            imp_min, imp_max = min(recent_impostor), max(recent_impostor)
                            print(f"Impostor range: [{imp_min:.2f}, {imp_max:.2f}]")
                
                print(f"  PG loss: {update_stats['pg_loss']:7.4f} | "
                      f"V loss: {update_stats['v_loss']:7.4f} | "
                      f"Entropy: {update_stats['entropy']:6.4f} | "
                      f"KL: {update_stats['approx_kl']:6.4f}")
                print(f"  Critic values: Survivor {update_stats['mean_survivor_value']:7.3f} | "
                      f"Impostor {update_stats['mean_impostor_value']:7.3f}")
                print()
                
                # Write structured log entry (JSONL format)
                log_entry = {
                    "step": int(global_step),
                    "update": int(num_updates),
                    "mean_reward": float(mean_reward),
                    "mean_ep_len": float(mean_ep_len),
                    "survivor_reward": float(survivor_reward_mean),
                    "impostor_reward": float(impostor_reward_mean),
                    "impostor_win_rate": float(impostor_win_rate),
                    "impostor_ejected_rate": float(impostor_ejected_rate),
                    "entropy": float(update_stats['entropy']),
                    "value_loss": float(update_stats['v_loss']),
                    "policy_loss": float(update_stats['pg_loss']),
                }
                log_file.write(json.dumps(log_entry) + "\n")
                log_file.flush()
            
            if args.debug_rewards and num_updates % args.debug_rewards_interval == 0 and all_debug_stats:
                print_reward_debug(all_debug_stats[-50:], len(all_episode_rewards))
                all_debug_stats = []
            
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
    finally:
        log_file.close()
        print(f"Training log saved to {log_file_path}")
    
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
    
    parser.add_argument("-c", "--config", type=str, help="Path to YAML config")
    
    parser.add_argument("--num-survivors", type=int)
    parser.add_argument("--num-impostors", type=int)
    
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--num-minibatches", type=int)
    parser.add_argument("--update-epochs", type=int)
    
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--gae-lambda", type=float)
    parser.add_argument("--clip-coef", type=float)
    parser.add_argument("--no-clip-vloss", action="store_true")
    parser.add_argument("--vf-coef", type=float)
    parser.add_argument("--ent-coef", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--target-kl", type=float)
    
    parser.add_argument("--hidden-dim", type=int)
    
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--checkpoint-timesteps", type=int, help="Save checkpoints by timesteps instead of updates")
    parser.add_argument("--checkpoint-dir", type=str)
    
    parser.add_argument("--debug-rewards", action="store_true", help="Enable detailed reward breakdown logging")
    parser.add_argument("--debug-rewards-interval", type=int, default=10, help="Print reward debug every N updates")
    
    cli = parser.parse_args()
    
    args = Args.from_yaml(cli.config) if cli.config else Args()
    
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
        debug_rewards=cli.debug_rewards if cli.debug_rewards else None,
        debug_rewards_interval=cli.debug_rewards_interval if cli.debug_rewards_interval != 10 else None,
    )
    
    if args.game_config and (cli.num_survivors or cli.num_impostors):
        if cli.num_survivors:
            args.game_config["num_survivors"] = cli.num_survivors
        if cli.num_impostors:
            args.game_config["num_impostors"] = cli.num_impostors
    
    train(args)


if __name__ == "__main__":
    main()

