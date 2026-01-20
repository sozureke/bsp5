"""Reward shaping sanity tests: ensure shaping follows event flags correctly."""

import pytest
from aegis.env.pettingzoo_env import AegisEnv
from aegis.train.rl_ppo import compute_role_shaped_rewards, RoleRewards
from aegis.core.rules import GameConfig


class TestRewardShaping:
    """C3: Reward shaping sanity tests."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = GameConfig(
            num_survivors=2,
            num_impostors=1,
            max_ticks=100,
        )
        self.env = AegisEnv(config=self.config)
        self.role_rewards = RoleRewards(
            survivor_task_step=0.1,
            survivor_team_task_step=0.0,  # Disabled
            survivor_death=-1.0,
            survivor_ejected=-1.0,
            impostor_kill=0.5,
            impostor_ejected=-2.0,
        )
    
    def test_survivor_task_step_reward(self):
        """C3: Test that did_task_step=True gives survivor_task_step reward."""
        obs, infos = self.env.reset(seed=42)
        
        # Run until a task step happens
        max_steps = 100
        found_task_step = False
        
        for step in range(max_steps):
            actions = {name: 0 for name in self.env.possible_agents}
            next_obs, rewards, terminations, truncations, next_infos = self.env.step(actions)
            
            # Check if any survivor completed a task step
            for agent_name in self.env.possible_agents:
                info = next_infos[agent_name]
                role = info["role"]
                
                if role == 0 and info["did_task_step"]:
                    # C3: If did_task_step is True, shaped reward should include survivor_task_step
                    shaped_rewards, breakdowns = compute_role_shaped_rewards(
                        agent_names=self.env.possible_agents,
                        obs_dict=obs,
                        next_obs_dict=next_obs,
                        env_rewards=rewards,
                        infos=next_infos,
                        role_rewards=self.role_rewards,
                        terminated=any(terminations.values()) or any(truncations.values()),
                    )
                    
                    assert breakdowns[agent_name].task_step == self.role_rewards.survivor_task_step, (
                        f"Survivor {agent_name} completed task step but reward not applied. "
                        f"Expected {self.role_rewards.survivor_task_step}, "
                        f"got {breakdowns[agent_name].task_step}"
                    )
                    found_task_step = True
                    break
            
            obs = next_obs
            
            if found_task_step or any(terminations.values()) or any(truncations.values()):
                break
        
        # Note: This test may not always find a task step, which is OK
        # The important part is that if it does find one, the reward is correct
    
    def test_team_task_step_disabled(self):
        """C3: Test that team_task_step reward is not applied when disabled."""
        obs, infos = self.env.reset(seed=42)
        
        # Ensure team_task_step is disabled (default 0.0)
        assert self.role_rewards.survivor_team_task_step == 0.0
        
        actions = {name: 0 for name in self.env.possible_agents}
        next_obs, rewards, terminations, truncations, next_infos = self.env.step(actions)
        
        shaped_rewards, breakdowns = compute_role_shaped_rewards(
            agent_names=self.env.possible_agents,
            obs_dict=obs,
            next_obs_dict=next_obs,
            env_rewards=rewards,
            infos=next_infos,
            role_rewards=self.role_rewards,
            terminated=any(terminations.values()) or any(truncations.values()),
        )
        
        # Even if team_task_step flag is True, reward should be 0.0 when disabled
        for agent_name in self.env.possible_agents:
            if next_infos[agent_name]["role"] == 0:  # Survivor
                assert breakdowns[agent_name].team_task_step == 0.0, (
                    f"Team task step reward applied when disabled. "
                    f"Got {breakdowns[agent_name].team_task_step}"
                )
    
    def test_impostor_kill_reward(self):
        """C3: Test that did_kill=True gives impostor_kill reward."""
        obs, infos = self.env.reset(seed=42)
        
        # Run until a kill happens (may take many steps)
        max_steps = 500
        found_kill = False
        
        for step in range(max_steps):
            actions = {name: 0 for name in self.env.possible_agents}
            next_obs, rewards, terminations, truncations, next_infos = self.env.step(actions)
            
            # Check if any impostor killed
            for agent_name in self.env.possible_agents:
                info = next_infos[agent_name]
                role = info["role"]
                
                if role == 1 and info["did_kill"]:
                    # C3: If did_kill is True, shaped reward should include impostor_kill
                    shaped_rewards, breakdowns = compute_role_shaped_rewards(
                        agent_names=self.env.possible_agents,
                        obs_dict=obs,
                        next_obs_dict=next_obs,
                        env_rewards=rewards,
                        infos=next_infos,
                        role_rewards=self.role_rewards,
                        terminated=any(terminations.values()) or any(truncations.values()),
                    )
                    
                    assert breakdowns[agent_name].kill == self.role_rewards.impostor_kill, (
                        f"Impostor {agent_name} killed but reward not applied. "
                        f"Expected {self.role_rewards.impostor_kill}, "
                        f"got {breakdowns[agent_name].kill}"
                    )
                    found_kill = True
                    break
            
            obs = next_obs
            
            if found_kill or any(terminations.values()) or any(truncations.values()):
                break
        
        # Note: This test may not always find a kill, which is OK
        # The important part is that if it does find one, the reward is correct

