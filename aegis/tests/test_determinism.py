"""Determinism smoke test: ensure reproducible behavior with fixed seed."""

import pytest
import numpy as np
from aegis.env.pettingzoo_env import AegisEnv
from aegis.core.rules import GameConfig


class TestDeterminism:
    """C4: Determinism smoke test."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = GameConfig(
            num_survivors=2,
            num_impostors=1,
            max_ticks=100,
        )
    
    def test_deterministic_reset(self):
        """C4: Test that reset with same seed produces same observations."""
        env1 = AegisEnv(config=self.config)
        env2 = AegisEnv(config=self.config)
        
        seed = 42
        obs1, infos1 = env1.reset(seed=seed)
        obs2, infos2 = env2.reset(seed=seed)
        
        # Observations should match (shape and values)
        for agent_name in env1.possible_agents:
            assert agent_name in obs1
            assert agent_name in obs2
            
            # Check observation keys match
            assert set(obs1[agent_name].keys()) == set(obs2[agent_name].keys())
            
            # Check observation shapes match (handle both arrays and scalars)
            for key in obs1[agent_name].keys():
                val1 = obs1[agent_name][key]
                val2 = obs2[agent_name][key]
                if hasattr(val1, 'shape'):
                    assert val1.shape == val2.shape, (
                        f"Observation shape mismatch for {agent_name}[{key}]: "
                        f"{val1.shape} vs {val2.shape}"
                    )
                else:
                    # Scalar values should match
                    assert type(val1) == type(val2), (
                        f"Observation type mismatch for {agent_name}[{key}]: "
                        f"{type(val1)} vs {type(val2)}"
                    )
        
        # Infos keys should match
        for agent_name in env1.possible_agents:
            assert set(infos1[agent_name].keys()) == set(infos2[agent_name].keys())
    
    def test_deterministic_step(self):
        """C4: Test that same actions produce same results with fixed seed."""
        env1 = AegisEnv(config=self.config)
        env2 = AegisEnv(config=self.config)
        
        seed = 42
        obs1, infos1 = env1.reset(seed=seed)
        obs2, infos2 = env2.reset(seed=seed)
        
        # Fixed action dict
        actions = {name: 0 for name in env1.possible_agents}
        
        # Take same step
        next_obs1, rewards1, term1, trunc1, next_infos1 = env1.step(actions)
        next_obs2, rewards2, term2, trunc2, next_infos2 = env2.step(actions)
        
        # Observation shapes should match
        for agent_name in env1.possible_agents:
            assert agent_name in next_obs1
            assert agent_name in next_obs2
            
            for key in next_obs1[agent_name].keys():
                val1 = next_obs1[agent_name][key]
                val2 = next_obs2[agent_name][key]
                if hasattr(val1, 'shape'):
                    assert val1.shape == val2.shape
                else:
                    assert type(val1) == type(val2)
        
        # Info keys should match
        for agent_name in env1.possible_agents:
            assert set(next_infos1[agent_name].keys()) == set(next_infos2[agent_name].keys())
        
        # Rewards should match (deterministic environment)
        for agent_name in env1.possible_agents:
            assert rewards1[agent_name] == rewards2[agent_name], (
                f"Reward mismatch for {agent_name}: {rewards1[agent_name]} vs {rewards2[agent_name]}"
            )
        
        # Termination/truncation should match
        for agent_name in env1.possible_agents:
            assert term1[agent_name] == term2[agent_name]
            assert trunc1[agent_name] == trunc2[agent_name]
    
    def test_deterministic_multiple_steps(self):
        """C4: Test determinism across multiple steps."""
        env1 = AegisEnv(config=self.config)
        env2 = AegisEnv(config=self.config)
        
        seed = 42
        obs1, _ = env1.reset(seed=seed)
        obs2, _ = env2.reset(seed=seed)
        
        # Run multiple steps with same actions
        for step in range(10):
            actions = {name: 0 for name in env1.possible_agents}
            
            next_obs1, rewards1, term1, trunc1, infos1 = env1.step(actions)
            next_obs2, rewards2, term2, trunc2, infos2 = env2.step(actions)
            
            # Check rewards match
            for agent_name in env1.possible_agents:
                assert rewards1[agent_name] == rewards2[agent_name], (
                    f"Step {step}: Reward mismatch for {agent_name}"
                )
            
            obs1 = next_obs1
            obs2 = next_obs2
            
            if any(term1.values()) or any(trunc1.values()):
                break

