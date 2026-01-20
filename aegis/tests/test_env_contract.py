"""Contract tests: validate env→trainer interface (infos schema and types)."""

import pytest
import numpy as np
from aegis.env.pettingzoo_env import AegisEnv
from aegis.core.rules import GameConfig


class TestInfosContract:
    """Test that infos dict conforms to expected schema and types."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = GameConfig(
            num_survivors=2,
            num_impostors=1,
            max_ticks=100,
        )
        self.env = AegisEnv(config=self.config)
    
    def test_infos_schema_after_reset(self):
        """C1: Contract test - infos schema after reset."""
        obs, infos = self.env.reset(seed=42)
        
        # Reset only returns minimal info (role), but should still be valid
        for agent_name in self.env.possible_agents:
            info = infos[agent_name]
            
            # At minimum, role should be present and valid
            assert "role" in info
            assert isinstance(info["role"], int)
            assert info["role"] in {0, 1}  # SURVIVOR=0, IMPOSTOR=1
    
    def test_infos_schema_after_step(self):
        """C1: Contract test - infos schema after step."""
        obs, infos = self.env.reset(seed=42)
        
        # Take one step with valid actions
        actions = {name: 0 for name in self.env.possible_agents}  # All no-op actions
        next_obs, rewards, terminations, truncations, next_infos = self.env.step(actions)
        
        for agent_name in self.env.possible_agents:
            info = next_infos[agent_name]
            
            # Required keys exist
            assert "role" in info
            assert "alive" in info
            assert "winner" in info
            assert "events" in info
            assert "did_kill" in info
            assert "did_task_step" in info
            assert "was_killed" in info
            assert "was_ejected" in info
            assert "team_task_step" in info
            assert "did_report" in info
            assert "ejection_benefited" in info
            assert "ejection_penalized" in info
            
            # Type checks
            assert isinstance(info["role"], int)
            assert info["role"] in {0, 1}
            assert isinstance(info["alive"], bool)
            assert info["winner"] is None or isinstance(info["winner"], int)
            if info["winner"] is not None:
                assert info["winner"] in {0, 1}
            assert isinstance(info["events"], list)
            assert isinstance(info["did_kill"], bool)
            assert isinstance(info["did_task_step"], bool)
            assert isinstance(info["was_killed"], bool)
            assert isinstance(info["was_ejected"], bool)
            assert isinstance(info["team_task_step"], bool)
            assert isinstance(info["did_report"], bool)
            assert isinstance(info["ejection_benefited"], bool)
            assert isinstance(info["ejection_penalized"], bool)
    
    def test_events_always_present(self):
        """C1: Contract test - events list always present after step (required for reward shaping)."""
        obs, infos = self.env.reset(seed=42)
        
        # After step (events are populated in step, not reset)
        actions = {name: 0 for name in self.env.possible_agents}
        next_obs, rewards, terminations, truncations, next_infos = self.env.step(actions)
        
        for agent_name in self.env.possible_agents:
            assert "events" in next_infos[agent_name], (
                f"Missing 'events' key in infos for {agent_name}. "
                "Reward shaping requires event tags."
            )
            assert isinstance(next_infos[agent_name]["events"], list), (
                f"'events' must be a list, got {type(next_infos[agent_name]['events'])}"
            )
    
    def test_role_consistency(self):
        """C1: Contract test - role is consistent across steps."""
        obs, infos = self.env.reset(seed=42)
        
        roles = {name: infos[name]["role"] for name in self.env.possible_agents}
        
        # Role should not change during episode
        for _ in range(10):
            actions = {name: 0 for name in self.env.possible_agents}
            next_obs, rewards, terminations, truncations, next_infos = self.env.step(actions)
            
            for agent_name in self.env.possible_agents:
                assert next_infos[agent_name]["role"] == roles[agent_name]
            
            if any(terminations.values()) or any(truncations.values()):
                break

