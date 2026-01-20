"""Core invariant tests: prevent regressions in game rules."""

import pytest
import numpy as np
from aegis.env.pettingzoo_env import AegisEnv
from aegis.core.rules import GameConfig


class TestImpostorInvariant:
    """C2: Core invariant test - impostor cannot die from kill."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = GameConfig(
            num_survivors=2,
            num_impostors=1,
            max_ticks=500,
        )
        self.env = AegisEnv(config=self.config)
    
    def test_impostor_cannot_be_killed(self):
        """C2: Test that impostor elimination always has was_ejected=True, was_killed=False."""
        obs, infos = self.env.reset(seed=42)
        
        # Track roles (get from first step since reset only has role)
        roles = {}
        for agent_name in self.env.possible_agents:
            roles[agent_name] = infos[agent_name]["role"]
        
        # Run for many steps with random valid actions
        max_steps = 200
        for step in range(max_steps):
            # Get valid actions (use no-op or random valid action)
            actions = {}
            for agent_name in self.env.possible_agents:
                # Use action 0 (no-op) as safe default
                actions[agent_name] = 0
            
            # Track alive state before step (from obs since reset doesn't have alive in infos)
            was_alive = {name: obs[name]["self_alive"] for name in self.env.possible_agents}
            
            next_obs, rewards, terminations, truncations, next_infos = self.env.step(actions)
            
            # Check elimination invariants
            for agent_name in self.env.possible_agents:
                role = roles[agent_name]
                was_alive_before = was_alive[agent_name]
                is_alive_after = next_infos[agent_name]["alive"]
                was_killed = next_infos[agent_name]["was_killed"]
                was_ejected = next_infos[agent_name]["was_ejected"]
                
                if was_alive_before and not is_alive_after:
                    # Agent died this step
                    if role == 1:  # IMPOSTOR
                        # C2: Impostor invariant - can only be ejected
                        assert was_ejected is True, (
                            f"Impostor {agent_name} died but was_ejected=False. "
                            "Impostors can only be ejected, never killed."
                        )
                        assert was_killed is False, (
                            f"Impostor {agent_name} died with was_killed=True. "
                            "Impostors cannot be killed."
                        )
                    else:  # SURVIVOR
                        # Survivor: exactly one of was_killed or was_ejected must be True
                        assert (was_killed != was_ejected), (
                            f"Survivor {agent_name} died but elimination flags inconsistent. "
                            f"was_killed={was_killed}, was_ejected={was_ejected}. "
                            "Exactly one must be True."
                        )
            
            obs = next_obs
            infos = next_infos
            
            if any(terminations.values()) or any(truncations.values()):
                break
    
    def test_survivor_elimination_mutually_exclusive(self):
        """C2: Test that survivor elimination flags are mutually exclusive."""
        obs, infos = self.env.reset(seed=42)
        
        roles = {}
        for agent_name in self.env.possible_agents:
            roles[agent_name] = infos[agent_name]["role"]
        
        max_steps = 200
        obs, infos = self.env.reset(seed=42)
        for step in range(max_steps):
            actions = {name: 0 for name in self.env.possible_agents}
            was_alive = {name: obs[name]["self_alive"] for name in self.env.possible_agents}
            
            next_obs, rewards, terminations, truncations, next_infos = self.env.step(actions)
            
            for agent_name in self.env.possible_agents:
                role = roles[agent_name]
                was_alive_before = was_alive[agent_name]
                is_alive_after = next_infos[agent_name]["alive"]
                was_killed = next_infos[agent_name]["was_killed"]
                was_ejected = next_infos[agent_name]["was_ejected"]
                
                if was_alive_before and not is_alive_after:
                    if role == 0:  # SURVIVOR
                        # Exactly one must be True
                        assert (was_killed != was_ejected), (
                            f"Survivor {agent_name} died but flags inconsistent: "
                            f"was_killed={was_killed}, was_ejected={was_ejected}"
                        )
            
            infos = next_infos
            
            if any(terminations.values()) or any(truncations.values()):
                break

