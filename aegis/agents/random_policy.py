"""Random policy that selects valid actions uniformly."""

from __future__ import annotations

import numpy as np
from typing import Optional


class RandomPolicy:
    """Random policy that samples uniformly from valid actions."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
    
    def get_action(self, observation: dict) -> int:
        """
        Select a random valid action based on the action mask.
        
        Args:
            observation: Observation dictionary with 'action_mask' key
        
        Returns:
            Selected action index
        """
        action_mask = observation.get("action_mask", None)
        
        if action_mask is None:

            return 0
        

        valid_actions = np.where(action_mask > 0)[0]
        
        if len(valid_actions) == 0:
            return 0
        

        return int(self.rng.choice(valid_actions))
    
    def get_actions(self, observations: dict[str, dict]) -> dict[str, int]:
        """
        Get actions for multiple agents.
        
        Args:
            observations: Dict mapping agent names to observations
        
        Returns:
            Dict mapping agent names to actions
        """
        return {
            agent: self.get_action(obs)
            for agent, obs in observations.items()
        }

