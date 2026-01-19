"""Heuristic impostor policy."""

from __future__ import annotations

import numpy as np
from typing import Optional

from aegis.core.rules import GameConfig
from aegis.comms.tokens import TokenVocab


class HeuristicImpostorPolicy:
    """
    Heuristic policy for impostors.
    
    Behavior:
    - Roam around
    - Try to kill isolated survivors (same room alone)
    - Occasionally close doors
    - Vote to eject a survivor if accusations exist, else skip
    - Send accusation messages against random survivors
    """
    
    def __init__(
        self,
        config: GameConfig,
        num_rooms: int = 9,
        num_agents: int = 5,
        seed: Optional[int] = None,
    ):
        self.config = config
        self.num_rooms = num_rooms
        self.num_agents = num_agents
        self.rng = np.random.default_rng(seed)
        
        # Compute action offsets
        n_rooms = num_rooms
        n_agents = num_agents
        n_doors = 12
        n_tokens = 20
        
        self.offset_move = 0
        self.offset_work = n_rooms
        self.offset_report = n_rooms + 1
        self.offset_kill = n_rooms + 2
        self.offset_close = n_rooms + 2 + n_agents
        self.offset_token = n_rooms + 2 + n_agents + n_doors
        self.offset_vote = n_rooms + 2 + n_agents + n_doors + n_tokens
        self.offset_vote_skip = n_rooms + 2 + n_agents + n_doors + n_tokens + n_agents
    
    def get_action(self, observation: dict, agent_name: str = "") -> int:
        """
        Select action based on heuristic rules.
        
        Args:
            observation: Observation dictionary
            agent_name: Agent name (for debugging)
        
        Returns:
            Selected action index
        """
        action_mask = observation.get("action_mask", None)
        if action_mask is None:
            return 0
        
        phase = observation.get("phase", 0)
        self_room = observation.get("self_room", 0)
        visible_agents = observation.get("visible_agents", np.zeros((self.num_agents, 3)))
        kill_cooldown = observation.get("kill_cooldown", np.array([1.0]))
        door_cooldown = observation.get("door_cooldown", np.array([1.0]))
        meeting_messages = observation.get("meeting_messages", np.full((50, 2), -1))
        
        # Phase 0: Free play
        if phase == 0:
            # Check for kill opportunity
            if kill_cooldown[0] == 0:
                # Find survivors in same room
                targets_in_room = []
                for agent_id in range(self.num_agents):
                    seen, room, alive = visible_agents[agent_id]
                    if seen and alive and room == self_room:
                        kill_action = self.offset_kill + agent_id
                        if action_mask[kill_action] > 0:
                            targets_in_room.append(agent_id)
                
                # Check if isolated (only us and one other agent visible in room)
                visible_in_room = sum(
                    1 for i in range(self.num_agents)
                    if visible_agents[i, 0] and visible_agents[i, 1] == self_room
                )
                
                if targets_in_room and visible_in_room <= 2:
                    # Good kill opportunity
                    target = self.rng.choice(targets_in_room)
                    return self.offset_kill + target
            
            # Occasionally close a door (10% chance)
            if door_cooldown[0] == 0 and self.rng.random() < 0.1:
                close_actions = [
                    a for a in range(self.offset_close, self.offset_token)
                    if action_mask[a] > 0
                ]
                if close_actions:
                    return int(self.rng.choice(close_actions))
            
            # Roam: move to a random room
            if self.rng.random() < 0.5:
                move_actions = [
                    a for a in range(self.offset_move, self.offset_work)
                    if action_mask[a] > 0 and a != self.offset_move + self_room
                ]
                if move_actions:
                    return int(self.rng.choice(move_actions))
            
            return self._random_valid_action(action_mask)
        
        # Phase 1: Meeting
        elif phase == 1:
            # Send accusation against a random survivor (30% chance)
            if self.rng.random() < 0.3:
                # Accuse a random agent
                target = int(self.rng.integers(0, self.num_agents))
                token = TokenVocab.make_suspect(target)
                token_action = self.offset_token + token
                if action_mask[token_action] > 0:
                    return token_action
            
            return self._random_valid_action(action_mask)
        
        # Phase 2: Voting
        elif phase == 2:
            # Try to vote for someone being accused (preferably survivors)
            accusations = self._count_accusations(meeting_messages)
            
            if accusations:
                # Vote for most accused
                most_accused = max(accusations, key=accusations.get)
                vote_action = self.offset_vote + most_accused
                if action_mask[vote_action] > 0:
                    return vote_action
            
            # Random vote for anyone alive
            vote_actions = [
                a for a in range(self.offset_vote, self.offset_vote_skip)
                if action_mask[a] > 0
            ]
            if vote_actions:
                return int(self.rng.choice(vote_actions))
            
            # Skip
            if action_mask[self.offset_vote_skip] > 0:
                return self.offset_vote_skip
            
            return self._random_valid_action(action_mask)
        
        return self._random_valid_action(action_mask)
    
    def _count_accusations(self, meeting_messages: np.ndarray) -> dict[int, int]:
        """Count SUSPECT tokens from messages."""
        accusations = {}
        for sender_id, token_id in meeting_messages:
            if sender_id < 0:
                continue
            target = TokenVocab.get_suspect_target(token_id)
            if target is not None:
                accusations[target] = accusations.get(target, 0) + 1
        return accusations
    
    def _random_valid_action(self, action_mask: np.ndarray) -> int:
        """Select random valid action."""
        valid_actions = np.where(action_mask > 0)[0]
        if len(valid_actions) == 0:
            return 0
        return int(self.rng.choice(valid_actions))
    
    def get_actions(self, observations: dict[str, dict]) -> dict[str, int]:
        """Get actions for multiple agents."""
        return {
            agent: self.get_action(obs, agent)
            for agent, obs in observations.items()
        }

