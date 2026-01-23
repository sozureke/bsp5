"""Heuristic survivor policy."""

from __future__ import annotations

import numpy as np
from typing import Optional

from aegis.core.rules import GameConfig
from aegis.comms.tokens import TokenVocab


class HeuristicSurvivorPolicy:
    """
    Heuristic policy for survivors.
    
    Behavior:
    - Go to next task location and work
    - If sees body in room -> report
    - During voting: vote most frequently accused if any, else skip
    - If evac active and tasks done, go to evac room
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
        evac_active = observation.get("evac_active", 0)
        task_state = observation.get("task_state", np.array([-1, -1, -1, -1, -1]))
        visible_bodies = observation.get("visible_bodies", np.zeros((self.num_agents, 3)))
        meeting_messages = observation.get("meeting_messages", np.full((50, 2), -1))
        

        if phase == 0:

            report_action = self.offset_report
            if action_mask[report_action] > 0:
                return report_action
            

            if task_state[0] >= 0:
                target_room = task_state[4]
                if target_room >= 0 and target_room < self.num_rooms:
                    if self_room == target_room:

                        work_action = self.offset_work
                        if action_mask[work_action] > 0:
                            return work_action
                    else:

                        move_action = self.offset_move + target_room
                        if action_mask[move_action] > 0:
                            return move_action
            

            if evac_active:
                evac_room = self.config.evac_room
                if self_room != evac_room:
                    move_action = self.offset_move + evac_room
                    if action_mask[move_action] > 0:
                        return move_action
            

            return self._random_valid_action(action_mask)
        

        elif phase == 1:

            if self.rng.random() < 0.3:

                token = TokenVocab.make_i_was_in_room(self_room % 3)
                token_action = self.offset_token + token
                if action_mask[token_action] > 0:
                    return token_action
            

            return self._random_valid_action(action_mask)
        

        elif phase == 2:

            accusations = self._count_accusations(meeting_messages)
            

            if accusations:
                most_accused = max(accusations, key=accusations.get)
                vote_action = self.offset_vote + most_accused
                if action_mask[vote_action] > 0:
                    return vote_action
            

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

