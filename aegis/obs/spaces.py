from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from aegis.core.rules import GameConfig


class ObservationSpaces:
    """Factory for creating Gymnasium observation and action spaces."""
    
    def __init__(self, config: GameConfig, num_rooms: int, num_edges: int):
        self.config = config
        self.num_rooms = num_rooms
        self.num_edges = num_edges
        self.num_agents = config.num_agents
        self.num_tokens = 20
        self.num_comm_actions = 17
        self.max_bodies = config.num_agents
        self.max_messages = 50
        self.max_tasks = config.tasks_per_survivor
        

        self.action_space_size = self._compute_action_space_size()
    
    def _compute_action_space_size(self) -> int:
        """Compute total action space size."""









        return (
            self.num_rooms +
            1 +
            1 +
            self.num_agents +
            self.num_edges +
            self.num_tokens +
            self.num_comm_actions +
            self.num_agents +
            1
        )
    
    def get_observation_space(self) -> spaces.Dict:
        """Create the observation space as a Gymnasium Dict space."""
        return spaces.Dict({

            "phase": spaces.Discrete(3),
            

            "self_room": spaces.Discrete(self.num_rooms),
            "self_role": spaces.Discrete(2),
            "self_alive": spaces.Discrete(2),
            

            "evac_active": spaces.Discrete(2),
            


            "visible_agents": spaces.Box(
                low=0,
                high=max(self.num_rooms, 2),
                shape=(self.num_agents, 3),
                dtype=np.int32
            ),
            


            "visible_bodies": spaces.Box(
                low=-1,
                high=max(self.num_rooms, 1000),
                shape=(self.max_bodies, 3),
                dtype=np.int32
            ),
            

            "door_states": spaces.Box(
                low=0,
                high=max(2, 100),
                shape=(self.num_edges, 2),
                dtype=np.int32
            ),
            


            "task_state": spaces.Box(
                low=-1,
                high=max(self.max_tasks, self.num_rooms, 100),
                shape=(5,),
                dtype=np.int32
            ),
            

            "team_task_progress": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            ),
            

            "kill_cooldown": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "door_cooldown": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            

            "meeting_messages": spaces.Box(
                low=-1,
                high=self.num_tokens,
                shape=(self.max_messages, 2),
                dtype=np.int32
            ),
            

            "meeting_reporter": spaces.Discrete(self.num_agents + 1),
            "meeting_body_room": spaces.Discrete(self.num_rooms + 1),
            

            "knows_role_of": spaces.Discrete(self.num_agents + 1),
            "known_role": spaces.Discrete(3),
            

            "tick": spaces.Box(low=0, high=self.config.max_ticks, shape=(1,), dtype=np.int32),
            

            "trust_vector": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.num_agents,),
                dtype=np.float32
            ),
            

            "action_mask": spaces.Box(
                low=0,
                high=1,
                shape=(self.action_space_size,),
                dtype=np.int8
            ),
        })
    
    def get_action_space(self) -> spaces.Discrete:
        """Create the action space."""
        return spaces.Discrete(self.action_space_size)

