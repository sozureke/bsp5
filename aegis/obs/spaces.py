"""Gymnasium spaces for observations and action masks."""

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
        self.num_tokens = 20  # Token vocabulary size
        self.max_bodies = config.num_agents  # Max possible bodies
        self.max_messages = 50  # Max messages to track per meeting
        self.max_tasks = config.tasks_per_survivor
        
        # Compute action space size
        self.action_space_size = self._compute_action_space_size()
    
    def _compute_action_space_size(self) -> int:
        """Compute total action space size."""
        # MOVE_TO_ROOM[r] for each room
        # WORK
        # REPORT
        # KILL[target] for each agent
        # CLOSE_DOOR[door] for each door
        # SEND_TOKEN[token] for each token
        # VOTE[agent] for each agent
        # VOTE_SKIP
        return (
            self.num_rooms +  # MOVE
            1 +  # WORK
            1 +  # REPORT
            self.num_agents +  # KILL
            self.num_edges +  # CLOSE_DOOR
            self.num_tokens +  # SEND_TOKEN
            self.num_agents +  # VOTE
            1  # VOTE_SKIP
        )
    
    def get_observation_space(self) -> spaces.Dict:
        """Create the observation space as a Gymnasium Dict space."""
        return spaces.Dict({
            # Phase: 0=free_play, 1=meeting, 2=voting
            "phase": spaces.Discrete(3),
            
            # Agent's own state
            "self_room": spaces.Discrete(self.num_rooms),
            "self_role": spaces.Discrete(2),  # 0=survivor, 1=impostor
            "self_alive": spaces.Discrete(2),
            
            # Evac status
            "evac_active": spaces.Discrete(2),
            
            # Visible agents: (seen, room, alive) for each agent
            # seen=0 means not visible, room=-1 (encoded as num_rooms)
            "visible_agents": spaces.Box(
                low=0,
                high=max(self.num_rooms, 2),
                shape=(self.num_agents, 3),
                dtype=np.int32
            ),
            
            # Visible bodies: (seen, room, age) for max_bodies
            # room and age can be -1 when body is not seen
            "visible_bodies": spaces.Box(
                low=-1,
                high=max(self.num_rooms, 1000),
                shape=(self.max_bodies, 3),
                dtype=np.int32
            ),
            
            # Door states: (is_open, timer_remaining) for each door
            "door_states": spaces.Box(
                low=0,
                high=max(2, 100),
                shape=(self.num_edges, 2),
                dtype=np.int32
            ),
            
            # Task state for survivors
            # (task_idx, step_idx, progress, ticks_remaining, target_room) for current task
            "task_state": spaces.Box(
                low=-1,
                high=max(self.max_tasks, self.num_rooms, 100),
                shape=(5,),
                dtype=np.int32
            ),
            
            # Team task progress (normalized)
            "team_task_progress": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            ),
            
            # Cooldowns (normalized)
            "kill_cooldown": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "door_cooldown": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            
            # Meeting messages: last N tokens heard (token_id, -1 for none)
            "meeting_messages": spaces.Box(
                low=-1,
                high=self.num_tokens,
                shape=(self.max_messages, 2),  # (sender_id, token_id)
                dtype=np.int32
            ),
            
            # Meeting info
            "meeting_reporter": spaces.Discrete(self.num_agents + 1),  # +1 for no meeting
            "meeting_body_room": spaces.Discrete(self.num_rooms + 1),  # +1 for no meeting
            
            # Knower info (if applicable)
            "knows_role_of": spaces.Discrete(self.num_agents + 1),  # +1 for none
            "known_role": spaces.Discrete(3),  # 0=survivor, 1=impostor, 2=unknown
            
            # Tick info
            "tick": spaces.Box(low=0, high=self.config.max_ticks, shape=(1,), dtype=np.int32),
            
            # Action mask
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

