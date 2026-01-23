from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class GameConfig:
    """Game configuration with all tunable parameters."""
    

    num_survivors: int = 4
    num_impostors: int = 1
    

    num_rooms: int = 9
    evac_room: int = 4
    vision_radius: int = 2
    

    tasks_per_survivor: int = 4
    task_ticks_single: int = 5
    task_ticks_two_step: int = 3
    two_step_task_probability: float = 0.3
    

    kill_cooldown: int = 30
    initial_kill_delay: int = 0
    

    door_close_duration: int = 15
    door_cooldown: int = 40
    protect_evac_door: bool = True
    

    meeting_discussion_ticks: int = 60
    meeting_voting_ticks: int = 30
    

    evac_ticks_required: int = 5
    max_ticks: int = 1000
    min_episode_ticks_before_impostor_win: int = 0
    

    enable_knower: bool = False
    

    max_messages_per_meeting: int = 10
    comm_mode: str =  "broadcast"
    

    enable_trust_comm: bool = False
    trust_delay_ticks: int = 5
    trust_initial_value: float = 0.5
    trust_support_delta: float = 0.15
    trust_accuse_delta: float = -0.20
    trust_defend_delta: float = 0.05
    trust_question_delta: float = -0.08
    trust_voting_weight: float = 0.3
    

    voting_confidence_threshold: float = 0.0
    

    memory_mode: str =  "none"
    

    suspicion_delay_ticks: int = 30
    

    comm_exploration_boost: float = 0.5
    comm_exploration_decay_steps: int = 100000
    
    meeting_observation_degradation: float = 0.0
    meeting_suspicion_noise: float = 0.0
    

    voting_noise_base: float = 0.0
    voting_coordination_threshold: int = 2
    voting_noise_with_coordination: float = 0.0
    

    reward_win: float = 1.0
    reward_loss: float = -1.0
    reward_task_step: float = 0.05
    reward_correct_report: float = 0.2
    reward_impostor_ejected_survivor: float = 0.3
    reward_survivor_ejected_survivor: float = -0.3
    reward_kill: float = 0.3
    reward_impostor_ejected_impostor: float = -0.4
    reward_survivor_ejected_impostor: float = 0.2
    reward_time_penalty: float = -0.001
    reward_proximity_penalty: float = -0.01
    proximity_penalty_ticks: int = 1
    

    seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""

        if self.enable_trust_comm:
            assert 0.0 <= self.trust_initial_value <= 1.0,                f"trust_initial_value must be in [0, 1], got {self.trust_initial_value}"
            assert 0.0 <= self.trust_voting_weight <= 1.0,                f"trust_voting_weight must be in [0, 1], got {self.trust_voting_weight}"
            assert self.trust_delay_ticks >= 0,                f"trust_delay_ticks must be non-negative, got {self.trust_delay_ticks}"
        

        assert 0.0 <= self.voting_confidence_threshold <= 1.0,            f"voting_confidence_threshold must be in [0, 1], got {self.voting_confidence_threshold}"
    
    @classmethod
    def from_yaml(cls, path: str) ->  "GameConfig":
        """Load config from YAML file."""
        with open(path,  "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        with open(path,  "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    @property
    def num_agents(self) -> int:
        """Total number of agents."""
        return self.num_survivors + self.num_impostors



def create_grid_map(rows: int = 3, cols: int = 3) -> tuple[list[int], list[tuple[int, int]]]:
    """Create a grid map with rooms connected horizontally and vertically."""
    rooms = list(range(rows * cols))
    edges = []
    
    for r in range(rows):
        for c in range(cols):
            room_id = r * cols + c

            if c < cols - 1:
                edges.append((room_id, room_id + 1))

            if r < rows - 1:
                edges.append((room_id, room_id + cols))
    
    return rooms, edges


def create_default_map() -> tuple[list[int], list[tuple[int, int]]]:
    """Create the default 3x3 grid map."""
    return create_grid_map(3, 3)

