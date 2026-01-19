"""Constants and configuration for the game."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class GameConfig:
    """Game configuration with all tunable parameters."""
    
    # Agent counts
    num_survivors: int = 4
    num_impostors: int = 1
    
    # Map settings
    num_rooms: int = 9  # Default 3x3 grid
    evac_room: int = 4  # Center room for 3x3
    vision_radius: int = 2  # Graph distance for visibility
    
    # Task settings
    tasks_per_survivor: int = 4
    task_ticks_single: int = 5  # Ticks to complete single_point task
    task_ticks_two_step: int = 3  # Ticks per step for two_step task
    two_step_task_probability: float = 0.3
    
    # Kill settings
    kill_cooldown: int = 30
    
    # Door settings
    door_close_duration: int = 15
    door_cooldown: int = 40
    protect_evac_door: bool = True  # Prevent closing doors to evac room
    
    # Meeting settings
    meeting_discussion_ticks: int = 60
    meeting_voting_ticks: int = 30
    
    # Win conditions
    evac_ticks_required: int = 5  # Consecutive ticks in evac room to win
    max_ticks: int = 1000  # Episode timeout
    
    # Knower modifier
    enable_knower: bool = False
    
    # Communication settings
    max_messages_per_meeting: int = 10  # Per agent
    comm_mode: str = "broadcast"  # "broadcast" or "local"
    
    # Memory mode
    memory_mode: str = "none"  # "none", "aggregated", "recurrent"
    
    # Rewards
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
    reward_proximity_penalty: float = -0.01  # penalty for being near impostor
    proximity_penalty_ticks: int = 1  # apply if in same room for N ticks (1 = immediate)
    
    # Seed
    seed: Optional[int] = None
    
    @classmethod
    def from_yaml(cls, path: str) -> "GameConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    @property
    def num_agents(self) -> int:
        """Total number of agents."""
        return self.num_survivors + self.num_impostors


# Default map layouts
def create_grid_map(rows: int = 3, cols: int = 3) -> tuple[list[int], list[tuple[int, int]]]:
    """Create a grid map with rooms connected horizontally and vertically."""
    rooms = list(range(rows * cols))
    edges = []
    
    for r in range(rows):
        for c in range(cols):
            room_id = r * cols + c
            # Connect to right neighbor
            if c < cols - 1:
                edges.append((room_id, room_id + 1))
            # Connect to bottom neighbor
            if r < rows - 1:
                edges.append((room_id, room_id + cols))
    
    return rooms, edges


def create_default_map() -> tuple[list[int], list[tuple[int, int]]]:
    """Create the default 3x3 grid map."""
    return create_grid_map(3, 3)

