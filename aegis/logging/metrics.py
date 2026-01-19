"""Episode summary metrics."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json

from aegis.core.world import WorldState, Role


@dataclass
class EpisodeMetrics:
    """Metrics collected during an episode."""
    
    # Episode outcome
    episode_length: int = 0
    winner: Optional[int] = None  # 0=Survivor, 1=Impostor
    win_reason: str = ""
    
    # Event counts
    kills: int = 0
    meetings: int = 0
    ejections: int = 0
    tasks_completed: int = 0
    messages_sent: int = 0
    
    # Task completion
    total_task_steps: int = 0
    completed_task_steps: int = 0
    
    # Voting accuracy (for analysis)
    correct_ejections: int = 0  # Impostors ejected
    wrong_ejections: int = 0     # Survivors ejected
    
    # Survivors at end
    survivors_alive: int = 0
    impostors_alive: int = 0
    
    def finalize(self, world: WorldState) -> None:
        """Finalize metrics at episode end."""
        self.episode_length = world.tick
        self.winner = world.winner
        
        # Count survivors
        self.survivors_alive = len(world.alive_survivors())
        self.impostors_alive = len(world.alive_impostors())
        
        # Task stats
        self.total_task_steps = world.total_task_steps()
        self.completed_task_steps = world.team_task_progress()
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: dict) -> "EpisodeMetrics":
        """Create from dictionary."""
        return cls(**data)


class MetricsAggregator:
    """Aggregates metrics across multiple episodes."""
    
    def __init__(self):
        self.episodes: list[EpisodeMetrics] = []
    
    def add(self, metrics: EpisodeMetrics) -> None:
        """Add episode metrics."""
        self.episodes.append(metrics)
    
    def summary(self) -> dict:
        """Compute summary statistics."""
        if not self.episodes:
            return {}
        
        n = len(self.episodes)
        
        survivor_wins = sum(1 for m in self.episodes if m.winner == Role.SURVIVOR)
        impostor_wins = sum(1 for m in self.episodes if m.winner == Role.IMPOSTOR)
        
        return {
            "num_episodes": n,
            "survivor_win_rate": survivor_wins / n,
            "impostor_win_rate": impostor_wins / n,
            "avg_episode_length": sum(m.episode_length for m in self.episodes) / n,
            "avg_kills": sum(m.kills for m in self.episodes) / n,
            "avg_meetings": sum(m.meetings for m in self.episodes) / n,
            "avg_ejections": sum(m.ejections for m in self.episodes) / n,
            "avg_tasks_completed": sum(m.tasks_completed for m in self.episodes) / n,
            "avg_messages_sent": sum(m.messages_sent for m in self.episodes) / n,
            "task_completion_rate": (
                sum(m.completed_task_steps for m in self.episodes) /
                max(1, sum(m.total_task_steps for m in self.episodes))
            ),
        }
    
    def to_csv(self, path: str) -> None:
        """Export all episodes to CSV."""
        import csv
        
        if not self.episodes:
            return
        
        fieldnames = list(self.episodes[0].to_dict().keys())
        
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in self.episodes:
                writer.writerow(m.to_dict())

