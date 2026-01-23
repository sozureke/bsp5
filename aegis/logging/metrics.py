from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json

from aegis.core.world import WorldState, Role


@dataclass
class EpisodeMetrics:
    """Metrics collected during an episode."""
    

    episode_length: int = 0
    winner: Optional[int] = None
    win_reason: str =  ""
    

    kills: int = 0
    meetings: int = 0
    ejections: int = 0
    tasks_completed: int = 0
    messages_sent: int = 0
    

    total_task_steps: int = 0
    completed_task_steps: int = 0
    

    correct_ejections: int = 0
    wrong_ejections: int = 0
    failed_votes_low_confidence: int = 0
    max_voting_scores: list[float] = field(default_factory=list)
    

    survivors_alive: int = 0
    impostors_alive: int = 0
    

    comm_support_count: int = 0
    comm_accuse_count: int = 0
    comm_defend_count: int = 0
    comm_question_count: int = 0
    comm_noop_count: int = 0
    

    trust_support_avg_delta: float = 0.0
    trust_accuse_avg_delta: float = 0.0
    trust_defend_avg_delta: float = 0.0
    trust_question_avg_delta: float = 0.0
    
    def finalize(self, world: WorldState, trust_manager = None) -> None:
        """Finalize metrics at episode end."""
        self.episode_length = world.tick
        self.winner = world.winner
        

        self.survivors_alive = len(world.alive_survivors())
        self.impostors_alive = len(world.alive_impostors())
        

        self.total_task_steps = world.total_task_steps()
        self.completed_task_steps = world.team_task_progress()
        

        if trust_manager is not None:
            trust_stats = trust_manager.get_statistics()
            self.comm_support_count = trust_stats.get("support_count", 0)
            self.comm_accuse_count = trust_stats.get("accuse_count", 0)
            self.comm_defend_count = trust_stats.get("defend_count", 0)
            self.comm_question_count = trust_stats.get("question_count", 0)
            self.trust_support_avg_delta = trust_stats.get("support_avg_delta", 0.0)
            self.trust_accuse_avg_delta = trust_stats.get("accuse_avg_delta", 0.0)
            self.trust_defend_avg_delta = trust_stats.get("defend_avg_delta", 0.0)
            self.trust_question_avg_delta = trust_stats.get("question_avg_delta", 0.0)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: dict) ->  "EpisodeMetrics":
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
        
        with open(path,  "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in self.episodes:
                writer.writerow(m.to_dict())

