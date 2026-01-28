"""Aggregate metrics from all experiments into a single JSON file."""

from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class Meeting:
    """Represents a single meeting."""
    tick_start: int
    tick_end: int
    reporter_id: int
    body_room: int
    comm_actions: List[Dict]
    votes: dict[int, Optional[int]]
    ejected_id: Optional[int] = None
    ejected_role: Optional[int] = None
    
    @property
    def duration(self) -> int:
        return self.tick_end - self.tick_start
    
    @property
    def num_comm_actions(self) -> int:
        return len(self.comm_actions)
    
    @property
    def has_ejection(self) -> bool:
        return self.ejected_id is not None


def extract_metrics(events_path: Path) -> Dict:
    """Extract all metrics from an events_all.jsonl file."""
    print(f"Processing {events_path.name}...")
    
    events = []
    with open(events_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    
    # Parse meetings
    meetings = []
    for event in events:
        if event.get("event_type") == "meeting_summary":
            meetings.append(Meeting(
                tick_start=event["meeting_tick_start"],
                tick_end=event["tick"],
                reporter_id=event["reporter_id"],
                body_room=event["body_room"],
                comm_actions=event.get("comm_actions", []),
                votes=event.get("votes", {}),
                ejected_id=event.get("ejected_id"),
                ejected_role=event.get("ejected_role"),
            ))
    
    # Win statistics
    win_events = [e for e in events if e.get("event_type") == "win"]
    total_games = len(win_events)
    survivor_wins = sum(1 for w in win_events if w.get("winner") == 0)
    impostor_wins = sum(1 for w in win_events if w.get("winner") == 1)
    win_reasons = Counter(w.get("reason") for w in win_events)
    
    # Meeting statistics
    total_meetings = len(meetings)
    meetings_with_ejection = sum(1 for m in meetings if m.has_ejection)
    ejection_rate = (meetings_with_ejection / total_meetings * 100) if total_meetings > 0 else 0
    
    # Correct ejection rate (ejected_role == 1 means impostor)
    ejections = [m for m in meetings if m.has_ejection]
    correct_ejections = sum(1 for e in ejections if e.ejected_role == 1)
    correct_ejection_rate = (correct_ejections / len(ejections) * 100) if ejections else 0
    
    # Communication statistics
    all_comm_actions = []
    for meeting in meetings:
        all_comm_actions.extend(meeting.comm_actions)
    
    comm_actions_total = len(all_comm_actions)
    avg_comm_per_meeting = comm_actions_total / total_meetings if total_meetings > 0 else 0
    
    # Communication by type
    comm_by_type = Counter(a.get("action_name", "UNKNOWN") for a in all_comm_actions)
    
    # Communication by role
    survivor_comm = [a for a in all_comm_actions if a.get("sender_role") == 0]
    impostor_comm = [a for a in all_comm_actions if a.get("sender_role") == 1]
    
    # Communication by role and type (detailed breakdown)
    comm_by_role_type = {
        "survivor": {
            "ACCUSE": 0,
            "SUPPORT": 0,
            "QUESTION": 0,
            "DEFEND_SELF": 0,
            "NO_OP": 0,
        },
        "impostor": {
            "ACCUSE": 0,
            "SUPPORT": 0,
            "QUESTION": 0,
            "DEFEND_SELF": 0,
            "NO_OP": 0,
        },
    }
    
    for action in survivor_comm:
        action_name = action.get("action_name", "")
        if action_name.startswith("ACCUSE"):
            comm_by_role_type["survivor"]["ACCUSE"] += 1
        elif action_name.startswith("SUPPORT"):
            comm_by_role_type["survivor"]["SUPPORT"] += 1
        elif action_name.startswith("QUESTION"):
            comm_by_role_type["survivor"]["QUESTION"] += 1
        elif action_name == "DEFEND_SELF":
            comm_by_role_type["survivor"]["DEFEND_SELF"] += 1
        elif action_name == "NO_OP":
            comm_by_role_type["survivor"]["NO_OP"] += 1
    
    for action in impostor_comm:
        action_name = action.get("action_name", "")
        if action_name.startswith("ACCUSE"):
            comm_by_role_type["impostor"]["ACCUSE"] += 1
        elif action_name.startswith("SUPPORT"):
            comm_by_role_type["impostor"]["SUPPORT"] += 1
        elif action_name.startswith("QUESTION"):
            comm_by_role_type["impostor"]["QUESTION"] += 1
        elif action_name == "DEFEND_SELF":
            comm_by_role_type["impostor"]["DEFEND_SELF"] += 1
        elif action_name == "NO_OP":
            comm_by_role_type["impostor"]["NO_OP"] += 1
    
    # Episode statistics
    episode_lengths = [e.get("tick", 0) for e in win_events]
    avg_episode_length = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0
    
    # Event counts
    event_types = Counter(e.get("event_type") for e in events)
    
    metrics = {
        "experiment_name": events_path.parent.name,
        "total_events": len(events),
        "total_games": total_games,
        "wins": {
            "survivor": survivor_wins,
            "impostor": impostor_wins,
            "survivor_rate": (survivor_wins / total_games * 100) if total_games > 0 else 0,
            "impostor_rate": (impostor_wins / total_games * 100) if total_games > 0 else 0,
            "win_reasons": dict(win_reasons),
        },
        "meetings": {
            "total": total_meetings,
            "with_ejection": meetings_with_ejection,
            "ejection_rate_pct": ejection_rate,
            "correct_ejections": correct_ejections,
            "correct_ejection_rate_pct": correct_ejection_rate,
            "avg_duration_ticks": sum(m.duration for m in meetings) / total_meetings if total_meetings > 0 else 0,
        },
        "communication": {
            "total_actions": comm_actions_total,
            "avg_per_meeting": avg_comm_per_meeting,
            "by_type": dict(comm_by_type),
            "by_role": {
                "survivor": len(survivor_comm),
                "impostor": len(impostor_comm),
            },
            "by_role_type": comm_by_role_type,
        },
        "episodes": {
            "avg_length_ticks": avg_episode_length,
        },
        "event_counts": dict(event_types),
    }
    
    return metrics


def main():
    checkpoints_dir = Path("checkpoints")
    
    if not checkpoints_dir.exists():
        print(f"Error: {checkpoints_dir} directory not found")
        return 1
    
    # Find all events_all.jsonl files
    events_files = list(checkpoints_dir.glob("*/events_all.jsonl"))
    
    if not events_files:
        print(f"No events_all.jsonl files found in {checkpoints_dir}")
        print("Make sure you've merged events for each experiment:")
        print("python -m aegis.analysis.merge_events checkpoints/<experiment_name>")
        return 1
    
    print(f"Found {len(events_files)} experiments to process\n")
    
    all_metrics = {}
    
    for events_file in sorted(events_files):
        try:
            metrics = extract_metrics(events_file)
            all_metrics[metrics["experiment_name"]] = metrics
            print(f"✓ {metrics['experiment_name']}: {metrics['total_games']} games, {metrics['meetings']['total']} meetings")
        except Exception as e:
            print(f"✗ Error processing {events_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save to JSON
    output_file = Path("results/all_metrics.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n✓ Aggregated metrics saved to {output_file}")
    print(f"\nExperiments processed: {len(all_metrics)}")
    
    # Print summary
    if all_metrics:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for exp_name, metrics in sorted(all_metrics.items()):
            wins = metrics["wins"]
            print(f"\n{exp_name}:")
            print(f"Games: {metrics['total_games']}")
            print(f"Survivor win rate: {wins['survivor_rate']:.1f}%")
            print(f"Impostor win rate: {wins['impostor_rate']:.1f}%")
            print(f"Meetings: {metrics['meetings']['total']}")
            print(f"Correct ejection rate: {metrics['meetings']['correct_ejection_rate_pct']:.1f}%")
            print(f"Avg comm per meeting: {metrics['communication']['avg_per_meeting']:.1f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
