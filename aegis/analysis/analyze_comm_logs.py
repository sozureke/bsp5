import argparse
from collections import Counter, defaultdict
from pathlib import Path

from aegis.logging.event_writer import EventReader


def analyze_comm_actions(log_path: str):
    """Analyze communication actions from a JSONL event log."""
    print("=" * 70)
    print(f"Analyzing Communication Logs: {log_path}")
    print("=" * 70)
    
    reader = EventReader(log_path)
    

    comm_events = reader.filter_by_type("comm_action")
    
    if not comm_events:
        print("\nNo communication action events found in log.")
        print("Make sure enable_trust_comm=true in your config.")
        return
    
    print(f"\nFound {len(comm_events)} communication actions\n")
    

    print("1. Action Type Distribution")
    print("-" * 70)
    action_counts = Counter(e["action_name"] for e in comm_events)
    for action_name, count in action_counts.most_common():
        percentage = (count / len(comm_events)) * 100
        print(f"{action_name:20s}: {count:4d} ({percentage:5.1f}%)")
    

    print("\n2. Communication by Agent")
    print("-" * 70)
    agent_counts = Counter(e["sender_id"] for e in comm_events)
    for agent_id, count in sorted(agent_counts.items()):
        print(f"Agent {agent_id}: {count:4d} actions")
    

    print("\n3. Temporal Distribution")
    print("-" * 70)
    ticks = [e["tick"] for e in comm_events]
    if ticks:
        min_tick, max_tick = min(ticks), max(ticks)
        print(f"First action: tick {min_tick}")
        print(f"Last action:  tick {max_tick}")
        print(f"Duration:     {max_tick - min_tick} ticks")
        

        early = sum(1 for t in ticks if t < max_tick / 3)
        mid = sum(1 for t in ticks if max_tick / 3 <= t < 2 * max_tick / 3)
        late = sum(1 for t in ticks if t >= 2 * max_tick / 3)
        print(f"\nEarly game: {early} ({(early/len(ticks))*100:.1f}%)")
        print(f"Mid game:   {mid} ({(mid/len(ticks))*100:.1f}%)")
        print(f"Late game:  {late} ({(late/len(ticks))*100:.1f}%)")
    

    print("\n4. Target Analysis")
    print("-" * 70)
    

    targets = defaultdict(lambda: {"support": 0, "accuse": 0, "question": 0})
    for event in comm_events:
        action_name = event["action_name"]
        

        if "SUPPORT(" in action_name or "ACCUSE(" in action_name or "QUESTION(" in action_name:
            try:
                target_id = int(action_name.split("(")[1].split(")")[0])
                if "SUPPORT" in action_name:
                    targets[target_id]["support"] += 1
                elif "ACCUSE" in action_name:
                    targets[target_id]["accuse"] += 1
                elif "QUESTION" in action_name:
                    targets[target_id]["question"] += 1
            except (IndexError, ValueError):
                pass
    
    for target_id in sorted(targets.keys()):
        counts = targets[target_id]
        total = sum(counts.values())
        print(f"Agent {target_id}: "
              f"SUPPORT={counts['support']}, "
              f"ACCUSE={counts['accuse']}, "
              f"QUESTION={counts['question']} "
              f"(total={total})")
    

    print("\n5. Action Patterns")
    print("-" * 70)
    

    agent_sequences = defaultdict(list)
    for event in sorted(comm_events, key=lambda e: e["tick"]):
        agent_id = event["sender_id"]
        action_type = event["action_name"].split("(")[0]
        agent_sequences[agent_id].append(action_type)
    

    patterns = Counter()
    for agent_id, actions in agent_sequences.items():
        for i in range(len(actions) - 1):
            pattern = f"{actions[i]} → {actions[i+1]}"
            patterns[pattern] += 1
    
    if patterns:
        print("Most common action transitions:")
        for pattern, count in patterns.most_common(5):
            print(f"{pattern}: {count} times")
    else:
        print("No repeated patterns found (single actions per agent)")
    
    print("\n" + "=" * 70)


def compare_logs(log_paths: list[str]):
    """Compare communication patterns across multiple logs."""
    print("=" * 70)
    print("Comparing Communication Patterns Across Episodes")
    print("=" * 70)
    
    all_stats = []
    
    for log_path in log_paths:
        reader = EventReader(log_path)
        comm_events = reader.filter_by_type("comm_action")
        
        if not comm_events:
            continue
        
        action_counts = Counter(e["action_name"].split("(")[0] for e in comm_events)
        total = len(comm_events)
        
        stats = {
            "log": Path(log_path).name,
            "total": total,
            "support_pct": (action_counts.get("SUPPORT", 0) / total) * 100 if total > 0 else 0,
            "accuse_pct": (action_counts.get("ACCUSE", 0) / total) * 100 if total > 0 else 0,
            "defend_pct": (action_counts.get("DEFEND_SELF", 0) / total) * 100 if total > 0 else 0,
            "question_pct": (action_counts.get("QUESTION", 0) / total) * 100 if total > 0 else 0,
        }
        all_stats.append(stats)
    
    if not all_stats:
        print("\nNo communication events found in any logs.")
        return
    
    print(f"\n{len(all_stats)} episodes with communication data:")
    print("-" * 70)
    print(f"{'Episode':<30s} {'Total':>6s} {'SUPPORT':>8s} {'ACCUSE':>8s} {'DEFEND':>8s} {'QUESTION':>8s}")
    print("-" * 70)
    
    for stats in all_stats:
        print(f"{stats['log']:<30s} {stats['total']:>6d} "
              f"{stats['support_pct']:>7.1f}% "
              f"{stats['accuse_pct']:>7.1f}% "
              f"{stats['defend_pct']:>7.1f}% "
              f"{stats['question_pct']:>7.1f}%")
    

    if len(all_stats) > 1:
        print("-" * 70)
        avg_total = sum(s["total"] for s in all_stats) / len(all_stats)
        avg_support = sum(s["support_pct"] for s in all_stats) / len(all_stats)
        avg_accuse = sum(s["accuse_pct"] for s in all_stats) / len(all_stats)
        avg_defend = sum(s["defend_pct"] for s in all_stats) / len(all_stats)
        avg_question = sum(s["question_pct"] for s in all_stats) / len(all_stats)
        
        print(f"{'AVERAGE':<30s} {avg_total:>6.1f} "
              f"{avg_support:>7.1f}% "
              f"{avg_accuse:>7.1f}% "
              f"{avg_defend:>7.1f}% "
              f"{avg_question:>7.1f}%")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze communication action logs")
    parser.add_argument("logs", nargs="+", help="Path(s) to JSONL event log file(s)")
    parser.add_argument("--compare", action="store_true", 
                       help="Compare patterns across multiple logs")
    
    args = parser.parse_args()
    
    if args.compare and len(args.logs) > 1:
        compare_logs(args.logs)
    else:
        for log_path in args.logs:
            analyze_comm_actions(log_path)
            if len(args.logs) > 1:
                print("\n")


if __name__ == "__main__":
    main()