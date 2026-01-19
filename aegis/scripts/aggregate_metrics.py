#!/usr/bin/env python3
"""Aggregate metrics from JSONL event logs into CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict


def process_event_log(log_path: str) -> dict:
    """
    Process a single event log file and extract metrics.
    
    Args:
        log_path: Path to JSONL event log
    
    Returns:
        Dictionary of extracted metrics
    """
    metrics = {
        "file": Path(log_path).name,
        "total_ticks": 0,
        "kills": 0,
        "meetings": 0,
        "ejections": 0,
        "tasks_completed": 0,
        "messages_sent": 0,
        "winner": None,
        "win_reason": None,
    }
    
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            event = json.loads(line)
            event_type = event.get("event_type")
            tick = event.get("tick", 0)
            
            metrics["total_ticks"] = max(metrics["total_ticks"], tick)
            
            if event_type == "kill":
                metrics["kills"] += 1
            elif event_type == "meeting_start":
                metrics["meetings"] += 1
            elif event_type == "ejection":
                metrics["ejections"] += 1
            elif event_type == "task_complete":
                metrics["tasks_completed"] += 1
            elif event_type == "message_sent":
                metrics["messages_sent"] += 1
            elif event_type == "win":
                metrics["winner"] = event.get("winner")
                metrics["win_reason"] = event.get("reason")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Aggregate event logs to CSV")
    parser.add_argument(
        "input",
        nargs="+",
        help="Input JSONL files or directory",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="metrics_summary.csv",
        help="Output CSV file",
    )
    
    args = parser.parse_args()
    
    # Collect input files
    input_files = []
    for path in args.input:
        p = Path(path)
        if p.is_dir():
            input_files.extend(p.glob("*.jsonl"))
        elif p.exists():
            input_files.append(p)
    
    if not input_files:
        print("No input files found")
        return
    
    print(f"Processing {len(input_files)} files...")
    
    # Process all files
    all_metrics = []
    for file_path in input_files:
        try:
            metrics = process_event_log(str(file_path))
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if not all_metrics:
        print("No metrics extracted")
        return
    
    # Write CSV
    fieldnames = list(all_metrics[0].keys())
    
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)
    
    print(f"Wrote {len(all_metrics)} records to {args.output}")
    
    # Print summary
    survivor_wins = sum(1 for m in all_metrics if m["winner"] == 0)
    impostor_wins = sum(1 for m in all_metrics if m["winner"] == 1)
    total = len(all_metrics)
    
    print(f"\nSummary:")
    print(f"  Survivor wins: {survivor_wins}/{total} ({100*survivor_wins/total:.1f}%)")
    print(f"  Impostor wins: {impostor_wins}/{total} ({100*impostor_wins/total:.1f}%)")
    print(f"  Avg ticks: {sum(m['total_ticks'] for m in all_metrics)/total:.1f}")
    print(f"  Avg kills: {sum(m['kills'] for m in all_metrics)/total:.2f}")


if __name__ == "__main__":
    main()

