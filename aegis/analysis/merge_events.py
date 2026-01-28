import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict


def merge_events(experiment_dir: Path, output_file: str = None) -> Path:
    """
    Merge all events.jsonl files from seed subdirectories.
    
    Args:
        experiment_dir: Path to experiment directory (e.g., checkpoints/E1_baseline)
        output_file: Optional output file path. Defaults to experiment_dir/events_all.jsonl
    
    Returns:
        Path to merged events file
    """
    experiment_dir = Path(experiment_dir)
    
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    seed_dirs = sorted([d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
    
    if not seed_dirs:
        raise ValueError(f"No seed directories found in {experiment_dir}")
    
    print(f"Found {len(seed_dirs)} seed directories in {experiment_dir}")
    
    all_events: List[Dict] = []
    
    for seed_dir in seed_dirs:
        events_file = seed_dir / "events.jsonl"
        if not events_file.exists():
            print(f"Warning: {events_file} not found, skipping")
            continue
        
        seed = seed_dir.name.replace("seed_", "")
        print(f"Loading {seed_dir.name}...", end=" ")
        
        with open(events_file, "r") as f:
            count = 0
            for line in f:
                line = line.strip()
                if line:
                    event = json.loads(line)
                    event["_seed"] = int(seed)
                    all_events.append(event)
                    count += 1
        print(f"{count} events")
    
    if not all_events:
        raise ValueError("No events found in any seed directory")
    
    all_events.sort(key=lambda e: (e.get("tick", 0), e.get("_seed", 0)))
    
    if output_file is None:
        output_file = experiment_dir / "events_all.jsonl"
    else:
        output_file = Path(output_file)
    
    print(f"\nWriting {len(all_events)} merged events to {output_file}")
    
    with open(output_file, "w") as f:
        for event in all_events:
            seed = event.pop("_seed")
            f.write(json.dumps(event) + "\n")
    
    print(f"✓ Merged events saved to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Merge events.jsonl from multiple seeds into a single file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge events from all seeds in E1_baseline
  python -m aegis.analysis.merge_events checkpoints/E1_baseline

  # Merge and specify output file
  python -m aegis.analysis.merge_events checkpoints/E1_baseline -o checkpoints/E1_baseline/merged.jsonl
        """
    )
    
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to experiment directory (e.g., checkpoints/E1_baseline)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path (default: experiment_dir/events_all.jsonl)"
    )
    
    args = parser.parse_args()
    
    try:
        output_path = merge_events(Path(args.experiment_dir), args.output)
        print(f"\nSuccess! Merged events available at: {output_path}")
        print(f"\nYou can now analyze with:")
        print(f"python -m aegis.analysis.analyze_events {output_path} --summary")
        return 0
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 


if __name__ == "__main__":
    sys.exit(main())

