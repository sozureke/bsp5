import argparse
import os
import subprocess
import sys
from pathlib import Path

from aegis.train.seeds import SEEDS, SEEDS_TEST


def run_experiment(config_path: str, seed: int, dry_run: bool = False):
    """Run a single experiment with a given seed."""
    configs_dir = Path(__file__).parent.parent / "train" / "configs"
    config_file = Path(config_path)
    
    if not config_file.is_absolute():
        config_file = configs_dir / config_path
    
    experiment_name = config_file.stem.replace(".config", "")
    
    checkpoint_dir = Path("checkpoints")
    experiment_dir = checkpoint_dir / experiment_name
    seed_dir = experiment_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    event_log_path = seed_dir / "events.jsonl"
    
    cmd = [
        sys.executable,
        "-m", "aegis.train.rl_ppo",
        "-c", str(config_file),
        "--seed", str(seed),
        "--checkpoint-dir", str(seed_dir),
    ]
    
    if dry_run:
        print(f"[DRY RUN] {' '.join(cmd)}")
        print(f"  Event log: {event_log_path}")
        return True
    
    print(f"\n{'='*70}")
    print(f"Running: {experiment_name} with seed {seed}")
    print(f"Output directory: {seed_dir}")
    print(f"Event log: {event_log_path}")
    print(f"{'='*70}\n")
    
    try:
        env = os.environ.copy()
        env["AEGIS_EVENT_LOG_PATH"] = str(event_log_path)
        result = subprocess.run(cmd, check=True, env=env)
        print(f"\n✓ Completed: {experiment_name} with seed {seed}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {experiment_name} with seed {seed}")
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments with shared seed set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run E1 with all seeds
  python -m aegis.scripts.run_experiments e1_baseline.config.yaml

  # Run E1 with test seeds (first 10)
  python -m aegis.scripts.run_experiments e1_baseline.config.yaml --test

  # Run multiple experiments with all seeds
  python -m aegis.scripts.run_experiments e1_baseline.config.yaml e2_trust_only.config.yaml

  # Dry run (show commands without executing)
  python -m aegis.scripts.run_experiments e1_baseline.config.yaml --dry-run
        """
    )
    
    parser.add_argument(
        "configs",
        nargs="+",
        help="Config file(s) to run (relative to aegis/train/configs/)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use test seed set (first 10 seeds) instead of full set"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands without executing"
    )
    
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="Limit number of seeds to use (for testing)"
    )
    args = parser.parse_args()
    configs_dir = Path(__file__).parent.parent / "train" / "configs"
    
    seeds = SEEDS_TEST if args.test else SEEDS
    if args.max_seeds:
        seeds = seeds[:args.max_seeds]
    
    print(f"Using {len(seeds)} seeds: {seeds[:5]}..." if len(seeds) > 5 else f"Using {len(seeds)} seeds: {seeds}")
    print(f"Running {len(args.configs)} experiment(s): {', '.join(args.configs)}")
    print(f"Total runs: {len(args.configs) * len(seeds)}")
    
    if args.dry_run:
        print("\n[DRY RUN MODE - No experiments will be executed]\n")
    
    results = []
    
    for config_name in args.configs:
        config_path = configs_dir / config_name
        if not config_path.exists():
            print(f"Configuration not found: {config_path}")
            continue
        
        for seed in seeds:
            success = run_experiment(str(config_path), seed, dry_run=args.dry_run)
            results.append((config_name, seed, success))
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    successful = sum(1 for _, _, success in results if success)
    total = len(results)
    
    print(f"Successful: {successful}/{total}")
    print(f"Failed: {total - successful}/{total}")
    
    if total - successful > 0:
        print("\nFailed runs:")
        for config_name, seed, success in results:
            if not success:
                print(f"- {config_name} (seed {seed})")
    
    if successful > 0 and not args.dry_run:
        print(f"\n{'='*70}")
        print("NEXT STEPS")
        print(f"{'='*70}")
        print("To merge events from all seeds for analysis, run:")
        for config_name in set(cn for cn, _, _ in results):
            config_file = configs_dir / config_name
            experiment_name = config_file.stem.replace(".config", "")
            experiment_dir = Path("checkpoints") / experiment_name
            print(f"python -m aegis.scripts.merge_events {experiment_dir}")
        print("\nThen analyze the merged file:")
        for config_name in set(cn for cn, _, _ in results):
            config_file = configs_dir / config_name
            experiment_name = config_file.stem.replace(".config", "")
            print(f"  python -m aegis.scripts.analyze_events checkpoints/{experiment_name}/events_all.jsonl --summary")
    
    return 0 if successful == total else 1


if __name__ == "__main__":
    sys.exit(main())

