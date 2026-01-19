#!/usr/bin/env python3
"""Run episodes with heuristic agents and simple text rendering."""

from __future__ import annotations

import argparse
import time
from typing import Optional

from aegis.core.rules import GameConfig
from aegis.core.world import Role
from aegis.env.pettingzoo_env import AegisEnv
from aegis.agents.random_policy import RandomPolicy
from aegis.agents.heuristic_survivor import HeuristicSurvivorPolicy
from aegis.agents.heuristic_impostor import HeuristicImpostorPolicy
from aegis.logging.metrics import MetricsAggregator


def run_episode(
    env: AegisEnv,
    survivor_policy: HeuristicSurvivorPolicy,
    impostor_policy: HeuristicImpostorPolicy,
    render: bool = False,
    delay: float = 0.0,
) -> dict:
    """
    Run a single episode with heuristic agents.
    
    Args:
        env: AEGIS environment
        survivor_policy: Policy for survivors
        impostor_policy: Policy for impostors
        render: Whether to render each step
        delay: Delay between steps (for visualization)
    
    Returns:
        Episode metrics dictionary
    """
    observations, infos = env.reset()
    
    # Track roles
    agent_roles = {name: infos[name]["role"] for name in env.possible_agents}
    
    if render:
        print("\n" + "=" * 60)
        print("EPISODE START")
        print("=" * 60)
        print(f"Roles: {agent_roles}")
        env.render()
    
    done = False
    step_count = 0
    
    while not done:
        # Get actions for all active agents
        actions = {}
        
        for agent_name in env.agents:
            obs = observations[agent_name]
            role = agent_roles[agent_name]
            
            if role == Role.SURVIVOR:
                action = survivor_policy.get_action(obs, agent_name)
            else:
                action = impostor_policy.get_action(obs, agent_name)
            
            actions[agent_name] = action
        
        # Step environment
        observations, rewards, terminations, truncations, infos = env.step(actions)
        step_count += 1
        
        # Check if done
        done = all(terminations.values()) or all(truncations.values())
        
        if render:
            print(f"\n--- Step {step_count} ---")
            env.render()
            
            # Show rewards
            nonzero_rewards = {k: v for k, v in rewards.items() if v != 0}
            if nonzero_rewards:
                print(f"Rewards: {nonzero_rewards}")
            
            if delay > 0:
                time.sleep(delay)
    
    # Get final info
    winner = None
    for name, info in infos.items():
        if info.get("winner") is not None:
            winner = info["winner"]
            break
    
    return {
        "steps": step_count,
        "winner": "SURVIVOR" if winner == Role.SURVIVOR else "IMPOSTOR",
        "metrics": env.metrics.to_dict() if hasattr(env, "metrics") else {},
    }


def main():
    parser = argparse.ArgumentParser(description="Run AEGIS episodes with heuristic agents")
    parser.add_argument(
        "--episodes", "-n",
        type=int,
        default=5,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--render", "-r",
        action="store_true",
        help="Render environment output",
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=0.0,
        help="Delay between steps (seconds)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--survivors",
        type=int,
        default=4,
        help="Number of survivors",
    )
    parser.add_argument(
        "--impostors",
        type=int,
        default=1,
        help="Number of impostors",
    )
    parser.add_argument(
        "--vision",
        type=int,
        default=2,
        help="Vision radius",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=500,
        help="Maximum ticks per episode",
    )
    
    args = parser.parse_args()
    
    # Create config
    config = GameConfig(
        num_survivors=args.survivors,
        num_impostors=args.impostors,
        vision_radius=args.vision,
        max_ticks=args.max_ticks,
        seed=args.seed,
    )
    
    # Create environment
    env = AegisEnv(config=config, render_mode="human" if args.render else None)
    
    # Create policies
    num_rooms = len(env.engine.rooms)
    num_agents = config.num_agents
    
    survivor_policy = HeuristicSurvivorPolicy(
        config=config,
        num_rooms=num_rooms,
        num_agents=num_agents,
        seed=args.seed,
    )
    
    impostor_policy = HeuristicImpostorPolicy(
        config=config,
        num_rooms=num_rooms,
        num_agents=num_agents,
        seed=args.seed,
    )
    
    # Run episodes
    aggregator = MetricsAggregator()
    survivor_wins = 0
    impostor_wins = 0
    
    print(f"\nRunning {args.episodes} episodes...")
    print(f"Config: {args.survivors} survivors, {args.impostors} impostors, vision={args.vision}")
    print("-" * 60)
    
    for ep in range(args.episodes):
        result = run_episode(
            env=env,
            survivor_policy=survivor_policy,
            impostor_policy=impostor_policy,
            render=args.render,
            delay=args.delay,
        )
        
        if result["winner"] == "SURVIVOR":
            survivor_wins += 1
        else:
            impostor_wins += 1
        
        if hasattr(env, "metrics"):
            aggregator.add(env.metrics)
        
        print(f"Episode {ep + 1}: {result['winner']} wins in {result['steps']} steps")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total episodes: {args.episodes}")
    print(f"Survivor wins: {survivor_wins} ({100*survivor_wins/args.episodes:.1f}%)")
    print(f"Impostor wins: {impostor_wins} ({100*impostor_wins/args.episodes:.1f}%)")
    
    if aggregator.episodes:
        summary = aggregator.summary()
        print(f"\nAverage episode length: {summary['avg_episode_length']:.1f}")
        print(f"Average kills: {summary['avg_kills']:.2f}")
        print(f"Average meetings: {summary['avg_meetings']:.2f}")
        print(f"Task completion rate: {100*summary['task_completion_rate']:.1f}%")
    
    env.close()


if __name__ == "__main__":
    main()

