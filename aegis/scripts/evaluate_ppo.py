from __future__ import annotations

import argparse
import time
from typing import Optional

import numpy as np
import torch

from aegis.core.rules import GameConfig
from aegis.core.world import Role
from aegis.env.pettingzoo_env import AegisEnv
from aegis.train.rl_ppo import ActorCritic, flatten_obs, get_flat_obs_dim, get_action_dim, Args


def evaluate(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    num_episodes: int = 100,
    render: bool = False,
    delay: float = 0.0,
    seed: Optional[int] = None,
    device: str = "cpu",
):
    """Evaluate a trained PPO model."""
    
    # Load args from config or use defaults
    if config_path:
        args = Args.from_yaml(config_path)
    else:
        args = Args()
    
    if seed is not None:
        args = args.override(seed=seed)
    
    # Create environment
    if args.game_config:
        config = GameConfig(**args.game_config, seed=args.seed)
    else:
        config = GameConfig(seed=args.seed)
    
    env = AegisEnv(config=config, render_mode="human" if render else None)
    
    # Get dimensions
    obs_dim = get_flat_obs_dim(env)
    action_dim = get_action_dim(env)
    
    # Create and load model
    agent = ActorCritic(obs_dim, action_dim, args.hidden_dim).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()
    
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Global step: {checkpoint.get('global_step', 'N/A')}")
    print(f"  Num updates: {checkpoint.get('num_updates', 'N/A')}")
    print()
    
    # Run evaluation
    agent_names = env.possible_agents
    survivor_wins = 0
    impostor_wins = 0
    all_rewards = []
    all_lengths = []
    
    print(f"Evaluating on {num_episodes} episodes...")
    print("-" * 60)
    
    for ep in range(num_episodes):
        obs_dict, infos = env.reset()
        agent_roles = {name: infos[name]["role"] for name in agent_names}
        
        if render:
            print(f"\n{'='*60}")
            print(f"EPISODE {ep + 1}")
            print(f"{'='*60}")
            print(f"Roles: {agent_roles}")
            env.render()
        
        done = False
        episode_reward = 0.0
        step_count = 0
        
        while not done:
            obs_flat = []
            masks = []
            roles = []
            for name in agent_names:
                obs_flat.append(flatten_obs(obs_dict[name]))
                masks.append(obs_dict[name]["action_mask"])
                roles.append(obs_dict[name]["self_role"])
            
            obs_tensor = torch.tensor(np.array(obs_flat), dtype=torch.float32, device=device)
            mask_tensor = torch.tensor(np.array(masks), dtype=torch.float32, device=device)
            roles_tensor = torch.tensor(np.array(roles), dtype=torch.long, device=device)
            
            # Get actions (deterministic for evaluation)
            with torch.no_grad():
                actions, _, _, _ = agent.get_action_and_value(obs_tensor, mask_tensor, roles_tensor)
            
            actions_dict = {name: actions[i].item() for i, name in enumerate(agent_names)}
            
            # Step
            obs_dict, rewards, terminations, truncations, infos = env.step(actions_dict)
            episode_reward += sum(rewards.values())
            step_count += 1
            
            done = all(terminations.values()) or all(truncations.values())
            
            if render:
                print(f"\n--- Step {step_count} ---")
                env.render()
                nonzero_rewards = {k: v for k, v in rewards.items() if v != 0}
                if nonzero_rewards:
                    print(f"Rewards: {nonzero_rewards}")
                if delay > 0:
                    time.sleep(delay)
        
        # Determine winner
        winner = None
        for info in infos.values():
            if info.get("winner") is not None:
                winner = info["winner"]
                break
        
        if winner == Role.SURVIVOR:
            survivor_wins += 1
            winner_str = "SURVIVOR"
        else:
            impostor_wins += 1
            winner_str = "IMPOSTOR"
        
        all_rewards.append(episode_reward / len(agent_names))
        all_lengths.append(step_count)
        
        print(f"Episode {ep + 1}: {winner_str} wins in {step_count} steps (reward: {episode_reward/len(agent_names):.3f})")
    
    # Summary
    print()
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episodes: {num_episodes}")
    print()
    print(f"Survivor wins: {survivor_wins} ({100*survivor_wins/num_episodes:.1f}%)")
    print(f"Impostor wins: {impostor_wins} ({100*impostor_wins/num_episodes:.1f}%)")
    print()
    print(f"Mean reward: {np.mean(all_rewards):.3f} ± {np.std(all_rewards):.3f}")
    print(f"Mean episode length: {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f}")
    
    env.close()
    
    return {
        "survivor_wins": survivor_wins,
        "impostor_wins": impostor_wins,
        "win_rate_survivor": survivor_wins / num_episodes,
        "mean_reward": np.mean(all_rewards),
        "std_reward": np.std(all_rewards),
        "mean_length": np.mean(all_lengths),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO model")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("-c", "--config", type=str, help="Path to YAML config")
    parser.add_argument("-n", "--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("-r", "--render", action="store_true", help="Render episodes")
    parser.add_argument("-d", "--delay", type=float, default=0.0, help="Delay between steps")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    
    args = parser.parse_args()
    
    evaluate(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        num_episodes=args.episodes,
        render=args.render,
        delay=args.delay,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()

