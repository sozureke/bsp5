"""SuperSuit wrappers and environment factory."""

from __future__ import annotations

from typing import Optional
import supersuit as ss

from aegis.core.rules import GameConfig
from aegis.env.pettingzoo_env import AegisEnv


def make_env(
    config: Optional[GameConfig] = None,
    render_mode: Optional[str] = None,
    log_events: bool = False,
    log_path: Optional[str] = None,
    apply_wrappers: bool = True,
) -> AegisEnv:
    """
    Create an AEGIS environment with optional SuperSuit wrappers.
    
    Args:
        config: Game configuration
        render_mode: Render mode ("human" or "ansi")
        log_events: Whether to log events to file
        log_path: Path for event log file
        apply_wrappers: Whether to apply SuperSuit wrappers
    
    Returns:
        Wrapped environment
    """
    env = AegisEnv(
        config=config,
        render_mode=render_mode,
        log_events=log_events,
        log_path=log_path,
    )
    
    if apply_wrappers:


        env = ss.pad_observations_v0(env)
        

        env = ss.pad_action_space_v0(env)
    
    return env


def make_vec_env(
    num_envs: int = 4,
    config: Optional[GameConfig] = None,
    **kwargs,
):
    """
    Create vectorized environments for parallel training.
    
    Args:
        num_envs: Number of parallel environments
        config: Game configuration
        **kwargs: Additional arguments for make_env
    
    Returns:
        Vectorized environment
    """
    def env_fn():
        return make_env(config=config, **kwargs)
    

    env = env_fn()
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_envs, num_cpus=1, base_class="gymnasium")
    
    return env

