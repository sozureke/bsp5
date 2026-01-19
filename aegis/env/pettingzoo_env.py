"""PettingZoo ParallelEnv wrapper around core engine."""

from __future__ import annotations

from typing import Optional, Any
import functools

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from aegis.core.world import WorldState, Role, Phase
from aegis.core.rules import GameConfig
from aegis.core.engine import StepEngine
from aegis.core.events import Event, EventType
from aegis.obs.builder import ObservationBuilder
from aegis.obs.spaces import ObservationSpaces
from aegis.logging.event_writer import EventWriter
from aegis.logging.metrics import EpisodeMetrics


class AegisEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv for the AEGIS communication-deduction game.
    
    All agents act simultaneously each tick.
    """
    
    metadata = {"render_modes": ["human", "ansi"], "name": "aegis_v0"}
    
    def __init__(
        self,
        config: Optional[GameConfig] = None,
        render_mode: Optional[str] = None,
        log_events: bool = False,
        log_path: Optional[str] = None,
    ):
        super().__init__()
        
        self.config = config or GameConfig()
        self.render_mode = render_mode
        self.log_events = log_events
        
        # Initialize engine
        self.engine = StepEngine(self.config)
        
        # Initialize observation builder
        self.obs_builder = ObservationBuilder(self.engine, self.config)
        
        # Initialize observation spaces helper
        self.obs_spaces = ObservationSpaces(
            self.config,
            num_rooms=len(self.engine.rooms),
            num_edges=len(self.engine.edges),
        )
        
        # Agent IDs
        self.possible_agents = [f"agent_{i}" for i in range(self.config.num_agents)]
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}
        
        # Event logging
        self.event_writer = EventWriter(log_path) if log_events and log_path else None
        
        # Episode metrics
        self.metrics = EpisodeMetrics()
        
        # State
        self.world: Optional[WorldState] = None
        self.agents: list[str] = []
        self._cumulative_rewards: dict[str, float] = {}
        
    def _agent_id_to_name(self, agent_id: int) -> str:
        """Convert numeric agent ID to string name."""
        return f"agent_{agent_id}"
    
    def _name_to_agent_id(self, name: str) -> int:
        """Convert string name to numeric agent ID."""
        return self.agent_name_mapping[name]
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Dict:
        """Return observation space for an agent."""
        return self.obs_spaces.get_observation_space()
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Discrete:
        """Return action space for an agent."""
        return self.obs_spaces.get_action_space()
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict[str, dict], dict[str, dict]]:
        """Reset the environment."""
        # Reset engine with seed
        actual_seed = seed if seed is not None else self.config.seed
        self.world, init_events = self.engine.create_initial_state(actual_seed)
        
        # Reset agent list (all agents alive)
        self.agents = list(self.possible_agents)
        
        # Reset cumulative rewards
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        
        # Reset metrics
        self.metrics = EpisodeMetrics()
        
        # Log initial events
        if self.event_writer:
            for event in init_events:
                self.event_writer.write(event)
        
        # Build observations
        observations = {}
        infos = {}
        for agent_name in self.agents:
            agent_id = self._name_to_agent_id(agent_name)
            observations[agent_name] = self.obs_builder.build(self.world, agent_id)
            infos[agent_name] = {"role": int(self.world.agents[agent_id].role)}
        
        return observations, infos
    
    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[
        dict[str, dict],  # observations
        dict[str, float],  # rewards
        dict[str, bool],  # terminations
        dict[str, bool],  # truncations
        dict[str, dict],  # infos
    ]:
        """Execute one step with joint actions."""
        if self.world is None:
            raise RuntimeError("Environment not reset")
        
        # Convert string agent names to numeric IDs
        joint_actions = {}
        for agent_name, action in actions.items():
            if agent_name in self.agents:
                agent_id = self._name_to_agent_id(agent_name)
                joint_actions[agent_id] = action
        
        # Fill in default actions for agents that didn't provide one
        for agent_name in self.agents:
            agent_id = self._name_to_agent_id(agent_name)
            if agent_id not in joint_actions:
                # Default to first valid action (usually MOVE_0 / stay)
                joint_actions[agent_id] = 0
        
        # Step the simulation
        next_world, events = self.engine.step(self.world, joint_actions)
        
        # Log events
        if self.event_writer:
            for event in events:
                self.event_writer.write(event)
        
        # Update metrics
        self._update_metrics(events)
        
        # Compute rewards
        rewards = self._compute_rewards(self.world, next_world, events, joint_actions)
        
        # Update world
        self.world = next_world
        
        # Build observations
        observations = {}
        terminations = {}
        truncations = {}
        infos = {}
        
        for agent_name in self.possible_agents:
            agent_id = self._name_to_agent_id(agent_name)
            agent = self.world.agents[agent_id]
            
            observations[agent_name] = self.obs_builder.build(self.world, agent_id)
            terminations[agent_name] = self.world.terminated
            truncations[agent_name] = self.world.truncated
            infos[agent_name] = {
                "role": int(agent.role),
                "alive": agent.alive,
                "winner": self.world.winner if self.world.terminated else None,
            }
        
        # Update active agents list (remove dead agents from active list)
        self.agents = [
            name for name in self.possible_agents
            if self.world.agents[self._name_to_agent_id(name)].alive
        ]
        
        # If terminated, finalize metrics
        if self.world.terminated or self.world.truncated:
            self.metrics.finalize(self.world)
            if self.event_writer:
                self.event_writer.flush()
        
        return observations, rewards, terminations, truncations, infos
    
    def _compute_rewards(
        self,
        old_world: WorldState,
        new_world: WorldState,
        events: list[Event],
        actions: dict[int, int],
    ) -> dict[str, float]:
        """Compute rewards for all agents."""
        rewards = {agent: 0.0 for agent in self.possible_agents}
        config = self.config
        
        # Process events
        for event in events:
            if event.event_type == EventType.WIN:
                winner = Role(event.data["winner"])
                for agent_name in self.possible_agents:
                    agent_id = self._name_to_agent_id(agent_name)
                    agent = new_world.agents[agent_id]
                    if agent.role == winner:
                        rewards[agent_name] += config.reward_win
                    else:
                        rewards[agent_name] += config.reward_loss
            
            elif event.event_type == EventType.KILL:
                killer_id = event.data["killer_id"]
                killer_name = self._agent_id_to_name(killer_id)
                rewards[killer_name] += config.reward_kill
            
            elif event.event_type == EventType.TASK_STEP_COMPLETE:
                # Team reward for task progress
                for agent_name in self.possible_agents:
                    agent_id = self._name_to_agent_id(agent_name)
                    if new_world.agents[agent_id].role == Role.SURVIVOR:
                        rewards[agent_name] += config.reward_task_step
            
            elif event.event_type == EventType.REPORT:
                reporter_id = event.data["reporter_id"]
                reporter_name = self._agent_id_to_name(reporter_id)
                rewards[reporter_name] += config.reward_correct_report
            
            elif event.event_type == EventType.EJECTION:
                ejected_id = event.data["ejected_id"]
                ejected_role = Role(event.data["role"])
                
                for agent_name in self.possible_agents:
                    agent_id = self._name_to_agent_id(agent_name)
                    agent = new_world.agents[agent_id]
                    
                    if agent.role == Role.SURVIVOR:
                        if ejected_role == Role.IMPOSTOR:
                            rewards[agent_name] += config.reward_survivor_ejected_impostor
                        else:
                            rewards[agent_name] += config.reward_survivor_ejected_survivor
                    else:  # Impostor
                        if ejected_role == Role.SURVIVOR:
                            rewards[agent_name] += config.reward_impostor_ejected_survivor
                        else:
                            rewards[agent_name] += config.reward_impostor_ejected_impostor
        
        # Time penalty for all alive agents
        for agent_name in self.possible_agents:
            agent_id = self._name_to_agent_id(agent_name)
            if new_world.agents[agent_id].alive:
                rewards[agent_name] += config.reward_time_penalty
        
        # Proximity penalty for survivors near impostors (reward shaping)
        for agent_name in self.possible_agents:
            agent_id = self._name_to_agent_id(agent_name)
            agent = new_world.agents[agent_id]
            
            if agent.role == Role.SURVIVOR and agent.alive:
                # Check if in same room as any impostor
                impostor_in_room = False
                for other_id, other in new_world.agents.items():
                    if (other.role == Role.IMPOSTOR and 
                        other.alive and 
                        other.room == agent.room):
                        impostor_in_room = True
                        break
                
                if impostor_in_room:
                    # Update proximity counter
                    agent.proximity_to_impostor_ticks += 1
                    # Apply penalty if threshold reached
                    if agent.proximity_to_impostor_ticks >= config.proximity_penalty_ticks:
                        rewards[agent_name] += config.reward_proximity_penalty
                else:
                    # Reset counter when not near impostor
                    agent.proximity_to_impostor_ticks = 0
        
        return rewards
    
    def _update_metrics(self, events: list[Event]) -> None:
        """Update episode metrics based on events."""
        for event in events:
            if event.event_type == EventType.KILL:
                self.metrics.kills += 1
            elif event.event_type == EventType.MEETING_START:
                self.metrics.meetings += 1
            elif event.event_type == EventType.TASK_STEP_COMPLETE:
                self.metrics.tasks_completed += 1
            elif event.event_type == EventType.EJECTION:
                self.metrics.ejections += 1
            elif event.event_type == EventType.MESSAGE_SENT:
                self.metrics.messages_sent += 1
    
    def render(self) -> Optional[str]:
        """Render the environment."""
        if self.world is None:
            return None
        
        if self.render_mode == "ansi" or self.render_mode == "human":
            return self._render_text()
        
        return None
    
    def _render_text(self) -> str:
        """Render environment as text."""
        lines = []
        world = self.world
        
        lines.append(f"=== Tick {world.tick} | Phase: {world.phase.name} ===")
        lines.append(f"EVAC Active: {world.evac_active} | EVAC Counter: {world.evac_tick_counter}")
        
        # Room layout (3x3 grid)
        lines.append("\nMap:")
        for row in range(3):
            row_str = ""
            for col in range(3):
                room_id = row * 3 + col
                agents_in_room = [
                    a for a in world.agents.values()
                    if a.room == room_id and a.alive
                ]
                bodies_in_room = [b for b in world.bodies if b.room == room_id]
                
                if room_id == world.evac_room:
                    cell = f"[E{room_id}]"
                else:
                    cell = f"[R{room_id}]"
                
                if agents_in_room:
                    agent_chars = []
                    for a in agents_in_room:
                        char = "I" if a.role == Role.IMPOSTOR else "S"
                        agent_chars.append(f"{char}{a.agent_id}")
                    cell += "(" + ",".join(agent_chars) + ")"
                
                if bodies_in_room:
                    cell += " X"
                
                row_str += f"{cell:20}"
            lines.append(row_str)
        
        # Closed doors
        closed_doors = [e for e, d in world.doors.items() if not d.is_open]
        if closed_doors:
            lines.append(f"\nClosed doors: {closed_doors}")
        
        # Agent status
        lines.append("\nAgents:")
        for agent in world.agents.values():
            role = "IMP" if agent.role == Role.IMPOSTOR else "SRV"
            status = "ALIVE" if agent.alive else "DEAD"
            task_info = ""
            if agent.role == Role.SURVIVOR:
                completed = sum(1 for t in agent.tasks if t.completed)
                task_info = f" | Tasks: {completed}/{len(agent.tasks)}"
            cooldown_info = ""
            if agent.role == Role.IMPOSTOR:
                cooldown_info = f" | Kill CD: {agent.kill_cooldown}, Door CD: {agent.door_cooldown}"
            lines.append(f"  Agent {agent.agent_id} [{role}] {status} in Room {agent.room}{task_info}{cooldown_info}")
        
        # Meeting info
        if world.meeting:
            lines.append(f"\nMeeting: Reporter={world.meeting.reporter_id}, Body in Room {world.meeting.body_location}")
            lines.append(f"Timer: {world.meeting.phase_timer} | Messages: {len(world.meeting.messages)}")
            lines.append(f"Votes: {world.meeting.votes}")
        
        # Task progress
        lines.append(f"\nTeam Task Progress: {world.team_task_progress()}/{world.total_task_steps()}")
        
        if world.terminated:
            winner = "SURVIVORS" if world.winner == Role.SURVIVOR else "IMPOSTORS"
            lines.append(f"\n*** GAME OVER: {winner} WIN! ***")
        
        result = "\n".join(lines)
        
        if self.render_mode == "human":
            print(result)
        
        return result
    
    def close(self) -> None:
        """Close the environment."""
        if self.event_writer:
            self.event_writer.close()

