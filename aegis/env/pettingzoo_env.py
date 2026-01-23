from __future__ import annotations

from typing import Optional, Any
from dataclasses import dataclass
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
from aegis.comms.trust import TrustManager
from aegis.comms.actions import CommVocab


@dataclass
class SuspectEvent:
    """Represents a suspicious event that may become active after delay."""
    target_agent: int
    event_type: str
    tick_created: int
    suspicion_value: float


class AegisEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv for the AEGIS communication-deduction game.
    
    All agents act simultaneously each tick.
    """
    
    metadata = {"render_modes": ["human",  "ansi"],  "name":  "aegis_v0"}
    
    def __init__(
        self,
        config: Optional[GameConfig] = None,
        render_mode: Optional[str] = None,
        log_events: bool = False,
        log_path: Optional[str] = None,
        log_event_types: Optional[list[str]] = None,
    ):
        super().__init__()
        
        self.config = config or GameConfig()
        self.render_mode = render_mode
        self.log_events = log_events
        self.log_event_types = set(log_event_types) if log_event_types else None
        
        self.engine = StepEngine(self.config)
        
        self.trust_manager: Optional[TrustManager] = None
        if self.config.enable_trust_comm:
            self.trust_manager = TrustManager(
                num_agents=self.config.num_agents,
                delay_ticks=self.config.trust_delay_ticks,
                initial_trust=self.config.trust_initial_value,
            )
        
        self.obs_builder = ObservationBuilder(self.engine, self.config, trust_manager=self.trust_manager)
        
        self.obs_spaces = ObservationSpaces(
            self.config,
            num_rooms=len(self.engine.rooms),
            num_edges=len(self.engine.edges),
        )
        
        self.possible_agents = [f"agent_{i}" for i in range(self.config.num_agents)]
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}
        
        self.event_writer = EventWriter(log_path) if log_events and log_path else None
        
        self.metrics = EpisodeMetrics()
        
        self.world: Optional[WorldState] = None
        self.agents: list[str] = []
        self._cumulative_rewards: dict[str, float] = {}
        
        self._comm_action_selections: int = 0
        self._total_meeting_steps: int = 0
        
        self.suspicion_score: dict[int, float] = {}
        
        self.ticks_since_kill: dict[int, int] = {}
        
        self.proximity_ticks: dict[int, int] = {}
        
        self.pending_suspicions: dict[int, list[SuspectEvent]] = {}
        self.active_suspicions: dict[int, list[SuspectEvent]] = {}
        
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
        actual_seed = seed if seed is not None else self.config.seed
        self.world, init_events = self.engine.create_initial_state(actual_seed)
        
        self.agents = list(self.possible_agents)
        
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        
        self._comm_action_selections = 0
        self._total_meeting_steps = 0
        
        self.metrics = EpisodeMetrics()
        
        self.suspicion_score = {i: 0.0 for i in range(self.config.num_agents)}
        
        self.ticks_since_kill = {i: 0 for i in range(self.config.num_agents)}
        
        self.proximity_ticks = {i: 0 for i in range(self.config.num_agents)}
        
        self.pending_suspicions = {i: [] for i in range(self.config.num_agents)}
        self.active_suspicions = {i: [] for i in range(self.config.num_agents)}
        
        if self.trust_manager is not None:
            self.trust_manager.reset(initial_trust=self.config.trust_initial_value)
        
        if self.event_writer:
            for event in init_events:
                if self._should_log_event(event):
                    self.event_writer.write(event)
        

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
        dict[str, dict],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """Execute one step with joint actions."""
        if self.world is None:
            raise RuntimeError("Environment not reset")
        
        joint_actions = {}
        for agent_name, action in actions.items():
            if agent_name in self.agents:
                agent_id = self._name_to_agent_id(agent_name)
                joint_actions[agent_id] = action
        
        for agent_name in self.agents:
            agent_id = self._name_to_agent_id(agent_name)
            if agent_id not in joint_actions:
                joint_actions[agent_id] = 0
        
        old_world = self.world
        
        if old_world.phase == Phase.MEETING:
            self._total_meeting_steps += 1
            for agent_name, action in actions.items():
                if agent_name in self.agents:
                    agent_id = self._name_to_agent_id(agent_name)
                    action_type, param = self.engine.parse_action(action)
                    if action_type ==  "comm_action":
                        self._comm_action_selections += 1
        
        self._update_proximity_tracking(old_world)        
        self._activate_delayed_suspicions(self.world.tick)        
        self._decay_suspicion(old_world)
        
        if self.trust_manager is not None and self.world.meeting is not None:
            self._process_comm_actions(self.world, joint_actions)
            self.trust_manager.activate_delayed_updates(self.world.tick)
        
        suspicion_scores = self._compute_voting_scores()
        next_world, events = self.engine.step(self.world, joint_actions, suspicion_scores=suspicion_scores)
        
        if self.event_writer:
            for event in events:
                if self._should_log_event(event):
                    self.event_writer.write(event)
        
        self._update_suspicion_from_events(events, old_world, next_world)
        
        for event in events:
            if event.event_type == EventType.MEETING_END:
                for agent_id in range(self.config.num_agents):
                    self.pending_suspicions[agent_id] = []
                if self.trust_manager is not None:
                    self.trust_manager.clear_pending_updates()
                break
        
        self._update_metrics(events)
        
        agent_events = self._extract_agent_events(events, old_world, next_world)
        
        rewards = self._compute_rewards(old_world, next_world, events, joint_actions)
        
        authoritative_winner = None
        if next_world.terminated or next_world.truncated:
            if next_world.winner is not None:
                authoritative_winner = int(next_world.winner)
        
        observations = {}
        terminations = {}
        truncations = {}
        infos = {}
        
        for agent_name in self.possible_agents:
            agent_id = self._name_to_agent_id(agent_name)
            agent = next_world.agents[agent_id]
            
            observations[agent_name] = self.obs_builder.build(next_world, agent_id)
            terminations[agent_name] = next_world.terminated
            truncations[agent_name] = next_world.truncated
            
            agent_event_info = agent_events.get(agent_name, {})
            
            infos[agent_name] = {
                "role": int(agent.role),
                "alive": agent.alive,
                "winner": authoritative_winner,
                "events": agent_event_info.get("events", []),
                "did_kill": agent_event_info.get("did_kill", False),
                "did_task_step": agent_event_info.get("did_task_step", False),
                "was_killed": agent_event_info.get("was_killed", False),
                "was_ejected": agent_event_info.get("was_ejected", False),
                "team_task_step": agent_event_info.get("team_task_step", False),
                "did_report": agent_event_info.get("did_report", False),
                "ejection_benefited": agent_event_info.get("ejection_benefited", False),
                "ejection_penalized": agent_event_info.get("ejection_penalized", False),
                "correct_accuse": agent_event_info.get("correct_accuse", False),
                "false_accuse": agent_event_info.get("false_accuse", False),
            }
            
            was_alive = old_world.agents[agent_id].alive if agent_id in old_world.agents else True
            if not agent.alive and was_alive:
                was_killed = agent_event_info.get("was_killed", False)
                was_ejected = agent_event_info.get("was_ejected", False)
                
                if agent.role == Role.IMPOSTOR:
                    if was_killed:
                        raise RuntimeError(
                            f"Invariant violated: impostor {agent_id} was killed. "
                            "Impostors can only be ejected, never killed. Check engine logic."
                        )
                    if not was_ejected:
                        raise RuntimeError(
                            f"Invariant violated: impostor {agent_id} died but was_ejected=False. "
                            "Impostors can only leave via ejection."
                        )
                else:
                    if was_killed == was_ejected:
                        raise RuntimeError(
                            f"Invariant violated: survivor {agent_id} died but elimination flags inconsistent. "
                            f"was_killed={was_killed}, was_ejected={was_ejected}. "
                            "Exactly one must be True."
                        )
        
        self.world = next_world
        
        self.agents = [
            name for name in self.possible_agents
            if self.world.agents[self._name_to_agent_id(name)].alive
        ]
        
        if self.world.terminated or self.world.truncated:
            self.metrics.finalize(self.world, trust_manager=self.trust_manager)
            if self.event_writer:
                self.event_writer.flush()
        
        return observations, rewards, terminations, truncations, infos
    
    def _extract_agent_events(
        self,
        events: list[Event],
        old_world: WorldState,
        next_world: WorldState,
    ) -> dict[str, dict]:
        """Extract per-agent event tags from step events."""
        agent_events = {name: {"events": []} for name in self.possible_agents}
        
        for event in events:
            if event.event_type == EventType.KILL:
                killer_id = event.data["killer_id"]
                victim_id = event.data["victim_id"]
                killer_name = self._agent_id_to_name(killer_id)
                victim_name = self._agent_id_to_name(victim_id)
                
                victim_role = next_world.agents[victim_id].role
                if victim_role == Role.IMPOSTOR:
                    raise RuntimeError(
                        f"Invariant violated: KILL event targets impostor {victim_id}. "
                        "Impostors can only be ejected, never killed. Check engine logic."
                    )
                
                agent_events[killer_name]["events"].append("kill")
                agent_events[killer_name]["did_kill"] = True
                agent_events[victim_name]["events"].append("killed")
                agent_events[victim_name]["was_killed"] = True
            
            elif event.event_type == EventType.TASK_STEP_COMPLETE:
                agent_id = event.data["agent_id"]
                agent_name = self._agent_id_to_name(agent_id)
                agent_events[agent_name]["events"].append("task_step")
                agent_events[agent_name]["did_task_step"] = True
                
                for name in self.possible_agents:
                    aid = self._name_to_agent_id(name)
                    if next_world.agents[aid].role == Role.SURVIVOR:
                        agent_events[name]["team_task_step"] = True
            
            elif event.event_type == EventType.REPORT:
                reporter_id = event.data["reporter_id"]
                reporter_name = self._agent_id_to_name(reporter_id)
                agent_events[reporter_name]["events"].append("report")
                agent_events[reporter_name]["did_report"] = True
            
            elif event.event_type == EventType.EJECTION:
                ejected_id = event.data["ejected_id"]
                ejected_role = Role(event.data["role"])
                ejected_name = self._agent_id_to_name(ejected_id)
                
                agent_events[ejected_name]["events"].append("ejected")
                agent_events[ejected_name]["was_ejected"] = True
                
                for name in self.possible_agents:
                    aid = self._name_to_agent_id(name)
                    agent_role = next_world.agents[aid].role
                    
                    if agent_role == Role.SURVIVOR:
                        if ejected_role == Role.IMPOSTOR:
                            agent_events[name]["ejection_benefited"] = True
                        else:
                            agent_events[name]["ejection_penalized"] = True
                    else:
                        if ejected_role == Role.SURVIVOR:
                            agent_events[name]["ejection_benefited"] = True
                        else:
                            agent_events[name]["ejection_penalized"] = True
                
                if old_world.meeting is not None:
                    for tick, sender_id, action_id in old_world.meeting.comm_actions:
                        accuse_target = CommVocab.get_accuse_target(action_id)
                        if accuse_target == ejected_id:
                            accuser_name = self._agent_id_to_name(sender_id)
                            
                            if ejected_role == Role.IMPOSTOR:
                                agent_events[accuser_name]["correct_accuse"] = True
                            else:
                                agent_events[accuser_name]["false_accuse"] = True
            
            elif event.event_type == EventType.WIN:
                winner = Role(event.data["winner"])
                for name in self.possible_agents:
                    aid = self._name_to_agent_id(name)
                    if next_world.agents[aid].role == winner:
                        agent_events[name]["events"].append("win")
                    else:
                        agent_events[name]["events"].append("loss")
        
        return agent_events
    
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
                    else:
                        if ejected_role == Role.SURVIVOR:
                            rewards[agent_name] += config.reward_impostor_ejected_survivor
                        else:
                            rewards[agent_name] += config.reward_impostor_ejected_impostor
        
        for agent_name in self.possible_agents:
            agent_id = self._name_to_agent_id(agent_name)
            if new_world.agents[agent_id].alive:
                rewards[agent_name] += config.reward_time_penalty
        
        for agent_name in self.possible_agents:
            agent_id = self._name_to_agent_id(agent_name)
            agent = new_world.agents[agent_id]
            
            if agent.role == Role.SURVIVOR and agent.alive:
                impostor_in_room = False
                for other_id, other in new_world.agents.items():
                    if (other.role == Role.IMPOSTOR and 
                        other.alive and 
                        other.room == agent.room):
                        impostor_in_room = True
                        break
                
                if impostor_in_room:
                    agent.proximity_to_impostor_ticks += 1
                    if agent.proximity_to_impostor_ticks >= config.proximity_penalty_ticks:
                        rewards[agent_name] += config.reward_proximity_penalty
                else:
                    agent.proximity_to_impostor_ticks = 0
        
        return rewards
    
    def _update_proximity_tracking(self, world: WorldState) -> None:
        """Track how long agents stay near other agents (for impostor suspicion)."""
        for agent_id, agent in world.agents.items():
            if not agent.alive:
                self.proximity_ticks[agent_id] = 0
                continue
            

            other_agents_in_room = 0
            for other_id, other in world.agents.items():
                if other_id != agent_id and other.alive and other.room == agent.room:
                    other_agents_in_room += 1
            
            if other_agents_in_room > 0:
                self.proximity_ticks[agent_id] = self.proximity_ticks.get(agent_id, 0) + 1
            else:
                self.proximity_ticks[agent_id] = 0
    
    def _decay_suspicion(self, world: WorldState) -> None:
        """
        Apply temporal decay to suspicion scores.
        
        Requirements:
        - -0.1 slowly over time if no kills occur (for impostors)
        """
        for agent_id, agent in world.agents.items():
            if not agent.alive:
                continue
            
            if agent.role == Role.IMPOSTOR:
                ticks_since_kill = self.ticks_since_kill.get(agent_id, 0)
                if ticks_since_kill > 0 and ticks_since_kill % 10 == 0:
                    self.suspicion_score[agent_id] = max(0.0, self.suspicion_score[agent_id] - 0.1)
                
                self.ticks_since_kill[agent_id] = ticks_since_kill + 1
            
            self.suspicion_score[agent_id] = max(0.0, min(1.0, self.suspicion_score[agent_id]))
    
    def _update_suspicion_from_events(
        self,
        events: list[Event],
        old_world: WorldState,
        new_world: WorldState,
    ) -> None:
        """
        Buffer suspicious events for delayed activation.
        
        This is internal environment state only - not exposed to agents.
        
        Events are stored in pending_suspicions and become active after
        suspicion_delay_ticks have passed.
        
        Requirements:
        - Impostor kills: +0.3 if at least one witness is nearby
        - Impostor near body: +0.2
        - Impostor stays frequently near other agents: +0.1 (when proximity_ticks >= 5)
        - Survivor near body: +0.05
        - Survivor task progress: -0.05
        """
        current_tick = new_world.tick
        
        for event in events:
            if event.event_type == EventType.KILL:
                killer_id = event.data["killer_id"]
                victim_id = event.data["victim_id"]
                kill_room = event.data["room"]
                
                killer = old_world.agents.get(killer_id)
                if killer and killer.role == Role.IMPOSTOR:
                    witnesses = []
                    for agent_id, agent in old_world.agents.items():
                        if agent_id == killer_id or agent_id == victim_id:
                            continue
                        if not agent.alive:
                            continue
                        if agent.room == kill_room:
                            dist = self.engine.get_graph_distance(old_world, agent.room, kill_room)
                            if dist <= self.config.vision_radius:
                                witnesses.append(agent_id)
                    
                    if len(witnesses) > 0:
                        suspect_event = SuspectEvent(
                            target_agent=killer_id,
                            event_type="witness_kill",
                            tick_created=current_tick,
                            suspicion_value=0.3
                        )
                        self.pending_suspicions[killer_id].append(suspect_event)
                    

                    self.ticks_since_kill[killer_id] = 0
            
            elif event.event_type == EventType.REPORT:
                reporter_id = event.data["reporter_id"]
                body_room = event.data["room"]
                

                for body in old_world.bodies:
                    if body.room == body_room:
                        body_age = old_world.tick - body.tick_killed
                        if body_age <= 10:
                            for agent_id, agent in old_world.agents.items():
                                if agent_id == reporter_id:
                                    continue
                                if not agent.alive:
                                    continue
                                if agent.room == body_room:
                                    if agent.role == Role.IMPOSTOR:
                                        suspect_event = SuspectEvent(
                                            target_agent=agent_id,
                                            event_type="near_body",
                                            tick_created=current_tick,
                                            suspicion_value=0.2
                                        )
                                        self.pending_suspicions[agent_id].append(suspect_event)
                                    else:
                                        suspect_event = SuspectEvent(
                                            target_agent=agent_id,
                                            event_type="near_body",
                                            tick_created=current_tick,
                                            suspicion_value=0.05
                                        )
                                        self.pending_suspicions[agent_id].append(suspect_event)
            
            elif event.event_type == EventType.TASK_STEP_COMPLETE:
                agent_id = event.data["agent_id"]
                agent = new_world.agents.get(agent_id)
                if agent and agent.role == Role.SURVIVOR:
                    self.suspicion_score[agent_id] = max(0.0, self.suspicion_score[agent_id] - 0.05)
        
        for agent_id, agent in new_world.agents.items():
            if not agent.alive:
                continue
            if agent.role == Role.IMPOSTOR:
                proximity_count = self.proximity_ticks.get(agent_id, 0)
                if proximity_count >= 5 and proximity_count % 5 == 0:
                    suspect_event = SuspectEvent(
                        target_agent=agent_id,
                        event_type="suspicious_proximity",
                        tick_created=current_tick,
                        suspicion_value=0.1
                    )
                    self.pending_suspicions[agent_id].append(suspect_event)
    
    def _activate_delayed_suspicions(self, current_tick: int) -> None:
        """
        Activate pending suspicions that have passed the delay threshold.
        
        Moves events from pending_suspicions to active_suspicions if:
        current_tick - event.tick_created >= suspicion_delay_ticks
        
        Then updates suspicion scores based on newly activated events.
        """
        delay = self.config.suspicion_delay_ticks
        
        for agent_id in range(self.config.num_agents):
            pending = self.pending_suspicions.get(agent_id, [])
            if not pending:
                continue
            
            still_pending = []
            newly_active = []
            
            for event in pending:
                if current_tick - event.tick_created >= delay:
                    newly_active.append(event)
                else:
                    still_pending.append(event)
            
            self.pending_suspicions[agent_id] = still_pending
            
            for event in newly_active:
                self.suspicion_score[agent_id] += event.suspicion_value
            
            self.active_suspicions[agent_id].extend(newly_active)
        
        for agent_id in self.suspicion_score:
            self.suspicion_score[agent_id] = max(0.0, min(1.0, self.suspicion_score[agent_id]))
    
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
            elif event.event_type == EventType.COMM_ACTION:
                action_name = event.data.get("action_name",  "")
                if  "SUPPORT" in action_name:
                    self.metrics.comm_support_count += 1
                elif  "ACCUSE" in action_name:
                    self.metrics.comm_accuse_count += 1
                elif  "DEFEND" in action_name:
                    self.metrics.comm_defend_count += 1
                elif  "QUESTION" in action_name:
                    self.metrics.comm_question_count += 1
                elif  "NO_OP" in action_name:
                    self.metrics.comm_noop_count += 1
            elif event.event_type == EventType.VOTE_FAILED_LOW_CONFIDENCE:
                self.metrics.failed_votes_low_confidence += 1
                max_score = event.data.get("max_score", 0.0)
                self.metrics.max_voting_scores.append(max_score)
    
    def _process_comm_actions(self, world: WorldState, joint_actions: dict[int, int]) -> None:
        """
        Process communication actions and buffer them for delayed trust updates.
        
        This method is called during meeting phases when trust communication is enabled.
        """
        if self.trust_manager is None or world.meeting is None:
            return
        
        for agent_id, action in joint_actions.items():
            agent = world.agents[agent_id]
            if not agent.alive:
                continue
            
            action_type, param = self.engine.parse_action(action)
            if action_type ==  "comm_action":
                comm_action_id = param
                self.trust_manager.buffer_comm_action(
                    sender_id=agent_id,
                    action_id=comm_action_id,
                    current_tick=world.tick,
                    support_delta=self.config.trust_support_delta,
                    accuse_delta=self.config.trust_accuse_delta,
                    defend_delta=self.config.trust_defend_delta,
                    question_delta=self.config.trust_question_delta,
                )
    
    def _compute_voting_scores(self) -> dict[int, float]:
        """
        Compute voting scores by combining suspicion scores with trust.
        
        Voting score = suspicion * (1 - trust_weight) + distrust * trust_weight
        
        This allows for controlled ablation studies:
        - trust_weight = 0.0: pure suspicion-based voting (baseline)
        - trust_weight = 1.0: pure trust-based voting
        - trust_weight in (0, 1): hybrid approach
        
        During MEETING phase, adds noise to suspicion scores to make individual
        observations insufficient for reliable voting decisions.
        
        Returns:
            dict mapping agent_id -> voting score [0, 1]
        """
        assert 0.0 <= self.config.trust_voting_weight <= 1.0,            f"trust_voting_weight must be in [0, 1], got {self.config.trust_voting_weight}"
        
        voting_scores = {}
        
        in_meeting = self.world is not None and self.world.phase == Phase.MEETING
        meeting_noise = self.config.meeting_suspicion_noise if in_meeting else 0.0
        
        if self.trust_manager is None or self.config.trust_voting_weight == 0.0:
            for agent_id in range(self.config.num_agents):
                base_score = self.suspicion_score[agent_id]
                
                if meeting_noise > 0:
                    noise = (self.engine._rng.random() - 0.5) * meeting_noise
                    base_score = base_score + noise
                
                voting_scores[agent_id] = max(0.0, min(1.0, base_score))
        else:
            distrust_scores = self.trust_manager.get_distrust_scores()
            trust_weight = self.config.trust_voting_weight
            
            for agent_id in range(self.config.num_agents):
                suspicion = self.suspicion_score[agent_id]
                
                if meeting_noise > 0:
                    noise = (self.engine._rng.random() - 0.5) * meeting_noise
                    suspicion = max(0.0, min(1.0, suspicion + noise))
                
                distrust = distrust_scores.get(agent_id, 0.5)
                
                voting_scores[agent_id] = (
                    suspicion * (1.0 - trust_weight) + distrust * trust_weight
                )
                
                voting_scores[agent_id] = max(0.0, min(1.0, voting_scores[agent_id]))
                
                assert 0.0 <= voting_scores[agent_id] <= 1.0,                    f"Voting score out of bounds for agent {agent_id}: {voting_scores[agent_id]}"
        
        return voting_scores
    
    def render(self) -> Optional[str]:
        """Render the environment."""
        if self.world is None:
            return None
        
        if self.render_mode ==  "ansi" or self.render_mode ==  "human":
            return self._render_text()
        
        return None
    
    def _render_text(self) -> str:
        """Render environment as text."""
        lines = []
        world = self.world
        
        lines.append(f"=== Tick {world.tick} | Phase: {world.phase.name} ===")
        lines.append(f"EVAC Active: {world.evac_active} | EVAC Counter: {world.evac_tick_counter}")
        

        lines.append("\nMap:")
        for row in range(3):
            row_str =  ""
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
                        char =  "I" if a.role == Role.IMPOSTOR else  "S"
                        agent_chars.append(f"{char}{a.agent_id}")
                    cell +=  "(" +  ",".join(agent_chars) +  ")"
                
                if bodies_in_room:
                    cell +=  " X"
                
                row_str += f"{cell:20}"
            lines.append(row_str)
        

        closed_doors = [e for e, d in world.doors.items() if not d.is_open]
        if closed_doors:
            lines.append(f"\nClosed doors: {closed_doors}")
        

        lines.append("\nAgents:")
        for agent in world.agents.values():
            role =  "IMP" if agent.role == Role.IMPOSTOR else  "SRV"
            status =  "ALIVE" if agent.alive else  "DEAD"
            task_info =  ""
            if agent.role == Role.SURVIVOR:
                completed = sum(1 for t in agent.tasks if t.completed)
                task_info = f" | Tasks: {completed}/{len(agent.tasks)}"
            cooldown_info =  ""
            if agent.role == Role.IMPOSTOR:
                cooldown_info = f" | Kill CD: {agent.kill_cooldown}, Door CD: {agent.door_cooldown}"
            lines.append(f"  Agent {agent.agent_id} [{role}] {status} in Room {agent.room}{task_info}{cooldown_info}")
        

        if world.meeting:
            lines.append(f"\nMeeting: Reporter={world.meeting.reporter_id}, Body in Room {world.meeting.body_location}")
            lines.append(f"Timer: {world.meeting.phase_timer} | Messages: {len(world.meeting.messages)}")
            lines.append(f"Votes: {world.meeting.votes}")
        

        lines.append(f"\nTeam Task Progress: {world.team_task_progress()}/{world.total_task_steps()}")
        
        if world.terminated:
            winner =  "SURVIVORS" if world.winner == Role.SURVIVOR else  "IMPOSTORS"
            lines.append(f"\n*** GAME OVER: {winner} WIN! ***")
        
        result =  "\n".join(lines)
        
        if self.render_mode ==  "human":
            print(result)
        
        return result
    
    def _should_log_event(self, event: Event) -> bool:
        """
        Check if event should be logged based on log_event_types filter.
        
        Args:
            event: Event to check
            
        Returns:
            True if event should be logged, False otherwise
        """

        if self.log_event_types is None:
            return True
        

        return event.event_type.value in self.log_event_types
    
    def close(self) -> None:
        """Close the environment."""
        if self.event_writer:
            self.event_writer.close()

