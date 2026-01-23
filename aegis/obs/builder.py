from __future__ import annotations

from typing import Optional
import numpy as np
import random

from aegis.core.world import WorldState, Phase, Role
from aegis.core.rules import GameConfig
from aegis.core.engine import StepEngine
from aegis.comms.actions import CommVocab


class ObservationBuilder:
    """Builds observations and action masks for agents."""
    
    def __init__(self, engine: StepEngine, config: GameConfig, trust_manager=None):
        self.engine = engine
        self.config = config
        self.trust_manager = trust_manager
        self.num_rooms = len(engine.rooms)
        self.num_edges = len(engine.edges)
        self.num_agents = config.num_agents
        self.num_tokens = 20
        self.num_comm_actions = CommVocab.VOCAB_SIZE
        self.max_bodies = config.num_agents
        self.max_messages = 50
        self.max_tasks = config.tasks_per_survivor
        self._rng = random.Random(config.seed if config.seed is not None else 42)
    
    def build(self, world: WorldState, agent_id: int) -> dict:
        """Build observation dictionary for an agent."""
        agent = world.agents[agent_id]
        

        phase = int(world.phase)
        

        self_room = agent.room
        self_role = int(agent.role)
        self_alive = int(agent.alive)
        

        evac_active = int(world.evac_active)
        

        visible_agents = np.zeros((self.num_agents, 3), dtype=np.int32)
        if agent.alive:

            if world.phase == Phase.MEETING and self.config.meeting_observation_degradation > 0:

                degradation_factor = self.config.meeting_observation_degradation
                visible_set = set()
                for other_id, other in world.agents.items():
                    if other_id == agent_id:

                        visible_set.add(other_id)
                    elif other.alive and other.room == agent.room:

                        if self._rng.random() >= degradation_factor:
                            visible_set.add(other_id)
            else:

                visible_set = self.engine.get_visible_agents(world, agent_id)
            
            for other_id, other in world.agents.items():
                if other_id == agent_id:

                    visible_agents[other_id] = [1, agent.room, int(agent.alive)]
                elif other_id in visible_set:
                    visible_agents[other_id] = [1, other.room, int(other.alive)]
                else:
                    visible_agents[other_id] = [0, self.num_rooms, 0]
        

        visible_bodies = np.full((self.max_bodies, 3), -1, dtype=np.int32)
        visible_bodies[:, 0] = 0
        if agent.alive:
            if world.phase == Phase.MEETING and self.config.meeting_observation_degradation > 0:
                visible_body_list = [
                    b for b in world.bodies 
                    if b.room == agent.room and self._rng.random() >= self.config.meeting_observation_degradation
                ]
            else:
                visible_body_list = self.engine.get_visible_bodies(world, agent_id)
            
            for i, body in enumerate(visible_body_list[:self.max_bodies]):
                age = world.tick - body.tick_killed
                visible_bodies[i] = [1, body.room, age]
        

        door_states = np.zeros((self.num_edges, 2), dtype=np.int32)
        for i, edge in enumerate(self.engine.edges):
            door = world.doors[edge]
            door_states[i] = [int(door.is_open), door.closed_timer]
        

        task_state = np.full(5, -1, dtype=np.int32)
        if agent.role == Role.SURVIVOR:
            for task_idx, task in enumerate(agent.tasks):
                if not task.completed:
                    target_room = task.current_room()
                    if target_room is not None:
                        task_state = np.array([
                            task_idx,
                            task.current_step,
                            task.progress,
                            task.ticks_remaining(),
                            target_room
                        ], dtype=np.int32)
                    break
        

        total_steps = world.total_task_steps()
        if total_steps > 0:
            team_progress = world.team_task_progress() / total_steps
        else:
            team_progress = 1.0
        team_task_progress = np.array([team_progress], dtype=np.float32)
        

        kill_cd = agent.kill_cooldown / self.config.kill_cooldown if self.config.kill_cooldown > 0 else 0.0
        door_cd = agent.door_cooldown / self.config.door_cooldown if self.config.door_cooldown > 0 else 0.0
        kill_cooldown = np.array([kill_cd], dtype=np.float32)
        door_cooldown = np.array([door_cd], dtype=np.float32)
        

        meeting_messages = np.full((self.max_messages, 2), -1, dtype=np.int32)
        if world.meeting is not None:
            for i, (tick, sender_id, token_id) in enumerate(world.meeting.messages[-self.max_messages:]):
                meeting_messages[i] = [sender_id, token_id]
        

        if world.meeting is not None:
            meeting_reporter = world.meeting.reporter_id
            meeting_body_room = world.meeting.body_location
        else:
            meeting_reporter = self.num_agents
            meeting_body_room = self.num_rooms
        

        if agent.knows_role_of is not None:
            knows_role_of = agent.knows_role_of
            known_role = int(world.agents[agent.knows_role_of].role)
        else:
            knows_role_of = self.num_agents
            known_role = 2
        

        tick = np.array([world.tick], dtype=np.int32)
        

        if self.trust_manager is not None:
            trust_vector = self.trust_manager.get_trust_vector(agent_id)
        else:
            trust_vector = np.full(self.num_agents, 0.5, dtype=np.float32)
        

        action_mask = self._build_action_mask(world, agent_id)
        
        return {
            "phase": phase,
            "self_room": self_room,
            "self_role": self_role,
            "self_alive": self_alive,
            "evac_active": evac_active,
            "visible_agents": visible_agents,
            "visible_bodies": visible_bodies,
            "door_states": door_states,
            "task_state": task_state,
            "team_task_progress": team_task_progress,
            "kill_cooldown": kill_cooldown,
            "door_cooldown": door_cooldown,
            "meeting_messages": meeting_messages,
            "meeting_reporter": meeting_reporter,
            "meeting_body_room": meeting_body_room,
            "knows_role_of": knows_role_of,
            "known_role": known_role,
            "tick": tick,
            "trust_vector": trust_vector,
            "action_mask": action_mask,
        }
    
    def _build_action_mask(self, world: WorldState, agent_id: int) -> np.ndarray:
        """Build action mask for an agent."""
        agent = world.agents[agent_id]
        engine = self.engine
        

        mask = np.zeros(engine.action_space_size, dtype=np.int8)
        
        if not agent.alive:

            mask[0] = 1
            return mask
        
        if world.phase == Phase.FREE_PLAY:

            for room in range(self.num_rooms):
                mask[engine.offset_move + room] = 1
            

            if agent.role == Role.SURVIVOR:
                for task in agent.tasks:
                    if not task.completed:
                        target_room = task.current_room()
                        if target_room is not None and agent.room == target_room:
                            mask[engine.offset_work] = 1
                        break
            

            for body in world.bodies:
                if body.room == agent.room:
                    mask[engine.offset_report] = 1
                    break
            

            if agent.role == Role.IMPOSTOR and agent.kill_cooldown == 0:
                for other_id, other in world.agents.items():
                    if other_id != agent_id and other.alive and other.room == agent.room:
                        mask[engine.offset_kill + other_id] = 1
            

            if agent.role == Role.IMPOSTOR and agent.door_cooldown == 0:
                for i, edge in enumerate(engine.edges):
                    door = world.doors[edge]
                    if door.is_open:

                        if self.config.protect_evac_door and world.evac_room in edge:
                            continue
                        mask[engine.offset_close + i] = 1
        
        elif world.phase == Phase.MEETING:

            for room in range(self.num_rooms):
                mask[engine.offset_move + room] = 1
            

            for token_id in range(self.num_tokens):
                mask[engine.offset_token + token_id] = 1
            

            if self.config.enable_trust_comm:
                for comm_id in range(self.num_comm_actions):
                    mask[engine.offset_comm + comm_id] = 1
        
        elif world.phase == Phase.VOTING:

            for room in range(self.num_rooms):
                mask[engine.offset_move + room] = 1
            

            if not agent.has_voted:
                for other_id, other in world.agents.items():
                    if other.alive:
                        mask[engine.offset_vote + other_id] = 1
                mask[engine.offset_vote_skip] = 1
            

            for token_id in range(self.num_tokens):
                mask[engine.offset_token + token_id] = 1
            

            if self.config.enable_trust_comm:
                for comm_id in range(self.num_comm_actions):
                    mask[engine.offset_comm + comm_id] = 1
        

        if mask.sum() == 0:
            mask[0] = 1
        
        return mask
    
    def build_all(self, world: WorldState) -> dict[int, dict]:
        """Build observations for all agents."""
        return {agent_id: self.build(world, agent_id) for agent_id in world.agents}

