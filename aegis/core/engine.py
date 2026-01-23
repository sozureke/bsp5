"""Deterministic step engine for the game simulation."""

from __future__ import annotations

from typing import Optional
import random
import networkx as nx

from aegis.core.world import (
    WorldState,
    AgentState,
    TaskState,
    DoorState,
    BodyState,
    MeetingState,
    Phase,
    Role,
    TaskType,
)
from aegis.core.rules import GameConfig, create_default_map
from aegis.core.events import (
    Event,
    EventType,
    kill_event,
    report_event,
    meeting_start_event,
    meeting_end_event,
    meeting_summary_event,
    vote_cast_event,
    ejection_event,
    vote_failed_event,
    door_close_event,
    door_open_event,
    task_progress_event,
    task_step_complete_event,
    task_complete_event,
    message_sent_event,
    comm_action_event,
    move_event,
    phase_change_event,
    evac_activated_event,
    evac_progress_event,
    win_event,
    knower_reveal_event,
)


class ActionType:
    """Action type constants."""
    NOOP = 0
    MOVE_BASE = 1  # MOVE_BASE + room_id
    WORK = -1  # Set dynamically
    REPORT = -2
    KILL_BASE = -3  # KILL_BASE - target_id
    CLOSE_DOOR_BASE = -100  # CLOSE_DOOR_BASE - door_idx
    SEND_TOKEN_BASE = -200  # SEND_TOKEN_BASE - token_id
    VOTE_BASE = -300  # VOTE_BASE - target_id
    VOTE_SKIP = -400


class StepEngine:
    """Deterministic step engine for the game simulation."""
    
    def __init__(self, config: Optional[GameConfig] = None):
        self.config = config or GameConfig()
        self._rng = random.Random(self.config.seed)
        self._build_map()
        self._compute_action_offsets()
    
    def _build_map(self) -> None:
        """Build the map graph."""
        rooms, edges = create_default_map()
        self.rooms = rooms
        self.edges = [tuple(sorted(e)) for e in edges]
        
        self.graph = nx.Graph()
        self.graph.add_nodes_from(rooms)
        self.graph.add_edges_from(self.edges)
        
        # Create edge index mapping
        self.edge_to_idx = {e: i for i, e in enumerate(self.edges)}
        self.idx_to_edge = {i: e for e, i in self.edge_to_idx.items()}
    
    def _compute_action_offsets(self) -> None:
        """Compute action space offsets."""
        n_rooms = len(self.rooms)
        n_agents = self.config.num_agents
        n_doors = len(self.edges)
        n_tokens = 20  # Token vocab size
        n_comm_actions = 17  # Communication action vocab size (NO_OP + 5*SUPPORT + 5*ACCUSE + DEFEND + 5*QUESTION)
        
        self.offset_move = 0
        self.offset_work = n_rooms
        self.offset_report = n_rooms + 1
        self.offset_kill = n_rooms + 2
        self.offset_close = n_rooms + 2 + n_agents
        self.offset_token = n_rooms + 2 + n_agents + n_doors
        self.offset_comm = n_rooms + 2 + n_agents + n_doors + n_tokens
        self.offset_vote = n_rooms + 2 + n_agents + n_doors + n_tokens + n_comm_actions
        self.offset_vote_skip = n_rooms + 2 + n_agents + n_doors + n_tokens + n_comm_actions + n_agents
        self.action_space_size = self.offset_vote_skip + 1
    
    def parse_action(self, action: int) -> tuple[str, int]:
        """Parse action integer into (action_type, param)."""
        if action < self.offset_work:
            return ("move", action - self.offset_move)
        elif action == self.offset_work:
            return ("work", 0)
        elif action == self.offset_report:
            return ("report", 0)
        elif action < self.offset_close:
            return ("kill", action - self.offset_kill)
        elif action < self.offset_token:
            return ("close_door", action - self.offset_close)
        elif action < self.offset_comm:
            return ("send_token", action - self.offset_token)
        elif action < self.offset_vote:
            return ("comm_action", action - self.offset_comm)
        elif action < self.offset_vote_skip:
            return ("vote", action - self.offset_vote)
        elif action == self.offset_vote_skip:
            return ("vote_skip", 0)
        else:
            return ("noop", 0)
    
    def action_to_int(self, action_type: str, param: int = 0) -> int:
        """Convert action type and param to action integer."""
        if action_type == "move":
            return self.offset_move + param
        elif action_type == "work":
            return self.offset_work
        elif action_type == "report":
            return self.offset_report
        elif action_type == "kill":
            return self.offset_kill + param
        elif action_type == "close_door":
            return self.offset_close + param
        elif action_type == "send_token":
            return self.offset_token + param
        elif action_type == "comm_action":
            return self.offset_comm + param
        elif action_type == "vote":
            return self.offset_vote + param
        elif action_type == "vote_skip":
            return self.offset_vote_skip
        else:
            return 0
    
    def create_initial_state(self, seed: Optional[int] = None) -> tuple[WorldState, list[Event]]:
        """Create initial world state."""
        if seed is not None:
            self._rng = random.Random(seed)
        
        events = []
        
        # Assign roles
        agent_ids = list(range(self.config.num_agents))
        self._rng.shuffle(agent_ids)
        impostor_ids = set(agent_ids[:self.config.num_impostors])
        
        # Create agents
        agents = {}
        for i in range(self.config.num_agents):
            role = Role.IMPOSTOR if i in impostor_ids else Role.SURVIVOR
            room = self._rng.choice(self.rooms)
            
            agent = AgentState(
                agent_id=i,
                role=role,
                room=room,
                kill_cooldown=self.config.kill_cooldown if role == Role.IMPOSTOR else 0,
            )
            
            # Assign tasks to survivors
            if role == Role.SURVIVOR:
                agent.tasks = self._generate_tasks()
            
            agents[i] = agent
        
        # Knower modifier
        if self.config.enable_knower:
            knower_id = self._rng.choice(agent_ids)
            other_ids = [i for i in agent_ids if i != knower_id]
            target_id = self._rng.choice(other_ids)
            agents[knower_id].knows_role_of = target_id
            agents[knower_id].memory_events.append({
                "type": "knower_reveal",
                "target_id": target_id,
                "target_role": agents[target_id].role,
            })
            events.append(knower_reveal_event(0, knower_id, target_id, agents[target_id].role))
        
        # Create doors (all open initially)
        doors = {}
        for edge in self.edges:
            doors[edge] = DoorState(edge=edge)
        
        world = WorldState(
            tick=0,
            phase=Phase.FREE_PLAY,
            agents=agents,
            bodies=[],
            doors=doors,
            rooms=self.rooms,
            edges=self.edges,
            evac_room=self.config.evac_room,
            config=self.config,
        )
        
        return world, events
    
    def _generate_tasks(self) -> list[TaskState]:
        """Generate random tasks for a survivor."""
        tasks = []
        available_rooms = [r for r in self.rooms if r != self.config.evac_room]
        
        for _ in range(self.config.tasks_per_survivor):
            if self._rng.random() < self.config.two_step_task_probability:
                # Two-step task
                rooms = self._rng.sample(available_rooms, 2)
                ticks = [self.config.task_ticks_two_step, self.config.task_ticks_two_step]
                task = TaskState(
                    task_type=TaskType.TWO_STEP,
                    rooms=rooms,
                    ticks_required=ticks,
                )
            else:
                # Single-point task
                room = self._rng.choice(available_rooms)
                task = TaskState(
                    task_type=TaskType.SINGLE_POINT,
                    rooms=[room],
                    ticks_required=[self.config.task_ticks_single],
                )
            tasks.append(task)
        
        return tasks
    
    def compute_shortest_path(self, world: WorldState, from_room: int, to_room: int) -> list[int]:
        """Compute shortest path avoiding closed doors."""
        if from_room == to_room:
            return []
        
        # Build graph with open edges only
        open_graph = nx.Graph()
        open_graph.add_nodes_from(self.rooms)
        for edge, door in world.doors.items():
            if door.is_open:
                open_graph.add_edge(*edge)
        
        try:
            path = nx.shortest_path(open_graph, from_room, to_room)
            return path[1:]  # Exclude current room
        except nx.NetworkXNoPath:
            return []
    
    def get_graph_distance(self, world: WorldState, room_a: int, room_b: int) -> int:
        """Get graph distance between two rooms (ignoring closed doors for vision)."""
        if room_a == room_b:
            return 0
        try:
            return nx.shortest_path_length(self.graph, room_a, room_b)
        except nx.NetworkXNoPath:
            return float('inf')
    
    def get_visible_agents(self, world: WorldState, agent_id: int) -> set[int]:
        """Get agents visible to the given agent."""
        agent = world.agents[agent_id]
        if not agent.alive:
            return set()
        
        visible = set()
        for other_id, other in world.agents.items():
            if other_id == agent_id:
                continue
            dist = self.get_graph_distance(world, agent.room, other.room)
            if dist <= self.config.vision_radius:
                visible.add(other_id)
        
        return visible
    
    def get_visible_bodies(self, world: WorldState, agent_id: int) -> list[BodyState]:
        """Get bodies visible to the given agent."""
        agent = world.agents[agent_id]
        if not agent.alive:
            return []
        
        visible = []
        for body in world.bodies:
            dist = self.get_graph_distance(world, agent.room, body.room)
            if dist <= self.config.vision_radius:
                visible.append(body)
        
        return visible
    
    def step(
        self, 
        world: WorldState, 
        joint_actions: dict[int, int],
        suspicion_scores: Optional[dict[int, float]] = None,
    ) -> tuple[WorldState, list[Event]]:
        """
        Execute one simulation tick.
        
        Order:
        1) Decrement cooldowns and door timers
        2) Apply movement
        3) Apply kills
        4) Apply reports -> enter MEETING
        5) Apply WORK for tasks
        6) Handle phase timers for MEETING/VOTING
        7) On VOTING end: compute ejection
        8) Compute win/termination conditions
        
        Args:
            world: Current world state
            joint_actions: Actions for all agents
            suspicion_scores: Optional dict mapping agent_id -> suspicion score [0.0, 1.0]
                            Used for probabilistic ejection selection
        """
        next_world = world.copy()
        next_world.tick += 1
        events = []
        
        # 1) Decrement cooldowns and door timers
        events.extend(self._decrement_timers(next_world))
        
        if next_world.phase == Phase.FREE_PLAY:
            # 2) Apply movement
            events.extend(self._apply_movement(next_world, joint_actions))
            
            # 3) Apply kills
            events.extend(self._apply_kills(next_world, joint_actions))
            
            # 4) Apply reports
            report_triggered = self._apply_reports(next_world, joint_actions, events)
            
            if not report_triggered:
                # 5) Apply WORK for tasks
                events.extend(self._apply_work(next_world, joint_actions))
                
                # Apply door closes
                events.extend(self._apply_door_closes(next_world, joint_actions))
        
        elif next_world.phase == Phase.MEETING:
            # Handle meeting messages
            events.extend(self._apply_messages(next_world, joint_actions))
            
            # 6) Handle meeting timer
            if next_world.meeting:
                next_world.meeting.phase_timer += 1
                if next_world.meeting.phase_timer >= self.config.meeting_discussion_ticks:
                    # Transition to VOTING
                    events.append(phase_change_event(next_world.tick, Phase.MEETING, Phase.VOTING))
                    next_world.phase = Phase.VOTING
                    next_world.meeting.phase_timer = 0
                    # Reset vote state for all alive agents
                    for agent in next_world.alive_agents():
                        agent.has_voted = False
                        agent.vote_target = None
        
        elif next_world.phase == Phase.VOTING:
            # Handle votes
            events.extend(self._apply_votes(next_world, joint_actions))
            
            # 7) Handle voting timer
            if next_world.meeting:
                next_world.meeting.phase_timer += 1
                if next_world.meeting.phase_timer >= self.config.meeting_voting_ticks:
                    # End voting, compute ejection (with optional suspicion scores)
                    events.extend(self._resolve_voting(next_world, suspicion_scores=suspicion_scores))
        
        # Check evac activation
        events.extend(self._check_evac_activation(next_world))
        
        # 8) Compute win/termination conditions
        events.extend(self._check_win_conditions(next_world))
        
        return next_world, events
    
    def _decrement_timers(self, world: WorldState) -> list[Event]:
        """Decrement cooldowns and door timers."""
        events = []
        
        for agent in world.agents.values():
            if agent.kill_cooldown > 0:
                agent.kill_cooldown -= 1
            if agent.door_cooldown > 0:
                agent.door_cooldown -= 1
        
        for edge, door in world.doors.items():
            if door.closed_timer > 0:
                door.closed_timer -= 1
                if door.closed_timer == 0:
                    events.append(door_open_event(world.tick, edge))
        
        return events
    
    def _apply_movement(self, world: WorldState, joint_actions: dict[int, int]) -> list[Event]:
        """Apply movement actions."""
        events = []
        
        for agent_id, action in joint_actions.items():
            agent = world.agents[agent_id]
            if not agent.alive:
                continue
            
            action_type, param = self.parse_action(action)
            if action_type == "move":
                target_room = param
                if target_room == agent.room:
                    continue
                
                if target_room < 0 or target_room >= len(self.rooms):
                    continue
                
                # Compute path if target changed
                if agent.target_room != target_room:
                    agent.target_room = target_room
                    agent.path = self.compute_shortest_path(world, agent.room, target_room)
                
                # Move one step along path
                if agent.path:
                    next_room = agent.path[0]
                    # Check if edge is open
                    edge = tuple(sorted((agent.room, next_room)))
                    if world.doors[edge].is_open:
                        old_room = agent.room
                        agent.room = next_room
                        agent.path = agent.path[1:]
                        events.append(move_event(world.tick, agent_id, old_room, next_room))
        
        return events
    
    def _apply_kills(self, world: WorldState, joint_actions: dict[int, int]) -> list[Event]:
        """Apply kill actions."""
        events = []
        
        # Check if initial kill delay has passed
        if world.tick < self.config.initial_kill_delay:
            return events  # No kills allowed before initial delay
        
        for agent_id, action in joint_actions.items():
            agent = world.agents[agent_id]
            if not agent.alive or agent.role != Role.IMPOSTOR:
                continue
            if agent.kill_cooldown > 0:
                continue
            
            action_type, param = self.parse_action(action)
            if action_type == "kill":
                target_id = param
                if target_id == agent_id:
                    continue
                
                target = world.agents.get(target_id)
                if target is None or not target.alive:
                    continue
                
                # Impostors cannot be killed, only ejected
                if target.role == Role.IMPOSTOR:
                    continue
                
                # Must be in same room
                if target.room != agent.room:
                    continue
                
                # Execute kill
                target.alive = False
                agent.kill_cooldown = self.config.kill_cooldown
                
                # Create body
                body = BodyState(
                    agent_id=target_id,
                    room=target.room,
                    tick_killed=world.tick,
                )
                world.bodies.append(body)
                
                events.append(kill_event(world.tick, agent_id, target_id, target.room))
        
        return events
    
    def _apply_reports(self, world: WorldState, joint_actions: dict[int, int], events: list[Event]) -> bool:
        """Apply report actions. Returns True if a meeting was triggered."""
        for agent_id, action in joint_actions.items():
            agent = world.agents[agent_id]
            if not agent.alive:
                continue
            
            action_type, _ = self.parse_action(action)
            if action_type == "report":
                # Check if there's a body in the agent's room
                body_in_room = None
                for body in world.bodies:
                    if body.room == agent.room:
                        body_in_room = body
                        break
                
                if body_in_room is not None:
                    # Start meeting
                    events.append(report_event(world.tick, agent_id, body_in_room.agent_id, agent.room))
                    events.append(meeting_start_event(world.tick, agent_id, body_in_room.agent_id, agent.room))
                    events.append(phase_change_event(world.tick, Phase.FREE_PLAY, Phase.MEETING))
                    
                    world.phase = Phase.MEETING
                    world.meeting = MeetingState(
                        reporter_id=agent_id,
                        body_location=agent.room,
                        body_agent_id=body_in_room.agent_id,
                        tick_start=world.tick,  # Track when meeting started for logging
                    )
                    
                    # Remove the reported body
                    world.bodies.remove(body_in_room)
                    
                    return True
        
        return False
    
    def _apply_work(self, world: WorldState, joint_actions: dict[int, int]) -> list[Event]:
        """Apply work actions for tasks."""
        events = []
        
        for agent_id, action in joint_actions.items():
            agent = world.agents[agent_id]
            if not agent.alive or agent.role != Role.SURVIVOR:
                continue
            
            action_type, _ = self.parse_action(action)
            if action_type == "work":
                # Find current incomplete task
                for task_idx, task in enumerate(agent.tasks):
                    if task.completed:
                        continue
                    
                    current_room = task.current_room()
                    if current_room is not None and agent.room == current_room:
                        # Progress the task
                        task.progress += 1
                        events.append(task_progress_event(
                            world.tick, agent_id, task_idx, task.current_step, task.progress
                        ))
                        
                        # Check if step completed
                        if task.progress >= task.ticks_required[task.current_step]:
                            events.append(task_step_complete_event(
                                world.tick, agent_id, task_idx, task.current_step
                            ))
                            
                            # Move to next step or complete task
                            task.current_step += 1
                            task.progress = 0
                            
                            if task.current_step >= task.total_steps():
                                task.completed = True
                                events.append(task_complete_event(world.tick, agent_id, task_idx))
                    
                    break  # Only work on one task at a time
        
        return events
    
    def _apply_door_closes(self, world: WorldState, joint_actions: dict[int, int]) -> list[Event]:
        """Apply door close actions."""
        events = []
        
        for agent_id, action in joint_actions.items():
            agent = world.agents[agent_id]
            if not agent.alive or agent.role != Role.IMPOSTOR:
                continue
            if agent.door_cooldown > 0:
                continue
            
            action_type, param = self.parse_action(action)
            if action_type == "close_door":
                door_idx = param
                if door_idx < 0 or door_idx >= len(self.edges):
                    continue
                
                edge = self.edges[door_idx]
                door = world.doors[edge]
                
                # Check if protecting evac door
                if self.config.protect_evac_door:
                    if world.evac_room in edge:
                        continue
                
                if door.is_open:
                    door.closed_timer = self.config.door_close_duration
                    agent.door_cooldown = self.config.door_cooldown
                    events.append(door_close_event(world.tick, agent_id, edge))
        
        return events
    
    def _apply_messages(self, world: WorldState, joint_actions: dict[int, int]) -> list[Event]:
        """Apply message sending and communication actions during meeting."""
        events = []
        
        if world.meeting is None:
            return events
        
        for agent_id, action in joint_actions.items():
            agent = world.agents[agent_id]
            if not agent.alive:
                continue
            
            action_type, param = self.parse_action(action)
            if action_type == "send_token":
                token_id = param
                world.meeting.messages.append((world.tick, agent_id, token_id))
                events.append(message_sent_event(world.tick, agent_id, token_id))
            
            elif action_type == "comm_action":
                # Communication actions for trust system
                comm_action_id = param
                world.meeting.comm_actions.append((world.tick, agent_id, comm_action_id))
                # Log communication action event with full context for scientific analysis
                from aegis.comms.actions import CommVocab
                action_name = CommVocab.action_to_string(comm_action_id)
                
                # Extract target information if applicable
                target_id = None
                if CommVocab.is_support(comm_action_id):
                    target_id = CommVocab.get_support_target(comm_action_id)
                elif CommVocab.is_accuse(comm_action_id):
                    target_id = CommVocab.get_accuse_target(comm_action_id)
                elif CommVocab.is_question(comm_action_id):
                    target_id = CommVocab.get_question_target(comm_action_id)
                
                # Get roles for sender and target
                sender_role = int(agent.role)
                target_role = None
                if target_id is not None and target_id < len(world.agents):
                    target_agent = world.agents.get(target_id)
                    if target_agent is not None:
                        target_role = int(target_agent.role)
                
                # Find meeting start tick (first tick of current meeting phase)
                meeting_tick_start = None
                # We need to track when meeting started - check if we can infer from phase
                # For now, we'll use the reporter_id and body_location from meeting state
                
                events.append(comm_action_event(
                    tick=world.tick,
                    sender_id=agent_id,
                    action_id=comm_action_id,
                    action_name=action_name,
                    sender_role=sender_role,
                    target_id=target_id,
                    target_role=target_role,
                    meeting_tick_start=world.meeting.tick_start,
                    meeting_reporter_id=world.meeting.reporter_id,
                    meeting_body_room=world.meeting.body_location,
                ))
        
        return events
    
    def _apply_votes(self, world: WorldState, joint_actions: dict[int, int]) -> list[Event]:
        """Apply voting actions."""
        events = []
        
        if world.meeting is None:
            return events
        
        for agent_id, action in joint_actions.items():
            agent = world.agents[agent_id]
            if not agent.alive or agent.has_voted:
                continue
            
            action_type, param = self.parse_action(action)
            if action_type == "vote":
                target_id = param
                if target_id < 0 or target_id >= self.config.num_agents:
                    continue
                target = world.agents.get(target_id)
                if target is None or not target.alive:
                    continue
                
                agent.has_voted = True
                agent.vote_target = target_id
                world.meeting.votes[agent_id] = target_id
                events.append(vote_cast_event(world.tick, agent_id, target_id))
                
            elif action_type == "vote_skip":
                agent.has_voted = True
                agent.vote_target = None
                world.meeting.votes[agent_id] = None
                events.append(vote_cast_event(world.tick, agent_id, None))
        
        return events
    
    def _resolve_voting(
        self, 
        world: WorldState,
        suspicion_scores: Optional[dict[int, float]] = None,
    ) -> list[Event]:
        """
        Resolve voting and potentially eject someone.
        
        Voting mechanism: Probabilistic selection based on voting scores.
        P(eject agent i) ∝ voting_score[i] with noise that depends on coordination.
        
        Voting Confidence Gating:
        - Only agents with voting_score >= voting_confidence_threshold are eligible
        - This forces evidence accumulation via communication
        
        Coordination-Based Noise Reduction:
        - Without coordination (few ACCUSE actions on same target): high voting noise
        - With coordination (multiple ACCUSE actions on same target): reduced noise
        - This makes ACCUSE actions structurally necessary for reliable voting
        
        This is environment-internal causal memory - agents do not have direct
        access to voting scores. Voting is NOT trained by agents.
        
        Args:
            world: Current world state
            suspicion_scores: Optional dict mapping agent_id -> voting score [0.0, 1.0]
                            (renamed from suspicion_scores for clarity - these are composite scores)
                            If None, voting falls back to deterministic vote counting
        """
        events = []
        
        if world.meeting is None:
            return events
        
        # Count ACCUSE actions per target to detect coordination
        from aegis.comms.actions import CommVocab
        accuse_counts: dict[int, int] = {}  # target_id -> count of ACCUSE actions
        for tick, sender_id, action_id in world.meeting.comm_actions:
            if CommVocab.is_accuse(action_id):
                target_id = CommVocab.get_accuse_target(action_id)
                if target_id is not None:
                    accuse_counts[target_id] = accuse_counts.get(target_id, 0) + 1
        
        # Determine noise level based on coordination
        # Find the target with most accusations
        max_accusations = max(accuse_counts.values()) if accuse_counts else 0
        has_coordination = max_accusations >= self.config.voting_coordination_threshold
        
        # Set noise level: high without coordination, low with coordination
        if has_coordination:
            voting_noise = self.config.voting_noise_with_coordination
        else:
            voting_noise = self.config.voting_noise_base
        
        # Count base votes
        vote_counts: dict[Optional[int], int] = {}
        for voter_id, target_id in world.meeting.votes.items():
            vote_counts[target_id] = vote_counts.get(target_id, 0) + 1
        
        # Also count agents who didn't vote as skips
        for agent in world.alive_agents():
            if agent.agent_id not in world.meeting.votes:
                vote_counts[None] = vote_counts.get(None, 0) + 1
        
        ejected_id = None
        confidence_threshold = self.config.voting_confidence_threshold
        max_voting_score = 0.0  # Track for logging
        
        if suspicion_scores is not None:
            alive_agent_ids = [a.agent_id for a in world.alive_agents()]
            
            # VOTING CONFIDENCE GATING: Filter to eligible candidates
            eligible_candidates = []
            eligible_scores = []
            
            for agent_id in alive_agent_ids:
                score = suspicion_scores.get(agent_id, 0.0)
                max_voting_score = max(max_voting_score, score)
                
                # Only consider if score meets threshold
                if score >= confidence_threshold:
                    eligible_candidates.append(agent_id)
                    eligible_scores.append(score)
            
            if eligible_candidates:
                # Add noise based on coordination level
                # High noise without coordination makes voting unreliable
                # Low noise with coordination makes voting more reliable
                noisy_scores = []
                for score in eligible_scores:
                    # Add noise: uniform in [-voting_noise/2, +voting_noise/2]
                    noise = self._rng.uniform(-voting_noise / 2, voting_noise / 2)
                    noisy_scores.append(max(0.0, min(1.0, score + noise)))
                
                total_weight = sum(noisy_scores)
                if total_weight > 0:
                    probabilities = [w / total_weight for w in noisy_scores]
                    
                    # Sample from distribution
                    selected_idx = self._rng.choices(
                        range(len(eligible_candidates)),
                        weights=probabilities,
                        k=1
                    )[0]
                    ejected_id = eligible_candidates[selected_idx]
            else:
                # No eligible candidates (all below threshold) → no ejection
                # Log this event for analysis
                events.append(vote_failed_event(
                    world.tick, 
                    max_voting_score, 
                    confidence_threshold,
                    len(alive_agent_ids)
                ))
        else:
            if has_coordination:
                max_votes = max(vote_counts.values()) if vote_counts else 0
                max_targets = [t for t, v in vote_counts.items() if v == max_votes]
                if len(max_targets) == 1 and max_targets[0] is not None:
                    ejected_id = max_targets[0]
            else:
                noisy_vote_counts = {}
                for target, count in vote_counts.items():
                    max_vote_value = max(vote_counts.values()) if vote_counts else 1
                    noise_offset = int(self._rng.uniform(-voting_noise * max_vote_value, voting_noise * max_vote_value))
                    noisy_vote_counts[target] = max(0, count + noise_offset)
                
                max_votes = max(noisy_vote_counts.values()) if noisy_vote_counts else 0
                max_targets = [t for t, v in noisy_vote_counts.items() if v == max_votes]
                if len(max_targets) == 1 and max_targets[0] is not None:
                    ejected_id = max_targets[0]
        
        # Eject the selected agent (if eligible)
        ejected_role = None
        if ejected_id is not None:
            ejected = world.agents[ejected_id]
            ejected.alive = False
            ejected_role = int(ejected.role)
            # No body created for ejection
            events.append(ejection_event(
                world.tick, ejected_id, ejected.role,
                {k: v for k, v in world.meeting.votes.items()}
            ))
        
        # Create meeting summary event with all communications and voting results
        # This is useful for scientific analysis: complete meeting context in one event
        from aegis.comms.actions import CommVocab
        comm_actions_summary = []
        for tick, sender_id, action_id in world.meeting.comm_actions:
            action_name = CommVocab.action_to_string(action_id)
            sender_agent = world.agents.get(sender_id)
            sender_role = int(sender_agent.role) if sender_agent else None
            
            # Extract target info
            target_id = None
            target_role = None
            if CommVocab.is_support(action_id):
                target_id = CommVocab.get_support_target(action_id)
            elif CommVocab.is_accuse(action_id):
                target_id = CommVocab.get_accuse_target(action_id)
            elif CommVocab.is_question(action_id):
                target_id = CommVocab.get_question_target(action_id)
            
            if target_id is not None and target_id < len(world.agents):
                target_agent = world.agents.get(target_id)
                if target_agent:
                    target_role = int(target_agent.role)
            
            comm_actions_summary.append({
                "tick": tick,
                "sender_id": sender_id,
                "sender_role": sender_role,
                "action_id": action_id,
                "action_name": action_name,
                "target_id": target_id,
                "target_role": target_role,
            })
        
        events.append(meeting_summary_event(
            tick=world.tick,
            meeting_tick_start=world.meeting.tick_start,
            reporter_id=world.meeting.reporter_id,
            body_room=world.meeting.body_location,
            comm_actions=comm_actions_summary,
            votes={k: v for k, v in world.meeting.votes.items()},
            ejected_id=ejected_id,
            ejected_role=ejected_role,
        ))
        
        # End meeting
        events.append(meeting_end_event(world.tick))
        events.append(phase_change_event(world.tick, Phase.VOTING, Phase.FREE_PLAY))
        
        world.phase = Phase.FREE_PLAY
        world.meeting = None
        
        # Reset vote state
        for agent in world.agents.values():
            agent.has_voted = False
            agent.vote_target = None
        
        return events
    
    def _check_evac_activation(self, world: WorldState) -> list[Event]:
        """Check if EVAC should be activated."""
        events = []
        
        if world.evac_active:
            return events
        
        # Condition A: All tasks complete
        if world.all_tasks_complete():
            world.evac_active = True
            events.append(evac_activated_event(world.tick, "all_tasks_complete"))
            return events
        
        # Condition B: All impostors eliminated
        if len(world.alive_impostors()) == 0:
            world.evac_active = True
            events.append(evac_activated_event(world.tick, "all_impostors_eliminated"))
            return events
        
        return events
    
    def _check_win_conditions(self, world: WorldState) -> list[Event]:
        """Check win/termination conditions."""
        events = []
        
        if world.terminated:
            return events
        
        alive_survivors = world.alive_survivors()
        alive_impostors = world.alive_impostors()
        
        # Environment constraint: impostor cannot win too early (gives survivors time to coordinate)
        can_impostor_win = world.tick >= self.config.min_episode_ticks_before_impostor_win
        
        # Impostor win: alive impostors >= alive survivors
        if can_impostor_win and len(alive_impostors) >= len(alive_survivors) and len(alive_survivors) > 0:
            world.terminated = True
            world.winner = Role.IMPOSTOR
            events.append(win_event(world.tick, Role.IMPOSTOR, "impostor_parity"))
            return events
        
        # All survivors dead
        if can_impostor_win and len(alive_survivors) == 0:
            world.terminated = True
            world.winner = Role.IMPOSTOR
            events.append(win_event(world.tick, Role.IMPOSTOR, "all_survivors_dead"))
            return events
        
        # All impostors dead (survivors win)
        if len(alive_impostors) == 0:
            world.terminated = True
            world.winner = Role.SURVIVOR
            events.append(win_event(world.tick, Role.SURVIVOR, "all_impostors_eliminated"))
            return events
        
        # Survivor win via EVAC
        if world.evac_active and world.phase == Phase.FREE_PLAY:
            survivors_in_evac = [a for a in alive_survivors if a.room == world.evac_room]
            
            if len(survivors_in_evac) >= 2:
                world.evac_tick_counter += 1
                events.append(evac_progress_event(
                    world.tick, world.evac_tick_counter,
                    [a.agent_id for a in survivors_in_evac]
                ))
                
                if world.evac_tick_counter >= self.config.evac_ticks_required:
                    world.terminated = True
                    world.winner = Role.SURVIVOR
                    events.append(win_event(world.tick, Role.SURVIVOR, "evac_success"))
            else:
                world.evac_tick_counter = 0
        
        # Timeout truncation
        if world.tick >= self.config.max_ticks:
            world.truncated = True
            # Impostors win on timeout if not already terminated
            if not world.terminated:
                world.terminated = True
                world.winner = Role.IMPOSTOR
                events.append(win_event(world.tick, Role.IMPOSTOR, "timeout"))
        
        return events

