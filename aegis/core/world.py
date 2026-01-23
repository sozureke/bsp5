from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, TYPE_CHECKING
import copy

if TYPE_CHECKING:
    from aegis.core.rules import GameConfig


class Phase(IntEnum):
    """Game phase enum."""
    FREE_PLAY = 0
    MEETING = 1
    VOTING = 2


class Role(IntEnum):
    """Agent role enum."""
    SURVIVOR = 0
    IMPOSTOR = 1


class TaskType(IntEnum):
    """Task type enum."""
    SINGLE_POINT = 0  # Stand in target room for T_task ticks
    TWO_STEP = 1      # Visit room A then room B sequentially


@dataclass
class TaskState:
    """State of a single task assigned to an agent."""
    task_type: TaskType
    rooms: list[int]           # [room] for single_point, [room_a, room_b] for two_step
    ticks_required: list[int]  # ticks needed per step
    current_step: int = 0      # 0 or 1 for two_step
    progress: int = 0          # ticks spent on current step
    completed: bool = False
    
    def total_steps(self) -> int:
        """Return total number of steps in this task."""
        return len(self.rooms)
    
    def current_room(self) -> Optional[int]:
        """Return current target room, or None if completed."""
        if self.completed or self.current_step >= len(self.rooms):
            return None
        return self.rooms[self.current_step]
    
    def ticks_remaining(self) -> int:
        """Return ticks remaining for current step."""
        if self.completed or self.current_step >= len(self.rooms):
            return 0
        return self.ticks_required[self.current_step] - self.progress


@dataclass
class DoorState:
    """State of a door (edge) in the map."""
    edge: tuple[int, int]  # (room_a, room_b) sorted
    closed_timer: int = 0  # ticks remaining until reopens; 0 = open
    
    @property
    def is_open(self) -> bool:
        return self.closed_timer == 0


@dataclass 
class BodyState:
    """State of a dead body."""
    agent_id: int
    room: int
    tick_killed: int


@dataclass
class AgentState:
    """State of a single agent."""
    agent_id: int
    role: Role
    alive: bool = True
    room: int = 0
    kill_cooldown: int = 0
    door_cooldown: int = 0
    tasks: list[TaskState] = field(default_factory=list)
    target_room: Optional[int] = None  # current movement target
    path: list[int] = field(default_factory=list)  # A* path to target
    
    # Meeting/voting state
    has_voted: bool = False
    vote_target: Optional[int] = None  # agent_id or -1 for skip
    
    # Knower modifier: knows another agent's role
    knows_role_of: Optional[int] = None
    
    # Memory events (for aggregated memory mode)
    memory_events: list[dict] = field(default_factory=list)
    
    # Proximity tracking for reward shaping
    proximity_to_impostor_ticks: int = 0  # consecutive ticks near impostor
    
    def completed_task_steps(self) -> int:
        """Return total completed task steps."""
        total = 0
        for task in self.tasks:
            if task.completed:
                total += task.total_steps()
            else:
                total += task.current_step
        return total
    
    def all_tasks_complete(self) -> bool:
        """Check if all tasks are complete."""
        return all(t.completed for t in self.tasks)


@dataclass
class MeetingState:
    """State during a meeting."""
    reporter_id: int
    body_location: int
    body_agent_id: int
    tick_start: int  # Tick when meeting started (for logging and analysis)
    phase_timer: int = 0  # ticks elapsed in current meeting phase
    messages: list[tuple[int, int, int]] = field(default_factory=list)  # (tick, sender_id, token_id)
    votes: dict[int, Optional[int]] = field(default_factory=dict)  # voter_id -> target_id or None (skip)
    comm_actions: list[tuple[int, int, int]] = field(default_factory=list)  # (tick, sender_id, action_id) for trust-based comm


@dataclass
class WorldState:
    """Complete world state for one tick."""
    tick: int
    phase: Phase
    agents: dict[int, AgentState]  # agent_id -> AgentState
    bodies: list[BodyState]
    doors: dict[tuple[int, int], DoorState]  # sorted edge -> DoorState
    meeting: Optional[MeetingState] = None
    
    # Map structure (set once at init, not mutated)
    rooms: list[int] = field(default_factory=list)  # list of room ids
    edges: list[tuple[int, int]] = field(default_factory=list)  # list of (room_a, room_b)
    evac_room: int = 0
    
    # Win tracking
    evac_active: bool = False
    evac_tick_counter: int = 0  # consecutive ticks with >=2 survivors in evac
    winner: Optional[Role] = None
    terminated: bool = False
    truncated: bool = False
    
    # Config reference (for convenience)
    config: Optional["GameConfig"] = None
    
    def alive_agents(self) -> list[AgentState]:
        """Return list of alive agents."""
        return [a for a in self.agents.values() if a.alive]
    
    def alive_survivors(self) -> list[AgentState]:
        """Return alive survivors."""
        return [a for a in self.agents.values() if a.alive and a.role == Role.SURVIVOR]
    
    def alive_impostors(self) -> list[AgentState]:
        """Return alive impostors."""
        return [a for a in self.agents.values() if a.alive and a.role == Role.IMPOSTOR]
    
    def team_task_progress(self) -> int:
        """Return total completed task steps across all survivors."""
        return sum(a.completed_task_steps() for a in self.agents.values() if a.role == Role.SURVIVOR)
    
    def total_task_steps(self) -> int:
        """Return total task steps needed for all survivors."""
        total = 0
        for a in self.agents.values():
            if a.role == Role.SURVIVOR:
                for t in a.tasks:
                    total += t.total_steps()
        return total
    
    def all_tasks_complete(self) -> bool:
        """Check if all survivor tasks are complete."""
        return self.team_task_progress() >= self.total_task_steps()
    
    def copy(self) -> "WorldState":
        """Return a deep copy of this state."""
        return copy.deepcopy(self)

