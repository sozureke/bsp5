"""Typed events for logging."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Any


class EventType(str, Enum):
    """Event type enum."""
    TICK_START = "tick_start"
    TICK_END = "tick_end"
    PHASE_CHANGE = "phase_change"
    MOVE = "move"
    KILL = "kill"
    REPORT = "report"
    MEETING_START = "meeting_start"
    MEETING_END = "meeting_end"
    VOTE_CAST = "vote_cast"
    EJECTION = "ejection"
    VOTE_FAILED_LOW_CONFIDENCE = "vote_failed_low_confidence"
    DOOR_CLOSE = "door_close"
    DOOR_OPEN = "door_open"
    TASK_PROGRESS = "task_progress"
    TASK_STEP_COMPLETE = "task_step_complete"
    TASK_COMPLETE = "task_complete"
    MESSAGE_SENT = "message_sent"
    COMM_ACTION = "comm_action"
    EVAC_ACTIVATED = "evac_activated"
    EVAC_PROGRESS = "evac_progress"
    WIN = "win"
    KNOWER_REVEAL = "knower_reveal"
    MEETING_SUMMARY = "meeting_summary"


@dataclass
class Event:
    """Base event class."""
    event_type: EventType
    tick: int
    data: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert event to dictionary."""
        return {
            "event_type": self.event_type.value,
            "tick": self.tick,
            **self.data
        }


def make_event(event_type: EventType, tick: int, **kwargs) -> Event:
    """Factory function to create events."""
    return Event(event_type=event_type, tick=tick, data=kwargs)



def kill_event(tick: int, killer_id: int, victim_id: int, room: int) -> Event:
    return make_event(EventType.KILL, tick, killer_id=killer_id, victim_id=victim_id, room=room)


def report_event(tick: int, reporter_id: int, body_id: int, room: int) -> Event:
    return make_event(EventType.REPORT, tick, reporter_id=reporter_id, body_id=body_id, room=room)


def meeting_start_event(tick: int, reporter_id: int, body_id: int, room: int) -> Event:
    return make_event(EventType.MEETING_START, tick, reporter_id=reporter_id, body_id=body_id, room=room)


def meeting_end_event(tick: int) -> Event:
    return make_event(EventType.MEETING_END, tick)


def vote_cast_event(tick: int, voter_id: int, target_id: Optional[int]) -> Event:
    return make_event(EventType.VOTE_CAST, tick, voter_id=voter_id, target_id=target_id)


def ejection_event(tick: int, ejected_id: int, role: int, votes: dict) -> Event:
    return make_event(EventType.EJECTION, tick, ejected_id=ejected_id, role=role, votes=votes)


def vote_failed_event(tick: int, max_score: float, threshold: float, num_candidates: int) -> Event:
    return make_event(EventType.VOTE_FAILED_LOW_CONFIDENCE, tick, 
                     max_score=max_score, threshold=threshold, num_candidates=num_candidates)


def door_close_event(tick: int, agent_id: int, edge: tuple[int, int]) -> Event:
    return make_event(EventType.DOOR_CLOSE, tick, agent_id=agent_id, edge=list(edge))


def door_open_event(tick: int, edge: tuple[int, int]) -> Event:
    return make_event(EventType.DOOR_OPEN, tick, edge=list(edge))


def task_progress_event(tick: int, agent_id: int, task_idx: int, step: int, progress: int) -> Event:
    return make_event(EventType.TASK_PROGRESS, tick, agent_id=agent_id, task_idx=task_idx, step=step, progress=progress)


def task_step_complete_event(tick: int, agent_id: int, task_idx: int, step: int) -> Event:
    return make_event(EventType.TASK_STEP_COMPLETE, tick, agent_id=agent_id, task_idx=task_idx, step=step)


def task_complete_event(tick: int, agent_id: int, task_idx: int) -> Event:
    return make_event(EventType.TASK_COMPLETE, tick, agent_id=agent_id, task_idx=task_idx)


def message_sent_event(tick: int, sender_id: int, token_id: int) -> Event:
    return make_event(EventType.MESSAGE_SENT, tick, sender_id=sender_id, token_id=token_id)


def comm_action_event(
    tick: int, 
    sender_id: int, 
    action_id: int, 
    action_name: str,
    sender_role: Optional[int] = None,
    target_id: Optional[int] = None,
    target_role: Optional[int] = None,
    meeting_tick_start: Optional[int] = None,
    meeting_reporter_id: Optional[int] = None,
    meeting_body_room: Optional[int] = None,
) -> Event:
    """
    Create a communication action event with full context for scientific analysis.
    
    Args:
        tick: Current game tick
        sender_id: Agent who sent the communication
        action_id: Communication action ID
        action_name: Human-readable action name (e.g., "ACCUSE(1)")
        sender_role: Role of sender (0=survivor, 1=impostor)
        target_id: Target agent ID (if applicable, e.g., for ACCUSE/SUPPORT/QUESTION)
        target_role: Role of target (0=survivor, 1=impostor, if applicable)
        meeting_tick_start: Tick when current meeting started
        meeting_reporter_id: Agent who reported the body (triggered meeting)
        meeting_body_room: Room where the body was found
    """
    data = {
        "sender_id": sender_id,
        "action_id": action_id,
        "action_name": action_name,
    }
    

    if sender_role is not None:
        data["sender_role"] = sender_role
    if target_id is not None:
        data["target_id"] = target_id
    if target_role is not None:
        data["target_role"] = target_role
    if meeting_tick_start is not None:
        data["meeting_tick_start"] = meeting_tick_start
    if meeting_reporter_id is not None:
        data["meeting_reporter_id"] = meeting_reporter_id
    if meeting_body_room is not None:
        data["meeting_body_room"] = meeting_body_room
    
    return make_event(EventType.COMM_ACTION, tick, **data)


def move_event(tick: int, agent_id: int, from_room: int, to_room: int) -> Event:
    return make_event(EventType.MOVE, tick, agent_id=agent_id, from_room=from_room, to_room=to_room)


def phase_change_event(tick: int, old_phase: int, new_phase: int) -> Event:
    return make_event(EventType.PHASE_CHANGE, tick, old_phase=old_phase, new_phase=new_phase)


def evac_activated_event(tick: int, reason: str) -> Event:
    return make_event(EventType.EVAC_ACTIVATED, tick, reason=reason)


def evac_progress_event(tick: int, count: int, agents_in_evac: list[int]) -> Event:
    return make_event(EventType.EVAC_PROGRESS, tick, count=count, agents_in_evac=agents_in_evac)


def win_event(tick: int, winner: int, reason: str) -> Event:
    return make_event(EventType.WIN, tick, winner=winner, reason=reason)


def knower_reveal_event(tick: int, knower_id: int, target_id: int, target_role: int) -> Event:
    return make_event(EventType.KNOWER_REVEAL, tick, knower_id=knower_id, target_id=target_id, target_role=target_role)


def meeting_summary_event(
    tick: int,
    meeting_tick_start: int,
    reporter_id: int,
    body_room: int,
    comm_actions: list[dict],
    votes: dict[int, Optional[int]],
    ejected_id: Optional[int] = None,
    ejected_role: Optional[int] = None,
) -> Event:
    """
    Create a meeting summary event with all communications and voting results.
    
    Useful for scientific analysis: shows complete meeting context, all communications,
    and final outcome in one event.
    
    Args:
        tick: Current tick (end of meeting)
        meeting_tick_start: Tick when meeting started
        reporter_id: Agent who reported the body
        body_room: Room where body was found
        comm_actions: List of communication actions during meeting (with full context)
        votes: Final votes cast (voter_id -> target_id or None)
        ejected_id: Agent who was ejected (if any)
        ejected_role: Role of ejected agent (0=survivor, 1=impostor, if applicable)
    """
    return make_event(
        EventType.MEETING_SUMMARY,
        tick,
        meeting_tick_start=meeting_tick_start,
        reporter_id=reporter_id,
        body_room=body_room,
        comm_actions=comm_actions,
        votes=votes,
        ejected_id=ejected_id,
        ejected_role=ejected_role,
    )

