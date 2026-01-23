"""Discrete communication actions for trust-based interaction."""

from __future__ import annotations

from enum import IntEnum
from typing import Optional


class CommAction(IntEnum):
    """Discrete communication actions for agent interaction."""
    NO_OP = 0
    SUPPORT_0 = 1
    SUPPORT_1 = 2
    SUPPORT_2 = 3
    SUPPORT_3 = 4
    SUPPORT_4 = 5
    ACCUSE_0 = 6
    ACCUSE_1 = 7
    ACCUSE_2 = 8
    ACCUSE_3 = 9
    ACCUSE_4 = 10
    DEFEND_SELF = 11
    QUESTION_0 = 12
    QUESTION_1 = 13
    QUESTION_2 = 14
    QUESTION_3 = 15
    QUESTION_4 = 16


class CommVocab:
    """Communication vocabulary helper class."""
    
    VOCAB_SIZE = 17
    

    NO_OP = 0
    SUPPORT_BASE = 1
    SUPPORT_END = 6
    ACCUSE_BASE = 6
    ACCUSE_END = 11
    DEFEND_SELF = 11
    QUESTION_BASE = 12
    QUESTION_END = 17
    
    @classmethod
    def is_support(cls, action_id: int) -> bool:
        """Check if action is SUPPORT."""
        return cls.SUPPORT_BASE <= action_id < cls.SUPPORT_END
    
    @classmethod
    def is_accuse(cls, action_id: int) -> bool:
        """Check if action is ACCUSE."""
        return cls.ACCUSE_BASE <= action_id < cls.ACCUSE_END
    
    @classmethod
    def is_question(cls, action_id: int) -> bool:
        """Check if action is QUESTION."""
        return cls.QUESTION_BASE <= action_id < cls.QUESTION_END
    
    @classmethod
    def is_defend(cls, action_id: int) -> bool:
        """Check if action is DEFEND_SELF."""
        return action_id == cls.DEFEND_SELF
    
    @classmethod
    def get_support_target(cls, action_id: int) -> Optional[int]:
        """Get the target agent ID for SUPPORT, or None."""
        if cls.is_support(action_id):
            return action_id - cls.SUPPORT_BASE
        return None
    
    @classmethod
    def get_accuse_target(cls, action_id: int) -> Optional[int]:
        """Get the target agent ID for ACCUSE, or None."""
        if cls.is_accuse(action_id):
            return action_id - cls.ACCUSE_BASE
        return None
    
    @classmethod
    def get_question_target(cls, action_id: int) -> Optional[int]:
        """Get the target agent ID for QUESTION, or None."""
        if cls.is_question(action_id):
            return action_id - cls.QUESTION_BASE
        return None
    
    @classmethod
    def make_support(cls, agent_id: int) -> int:
        """Create SUPPORT action for given agent."""
        return cls.SUPPORT_BASE + agent_id
    
    @classmethod
    def make_accuse(cls, agent_id: int) -> int:
        """Create ACCUSE action for given agent."""
        return cls.ACCUSE_BASE + agent_id
    
    @classmethod
    def make_question(cls, agent_id: int) -> int:
        """Create QUESTION action for given agent."""
        return cls.QUESTION_BASE + agent_id
    
    @classmethod
    def action_to_string(cls, action_id: int) -> str:
        """Convert action ID to human-readable string."""
        if action_id == cls.NO_OP:
            return "NO_OP"
        elif cls.is_support(action_id):
            target = cls.get_support_target(action_id)
            return f"SUPPORT({target})"
        elif cls.is_accuse(action_id):
            target = cls.get_accuse_target(action_id)
            return f"ACCUSE({target})"
        elif cls.is_defend(action_id):
            return "DEFEND_SELF"
        elif cls.is_question(action_id):
            target = cls.get_question_target(action_id)
            return f"QUESTION({target})"
        else:
            return f"UNKNOWN_ACTION_{action_id}"

