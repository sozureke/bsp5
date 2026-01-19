"""Small discrete token vocabulary for agent communication."""

from __future__ import annotations

from enum import IntEnum
from dataclasses import dataclass
from typing import Optional


class Token(IntEnum):
    """Token vocabulary for agent communication."""
    # Meta tokens
    SKIP = 0
    CONFIRM = 1
    DENY = 2
    
    # Accusation tokens (parameterized by agent index in higher bits)
    SUSPECT_0 = 3
    SUSPECT_1 = 4
    SUSPECT_2 = 5
    SUSPECT_3 = 6
    SUSPECT_4 = 7
    
    # Clear tokens
    CLEAR_0 = 8
    CLEAR_1 = 9
    CLEAR_2 = 10
    CLEAR_3 = 11
    CLEAR_4 = 12
    
    # Location tokens
    BODY_IN_ROOM_0 = 13
    BODY_IN_ROOM_1 = 14
    BODY_IN_ROOM_2 = 15
    
    # Personal location
    I_WAS_IN_ROOM_0 = 16
    I_WAS_IN_ROOM_1 = 17
    I_WAS_IN_ROOM_2 = 18
    
    # Door status
    DOOR_CLOSED = 19


class TokenVocab:
    """Token vocabulary helper class."""
    
    VOCAB_SIZE = 20
    
    # Token ranges for interpretation
    SUSPECT_BASE = 3
    SUSPECT_END = 8
    CLEAR_BASE = 8
    CLEAR_END = 13
    BODY_BASE = 13
    BODY_END = 16
    I_WAS_BASE = 16
    I_WAS_END = 19
    
    @classmethod
    def is_suspect(cls, token_id: int) -> bool:
        """Check if token is a SUSPECT token."""
        return cls.SUSPECT_BASE <= token_id < cls.SUSPECT_END
    
    @classmethod
    def is_clear(cls, token_id: int) -> bool:
        """Check if token is a CLEAR token."""
        return cls.CLEAR_BASE <= token_id < cls.CLEAR_END
    
    @classmethod
    def is_body_report(cls, token_id: int) -> bool:
        """Check if token is a body report."""
        return cls.BODY_BASE <= token_id < cls.BODY_END
    
    @classmethod
    def is_location_report(cls, token_id: int) -> bool:
        """Check if token is a personal location report."""
        return cls.I_WAS_BASE <= token_id < cls.I_WAS_END
    
    @classmethod
    def get_suspect_target(cls, token_id: int) -> Optional[int]:
        """Get the agent ID being suspected, or None."""
        if cls.is_suspect(token_id):
            return token_id - cls.SUSPECT_BASE
        return None
    
    @classmethod
    def get_clear_target(cls, token_id: int) -> Optional[int]:
        """Get the agent ID being cleared, or None."""
        if cls.is_clear(token_id):
            return token_id - cls.CLEAR_BASE
        return None
    
    @classmethod
    def get_body_room(cls, token_id: int) -> Optional[int]:
        """Get the room ID for body report, or None."""
        if cls.is_body_report(token_id):
            return token_id - cls.BODY_BASE
        return None
    
    @classmethod
    def get_location_room(cls, token_id: int) -> Optional[int]:
        """Get the room ID for location report, or None."""
        if cls.is_location_report(token_id):
            return token_id - cls.I_WAS_BASE
        return None
    
    @classmethod
    def make_suspect(cls, agent_id: int) -> int:
        """Create SUSPECT token for given agent."""
        return cls.SUSPECT_BASE + agent_id
    
    @classmethod
    def make_clear(cls, agent_id: int) -> int:
        """Create CLEAR token for given agent."""
        return cls.CLEAR_BASE + agent_id
    
    @classmethod
    def make_body_in_room(cls, room_id: int) -> int:
        """Create BODY_IN_ROOM token."""
        return cls.BODY_BASE + min(room_id, 2)  # Clamp to available tokens
    
    @classmethod
    def make_i_was_in_room(cls, room_id: int) -> int:
        """Create I_WAS_IN_ROOM token."""
        return cls.I_WAS_BASE + min(room_id, 2)  # Clamp to available tokens
    
    @classmethod
    def token_to_string(cls, token_id: int) -> str:
        """Convert token ID to human-readable string."""
        if token_id == Token.SKIP:
            return "SKIP"
        elif token_id == Token.CONFIRM:
            return "CONFIRM"
        elif token_id == Token.DENY:
            return "DENY"
        elif cls.is_suspect(token_id):
            return f"SUSPECT_{token_id - cls.SUSPECT_BASE}"
        elif cls.is_clear(token_id):
            return f"CLEAR_{token_id - cls.CLEAR_BASE}"
        elif cls.is_body_report(token_id):
            return f"BODY_IN_ROOM_{token_id - cls.BODY_BASE}"
        elif cls.is_location_report(token_id):
            return f"I_WAS_IN_ROOM_{token_id - cls.I_WAS_BASE}"
        elif token_id == Token.DOOR_CLOSED:
            return "DOOR_CLOSED"
        else:
            return f"TOKEN_{token_id}"

