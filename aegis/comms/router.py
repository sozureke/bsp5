from __future__ import annotations

from typing import Optional
from dataclasses import dataclass, field

from aegis.core.world import WorldState


@dataclass
class Message:
    """A message sent by an agent."""
    tick: int
    sender_id: int
    token_id: int
    recipients: list[int] = field(default_factory=list)


class MessageRouter:
    """Routes messages between agents during meetings."""
    
    def __init__(self, mode: str =  "broadcast", bandwidth_cap: int = 10):
        """
        Initialize message router.
        
        Args:
            mode: "broadcast" (all agents receive) or "local" (same room only)
            bandwidth_cap: Max messages per agent per meeting
        """
        self.mode = mode
        self.bandwidth_cap = bandwidth_cap
        self._message_counts: dict[int, int] = {}
    
    def reset(self) -> None:
        """Reset for new meeting."""
        self._message_counts = {}
    
    def can_send(self, agent_id: int) -> bool:
        """Check if agent can send another message."""
        return self._message_counts.get(agent_id, 0) < self.bandwidth_cap
    
    def route_message(
        self,
        world: WorldState,
        sender_id: int,
        token_id: int,
    ) -> Optional[Message]:
        """
        Route a message from sender.
        
        Returns Message with recipients, or None if bandwidth exceeded.
        """
        if not self.can_send(sender_id):
            return None
        
        sender = world.agents[sender_id]
        if not sender.alive:
            return None
        
        if self.mode ==  "broadcast":
            recipients = [
                a.agent_id for a in world.alive_agents()
                if a.agent_id != sender_id
            ]
        else:
            recipients = [
                a.agent_id for a in world.alive_agents()
                if a.agent_id != sender_id and a.room == sender.room
            ]
        
        self._message_counts[sender_id] = self._message_counts.get(sender_id, 0) + 1
        
        return Message(
            tick=world.tick,
            sender_id=sender_id,
            token_id=token_id,
            recipients=recipients,
        )
    
    def get_messages_for_agent(
        self,
        agent_id: int,
        all_messages: list[Message],
    ) -> list[Message]:
        """Filter messages to those received by a specific agent."""
        return [
            msg for msg in all_messages
            if agent_id in msg.recipients or len(msg.recipients) == 0
        ]

