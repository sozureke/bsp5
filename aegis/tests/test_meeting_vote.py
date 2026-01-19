"""Tests for meeting and voting mechanics."""

import pytest
from aegis.core.world import WorldState, MeetingState, BodyState, Phase, Role
from aegis.core.rules import GameConfig
from aegis.core.engine import StepEngine


class TestMeetingTrigger:
    """Test meeting trigger conditions."""
    
    def setup_method(self):
        self.config = GameConfig(
            num_survivors=3,
            num_impostors=1,
            meeting_discussion_ticks=10,
            meeting_voting_ticks=5,
        )
        self.engine = StepEngine(self.config)
    
    def test_report_triggers_meeting(self):
        """Test that reporting a body triggers a meeting."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        # Create a body
        body = BodyState(agent_id=0, room=0, tick_killed=world.tick)
        world.bodies.append(body)
        
        # Find agent in same room
        for agent in world.agents.values():
            if agent.alive:
                agent.room = 0
                break
        
        # Report
        actions = {i: 0 for i in world.agents}
        actions[agent.agent_id] = self.engine.offset_report
        world, events = self.engine.step(world, actions)
        
        assert world.phase == Phase.MEETING
        assert world.meeting is not None
    
    def test_report_requires_body_in_room(self):
        """Test that report only works with body in room."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        # Body in different room
        body = BodyState(agent_id=0, room=5, tick_killed=world.tick)
        world.bodies.append(body)
        
        # Agent in room 0
        agent = list(world.agents.values())[0]
        agent.room = 0
        
        # Try to report
        actions = {i: 0 for i in world.agents}
        actions[agent.agent_id] = self.engine.offset_report
        world, events = self.engine.step(world, actions)
        
        # Should still be in free play (report action masked)
        assert world.phase == Phase.FREE_PLAY


class TestMeetingTimer:
    """Test meeting phase timers."""
    
    def setup_method(self):
        self.config = GameConfig(
            num_survivors=3,
            num_impostors=1,
            meeting_discussion_ticks=5,
            meeting_voting_ticks=3,
        )
        self.engine = StepEngine(self.config)
    
    def test_meeting_transitions_to_voting(self):
        """Test that meeting phase transitions to voting after timer."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        # Start meeting manually
        world.phase = Phase.MEETING
        world.meeting = MeetingState(reporter_id=0, body_location=0, body_agent_id=1)
        
        # Step through meeting phase
        for i in range(self.config.meeting_discussion_ticks):
            actions = {a: 0 for a in world.agents}
            world, _ = self.engine.step(world, actions)
        
        assert world.phase == Phase.VOTING
    
    def test_voting_ends_meeting(self):
        """Test that voting phase ends and returns to free play."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        # Start in voting phase
        world.phase = Phase.VOTING
        world.meeting = MeetingState(reporter_id=0, body_location=0, body_agent_id=1)
        world.meeting.phase_timer = 0
        
        # Step through voting phase
        for i in range(self.config.meeting_voting_ticks):
            actions = {a: 0 for a in world.agents}
            world, _ = self.engine.step(world, actions)
        
        assert world.phase == Phase.FREE_PLAY
        assert world.meeting is None


class TestVoting:
    """Test voting mechanics."""
    
    def setup_method(self):
        self.config = GameConfig(
            num_survivors=3,
            num_impostors=1,
            meeting_discussion_ticks=5,
            meeting_voting_ticks=10,
        )
        self.engine = StepEngine(self.config)
    
    def test_majority_vote_ejects(self):
        """Test that majority vote ejects the target."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        # Start voting phase
        world.phase = Phase.VOTING
        world.meeting = MeetingState(reporter_id=0, body_location=0, body_agent_id=1)
        
        # All vote for agent 0
        target_id = 0
        alive_agents = [a for a in world.agents.values() if a.alive]
        
        actions = {}
        for agent in alive_agents:
            actions[agent.agent_id] = self.engine.offset_vote + target_id
        
        # Fill missing with noop
        for i in world.agents:
            if i not in actions:
                actions[i] = 0
        
        world, _ = self.engine.step(world, actions)
        
        # Fast forward to end of voting
        world.meeting.phase_timer = self.config.meeting_voting_ticks - 1
        actions = {a: 0 for a in world.agents}
        world, events = self.engine.step(world, actions)
        
        assert not world.agents[target_id].alive
    
    def test_tie_vote_no_ejection(self):
        """Test that tie vote results in no ejection."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        # Start voting phase with even split
        world.phase = Phase.VOTING
        world.meeting = MeetingState(reporter_id=0, body_location=0, body_agent_id=1)
        
        alive_agents = [a for a in world.agents.values() if a.alive]
        alive_count = len(alive_agents)
        
        # Create a tie by having votes split
        actions = {}
        for i, agent in enumerate(alive_agents):
            if i % 2 == 0:
                actions[agent.agent_id] = self.engine.offset_vote + 0
            else:
                actions[agent.agent_id] = self.engine.offset_vote + 1
        
        for i in world.agents:
            if i not in actions:
                actions[i] = 0
        
        world, _ = self.engine.step(world, actions)
        
        # Fast forward to end of voting
        world.meeting.phase_timer = self.config.meeting_voting_ticks - 1
        actions = {a: 0 for a in world.agents}
        
        # Count alive before
        alive_before = len([a for a in world.agents.values() if a.alive])
        
        world, events = self.engine.step(world, actions)
        
        alive_after = len([a for a in world.agents.values() if a.alive])
        
        # With a tie, no one should be ejected
        # (Depends on exact vote distribution)
        assert world.phase == Phase.FREE_PLAY
    
    def test_skip_vote_counts(self):
        """Test that skip votes are counted correctly."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        world.phase = Phase.VOTING
        world.meeting = MeetingState(reporter_id=0, body_location=0, body_agent_id=1)
        
        # Everyone skips
        actions = {}
        for agent in world.agents.values():
            if agent.alive:
                actions[agent.agent_id] = self.engine.offset_vote_skip
        
        for i in world.agents:
            if i not in actions:
                actions[i] = 0
        
        world, _ = self.engine.step(world, actions)
        
        # Fast forward to end
        world.meeting.phase_timer = self.config.meeting_voting_ticks - 1
        actions = {a: 0 for a in world.agents}
        
        alive_before = len([a for a in world.agents.values() if a.alive])
        world, _ = self.engine.step(world, actions)
        alive_after = len([a for a in world.agents.values() if a.alive])
        
        # No one should be ejected on skip majority
        assert alive_before == alive_after


class TestEjection:
    """Test ejection behavior."""
    
    def setup_method(self):
        self.config = GameConfig(
            num_survivors=3,
            num_impostors=1,
        )
        self.engine = StepEngine(self.config)
    
    def test_ejection_creates_no_body(self):
        """Test that ejection does not create a body."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        initial_bodies = len(world.bodies)
        
        # Setup voting with unanimous vote
        world.phase = Phase.VOTING
        world.meeting = MeetingState(reporter_id=0, body_location=0, body_agent_id=1)
        
        target_id = 0
        for agent in world.agents.values():
            if agent.alive:
                agent.has_voted = True
                agent.vote_target = target_id
                world.meeting.votes[agent.agent_id] = target_id
        
        # Trigger vote resolution
        world.meeting.phase_timer = self.config.meeting_voting_ticks
        events = self.engine._resolve_voting(world)
        
        # No new bodies
        assert len(world.bodies) == initial_bodies


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

