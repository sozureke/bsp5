"""Tests for vision radius and action masking."""

import pytest
import numpy as np
from aegis.core.world import WorldState, Phase, Role
from aegis.core.rules import GameConfig
from aegis.core.engine import StepEngine
from aegis.obs.builder import ObservationBuilder


class TestVisionRadius:
    """Test vision radius mechanics."""
    
    def setup_method(self):
        self.config = GameConfig(
            num_survivors=3,
            num_impostors=1,
            vision_radius=1,  # Can only see adjacent rooms
        )
        self.engine = StepEngine(self.config)
        self.obs_builder = ObservationBuilder(self.engine, self.config)
    
    def test_agents_in_same_room_visible(self):
        """Test that agents in the same room are visible."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        # Put all agents in room 0
        for agent in world.agents.values():
            agent.room = 0
        
        # Check visibility for first agent
        agent_id = 0
        visible = self.engine.get_visible_agents(world, agent_id)
        
        # All other agents should be visible
        expected = set(world.agents.keys()) - {agent_id}
        assert visible == expected
    
    def test_agents_beyond_radius_not_visible(self):
        """Test that agents beyond vision radius are not visible."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        # Put observer in room 0
        observer_id = 0
        world.agents[observer_id].room = 0
        
        # Put another agent far away (room 8 in 3x3 grid)
        # Distance from room 0 to room 8 is 4 (0->1->2->5->8 or similar)
        other_id = 1
        world.agents[other_id].room = 8
        
        visible = self.engine.get_visible_agents(world, observer_id)
        
        # With vision_radius=1, room 8 should not be visible from room 0
        dist = self.engine.get_graph_distance(world, 0, 8)
        if dist > self.config.vision_radius:
            assert other_id not in visible
    
    def test_bodies_within_radius_visible(self):
        """Test that bodies within vision radius are visible."""
        world, _ = self.engine.create_initial_state(seed=42)
        from aegis.core.world import BodyState
        
        observer_id = 0
        world.agents[observer_id].room = 0
        
        # Add body in adjacent room (room 1)
        body = BodyState(agent_id=99, room=1, tick_killed=0)
        world.bodies.append(body)
        
        visible_bodies = self.engine.get_visible_bodies(world, observer_id)
        
        # Room 1 is adjacent to room 0 (distance 1)
        assert len(visible_bodies) == 1
        assert visible_bodies[0].room == 1
    
    def test_bodies_beyond_radius_not_visible(self):
        """Test that bodies beyond vision radius are not visible."""
        world, _ = self.engine.create_initial_state(seed=42)
        from aegis.core.world import BodyState
        
        observer_id = 0
        world.agents[observer_id].room = 0
        
        # Add body in far room
        body = BodyState(agent_id=99, room=8, tick_killed=0)
        world.bodies.append(body)
        
        visible_bodies = self.engine.get_visible_bodies(world, observer_id)
        
        # Room 8 is far from room 0
        dist = self.engine.get_graph_distance(world, 0, 8)
        if dist > self.config.vision_radius:
            assert len(visible_bodies) == 0


class TestActionMask:
    """Test action mask generation."""
    
    def setup_method(self):
        self.config = GameConfig(
            num_survivors=3,
            num_impostors=1,
        )
        self.engine = StepEngine(self.config)
        self.obs_builder = ObservationBuilder(self.engine, self.config)
    
    def test_dead_agents_limited_actions(self):
        """Test that dead agents have very limited actions."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        # Kill an agent
        dead_id = 0
        world.agents[dead_id].alive = False
        
        obs = self.obs_builder.build(world, dead_id)
        mask = obs["action_mask"]
        
        # Dead agents should have minimal valid actions
        valid_count = np.sum(mask)
        assert valid_count <= 2  # Usually just one no-op action
    
    def test_kill_masked_when_on_cooldown(self):
        """Test that kill actions are masked when on cooldown."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        impostor = None
        for agent in world.agents.values():
            if agent.role == Role.IMPOSTOR:
                impostor = agent
                impostor.kill_cooldown = 10  # On cooldown
                break
        
        obs = self.obs_builder.build(world, impostor.agent_id)
        mask = obs["action_mask"]
        
        # Kill actions should be masked
        kill_actions = range(self.engine.offset_kill, self.engine.offset_close)
        for action in kill_actions:
            assert mask[action] == 0
    
    def test_kill_masked_for_survivors(self):
        """Test that survivors cannot kill."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        survivor = None
        for agent in world.agents.values():
            if agent.role == Role.SURVIVOR:
                survivor = agent
                break
        
        obs = self.obs_builder.build(world, survivor.agent_id)
        mask = obs["action_mask"]
        
        # All kill actions should be masked for survivors
        kill_actions = range(self.engine.offset_kill, self.engine.offset_close)
        for action in kill_actions:
            assert mask[action] == 0
    
    def test_work_masked_when_not_in_task_room(self):
        """Test that WORK is masked when not in the correct room."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        survivor = None
        for agent in world.agents.values():
            if agent.role == Role.SURVIVOR and agent.tasks:
                survivor = agent
                break
        
        if survivor and survivor.tasks:
            task = survivor.tasks[0]
            target_room = task.current_room()
            
            # Put survivor in wrong room
            if target_room is not None:
                survivor.room = (target_room + 1) % len(self.engine.rooms)
                
                obs = self.obs_builder.build(world, survivor.agent_id)
                mask = obs["action_mask"]
                
                # WORK should be masked
                assert mask[self.engine.offset_work] == 0
    
    def test_work_enabled_in_task_room(self):
        """Test that WORK is enabled when in the correct room."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        survivor = None
        for agent in world.agents.values():
            if agent.role == Role.SURVIVOR and agent.tasks:
                survivor = agent
                break
        
        if survivor and survivor.tasks:
            task = survivor.tasks[0]
            target_room = task.current_room()
            
            if target_room is not None:
                # Put survivor in correct room
                survivor.room = target_room
                
                obs = self.obs_builder.build(world, survivor.agent_id)
                mask = obs["action_mask"]
                
                # WORK should be enabled
                assert mask[self.engine.offset_work] == 1
    
    def test_report_masked_without_body(self):
        """Test that REPORT is masked when no body in room."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        agent = list(world.agents.values())[0]
        agent.room = 0
        world.bodies = []  # No bodies
        
        obs = self.obs_builder.build(world, agent.agent_id)
        mask = obs["action_mask"]
        
        # REPORT should be masked
        assert mask[self.engine.offset_report] == 0
    
    def test_report_enabled_with_body(self):
        """Test that REPORT is enabled when body in room."""
        world, _ = self.engine.create_initial_state(seed=42)
        from aegis.core.world import BodyState
        
        agent = list(world.agents.values())[0]
        agent.room = 0
        
        # Add body in same room
        body = BodyState(agent_id=99, room=0, tick_killed=0)
        world.bodies.append(body)
        
        obs = self.obs_builder.build(world, agent.agent_id)
        mask = obs["action_mask"]
        
        # REPORT should be enabled
        assert mask[self.engine.offset_report] == 1
    
    def test_vote_actions_only_in_voting_phase(self):
        """Test that vote actions are only available in voting phase."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        agent = list(world.agents.values())[0]
        
        # In free play, votes should be masked
        world.phase = Phase.FREE_PLAY
        obs = self.obs_builder.build(world, agent.agent_id)
        mask = obs["action_mask"]
        
        vote_actions = range(self.engine.offset_vote, self.engine.offset_vote_skip + 1)
        for action in vote_actions:
            assert mask[action] == 0
        
        # In voting, votes should be enabled
        world.phase = Phase.VOTING
        from aegis.core.world import MeetingState
        world.meeting = MeetingState(reporter_id=0, body_location=0, body_agent_id=1)
        
        obs = self.obs_builder.build(world, agent.agent_id)
        mask = obs["action_mask"]
        
        # At least skip should be enabled
        assert mask[self.engine.offset_vote_skip] == 1


class TestObservationContent:
    """Test observation content correctness."""
    
    def setup_method(self):
        self.config = GameConfig(
            num_survivors=3,
            num_impostors=1,
            vision_radius=2,
        )
        self.engine = StepEngine(self.config)
        self.obs_builder = ObservationBuilder(self.engine, self.config)
    
    def test_visible_agents_correct(self):
        """Test that visible_agents observation is correct."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        # Setup specific positions
        world.agents[0].room = 0
        world.agents[1].room = 0  # Same room
        world.agents[2].room = 1  # Adjacent
        world.agents[3].room = 8  # Far
        
        obs = self.obs_builder.build(world, 0)
        visible_agents = obs["visible_agents"]
        
        # Agent 0 sees itself
        assert visible_agents[0, 0] == 1  # seen
        
        # Agent 1 in same room - visible
        assert visible_agents[1, 0] == 1
        
        # Agent 2 adjacent - visible (within radius 2)
        assert visible_agents[2, 0] == 1
    
    def test_phase_observation_correct(self):
        """Test that phase is correctly observed."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        for phase in [Phase.FREE_PLAY, Phase.MEETING, Phase.VOTING]:
            world.phase = phase
            if phase in [Phase.MEETING, Phase.VOTING]:
                from aegis.core.world import MeetingState
                world.meeting = MeetingState(reporter_id=0, body_location=0, body_agent_id=1)
            
            obs = self.obs_builder.build(world, 0)
            assert obs["phase"] == int(phase)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

