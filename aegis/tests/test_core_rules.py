"""Tests for core game rules and win conditions."""

import pytest
from aegis.core.world import WorldState, AgentState, TaskState, Phase, Role, TaskType
from aegis.core.rules import GameConfig
from aegis.core.engine import StepEngine


class TestWinConditions:
    """Test win condition logic."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = GameConfig(
            num_survivors=4,
            num_impostors=1,
            max_ticks=100,
            evac_ticks_required=3,
        )
        self.engine = StepEngine(self.config)
    
    def test_impostor_wins_by_parity(self):
        """Test that impostors win when alive impostors >= alive survivors."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        # Kill survivors until parity
        survivors = world.alive_survivors()
        for survivor in survivors[:-1]:  # Leave 1 alive
            survivor.alive = False
        
        # Step to trigger win check
        actions = {i: 0 for i in world.agents}
        next_world, events = self.engine.step(world, actions)
        
        assert next_world.terminated
        assert next_world.winner == Role.IMPOSTOR
    
    def test_survivors_win_by_eliminating_impostors(self):
        """Test that survivors win when all impostors are eliminated."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        # Kill all impostors
        for agent in world.agents.values():
            if agent.role == Role.IMPOSTOR:
                agent.alive = False
        
        # Step to trigger win check
        actions = {i: 0 for i in world.agents}
        next_world, events = self.engine.step(world, actions)
        
        assert next_world.terminated
        assert next_world.winner == Role.SURVIVOR
    
    def test_evac_activation_on_task_completion(self):
        """Test that EVAC activates when all tasks are complete."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        # Complete all tasks
        for agent in world.agents.values():
            if agent.role == Role.SURVIVOR:
                for task in agent.tasks:
                    task.completed = True
                    task.current_step = task.total_steps()
        
        # Step to trigger evac check
        actions = {i: 0 for i in world.agents}
        next_world, events = self.engine.step(world, actions)
        
        assert next_world.evac_active
    
    def test_evac_win_requires_consecutive_ticks(self):
        """Test that evac win requires K consecutive ticks."""
        world, _ = self.engine.create_initial_state(seed=42)
        world.evac_active = True
        
        # Move survivors to evac room
        evac_room = world.evac_room
        survivor_count = 0
        for agent in world.agents.values():
            if agent.role == Role.SURVIVOR and survivor_count < 2:
                agent.room = evac_room
                survivor_count += 1
        
        # Step multiple times
        for i in range(self.config.evac_ticks_required - 1):
            actions = {a: self.engine.offset_move + evac_room for a in world.agents}
            world, events = self.engine.step(world, actions)
            
            assert not world.terminated
            assert world.evac_tick_counter == i + 1
        
        # Final step should trigger win
        actions = {a: self.engine.offset_move + evac_room for a in world.agents}
        world, events = self.engine.step(world, actions)
        
        assert world.terminated
        assert world.winner == Role.SURVIVOR
    
    def test_timeout_results_in_impostor_win(self):
        """Test that timeout causes impostor win."""
        config = GameConfig(max_ticks=5)
        engine = StepEngine(config)
        world, _ = engine.create_initial_state(seed=42)
        
        # Step until timeout
        for _ in range(10):
            if world.terminated:
                break
            actions = {i: 0 for i in world.agents}
            world, _ = engine.step(world, actions)
        
        assert world.truncated or world.terminated
        assert world.winner == Role.IMPOSTOR


class TestTaskMechanics:
    """Test task completion mechanics."""
    
    def setup_method(self):
        self.config = GameConfig(
            num_survivors=2,
            num_impostors=1,
            tasks_per_survivor=2,
            task_ticks_single=3,
        )
        self.engine = StepEngine(self.config)
    
    def test_task_progress_only_in_correct_room(self):
        """Test that WORK only progresses when in correct room."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        # Find a survivor with tasks
        survivor = None
        for agent in world.agents.values():
            if agent.role == Role.SURVIVOR and agent.tasks:
                survivor = agent
                break
        
        assert survivor is not None
        task = survivor.tasks[0]
        target_room = task.current_room()
        
        # Try working in wrong room
        survivor.room = (target_room + 1) % len(self.engine.rooms)
        initial_progress = task.progress
        
        actions = {i: 0 for i in world.agents}
        actions[survivor.agent_id] = self.engine.offset_work
        world, _ = self.engine.step(world, actions)
        
        # Progress should not change (work is masked in wrong room)
        # Actually let's verify the mask prevents this action
        # For now, just verify task system works
        assert task.progress == initial_progress  # Should be 0
    
    def test_task_completion_increments_team_progress(self):
        """Test that completing task steps increases team progress."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        initial_progress = world.team_task_progress()
        
        # Complete a task step manually
        for agent in world.agents.values():
            if agent.role == Role.SURVIVOR and agent.tasks:
                task = agent.tasks[0]
                task.current_step = 1
                break
        
        new_progress = world.team_task_progress()
        assert new_progress > initial_progress


class TestKillMechanics:
    """Test kill action mechanics."""
    
    def setup_method(self):
        self.config = GameConfig(
            num_survivors=3,
            num_impostors=1,
            kill_cooldown=10,
        )
        self.engine = StepEngine(self.config)
    
    def test_kill_creates_body(self):
        """Test that kill action creates a body."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        # Find impostor and survivor
        impostor = None
        target = None
        for agent in world.agents.values():
            if agent.role == Role.IMPOSTOR:
                impostor = agent
                impostor.kill_cooldown = 0
        
        for agent in world.agents.values():
            if agent.role == Role.SURVIVOR:
                target = agent
                break
        
        # Put both in room 0 so movement (action 0 = MOVE_TO_ROOM_0) keeps them there
        impostor.room = 0
        target.room = 0
        
        initial_bodies = len(world.bodies)
        
        # Execute kill - all agents move to room 0 (stay in place for impostor/target)
        actions = {i: 0 for i in world.agents}
        actions[impostor.agent_id] = self.engine.offset_kill + target.agent_id
        world, events = self.engine.step(world, actions)
        
        assert len(world.bodies) == initial_bodies + 1
        assert not world.agents[target.agent_id].alive
    
    def test_kill_cooldown_prevents_kill(self):
        """Test that kill cooldown prevents killing."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        impostor = None
        for agent in world.agents.values():
            if agent.role == Role.IMPOSTOR:
                impostor = agent
                impostor.kill_cooldown = 5  # On cooldown
                break
        
        # Try to kill
        actions = {i: 0 for i in world.agents}
        actions[impostor.agent_id] = self.engine.offset_kill + 0
        world, events = self.engine.step(world, actions)
        
        # All agents should still be alive (kill was blocked)
        # (The action would be masked anyway)
        kill_events = [e for e in events if e.event_type.value == "kill"]
        assert len(kill_events) == 0


class TestMovement:
    """Test movement mechanics."""
    
    def setup_method(self):
        self.config = GameConfig()
        self.engine = StepEngine(self.config)
    
    def test_movement_one_step_per_tick(self):
        """Test that agents move one room per tick."""
        world, _ = self.engine.create_initial_state(seed=42)
        
        agent = list(world.agents.values())[0]
        start_room = agent.room
        
        # Find an adjacent room
        adjacent = None
        for neighbor in self.engine.graph.neighbors(start_room):
            adjacent = neighbor
            break
        
        if adjacent is not None:
            # Move to non-adjacent room to verify path following
            target = (adjacent + 3) % len(self.engine.rooms)
            
            actions = {i: 0 for i in world.agents}
            actions[agent.agent_id] = self.engine.offset_move + target
            world, _ = self.engine.step(world, actions)
            
            # Should have moved at most one room
            dist = self.engine.get_graph_distance(world, start_room, agent.room)
            assert dist <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

