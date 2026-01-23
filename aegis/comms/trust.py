from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from aegis.comms.actions import CommVocab


@dataclass
class TrustUpdate:
    """A delayed trust update event."""
    target_agent: int
    trust_delta: float
    source_action: str
    tick_created: int


class TrustManager:
    """
    Manages trust matrix and delayed trust updates.
    
    Trust matrix: trust[observer][target] ∈ [0, 1]
    
    Trust effects are delayed by a configurable number of ticks to prevent
    immediate influence on voting decisions.
    """
    
    def __init__(
        self,
        num_agents: int,
        delay_ticks: int = 5,
        initial_trust: float = 0.5,
    ):
        """
        Initialize trust manager.
        
        Args:
            num_agents: Number of agents in the game
            delay_ticks: Number of ticks to delay trust updates
            initial_trust: Initial trust value for all agent pairs
        """

        assert num_agents > 0,  "num_agents must be positive"
        assert delay_ticks >= 0,  "delay_ticks must be non-negative"
        assert 0.0 <= initial_trust <= 1.0,  "initial_trust must be in [0, 1]"
        
        self.num_agents = num_agents
        self.delay_ticks = delay_ticks
        


        self.trust_matrix = np.full((num_agents, num_agents), initial_trust, dtype=np.float32)
        

        for i in range(num_agents):
            self.trust_matrix[i, i] = 1.0
        

        self.pending_updates: dict[int, list[TrustUpdate]] = {i: [] for i in range(num_agents)}
        

        self.trust_deltas: dict[str, list[float]] = {
            "support": [],
            "accuse": [],
            "defend": [],
            "question": [],
        }
    
    def reset(self, initial_trust: float = 0.5) -> None:
        """Reset trust matrix and pending updates."""
        self.trust_matrix = np.full((self.num_agents, self.num_agents), initial_trust, dtype=np.float32)
        for i in range(self.num_agents):
            self.trust_matrix[i, i] = 1.0
        
        self.pending_updates = {i: [] for i in range(self.num_agents)}
        self.trust_deltas = {
            "support": [],
            "accuse": [],
            "defend": [],
            "question": [],
        }
    
    def buffer_comm_action(
        self,
        sender_id: int,
        action_id: int,
        current_tick: int,
        support_delta: float = 0.15,
        accuse_delta: float = -0.20,
        defend_delta: float = 0.05,
        question_delta: float = -0.08,
    ) -> None:
        """
        Buffer a communication action for delayed trust update.
        
        Args:
            sender_id: Agent who sent the communication
            action_id: Communication action ID
            current_tick: Current game tick
            support_delta: Trust increase for SUPPORT actions (positive)
            accuse_delta: Trust decrease for ACCUSE actions (negative)
            defend_delta: Trust increase for DEFEND_SELF (positive)
            question_delta: Trust decrease for QUESTION actions (negative)
        """
        if CommVocab.is_support(action_id):
            target_id = CommVocab.get_support_target(action_id)
            if target_id is not None and target_id < self.num_agents:
                update = TrustUpdate(
                    target_agent=target_id,
                    trust_delta=support_delta,
                    source_action="support",
                    tick_created=current_tick,
                )
                self.pending_updates[target_id].append(update)
        
        elif CommVocab.is_accuse(action_id):
            target_id = CommVocab.get_accuse_target(action_id)
            if target_id is not None and target_id < self.num_agents:
                update = TrustUpdate(
                    target_agent=target_id,
                    trust_delta=accuse_delta,
                    source_action="accuse",
                    tick_created=current_tick,
                )
                self.pending_updates[target_id].append(update)
        
        elif CommVocab.is_defend(action_id):

            update = TrustUpdate(
                target_agent=sender_id,
                trust_delta=defend_delta,
                source_action="defend",
                tick_created=current_tick,
            )
            self.pending_updates[sender_id].append(update)
        
        elif CommVocab.is_question(action_id):
            target_id = CommVocab.get_question_target(action_id)
            if target_id is not None and target_id < self.num_agents:
                update = TrustUpdate(
                    target_agent=target_id,
                    trust_delta=question_delta,
                    source_action="question",
                    tick_created=current_tick,
                )
                self.pending_updates[target_id].append(update)
    
    def activate_delayed_updates(self, current_tick: int) -> None:
        """
        Activate pending trust updates that have passed the delay threshold.
        
        Updates trust_matrix with all pending updates that are old enough.
        """
        for target_id in range(self.num_agents):
            pending = self.pending_updates.get(target_id, [])
            if not pending:
                continue
            

            still_pending = []
            newly_active = []
            
            for update in pending:
                if current_tick - update.tick_created >= self.delay_ticks:
                    newly_active.append(update)
                else:
                    still_pending.append(update)
            

            self.pending_updates[target_id] = still_pending
            

            for update in newly_active:

                for observer_id in range(self.num_agents):
                    if observer_id != target_id:
                        self.trust_matrix[observer_id, target_id] += update.trust_delta
                        self.trust_matrix[observer_id, target_id] = np.clip(
                            self.trust_matrix[observer_id, target_id], 0.0, 1.0
                        )
                

                self.trust_deltas[update.source_action].append(update.trust_delta)
        

        for i in range(self.num_agents):
            assert self.trust_matrix[i, i] == 1.0, f"Self-trust violated for agent {i}"
    
    def clear_pending_updates(self) -> None:
        """Clear all pending trust updates (called when meeting ends)."""
        self.pending_updates = {i: [] for i in range(self.num_agents)}
    
    def get_trust_vector(self, observer_id: int) -> np.ndarray:
        """Get trust values from one observer to all other agents."""
        return self.trust_matrix[observer_id].copy()
    
    def get_distrust_scores(self) -> dict[int, float]:
        """
        Get distrust scores for all agents (for voting integration).
        
        Distrust = 1 - avg_trust, where avg_trust is the average trust
        that all other agents have in a given agent.
        
        Returns:
            dict mapping agent_id -> distrust score [0, 1]
        """
        distrust = {}
        for target_id in range(self.num_agents):

            trust_values = []
            for observer_id in range(self.num_agents):
                if observer_id != target_id:
                    trust_values.append(self.trust_matrix[observer_id, target_id])
            
            avg_trust = np.mean(trust_values) if trust_values else 0.5
            distrust[target_id] = 1.0 - avg_trust
        
        return distrust
    
    def get_statistics(self) -> dict:
        """
        Get aggregated statistics for logging.
        
        Returns:
            dict with count and average delta per action type
        """
        stats = {}
        for action_type, deltas in self.trust_deltas.items():
            if deltas:
                stats[f"{action_type}_count"] = len(deltas)
                stats[f"{action_type}_avg_delta"] = float(np.mean(deltas))
            else:
                stats[f"{action_type}_count"] = 0
                stats[f"{action_type}_avg_delta"] = 0.0
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset statistics tracking (called at episode start)."""
        self.trust_deltas = {
            "support": [],
            "accuse": [],
            "defend": [],
            "question": [],
        }

