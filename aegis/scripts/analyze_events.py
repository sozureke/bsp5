from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class Meeting:
    """Represents a single meeting with all its communications."""
    tick_start: int
    tick_end: int
    reporter_id: int
    body_room: int
    comm_actions: List[Dict]
    votes: dict[int, Optional[int]]
    ejected_id: Optional[int] = None
    ejected_role: Optional[int] = None
    
    @property
    def duration(self) -> int:
        return self.tick_end - self.tick_start
    
    @property
    def num_comm_actions(self) -> int:
        return len(self.comm_actions)
    
    @property
    def has_ejection(self) -> bool:
        return self.ejected_id is not None


class EventAnalyzer:
    """Analyzer for events.jsonl files."""
    
    def __init__(self, events_path: str):
        self.events_path = Path(events_path)
        self.events: List[Dict] = []
        self.meetings: list[Meeting] = []
        self._load_events()
        self._parse_meetings()
    
    def _load_events(self):
        """Load all events from JSONL file."""
        if not self.events_path.exists():
            raise FileNotFoundError(f"Events file not found: {self.events_path}")
        
        with open(self.events_path,  "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.events.append(json.loads(line))
        
        print(f"Loaded {len(self.events)} events from {self.events_path}")
    
    def _parse_meetings(self):
        """Parse meeting_summary events into Meeting objects."""
        for event in self.events:
            if event.get("event_type") ==  "meeting_summary":
                meeting = Meeting(
                    tick_start=event["meeting_tick_start"],
                    tick_end=event["tick"],
                    reporter_id=event["reporter_id"],
                    body_room=event["body_room"],
                    comm_actions=event.get("comm_actions", []),
                    votes=event.get("votes", {}),
                    ejected_id=event.get("ejected_id"),
                    ejected_role=event.get("ejected_role"),
                )
                self.meetings.append(meeting)
        
        print(f"Found {len(self.meetings)} meetings")
    
    def summary(self):
        """Print summary statistics."""
        print("\n" +  "=" * 70)
        print("EVENT ANALYSIS SUMMARY")
        print("=" * 70)
        

        event_types = Counter(e.get("event_type") for e in self.events)
        print(f"\nEvent Types:")
        for event_type, count in event_types.most_common():
            print(f"  {event_type:25s}: {count:6d}")
        

        if self.meetings:
            print(f"\nMeeting Statistics:")
            print(f"  Total meetings: {len(self.meetings)}")
            print(f"  Meetings with ejections: {sum(1 for m in self.meetings if m.has_ejection)}")
            print(f"  Average duration: {sum(m.duration for m in self.meetings) / len(self.meetings):.1f} ticks")
            print(f"  Average comm actions per meeting: {sum(m.num_comm_actions for m in self.meetings) / len(self.meetings):.1f}")
            

            all_comm_actions = []
            for meeting in self.meetings:
                all_comm_actions.extend(meeting.comm_actions)
            
            if all_comm_actions:
                action_types = Counter(a.get("action_name",  "UNKNOWN") for a in all_comm_actions)
                print(f"\nCommunication Actions:")
                for action, count in action_types.most_common():
                    print(f"  {action:20s}: {count:6d}")
                

                survivor_actions = [a for a in all_comm_actions if a.get("sender_role") == 0]
                impostor_actions = [a for a in all_comm_actions if a.get("sender_role") == 1]
                
                print(f"\nActions by Role:")
                print(f"  Survivors: {len(survivor_actions)} actions")
                print(f"  Impostors: {len(impostor_actions)} actions")
                
                if survivor_actions:
                    surv_action_types = Counter(a.get("action_name") for a in survivor_actions)
                    print(f"\n  Survivor action distribution:")
                    for action, count in surv_action_types.most_common(5):
                        print(f"    {action:20s}: {count:6d}")
                
                if impostor_actions:
                    imp_action_types = Counter(a.get("action_name") for a in impostor_actions)
                    print(f"\n  Impostor action distribution:")
                    for action, count in imp_action_types.most_common(5):
                        print(f"    {action:20s}: {count:6d}")
        

        win_events = [e for e in self.events if e.get("event_type") ==  "win"]
        if win_events:
            winners = Counter(e.get("winner") for e in win_events)
            print(f"\nGame Outcomes:")
            for winner, count in winners.items():
                winner_name =  "SURVIVORS" if winner == 0 else  "IMPOSTORS"
                print(f"  {winner_name}: {count}")
        
        print("=" * 70)
    
    def show_meeting(self, meeting_idx: int):
        """Show detailed information about a specific meeting."""
        if meeting_idx < 0 or meeting_idx >= len(self.meetings):
            print(f"Invalid meeting index: {meeting_idx} (valid range: 0-{len(self.meetings)-1})")
            return
        
        meeting = self.meetings[meeting_idx]
        
        print("\n" +  "=" * 70)
        print(f"MEETING #{meeting_idx}")
        print("=" * 70)
        print(f"Started: tick {meeting.tick_start}")
        print(f"Ended: tick {meeting.tick_end} (duration: {meeting.duration} ticks)")
        print(f"Reporter: agent_{meeting.reporter_id}")
        print(f"Body found in room: {meeting.body_room}")
        print(f"Total communications: {meeting.num_comm_actions}")
        
        if meeting.has_ejection:
            role_name =  "SURVIVOR" if meeting.ejected_role == 0 else  "IMPOSTOR"
            print(f"Ejected: agent_{meeting.ejected_id} ({role_name})")
        else:
            print("Ejected: None")
        
        print(f"\nVotes:")
        for voter_id, target_id in meeting.votes.items():
            if target_id is not None:
                print(f"  agent_{voter_id} -> agent_{target_id}")
            else:
                print(f"  agent_{voter_id} -> SKIP")
        
        print(f"\nCommunication Timeline:")
        for i, comm in enumerate(meeting.comm_actions, 1):
            sender_id = comm.get("sender_id")
            sender_role =  "SURVIVOR" if comm.get("sender_role") == 0 else  "IMPOSTOR"
            action_name = comm.get("action_name",  "UNKNOWN")
            target_id = comm.get("target_id")
            
            line = f"  [{comm.get('tick',  '?')}] agent_{sender_id} ({sender_role}): {action_name}"
            if target_id is not None:
                target_role = comm.get("target_role")
                if target_role is not None:
                    target_role_name =  "SURVIVOR" if target_role == 0 else  "IMPOSTOR"
                    line += f" -> agent_{target_id} ({target_role_name})"
                else:
                    line += f" -> agent_{target_id}"
            print(line)
        
        print("=" * 70)
    
    def find_interesting_meetings(self, min_comm_actions: int = 10, require_ejection: bool = False):
        """Find meetings with interesting patterns."""
        candidates = []
        
        for i, meeting in enumerate(self.meetings):
            if meeting.num_comm_actions < min_comm_actions:
                continue
            if require_ejection and not meeting.has_ejection:
                continue
            

            accuse_count = sum(1 for a in meeting.comm_actions if  "ACCUSE" in a.get("action_name",  ""))
            

            accuse_targets = defaultdict(int)
            for a in meeting.comm_actions:
                if  "ACCUSE" in a.get("action_name",  ""):
                    target_id = a.get("target_id")
                    if target_id is not None:
                        accuse_targets[target_id] += 1
            
            max_coordination = max(accuse_targets.values()) if accuse_targets else 0
            
            candidates.append({
                "index": i,
                "meeting": meeting,
                "accuse_count": accuse_count,
                "max_coordination": max_coordination,
                "has_ejection": meeting.has_ejection,
            })
        

        candidates.sort(key=lambda x: (x["max_coordination"], x["accuse_count"]), reverse=True)
        
        return candidates
    
    def analyze_impostor_strategies(self):
        """Analyze communication strategies used by impostors."""
        print("\n" +  "=" * 70)
        print("IMPOSTOR STRATEGY ANALYSIS")
        print("=" * 70)
        
        impostor_actions = []
        for meeting in self.meetings:
            for comm in meeting.comm_actions:
                if comm.get("sender_role") == 1:
                    impostor_actions.append({
                        "meeting_idx": self.meetings.index(meeting),
                        "action": comm,
                    })
        
        if not impostor_actions:
            print("No impostor communication actions found.")
            return
        
        print(f"\nTotal impostor actions: {len(impostor_actions)}")
        

        action_types = Counter(a["action"].get("action_name") for a in impostor_actions)
        print(f"\nImpostor Action Distribution:")
        for action, count in action_types.most_common():
            print(f"  {action:20s}: {count:6d}")
        

        accuse_actions = [a for a in impostor_actions if  "ACCUSE" in a["action"].get("action_name",  "")]
        support_actions = [a for a in impostor_actions if  "SUPPORT" in a["action"].get("action_name",  "")]
        
        if accuse_actions:
            print(f"\nImpostor ACCUSE targets:")
            accuse_targets = Counter(
                a["action"].get("target_role") for a in accuse_actions 
                if a["action"].get("target_role") is not None
            )
            for role, count in accuse_targets.items():
                role_name =  "SURVIVOR" if role == 0 else  "IMPOSTOR"
                print(f"  {role_name}: {count}")
        
        if support_actions:
            print(f"\nImpostor SUPPORT targets:")
            support_targets = Counter(
                a["action"].get("target_role") for a in support_actions 
                if a["action"].get("target_role") is not None
            )
            for role, count in support_targets.items():
                role_name =  "SURVIVOR" if role == 0 else  "IMPOSTOR"
                print(f"  {role_name}: {count}")
        
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze events.jsonl from training")
    parser.add_argument("events_path", type=str, help="Path to events.jsonl file")
    parser.add_argument("--summary", action="store_true", help="Show summary statistics")
    parser.add_argument("--meeting", type=int, help="Show detailed view of meeting by index")
    parser.add_argument("--find-interesting", action="store_true", 
                       help="Find meetings with interesting patterns")
    parser.add_argument("--min-comm", type=int, default=10,
                       help="Minimum communication actions for interesting meetings")
    parser.add_argument("--require-ejection", action="store_true",
                       help="Only show meetings with ejections")
    parser.add_argument("--impostor-strategies", action="store_true",
                       help="Analyze impostor communication strategies")
    
    args = parser.parse_args()
    
    analyzer = EventAnalyzer(args.events_path)
    
    if args.summary:
        analyzer.summary()
    
    if args.meeting is not None:
        analyzer.show_meeting(args.meeting)
    
    if args.find_interesting:
        candidates = analyzer.find_interesting_meetings(
            min_comm_actions=args.min_comm,
            require_ejection=args.require_ejection
        )
        print(f"\nFound {len(candidates)} interesting meetings:")
        for cand in candidates[:10]:
            meeting = cand["meeting"]
            print(f"\n  Meeting #{cand['index']}:")
            print(f"    Duration: {meeting.duration} ticks")
            print(f"    Comm actions: {meeting.num_comm_actions}")
            print(f"    ACCUSE count: {cand['accuse_count']}")
            print(f"    Max coordination: {cand['max_coordination']} ACCUSE on same target")
            print(f"    Ejected: {'Yes' if meeting.has_ejection else  'No'}")
    
    if args.impostor_strategies:
        analyzer.analyze_impostor_strategies()
    

    if not any([args.summary, args.meeting is not None, args.find_interesting, args.impostor_strategies]):
        analyzer.summary()


if __name__ ==  "__main__":
    main()

