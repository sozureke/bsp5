"""JSONL event logger for tick-level events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, TextIO, Any
from datetime import datetime

import numpy as np

from aegis.core.events import Event


class EventWriter:
    """Writes events to a JSONL file."""
    
    def __init__(self, path: Optional[str] = None, auto_timestamp: bool = True):
        """
        Initialize event writer.
        
        Args:
            path: Output file path. If None, generates timestamped path.
            auto_timestamp: Add timestamp to filename if path not specified.
        """
        if path is None:
            if auto_timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"events_{timestamp}.jsonl"
            else:
                path = "events.jsonl"
        
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: Optional[TextIO] = None
        self._event_count = 0
    
    def _ensure_open(self) -> TextIO:
        """Ensure file is open for writing."""
        if self._file is None:
            self._file = open(self.path, "a")
        return self._file
    
    @staticmethod
    def _convert_to_json_serializable(obj: Any) -> Any:
        """Recursively convert NumPy types and other non-serializable types to native Python types."""
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: EventWriter._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [EventWriter._convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    def write(self, event: Event) -> None:
        """Write an event to the log."""
        f = self._ensure_open()
        event_dict = event.to_dict()
        # Convert NumPy types to native Python types for JSON serialization
        serializable_dict = self._convert_to_json_serializable(event_dict)
        f.write(json.dumps(serializable_dict) + "\n")
        self._event_count += 1
    
    def write_many(self, events: list[Event]) -> None:
        """Write multiple events."""
        for event in events:
            self.write(event)
    
    def flush(self) -> None:
        """Flush the file buffer."""
        if self._file:
            self._file.flush()
    
    def close(self) -> None:
        """Close the file."""
        if self._file:
            self._file.close()
            self._file = None
    
    def __enter__(self) -> "EventWriter":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    @property
    def event_count(self) -> int:
        """Number of events written."""
        return self._event_count


class EventReader:
    """Reads events from a JSONL file."""
    
    def __init__(self, path: str):
        """
        Initialize event reader.
        
        Args:
            path: Input file path
        """
        self.path = Path(path)
    
    def read_all(self) -> list[dict]:
        """Read all events from the file."""
        events = []
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events
    
    def iterate(self):
        """Iterate over events lazily."""
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    
    def filter_by_type(self, event_type: str) -> list[dict]:
        """Get all events of a specific type."""
        return [e for e in self.read_all() if e.get("event_type") == event_type]
    
    def filter_by_tick(self, tick: int) -> list[dict]:
        """Get all events at a specific tick."""
        return [e for e in self.read_all() if e.get("tick") == tick]
    
    def get_tick_range(self) -> tuple[int, int]:
        """Get the range of ticks in the log."""
        events = self.read_all()
        if not events:
            return (0, 0)
        ticks = [e.get("tick", 0) for e in events]
        return (min(ticks), max(ticks))

