"""Simulation event types."""

from dataclasses import dataclass
from enum import Enum, auto


class EventType(Enum):
    TASK_DISPATCH = auto()
    TASK_COMPLETE = auto()
    REALLOCATION = auto()
    FEEDBACK_CHECK = auto()
    QPU_CALIBRATION = auto()


@dataclass
class Event:
    timestamp: float
    event_type: EventType

    def __lt__(self, other):
        return self.timestamp < other.timestamp


@dataclass
class TaskDispatchEvent(Event):
    task_id: str = ""
    device_id: str = ""
    event_type: EventType = EventType.TASK_DISPATCH


@dataclass
class TaskCompleteEvent(Event):
    task_id: str = ""
    device_id: str = ""
    event_type: EventType = EventType.TASK_COMPLETE


@dataclass
class ReallocationEvent(Event):
    task_id: str = ""
    from_device_id: str = ""
    to_device_id: str = ""
    reason: str = ""
    event_type: EventType = EventType.REALLOCATION
