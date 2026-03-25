"""Simulated clock for discrete-event simulation."""


class SimulatedClock:
    """Tracks simulation time."""

    def __init__(self):
        self.current_tick: float = 0.0

    def advance_to(self, tick: float) -> None:
        self.current_tick = tick

    def advance_by(self, delta: float) -> None:
        self.current_tick += delta

    def __repr__(self) -> str:
        return f"Clock(tick={self.current_tick:.1f})"
