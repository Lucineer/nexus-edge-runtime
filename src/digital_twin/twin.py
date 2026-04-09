"""Nexus Edge Digital Twin — state mirroring and predictive simulation.

Real-time state mirror, predictive simulation, anomaly detection,
and virtual replay for autonomous agents.
"""
import math, time, random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any


@dataclass
class TwinState:
    agent_id: str
    timestamp: float = 0.0
    position: Tuple[float, float, float] = (0, 0, 0)  # x, y, z
    velocity: Tuple[float, float, float] = (0, 0, 0)
    orientation: Tuple[float, float, float] = (0, 0, 0)  # roll, pitch, yaw
    battery_pct: float = 100.0
    temperature: float = 25.0
    motor_active: bool = False
    sensor_health: Dict[str, float] = field(default_factory=dict)  # sensor -> health
    anomaly_flags: List[str] = field(default_factory=list)


@dataclass
class PredictedState:
    timestamp: float
    position: Tuple[float, float, float]
    battery_pct: float
    confidence: float


class StateMirror:
    """Real-time state mirroring with history buffer."""

    def __init__(self, buffer_seconds: float = 300):
        self.buffer_seconds = buffer_seconds
        self.states: Dict[str, List[TwinState]] = {}
        self.current: Dict[str, TwinState] = {}

    def update(self, state: TwinState) -> None:
        now = time.time()
        state.timestamp = state.timestamp or now
        self.current[state.agent_id] = state

        if state.agent_id not in self.states:
            self.states[state.agent_id] = []

        self.states[state.agent_id].append(state)

        # Prune old entries
        cutoff = now - self.buffer_seconds
        self.states[state.agent_id] = [
            s for s in self.states[state.agent_id] if s.timestamp >= cutoff]

    def get_history(self, agent_id: str,
                    duration: float = 0) -> List[TwinState]:
        if agent_id not in self.states:
            return []
        if duration <= 0:
            return self.states[agent_id]
        cutoff = time.time() - duration
        return [s for s in self.states[agent_id] if s.timestamp >= cutoff]

    def get_state_delta(self, agent_id: str) -> Dict[str, float]:
        """Compute rate of change for key fields."""
        history = self.get_history(agent_id, duration=10)
        if len(history) < 2:
            return {}
        first, last = history[0], history[-1]
        dt = last.timestamp - first.timestamp
        if dt <= 0:
            return {}
        return {
            "dx": (last.position[0] - first.position[0]) / dt,
            "dy": (last.position[1] - first.position[1]) / dt,
            "dz": (last.position[2] - first.position[2]) / dt,
            "battery_drain": (first.battery_pct - last.battery_pct) / dt,
            "temp_change": (last.temperature - first.temperature) / dt,
        }


class PredictiveSimulator:
    """Forward simulation of agent state."""

    def __init__(self):
        self.power_models: Dict[str, float] = {}  # agent -> watts consumed

    def set_power_model(self, agent_id: str, watts_active: float,
                       watts_idle: float = 0.5):
        self.power_models[agent_id] = watts_active

    def predict(self, state: TwinState, duration_s: float,
               steps: int = 10) -> List[PredictedState]:
        """Predict future states using simple kinematics + power model."""
        predictions = []
        dt = duration_s / steps

        x, y, z = state.position
        vx, vy, vz = state.velocity
        bat = state.battery_pct
        now = state.timestamp or time.time()

        watts = self.power_models.get(state.agent_id, 10.0)

        for i in range(steps):
            t = now + (i + 1) * dt
            x += vx * dt
            y += vy * dt
            z += vz * dt

            drain_rate = watts * dt / 3600  # rough: W * s / 3600 = Wh used
            bat -= drain_rate * 2  # simplified drain model
            bat = max(0, bat)

            confidence = max(0, 1.0 - i / steps * 0.5)
            predictions.append(PredictedState(
                t, (x, y, z), bat, confidence))

        return predictions

    def estimate_battery_remaining(self, state: TwinState) -> float:
        """Estimate time until battery depletion (seconds)."""
        watts = self.power_models.get(state.agent_id, 10.0)
        drain_per_sec = watts / 3600 * 2  # simplified
        if drain_per_sec <= 0:
            return float('inf')
        return state.battery_pct / drain_per_sec


class AnomalyDetector:
    """Detect anomalies by comparing actual vs predicted state."""

    def __init__(self):
        self.baselines: Dict[str, Dict[str, Tuple[float, float]]] = {}
        # agent_id -> {field: (mean, std)}

    def learn_baseline(self, agent_id: str, states: List[TwinState]) -> None:
        if not states:
            return
        fields = ["battery_pct", "temperature"]
        self.baselines[agent_id] = {}

        for field_name in fields:
            values = [getattr(s, field_name) for s in states]
            mean = sum(values) / len(values)
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
            self.baselines[agent_id][field_name] = (mean, std)

    def check(self, state: TwinState,
              sigma_threshold: float = 3.0) -> List[str]:
        anomalies = []
        baseline = self.baselines.get(state.agent_id, {})

        for field_name, (mean, std) in baseline.items():
            value = getattr(state, field_name)
            if std > 0:
                z = abs(value - mean) / std
                if z > sigma_threshold:
                    anomalies.append(
                        f"{field_name}: {value:.1f} "
                        f"({z:.1f}σ from baseline {mean:.1f}±{std:.1f})")

        # Battery critical
        if state.battery_pct < 10:
            anomalies.append(f"CRITICAL: battery at {state.battery_pct:.1f}%")

        # Temperature high
        if state.temperature > 70:
            anomalies.append(f"CRITICAL: temperature {state.temperature:.1f}C")

        state.anomaly_flags = anomalies
        return anomalies


def demo():
    print("=== Digital Twin ===\n")
    random.seed(42)

    mirror = StateMirror(buffer_seconds=60)
    simulator = PredictiveSimulator()
    detector = AnomalyDetector()

    # Simulate agent trajectory
    agent_id = "auv_alpha"
    simulator.set_power_model(agent_id, 15.0)

    print("--- Simulating 60s of telemetry ---")
    states = []
    for t in range(0, 60, 2):
        state = TwinState(
            agent_id=agent_id,
            timestamp=time.time() - 60 + t,
            position=(100 + t * 0.5, 200 + t * 0.3, 5.0),
            velocity=(0.5, 0.3, 0),
            orientation=(0, 5, 45 + t * 0.1),
            battery_pct=100 - t * 0.15,
            temperature=25 + random.gauss(0, 0.5),
            motor_active=True,
            sensor_health={"sonar": 0.95 + random.gauss(0, 0.02),
                          "imu": 0.98 + random.gauss(0, 0.01)}
        )
        mirror.update(state)
        states.append(state)

    # State delta
    delta = mirror.get_state_delta(agent_id)
    print(f"  Velocity: ({delta.get('dx',0):.2f}, {delta.get('dy',0):.2f}, {delta.get('dz',0):.2f}) m/s")
    print(f"  Battery drain: {delta.get('battery_drain',0):.3f} %/s")

    # Current state
    current = mirror.current[agent_id]
    print(f"  Current: pos=({current.position[0]:.0f},{current.position[1]:.0f},{current.position[2]:.0f})")
    print(f"  Battery: {current.battery_pct:.1f}%, Temp: {current.temperature:.1f}C")

    # Predictive simulation
    print("\n--- Predictive Simulation (30s forward) ---")
    predictions = simulator.predict(current, 30, 5)
    for p in predictions:
        print(f"  +{p.timestamp-current.timestamp:.0f}s: "
              f"pos=({p.position[0]:.0f},{p.position[1]:.0f}) "
              f"bat={p.battery_pct:.1f}% conf={p.confidence:.2f}")

    # Battery estimate
    remaining = simulator.estimate_battery_remaining(current)
    print(f"\n  Estimated battery life: {remaining:.0f}s ({remaining/60:.1f}min)")

    # Anomaly detection
    print("\n--- Anomaly Detection ---")
    detector.learn_baseline(agent_id, states[:-5])  # train on first 55s
    normal = states[-5]
    anomalies = detector.check(normal)
    print(f"  Normal state: {len(anomalies)} anomalies")

    # Inject anomaly
    anomalous = TwinState(agent_id, temperature=85.0, battery_pct=8.0)
    anomalies = detector.check(anomalous)
    print(f"  Anomalous state: {len(anomalies)} anomalies")
    for a in anomalies:
        print(f"    {a}")

    # History size
    print(f"\n--- Mirror Stats ---")
    print(f"  Buffered states: {len(mirror.states[agent_id])}")
    print(f"  Tracked agents: {len(mirror.current)}")


if __name__ == "__main__":
    demo()
