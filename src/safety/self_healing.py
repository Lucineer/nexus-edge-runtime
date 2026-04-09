"""Nexus Edge Self-Healing — automatic failure recovery.

Component health monitoring, fault classification, recovery strategies,
and graceful degradation for autonomous agents.
"""
import time, random, enum, math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable


class ComponentType(enum.Enum):
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    COMMUNICATION = "comms"
    COMPUTATION = "compute"
    POWER = "power"
    NAVIGATION = "nav"
    MEMORY = "memory"


class FaultSeverity(enum.Enum):
    NONE = 0
    WARNING = 1
    MINOR = 2
    MAJOR = 3
    CRITICAL = 4


class RecoveryAction(enum.Enum):
    RESTART = "restart"
    RESET = "reset"
    SWITCH_BACKUP = "switch_backup"
    REDUCE_LOAD = "reduce_load"
    SAFE_MODE = "safe_mode"
    ESCALATE = "escalate"
    IGNORE = "ignore"


@dataclass
class ComponentHealth:
    component_id: str
    comp_type: ComponentType
    health: float = 1.0  # 0=dead, 1=perfect
    last_heartbeat: float = 0.0
    error_count: int = 0
    consecutive_errors: int = 0
    last_error: str = ""
    recovery_attempts: int = 0
    has_backup: bool = False
    is_primary: bool = True


@dataclass
class FaultEvent:
    component_id: str
    severity: FaultSeverity
    description: str
    timestamp: float = 0.0
    recovery_action: RecoveryAction = RecoveryAction.IGNORE
    resolved: bool = False


class HealthMonitor:
    """Track component health with heartbeat monitoring."""

    def __init__(self, heartbeat_timeout_ms: float = 2000,
                 error_threshold: int = 5):
        self.timeout = heartbeat_timeout_ms / 1000
        self.error_threshold = error_threshold
        self.components: Dict[str, ComponentHealth] = {}
        self.fault_history: List[FaultEvent] = []

    def register(self, comp: ComponentHealth) -> None:
        self.components[comp.component_id] = comp

    def heartbeat(self, component_id: str) -> None:
        comp = self.components.get(component_id)
        if comp:
            comp.last_heartbeat = time.time()
            comp.consecutive_errors = 0
            comp.health = min(1.0, comp.health + 0.1)

    def report_error(self, component_id: str, error: str) -> FaultSeverity:
        comp = self.components.get(component_id)
        if not comp:
            return FaultSeverity.NONE

        comp.error_count += 1
        comp.consecutive_errors += 1
        comp.last_error = error
        comp.health = max(0, comp.health - 0.15)

        if comp.consecutive_errors >= self.error_threshold:
            severity = FaultSeverity.CRITICAL
            comp.health = 0.0
        elif comp.consecutive_errors >= 3:
            severity = FaultSeverity.MAJOR
        elif comp.consecutive_errors >= 1:
            severity = FaultSeverity.MINOR
        else:
            severity = FaultSeverity.WARNING

        event = FaultEvent(component_id, severity, error, time.time())
        self.fault_history.append(event)
        return severity

    def check_health(self) -> List[FaultEvent]:
        now = time.time()
        events = []
        for cid, comp in self.components.items():
            if comp.last_heartbeat > 0:
                age = now - comp.last_heartbeat
                if age > self.timeout:
                    comp.health = max(0, comp.health - 0.05)
                    events.append(FaultEvent(
                        cid, FaultSeverity.MAJOR,
                        f"Missed heartbeat ({age:.1f}s)", now))
        return events


class RecoveryManager:
    """Execute recovery strategies based on fault classification."""

    STRATEGIES = {
        (FaultSeverity.WARNING, ComponentType.SENSOR): RecoveryAction.IGNORE,
        (FaultSeverity.MINOR, ComponentType.SENSOR): RecoveryAction.RESTART,
        (FaultSeverity.MAJOR, ComponentType.SENSOR): RecoveryAction.SWITCH_BACKUP,
        (FaultSeverity.CRITICAL, ComponentType.SENSOR): RecoveryAction.REDUCE_LOAD,
        (FaultSeverity.WARNING, ComponentType.ACTUATOR): RecoveryAction.RESET,
        (FaultSeverity.MINOR, ComponentType.ACTUATOR): RecoveryAction.RESET,
        (FaultSeverity.MAJOR, ComponentType.ACTUATOR): RecoveryAction.SAFE_MODE,
        (FaultSeverity.CRITICAL, ComponentType.ACTUATOR): RecoveryAction.SAFE_MODE,
        (FaultSeverity.MINOR, ComponentType.COMMUNICATION): RecoveryAction.RESTART,
        (FaultSeverity.MAJOR, ComponentType.COMMUNICATION): RecoveryAction.SWITCH_BACKUP,
        (FaultSeverity.CRITICAL, ComponentType.COMMUNICATION): RecoveryAction.ESCALATE,
        (FaultSeverity.WARNING, ComponentType.POWER): RecoveryAction.REDUCE_LOAD,
        (FaultSeverity.MAJOR, ComponentType.POWER): RecoveryAction.SAFE_MODE,
        (FaultSeverity.CRITICAL, ComponentType.POWER): RecoveryAction.SAFE_MODE,
    }

    def __init__(self, monitor: HealthMonitor):
        self.monitor = monitor

    def determine_action(self, event: FaultEvent) -> RecoveryAction:
        comp = self.monitor.components.get(event.component_id)
        if not comp:
            return RecoveryAction.ESCALATE
        action = self.STRATEGIES.get((event.severity, comp.comp_type))
        if action is None:
            return RecoveryAction.ESCALATE
        return action

    def execute_recovery(self, event: FaultEvent) -> Dict:
        action = self.determine_action(event)
        event.recovery_action = action
        comp = self.monitor.components.get(event.component_id)

        result = {"action": action.name, "component": event.component_id,
                 "severity": event.severity.name}

        if action == RecoveryAction.RESTART:
            if comp:
                comp.consecutive_errors = 0
                comp.health = min(1.0, comp.health + 0.3)
                comp.recovery_attempts += 1
            result["success"] = True

        elif action == RecoveryAction.SWITCH_BACKUP:
            if comp and comp.has_backup:
                comp.is_primary = False
                comp.health = 0.0
                # Activate backup
                backup_id = comp.component_id + "_backup"
                backup = self.monitor.components.get(backup_id)
                if backup:
                    backup.is_primary = True
                    backup.health = 1.0
                result["success"] = True
                result["backup"] = backup_id
            else:
                result["success"] = False
                result["reason"] = "No backup available"

        elif action == RecoveryAction.REDUCE_LOAD:
            if comp:
                comp.health = max(0.3, comp.health)
                comp.recovery_attempts += 1
            result["success"] = True

        elif action == RecoveryAction.SAFE_MODE:
            if comp:
                comp.health = 0.3
            result["success"] = True
            result["mode"] = "safe"

        else:
            result["success"] = action == RecoveryAction.IGNORE

        return result


class GracefulDegradation:
    """Manage system behavior under degraded conditions."""

    def __init__(self):
        self.degradation_level = 0  # 0=full, 1=reduced, 2=minimal, 3=safe

    def assess(self, components: Dict[str, ComponentHealth]) -> int:
        total_health = sum(c.health for c in components.values())
        avg_health = total_health / len(components) if components else 0

        critical_count = sum(1 for c in components.values() if c.health < 0.3)
        nav_alive = any(c.comp_type == ComponentType.NAVIGATION and c.health > 0.5
                       for c in components.values())
        comms_alive = any(c.comp_type == ComponentType.COMMUNICATION and c.health > 0.5
                        for c in components.values())

        if critical_count >= 3 or avg_health < 0.2:
            self.degradation_level = 3  # safe mode
        elif critical_count >= 2 or not nav_alive:
            self.degradation_level = 2  # minimal
        elif critical_count >= 1 or not comms_alive:
            self.degradation_level = 1  # reduced
        else:
            self.degradation_level = 0  # full

        return self.degradation_level

    def get_capabilities(self) -> Dict[str, bool]:
        levels = {
            0: {"navigation": True, "comms": True, "sensors": True,
                "actuators": True, "computation": True},
            1: {"navigation": True, "comms": True, "sensors": True,
                "actuators": True, "computation": True},
            2: {"navigation": True, "comms": True, "sensors": False,
                "actuators": False, "computation": True},
            3: {"navigation": False, "comms": True, "sensors": False,
                "actuators": False, "computation": True},
        }
        return levels.get(self.degradation_level, levels[0])


def demo():
    print("=== Self-Healing System ===\n")
    random.seed(42)

    monitor = HealthMonitor()
    recovery = RecoveryManager(monitor)
    degradation = GracefulDegradation()

    # Register components
    components = [
        ComponentHealth("sonar_front", ComponentType.SENSOR, has_backup=True),
        ComponentHealth("sonar_front_backup", ComponentType.SENSOR, has_backup=True, is_primary=False),
        ComponentHealth("thruster_port", ComponentType.ACTUATOR),
        ComponentHealth("thruster_starboard", ComponentType.ACTUATOR),
        ComponentHealth("gps", ComponentType.NAVIGATION),
        ComponentHealth("imu", ComponentType.NAVIGATION, has_backup=True),
        ComponentHealth("radio", ComponentType.COMMUNICATION, has_backup=True),
        ComponentHealth("radio_backup", ComponentType.COMMUNICATION, has_backup=True, is_primary=False),
        ComponentHealth("battery", ComponentType.POWER),
        ComponentHealth("jetson_compute", ComponentType.COMPUTATION),
    ]
    for c in components:
        monitor.register(c)

    # Normal operation
    for c in components:
        monitor.heartbeat(c.component_id)
    level = degradation.assess(monitor.components)
    caps = degradation.get_capabilities()
    print(f"--- Normal State: degradation={level}, capabilities={caps}")

    # Sensor failure scenario
    print("\n--- Sensor Failure Scenario ---")
    for i in range(6):
        sev = monitor.report_error("sonar_front", f"No ping response (attempt {i+1})")
        print(f"  Error {i+1}: {sev.name}")
    action = recovery.execute_recovery(FaultEvent("sonar_front", FaultSeverity.CRITICAL, "No ping"))
    print(f"  Recovery: {action}")

    # Actuator failure
    print("\n--- Actuator Failure ---")
    monitor.report_error("thruster_port", "Motor stall detected")
    monitor.report_error("thruster_port", "Overcurrent")
    monitor.report_error("thruster_port", "Motor stall detected")
    action = recovery.execute_recovery(FaultEvent("thruster_port", FaultSeverity.MAJOR, "Motor stall"))
    print(f"  Recovery: {action}")

    # Assess degradation
    level = degradation.assess(monitor.components)
    caps = degradation.get_capabilities()
    print(f"\n--- Degradation Assessment ---")
    print(f"  Level: {level} ({['FULL', 'REDUCED', 'MINIMAL', 'SAFE'][level]})")
    print(f"  Capabilities: {caps}")

    # Health summary
    print(f"\n--- Component Health ---")
    for cid, comp in sorted(monitor.components.items(), key=lambda x: x[1].health):
        status = "OK" if comp.health > 0.8 else "WARN" if comp.health > 0.5 else "FAIL"
        print(f"  {cid:25s}: {comp.health:.2f} [{status}] errors={comp.error_count}")


if __name__ == "__main__":
    demo()
