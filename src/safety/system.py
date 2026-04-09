"""Nexus Edge Safety System — 4-tier defense-in-depth architecture.

Tier 1: HARDWARE — Kill switch, watchdog IC, polyfuses (<1us)
Tier 2: FIRMWARE — ISR guard, safe-state outputs, stack canary (<1ms)
Tier 3: SUPERVISORY — Heartbeat monitoring, state machine, watchdog daemon (<100ms)
Tier 4: APPLICATION — Trust-score-gated autonomy (L0-L5), bytecode validation (<1s)

Autonomy Levels:
  L0 Manual     — all automation disabled
  L1 Assisted   — human approves every action
  L2 Supervised — human can veto
  L3 Conditional— trust-score-gated
  L4 High       — fleet cooperation enabled
  L5 Full       — emergency-only human intervention
"""
import enum, time, math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple


class SafetyTier(enum.IntEnum):
    HARDWARE = 1
    FIRMWARE = 2
    SUPERVISORY = 3
    APPLICATION = 4


class AutonomyLevel(enum.IntEnum):
    MANUAL = 0
    ASSISTED = 1
    SUPERVISED = 2
    CONDITIONAL = 3
    HIGH = 4
    FULL = 5


class SafetyState(enum.IntEnum):
    NORMAL = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3
    SAFE_STATE = 4


@dataclass
class SafetyViolation:
    tier: SafetyTier
    kind: str
    description: str
    timestamp: float = 0.0
    resolved: bool = False


@dataclass
class SafetyConfig:
    heartbeat_timeout_ms: int = 500
    watchdog_timeout_ms: int = 2000
    max_bytecode_size: int = 4096
    max_stack_depth: int = 1024
    max_instructions: int = 100000
    trust_threshold_L3: float = 0.6
    trust_threshold_L4: float = 0.8
    trust_threshold_L5: float = 0.95
    safe_states: Dict[int, float] = field(default_factory=dict)  # pin -> safe value


class HardwareSafety:
    """Tier 1: Hardware safety (simulated)."""

    def __init__(self, config: SafetyConfig):
        self.config = config
        self.kill_switch_active = False
        self.watchdog_triggered = False
        self._watchdog_last_feed = 0.0

    def arm(self) -> None:
        self.kill_switch_active = False
        self.watchdog_triggered = False
        self._watchdog_last_feed = time.time()

    def feed_watchdog(self) -> None:
        self._watchdog_last_feed = time.time()
        self.watchdog_triggered = False

    def check(self) -> Tuple[bool, List[SafetyViolation]]:
        violations = []
        safe = True

        # Watchdog check
        if self._watchdog_last_feed > 0:
            elapsed = (time.time() - self._watchdog_last_feed) * 1000
            if elapsed > self.config.watchdog_timeout_ms:
                violations.append(SafetyViolation(
                    SafetyTier.HARDWARE, "watchdog_timeout",
                    f"Watchdog not fed in {elapsed:.0f}ms"))
                self.watchdog_triggered = True
                safe = False

        return safe, violations


class FirmwareSafety:
    """Tier 2: Firmware-level safety (simulated)."""

    def __init__(self, config: SafetyConfig):
        self.config = config
        self.stack_canary = 0xDEADBEEF
        self.isr_guard_active = False
        self.safe_state_applied = False

    def check_stack(self, stack_depth: int) -> Tuple[bool, Optional[SafetyViolation]]:
        if stack_depth > self.config.max_stack_depth:
            return False, SafetyViolation(
                SafetyTier.FIRMWARE, "stack_overflow",
                f"Stack depth {stack_depth} exceeds max {self.config.max_stack_depth}")
        return True, None

    def apply_safe_state(self, pins: Dict[int, float],
                        safe_states: Dict[int, float]) -> Dict[int, float]:
        result = {}
        for pin in pins:
            if pin in safe_states:
                result[pin] = safe_states[pin]
            else:
                result[pin] = 0.0
        self.safe_state_applied = True
        return result

    def check(self) -> Tuple[bool, List[SafetyViolation]]:
        violations = []
        safe = True
        if self.stack_canary != 0xDEADBEEF:
            violations.append(SafetyViolation(SafetyTier.FIRMWARE, "canary_corrupt",
                "Stack canary corrupted"))
            safe = False
        return safe, violations


class SupervisorySafety:
    """Tier 3: Supervisory heartbeat monitoring."""

    def __init__(self, config: SafetyConfig):
        self.config = config
        self.heartbeats: Dict[str, float] = {}  # node_id -> last heartbeat time
        self.state_machine = SafetyState.NORMAL

    def record_heartbeat(self, node_id: str) -> None:
        self.heartbeats[node_id] = time.time()

    def check(self) -> Tuple[bool, List[SafetyViolation]]:
        violations = []
        safe = True
        now = time.time()

        for node_id, last_hb in self.heartbeats.items():
            elapsed = (now - last_hb) * 1000
            if elapsed > self.config.heartbeat_timeout_ms:
                violations.append(SafetyViolation(
                    SafetyTier.SUPERVISORY, "heartbeat_timeout",
                    f"Node {node_id}: no heartbeat in {elapsed:.0f}ms"))
                safe = False

        if violations:
            self.state_machine = SafetyState.CRITICAL
        elif len(self.heartbeats) == 0:
            self.state_machine = SafetyState.WARNING
        else:
            self.state_machine = SafetyState.NORMAL

        return safe, violations


class ApplicationSafety:
    """Tier 4: Application-level safety, bytecode validation, trust gates."""

    def __init__(self, config: SafetyConfig):
        self.config = config
        self.current_autonomy = AutonomyLevel.MANUAL
        self.violations: List[SafetyViolation] = []

    def validate_bytecode(self, bytecode: bytes) -> Tuple[bool, List[SafetyViolation]]:
        errors = []
        if len(bytecode) > self.config.max_bytecode_size:
            errors.append(SafetyViolation(SafetyTier.APPLICATION, "bytecode_too_large",
                f"Bytecode {len(bytecode)}B exceeds max {self.config.max_bytecode_size}B"))
        if len(bytecode) % 8 != 0:
            errors.append(SafetyViolation(SafetyTier.APPLICATION, "bytecode_misaligned",
                f"Bytecode length {len(bytecode)} not multiple of 8"))
        # Check for valid opcodes
        valid_ops = set(range(0x20))  # 0x00-0x1F
        for i in range(0, min(len(bytecode), 8192), 8):
            op = bytecode[i]
            if op not in valid_ops and op != 0x20 and op != 0x21:
                errors.append(SafetyViolation(SafetyTier.APPLICATION, "invalid_opcode",
                    f"Opcode 0x{op:02X} at instruction {i//8}"))
                if len(errors) > 5:
                    break
        return len(errors) == 0, errors

    def check_trust_gate(self, trust_score: float,
                         requested_level: int) -> Tuple[bool, str]:
        thresholds = {
            1: 0.2, 2: 0.4, 3: self.config.trust_threshold_L3,
            4: self.config.trust_threshold_L4, 5: self.config.trust_threshold_L5,
        }
        required = thresholds.get(requested_level, 1.0)
        if trust_score >= required:
            self.current_autonomy = AutonomyLevel(requested_level)
            return True, f"Granted L{requested_level} (score={trust_score:.2f} >= {required:.2f})"
        return False, f"Denied L{requested_level} (score={trust_score:.2f} < {required:.2f})"

    def check(self) -> Tuple[bool, List[SafetyViolation]]:
        return len(self.violations) == 0, self.violations


class SafetySystem:
    """Integrated 4-tier safety system."""

    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()
        self.tier1 = HardwareSafety(self.config)
        self.tier2 = FirmwareSafety(self.config)
        self.tier3 = SupervisorySafety(self.config)
        self.tier4 = ApplicationSafety(self.config)
        self.state = SafetyState.NORMAL
        self.audit_log: List[SafetyViolation] = []

    def arm(self) -> None:
        self.tier1.arm()
        self.state = SafetyState.NORMAL
        self.audit_log.clear()

    def check_all(self) -> Dict:
        results = {}
        overall_safe = True
        self.audit_log.clear()

        for name, tier in [("HARDWARE", self.tier1), ("FIRMWARE", self.tier2),
                           ("SUPERVISORY", self.tier3), ("APPLICATION", self.tier4)]:
            safe, violations = tier.check()
            results[name] = {"safe": safe, "violations": len(violations)}
            if not safe:
                overall_safe = False
            self.audit_log.extend(violations)

        if not overall_safe:
            critical = any(v.kind == "watchdog_timeout" for v in self.audit_log)
            self.state = SafetyState.EMERGENCY if critical else SafetyState.WARNING
        else:
            self.state = SafetyState.NORMAL

        return {
            "overall_safe": overall_safe,
            "state": self.state.name,
            "tiers": results,
            "total_violations": len(self.audit_log),
        }

    def emergency_stop(self) -> Dict[int, float]:
        self.state = SafetyState.SAFE_STATE
        self.tier1.kill_switch_active = True
        return self.tier2.apply_safe_state(
            {i: 0.0 for i in range(16)}, self.config.safe_states)


def demo():
    print("=== Nexus Edge Safety System ===\n")

    config = SafetyConfig(
        heartbeat_timeout_ms=200,
        watchdog_timeout_ms=1000,
        trust_threshold_L3=0.6,
        trust_threshold_L4=0.8,
        trust_threshold_L5=0.95,
        safe_states={0: 0.0, 1: 0.0, 2: 0.0, 3: 1500.0})  # pin 3 = neutral throttle
    safety = SafetySystem(config)
    safety.arm()

    # Normal check
    print("--- Normal State ---")
    result = safety.check_all()
    print(f"  Safe: {result['overall_safe']}, State: {result['state']}")

    # Bytecode validation
    print("\n--- Bytecode Validation ---")
    valid_bc = bytes([0x01, 0, 0, 0, 0x42, 0, 0, 0]) * 10  # 10 PUSH_I8 instructions
    ok, errs = safety.tier4.validate_bytecode(valid_bc)
    print(f"  Valid bytecode: {ok}, errors: {len(errs)}")

    bad_bc = bytes([0xFF, 0, 0, 0, 0, 0, 0, 0])  # invalid opcode
    ok, errs = safety.tier4.validate_bytecode(bad_bc)
    print(f"  Invalid opcode: {ok}, errors: {[e.description for e in errs]}")

    # Trust gates
    print("\n--- Trust Gates ---")
    for score, level in [(0.3, 2), (0.7, 3), (0.85, 4), (0.96, 5), (0.5, 4)]:
        ok, msg = safety.tier4.check_trust_gate(score, level)
        print(f"  Score={score:.1f} L{level}: {'GRANT' if ok else 'DENY'} — {msg}")

    # Heartbeat timeout
    print("\n--- Heartbeat Timeout ---")
    safety.tier3.record_heartbeat("auv_1")
    import time as _t
    _t.sleep(0.3)  # > 200ms timeout
    result = safety.check_all()
    print(f"  Safe: {result['overall_safe']}, State: {result['state']}")
    for name, tier in result['tiers'].items():
        print(f"  {name}: safe={tier['safe']}, violations={tier['violations']}")

    # Emergency stop
    print("\n--- Emergency Stop ---")
    safe_pins = safety.emergency_stop()
    print(f"  State: {safety.state.name}")
    print(f"  Kill switch: {safety.tier1.kill_switch_active}")
    print(f"  Safe pin states: {safe_pins}")

    # Stack overflow
    print("\n--- Stack Overflow Detection ---")
    ok, err = safety.tier2.check_stack(1025)
    print(f"  Depth 1025: {'OK' if ok else err.description}")
    ok, err = safety.tier2.check_stack(500)
    print(f"  Depth 500: {'OK' if ok else err.description}")


if __name__ == "__main__":
    demo()
