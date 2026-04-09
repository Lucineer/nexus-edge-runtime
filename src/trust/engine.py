"""INCREMENTS Trust Engine — multi-dimensional trust for autonomous agents.

Dimensions:
  T_history     — EMA of binary outcomes
  T_capability  — capability match score (0-1)
  T_latency     — communication latency factor
  T_consistency — behavioral consistency (1 - CV)

Composite: T(a,b,t) = alpha*T_hist + beta*T_cap + gamma*T_lat + delta*T_con
Autonomy levels: L0 (manual) through L5 (full autonomy)
"""
import math, time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Deque


@dataclass
class TrustDimensions:
    history: float = 0.5
    capability: float = 0.5
    latency: float = 0.5
    consistency: float = 0.5


@dataclass
class TrustWeights:
    alpha: float = 0.35   # history
    beta: float = 0.25    # capability
    gamma: float = 0.20   # latency
    delta: float = 0.20   # consistency


class AutonomyLevel:
    MANUAL = 0           # L0: all automation disabled
    ASSISTED = 1         # L1: human approves every action
    SUPERVISED = 2       # L2: human can veto
    CONDITIONAL = 3      # L3: trust-score-gated
    HIGH = 4             # L4: fleet cooperation enabled
    FULL = 5             # L5: emergency-only human intervention

    NAMES = {
        0: "MANUAL", 1: "ASSISTED", 2: "SUPERVISED",
        3: "CONDITIONAL", 4: "HIGH", 5: "FULL"
    }

    @classmethod
    def from_trust(cls, score: float) -> int:
        if score < 0.2: return cls.MANUAL
        if score < 0.4: return cls.ASSISTED
        if score < 0.6: return cls.SUPERVISED
        if score < 0.8: return cls.CONDITIONAL
        if score < 0.95: return cls.HIGH
        return cls.FULL


@dataclass
class InteractionRecord:
    agent_a: str
    agent_b: str
    success: bool
    latency_ms: float
    timestamp: float = 0.0
    context: str = ""


@dataclass
class TrustRelationship:
    agent_a: str
    agent_b: str
    dimensions: TrustDimensions = field(default_factory=TrustDimensions)
    composite_score: float = 0.5
    interaction_count: int = 0
    autonomy_level: int = 0
    last_interaction: float = 0.0
    records: Deque[InteractionRecord] = field(
        default_factory=lambda: deque(maxlen=200))


class TrustEngine:
    """INCREMENTS multi-dimensional trust engine."""

    def __init__(self, weights: Optional[TrustWeights] = None,
                 decay_rate: float = 0.001):
        self.weights = weights or TrustWeights()
        self.decay = decay_rate
        self.relationships: Dict[Tuple[str, str], TrustRelationship] = {}

    def _key(self, a: str, b: str) -> Tuple[str, str]:
        return (a, b)

    def record_interaction(self, agent_a: str, agent_b: str,
                           success: bool, latency_ms: float = 0.0,
                           capability_score: float = None,
                           context: str = "") -> TrustRelationship:
        key = self._key(agent_a, agent_b)
        if key not in self.relationships:
            self.relationships[key] = TrustRelationship(
                agent_a=agent_a, agent_b=agent_b)

        rel = self.relationships[key]
        rel.interaction_count += 1
        rel.last_interaction = time.time()

        # Record
        rec = InteractionRecord(agent_a, agent_b, success, latency_ms,
                                time.time(), context)
        rel.records.append(rec)

        # History (EMA)
        alpha = 0.3
        if rel.interaction_count == 1:
            rel.dimensions.history = 1.0 if success else 0.0
        else:
            val = 1.0 if success else 0.0
            rel.dimensions.history = (alpha * val +
                (1 - alpha) * rel.dimensions.history)

        # Capability
        if capability_score is not None:
            beta = 0.2
            if rel.interaction_count == 1:
                rel.dimensions.capability = capability_score
            else:
                rel.dimensions.capability = (beta * capability_score +
                    (1 - beta) * rel.dimensions.capability)

        # Latency (inverse: low latency = high trust)
        if latency_ms > 0:
            target_ms = 100.0
            latency_score = max(0, 1.0 - (latency_ms / (target_ms * 10)))
            rel.dimensions.latency = (0.15 * latency_score +
                0.85 * rel.dimensions.latency)

        # Consistency (1 - coefficient of variation of outcomes)
        if len(rel.records) >= 5:
            outcomes = [1.0 if r.success else 0.0 for r in list(rel.records)[-50:]]
            mean = sum(outcomes) / len(outcomes)
            if mean > 0 and mean < 1:
                std = math.sqrt(sum((x - mean) ** 2 for x in outcomes) / len(outcomes))
                cv = std / mean if mean > 0 else 0
                rel.dimensions.consistency = max(0, 1.0 - cv)

        # Composite
        w = self.weights
        d = rel.dimensions
        rel.composite_score = (w.alpha * d.history + w.beta * d.capability +
            w.gamma * d.latency + w.delta * d.consistency)

        # Apply decay (trust decays without interaction)
        age = time.time() - rel.last_interaction if rel.last_interaction > 0 else 0
        if age > 3600:  # 1 hour
            decay_factor = math.exp(-self.decay * age / 3600)
            rel.composite_score *= decay_factor

        # Update autonomy level
        rel.autonomy_level = AutonomyLevel.from_trust(rel.composite_score)

        return rel

    def get_trust(self, agent_a: str, agent_b: str) -> Optional[TrustRelationship]:
        return self.relationships.get(self._key(agent_a, agent_b))

    def get_autonomy_level(self, agent_a: str, agent_b: str) -> int:
        rel = self.get_trust(agent_a, agent_b)
        if rel is None:
            return AutonomyLevel.MANUAL
        return rel.autonomy_level

    def can_delegate(self, agent_a: str, agent_b: str,
                     min_level: int = 3) -> Tuple[bool, str]:
        rel = self.get_trust(agent_a, agent_b)
        if rel is None:
            return False, "No trust history"
        if rel.autonomy_level < min_level:
            return False, (f"Autonomy L{rel.autonomy_level} < L{min_level} required "
                          f"(score={rel.composite_score:.2f})")
        if rel.composite_score < 0.6:
            return False, f"Trust score {rel.composite_score:.2f} below threshold"
        return True, "OK"

    def propagate_trust(self, source: str, target: str,
                        depth: int = 2) -> Dict[str, float]:
        """Transitive trust: A trusts B, B trusts C → A partially trusts C."""
        trust_map: Dict[str, float] = {}

        def _propagate(node: str, current_depth: int, score: float):
            if current_depth > depth:
                return
            for key, rel in self.relationships.items():
                a, b = key
                if a == node:
                    transitive = score * rel.composite_score * 0.5
                    if b not in trust_map or transitive > trust_map[b]:
                        trust_map[b] = transitive
                        _propagate(b, current_depth + 1, transitive)

        rel = self.get_trust(source, target)
        if rel:
            trust_map[target] = rel.composite_score
            _propagate(target, 1, rel.composite_score)
        return trust_map

    def fleet_summary(self) -> Dict:
        if not self.relationships:
            return {"agents": 0, "relationships": 0, "avg_trust": 0}
        agents = set()
        total = 0
        for (a, b), rel in self.relationships.items():
            agents.add(a)
            agents.add(b)
            total += rel.composite_score
        return {
            "agents": len(agents),
            "relationships": len(self.relationships),
            "avg_trust": round(total / len(self.relationships), 3),
            "min_trust": round(min(r.composite_score for r in self.relationships.values()), 3),
            "max_trust": round(max(r.composite_score for r in self.relationships.values()), 3),
        }


def demo():
    print("=== INCREMENTS Trust Engine ===\n")

    engine = TrustEngine()

    # Build trust between agents
    agents = ["auv_alpha", "auv_beta", "auv_gamma", "jetson_ground", "surface_relay"]

    # Simulate interactions
    for i in range(50):
        for a in agents:
            for b in agents:
                if a == b:
                    continue
                # Realistic success rates based on "experience"
                base_rate = {"auv_alpha": 0.9, "auv_beta": 0.7, "auv_gamma": 0.8,
                            "jetson_ground": 0.95, "surface_relay": 0.85}.get(a, 0.75)
                success = (i * 0.01 + base_rate * 0.5) > random.random() * 0.7
                latency = random.gauss(50, 20) if success else random.gauss(200, 80)
                cap = random.gauss(base_rate, 0.1)
                engine.record_interaction(a, b, success, max(1, latency),
                                         min(1, max(0, cap)))

    # Fleet summary
    print("--- Fleet Summary ---")
    summary = engine.fleet_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Specific relationships
    print("\n--- Key Relationships ---")
    for pair in [("auv_alpha", "auv_beta"), ("auv_alpha", "jetson_ground"),
                 ("auv_gamma", "surface_relay")]:
        rel = engine.get_trust(*pair)
        if rel:
            d = rel.dimensions
            level = AutonomyLevel.NAMES[rel.autonomy_level]
            print(f"  {pair[0]} → {pair[1]}:")
            print(f"    Score: {rel.composite_score:.3f} (L{rel.autonomy_level} {level})")
            print(f"    Hist={d.history:.2f} Cap={d.capability:.2f} "
                  f"Lat={d.latency:.2f} Con={d.consistency:.2f}")
            print(f"    Interactions: {rel.interaction_count}")

    # Delegation checks
    print("\n--- Delegation Checks ---")
    for pair in [("auv_alpha", "auv_beta"), ("auv_gamma", "auv_alpha"),
                 ("surface_relay", "auv_beta")]:
        ok, reason = engine.can_delegate(*pair, min_level=3)
        print(f"  {pair[0]} → {pair[1]}: {'ALLOW' if ok else 'DENY'} ({reason})")

    # Transitive trust
    print("\n--- Transitive Trust (auv_alpha, depth=2) ---")
    trust_map = engine.propagate_trust("auv_alpha", "auv_beta", depth=2)
    for agent, score in sorted(trust_map.items(), key=lambda x: -x[1]):
        print(f"  {agent}: {score:.3f}")


import random
random.seed(42)

if __name__ == "__main__":
    demo()
