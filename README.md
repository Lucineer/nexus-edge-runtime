# nexus-edge-runtime

Edge runtime for autonomous agents in the Cocapn fleet — generalized beyond maritime robotics for IoT, industrial, aerial, and marine domains.

## Core Modules

### Bytecode VM (`src/core/vm.py`) — 14.5K chars
32-opcode stack-based VM: 8-byte instructions, 32 registers (16 GP + 16 IO-mapped), 64KB memory, 1024-deep stack. Assembler, disassembler, bytecode validator. ESP32-S3 deployment + Jetson supervision.

### INCREMENTS Trust Engine (`src/trust/engine.py`) — 10.4K chars
Multi-dimensional trust: history (EMA), capability, latency, consistency. Composite scoring with configurable weights. Autonomy levels L0-L5. Transitive trust propagation. Trust decay.

### Wire Protocol (`src/wire/protocol.py`) — 5.8K chars
Length-prefixed framed protocol: `[PREAMBLE:2B][SRC:1B][DST:1B][TYPE:1B][SEQ:2B][LEN:2B][PAYLOAD:N][CRC16:2B]`. CRC-16/CCITT. Stream parser with partial buffering. 14 message types.

### Safety System (`src/safety/system.py`) — 11.4K chars
4-tier defense-in-depth:
- **Tier 1 HARDWARE**: Kill switch, watchdog IC (<1us)
- **Tier 2 FIRMWARE**: Stack canary, safe-state outputs (<1ms)
- **Tier 3 SUPERVISORY**: Heartbeat monitoring, state machine (<100ms)
- **Tier 4 APPLICATION**: Bytecode validation, trust-gated autonomy (<1s)

### Intent Compiler (`src/reflex/compiler.py`) — 14.8K chars
NL intent → IR → bytecode pipeline. Parses action + target + value + condition. Supports: maintain, navigate, monitor (conditional), alert.

### Fleet Coordination (`src/fleet/coordination.py`) — 11.4K chars
Task assignment with trust-aware scoring, capability matching, distance-weighted selection. Delegation with trust gates. Rendezvous planning (centroid). Formation planning (line, V, circle).

### Cooperative Perception (`src/perception/fusion.py`) — 9.7K chars
Bayesian sensor fusion with inverse-variance weighting. Trust-weighted consensus. Outlier detection (sigma threshold). Quality scoring. Data sharing with staleness filtering.

### Digital Twin (`src/digital_twin/twin.py`) — 8.8K chars
Real-time state mirroring with configurable history buffer. Forward predictive simulation (kinematics + power model). Anomaly detection with baseline learning (z-score). Battery life estimation.

## Architecture

```
NL Intent → Intent Parser → IR → Bytecode → VM Execution
                                           ↓
         Wire Protocol ←→ Agent Comms ←→ Trust Engine
                                           ↓
              Safety System (4 tiers) ↔ Autonomy Levels (L0-L5)
                                           ↓
        Fleet Coordination ←→ Cooperative Perception ←→ Digital Twin
```

## vs SuperInstance/nexus-runtime

Covers the same architectural patterns as [nexus-runtime](https://github.com/SuperInstance/nexus-runtime) but generalized:
- Domain-agnostic (not just maritime)
- Integrated with Cocapn fleet protocol
- Designed for mask-locked inference chip offload
- Trust-aware autonomy gating throughout

## Next: Additional Modules

- `src/navigation/` — dead reckoning, waypoint following, obstacle avoidance
- `src/mission/` — mission planning, execution monitoring, contingency
- `src/energy/` — power management, solar/recharge, budget allocation
- `src/maintenance/` — predictive maintenance, diagnostic scheduling
- `src/security/` — byzantine fault detection, encryption, auth
- `src/learning/` — reinforcement learning, skill acquisition
- `src/explainability/` — XAI decision logging, audit trail
- `src/sensor/` — sensor health monitoring, calibration, fusion
- `src/comms/` — MQTT bridge, mesh networking, relay routing
- `src/data_pipeline/` — telemetry ingestion, compression, storage
- `src/config/` — runtime configuration, hot-reload, schema validation
- `src/simulation/` — physics simulation, Monte Carlo scenarios
- `src/swarm/` — swarm behaviors, emergence detection, consensus
- `src/autonomy/` — adaptive autonomy, self-healing, reflex override
- `src/hardware/` — 11+ platform profiles (ESP32, Jetson, Pi, etc.)

## License

MIT — DiGennaro et al. (SuperInstance & Lucineer)
