"""Nexus Edge Fleet Coordination — task management, delegation, rendezvous.

Multi-agent task assignment with trust-aware delegation,
rendezvous planning, and task state machines.
"""
import math, random, time
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set


class TaskState(IntEnum):
    PENDING = 0
    ASSIGNED = 1
    ACCEPTED = 2
    IN_PROGRESS = 3
    COMPLETED = 4
    FAILED = 5
    CANCELLED = 6


class TaskPriority(IntEnum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Task:
    task_id: str
    description: str
    required_capability: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    state: TaskState = TaskState.PENDING
    assignee: str = ""
    creator: str = ""
    position: Tuple[float, float] = (0, 0)
    payload: float = 0.0
    deadline_ms: float = 0
    max_agents: int = 1
    assigned_agents: List[str] = field(default_factory=list)
    result: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class AgentCapability:
    agent_id: str
    capabilities: Dict[str, float] = field(default_factory=dict)  # cap -> proficiency (0-1)
    position: Tuple[float, float] = (0, 0)
    current_load: int = 0
    max_load: int = 3
    trust_score: float = 0.5


class TaskManager:
    """Fleet task assignment and management."""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.agents: Dict[str, AgentCapability] = {}
        self.task_counter = 0

    def register_agent(self, agent: AgentCapability) -> None:
        self.agents[agent.agent_id] = agent

    def create_task(self, description: str, capability: str = "",
                    priority: TaskPriority = TaskPriority.NORMAL,
                    position: Tuple[float, float] = (0, 0),
                    creator: str = "fleet") -> Task:
        self.task_counter += 1
        task_id = f"task_{self.task_counter:04d}"
        now = time.time()
        task = Task(task_id, description, capability, priority,
                   position=position, creator=creator,
                   created_at=now, updated_at=now)
        self.tasks[task_id] = task
        return task

    def assign_task(self, task_id: str, agent_id: str) -> bool:
        task = self.tasks.get(task_id)
        agent = self.agents.get(agent_id)
        if not task or not agent:
            return False
        if task.state != TaskState.PENDING and task.state != TaskState.ASSIGNED:
            return False
        if agent.current_load >= agent.max_load:
            return False
        if len(task.assigned_agents) >= task.max_agents:
            return False

        # Capability check
        if task.required_capability:
            proficiency = agent.capabilities.get(task.required_capability, 0)
            if proficiency < 0.3:
                return False

        if agent_id not in task.assigned_agents:
            task.assigned_agents.append(agent_id)
            agent.current_load += 1
        task.assignee = agent_id  # primary assignee
        task.state = TaskState.ASSIGNED
        task.updated_at = time.time()
        return True

    def auto_assign(self) -> List[str]:
        """Auto-assign pending tasks to best available agents."""
        assigned = []
        for task in self.tasks.values():
            if task.state != TaskState.PENDING:
                continue

            best_agent = None
            best_score = -1

            for agent in self.agents.values():
                if agent.current_load >= agent.max_load:
                    continue

                score = agent.trust_score * 0.4

                if task.required_capability:
                    proficiency = agent.capabilities.get(task.required_capability, 0)
                    if proficiency < 0.3:
                        continue
                    score += proficiency * 0.4
                else:
                    score += 0.4

                # Distance penalty
                dist = math.sqrt((agent.position[0] - task.position[0])**2 +
                                (agent.position[1] - task.position[1])**2)
                dist_score = max(0, 1.0 - dist / 1000.0)  # 1km max
                score += dist_score * 0.2

                if score > best_score:
                    best_score = score
                    best_agent = agent

            if best_agent:
                if self.assign_task(task.task_id, best_agent.agent_id):
                    assigned.append(task.task_id)

        return assigned

    def complete_task(self, task_id: str, result: str = "",
                      success: bool = True) -> bool:
        task = self.tasks.get(task_id)
        if not task:
            return False
        task.state = TaskState.COMPLETED if success else TaskState.FAILED
        task.result = result
        task.updated_at = time.time()
        for aid in task.assigned_agents:
            agent = self.agents.get(aid)
            if agent:
                agent.current_load = max(0, agent.current_load - 1)
        return True

    def delegate_task(self, task_id: str, from_agent: str,
                      to_agent: str, trust_score: float) -> Tuple[bool, str]:
        """Delegate a task between agents with trust check."""
        if trust_score < 0.6:
            return False, f"Trust {trust_score:.2f} below delegation threshold 0.6"
        task = self.tasks.get(task_id)
        if not task:
            return False, "Task not found"
        if from_agent not in task.assigned_agents:
            return False, f"{from_agent} not assigned to task"
        target = self.agents.get(to_agent)
        if not target:
            return False, f"Agent {to_agent} not found"
        if target.current_load >= target.max_load:
            return False, f"{to_agent} at max load"

        task.assigned_agents.append(to_agent)
        target.current_load += 1
        task.updated_at = time.time()
        return True, f"Task {task_id} delegated to {to_agent}"


class RendezvousPlanner:
    """Plan multi-agent rendezvous points."""

    def plan_rendezvous(self, agents: Dict[str, Tuple[float, float]],
                       weights: Optional[Dict[str, float]] = None) -> Tuple[float, float]:
        """Weighted centroid rendezvous point."""
        if not agents:
            return (0, 0)
        total_weight = 0
        wx, wy = 0, 0
        for aid, (x, y) in agents.items():
            w = weights.get(aid, 1.0) if weights else 1.0
            wx += x * w
            wy += y * w
            total_weight += w
        if total_weight == 0:
            total_weight = 1
        return (wx / total_weight, wy / total_weight)

    def plan_formation(self, agents: Dict[str, Tuple[float, float]],
                       shape: str = "line", spacing: float = 10.0,
                       heading: float = 0) -> Dict[str, Tuple[float, float]]:
        """Calculate formation positions for agents."""
        positions = {}
        agent_list = list(agents.keys())
        n = len(agent_list)

        if shape == "line":
            for i, aid in enumerate(agent_list):
                offset = (i - (n - 1) / 2) * spacing
                cx, cy = agents[aid]
                dx = offset * math.cos(math.radians(heading))
                dy = offset * math.sin(math.radians(heading))
                positions[aid] = (cx + dx, cy + dy)

        elif shape == "v":
            for i, aid in enumerate(agent_list):
                side = 1 if i % 2 == 0 else -1
                rank = (i // 2) + 1
                cx, cy = agents[aid]
                fx = rank * spacing * math.cos(math.radians(heading))
                fy = rank * spacing * math.sin(math.radians(heading))
                lx = side * rank * spacing * math.cos(math.radians(heading + 90))
                ly = side * rank * spacing * math.sin(math.radians(heading + 90))
                positions[aid] = (cx + fx + lx, cy + fy + ly)

        elif shape == "circle":
            radius = spacing * n / (2 * math.pi)
            for i, aid in enumerate(agent_list):
                angle = 2 * math.pi * i / n + math.radians(heading)
                cx, cy = agents[aid]
                positions[aid] = (cx + radius * math.cos(angle),
                                 cy + radius * math.sin(angle))

        return positions


def demo():
    print("=== Fleet Coordination ===\n")
    random.seed(42)

    tm = TaskManager()

    # Register agents
    agents = [
        AgentCapability("auv_alpha", {"navigation": 0.9, "survey": 0.8, "rescue": 0.7},
                       (100, 200), trust_score=0.85),
        AgentCapability("auv_beta", {"navigation": 0.7, "survey": 0.95, "rescue": 0.5},
                       (150, 100), trust_score=0.75),
        AgentCapability("auv_gamma", {"navigation": 0.8, "rescue": 0.9, "sample": 0.85},
                       (200, 150), trust_score=0.90),
        AgentCapability("surface_relay", {"communication": 0.99, "navigation": 0.6},
                       (0, 0), trust_score=0.95, max_load=10),
    ]
    for a in agents:
        tm.register_agent(a)

    # Create tasks
    tasks = [
        tm.create_task("Survey area A", "survey", TaskPriority.HIGH, (120, 180)),
        tm.create_task("Rescue at waypoint 3", "rescue", TaskPriority.CRITICAL, (180, 120)),
        tm.create_task("Collect water sample B", "sample", TaskPriority.NORMAL, (200, 200)),
        tm.create_task("Navigate to station", "navigation", TaskPriority.LOW, (100, 100)),
    ]

    # Auto-assign
    print("--- Auto Assignment ---")
    assigned = tm.auto_assign()
    for tid in assigned:
        t = tm.tasks[tid]
        print(f"  {tid}: {t.description} → {t.assignee} (agents: {t.assigned_agents})")

    # Manual delegation
    print("\n--- Delegation ---")
    ok, msg = tm.delegate_task("task_0001", "auv_beta", "auv_gamma", 0.85)
    print(f"  Delegate task_0001 beta→gamma: {'OK' if ok else 'DENY'} — {msg}")
    ok, msg = tm.delegate_task("task_0002", "auv_gamma", "auv_beta", 0.5)
    print(f"  Delegate task_0002 gamma→beta (trust=0.5): {'OK' if ok else 'DENY'} — {msg}")

    # Complete tasks
    print("\n--- Task Completion ---")
    tm.complete_task("task_0001", "Survey complete, 12 readings", True)
    tm.complete_task("task_0003", "Sample collected", True)
    for tid, task in tm.tasks.items():
        print(f"  {tid}: {task.state.name} — {task.description}")

    # Rendezvous
    print("\n--- Rendezvous Planning ---")
    rp = RendezvousPlanner()
    agent_pos = {a.agent_id: a.position for a in agents[:3]}
    rv = rp.plan_rendezvous(agent_pos)
    print(f"  Rendezvous point: ({rv[0]:.1f}, {rv[1]:.1f})")

    # Formation
    print("\n--- Formation Planning ---")
    for shape in ["line", "v", "circle"]:
        positions = rp.plan_formation(agent_pos, shape, spacing=20, heading=45)
        print(f"  {shape}: {', '.join(f'{k}=({v[0]:.0f},{v[1]:.0f})' for k,v in positions.items())}")

    # Summary
    print(f"\n--- Summary ---")
    print(f"  Tasks: {len(tm.tasks)} ({sum(1 for t in tm.tasks.values() if t.state == TaskState.COMPLETED)} done)")
    print(f"  Agents: {len(tm.agents)}")
    avg_load = sum(a.current_load for a in tm.agents.values()) / len(tm.agents)
    print(f"  Avg agent load: {avg_load:.1f}/{agents[0].max_load}")


if __name__ == "__main__":
    demo()
