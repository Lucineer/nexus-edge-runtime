"""Nexus Edge Navigation — dead reckoning, waypoint following, obstacle avoidance.

2D/3D position estimation, path planning with obstacle avoidance,
and waypoint sequencing with arrival detection.
"""
import math, random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class Pose:
    x: float = 0
    y: float = 0
    z: float = 0
    yaw: float = 0   # degrees
    pitch: float = 0
    roll: float = 0


@dataclass
class Waypoint:
    x: float
    y: float
    z: float = 0
    tolerance: float = 2.0  # arrival radius in meters
    speed_limit: float = 1.5  # m/s
    name: str = ""


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float  # obstacle size
    certainty: float = 1.0


class DeadReckoner:
    """Dead reckoning position estimation from velocity + heading."""

    def __init__(self, initial_pose: Pose = None):
        self.pose = initial_pose or Pose()
        self.history: List[Tuple[float, Pose]] = []  # (timestamp, pose)
        self._drift_factor = 0.01  # drift per second

    def update(self, vx: float, vy: float, vz: float,
               yaw_rate: float, dt: float, timestamp: float = 0) -> Pose:
        """Update position from velocity and yaw rate."""
        yaw_rad = math.radians(self.pose.yaw)
        # Rotate velocity to world frame
        world_vx = vx * math.cos(yaw_rad) - vy * math.sin(yaw_rad)
        world_vy = vx * math.sin(yaw_rad) + vy * math.cos(yaw_rad)

        self.pose.x += world_vx * dt
        self.pose.y += world_vy * dt
        self.pose.z += vz * dt
        self.pose.yaw += math.degrees(yaw_rate) * dt

        # Normalize yaw
        self.pose.yaw = self.pose.yaw % 360
        if self.pose.yaw > 180:
            self.pose.yaw -= 360

        # Apply drift
        drift = self._drift_factor * dt
        self.pose.x += random.gauss(0, drift)
        self.pose.y += random.gauss(0, drift)

        if timestamp:
            self.history.append((timestamp, Pose(
                self.pose.x, self.pose.y, self.pose.z, self.pose.yaw)))

        return self.pose

    def correct(self, x: float, y: float, z: float = None) -> float:
        """Correct position with external fix (GPS, etc). Returns error."""
        dx = x - self.pose.x
        dy = y - self.pose.y
        error = math.sqrt(dx**2 + dy**2)
        self.pose.x = x
        self.pose.y = y
        if z is not None:
            self.pose.z = z
        return error

    def distance_to(self, pose: Pose) -> float:
        return math.sqrt((self.pose.x - pose.x)**2 +
                        (self.pose.y - pose.y)**2 +
                        (self.pose.z - pose.z)**2)


class WaypointFollower:
    """Sequential waypoint navigation with arrival detection."""

    def __init__(self, tolerance: float = 2.0, speed: float = 1.5):
        self.waypoints: List[Waypoint] = []
        self.current_idx = 0
        self.tolerance = tolerance
        self.speed = speed
        self.completed: List[int] = []
        self.total_distance = 0.0
        self.traveled_distance = 0.0
        self._last_pose: Optional[Pose] = None

    def set_waypoints(self, waypoints: List[Waypoint]) -> None:
        self.waypoints = waypoints
        self.current_idx = 0
        self.completed = []
        self.total_distance = 0
        self.traveled_distance = 0
        self._last_pose = None
        for i in range(len(waypoints) - 1):
            a, b = waypoints[i], waypoints[i + 1]
            self.total_distance += math.sqrt((b.x - a.x)**2 + (b.y - a.y)**2)

    def get_target(self) -> Optional[Waypoint]:
        if self.current_idx >= len(self.waypoints):
            return None
        return self.waypoints[self.current_idx]

    def get_desired_heading(self, pose: Pose) -> Optional[float]:
        target = self.get_target()
        if not target:
            return None
        dx = target.x - pose.x
        dy = target.y - pose.y
        return math.degrees(math.atan2(dy, dx))

    def get_desired_speed(self, pose: Pose) -> float:
        target = self.get_target()
        if not target:
            return 0
        dist = math.sqrt((target.x - pose.x)**2 + (target.y - pose.y)**2)
        # Slow down near waypoint
        if dist < self.tolerance * 3:
            return target.speed_limit * max(0.1, dist / (self.tolerance * 3))
        return target.speed_limit

    def check_arrival(self, pose: Pose) -> bool:
        target = self.get_target()
        if not target:
            return False
        dist = math.sqrt((target.x - pose.x)**2 + (target.y - pose.y)**2)
        return dist <= target.tolerance

    def advance(self) -> Optional[Waypoint]:
        if self.current_idx < len(self.waypoints):
            self.completed.append(self.current_idx)
            self.current_idx += 1
            return self.get_target()
        return None

    def update(self, pose: Pose) -> Dict:
        if self._last_pose:
            dx = pose.x - self._last_pose.x
            dy = pose.y - self._last_pose.y
            self.traveled_distance += math.sqrt(dx**2 + dy**2)
        self._last_pose = Pose(pose.x, pose.y, pose.z, pose.yaw)

        arrived = self.check_arrival(pose)
        heading = self.get_desired_heading(pose)
        speed = self.get_desired_speed(pose)
        progress = self.traveled_distance / self.total_distance if self.total_distance > 0 else 0

        return {
            "arrived": arrived,
            "heading": heading,
            "speed": speed,
            "waypoint_idx": self.current_idx,
            "progress": round(progress * 100, 1),
            "remaining": len(self.waypoints) - self.current_idx,
        }


class ObstacleAvoidance:
    """Simple obstacle avoidance using potential fields."""

    def __init__(self, safety_radius: float = 5.0, repulsion_gain: float = 10.0):
        self.safety_radius = safety_radius
        self.repulsion_gain = repulsion_gain
        self.obstacles: List[Obstacle] = []

    def add_obstacle(self, obs: Obstacle) -> None:
        self.obstacles.append(obs)

    def compute_avoidance(self, pose: Pose, target: Waypoint) -> Tuple[float, float]:
        """Compute avoidance velocity adjustment."""
        # Attraction to target
        dx = target.x - pose.x
        dy = target.y - pose.y
        dist_to_target = math.sqrt(dx**2 + dy**2)
        if dist_to_target < 0.01:
            return (0, 0)
        fx = dx / dist_to_target
        fy = dy / dist_to_target

        # Repulsion from obstacles
        for obs in self.obstacles:
            ox = pose.x - obs.x
            oy = pose.y - obs.y
            dist = math.sqrt(ox**2 + oy**2)
            effective_dist = dist - obs.radius

            if effective_dist < self.safety_radius and dist > 0.01:
                strength = self.repulsion_gain * obs.certainty / (effective_dist**2 + 0.1)
                fx += (ox / dist) * strength
                fy += (oy / dist) * strength

        # Normalize
        mag = math.sqrt(fx**2 + fy**2)
        if mag > 0:
            fx /= mag
            fy /= mag

        return (fx, fy)


def demo():
    print("=== Navigation ===\n")
    random.seed(42)

    dr = DeadReckoner(Pose(0, 0, 0, 0))

    # Simulate movement
    print("--- Dead Reckoning (circular path) ---")
    for t in range(0, 20):
        vx = 1.0
        vy = 0.0
        yaw_rate = math.radians(10)  # 10 deg/s turn
        pose = dr.update(vx, vy, 0, yaw_rate, dt=1.0, timestamp=t)
        if t % 5 == 0:
            print(f"  t={t}s: pos=({pose.x:.1f}, {pose.y:.1f}), yaw={pose.yaw:.1f}deg")

    # GPS correction
    error = dr.correct(18.5, 3.2)
    print(f"\n  GPS correction: error={error:.2f}m")
    print(f"  Corrected pos: ({dr.pose.x:.1f}, {dr.pose.y:.1f})")

    # Waypoint following
    print("\n--- Waypoint Following ---")
    wf = WaypointFollower()
    wf.set_waypoints([
        Waypoint(10, 0, name="WP1"),
        Waypoint(10, 10, name="WP2"),
        Waypoint(0, 10, name="WP3"),
        Waypoint(0, 0, name="HOME", tolerance=3.0),
    ])

    pose = Pose(0, 0, 0, 0)
    for step in range(30):
        nav = wf.update(pose)
        heading = nav.get("heading", 0)
        speed = nav.get("speed", 0)
        if heading is not None:
            yaw_rad = math.radians(heading)
            pose.x += speed * math.cos(yaw_rad) * 1.0
            pose.y += speed * math.sin(yaw_rad) * 1.0
            pose.yaw = heading

        if nav["arrived"]:
            wf.advance()
            print(f"  Step {step}: ARRIVED at WP{nav['waypoint_idx']}, "
                  f"remaining={nav['remaining']}, progress={nav['progress']}%")

    # Obstacle avoidance
    print("\n--- Obstacle Avoidance ---")
    oa = ObstacleAvoidance(safety_radius=5, repulsion_gain=10)
    oa.add_obstacle(Obstacle(5, 5, 1.0, 1.0))
    oa.add_obstacle(Obstacle(3, 8, 0.5, 0.7))

    pose = Pose(0, 0, 0, 45)
    target = Waypoint(10, 10)
    fx, fy = oa.compute_avoidance(pose, target)
    print(f"  From (0,0) to (10,10) with obstacles:")
    print(f"  Avoidance vector: ({fx:.3f}, {fy:.3f})")


if __name__ == "__main__":
    demo()
