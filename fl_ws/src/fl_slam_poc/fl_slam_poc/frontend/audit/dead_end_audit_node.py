"""
Dead-end Audit Node (Golden Child).

Intentional "dead end" subscriber node for strict accountability:
- Subscribes to explicitly listed topics that are *not* consumed by GC v2 math.
- Tracks message counts and timestamps.
- Fails fast if required topics do not appear within a configured timeout.

No fallbacks:
- Message types are explicit and must be resolvable at startup.
- No auto-detection of message types.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from rosidl_runtime_py.utilities import get_message


@dataclass(frozen=True)
class TopicSpec:
    topic: str
    msg_type: str


def _parse_topic_spec(spec: str) -> TopicSpec:
    """
    Parse a topic spec string.

    Format: "<topic_name>|<fully_qualified_msg_type>"
    Example: "/camera/imu|sensor_msgs/msg/Imu"
    """
    parts = [p.strip() for p in spec.split("|") if p.strip()]
    if len(parts) != 2:
        raise ValueError(
            "Invalid topic spec. Expected '<topic>|<msg_type>', got: "
            f"{spec!r}"
        )
    return TopicSpec(topic=parts[0], msg_type=parts[1])


def _stamp_sec_from_msg(msg: Any) -> Optional[float]:
    header = getattr(msg, "header", None)
    if header is None:
        return None
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", None)
    nsec = getattr(stamp, "nanosec", None)
    if sec is None or nsec is None:
        return None
    return float(sec) + float(nsec) * 1e-9


class DeadEndAuditNode(Node):
    """Subscribes to explicitly listed topics for accountability only."""

    def __init__(self, parameter_overrides: Optional[Dict[str, Any]] = None) -> None:
        from rclpy.parameter import Parameter

        overrides = None
        if parameter_overrides:
            overrides = [Parameter(k, value=v) for k, v in parameter_overrides.items()]
        super().__init__("gc_dead_end_audit", parameter_overrides=overrides)

        self.declare_parameter("topic_specs", [])
        self.declare_parameter("required_topics", [])
        self.declare_parameter("required_timeout_sec", 10.0)
        self.declare_parameter("status_publish_period_sec", 5.0)
        self.declare_parameter("qos_reliability", "best_effort")
        self.declare_parameter("qos_depth", 10)

        topic_specs_raw = list(self.get_parameter("topic_specs").value)
        required_topics = set(self.get_parameter("required_topics").value)
        required_timeout_sec = float(self.get_parameter("required_timeout_sec").value)
        status_period_sec = float(self.get_parameter("status_publish_period_sec").value)
        qos_reliability = str(self.get_parameter("qos_reliability").value).lower()
        qos_depth = int(self.get_parameter("qos_depth").value)

        if qos_reliability == "reliable":
            reliability = ReliabilityPolicy.RELIABLE
        elif qos_reliability == "best_effort":
            reliability = ReliabilityPolicy.BEST_EFFORT
        else:
            raise ValueError(f"Unsupported qos_reliability: {qos_reliability!r}")

        qos = QoSProfile(
            reliability=reliability,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=qos_depth,
        )

        self._start_time = time.time()
        self._required_timeout_sec = required_timeout_sec
        self._required_topics = required_topics

        self._counts: Dict[str, int] = {}
        self._first_stamp_sec: Dict[str, Optional[float]] = {}
        self._last_stamp_sec: Dict[str, Optional[float]] = {}
        self._specs: List[TopicSpec] = []
        self._unavailable: List[Dict[str, str]] = []

        self._subs = []

        self.pub_status = self.create_publisher(String, "/gc/dead_end_status", 10)

        self._setup_subscriptions(topic_specs_raw, qos)
        self._log_startup_manifest()

        self._status_timer = self.create_timer(status_period_sec, self._publish_status)

    def _setup_subscriptions(self, topic_specs_raw: List[str], qos: QoSProfile) -> None:
        if not topic_specs_raw:
            raise ValueError(
                "DeadEndAuditNode requires non-empty 'topic_specs' parameter."
            )

        for raw in topic_specs_raw:
            spec = _parse_topic_spec(str(raw))
            try:
                msg_cls = get_message(spec.msg_type)
            except Exception as exc:
                # For audit-only topics, we still want accountability even when a
                # driver/interface package is not installed (e.g., livox_ros_driver).
                self._unavailable.append(
                    {"topic": spec.topic, "type": spec.msg_type, "error": f"{type(exc).__name__}: {exc}"}
                )
                self.get_logger().warn(
                    f"Dead-end audit: cannot resolve message type {spec.msg_type} for {spec.topic}: {exc}"
                )
                # Still record this spec so it appears in the manifest/status.
                self._specs.append(spec)
                self._counts[spec.topic] = 0
                self._first_stamp_sec[spec.topic] = None
                self._last_stamp_sec[spec.topic] = None
                continue
            cb = self._make_callback(spec.topic)
            sub = self.create_subscription(msg_cls, spec.topic, cb, qos)
            self._subs.append(sub)
            self._specs.append(spec)
            self._counts[spec.topic] = 0
            self._first_stamp_sec[spec.topic] = None
            self._last_stamp_sec[spec.topic] = None

    def _make_callback(self, topic: str) -> Callable[[Any], None]:
        def _cb(msg: Any) -> None:
            self._counts[topic] = int(self._counts.get(topic, 0)) + 1
            stamp_sec = _stamp_sec_from_msg(msg)
            if self._first_stamp_sec.get(topic) is None:
                self._first_stamp_sec[topic] = stamp_sec
            self._last_stamp_sec[topic] = stamp_sec

        return _cb

    def _log_startup_manifest(self) -> None:
        manifest = {
            "node": str(self.get_name()),
            "purpose": "dead_end_audit",
            "topic_specs": [{"topic": s.topic, "type": s.msg_type} for s in self._specs],
            "unavailable": list(self._unavailable),
            "required_topics": sorted(self._required_topics),
            "required_timeout_sec": float(self._required_timeout_sec),
        }
        self.get_logger().info("=" * 60)
        self.get_logger().info("GC DEAD-END AUDIT MANIFEST")
        self.get_logger().info("=" * 60)
        for spec in self._specs:
            self.get_logger().info(f"  subscribe: {spec.topic} [{spec.msg_type}]")
        if self._unavailable:
            self.get_logger().warn(f"  unavailable_types: {len(self._unavailable)} (see /gc/dead_end_status)")
        if self._required_topics:
            self.get_logger().info(f"  required: {sorted(self._required_topics)}")
        self.get_logger().info("=" * 60)
        self.get_logger().debug(json.dumps(manifest))

    def _publish_status(self) -> None:
        elapsed = time.time() - self._start_time

        missing_required = [
            t for t in sorted(self._required_topics) if int(self._counts.get(t, 0)) <= 0
        ]

        status = {
            "elapsed_sec": float(elapsed),
            "counts": dict(self._counts),
            "first_stamp_sec": dict(self._first_stamp_sec),
            "last_stamp_sec": dict(self._last_stamp_sec),
            "unavailable": list(self._unavailable),
            "missing_required": missing_required,
        }
        msg = String()
        msg.data = json.dumps(status)
        self.pub_status.publish(msg)

        if missing_required and elapsed >= self._required_timeout_sec:
            self.get_logger().error(
                "Dead-end audit missing required topics after "
                f"{self._required_timeout_sec:.3f}s: {missing_required}"
            )
            # Fail fast: stop this node's process so the failure is obvious.
            rclpy.shutdown()
            raise RuntimeError(
                f"Required topics not received within timeout: {missing_required}"
            )


def main() -> None:
    rclpy.init()
    node = DeadEndAuditNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
