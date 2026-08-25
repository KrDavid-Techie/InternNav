#!/usr/bin/env python3
"""Run the InternVLA-N1 real-world policy with the Dangjin A2 ROS interface.

The A2 control node consumes ``sensor_msgs/msg/Joy`` on ``/a2_control``:

* ``axes[1]`` is forward velocity with an inverted sign.
* ``axes[2]`` is yaw velocity with an inverted sign.

This client converts camera observations and odometry into the existing HTTP
model-server protocol, then converts the returned trajectory/action into Joy
messages.  A zero Joy message is published whenever the observation, model
response, or odometry becomes stale.
"""

import argparse
import copy
import io
import json
import math
import os
import threading
import time
from collections import deque

import numpy as np
import rclpy
import requests
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Joy
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

try:
    from controllers import Mpc_controller, PID_controller
except ModuleNotFoundError as exc:
    if exc.name in {'casadi', 'scipy'}:
        raise SystemExit(
            f"Missing A2 controller dependency: {exc.name}. "
            "Run: python3 -m pip install --user --no-deps -r requirements/a2_client.txt"
        ) from exc
    raise


def stamp_to_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def quaternion_to_yaw(q):
    """Return yaw using all quaternion components."""
    sin_yaw = 2.0 * (q.w * q.z + q.x * q.y)
    cos_yaw = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(sin_yaw, cos_yaw)


def encode_rgb(image):
    output = io.BytesIO()
    PILImage.fromarray(image).save(output, format="JPEG", quality=90)
    return output.getvalue()


def encode_depth(depth_meters, shape):
    """Encode depth in the same 0..10000 meter scale used by the server."""
    if depth_meters is None:
        depth = np.zeros(shape, dtype=np.uint16)
    else:
        depth = np.asarray(depth_meters)
        if depth.shape != shape:
            depth = np.zeros(shape, dtype=np.uint16)
        elif np.issubdtype(depth.dtype, np.integer):
            depth = depth.astype(np.float32) / 1000.0
            depth = np.clip(depth * 10000.0, 0.0, 65535.0).astype(np.uint16)
        else:
            depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            depth = np.clip(depth * 10000.0, 0.0, 65535.0).astype(np.uint16)

    output = io.BytesIO()
    PILImage.fromarray(depth).save(output, format="PNG")
    return output.getvalue()


def joy_axes_from_velocity(vx, yaw):
    """Map physical command values to the A2 Joy convention."""
    vx = float(np.clip(vx, -1.0, 1.0))
    yaw = float(np.clip(yaw, -1.0, 1.0))
    return [0.0, -vx, -yaw]


class A2InternVLANode(Node):
    def __init__(self, args):
        super().__init__("a2_internvla_client")
        self.args = args
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.running = True

        self.rgb_bytes = None
        self.rgb_image = None
        self.rgb_stamp = None
        self.depth_image = None
        self.camera_intrinsic = None
        self.odom = None
        self.odom_history = deque(maxlen=100)
        self.image_seq = 0
        self.last_processed_image_seq = -1
        self.last_image_wall_time = 0.0
        self.last_odom_wall_time = 0.0
        self.last_inference_wall_time = 0.0

        self.mode = "stop"
        self.goal = None
        self.mpc = None
        self.policy_init = True
        self.request_index = 0
        self.last_error_log = 0.0

        self.camera_sub = self.create_subscription(
            Image, args.camera_topic, self.on_image, qos_profile_sensor_data
        )
        self.odom_sub = self.create_subscription(
            Odometry, args.odom_topic, self.on_odom, qos_profile_sensor_data
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, args.camera_info_topic, self.on_camera_info, qos_profile_sensor_data
        )
        self.depth_sub = None
        if args.depth_topic:
            self.depth_sub = self.create_subscription(
                Image, args.depth_topic, self.on_depth, qos_profile_sensor_data
            )

        self.control_pub = self.create_publisher(Joy, args.control_topic, qos_profile_sensor_data)
        self.check_control_publishers()
        self.control_timer = self.create_timer(1.0 / args.control_rate, self.publish_control)
        self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.inference_thread.start()

        self.get_logger().info("Subscribed camera: %s", args.camera_topic)
        self.get_logger().info("Subscribed odometry: %s", args.odom_topic)
        self.get_logger().info("Publishing A2 Joy commands: %s", args.control_topic)
        self.get_logger().info("Model server: %s", args.server_url)
        self.get_logger().info("Instruction: %s", args.instruction)

    def check_control_publishers(self):
        # Give DDS graph discovery a moment to expose existing publishers.
        time.sleep(0.25)
        competitors = sorted(
            {
                f"{info.node_namespace.rstrip('/')}/{info.node_name}"
                for info in self.get_publishers_info_by_topic(self.args.control_topic)
                if info.node_name != self.get_name()
            }
        )
        if not competitors:
            return

        message = (
            f"Competing publisher(s) already exist on {self.args.control_topic}: "
            f"{', '.join(competitors)}. Concurrent velocity commands are unsafe."
        )
        if self.args.allow_competing_control_publisher:
            self.get_logger().warning(message)
            return
        raise RuntimeError(
            message + " Stop the competing publisher or explicitly pass --allow-competing-control-publisher."
        )

    def on_image(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            image = np.ascontiguousarray(image)
            with self.lock:
                self.rgb_image = image
                self.rgb_bytes = encode_rgb(image)
                self.rgb_stamp = stamp_to_seconds(msg.header.stamp)
                self.image_seq += 1
                self.last_image_wall_time = time.monotonic()
        except Exception as exc:  # pragma: no cover - depends on ROS image transport
            self.get_logger().error("Failed to decode camera image: %s", exc)

    def on_depth(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            with self.lock:
                self.depth_image = np.asarray(depth).copy()
        except Exception as exc:  # pragma: no cover - optional hardware path
            self.get_logger().warning("Failed to decode depth image: %s", exc)

    def on_camera_info(self, msg):
        if len(msg.k) != 9:
            return
        with self.lock:
            intrinsic = np.eye(4, dtype=np.float32)
            intrinsic[:3, :3] = np.asarray(msg.k, dtype=np.float32).reshape(3, 3)
            self.camera_intrinsic = intrinsic.tolist()

    def on_odom(self, msg):
        try:
            yaw = quaternion_to_yaw(msg.pose.pose.orientation)
            value = {
                "pose": [
                    float(msg.pose.pose.position.x),
                    float(msg.pose.pose.position.y),
                    float(yaw),
                ],
                "linear_x": float(msg.twist.twist.linear.x),
                "angular_z": float(msg.twist.twist.angular.z),
                "stamp": stamp_to_seconds(msg.header.stamp),
            }
            with self.lock:
                self.odom = value
                self.odom_history.append((value["stamp"], value["pose"]))
                self.last_odom_wall_time = time.monotonic()
        except Exception as exc:  # pragma: no cover - depends on ROS message data
            self.get_logger().error("Failed to read odometry: %s", exc)

    def nearest_odom(self, image_stamp):
        with self.lock:
            history = list(self.odom_history)
        if not history:
            return None
        return min(history, key=lambda item: abs(item[0] - image_stamp))[1]

    def make_request(self):
        with self.lock:
            if self.rgb_bytes is None or self.rgb_image is None or self.odom is None:
                return None
            image_bytes = self.rgb_bytes
            image = self.rgb_image.copy()
            depth = None if self.depth_image is None else self.depth_image.copy()
            image_stamp = self.rgb_stamp
            odom = copy.deepcopy(self.odom)
            intrinsic = copy.deepcopy(self.camera_intrinsic)

        odom_for_image = self.nearest_odom(image_stamp)
        if odom_for_image is None:
            odom_for_image = odom["pose"]

        payload = {
            "reset": self.policy_init,
            "idx": self.request_index,
            "instruction": self.args.instruction,
            "camera_intrinsic": intrinsic,
            "odom": odom_for_image,
        }
        return image_bytes, encode_depth(depth, image.shape[:2]), payload, odom_for_image

    def request_model(self, packet):
        image_bytes, depth_bytes, payload, odom = packet
        response = requests.post(
            self.args.server_url,
            files={
                "image": ("rgb_image", image_bytes, "image/jpeg"),
                "depth": ("depth_image", depth_bytes, "image/png"),
            },
            data={"json": json.dumps(payload)},
            timeout=self.args.request_timeout,
        )
        response.raise_for_status()
        return response.json(), odom

    @staticmethod
    def pose_matrix(pose):
        x, y, yaw = pose
        return np.array(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0, x],
                [math.sin(yaw), math.cos(yaw), 0.0, y],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def set_discrete_action(self, actions, odom):
        actions = [int(action) for action in actions]
        if not actions or 0 in actions:
            with self.lock:
                self.mode = "stop"
                self.goal = None
                self.mpc = None
            return

        actionable = [action for action in actions if action in (1, 2, 3)]
        if not actionable:
            with self.lock:
                self.mode = "stop"
            return

        goal = self.pose_matrix(odom)
        for action in actionable:
            if action == 1:
                yaw = math.atan2(goal[1, 0], goal[0, 0])
                goal[0, 3] += self.args.discrete_forward_distance * math.cos(yaw)
                goal[1, 3] += self.args.discrete_forward_distance * math.sin(yaw)
            elif action == 2:
                angle = math.radians(self.args.discrete_turn_degrees)
                rotation = np.array(
                    [[math.cos(angle), -math.sin(angle), 0.0],
                     [math.sin(angle), math.cos(angle), 0.0],
                     [0.0, 0.0, 1.0]]
                )
                goal[:3, :3] = rotation @ goal[:3, :3]
            elif action == 3:
                angle = -math.radians(self.args.discrete_turn_degrees)
                rotation = np.array(
                    [[math.cos(angle), -math.sin(angle), 0.0],
                     [math.sin(angle), math.cos(angle), 0.0],
                     [0.0, 0.0, 1.0]]
                )
                goal[:3, :3] = rotation @ goal[:3, :3]

        with self.lock:
            self.goal = goal
            self.mpc = None
            self.mode = "pid"

    def set_trajectory(self, trajectory, odom):
        points = []
        transform = self.pose_matrix(odom)
        for point in trajectory:
            if len(point) < 2:
                continue
            local_point = np.array([float(point[0]), float(point[1]), 0.0, 1.0])
            points.append((transform @ local_point)[:2])
        if len(points) < 2:
            with self.lock:
                self.mode = "stop"
                self.mpc = None
            return

        reference = np.asarray(points, dtype=np.float64)
        try:
            with self.lock:
                if self.mpc is None:
                    self.mpc = Mpc_controller(reference, v_max=self.args.max_linear_velocity,
                                               w_max=self.args.max_angular_velocity)
                else:
                    self.mpc.update_ref_traj(reference)
                self.mode = "mpc"
        except Exception as exc:
            self.get_logger().error("Failed to create MPC reference: %s", exc)
            with self.lock:
                self.mode = "stop"
                self.mpc = None

    def apply_response(self, response, odom):
        if "trajectory" in response:
            self.set_trajectory(response["trajectory"], odom)
        elif "discrete_action" in response:
            self.set_discrete_action(response["discrete_action"], odom)
        else:
            self.get_logger().warning("Model response contained no trajectory or discrete_action")
            with self.lock:
                self.mode = "stop"

    def inference_loop(self):
        while self.running and rclpy.ok():
            packet = None
            with self.lock:
                if self.image_seq != self.last_processed_image_seq:
                    packet = self.make_request()
                    if packet is not None:
                        self.last_processed_image_seq = self.image_seq

            if packet is None:
                time.sleep(0.02)
                continue

            try:
                response, odom = self.request_model(packet)
                with self.lock:
                    self.policy_init = False
                    self.request_index += 1
                    self.last_inference_wall_time = time.monotonic()
                self.apply_response(response, odom)
            except Exception as exc:
                now = time.monotonic()
                if now - self.last_error_log > 2.0:
                    self.get_logger().error("Model request failed; publishing stop: %s", exc)
                    self.last_error_log = now
                with self.lock:
                    self.mode = "stop"
                    self.mpc = None
            time.sleep(self.args.inference_period)

    def current_velocity(self):
        now = time.monotonic()
        with self.lock:
            stale = (
                self.last_image_wall_time == 0.0
                or self.last_odom_wall_time == 0.0
                or self.last_inference_wall_time == 0.0
                or now - self.last_image_wall_time > self.args.stale_timeout
                or now - self.last_odom_wall_time > self.args.stale_timeout
                or now - self.last_inference_wall_time > self.args.stale_timeout
            )
            mode = self.mode
            odom = None if self.odom is None else self.odom["pose"][:]
            goal = None if self.goal is None else self.goal.copy()
            mpc = self.mpc

        if stale or odom is None:
            return 0.0, 0.0

        try:
            if mode == "mpc" and mpc is not None:
                with self.lock:
                    controls, _ = mpc.solve(np.asarray(odom, dtype=np.float64))
                return float(controls[0, 0]), float(controls[0, 1])
            if mode == "pid" and goal is not None:
                current = self.pose_matrix(odom)
                velocity = [0.0, 0.0]
                with self.lock:
                    if self.odom is not None:
                        velocity = [self.odom["linear_x"], self.odom["angular_z"]]
                linear, angular, _, _ = self.pid.solve(current, goal, velocity)
                return max(0.0, float(linear)), float(angular)
        except Exception as exc:
            self.get_logger().error("Controller failed; publishing stop: %s", exc)
            with self.lock:
                self.mode = "stop"
                self.mpc = None
        return 0.0, 0.0

    def publish_control(self):
        linear, angular = self.current_velocity()
        message = Joy()
        message.header.stamp = self.get_clock().now().to_msg()
        message.axes = joy_axes_from_velocity(linear, angular)
        message.buttons = [0] * 10
        self.control_pub.publish(message)

    def stop(self):
        self.running = False
        for _ in range(3):
            message = Joy()
            message.header.stamp = self.get_clock().now().to_msg()
            message.axes = [0.0, 0.0, 0.0]
            message.buttons = [0] * 10
            self.control_pub.publish(message)
            time.sleep(0.02)

    @property
    def pid(self):
        if not hasattr(self, "_pid"):
            self._pid = PID_controller(
                Kp_trans=2.0,
                Kd_trans=0.0,
                Kp_yaw=1.5,
                Kd_yaw=0.0,
                max_v=self.args.max_linear_velocity,
                max_w=self.args.max_angular_velocity,
            )
        return self._pid


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instruction", "--prompt", dest="instruction",
        default=os.environ.get("INTERNVLA_INSTRUCTION", ""),
        help="Natural-language navigation instruction. English is recommended for the released checkpoint.",
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:5801/eval_dual")
    parser.add_argument("--camera-topic", default="/a2/front_camera/image_raw")
    parser.add_argument("--camera-info-topic", default="/a2/front_camera/camera_info")
    parser.add_argument("--depth-topic", default="", help="Optional depth topic; empty means invalid/zero depth.")
    parser.add_argument("--odom-topic", default="/grit_slam/odometry")
    parser.add_argument("--control-topic", default="/a2_control")
    parser.add_argument("--control-rate", type=float, default=10.0)
    parser.add_argument("--inference-period", type=float, default=0.05)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--stale-timeout", type=float, default=2.0)
    parser.add_argument("--max-linear-velocity", type=float, default=0.35)
    parser.add_argument("--max-angular-velocity", type=float, default=0.40)
    parser.add_argument("--discrete-forward-distance", type=float, default=0.25)
    parser.add_argument("--discrete-turn-degrees", type=float, default=15.0)
    parser.add_argument(
        "--allow-competing-control-publisher",
        action="store_true",
        help="Allow another node to publish to the control topic (unsafe unless arbitration is configured).",
    )
    args = parser.parse_args()
    if not args.instruction.strip():
        parser.error("--instruction is required (or set INTERNVLA_INSTRUCTION)")
    return args


def main():
    args = parse_args()
    rclpy.init()
    node = None
    try:
        node = A2InternVLANode(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
