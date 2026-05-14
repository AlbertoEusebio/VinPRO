#!/usr/bin/env python3
"""
VinPRO WP2 → WP3 bridge node.

Subscribes to the RealSense RGB stream, runs the ViNet Stacked Hourglass
Network inference pipeline, applies the pruning policy, and publishes
pixel-space cutting points to the control stack.

SUBSCRIBES
----------
  /camera/color/image_raw          sensor_msgs/Image

PUBLISHES
---------
  /pixel_coordinates               std_msgs/Int32MultiArray
      Flat list [u1, v1, u2, v2, ...] in the *original sensor* resolution.
  /cutting_point_markers           visualization_msgs/MarkerArray   [debug]

COORDINATE SPACES
-----------------
  Model input      : 1024 × 1024  (hard square resize of sensor frame)
  Network output   : 256  × 256   (H/4 × W/4 of input, i.e. DEFAULT_RESIZE)
  Node coordinates : 256  × 256   (output of extract_node_coordinates)
  Published coords : sensor_W × sensor_H

  Scaling:
      u_sensor = u_256 * sensor_W / 256
      v_sensor = v_256 * sensor_H / 256

  NOTE: the square resize distorts the aspect ratio. u_sensor and v_sensor
  are therefore the correct pixel indices into the original depth image even
  when sensor_W != sensor_H.
"""

import sys
import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from visualization_msgs.msg import MarkerArray, Marker
from cv_bridge import CvBridge

import cv2
import numpy as np
import torch

# ── Locate the WP2 package ────────────────────────────────────────────────────
# Expect VINET_PATH env var pointing at the wp2-computer-vision/ directory,
# or fall back to a sibling directory in the same repo.
_VINET_DEFAULT = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..", "..", "wp2-computer-vision",
)
_VINET_PATH = os.environ.get("VINET_PATH", os.path.abspath(_VINET_DEFAULT))
if _VINET_PATH not in sys.path:
    sys.path.insert(0, _VINET_PATH)

from vinet.model import StackedHourglassNetwork                         # noqa: E402
from vinet.inference import (                                            # noqa: E402
    extract_node_coordinates,
    construct_resistivity_graph,
    grapevine_structure_estimation,
    recover_heatmaps_vector_fields,
)
from vinet.config import (                                               # noqa: E402
    NODE_TYPES,
    BRANCH_TYPES,
    POSSIBLE_PARENTS,
    NUM_OUTPUT_CHANNELS,
    DEFAULT_RESIZE,
    DEFAULT_FRONT_CHANNELS,
    DEFAULT_HOURGLASS_CHANNELS,
    DEFAULT_CROP_SIZE,
)

from .pruning_policy import select_cutting_points                        # noqa: E402

# Heatmap output size from WP2 (H/4 × W/4 of the 1024 × 1024 input)
_HM_H, _HM_W = DEFAULT_RESIZE   # (256, 256)


class InferenceNode(Node):
    """ROS 2 node that wraps the WP2 ViNet pipeline."""

    def __init__(self):
        super().__init__("vinpro_inference")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("model_checkpoint", "")
        self.declare_parameter("front_channels", DEFAULT_FRONT_CHANNELS)
        self.declare_parameter("hourglass_channels", DEFAULT_HOURGLASS_CHANNELS)
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("publish_markers", True)

        ckpt = self.get_parameter("model_checkpoint").value
        front_ch = self.get_parameter("front_channels").value
        hg_ch = self.get_parameter("hourglass_channels").value
        device_str = self.get_parameter("device").value
        self._publish_markers = self.get_parameter("publish_markers").value

        if not ckpt:
            self.get_logger().fatal(
                "Parameter 'model_checkpoint' is not set. "
                "Pass it via perception_params.yaml."
            )
            raise RuntimeError("model_checkpoint required")

        # ── Model ─────────────────────────────────────────────────────────────
        self._device = torch.device(
            device_str if torch.cuda.is_available() else "cpu"
        )
        self._model = StackedHourglassNetwork(
            in_channels=3,
            front_channels=front_ch,
            hourglass_channels=hg_ch,
            num_output_channels=NUM_OUTPUT_CHANNELS,
        )
        self._model.load_state_dict(
            torch.load(ckpt, map_location=self._device, weights_only=True)
        )
        self._model.to(self._device).eval()
        self.get_logger().info(
            f"ViNet loaded from {ckpt} on {self._device} "
            f"({sum(p.numel() for p in self._model.parameters()):,} parameters)"
        )

        self._bridge = CvBridge()

        # ── Subscribers ───────────────────────────────────────────────────────
        self._image_sub = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self._image_callback,
            10,
        )

        # ── Publishers ────────────────────────────────────────────────────────
        # Flat [u1, v1, u2, v2, ...] in full sensor resolution.
        self._pixel_pub = self.create_publisher(
            Int32MultiArray, "/pixel_coordinates", 10
        )
        self._marker_pub = self.create_publisher(
            MarkerArray, "/cutting_point_markers_2d", 10
        )

    # ── Image callback ────────────────────────────────────────────────────────

    def _image_callback(self, msg: Image) -> None:
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"cv_bridge conversion failed: {exc}")
            return

        sensor_h, sensor_w = bgr.shape[:2]

        # 1 ─ Preprocess: hard-resize to DEFAULT_CROP_SIZE × DEFAULT_CROP_SIZE
        input_np = cv2.resize(
            bgr, (DEFAULT_CROP_SIZE, DEFAULT_CROP_SIZE), interpolation=cv2.INTER_CUBIC
        )
        # BGR → RGB, HWC → CHW, [0,255] → [0,1]
        tensor = (
            torch.from_numpy(input_np[:, :, ::-1].copy())
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(self._device)
        )

        # 2 ─ Inference (only stage-2 output is used, matching predict.py)
        with torch.no_grad():
            _, output2 = self._model(tensor)

        heatmaps, vector_fields = recover_heatmaps_vector_fields(
            output2[0].cpu(), resize=DEFAULT_RESIZE
        )
        # heatmaps:      (N_bt=5, N_nt=4, 256, 256)
        # vector_fields: (N_bt=5, 2,      256, 256)
        vf_np = vector_fields.numpy()

        # 3 ─ Extract nodes from every (branch_type, node_type) channel
        total_nodes: dict = {}
        for bname, bt in BRANCH_TYPES.items():
            for nname, nt in NODE_TYPES.items():
                hm = heatmaps[bt, nt].numpy()
                coords = extract_node_coordinates(hm)
                total_nodes[(bname, nname)] = coords

        # 4 ─ Build resistivity graph and estimate tree structure
        graph = construct_resistivity_graph(
            total_nodes,
            branch_types=BRANCH_TYPES,
            vector_fields=vf_np,
            possible_parents=POSSIBLE_PARENTS,
        )

        root_coords = total_nodes.get(("mainTrunk", "rootCrown"), [(0, 0)])[0]
        root_node = (root_coords, ("mainTrunk", "rootCrown"))
        tree = grapevine_structure_estimation(graph, root_node)

        # 5 ─ Pruning policy → list of cut dicts with "pixel" in 256×256 space
        cuts = select_cutting_points(tree)

        if not cuts:
            self.get_logger().warn("No cutting points detected in this frame")
            return

        # 6 ─ Scale coordinates from 256×256 heatmap space → sensor resolution
        #     u_sensor = u_256 * sensor_W / _HM_W
        #     v_sensor = v_256 * sensor_H / _HM_H
        scale_u = sensor_w / _HM_W
        scale_v = sensor_h / _HM_H

        flat: list[int] = []
        for cut in cuts:
            u256, v256 = cut["pixel"]
            u = int(round(u256 * scale_u))
            v = int(round(v256 * scale_v))
            # Clamp to sensor bounds
            u = max(0, min(u, sensor_w - 1))
            v = max(0, min(v, sensor_h - 1))
            flat.extend([u, v])

        # 7 ─ Publish pixel coordinates
        pixel_msg = Int32MultiArray()
        pixel_msg.data = flat
        self._pixel_pub.publish(pixel_msg)
        self.get_logger().info(
            f"Published {len(cuts)} cutting point(s): {list(zip(flat[::2], flat[1::2]))}"
        )

        # 8 ─ Optional 2D debug markers (image-space spheres at pixel coords)
        if self._publish_markers:
            self._publish_2d_markers(flat, msg.header)

    def _publish_2d_markers(self, flat: list[int], header) -> None:
        """Publish RViz markers for the cutting points (pixel space, z=0)."""
        markers = MarkerArray()
        for i, (u, v) in enumerate(zip(flat[::2], flat[1::2])):
            m = Marker()
            m.header = header
            m.ns = "cutting_pixels"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(u)
            m.pose.position.y = float(v)
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 10.0  # pixels
            m.color.r = 1.0
            m.color.a = 1.0
            m.lifetime = rclpy.duration.Duration(seconds=5).to_msg()
            markers.markers.append(m)
        self._marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
