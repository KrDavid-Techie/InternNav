# A2 real-world InternNav client

`a2_internvla_client.py` connects the InternVLA-N1 HTTP server to the Dangjin A2 ROS 2 interface.

Default topics:

| Purpose | Topic | Type |
| --- | --- | --- |
| RGB camera | `/a2/front_camera/res_360p/image_raw` | `sensor_msgs/msg/Image` |
| Camera intrinsics | `/a2/front_camera/res_360p/camera_info` | `sensor_msgs/msg/CameraInfo` |
| Odometry | `/grit_slam/odometry` | `nav_msgs/msg/Odometry` |
| A2 control | `/a2_control` | `sensor_msgs/msg/Joy` |

The A2 control node maps `Joy.axes[1]` to forward velocity and `Joy.axes[2]` to yaw, both with an inverted sign. The client publishes a zero command when the camera, odometry, or model response is stale.

## Start the model server

Run this on the machine with the InternVLA-N1 checkpoint:

```bash
python3 scripts/realworld/http_internvla_server.py \
  --device cuda:0 \
  --model_path checkpoints/InternVLA-N1 \
  --plan_step_gap 3
```

## Start the A2 client

Run this on the robot after sourcing ROS 2 Humble and the robot workspace:

```bash
source /opt/ros/humble/setup.bash
source /ros_ws/install/setup.bash

python3 scripts/realworld/a2_internvla_client.py \
  --server-url http://MODEL_SERVER_IP:5801/eval_dual \
  --instruction "Go forward to the red chair, turn left, and stop near the doorway."
```

The released InternVLA-N1 checkpoint was trained primarily with English instructions, so English prompts are recommended for the first hardware test. If `/grit_slam/odometry` is not publishing, the client stays stopped and logs the missing observation rather than sending motion commands.
