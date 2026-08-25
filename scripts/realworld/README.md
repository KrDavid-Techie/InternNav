# A2 real-world InternNav client

`a2_internvla_client.py` connects the InternVLA-N1 HTTP server to the Dangjin A2 ROS 2 interface.

Default topics:

| Purpose | Topic | Type |
| --- | --- | --- |
| RGB camera | `/a2/front_camera/image_raw` | `sensor_msgs/msg/Image` |
| Camera intrinsics | `/a2/front_camera/camera_info` | `sensor_msgs/msg/CameraInfo` |
| Odometry | `/grit_slam/odometry` | `nav_msgs/msg/Odometry` |
| A2 control | `/a2_control` | `sensor_msgs/msg/Joy` |

The A2 control node maps `Joy.axes[1]` to forward velocity and `Joy.axes[2]` to yaw, both with an inverted sign. The client publishes a zero command when the camera, odometry, or model response is stale.

## Start the model server with Docker Compose

Run this on the GPU machine that has the InternVLA-N1 checkpoint. The Compose service exposes the HTTP API on TCP port `5801`; the ROS client remains outside the container and connects to that port.

Do not run the full model server on the current `dangjin-a2` Jetson. It exposes about 15 GiB of shared CUDA memory, while the BF16 model shards alone occupy about 16.8 GB. The server performs a startup preflight and exits with a clear capacity error before loading when the model weights exceed the CUDA/shared-memory capacity. Run the Compose service on a larger NVIDIA GPU computer and point the robot client to that computer.

### Using the preloaded `cobiz:jetson` image

The Dangjin A2 Jetson image includes CUDA 12.6 but does not include PyTorch or the InternNav Python packages. The Compose build therefore installs the pinned CUDA 12.6/aarch64 PyTorch wheel before installing the model-server requirements. This variant assumes Python 3.10 and JetPack 6.2 compatibility; override `INTERNVLA_BASE_IMAGE` and `TORCH_WHEEL_URL` in `.env` when using a different Jetson software stack.

The checkpoint directory must contain the model directory and the DepthAnything checkpoint used by the asynchronous system-1, for example:

```text
checkpoints/
├── InternVLA-N1-DualVLN/
└── depth_anything_v2_metric_hypersim_vits.pth
```

If the downloaded model has another directory name, set `MODEL_DIR` in `.env` accordingly.

```bash
cp .env.example .env
# Edit CHECKPOINTS_DIR and MODEL_DIR if needed.

# Only needed when the NGC registry asks for credentials.
docker login nvcr.io
docker compose up -d --build
curl http://127.0.0.1:5801/health
docker compose logs -f model-server
```

The first build with `cobiz:jetson` downloads the PyTorch wheel, but does not copy the model into the image. The checkpoint must still exist under the host `checkpoints/` directory before `docker compose up`.

Startup also verifies the model index, every referenced safetensors shard, the DepthAnything checkpoint, CUDA availability, and device capacity. Container restarts and Docker JSON logs are bounded so a persistent startup failure cannot fill the robot disk. Debug image/text capture is disabled by default; set `INTERNVLA_DEBUG_OUTPUT_DIR=/app/output/debug` only when needed.

After the first build, `docker compose up -d` is sufficient. The default `sdpa` attention backend avoids requiring a FlashAttention wheel on Jetson; set `INTERNVLA_ATTN_IMPLEMENTATION=flash_attention_2` and `INSTALL_FLASH_ATTN=1` only when a compatible FlashAttention installation is available. The host must have NVIDIA Container Toolkit configured for GPU access.

The server also accepts a fallback `INTERNVLA_INSTRUCTION`, but the A2 client normally sends the instruction with every episode reset. The `/health` endpoint becomes available after the model has finished loading.

The asynchronous dual-system checkpoint also loads a DepthAnything component and can require more memory than a 16 GB Jetson provides. If the container exits with an out-of-memory error, run this model-server Compose service on a larger NVIDIA GPU host and point the robot client at that host's LAN address.

## Start the model server without Docker

Run this on the machine with the InternVLA-N1 checkpoint:

```bash
python3 scripts/realworld/http_internvla_server.py \
  --device cuda:0 \
  --model_path checkpoints/InternVLA-N1-DualVLN \
  --plan_step_gap 3
```

## Start the A2 client

Run this on the robot after sourcing ROS 2 Humble and the robot workspace:

```bash
source /opt/ros/humble/setup.bash
source /ros_ws/install/setup.bash

# CasADi has a CPython 3.10/aarch64 wheel. --no-deps preserves the system
# NumPy used by cv_bridge.
python3 -m pip install --user --no-deps -r requirements/a2_client.txt

# Confirm that the camera/odometry topics exist and inspect control publishers.
ros2 topic list | grep -E 'front_camera|grit_slam/odometry|a2_control'
ros2 topic info -v /a2_control

python3 scripts/realworld/a2_internvla_client.py \
  --server-url http://MODEL_SERVER_IP:5801/eval_dual \
  --instruction "Move forward a short distance and stop." \
  --max-linear-velocity 0.10 \
  --max-angular-velocity 0.15 \
  --request-timeout 120
```

The client refuses to start when another node is already publishing to `/a2_control`, because interleaved velocity commands are unsafe. Stop the competing publisher first. `--allow-competing-control-publisher` is available only for systems that have an explicit command arbiter.

The released InternVLA-N1 checkpoint was trained primarily with English instructions, so English prompts are recommended for the first hardware test. If `/grit_slam/odometry` is not publishing, the client stays stopped and logs the missing observation rather than sending motion commands.

For the first hardware test, lift the robot or use a clear area with an operator ready to trigger the emergency stop. Before starting the client, verify that the camera and odometry topics are publishing and that `/a2_control` has a subscriber. Pressing `Ctrl+C` makes the client publish three zero-velocity commands before it exits.
