import argparse
import json
import os
import threading
import time

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent
from model_server_preflight import run_preflight

app = Flask(__name__)
idx = 0
start_time = time.time()
runtime_args = None
agent = None
inference_lock = threading.Lock()
DEFAULT_INSTRUCTION = (
    "Turn around and walk out of this office. Turn towards your slight right at the chair. "
    "Move forward to the walkway and go near the red bin. You can see an open door on your right side, "
    "go inside the open door. Stop at the computer monitor"
)


@app.route("/health", methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': agent is not None})


@app.route("/eval_dual", methods=['POST'])
def eval_dual():
    global idx, start_time
    start_time = time.time()

    if agent is None:
        return jsonify({'error': 'Model is not ready'}), 503
    if not inference_lock.acquire(blocking=False):
        return jsonify({'error': 'Inference is already in progress'}), 429, {'Retry-After': '1'}

    try:
        if not {'image', 'depth'} <= set(request.files) or 'json' not in request.form:
            return jsonify({'error': 'image, depth, and json fields are required'}), 400
        try:
            data = json.loads(request.form['json'])
        except (TypeError, json.JSONDecodeError) as exc:
            return jsonify({'error': f'Invalid JSON payload: {exc}'}), 400

        image_file = request.files['image']
        depth_file = request.files['depth']

        try:
            image = np.asarray(Image.open(image_file.stream).convert('RGB'))
            depth = np.asarray(Image.open(depth_file.stream).convert('I')).astype(np.float32) / 10000.0
        except (OSError, ValueError) as exc:
            return jsonify({'error': f'Invalid image payload: {exc}'}), 400

        print(f"read http data cost {time.time() - start_time}")

        camera_pose = np.eye(4)
        instruction = str(data.get('instruction') or runtime_args.instruction).strip()
        if not instruction:
            return jsonify({'error': 'No navigation instruction was provided'}), 400

        camera_intrinsic = data.get('camera_intrinsic')
        if camera_intrinsic is not None:
            camera_intrinsic = np.asarray(camera_intrinsic, dtype=np.float32)
            if camera_intrinsic.shape not in ((3, 3), (4, 4)):
                camera_intrinsic = None
        if camera_intrinsic is None:
            camera_intrinsic = runtime_args.camera_intrinsic
        elif camera_intrinsic.shape == (3, 3):
            expanded = np.eye(4, dtype=np.float32)
            expanded[:3, :3] = camera_intrinsic
            camera_intrinsic = expanded
        if bool(data.get('reset', False)):
            idx = 0
            print("init reset model!!!")
            agent.reset()

        idx += 1

        look_down = False
        t0 = time.time()

        dual_sys_output = agent.step(
            image, depth, camera_pose, instruction, intrinsic=camera_intrinsic, look_down=look_down
        )
        if dual_sys_output.output_action is not None and dual_sys_output.output_action == [5]:
            dual_sys_output = agent.step(
                image, depth, camera_pose, instruction, intrinsic=camera_intrinsic, look_down=True
            )

        json_output = {}
        if dual_sys_output.output_action is not None:
            json_output['discrete_action'] = dual_sys_output.output_action
        elif dual_sys_output.output_trajectory is not None:
            json_output['trajectory'] = dual_sys_output.output_trajectory.tolist()
            if dual_sys_output.output_pixel is not None:
                json_output['pixel_goal'] = dual_sys_output.output_pixel
        else:
            return jsonify({'error': 'Model produced neither an action nor a trajectory'}), 422

        print(f"dual sys step {time.time() - t0}")
        print(f"json_output {json_output}")
        return jsonify(json_output)
    finally:
        inference_lock.release()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1")
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--plan_step_gap", type=int, default=3)
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=os.environ.get("INTERNVLA_ATTN_IMPLEMENTATION", "flash_attention_2"),
        choices=("flash_attention_2", "sdpa", "eager"),
        help="Transformers attention backend. Use sdpa when FlashAttention is unavailable.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=os.environ.get("INTERNVLA_INSTRUCTION") or DEFAULT_INSTRUCTION,
        help="Fallback instruction used when the client does not send one.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5801)
    parser.add_argument(
        "--debug-output-dir",
        default=os.environ.get("INTERNVLA_DEBUG_OUTPUT_DIR", ""),
        help="Optional directory for per-request debug images and text; disabled by default.",
    )
    args = parser.parse_args()
    runtime_args = args

    args.camera_intrinsic = np.array(
        [[386.5, 0.0, 328.9, 0.0], [0.0, 386.5, 244, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    run_preflight(args.model_path, args.device)
    agent = InternVLAN1AsyncAgent(args)
    agent.step(
        np.zeros((480, 640, 3)),
        np.zeros((480, 640)),
        np.eye(4),
        "hello",
        args.camera_intrinsic,
    )
    agent.reset()

    app.run(host=args.host, port=args.port, threaded=True)
