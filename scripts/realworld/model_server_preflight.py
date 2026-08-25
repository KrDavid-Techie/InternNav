"""Fail-fast checks for the real-world InternVLA model server."""

import argparse
import json
from pathlib import Path


GIB = 1024**3
DEPTH_CHECKPOINT = 'depth_anything_v2_metric_hypersim_vits.pth'


def format_gib(size):
    return f'{size / GIB:.1f} GiB'


def inspect_checkpoint(model_path):
    model_path = Path(model_path)
    errors = []

    if not model_path.is_dir():
        raise RuntimeError(f'Model directory does not exist: {model_path}')

    config_path = model_path / 'config.json'
    index_path = model_path / 'model.safetensors.index.json'
    for required_path in (config_path, index_path, model_path / 'preprocessor_config.json'):
        if not required_path.is_file():
            errors.append(f'Missing checkpoint file: {required_path}')

    config = {}
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f'Cannot read {config_path}: {exc}')

    shard_names = set()
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding='utf-8'))
            shard_names = set(index.get('weight_map', {}).values())
            if not shard_names:
                errors.append(f'No weight shards are listed in {index_path}')
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f'Cannot read {index_path}: {exc}')

    weight_bytes = 0
    for shard_name in sorted(shard_names):
        shard_path = model_path / shard_name
        if not shard_path.is_file():
            errors.append(f'Missing model shard: {shard_path}')
            continue
        shard_size = shard_path.stat().st_size
        if shard_size < 1024 * 1024:
            errors.append(f'Model shard is too small and may be an LFS pointer: {shard_path} ({shard_size} bytes)')
        weight_bytes += shard_size

    if 'async' in str(config.get('system1', '')):
        depth_path = model_path.parent / DEPTH_CHECKPOINT
        if not depth_path.is_file():
            errors.append(f'Missing DepthAnything checkpoint: {depth_path}')
        elif depth_path.stat().st_size < 90_000_000:
            errors.append(
                f'DepthAnything checkpoint is incomplete: {depth_path} ({depth_path.stat().st_size} bytes)'
            )

    if errors:
        raise RuntimeError('Checkpoint preflight failed:\n- ' + '\n- '.join(errors))

    return config, weight_bytes


def validate_cuda_capacity(device, weight_bytes):
    if not str(device).startswith('cuda'):
        return None

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f'CUDA device was requested but CUDA is unavailable: {device}')

    try:
        device_index = int(str(device).split(':', 1)[1]) if ':' in str(device) else 0
    except ValueError as exc:
        raise RuntimeError(f'Invalid CUDA device: {device}') from exc

    properties = torch.cuda.get_device_properties(device_index)
    if weight_bytes > properties.total_memory:
        raise RuntimeError(
            'Model weights exceed the CUDA/shared memory capacity before activations are allocated: '
            f'weights={format_gib(weight_bytes)}, device={properties.name}, '
            f'capacity={format_gib(properties.total_memory)}. '
            'Run the model server on a larger NVIDIA GPU computer and keep only the ROS client on this robot.'
        )
    return properties


def run_preflight(model_path, device):
    config, weight_bytes = inspect_checkpoint(model_path)
    properties = validate_cuda_capacity(device, weight_bytes)
    device_summary = 'CPU'
    if properties is not None:
        device_summary = f'{properties.name} ({format_gib(properties.total_memory)})'
    print(
        'Model-server preflight OK:',
        f"system1={config.get('system1')}",
        f'weights={format_gib(weight_bytes)}',
        f'device={device_summary}',
    )
    return config, weight_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_path')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    run_preflight(args.model_path, args.device)


if __name__ == '__main__':
    main()
