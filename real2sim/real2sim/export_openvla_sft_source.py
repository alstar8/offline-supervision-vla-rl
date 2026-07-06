import argparse
import json
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
WRIST_CAMERA_NAME = 'wrist_camera'
WRIST_INSET_TOP = 4
WRIST_INSET_LEFT = 4
WRIST_INSET_MARGIN = 4
WRIST_INSET_BORDER = 4
WRIST_INSET_HEIGHT = 224
WRIST_INSET_WIDTH = 168
MODEL_DB_PATH = (
    REPO_ROOT / 'ManiSkill' / 'mani_skill' / 'assets' / 'carrot' / 'more_carrot' / 'model_db.json'
)
MODEL_ID_KEY_PATHS = (
    ('reset_kwargs', 'options', 'probe_source_model_name'),
)
INSTRUCTION_KEY_PATHS = (
    ('reset_kwargs', 'options', 'trajectory_instruction'),
    ('reset_kwargs', 'options', 'language_instruction'),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert a VR demo session into per-episode NPZ files for the OpenVLA SFT dataset builder.'
    )
    parser.add_argument(
        '--session-dir',
        type=Path,
        required=True,
        help='Directory containing vr_human_demos.h5 and vr_human_demos.json.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Directory where converted *.npz episodes will be written.',
    )
    parser.add_argument(
        '--target-total-demos',
        type=int,
        default=100,
        help='Replicate episodes until this many NPZ files exist. Must be >= number of source episodes.',
    )
    parser.add_argument(
        '--trajectory-stem',
        type=str,
        default='vr_human_demos',
        help='Trajectory file stem inside --session-dir. Defaults to vr_human_demos.',
    )
    parser.add_argument(
        '--no-wrist-image',
        action='store_true',
        help='Use only the scene camera image instead of compositing the wrist camera inset.',
    )
    return parser.parse_args()


def _load_meta(session_dir: Path, trajectory_stem: str) -> dict:
    meta_path = session_dir / f'{trajectory_stem}.json'
    if not meta_path.exists():
        raise FileNotFoundError(f'Missing metadata JSON: {meta_path}')
    return json.loads(meta_path.read_text(encoding='utf-8'))


def _load_model_db() -> dict[str, dict]:
    if not MODEL_DB_PATH.exists():
        raise FileNotFoundError(f'Missing model DB JSON: {MODEL_DB_PATH}')

    model_db = json.loads(MODEL_DB_PATH.read_text(encoding='utf-8'))
    if not isinstance(model_db, dict) or not model_db:
        raise ValueError(f'Invalid or empty model DB JSON: {MODEL_DB_PATH}')
    return model_db


def _resize_nearest(images: np.ndarray, height: int, width: int) -> np.ndarray:
    if images.shape[1] == height and images.shape[2] == width:
        return images
    y_idx = np.linspace(0, images.shape[1] - 1, height).round().astype(np.int64)
    x_idx = np.linspace(0, images.shape[2] - 1, width).round().astype(np.int64)
    return images[:, y_idx][:, :, x_idx]


def _compose_wrist_inset(
    scene_rgb: np.ndarray,
    wrist_rgb: np.ndarray | None,
    *,
    bottom_right: bool = False,
) -> np.ndarray:
    scene_rgb = np.asarray(scene_rgb, dtype=np.uint8)
    if wrist_rgb is None:
        return scene_rgb

    wrist_rgb = _resize_nearest(np.asarray(wrist_rgb, dtype=np.uint8), WRIST_INSET_HEIGHT, WRIST_INSET_WIDTH)
    border = WRIST_INSET_BORDER
    inset_h = WRIST_INSET_HEIGHT
    inset_w = WRIST_INSET_WIDTH
    if bottom_right:
        margin = WRIST_INSET_MARGIN
        top = scene_rgb.shape[1] - inset_h - 2 * border - margin
        left = scene_rgb.shape[2] - inset_w - 2 * border - margin
    else:
        top = WRIST_INSET_TOP
        left = WRIST_INSET_LEFT

    out = scene_rgb.copy()
    out[:, top:top + inset_h + 2 * border, left:left + inset_w + 2 * border, :] = 0
    out[
        :,
        top + border:top + border + inset_h,
        left + border:left + border + inset_w,
        :,
    ] = wrist_rgb
    return out


def _validate_episode(
    traj_group: h5py.Group,
    episode_id: int,
    *,
    use_wrist_image: bool,
    wrist_inset_bottom_right: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = np.asarray(traj_group['actions'], dtype=np.float32)
    success = np.asarray(traj_group['success'], dtype=bool)
    obs = traj_group['obs']
    images = None
    wrist_rgb = None
    if 'sensor_data' in obs:
        sensor_data = obs['sensor_data']
        if '3rd_view_camera' in sensor_data:
            rgb = np.asarray(sensor_data['3rd_view_camera']['rgb'], dtype=np.uint8)
        elif 'openvla_image' in obs:
            rgb = np.asarray(obs['openvla_image'], dtype=np.uint8)
        else:
            raise ValueError(f'traj_{episode_id} has no 3rd_view_camera RGB or openvla_image.')
        if WRIST_CAMERA_NAME in sensor_data and 'rgb' in sensor_data[WRIST_CAMERA_NAME]:
            wrist_rgb = np.asarray(sensor_data[WRIST_CAMERA_NAME]['rgb'], dtype=np.uint8)
    elif 'openvla_image' in obs:
        rgb = np.asarray(obs['openvla_image'], dtype=np.uint8)
    else:
        raise ValueError(f'traj_{episode_id} has no sensor_data or openvla_image observations.')

    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f'traj_{episode_id} has invalid action shape {actions.shape}, expected [T, 7].')
    if success.shape[0] != actions.shape[0]:
        raise ValueError(
            f'traj_{episode_id} has success length {success.shape[0]} but action length {actions.shape[0]}.'
        )
    if rgb.shape[0] != actions.shape[0] + 1:
        raise ValueError(
            f'traj_{episode_id} has rgb length {rgb.shape[0]} but expected actions+1 = {actions.shape[0] + 1}.'
        )
    if rgb.shape[1:] != (480, 640, 3):
        raise ValueError(f'traj_{episode_id} has unexpected RGB shape {rgb.shape[1:]}, expected (480, 640, 3).')
    if wrist_rgb is not None and wrist_rgb.shape[0] != rgb.shape[0]:
        raise ValueError(
            f'traj_{episode_id} has wrist RGB length {wrist_rgb.shape[0]} but scene RGB length {rgb.shape[0]}.'
        )

    if use_wrist_image and wrist_rgb is not None:
        images = _compose_wrist_inset(
            rgb[:-1],
            wrist_rgb[:-1],
            bottom_right=wrist_inset_bottom_right,
        )
    elif images is None:
        images = rgb[:-1]
    return actions, success, images


def _is_airi_cubes_session(meta: dict) -> bool:
    task_mode = str(meta.get('task_mode', '')).strip().lower()
    scene_asset_dir = str(meta.get('scene_asset_dir', '')).replace('\\', '/').lower()
    env_info = meta.get('env_info', {}) if isinstance(meta.get('env_info', {}), dict) else {}
    env_id = str(env_info.get('env_id', '')).strip()
    env_kwargs = env_info.get('env_kwargs', {}) if isinstance(env_info.get('env_kwargs', {}), dict) else {}
    env_scene_asset_dir = str(env_kwargs.get('scene_asset_dir', '')).replace('\\', '/').lower()
    source_traj_path = str(meta.get('source_traj_path', '')).replace('\\', '/').lower()
    source_json_path = str(meta.get('source_json_path', '')).replace('\\', '/').lower()
    if task_mode in {'airi_cube_pickup', 'cube_pickup', 'pick_cube'}:
        return True
    if env_id in {'PutObjectOnPlateAiriCubesRecorder-v1', 'PickUpAiriCubeRecorder-v1'}:
        return True
    return (
        'airi_cubes' in scene_asset_dir
        or 'airi_cubes' in env_scene_asset_dir
        or 'airi_cubes' in source_traj_path
        or 'airi_cubes' in source_json_path
    )


def _get_nested_str(data: dict, path: tuple[str, ...]) -> str | None:
    value: object = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]

    if not isinstance(value, str):
        return None

    stripped = value.strip()
    if not stripped:
        return None
    return stripped


def _extract_model_id(episode: dict, episode_id: int) -> str:
    for path in MODEL_ID_KEY_PATHS:
        model_id = _get_nested_str(episode, path)
        if model_id is not None:
            return model_id

    supported_keys = ', '.join('.'.join(path) for path in MODEL_ID_KEY_PATHS)
    raise ValueError(
        f'No probe source model name found for episode_id={episode_id}. '
        f'Checked keys: {supported_keys}.'
    )


def _extract_instruction(episode: dict) -> str | None:
    for path in INSTRUCTION_KEY_PATHS:
        instruction = _get_nested_str(episode, path)
        if instruction is not None:
            return instruction
    return None


def _fallback_object_name(model_id: str) -> str:
    parts = model_id.split('_')
    if len(parts) >= 3 and parts[0].isdigit() and parts[-1].isdigit():
        object_name = ' '.join(parts[1:-1]).strip()
        if object_name:
            return object_name

    raise ValueError(
        f'Could not infer object name from model id {model_id!r}. '
        f'Add it to {MODEL_DB_PATH.name} or adjust the parser.'
    )


def _lookup_object_name(model_id: str, model_db: dict[str, dict]) -> str:
    entry = model_db.get(model_id)
    if isinstance(entry, dict):
        name = entry.get('name')
        if isinstance(name, str) and name.strip():
            return name.strip()

    return _fallback_object_name(model_id)


def _infer_instruction(
    episode: dict,
    episode_id: int,
    model_db: dict[str, dict],
) -> tuple[str, str, str, str]:
    recorded_instruction = _extract_instruction(episode)
    if recorded_instruction is not None:
        return 'recorded_task', 'recorded task', recorded_instruction, 'recorded'

    model_id = _extract_model_id(episode, episode_id)
    object_name = _lookup_object_name(model_id, model_db)

    fallback_instruction = f'Put the {object_name} on the plate'
    print(
        'WARNING: '
        f'episode_id={episode_id} has no explicit recorded instruction; '
        f'falling back to probe object name from {model_id!r}.'
    )
    return model_id, object_name, fallback_instruction, 'fallback_probe_object'


def _build_payload(
    *,
    instruction: str,
    actions: np.ndarray,
    success: np.ndarray,
    images: np.ndarray,
    source_episode_id: int,
    replica_index: int,
    action_filter_applied: bool,
) -> dict:
    info = [{'success': bool(v)} for v in success.tolist()]
    return {
        'instruction': instruction,
        'action': actions,
        'image': images,
        'info': np.asarray(info, dtype=object),
        'source_episode_id': int(source_episode_id),
        'replica_index': int(replica_index),
        'action_filter_applied': bool(action_filter_applied),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta = _load_meta(args.session_dir, args.trajectory_stem)
    model_db = _load_model_db()
    use_wrist_image = not bool(args.no_wrist_image)
    wrist_inset_bottom_right = _is_airi_cubes_session(meta)
    traj_path = args.session_dir / f'{args.trajectory_stem}.h5'
    if not traj_path.exists():
        raise FileNotFoundError(f'Missing trajectory H5: {traj_path}')

    episodes = list(meta['episodes'])
    if not episodes:
        raise ValueError('No episodes found in vr_human_demos.json.')
    if args.target_total_demos < len(episodes):
        raise ValueError(
            f'target_total_demos={args.target_total_demos} is smaller than source episode count {len(episodes)}.'
        )

    manifest = {
        'session_dir': str(args.session_dir.resolve()),
        'model_db_path': str(MODEL_DB_PATH.resolve()),
        'source_episode_count': len(episodes),
        'target_total_demos': args.target_total_demos,
        'use_wrist_image': use_wrist_image,
        'wrist_inset_bottom_right': wrist_inset_bottom_right,
        'airi_cubes_action_norm_clamp': {'enabled': False},
        'episodes': [],
    }

    with h5py.File(traj_path, 'r') as h5_file:
        for out_index in range(args.target_total_demos):
            source_index = out_index % len(episodes)
            episode = episodes[source_index]
            episode_id = int(episode['episode_id'])
            replica_index = out_index // len(episodes)
            model_id, object_name, instruction, instruction_source = _infer_instruction(
                episode,
                episode_id,
                model_db,
            )
            traj_group = h5_file[f'traj_{episode_id}']
            actions, success, images = _validate_episode(
                traj_group,
                episode_id,
                use_wrist_image=use_wrist_image,
                wrist_inset_bottom_right=wrist_inset_bottom_right,
            )
            action_norm_clamp = {'enabled': False}
            action_filter_applied = bool(
                episode.get('reset_kwargs', {})
                .get('options', {})
                .get('real2sim_action_filter_applied', False)
            )

            entry = {
                'source_index': source_index,
                'episode_id': episode_id,
                'probe_source_model_name': model_id,
                'object_name': object_name,
                'instruction': instruction,
                'instruction_source': instruction_source,
                'action_filter_applied': action_filter_applied,
                'num_steps': int(actions.shape[0]),
                'rgb_frames': int(images.shape[0]),
                'terminal_success': bool(success[-1]),
                'meta_success': bool(episode.get('success', False)),
                'output_file': f'episode_{out_index:04d}.npz',
                'replica_index': int(replica_index),
                'action_norm_clamp': action_norm_clamp,
            }
            manifest['episodes'].append(entry)
            payload = _build_payload(
                instruction=instruction,
                actions=actions,
                success=success,
                images=images,
                source_episode_id=episode_id,
                replica_index=replica_index,
                action_filter_applied=action_filter_applied,
            )
            out_path = args.output_dir / f'episode_{out_index:04d}.npz'
            np.savez_compressed(out_path, payload)
            if (out_index + 1) % 25 == 0 or out_index + 1 == args.target_total_demos:
                print(f'  exported {out_index + 1}/{args.target_total_demos} from {traj_path.name}', flush=True)

    (args.output_dir / 'conversion_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(
        f'Wrote {args.target_total_demos} demos to {args.output_dir} '
        f'from {len(episodes)} source episodes in {args.session_dir}'
    )


if __name__ == '__main__':
    main()
