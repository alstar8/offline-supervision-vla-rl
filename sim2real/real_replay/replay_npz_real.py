#!/usr/bin/env python3
"""
Replay a ManiSkill-generated trajectory (rl4vla_raw_episode*.npz) on real RC5 + AeroHand.

Adapted from rc5_aerohand_teleop/replay_jsonl_real.py. The hardware control path
(jog P-controller, E-stop, workspace clamps) is intentionally identical; only the
trajectory source differs.

The NPZ stores EE *deltas*, not absolute poses, so the trajectory is anchored at
the robot's pose when replay starts and integrated from there. Deltas are applied
in the frame they were recorded in, with no remapping.

Usage:
    # Dry-run: stats only, no hardware. ALWAYS run this first.
    python replay_npz_real.py ../runs/manual/pick_up_green_cube_ext_20260907_095442 --dry-run

    # Real hardware
    python replay_npz_real.py ../runs/manual/pick_up_green_cube_ext_20260907_095442

    # Half speed
    python replay_npz_real.py <run_dir> --speed 0.5

Ctrl+C at any point -> E-stop (holds robot, opens hand).
"""

import sys
import math
import time
import signal
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

# ── Constants ─────────────────────────────────────────────────────────────────
RC5_IP = '10.10.10.10'

# AeroHand poses. AeroHand.set_joint_positions() takes JOINT ANGLES IN DEGREES —
# it runs the tendon model (joints -> 7 actuators) internally, so nothing here may
# be pre-converted. The 7-value short form is expanded by the SDK as:
#
#   [0] thumb_cmc_abd            [1] thumb_cmc_flex     [2] thumb_mcp = thumb_ip
#   [3] index x3   [4] middle x3   [5] ring x3   [6] pinky x3
#
# Upper limits for those slots, from AeroHandConstants.joint_upper_limits:
HAND_SLOT_NAMES = ('thumb_abd', 'thumb_flex', 'thumb_mcp_ip',
                   'index', 'middle', 'ring', 'pinky')
HAND_SLOT_UPPER = (100.0, 55.0, 90.0, 90.0, 90.0, 90.0, 90.0)

# Hand-tuned presets from rc5_aerohand_teleop.
HAND_OPEN = [40.0, 20.0, 15.0, 30.0, 30.0, 30.0, 30.0]
HAND_CLOSE = [100.0, 55.0, 30.0, 60.0, 60.0, 60.0, 60.0]

# Posture the hand holds while the gripper channel says HOLD, and the one an OPEN
# command returns to. This is the 'side' robot_init_qpos profile — the posture
# reviewed on the rendered hand — with its joint angles converted rad -> deg. That
# profile gives every finger the same angle on all three phalanges, so it fits the
# 7-value short form exactly. The 'top' profile does not (its mcp/pip/dip differ),
# and would need the 16-value form:
#   [44.94, 8.58, 11.22, 20.05, 20.26, 25.15, 25.06, 15.25, 30.14, 20.05,
#    15.25, 30.15, 20.05, 1.81, 40.17, 20.06]
# thumb_abd lowered from the profile's 100 deg (the joint's upper limit) to 70:
# at 100 the thumb sticks out too far on the real hand.
HAND_HOLD = [40.0, 3.5, 14.0, 37.4, 29.4, 30.2, 29.4]

for _name, _v, _hi in zip(HAND_SLOT_NAMES, HAND_HOLD, HAND_SLOT_UPPER):
    if not 0.0 <= _v <= _hi:
        raise ValueError(f'HAND_HOLD[{_name}]={_v} outside the joint limit 0..{_hi}')

# Env control rate: OpenReal2SimEnv reports control_freq = 20 (read from the env on
# 2026-09-15). The earlier 30 was a guess and made replays run 1.5x too fast.
DEFAULT_CONTROL_HZ = 20.0

# Gripper channel is ternary: |v| <= eps -> hold, v > eps -> open, v < -eps -> close.
# Mirrors RCPresetHandController.set_action, and the epsilon must match the sim's
# RCPresetHandControllerConfig.hold_epsilon (1e-4, never overridden in this repo).
# A larger value here would swallow small commands the sim acted on.
GRIPPER_HOLD_EPS = 1e-4

# Jog velocity mode — mirrors teleop_record_webxr_by_controllers.py.
JOG_RATE_HZ       = 100    # Hz — RC5 requires >=100 Hz for stable jog
JOG_SPEED_MAX_POS = 0.20   # m/s   — normalization constant
JOG_SPEED_MAX_ROT = 0.30   # rad/s — normalization constant for rotation
JOG_KP_POS        = 2.0    # P-gain: v(m/s)   = err(m)   x KP
JOG_KP_ROT        = 1.5    # P-gain: w(rad/s) = err(rad) x KP
JOG_SPEED_ACTUAL_POS = 0.050  # m/s — measured actual speed at normalized 1.0

POS_DEDUP_M   = 0.0005   # 0.5 mm
ROT_DEDUP_DEG = 0.05     # 0.05 deg

# Workspace safety limits — clip target poses to the reachable zone.
WORKSPACE_X = (-0.60,  0.10)
WORKSPACE_Y = ( 0.15,  0.90)
WORKSPACE_Z = ( 0.08,  0.90)   # 80 mm — table surface

# ── E-stop ────────────────────────────────────────────────────────────────────
_stop = False


def _sigint_handler(sig, frame):
    global _stop
    _stop = True
    print('\n[E-STOP] Ctrl+C — stopping after current step.')


signal.signal(signal.SIGINT, _sigint_handler)


# ── Jog command ───────────────────────────────────────────────────────────────
def _jog_6dof(robot, var6: np.ndarray) -> None:
    """Send a non-blocking 6-DOF TCP velocity jog command.

    var6 — float[6] normalized to [-1..1]:
      [:3] = XYZ fraction of JOG_SPEED_MAX_POS
      [3:] = RxRyRz fraction of JOG_SPEED_MAX_ROT
    """
    from struct import pack as _struct_pack
    from API.source.models.classes.data_classes.command_templates import JogCommandTemplate as _JogTemplate
    from API.source.models.classes.enum_classes.controller_commands import JogModes as _Jm
    from API.source.models.constants import JOG_CMD_PACK_FORMAT as _JOG_FMT
    from API.source.features.tools import dataclass_to_tuple as _dc2t
    from API.source.models.classes.enum_classes.state_classes import OutComingMotionMode as _Omm
    tpl = _JogTemplate()
    tpl.mode = _Jm.ctrlr_coms_jog_mode_velocity
    tpl.var  = list(np.clip(var6, -1.0, 1.0).astype(float))
    robot.motion.linear._controller.send(_Omm.jog, _struct_pack(_JOG_FMT, *_dc2t(tpl)))


# ── Load NPZ ──────────────────────────────────────────────────────────────────
def _resolve_npz_path(target: Path) -> Path:
    if target.is_file():
        return target
    if not target.is_dir():
        raise FileNotFoundError(f'{target} not found')

    candidates = sorted(target.glob('rl4vla_raw_episode*.npz'))
    if not candidates:
        raise FileNotFoundError(f'No rl4vla_raw_episode*.npz in {target}')
    for c in candidates:
        if 'success' in c.name:
            return c
    return candidates[0]


def _load_npz(npz_path: Path) -> dict:
    archive = np.load(npz_path, allow_pickle=True)
    if 'arr_0' not in archive.files:
        raise ValueError(f'{npz_path}: expected key "arr_0", got {archive.files}')
    payload = archive['arr_0'].item()
    if not isinstance(payload, dict):
        raise ValueError(f'{npz_path}: arr_0 is {type(payload).__name__}, expected dict')

    actions = np.asarray(payload['action'], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise ValueError(f'{npz_path}: action has shape {actions.shape}, expected (N, 7)')
    return payload


def _episode_object_id(payload: dict) -> str:
    instruction = str(payload.get('instruction') or '')
    return instruction.split(':', 1)[1].strip() if ':' in instruction else ''


# ── Build waypoints ───────────────────────────────────────────────────────────
def _build_waypoints(
    actions: np.ndarray,
    start_pos: np.ndarray,
    start_rot: R,
    control_hz: float,
    hand_open: list = None,
    hand_close: list = None,
) -> tuple:
    """Integrate EE deltas into absolute waypoints anchored at (start_pos, start_rot).

    Returns:
        waypoints   — list of {pos, rot (xyz Euler deg), hand, t}
        hand_events — list of (waypoint_index, hand_positions)
    """
    hand_open = list(hand_open if hand_open is not None else HAND_OPEN)
    hand_close = list(hand_close if hand_close is not None else HAND_CLOSE)

    waypoints = []
    hand_events = []

    pos = np.array(start_pos, dtype=float)
    rot = start_rot
    hand = list(hand_open)

    prev_pos = None
    prev_rot_euler = None
    prev_hand = list(hand)

    for step, row in enumerate(actions):
        delta_pos = np.asarray(row[:3], dtype=float)
        delta_rpy = np.asarray(row[3:6], dtype=float)
        gripper = float(row[6])

        pos = pos + delta_pos
        if np.linalg.norm(delta_rpy) > 0.0:
            rot = R.from_rotvec(delta_rpy) * rot

        if gripper > GRIPPER_HOLD_EPS:
            hand = list(hand_open)
        elif gripper < -GRIPPER_HOLD_EPS:
            hand = list(hand_close)

        rot_euler = tuple(rot.as_euler('xyz', degrees=True))
        t = step / control_hz

        hand_changed = hand != prev_hand

        # Deduplicate poses, but never drop a step that changes the hand.
        if prev_pos is not None and not hand_changed:
            dp = float(np.linalg.norm(np.asarray(rot_euler) * 0 + pos - prev_pos))
            dr = max(abs(rot_euler[i] - prev_rot_euler[i]) for i in range(3))
            if dp < POS_DEDUP_M and dr < ROT_DEDUP_DEG:
                continue

        if hand_changed:
            hand_events.append((len(waypoints), list(hand)))

        waypoints.append({'pos': tuple(pos), 'rot': rot_euler, 'hand': list(hand), 't': t})
        prev_pos = pos.copy()
        prev_rot_euler = rot_euler
        prev_hand = list(hand)

    if not waypoints:
        raise ValueError('No waypoints produced from the episode.')
    return waypoints, hand_events


def _lift_overshoot(payload: dict, actions: np.ndarray):
    """Compare the commanded lift against the lift the sim actually achieved.

    Returns (commanded_m, achieved_m, ratio), or None if the episode carries no
    grasp event or no per-step object height.
    """
    info = payload.get('info')
    if not info or not isinstance(info[0], dict) or 'obj_height_above_table' not in info[0]:
        return None

    grasped = np.array([bool(i.get('is_src_obj_grasped')) for i in info])
    if not grasped.any():
        return None
    grasp_step = int(np.argmax(grasped))

    heights = np.array([float(i['obj_height_above_table']) for i in info])
    achieved = float(heights[-1] - heights[grasp_step])
    if achieved <= 1e-4:
        return None

    dz = actions[grasp_step:, 2]
    commanded = float(dz[dz > 0].sum())
    return commanded, achieved, commanded / achieved


def _clip_workspace(pos: tuple) -> tuple:
    p = list(pos)
    changed = False
    for i, (lo, hi) in enumerate([WORKSPACE_X, WORKSPACE_Y, WORKSPACE_Z]):
        if p[i] < lo:
            p[i] = lo
            changed = True
        elif p[i] > hi:
            p[i] = hi
            changed = True
    return tuple(p), changed


# ── Hardware init ─────────────────────────────────────────────────────────────
def _init_rc5():
    sys.path.insert(0, '/home/aermakov/github/ros2_rc5_control_pregrasp/python_api')
    from API.rc_api import RobotApi
    from API.source.models.classes.enum_classes.state_classes import (
        InComingControllerState as Ics, InComingSafetyStatus as Iss
    )
    robot = RobotApi(RC5_IP, show_std_traceback=True)
    if (
        robot.safety_status.get() == Iss.fault.name
        or robot.controller_state.get() == Ics.failure.name
    ):
        robot.controller_state.set('off')
    robot.controller_state.set('run', await_sec=120)
    return robot


# ── Execute (time-based jog) ──────────────────────────────────────────────────
def _execute(robot, hand, waypoints: list, hand_events: list, speed: float) -> None:
    """Replay the trajectory using continuous time-based jog.

    At each 100 Hz tick the target pose is interpolated (lerp pos, slerp rot) from
    the episode timeline scaled by `speed`. A P-controller turns the pose error into
    a jog velocity command. Hand events fire when playback time passes their waypoint.
    """
    import threading
    from scipy.spatial.transform import Slerp

    dt_jog = 1.0 / JOG_RATE_HZ

    t_rec   = np.array([wp['t'] for wp in waypoints], dtype=float)
    pos_arr = np.array([wp['pos'] for wp in waypoints], dtype=float)
    rots    = R.from_euler('xyz', [wp['rot'] for wp in waypoints], degrees=True)
    slerp   = Slerp(t_rec, rots)

    t0_rec   = t_rec[0]
    t1_rec   = t_rec[-1]
    duration = (t1_rec - t0_rec) / speed

    pos_arr[:, 0] = np.clip(pos_arr[:, 0], *WORKSPACE_X)
    pos_arr[:, 1] = np.clip(pos_arr[:, 1], *WORKSPACE_Y)
    pos_arr[:, 2] = np.clip(pos_arr[:, 2], *WORKSPACE_Z)

    hand_queue = sorted(
        [(waypoints[idx]['t'], hand_pos) for idx, hand_pos in hand_events],
        key=lambda x: x[0],
    )
    hand_iter = iter(hand_queue)
    next_hand = next(hand_iter, None)

    vel_pos_max = JOG_SPEED_MAX_POS * speed
    vel_rot_max = JOG_SPEED_MAX_ROT * speed

    print(f'  Waypoints: {len(waypoints)}  |  Hand events: {len(hand_events)}'
          f'  |  Speed: {speed:.1f}x  |  Duration: {duration:.0f} s')
    if not hand_events:
        print('  WARNING: no hand events — hand will not move.')

    _vel      = np.zeros(6)
    _vel_lock = threading.Lock()
    _jog_stop = threading.Event()

    t_wall_start = time.time()
    last_print   = t_wall_start

    def _jog_thread_fn():
        while not _stop and not _jog_stop.is_set():
            with _vel_lock:
                v = _vel.copy()
            _jog_6dof(robot, v)
            time.sleep(dt_jog)
        _jog_6dof(robot, np.zeros(6))

    jog_thread = threading.Thread(target=_jog_thread_fn, daemon=True, name='jog')
    jog_thread.start()

    dt_ctrl = 1.0 / 30.0

    while not _stop:
        t_elapsed = time.time() - t_wall_start
        t_play    = t0_rec + t_elapsed * speed

        while next_hand is not None and next_hand[0] <= t_play:
            try:
                hand.set_joint_positions(next_hand[1])
                print(f'  t+{t_elapsed:.0f}s: hand -> {[round(v, 1) for v in next_hand[1]]}')
            except Exception as e:
                print(f'  t+{t_elapsed:.0f}s: hand ERROR — {e}')
            next_hand = next(hand_iter, None)

        if t_elapsed >= duration:
            break

        t_clamp = float(np.clip(t_play, t0_rec, t1_rec))
        idx     = int(np.clip(np.searchsorted(t_rec, t_clamp, side='right') - 1, 0, len(waypoints) - 2))
        dt_seg  = float(t_rec[idx + 1] - t_rec[idx])
        alpha   = float(np.clip((t_clamp - t_rec[idx]) / dt_seg, 0.0, 1.0)) if dt_seg > 0 else 0.0

        target_pos = pos_arr[idx] + alpha * (pos_arr[idx + 1] - pos_arr[idx])
        target_rot = slerp(t_clamp)

        actual = robot.motion.linear.get_actual_position(orientation_units='deg')
        if actual is None:
            time.sleep(dt_ctrl)
            continue

        actual_pos = np.array(actual[:3])
        actual_rot = R.from_euler('xyz', actual[3:], degrees=True)

        pos_err = target_pos - actual_pos
        rot_err = (target_rot * actual_rot.inv()).as_rotvec()

        pos_vel = np.clip(pos_err * JOG_KP_POS, -vel_pos_max, vel_pos_max)
        rot_vel = np.clip(rot_err * JOG_KP_ROT, -vel_rot_max, vel_rot_max)

        # Feedforward: match target velocity so steady-state tracking error -> 0.
        if dt_seg > 0:
            target_vel_pos = (pos_arr[idx + 1] - pos_arr[idx]) / dt_seg * speed
            ff_pos = np.clip(target_vel_pos / JOG_SPEED_ACTUAL_POS, -1.0, 1.0)
        else:
            ff_pos = np.zeros(3)

        # Workspace velocity clamp: zero any component pushing further out of bounds.
        for j, (lo, hi) in enumerate([WORKSPACE_X, WORKSPACE_Y, WORKSPACE_Z]):
            if actual_pos[j] <= lo and pos_vel[j] < 0:
                pos_vel[j] = 0.0
                ff_pos[j]  = 0.0
            elif actual_pos[j] >= hi and pos_vel[j] > 0:
                pos_vel[j] = 0.0
                ff_pos[j]  = 0.0

        new_vel = np.zeros(6)
        new_vel[:3] = np.clip(pos_vel / JOG_SPEED_MAX_POS + ff_pos, -1.0, 1.0)
        new_vel[3:] = rot_vel / JOG_SPEED_MAX_ROT
        with _vel_lock:
            _vel[:] = new_vel

        if time.time() - last_print >= 5.0:
            pct = t_elapsed / duration * 100
            print(f'  {pct:.0f}%  t+{t_elapsed:.0f}s'
                  f'  target_z={target_pos[2]:.3f}  actual_z={actual_pos[2]:.3f}'
                  f'  err_z={pos_err[2]*100:.1f}cm  ff_z={ff_pos[2]:.2f}')
            last_print = time.time()

        time.sleep(dt_ctrl)

    with _vel_lock:
        _vel[:] = 0.0
    _jog_stop.set()
    jog_thread.join(timeout=0.5)
    _jog_6dof(robot, np.zeros(6))

    actual = robot.motion.linear.get_actual_position(orientation_units='deg')
    if actual and waypoints:
        final = waypoints[-1]['pos']
        dist  = math.sqrt(sum((actual[j] - final[j]) ** 2 for j in range(3)))
        print(f'  Done. Distance to final: {dist * 100:.1f} cm')


# ── Dry-run ───────────────────────────────────────────────────────────────────
def _dry_run(payload: dict, actions: np.ndarray, waypoints: list, hand_events: list,
             control_hz: float,
             anchored: bool = True, hand_open: list = None, hand_close: list = None) -> None:
    result = payload.get('result') or {}
    source = payload.get('source') or {}

    total_dist = 0.0
    n_clips = 0
    for i in range(1, len(waypoints)):
        p0 = np.asarray(waypoints[i - 1]['pos'])
        p1 = np.asarray(waypoints[i]['pos'])
        total_dist += float(np.linalg.norm(p1 - p0))
        _, clipped = _clip_workspace(waypoints[i]['pos'])
        if clipped:
            n_clips += 1

    net_xyz = actions[:, :3].sum(axis=0)

    print(f'  Instruction      : {payload.get("instruction")}')
    print(f'  Outcome          : {result.get("execution_outcome")} '
          f'(semantic_success={result.get("semantic_task_success")})')
    print(f'  Planner backend  : {source.get("planner_backend")} / {source.get("macro_route")}')
    print(f'  Action steps     : {len(actions)}  @ {control_hz:g} Hz -> {len(actions)/control_hz:.1f} s')
    print(f'  Rotation deltas  : '
          f'{"all zero (translation-only episode)" if not np.any(actions[:, 3:6]) else "present"}')
    print(f'  Waypoints (dedup): {len(waypoints)}')
    print(f'  Total EE travel  : {total_dist * 100:.1f} cm')
    print(f'  Workspace clips  : {n_clips if anchored else "n/a (no anchor pose)"}')
    print(f'  Hand events      : {len(hand_events)}'
          + (f' at steps {[i for i, _ in hand_events]}' if hand_events else ''))
    grip = np.asarray(actions[:, 6], dtype=float)
    uniq = np.unique(np.round(grip, 4))
    shown = ', '.join(f'{v:+g}' for v in uniq[:8]) + (' ...' if uniq.size > 8 else '')
    n_hold = int((np.abs(grip) <= GRIPPER_HOLD_EPS).sum())
    n_open = int((grip > GRIPPER_HOLD_EPS).sum())
    n_close = int((grip < -GRIPPER_HOLD_EPS).sum())
    print(f'  Gripper channel  : values [{shown}] -> '
          f'{n_open} open / {n_close} close / {n_hold} hold  (eps={GRIPPER_HOLD_EPS:g})')
    if hand_open is not None and hand_close is not None:
        width = max(len(n) for n in HAND_SLOT_NAMES)
        print('  AeroHand pose    : ' + ' '.join(f'{n:>{width}}' for n in HAND_SLOT_NAMES))
        print('    hold  (deg)    : ' + ' '.join(f'{v:>{width}.1f}' for v in hand_open))
        print('    close (deg)    : ' + ' '.join(f'{v:>{width}.1f}' for v in hand_close))
        print('    hold  = HAND_HOLD;  close = HAND_CLOSE  (joint angles, deg)')
        print('    limits (deg)   : ' + ' '.join(f'{v:>{width}.0f}' for v in HAND_SLOT_UPPER))
        print('    Sent to AeroHand.set_joint_positions(); the SDK runs the tendon')
        print('    model internally, so these are joint angles, not actuations.')
    print(f'  Start pose       : {[round(float(v), 4) for v in list(waypoints[0]["pos"]) + list(waypoints[0]["rot"])]}')
    print(f'  End pose         : {[round(float(v), 4) for v in list(waypoints[-1]["pos"]) + list(waypoints[-1]["rot"])]}')
    print(f'  Net XYZ delta    : {np.round(net_xyz, 4).tolist()} m (raw episode frame)')

    overshoot = _lift_overshoot(payload, actions)
    if overshoot is not None:
        commanded, achieved, ratio = overshoot
        print()
        print(f'  Lift commanded   : {commanded * 100:.1f} cm (sum of +dz after grasp)')
        print(f'  Lift achieved    : {achieved * 100:.1f} cm (object height, from episode info)')
        if ratio > 1.2:
            print(f'  *** WARNING: commanded lift overshoots achieved by {ratio:.1f}x. ***')
            print('  The NPZ stores TARGET deltas for a closed-loop controller: the sim planner')
            print('  kept issuing deltas while the arm lagged, so integrating them open-loop')
            print('  travels much further than the sim actually moved. Expect the same overshoot')
            print('  here. Start with --speed 0.3 and a hand on the E-stop, or export absolute')
            print('  TCP poses from the sim for a faithful replay.')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Replay a ManiSkill NPZ episode on real RC5 + AeroHand')
    parser.add_argument('episode',
                        help='Run directory (containing rl4vla_raw_episode*.npz) or the .npz itself')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Speed multiplier (default: 1.0)')
    parser.add_argument('--control-hz', type=float, default=DEFAULT_CONTROL_HZ,
                        help=f'Sim control rate the episode was recorded at (default: {DEFAULT_CONTROL_HZ:g})')
    parser.add_argument('--start-pose', type=float, nargs=6, default=None,
                        metavar=('X', 'Y', 'Z', 'RX', 'RY', 'RZ'),
                        help='Anchor pose for delta integration (default: robot pose at start; '
                             'required for --dry-run unless --assume-start-pose)')
    parser.add_argument('--dry-run', action='store_true', help='Stats only, no hardware')
    args = parser.parse_args()

    npz_path = _resolve_npz_path(Path(args.episode))
    payload = _load_npz(npz_path)
    actions = np.asarray(payload['action'], dtype=np.float64)

    object_id = _episode_object_id(payload)
    hand_open, hand_close = list(HAND_HOLD), list(HAND_CLOSE)

    print(f'Episode  : {npz_path}')
    print(f'Object   : {object_id or "<unknown>"}')
    print(f'Speed    : {args.speed:.1f}x   Control rate: {args.control_hz:g} Hz')
    print()

    if args.dry_run:
        # Deltas are frame-relative, so any anchor gives identical shape/length stats.
        start_pos = np.array(args.start_pose[:3]) if args.start_pose else np.zeros(3)
        start_rot = R.from_euler('xyz', args.start_pose[3:], degrees=True) if args.start_pose \
            else R.identity()
        waypoints, hand_events = _build_waypoints(
            actions, start_pos, start_rot, args.control_hz, hand_open, hand_close)
        print('[DRY RUN]')
        if not args.start_pose:
            print('  NOTE: no --start-pose given; poses below are relative to origin.')
        _dry_run(payload, actions, waypoints, hand_events,
                 args.control_hz, anchored=bool(args.start_pose),
                 hand_open=hand_open, hand_close=hand_close)
        return

    print('Connecting to RC5...')
    robot = _init_rc5()
    actual = robot.motion.linear.get_actual_position(orientation_units='deg')
    if actual is None:
        raise RuntimeError('RC5 did not report an actual position.')
    print(f'  RC5 OK — cartesian: {actual}')

    if args.start_pose:
        start_pos = np.array(args.start_pose[:3])
        start_rot = R.from_euler('xyz', args.start_pose[3:], degrees=True)
        print(f'  Anchoring at --start-pose: {args.start_pose}')
    else:
        start_pos = np.array(actual[:3])
        start_rot = R.from_euler('xyz', actual[3:], degrees=True)
        print('  Anchoring at the current robot pose.')

    waypoints, hand_events = _build_waypoints(
        actions, start_pos, start_rot, args.control_hz, hand_open, hand_close)

    print('Connecting to AeroHand...')
    from aero_open_sdk.aero_hand import AeroHand
    hand = AeroHand()
    print(f'  AeroHand OK — actuations: {hand.get_actuations()}')
    print()

    _dry_run(payload, actions, waypoints, hand_events,
             args.control_hz, hand_open=hand_open, hand_close=hand_close)
    print()

    try:
        hand.set_joint_positions(waypoints[0]['hand'])

        input('Ready. Press ENTER to start replay...')
        print('Starting replay.')

        _execute(robot, hand, waypoints, hand_events, args.speed)

    finally:
        print('Holding RC5, opening hand.')
        try:
            robot.motion.mode.set('hold')
        except Exception:
            pass
        try:
            hand.set_joint_positions(HAND_OPEN)
            time.sleep(0.5)
            hand.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
