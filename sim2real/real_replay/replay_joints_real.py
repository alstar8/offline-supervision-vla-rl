#!/usr/bin/env python3
"""
Replay a simulated grasp on the real RC5 + AeroHand from a joint trajectory, driving the arm
with continuous TCP velocity jog — the control path of replay_npz_real.py.

Input is joint_trajectory.npz from export_sim_joint_trajectory.py. Unlike the NPZ's Cartesian
deltas, recorded joint states carry no controller-vs-URDF frame mismatch and no open-loop
overshoot of target deltas.

Why TCP jog and not joint jog: the RC5 joint jog command carries only a direction per motor
(int8 -1/0/+1 at a fixed jog speed), so it cannot follow a trajectory smoothly. Every joint
state is converted to a TCP pose with the controller's own forward kinematics
(motion.kinematics.get_forward — same frame and tool as get_actual_position), and the TCP is
tracked in time with the P-controller and velocity feedforward of replay_npz_real.py (plus
an angular feedforward, since joint paths also rotate the TCP). The controller resolves the
joints itself, so the actual joints are watched against the recorded ones and the run aborts
if any drifts more than --max-joint-dev.

Timing follows the sim (control_freq states per second) scaled by --speed; the idle settle
at the start of the recording is skipped. Before each hand command the TCP is brought to that
state's pose within --pos-tol / --rot-tol-deg, the hand moves, and playback resumes after
--hand-wait.

Joint mapping (2026-09-15): the scene keys airy_table_scene14sep26_left_image[_metric] put the
sim base at the yaw where every sim joint equals the real RC5 joint (hover check on the real
robot), so q_real = q_sim. Trajectories recorded before that change used a base yaw 85.98 deg
larger and need joint0 + 85.98 deg. The offset is derived from the base yaw stored with the
trajectory (meta robot_base_pose, else the run's runtime_config.yaml); --j0-offset overrides
it. The yaw is known to about a degree (~1 cm at the object) — stop above the object first
(--until-height) and check alignment with locate_object_zed.py before a full grasp.

Each joint's whole trajectory is shifted by a multiple of 360 deg so its first state is the
equivalent angle closest to where the robot actually is. Without this a sim -180.5 deg
against a real +180.0 deg would spin the joint a full turn.

Usage:
    python3 replay_joints_real.py <run>/joint_trajectory.npz --dry-run [--read-robot]
    python3 replay_joints_real.py <run>/joint_trajectory.npz --until-height 0.15             # hover above object
    python3 replay_joints_real.py <run>/joint_trajectory.npz --until-height 0.15 --reverse   # back to start
    python3 replay_joints_real.py <run>/joint_trajectory.npz                                  # full grasp

Ctrl+C at any point -> jog stops, robot hold, hand opens.
"""

import sys
import json
import time
import signal
import argparse
import threading
import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# Hand presets, the RC5 connection helper and the jog command with its gains live in the
# Cartesian replay script; reuse them so both replays drive the robot identically.
_spec = importlib.util.spec_from_file_location("replay_npz_real", HERE / "replay_npz_real.py")
_npz_real = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_npz_real)
HAND_HOLD, HAND_CLOSE, HAND_OPEN = _npz_real.HAND_HOLD, _npz_real.HAND_CLOSE, _npz_real.HAND_OPEN
GRIPPER_HOLD_EPS = _npz_real.GRIPPER_HOLD_EPS
JOG_RATE_HZ = _npz_real.JOG_RATE_HZ
JOG_SPEED_MAX_POS, JOG_SPEED_MAX_ROT = _npz_real.JOG_SPEED_MAX_POS, _npz_real.JOG_SPEED_MAX_ROT
JOG_KP_POS, JOG_KP_ROT = _npz_real.JOG_KP_POS, _npz_real.JOG_KP_ROT
JOG_SPEED_ACTUAL_POS = _npz_real.JOG_SPEED_ACTUAL_POS
WORKSPACE = (_npz_real.WORKSPACE_X, _npz_real.WORKSPACE_Y, _npz_real.WORKSPACE_Z)
WS_LO = np.array([lim[0] for lim in WORKSPACE])
WS_HI = np.array([lim[1] for lim in WORKSPACE])
RC5_API_PATH = '/home/aermakov/github/ros2_rc5_control_pregrasp/python_api'

CONTROL_DT = 1.0 / 30.0       # pose read + velocity update period, as in replay_npz_real.py
IDLE_DEG = 0.05               # joint change below which the recorded start counts as the settle phase

# Controller TCP vs sim URDF (runs/calibration/stage1_ctrl_vs_urdf_fit.json, 3 samples):
# p_ctrl = Rz(-89.94 deg) * FK_urdf(q) * tool. Used only to preview controller-frame targets
# without the robot and to cross-check the controller's own forward kinematics.
CTRL_BASE_YAW_DEG = -89.94
CTRL_TOOL_IN_TCP_M = np.array([-0.0001, -0.0001, -0.0883])
FK_CROSSCHECK_TOL_M = 0.02

_stop = False


def _sigint_handler(sig, frame):
    global _stop
    _stop = True
    print('\n[E-STOP] Ctrl+C — stopping jog, holding robot, opening hand.')


signal.signal(signal.SIGINT, _sigint_handler)   # after importing replay_npz_real, which sets its own


class _Abort(Exception):
    pass


def _parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('trajectory', help='joint_trajectory.npz')
    ap.add_argument('--scene-key', default='airy_table_scene14sep26_left_image_metric',
                    help='scene whose table height is used for the TCP height checks')
    ap.add_argument('--j0-offset', type=float, default=None,
                    help='deg added to joint0 (sim -> real). Default: derived from the base yaw of the scene '
                         'the trajectory was recorded in (0 for scenes already in the real-robot convention)')
    ap.add_argument('--speed', type=float, default=0.3,
                    help='playback speed multiplier on the sim timeline (1.0 = sim speed). The jog reaches only '
                         '~5 cm/s at full command; the report shows the commanded TCP speed')
    ap.add_argument('--pos-tol', type=float, default=0.005,
                    help='TCP position tolerance (m) when arriving before a hand command and at the end')
    ap.add_argument('--rot-tol-deg', type=float, default=2.0, help='TCP orientation tolerance (deg) for those arrivals')
    ap.add_argument('--settle-timeout', type=float, default=4.0,
                    help='seconds allowed for an arrival; if still beyond 3x --pos-tol the run aborts')
    ap.add_argument('--max-track-err', type=float, default=0.05,
                    help='abort if the TCP is further than this from its moving target (m)')
    ap.add_argument('--max-joint-dev', type=float, default=8.0,
                    help='abort if any actual joint leaves the recorded joint path by more than this (deg)')
    ap.add_argument('--until-height', type=float, default=None,
                    help='stop before the grasp once the sim TCP is this high above the table (m)')
    ap.add_argument('--until-step', type=int, default=None, help='stop at this trajectory state index')
    ap.add_argument('--start-tol-deg', type=float, default=3.0, help='max start mismatch without --move-to-start')
    ap.add_argument('--move-to-start', action='store_true',
                    help='joint-move to the first state if the robot is elsewhere (one waypoint move, not jog)')
    ap.add_argument('--start-speed', type=float, default=10.0, help='joint speed for --move-to-start (deg/s)')
    ap.add_argument('--hand-wait', type=float, default=1.5, help='seconds to wait after a hand command')
    ap.add_argument('--read-robot', action='store_true',
                    help='dry-run: read current joints and query controller forward kinematics (no motion)')
    ap.add_argument('--reverse', action='store_true',
                    help='drive back from the stop state to the start along the same path, no hand events; '
                         'pass the same --j0-offset and --until-* as the forward run')
    ap.add_argument('--dry-run', action='store_true')
    return ap.parse_args(argv)


REQUIRED_KEYS = ('qpos', 'action', 'tcp_pose_world', 'meta_json')

# robot_base_pose yaw (deg) at which sim joint angles equal the real RC5 joints, per scene.
# Measured with a hover check on the real robot on 2026-09-15 (real joint0 = sim + 85.98 deg
# at the imported yaw of 50.593 deg). A trajectory recorded at another yaw needs
# joint0 offset = recorded yaw - calibrated yaw.
CALIBRATED_REAL_BASE_YAW_DEG = {
    'airy_table_scene14sep26_left_image': -35.387,
    'airy_table_scene14sep26_left_image_metric': -35.387,
}


def _recorded_base_pose(args, meta):
    """robot_base_pose the trajectory was recorded with (meta, else the run's runtime_config.yaml)."""
    base = meta.get('robot_base_pose')
    if base is None:
        runtime = Path(args.trajectory).with_name('runtime_config.yaml')
        if runtime.is_file():
            import yaml
            # runtime_config.yaml holds every scene of the config; take the recorded key's own pose.
            cfg = yaml.safe_load(runtime.read_text(encoding='utf-8')) or {}
            base = (((cfg.get('local') or {}).get(meta.get('key')) or {}).get('simulation') or {}).get('robot_base_pose')
    return base


def _resolve_j0_offset(args, meta, base):
    """Joint0 offset from the scene yaw the trajectory was recorded with, unless given."""
    import math
    key, yaw = meta.get('key'), None
    if base is not None:
        yaw = math.degrees(2 * math.atan2(float(base[6]), float(base[3])))
    auto = round(yaw - CALIBRATED_REAL_BASE_YAW_DEG[key], 3) if (yaw is not None and key in CALIBRATED_REAL_BASE_YAW_DEG) else None
    if args.j0_offset is None:
        if auto is None:
            raise SystemExit(f'Cannot derive the joint0 offset for key={key!r} (recorded yaw {yaw}); pass --j0-offset.')
        return auto, f'auto: recorded scene yaw {yaw:.3f} - calibrated {CALIBRATED_REAL_BASE_YAW_DEG[key]:.3f}'
    if auto is not None and abs(args.j0_offset - auto) > 0.5:
        print(f'WARNING: --j0-offset {args.j0_offset} differs from the {auto} derived from the recorded scene yaw.')
    return args.j0_offset, 'explicit --j0-offset'


def _check_trajectory_file(path: str) -> None:
    """Fail before touching the robot if the input is not a joint_trajectory.npz."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f'{p} not found')
    with np.load(p, allow_pickle=True) as d:
        keys = set(d.files)
    missing = [k for k in REQUIRED_KEYS if k not in keys]
    if not missing:
        return
    msg = f'{p}: not a joint_trajectory.npz (missing {missing}).'
    if 'arr_0' in keys:
        msg += ('\nThis is an rl4vla episode NPZ (end-effector deltas). It holds only part of the planner\'s env'
                '\nsteps, so joint angles cannot be recovered from it. Record a joint trajectory instead:'
                '\n  make record-joints KEY=airy_table_scene14sep26_left_image_metric TASK_OBJECT=<object>'
                '\nand pass the joint_trajectory.npz it writes into the new run directory.')
    sibling = p.with_name('joint_trajectory.npz')
    if sibling.is_file():
        msg += f'\nA joint trajectory exists next to it: {sibling}'
    raise SystemExit(msg)


def _read_actual_joints():
    sys.path.insert(0, RC5_API_PATH)
    from API.rc_api import RobotApi
    robot = RobotApi(_npz_real.RC5_IP, show_std_traceback=True)
    return robot, np.array(robot.motion.joint.get_actual_position(units='deg'), float)


def _approx_ctrl_positions(tcp_world, base, j0_offset):
    """Controller-frame TCP positions predicted from the sim TCP with the stage-1 fit (preview only)."""
    if base is None:
        return None
    rb = R.from_quat([base[4], base[5], base[6], base[3]])
    rt = R.from_quat(tcp_world[:, [4, 5, 6, 3]])
    p_base = rb.inv().apply(tcp_world[:, :3] - np.asarray(base[:3], float))
    p_tool = p_base + (rb.inv() * rt).apply(CTRL_TOOL_IN_TCP_M)
    # joint0 turns about base Z, so FK_urdf(q_sim + j0 offset) = Rz(offset) * FK_urdf(q_sim).
    return R.from_euler('z', CTRL_BASE_YAW_DEG + j0_offset, degrees=True).apply(p_tool)


def _workspace_excursion(pos):
    """Per-axis distance (m) by which each position lies outside the jog workspace."""
    return np.maximum(WS_LO - pos, 0.0) + np.maximum(pos - WS_HI, 0.0)


def _controller_fk(robot, q, path):
    """Controller-frame TCP pose (x, y, z m; rx, ry, rz deg) of each state, from the controller's own FK."""
    out = []
    for k in path:
        pose = robot.motion.kinematics.get_forward([float(v) for v in q[k]], units='deg')
        if pose is None:
            raise RuntimeError(f'controller forward kinematics failed for state {k}: {np.round(q[k], 2).tolist()}')
        out.append([float(v) for v in pose])
    return np.array(out)


def _plan(args, actual=None):
    d = np.load(args.trajectory, allow_pickle=True)
    qpos, action = d['qpos'], d['action']
    tcp_world = d['tcp_pose_world']
    meta = json.loads(str(d['meta_json']))
    hz = float(d['control_freq']) if 'control_freq' in d.files else 0.0
    if hz <= 0:
        hz = _npz_real.DEFAULT_CONTROL_HZ
    base = _recorded_base_pose(args, meta)
    args.j0_offset, j0_source = _resolve_j0_offset(args, meta, base)
    scene = json.loads((REPO / 'assets/scenes' / args.scene_key / 'simulation/scene.json').read_text())
    table_z = float(scene['groundplane_in_sim']['point'][2])
    tcp_h = tcp_world[:, 2] - table_z

    q_sim = np.degrees(qpos[:, :6])
    q_real = q_sim + np.array([args.j0_offset, 0, 0, 0, 0, 0])

    grip = action[:, 6]
    events, cmd = [], 'hold'                       # (state index before the command, pose name)
    for i, g in enumerate(grip):
        new = 'close' if g < -GRIPPER_HOLD_EPS else ('open' if g > GRIPPER_HOLD_EPS else None)
        if new is not None and new != cmd:
            events.append((i, new)); cmd = new
    first_close = next((i for i, n in events if n == 'close'), len(q_real) - 1)

    stop, reason = len(q_real) - 1, 'end of trajectory'
    if args.until_height is not None:
        below = [k for k in range(first_close + 1) if tcp_h[k] <= args.until_height]
        if below:
            stop, reason = below[0], f'sim TCP {100*tcp_h[below[0]]:.1f} cm above table (<= {100*args.until_height:.0f} cm)'
        else:
            stop, reason = first_close, 'TCP never went that low before the grasp — stopping at the grasp'
    if args.until_step is not None and args.until_step < stop:
        stop, reason = int(args.until_step), f'--until-step {args.until_step}'
    events = [(i, n) for i, n in events if i < stop]

    # Unwrap each joint by whole turns so the state the robot starts from is the
    # equivalent angle closest to where the robot actually is.
    shifts = np.zeros(6)
    if actual is not None:
        ref = stop if args.reverse else 0
        shifts = 360.0 * np.round((actual - q_real[ref]) / 360.0)
        q_real = q_real + shifts

    # Skip the recorded settle phase (the arm holds still while the scene stabilises), but
    # never skip past a hand command. The hand already starts in HOLD, so the initial OPEN
    # command at state 0 is dropped.
    moving = np.flatnonzero(np.abs(q_real[:stop + 1] - q_real[0]).max(axis=1) > IDLE_DEG)
    k0 = max(int(moving[0]) - 1, 0) if len(moving) else 0
    later = [i for i, _ in events if i > 0]
    if later:
        k0 = min(k0, later[0])
    events = [(i, n) for i, n in events if i >= k0 and i > 0]
    path = list(range(k0, stop + 1))
    if args.reverse:
        path, events = path[::-1], []
    return dict(q_sim=q_sim, q_real=q_real, shifts=shifts, tcp_h=tcp_h, tcp_world=tcp_world, table_z=table_z,
                events=events, stop=stop, reason=reason, path=path, k0=k0, hz=hz, meta=meta,
                first_close=first_close, j0_source=j0_source,
                ctrl_approx=_approx_ctrl_positions(tcp_world, base, args.j0_offset))


def _report(args, p, actual=None, fk=None):
    q, path, hz, m = p['q_real'], p['path'], p['hz'], p['meta']
    seq = np.array(path)
    duration = (len(path) - 1) / hz / args.speed
    v = (np.linalg.norm(np.diff(p['tcp_world'][seq, :3], axis=0), axis=1) * hz * args.speed
         if len(path) > 1 else np.zeros(1))
    print(f"Trajectory : {args.trajectory}")
    print(f"Source     : {m.get('source_npz')}  ({m.get('instruction')})")
    print(f"Sim run    : success={m.get('planner_success', m.get('replay_success'))}  "
          f"object lift={m.get('object_lift_m')} m  env steps={m.get('env_steps', m.get('steps'))}")
    print(f"States     : {len(q)} recorded; playing {path[0]} -> {path[-1]} ({len(path)} states, "
          f"{p['k0']} idle settle states skipped); stop: {p['reason']}")
    print(f"Hand events: {p['events'] or 'none (hand stays in HOLD)'}")
    print(f"Timing     : {hz:g} Hz sim x speed {args.speed:g} -> {duration:.0f} s of tracking, "
          f"plus arrivals and {args.hand_wait:g} s per hand event")
    print(f"TCP speed  : commanded peak {100*v.max():.1f} cm/s, p95 {100*np.percentile(v, 95):.1f} cm/s "
          f"(jog reaches ~{100*JOG_SPEED_ACTUAL_POS:.0f} cm/s at full command)")
    if np.percentile(v, 95) > JOG_SPEED_ACTUAL_POS:
        print('WARNING: the commanded TCP speed exceeds what the jog follows for much of the path — lower --speed.')
    print(f"j0 offset  : {args.j0_offset:+.2f} deg ({p['j0_source']});  360-deg shifts per joint: {p['shifts'].tolist()}")
    if args.reverse:
        print(f"Direction  : REVERSE, state {path[0]} -> {path[-1]}; hand events suppressed")
    print(f"Start sim  : {np.round(p['q_sim'][path[0]], 2).tolist()}")
    print(f"Start real : {np.round(q[path[0]], 2).tolist()}")
    print(f"Stop  real : {np.round(q[path[-1]], 2).tolist()}")
    print(f"Joint span : {np.round(q[seq].max(0) - q[seq].min(0), 2).tolist()} deg")
    print(f"TCP height : start {100*p['tcp_h'][path[0]]:.1f} cm, min {100*p['tcp_h'][seq].min():.1f} cm, "
          f"stop {100*p['tcp_h'][path[-1]]:.1f} cm above table (z={p['table_z']:.4f})")
    if fk is not None:
        pos, src = fk[:, :3], 'controller FK'
    elif p['ctrl_approx'] is not None:
        pos, src = p['ctrl_approx'][seq], 'stage-1 fit preview, no robot'
    else:
        pos, src = None, None
    if pos is not None:
        print(f"TCP targets: {src}: x [{pos[:, 0].min():.3f}, {pos[:, 0].max():.3f}]  "
              f"y [{pos[:, 1].min():.3f}, {pos[:, 1].max():.3f}]  z [{pos[:, 2].min():.3f}, {pos[:, 2].max():.3f}] m")
        exc = _workspace_excursion(pos)
        if exc.max() > 0:
            i, axis = np.unravel_index(int(np.argmax(exc)), exc.shape)
            print(f"WARNING: {int((exc.max(1) > 0).sum())} targets outside the jog workspace "
                  f"X{WORKSPACE[0]} Y{WORKSPACE[1]} Z{WORKSPACE[2]}, worst {1000*exc.max():.0f} mm on "
                  f"{'xyz'[axis]} at state {path[i]}; they are clipped as in replay_npz_real.py")
        if fk is not None and p['ctrl_approx'] is not None:
            diff = np.linalg.norm(fk[:, :3] - p['ctrl_approx'][seq], axis=1)
            print(f"FK check   : controller FK vs stage-1 fit, max {1000*diff.max():.1f} mm")
    if actual is not None:
        diff = q[path[0]] - actual
        print(f"Robot now  : {np.round(actual, 2).tolist()}")
        print(f"Start gap  : {np.round(diff, 2).tolist()} deg (max {np.abs(diff).max():.2f})")
    if np.abs(q).max() > 360:
        print('WARNING: a joint target exceeds 360 deg — check the controller joint range.')


def _execute(args, p, robot, hand, targets, recorder=None) -> bool:
    """Track the controller-frame TCP targets in time with velocity jog. True if the path completed.

    recorder.sample(t, joints_deg, tcp_pose, state) is called on every control cycle with the
    measured joints and the (fractional) trajectory state being tracked.
    """
    path, hz, n = p['path'], p['hz'], len(p['path'])
    t_rec = np.arange(n) / hz
    pos = np.clip(targets[:, :3], WS_LO, WS_HI)
    rots = R.from_euler('xyz', targets[:, 3:], degrees=True)
    slerp = Slerp(t_rec, rots) if n > 1 else None
    q_path = p['q_real'][np.array(path)]
    index_of = {k: j for j, k in enumerate(path)}
    events = [(index_of[i], name) for i, name in p['events']]
    zero3 = np.zeros(3)

    vel = np.zeros(6)
    lock = threading.Lock()
    done = threading.Event()

    def jog_loop():
        while not done.is_set() and not _stop:
            with lock:
                v = vel.copy()
            try:
                _npz_real._jog_6dof(robot, v)
            except Exception as e:
                print(f'  jog send error: {e}', flush=True)
            time.sleep(1.0 / JOG_RATE_HZ)
        try:
            _npz_real._jog_6dof(robot, np.zeros(6))
        except Exception:
            pass

    def command(v6):
        with lock:
            vel[:] = v6

    def target_at(t):
        if n == 1:
            return pos[0], rots[0], q_path[0], zero3, zero3, float(path[0])
        tc = float(np.clip(t, 0.0, t_rec[-1]))
        i = min(int(tc * hz), n - 2)
        a = tc * hz - i
        return (pos[i] + a * (pos[i + 1] - pos[i]), slerp(tc), q_path[i] + a * (q_path[i + 1] - q_path[i]),
                (pos[i + 1] - pos[i]) * hz, (rots[i + 1] * rots[i].inv()).as_rotvec() * hz,
                path[i] + a * (path[i + 1] - path[i]))

    def step(tp, tr, tq, ff_v, ff_w, state):
        """One control update: returns (pos_err, rot_err, joint_dev), None without feedback; raises _Abort."""
        actual = robot.motion.linear.get_actual_position(orientation_units='deg')
        joints = robot.motion.joint.get_actual_position(units='deg')
        if actual is None or joints is None:
            return None
        if recorder is not None:
            recorder.sample(time.monotonic(), joints, actual, state)
        ap = np.asarray(actual[:3], float)
        pe = tp - ap
        re = (tr * R.from_euler('xyz', actual[3:], degrees=True).inv()).as_rotvec()
        jdev = np.abs((np.asarray(joints, float) - tq + 180.0) % 360.0 - 180.0)
        if np.linalg.norm(pe) > args.max_track_err:
            raise _Abort(f'TCP is {100*np.linalg.norm(pe):.1f} cm from its target (--max-track-err {100*args.max_track_err:g} cm)')
        if jdev.max() > args.max_joint_dev:
            j = int(np.argmax(jdev))
            raise _Abort(f'joint{j} is {jdev[j]:.1f} deg off the recorded joint path (--max-joint-dev {args.max_joint_dev:g}): '
                         'the controller is taking a different arm configuration')
        pv = np.clip(pe * JOG_KP_POS, -JOG_SPEED_MAX_POS * args.speed, JOG_SPEED_MAX_POS * args.speed)
        rv = np.clip(re * JOG_KP_ROT, -JOG_SPEED_MAX_ROT * args.speed, JOG_SPEED_MAX_ROT * args.speed)
        ffp = np.clip(ff_v / JOG_SPEED_ACTUAL_POS, -1.0, 1.0)
        # Workspace velocity clamp: never push further out of bounds (as replay_npz_real.py).
        out = ((ap <= WS_LO) & (pv < 0)) | ((ap >= WS_HI) & (pv > 0))
        pv[out], ffp[out] = 0.0, 0.0
        command(np.concatenate([np.clip(pv / JOG_SPEED_MAX_POS + ffp, -1.0, 1.0),
                                np.clip((rv + ff_w) / JOG_SPEED_MAX_ROT, -1.0, 1.0)]))
        return pe, re, jdev

    def settle(j, label):
        """Bring the TCP to state j's pose; False means abort."""
        deadline = time.monotonic() + args.settle_timeout
        pe, re = np.full(3, np.inf), np.zeros(3)
        while time.monotonic() < deadline and not _stop:
            r = step(pos[j], rots[j], q_path[j], zero3, zero3, float(path[j]))
            if r is not None:
                pe, re, _ = r
                if np.linalg.norm(pe) <= args.pos_tol and np.degrees(np.linalg.norm(re)) <= args.rot_tol_deg:
                    break
            time.sleep(CONTROL_DT)
        command(np.zeros(6))
        if _stop:
            return False
        err, rerr = float(np.linalg.norm(pe)), float(np.degrees(np.linalg.norm(re)))
        print(f'  {label}: TCP {1000*err:.1f} mm / {rerr:.1f} deg from state {path[j]}', flush=True)
        if err <= args.pos_tol and rerr <= args.rot_tol_deg:
            return True
        if err <= 3 * args.pos_tol:
            print(f'  WARNING: not within tolerance after {args.settle_timeout:g} s; continuing', flush=True)
            return True
        print(f'  ABORT: could not reach state {path[j]}', flush=True)
        return False

    thread = threading.Thread(target=jog_loop, daemon=True, name='jog')
    thread.start()
    try:
        wall0, paused, ev = time.monotonic(), 0.0, 0
        last_print = wall0
        while not _stop:
            t = (time.monotonic() - wall0 - paused) * args.speed
            if ev < len(events) and t >= t_rec[events[ev][0]]:
                j, name = events[ev]
                ev += 1
                t_pause = time.monotonic()
                if not settle(j, f'arrival for hand {name}'):
                    return False
                print(f'  hand -> {name}', flush=True)
                hand.set_joint_positions(HAND_CLOSE if name == 'close' else HAND_HOLD)
                time.sleep(args.hand_wait)
                paused += time.monotonic() - t_pause
                continue
            if t >= t_rec[-1]:
                return settle(n - 1, 'end') and not _stop
            tp, tr, tq, fv, fw, state = target_at(t)
            r = step(tp, tr, tq, fv * args.speed, fw * args.speed, state)
            if r is not None and time.monotonic() - last_print >= 2.0:
                pe, _, jdev = r
                print(f'  t {t:5.1f}/{t_rec[-1]:.1f} s sim  state ~{path[min(int(t * hz), n - 1)]}  '
                      f'TCP err {1000*np.linalg.norm(pe):.0f} mm  max joint dev {jdev.max():.1f} deg', flush=True)
                last_print = time.monotonic()
            time.sleep(CONTROL_DT)
        return False
    except _Abort as e:
        print(f'  ABORT: {e}', flush=True)
        return False
    finally:
        command(np.zeros(6))
        done.set()
        thread.join(timeout=0.5)


def main(argv=None, recorder=None) -> int:
    """recorder (optional): open() before the start prompt, start(plan, args), sample(...) from the
    control loop, stop(completed) after the robot is set to hold. See record_paired_replay.py."""
    args = _parse(argv)
    if args.speed <= 0:
        raise SystemExit('--speed must be > 0')
    _check_trajectory_file(args.trajectory)
    if args.dry_run:
        robot, actual, fk = None, None, None
        if args.read_robot:
            robot, actual = _read_actual_joints()
        p = _plan(args, actual)
        if robot is not None:
            try:
                fk = _controller_fk(robot, p['q_real'], p['path'])
            except Exception as e:
                print(f'(controller forward kinematics query failed: {e})')
        print('[DRY RUN]' + ('' if actual is not None else '  (no --read-robot: 360-deg shifts not resolved)'))
        _report(args, p, actual, fk)
        return 0

    robot = _npz_real._init_rc5()
    actual = np.array(robot.motion.joint.get_actual_position(units='deg'), float)
    p = _plan(args, actual)
    fk = _controller_fk(robot, p['q_real'], p['path'])
    _report(args, p, actual, fk)
    if p['ctrl_approx'] is not None:
        diff = float(np.linalg.norm(fk[:, :3] - p['ctrl_approx'][np.array(p['path'])], axis=1).max())
        if diff > FK_CROSSCHECK_TOL_M:
            print(f'\nController FK differs from the sim TCP path by {100*diff:.1f} cm — wrong joint mapping or '
                  'tool setting on the controller. Not moving.')
            return 2
    start = p['q_real'][p['path'][0]]
    gap = np.abs(start - actual).max()
    if gap > args.start_tol_deg and not args.move_to_start:
        print(f'\nRobot is {gap:.1f} deg from the first state. Re-run with --move-to-start to go there first.')
        return 2

    from aero_open_sdk.aero_hand import AeroHand
    hand = AeroHand()
    print(f'AeroHand OK — actuations: {hand.get_actuations()}')
    completed = False
    try:
        hand.set_joint_positions(HAND_HOLD)
        if recorder is not None:
            recorder.open()
        if gap > args.start_tol_deg:
            input(f'Move {gap:.1f} deg to the first state at {args.start_speed:g} deg/s? ENTER to go, Ctrl+C then ENTER to abort...')
            if _stop:
                return 1
            robot.motion.joint.add_new_waypoint(angle_pose=[float(v) for v in start], speed=args.start_speed,
                                                accel=45.0, blend=0.0, units='deg')
            robot.motion.mode.set('move')
            while not robot.motion.wait_waypoint_completion(await_sec=0.2):
                if _stop:
                    return 1
        input('Ready. Press ENTER to start the jog replay (Ctrl+C then ENTER aborts)...')
        if _stop:
            print('Aborted before start.')
            return 1
        print('Starting.')
        if recorder is not None:
            recorder.start(p, args)
        if not _execute(args, p, robot, hand, fk, recorder):
            return 1
        final = np.array(robot.motion.joint.get_actual_position(units='deg'), float)
        print(f'Done. Final joints {np.round(final, 2).tolist()}, '
              f'target {np.round(p["q_real"][p["path"][-1]], 2).tolist()}')
        completed = True
    finally:
        try:
            time.sleep(0.15)      # let the last zero jog command land before switching mode
            robot.motion.mode.set('hold')
            print('Robot set to hold.')
        except Exception as e:
            print(f'WARNING: could not set hold: {e}')
        if recorder is not None:
            try:
                recorder.stop(completed)
            except Exception as e:
                print(f'WARNING: recorder stop failed: {e}')
        if not completed:
            # Ctrl+C, an abort or an exception: do not leave the hand gripping.
            try:
                hand.set_joint_positions(HAND_OPEN); time.sleep(0.5)
            except Exception:
                pass
        try:
            hand.close()
        except Exception:
            pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
