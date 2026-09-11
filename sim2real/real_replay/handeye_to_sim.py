#!/usr/bin/env python3
"""Turn a D405 hand-eye result into WRIST_CAMERA_LOCAL_P / WRIST_CAMERA_LOCAL_Q.

    calibrate_wrist_handeye.py solve --dir runs/handeye_X
    handeye_to_sim.py --dir runs/handeye_X

Every link in the chain below is measured or checked against data, not assumed:

  hand-eye X        OpenCV camera in the RC5 controller TCP frame
                    (calibrate_wrist_handeye.py solve -> handeye_result.json).
  T_tcp_prehand     URDF 'prehand' link in the controller TCP frame. The
                    controller reports a TCP with a tool offset on top of the
                    flange (~205 mm along it), so this is fitted from the
                    recorded joints + TCP through the URDF. URDF joint0 is
                    RC5 joint0 - 90 deg; the sim home qpos confirms it
                    (0.2853 rad = 16.35 deg against a real home of 106.35 deg).
  OpenCV -> SAPIEN  fixed axis swap. SAPIEN camera: x forward, y left, z up.
                    OpenCV optical: x right, y down, z forward.
  rot90             eval_openvla_real.py feeds the model np.rot90(k=-1) of the
                    raw D405 frame, which is that camera rolled -90 deg about
                    its optical axis. The sim camera must reproduce this
                    effective camera, not the raw one. The script prints the
                    raw and the compensated comparison so the sign is checked
                    on every run rather than trusted.
"""
from __future__ import annotations

import argparse
import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

SIM2REAL = Path(__file__).resolve().parents[1]
URDF = (
    SIM2REAL
    / "openreal2sim/simulation/maniskill/robot_assets/rc5_aero_hand/urdf_rc5_right_hand"
    / "Robot _with_right_hand_colored_visual_continuous.urdf"
)
ENV_PY = SIM2REAL / "openreal2sim/simulation/maniskill/envs/openr2s_ms_env.py"
JOINT0_OFFSET_DEG = -90.0
ROT90_ROLL_DEG = -90.0  # np.rot90(k=-1) in eval_openvla_real._prepare_wrist_inset

# SAPIEN camera axes expressed in the OpenCV optical frame.
_T_CV_SAPIEN = np.eye(4)
_T_CV_SAPIEN[:3, :3] = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)


def _T(R: np.ndarray, t) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = np.asarray(t, dtype=float).ravel()
    return M


def _load_urdf():
    joints = {}
    for j in ET.parse(URDF).getroot().findall("joint"):
        o, a = j.find("origin"), j.find("axis")
        xyz = (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()
        rpy = (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()
        joints[j.find("child").get("link")] = dict(
            name=j.get("name"),
            type=j.get("type"),
            parent=j.find("parent").get("link"),
            origin=_T(Rot.from_euler("xyz", [float(v) for v in rpy]).as_matrix(), [float(v) for v in xyz]),
            axis=np.array([float(v) for v in (a.get("xyz") if a is not None else "0 0 1").split()]),
        )
    return joints


def _chain(joints, link: str) -> list[str]:
    out = []
    while link in joints:
        out.append(link)
        link = joints[link]["parent"]
    return out[::-1]


def _fk(joints, q_rc5_deg, target: str) -> np.ndarray:
    active = [joints[c]["name"] for c in _chain(joints, "prehand") if joints[c]["type"] in ("revolute", "continuous")]
    q = list(q_rc5_deg)
    q[0] += JOINT0_OFFSET_DEG
    qmap = dict(zip(active, np.radians(q)))
    T = np.eye(4)
    for child in _chain(joints, target):
        j = joints[child]
        T = T @ j["origin"]
        if j["name"] in qmap:
            T = T @ _T(Rot.from_rotvec(j["axis"] * qmap[j["name"]]).as_matrix(), [0, 0, 0])
    return T


def _tcp_T(tcp_m_deg) -> np.ndarray:
    return _T(Rot.from_euler("xyz", tcp_m_deg[3:6], degrees=True).as_matrix(), tcp_m_deg[:3])


def _sim_camera_constants() -> tuple[np.ndarray, np.ndarray]:
    wanted = {"WRIST_CAMERA_LOCAL_P", "WRIST_CAMERA_LOCAL_Q"}
    found = {}
    for node in ast.parse(ENV_PY.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in wanted:
                found[node.targets[0].id] = np.array(ast.literal_eval(node.value), dtype=float)
    missing = wanted - found.keys()
    if missing:
        raise SystemExit(f"{sorted(missing)} not found in {ENV_PY}")
    return found["WRIST_CAMERA_LOCAL_P"], found["WRIST_CAMERA_LOCAL_Q"]


def _sim_to_X(T_tcp_prehand, p, q_wxyz) -> np.ndarray:
    """Sim camera constants (SAPIEN camera in prehand) -> OpenCV camera in TCP."""
    T_sapien_in_prehand = _T(Rot.from_quat(np.roll(q_wxyz, -1)).as_matrix(), p)
    return T_tcp_prehand @ T_sapien_in_prehand @ np.linalg.inv(_T_CV_SAPIEN)


def _X_to_sim(T_tcp_prehand, X) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV camera in TCP -> sim camera constants (SAPIEN camera in prehand)."""
    T = np.linalg.inv(T_tcp_prehand) @ X @ _T_CV_SAPIEN
    q = np.roll(Rot.from_matrix(T[:3, :3]).as_quat(), 1)
    return T[:3, 3], (q if q[0] >= 0 else -q)


def _angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees((Rot.from_matrix(Ra).inv() * Rot.from_matrix(Rb)).magnitude()))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, help="hand-eye run directory (after `solve`)")
    args = parser.parse_args(argv)

    run = Path(args.dir).expanduser().resolve()
    he = json.loads((run / "handeye_result.json").read_text())
    X_raw = _T(np.array(he["R_cam2gripper"]), he["t_cam2gripper_m"])
    print(f"hand-eye: method={he['method']}  shots={he['shots_used']}")

    # 1. controller tool frame on top of frame6, fitted over every recorded pose
    joints = _load_urdf()
    metas = [json.loads(p.read_text()) for p in sorted((run / "shots").glob("shot_*.json"))]
    tools = [np.linalg.inv(_fk(joints, m["joints_deg"], "frame6")) @ _tcp_T(m["tcp_m_deg"]) for m in metas]
    t_tool = np.mean([T[:3, 3] for T in tools], axis=0)
    rv = np.array([Rot.from_matrix(T[:3, :3]).as_rotvec() for T in tools])
    T_tool = _T(Rot.from_rotvec(rv.mean(0)).as_matrix(), t_tool)
    resid_mm = [
        np.linalg.norm((_fk(joints, m["joints_deg"], "frame6") @ T_tool)[:3, 3] - np.array(m["tcp_m_deg"][:3])) * 1000
        for m in metas
    ]
    print(f"\n[tool] controller TCP in frame6: t={(t_tool * 1000).round(2).tolist()} mm")
    print(f"       FK residual over {len(metas)} poses: max {max(resid_mm):.2f} mm")
    if max(resid_mm) > 2.0:
        print("       [warn] residual above 2 mm -- the joint mapping or the URDF does not match this robot")

    T_frame6_prehand = np.linalg.inv(_fk(joints, metas[0]["joints_deg"], "frame6")) @ _fk(
        joints, metas[0]["joints_deg"], "prehand"
    )
    T_tcp_prehand = np.linalg.inv(T_tool) @ T_frame6_prehand

    # 2. compare against the sim camera, raw and with the rot90 the eval applies
    p_sim, q_sim = _sim_camera_constants()
    X_sim = _sim_to_X(T_tcp_prehand, p_sim, q_sim)
    X_eff = X_raw @ _T(Rot.from_euler("z", ROT90_ROLL_DEG, degrees=True).as_matrix(), [0, 0, 0])
    X_flip = X_raw @ _T(Rot.from_euler("z", -ROT90_ROLL_DEG, degrees=True).as_matrix(), [0, 0, 0])
    print("\n[rotation] real camera vs sim camera")
    print(f"       raw D405 frame                : {_angle_deg(X_raw[:3, :3], X_sim[:3, :3]):6.2f} deg")
    print(f"       after rot90 (roll {ROT90_ROLL_DEG:+.0f} deg)  : {_angle_deg(X_eff[:3, :3], X_sim[:3, :3]):6.2f} deg")
    print(f"       opposite roll (sanity)        : {_angle_deg(X_flip[:3, :3], X_sim[:3, :3]):6.2f} deg")

    d_tcp = (X_eff[:3, 3] - X_sim[:3, 3]) * 1000
    print("\n[translation] camera origin in the TCP frame, real minus sim")
    print(f"       real={(X_eff[:3, 3] * 1000).round(2).tolist()} mm  sim={(X_sim[:3, 3] * 1000).round(2).tolist()} mm")
    print(f"       delta={d_tcp.round(2).tolist()} mm  (|delta|={np.linalg.norm(d_tcp):.2f} mm)")

    # 3. what the sim constants should become
    p_new, q_new = _X_to_sim(T_tcp_prehand, X_eff)
    print("\n[proposal] openr2s_ms_env.py")
    print(f"       WRIST_CAMERA_LOCAL_P = {[round(float(v), 5) for v in p_new]}   # was {p_sim.round(5).tolist()}")
    print(f"       WRIST_CAMERA_LOCAL_Q = {[round(float(v), 6) for v in q_new]}   # was {q_sim.round(6).tolist()}")

    out = {
        "source": str(run / "handeye_result.json"),
        "tool_t_in_frame6_m": t_tool.tolist(),
        "tool_fk_residual_max_mm": float(max(resid_mm)),
        "rotation_vs_sim_deg": {
            "raw": _angle_deg(X_raw[:3, :3], X_sim[:3, :3]),
            "rot90_compensated": _angle_deg(X_eff[:3, :3], X_sim[:3, :3]),
        },
        "translation_delta_tcp_mm": d_tcp.tolist(),
        "proposed_WRIST_CAMERA_LOCAL_P": p_new.tolist(),
        "proposed_WRIST_CAMERA_LOCAL_Q": q_new.tolist(),
        "current_WRIST_CAMERA_LOCAL_P": p_sim.tolist(),
        "current_WRIST_CAMERA_LOCAL_Q": q_sim.tolist(),
    }
    (run / "sim_camera_proposal.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {run / 'sim_camera_proposal.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
