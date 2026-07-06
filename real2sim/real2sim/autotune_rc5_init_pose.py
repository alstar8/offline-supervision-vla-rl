from dataclasses import dataclass
from pathlib import Path
import json

import imageio.v3 as iio
import numpy as np
import torch
import tyro

from real2sim.calibrate_rc5_pose import Args as CalibArgs
from real2sim.calibrate_rc5_pose import _build_env, _extract_obs_frame
from real2sim.openreal2sim_validation import ROBOT_BASE_POSE_P


TUNED_JOINT_IDXS = [1, 2, 4]
BASE_QPOS = np.array(
    [
        0.0, -0.10, 0.72, 0.0, -1.05, 0.0,
        0.65, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    dtype=np.float32,
)


@dataclass
class Args:
    output_dir: str = "real2sim/rc5_autotune"
    image_prefix: str = "rc5_autotune"
    seed: int = 0
    max_passes: int = 20


def _candidate_qposes() -> list[np.ndarray]:
    base = BASE_QPOS.copy()
    deltas = [
        (0.00, 0.00, 0.00),
        (0.00, 0.00, -0.12),
        (0.00, 0.00, +0.12),
        (0.00, -0.14, 0.00),
        (0.00, +0.14, 0.00),
        (-0.10, 0.00, 0.00),
        (+0.10, 0.00, 0.00),
        (0.00, -0.14, -0.12),
        (0.00, -0.14, +0.12),
        (0.00, +0.14, -0.12),
        (0.00, +0.14, +0.12),
        (-0.10, 0.00, -0.12),
        (-0.10, 0.00, +0.12),
        (+0.10, 0.00, -0.12),
        (+0.10, 0.00, +0.12),
        (-0.08, -0.18, -0.08),
        (+0.08, -0.18, -0.08),
        (-0.08, +0.18, +0.08),
        (+0.08, +0.18, +0.08),
        (0.00, -0.24, -0.18),
    ]
    qposes = []
    for dj1, dj2, dj4 in deltas:
        q = base.copy()
        q[1] += dj1
        q[2] += dj2
        q[4] += dj4
        qposes.append(q)
    return qposes


def _mask_stats(mask: torch.Tensor):
    pixels = int(mask.sum().item())
    if pixels == 0:
        return dict(pixels=0, cx=None, cy=None, xmin=None, xmax=None, ymin=None, ymax=None)
    ys, xs = torch.where(mask)
    return dict(
        pixels=pixels,
        cx=float(xs.float().mean().item()),
        cy=float(ys.float().mean().item()),
        xmin=int(xs.min().item()),
        xmax=int(xs.max().item()),
        ymin=int(ys.min().item()),
        ymax=int(ys.max().item()),
    )


def _score_trial(seg: torch.Tensor, ids: dict[str, int]) -> tuple[float, dict]:
    body5 = _mask_stats(seg == ids["body5"])
    body6 = _mask_stats(seg == ids["body6"])
    hand = _mask_stats(seg == ids["right_base_link"])

    score = 0.0
    details = {"body5": body5, "body6": body6, "hand": hand}

    if min(body5["pixels"], body6["pixels"], hand["pixels"]) <= 0:
        return -1e9, details

    # Prefer all three links being clearly visible.
    score += 0.002 * min(body5["pixels"], 20000)
    score += 0.002 * min(body6["pixels"], 20000)
    score += 0.002 * min(hand["pixels"], 20000)

    # Prefer hand visible below the arm chain in the image.
    if body5["cy"] < body6["cy"] < hand["cy"]:
        score += 250.0
    else:
        score -= 250.0

    # Prefer the chain to be more vertical than diagonal.
    x_std = np.std([body5["cx"], body6["cx"], hand["cx"]])
    score -= 3.0 * x_std

    # Keep the hand reasonably inside frame, not glued to an edge.
    if 80.0 <= hand["cx"] <= 560.0:
        score += 80.0
    else:
        score -= 80.0
    if 40.0 <= hand["cy"] <= 380.0:
        score += 80.0
    else:
        score -= 80.0

    # Prefer less extreme occlusion by keeping the visible arm bbox compact.
    xmin = min(body5["xmin"], body6["xmin"], hand["xmin"])
    xmax = max(body5["xmax"], body6["xmax"], hand["xmax"])
    ymin = min(body5["ymin"], body6["ymin"], hand["ymin"])
    ymax = max(body5["ymax"], body6["ymax"], hand["ymax"])
    bbox_area = float((xmax - xmin + 1) * (ymax - ymin + 1))
    score -= 0.002 * bbox_area
    details["bbox"] = dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, area=bbox_area, x_std=x_std)

    return score, details


def main(args: Args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calib_args = CalibArgs(
        output_dir=str(output_dir),
        image_prefix=args.image_prefix,
        observation_camera_mode="manual_best",
        show_debug_markers=False,
    )
    env = _build_env(calib_args, camera_mode="manual_best")

    best = None
    trials = []
    qposes = _candidate_qposes()[: args.max_passes]
    try:
        for idx, qpos in enumerate(qposes):
            obs, info = env.reset(
                seed=args.seed,
                options={
                    "reconfigure": True,
                    "show_debug_markers": False,
                    "robot_far_away": False,
                    "robot_base_pose_p": list(ROBOT_BASE_POSE_P),
                    "robot_init_qpos": qpos.tolist(),
                },
            )
            frame = _extract_obs_frame(obs)
            seg = obs["sensor_data"]["3rd_view_camera"]["segmentation"][0, ..., 0]
            links = env.unwrapped.agent.robot.links_map
            ids = {
                "body5": int(links["body5"].per_scene_id[0].item()),
                "body6": int(links["body6"].per_scene_id[0].item()),
                "right_base_link": int(links["right_base_link"].per_scene_id[0].item()),
            }
            score, details = _score_trial(seg, ids)

            image_path = output_dir / f"{args.image_prefix}_{idx:03d}.png"
            meta_path = output_dir / f"{args.image_prefix}_{idx:03d}.json"
            iio.imwrite(image_path, frame)
            payload = {
                "trial_index": idx,
                "score": score,
                "robot_base_pose_p": list(ROBOT_BASE_POSE_P),
                "robot_init_qpos": qpos.tolist(),
                "details": details,
                "info": {k: (v[0].item() if torch.is_tensor(v) else v) for k, v in info.items()},
            }
            meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[{idx + 1}/{len(qposes)}] score={score:.2f} saved={image_path.name}")

            trials.append(payload)
            if best is None or score > best["score"]:
                best = payload
    finally:
        env.close()

    summary_path = output_dir / f"{args.image_prefix}_summary.json"
    summary_path.write_text(json.dumps({"best": best, "trials": trials}, indent=2), encoding="utf-8")
    print("\nBest trial:")
    print(json.dumps(best, indent=2))
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
