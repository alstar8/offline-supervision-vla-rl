"""Shared policy for building pregrasp poses from grasp poses."""

from __future__ import annotations


def _coerce_like(value, template):
    template_type = type(template)
    try:
        return template_type(value)
    except TypeError:
        return value


def _make_pose_like(pose_type, p, q):
    try:
        return pose_type(p=p, q=q)
    except TypeError:
        try:
            return pose_type(p=p)
        except TypeError:
            return pose_type(p, q)


def build_pregrasp_pose_from_grasp(grasp_pose, retract_distance: float = 0.1):
    """Build pregrasp by retracting along local -Z from grasp pose."""

    pose_type = type(grasp_pose)
    identity_q = _coerce_like((1.0, 0.0, 0.0, 0.0), grasp_pose.q)
    local_offset_p = _coerce_like((0.0, 0.0, -float(retract_distance)), grasp_pose.p)
    local_offset_pose = _make_pose_like(pose_type, p=local_offset_p, q=identity_q)

    try:
        return grasp_pose * local_offset_pose
    except Exception:
        current_p = list(grasp_pose.p)
        current_p[2] = float(current_p[2]) + float(retract_distance)
        lifted_p = _coerce_like(current_p, grasp_pose.p)
        return _make_pose_like(pose_type, p=lifted_p, q=grasp_pose.q)
