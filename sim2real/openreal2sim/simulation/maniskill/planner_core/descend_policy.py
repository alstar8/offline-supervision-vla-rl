"""Shared policy for building descend poses from grasp poses."""

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


def build_descend_pose_from_grasp(grasp_pose, world_tweak_xyz=(0.0, 0.0, 0.0)):
    """Build descend by offsetting grasp position in world XYZ and preserving orientation."""

    target_p = [
        float(grasp_pose.p[0]) + float(world_tweak_xyz[0]),
        float(grasp_pose.p[1]) + float(world_tweak_xyz[1]),
        float(grasp_pose.p[2]) + float(world_tweak_xyz[2]),
    ]
    target_p = _coerce_like(target_p, grasp_pose.p)
    return _make_pose_like(type(grasp_pose), p=target_p, q=grasp_pose.q)
