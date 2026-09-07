from dataclasses import dataclass

from openreal2sim.simulation.maniskill.planner_core.descend_policy import (
    build_descend_pose_from_grasp,
)


@dataclass(frozen=True)
class FakePose:
    p: tuple[float, float, float]
    q: tuple[float, float, float, float]


def test_build_descend_pose_from_grasp_preserves_orientation():
    grasp_pose = FakePose(
        p=(0.0502, -1.2216, 0.1660),
        q=(0.5386, 0.4730, 0.5193, -0.4653),
    )

    descend_pose = build_descend_pose_from_grasp(grasp_pose, world_tweak_xyz=(0.0, 0.0, 0.045))

    assert descend_pose.q == grasp_pose.q
    assert descend_pose.p == (0.0502, -1.2216, 0.21100000000000002)


def test_build_descend_pose_from_grasp_default_matches_grasp_pose():
    grasp_pose = FakePose(
        p=(0.0502, -1.2216, 0.1660),
        q=(0.0, 0.6576, 0.7534, 0.0),
    )

    descend_pose = build_descend_pose_from_grasp(grasp_pose)

    assert descend_pose.q == grasp_pose.q
    assert descend_pose.p == grasp_pose.p


def test_build_descend_pose_from_grasp_with_zero_tweak_matches_grasp_pose():
    grasp_pose = FakePose(
        p=(0.0502, -1.2216, 0.1660),
        q=(0.0, 0.6576, 0.7534, 0.0),
    )

    descend_pose = build_descend_pose_from_grasp(grasp_pose, world_tweak_xyz=(0.0, 0.0, 0.0))

    assert descend_pose.q == grasp_pose.q
    assert descend_pose.p == grasp_pose.p
