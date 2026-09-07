from dataclasses import dataclass

from openreal2sim.simulation.maniskill.planner_core.pregrasp_policy import (
    build_pregrasp_pose_from_grasp,
)


@dataclass(frozen=True)
class FakePose:
    p: tuple[float, float, float]
    q: tuple[float, float, float, float]

    def __mul__(self, other):
        return FakePose(
            p=(
                self.p[0] + other.p[0],
                self.p[1] + other.p[1],
                self.p[2] - other.p[2],
            ),
            q=self.q,
        )


def test_build_pregrasp_pose_from_grasp_preserves_orientation():
    grasp_pose = FakePose(
        p=(0.0503, -1.2212, 0.1660),
        q=(0.0, 0.6605, 0.7508, 0.0),
    )

    pregrasp_pose = build_pregrasp_pose_from_grasp(grasp_pose, retract_distance=0.1)

    assert pregrasp_pose.q == grasp_pose.q
    assert pregrasp_pose.p == (0.0503, -1.2212, 0.266)
