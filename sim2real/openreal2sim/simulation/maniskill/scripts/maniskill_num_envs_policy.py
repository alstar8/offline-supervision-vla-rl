from __future__ import annotations

DEFAULT_RC5_SIM_BACKEND = "physx_cuda"


def resolve_requested_num_envs(
    requested_num_envs: int | None,
    *,
    sim_backend: str | None,
    render_backend: str | None,
    save_video: bool,
    vis: bool,
) -> int:
    """Preserve the caller-requested batch size without GPU-driven auto-expansion.

    ManiSkill already supports single-environment GPU physics when the caller
    explicitly passes ``sim_backend=physx_cuda``. We therefore keep ``num_envs``
    exactly as requested instead of silently rewriting ``1 -> 2``.
    """

    del sim_backend, render_backend, save_video, vis

    if requested_num_envs is None:
        return 1

    num_envs = int(requested_num_envs)
    if num_envs <= 0:
        raise ValueError("--num_envs must be >= 1")
    return num_envs
