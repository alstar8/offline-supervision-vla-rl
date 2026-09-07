try:
    from mani_skill.utils.registration import register_env
    from .envs.openr2s_ms_env import OpenReal2SimEnv
    from . import agents  # noqa: F401 - triggers @register_agent() for all OpenReal2Sim agents
except ModuleNotFoundError as exc:
    # Allow lightweight planner-core contract tests to import this package without
    # pulling the full ManiSkill runtime. Real environment usage still requires
    # ManiSkill to be installed and importable.
    if exc.name != "mani_skill":
        raise
