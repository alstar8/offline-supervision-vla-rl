"""
OpenReal2Sim robot agents (extensions for ManiSkill).
Import this module to register WidowX agents before creating env.
"""
from .widowx_openr2s import WidowX250SOpenR2S  # noqa: F401 - registers agent
from .widowx_bridgedataset import WidowX250SBridgeDatasetFlatTableOpenR2S  # noqa: F401 - registers agent
from .widowx_openr2s_rl import WidowX250SOpenR2S_RL  # noqa: F401 - registers agent
from .rc5_aero_hand_openr2s import RC5AeroHandOpenR2S  # noqa: F401 - registers agent
from .rc5_aero_hand_openr2s_rl import RC5AeroHandOpenR2S_RL  # noqa: F401 - registers agent
