"""
This module contains Heroic gym environment as well as HTTP client that
communicates with RL training server.
"""

from gym.envs.registration import register

register(
    id="Heroic-v0",
    entry_point="heroic_rl.gym_heroic.envs:HeroicEnv"
    # TODO figure out what deterministic does
)
