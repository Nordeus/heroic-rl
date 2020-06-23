"""
Contains various reward functions. Reward function takes a battle state and
returns a numerical reward.
"""

import numpy as np

from .obs import is_battle_done


class Rewards:
    @classmethod
    def all(cls):
        return [
            k
            for k, v in cls.__dict__.items()
            if k != "all"
            and (isinstance(v, staticmethod) or isinstance(v, classmethod))
        ]

    @staticmethod
    def clipped_with_jump(battle_state, clip=0.5, jump=0.5):
        left_player_hp_perc = battle_state["LeftPlayer"]["CastleHealthPercent"]
        right_player_hp_perc = battle_state["RightPlayer"]["CastleHealthPercent"]
        return np.clip(
            left_player_hp_perc - right_player_hp_perc, -clip, clip
        ) + jump * np.sign(left_player_hp_perc - right_player_hp_perc)

    @staticmethod
    def simple(battle_state):
        if not is_battle_done(battle_state):
            return 0.0
        left_player_hp_perc = battle_state["LeftPlayer"]["CastleHealthPercent"]
        right_player_hp_perc = battle_state["RightPlayer"]["CastleHealthPercent"]
        return np.sign(left_player_hp_perc - right_player_hp_perc)

    @staticmethod
    def dense(battle_state, max_hp=1.0, scaling_factor=1.0 / 50.0):
        left_player_hp_perc = battle_state["LeftPlayer"]["CastleHealthPercent"]
        right_player_hp_perc = battle_state["RightPlayer"]["CastleHealthPercent"]
        reward = scaling_factor * (left_player_hp_perc - right_player_hp_perc) / max_hp
        if is_battle_done(battle_state):
            reward += Rewards.simple(battle_state)
        return reward

    @staticmethod
    def value_estimate(battle_state):
        if is_battle_done(battle_state):
            return Rewards.simple(battle_state)
        else:
            return battle_state["LeftPlayer"]["EstimatorValue"] / 1000.0
