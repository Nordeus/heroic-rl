import logging
import uuid

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from termcolor import colored

from heroic_rl.agent import SpatialBuffer
from heroic_rl.train import obs
from heroic_rl.utils.mpi_tools import proc_id

from ..client import HeroicClient, Player

logger = logging.getLogger("heroic.env")


def create_action_space(cfg):
    return spaces.Dict(
        {
            "spell": spaces.Discrete(cfg.act_spell_shape),
            "spatial": spaces.Discrete(cfg.act_spatial_shape),
        }
    )


def create_observation_space(cfg):
    return spaces.Dict(
        {
            "spatial": spaces.Box(low=0, high=1000, shape=cfg.obs_spatial_shape),
            "non_spatial": spaces.Box(
                low=-30, high=1000, shape=cfg.obs_non_spatial_shape
            ),
            "mask_spell": spaces.Box(low=0, high=1, shape=cfg.obs_mask_spell_shape),
            "mask_spatial": spaces.Box(
                low=0, high=1, shape=cfg.obs_mask_spatial_shape,
            ),
            "if_spawn_spell": spaces.Box(
                low=0, high=1, shape=cfg.obs_if_spawn_spell_shape
            ),
        }
    )


class HeroicEnv(gym.Env):
    def __init__(self, cfg):
        self.cfg = cfg
        self.current_battle_id = None
        self.reward_fn = cfg.reward_fn
        self.reset_client()

        self.action_space = create_action_space(cfg)
        self.observation_space = create_observation_space(cfg)

        self.seed(cfg.seed)
        self.step_count = 0
        self.nb_episodes = 0
        self.battle_state = None

        # For logging purposes
        self.x_actions = np.zeros(cfg.map.num_bins)
        self.y_actions = np.zeros(cfg.map.num_lanes)
        self.spell_actions = np.zeros(cfg.spells.num_spells)

        self._spatial_buff = SpatialBuffer(cfg.architecture.num_stacked_past_exp)

        self._current_left_deck = None
        self._current_right_deck = None

    def reset_client(self):
        server_idx = proc_id() % len(self.cfg.servers)
        host, port = self.cfg.servers[server_idx].split(":")
        self.client = HeroicClient(host=host, port=port)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, left_action, right_action):
        self.step_count += 1
        client_left_action = obs.get_client_action(
            left_action, self.battle_state, self.cfg, player="LeftPlayer"
        )
        client_right_action = obs.get_client_action(
            right_action, self.battle_state, self.cfg, player="RightPlayer"
        )

        # save actions
        # TODO dimitrijer: this should not be here
        if proc_id() == 0 and left_action:
            spell = int(left_action["spell"][0])
            lane = int(left_action["spatial"][0] // self.cfg.map.num_bins)
            coordinate = int(left_action["spatial"][0] % self.cfg.map.num_bins)
            self.x_actions[coordinate] += 1
            self.y_actions[lane] += 1
            self.spell_actions[spell] += 1

        self.battle_state = self.client.step_battle(
            self.current_battle_id, client_left_action, client_right_action
        )

        done = obs.is_battle_done(self.battle_state)
        reward = self.reward_fn(self.battle_state)

        # log action distributions
        # TODO dimitrijer: this should not be here
        if done and self.nb_episodes % 20 == 0 and proc_id() == 0:
            logger.info(
                colored("X action distribution %s", "cyan"),
                self.x_actions / np.sum(self.x_actions),
            )
            logger.info(
                colored("Y action distribution %s", "cyan"),
                self.y_actions / np.sum(self.y_actions),
            )
            logger.info(
                colored("Spell action distribution: %s", "cyan"),
                self.spell_actions / np.sum(self.spell_actions),
            )
            self.x_actions = np.zeros(self.cfg.map.num_bins)
            self.y_actions = np.zeros(self.cfg.map.num_lanes)
            self.spell_actions = np.zeros(self.cfg.spells.num_spells)

        info = obs.battle_state_to_info(
            self.battle_state,
            self._spatial_buff,
            self._current_left_deck,
            self._current_right_deck,
            self.cfg,
        )
        return (info["o"], reward, done, info)

    def reset(self, left_brain=None, right_brain=None):
        self.current_battle_id = uuid.UUID(bytes=self.np_random.bytes(16))
        self._spatial_buff = SpatialBuffer(self.cfg.architecture.num_stacked_past_exp)

        all_decks = self.cfg.decks.get_decks()
        self._current_left_deck = all_decks[self.np_random.choice(len(all_decks))]

        # Pick opponent decks of similar arena level
        matching_decks = self.cfg.decks.get_decks(
            arena_level=self._current_left_deck.arena_level
        )
        self._current_right_deck = matching_decks[
            self.np_random.choice(len(matching_decks))
        ]

        castle_level = self.cfg.spells.get_castle_level(
            self._current_left_deck.arena_level
        )

        left_player = Player(
            castle_level=castle_level, deck=self._current_left_deck, brain=left_brain,
        )
        right_player = Player(
            castle_level=castle_level, deck=self._current_right_deck, brain=right_brain,
        )

        self.battle_state = self.client.create_new_battle(
            self.current_battle_id, left_player, right_player
        )
        self.nb_episodes += 1
        self.step_count = 0

        info = obs.battle_state_to_info(
            self.battle_state,
            self._spatial_buff,
            self._current_left_deck,
            self._current_right_deck,
            self.cfg,
        )
        return info["o"], info

    def render(self, mode="human"):
        return None

    def close(self):
        pass
