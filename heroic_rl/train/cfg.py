"""
Configuration classes and methods.
"""

import logging
import os
import os.path as osp
import subprocess
import sys
import time

import gym
import numpy as np

from ..utils import serialization
from .decks import DeckRepository
from .enums import CastingStrategy, Rarity, Spell
from .plan import Plans
from .rewards import Rewards

logger = logging.getLogger("heroic.train")


def _get_default_cpus():
    try:
        import psutil

        return psutil.cpu_count(logical=False)
    except ImportError:
        logger.warning(
            "Could not determine CPU count, install psutil for sane defaults"
        )
        return 4


def _get_default_cuda_devices():
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))

    # We need to list GPUs in a subprocess so we don't accidentally allocate
    # all GPUs.
    cmd = [
        sys.executable if sys.executable else "python3",
        "-c",
        "import tensorflow as tf; print(["
        + "int(device.name.split(':')[-1])"
        + "for device in tf.config.experimental.list_physical_devices('GPU')"
        + "])",
    ]

    try:
        visible_gpus = subprocess.check_output(cmd, env=os.environ)
        return eval(visible_gpus)
    except subprocess.CalledProcessError as cpe:
        logger.error("Failed to get visible GPUs, exit code %d" % cpe.returncode)
        sys.exit(1)


class DictCfg(object):
    """A hierarchial, dictionary-like config object.

    * Can be read like a dict.
    * Supports iteration with (k, v) pairs, like `dict.items()`.
    * Can be serialized to/from YAML.
    * Can be updated with a standard `dict` (nesting supported).
    * Private fields starting with underscore are excluded from serialization/update.
    * Two hooks for pre-update and post-update available.
    """

    @classmethod
    def from_yaml(cls, yaml_string):
        d = serialization.safe_load_yaml(yaml_string)
        cfg = cls()
        cfg.update(d)
        return cfg

    @classmethod
    def load(cls, yaml_file):
        with open(yaml_file, "r") as f:
            return cls.from_yaml(f.read())

    def _assert_update(self, d):
        if not isinstance(d, dict):
            raise ValueError("need a dict for update()")
        for k, v in d.items():
            if k not in self.__dict__:
                raise ValueError("%s is not cfg field" % k)
            if k.startswith("_"):
                raise ValueError("cannot set private cfg fields")
            if isinstance(self.__dict__[k], DictCfg):
                if not isinstance(v, dict):
                    raise ValueError("expecting dict for DictCfg field %s" % k)
                else:
                    self.__dict__[k]._assert_update(v)

    def update(self, d: dict):
        # Validate first
        self._assert_update(d)

        self._pre_update(d)

        # Do the updates
        for k, v in d.items():
            if isinstance(self.__dict__[k], DictCfg):
                self.__dict__[k].update(v)
            else:
                self.__dict__[k] = v

        self._post_update(d)

    def as_dict(self):
        return {k: v.as_dict() if isinstance(v, DictCfg) else v for k, v in self}

    def as_yaml(self):
        return serialization.safe_dump_yaml(self.as_dict())

    def save(self, path):
        with open(path, "w") as f:
            f.write(self.as_yaml())

    def _pre_update(self, d):
        pass

    def _post_update(self, d):
        pass

    def __getitem__(self, key):
        if key.startswith("_"):
            raise ValueError("cannot query private cfg fields")
        return self.__dict__[key]

    def __repr__(self):
        return self.__dict__.__repr__()

    def __iter__(self):
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                yield (k, v)


class HyperparametersCfg(DictCfg):

    DEFAULT_GAMMA = 1.0
    DEFAULT_CLIP_RATIO = 0.2
    DEFAULT_PI_LR = 3e-4
    DEFAULT_VF_LR = 1e-3
    DEFAULT_LR = 3e-4
    DEFAULT_GRAD_CLIPPING_ENABLED = True
    DEFAULT_MAX_GRAD_NORM = 0.5
    DEFAULT_TRAIN_PI_ITERS = 10
    DEFAULT_TRAIN_V_ITERS = 80
    # todo michalw: lambda 0.95 usually used (in previous experiments 0.97)
    DEFAULT_LAM = 0.95
    DEFAULT_TARGET_KL = 1.0
    DEFAULT_VALUE_CLIPPING_ENABLED = False
    DEFAULT_CLIP_RANGE_VF = 0.2
    DEFAULT_VF_LOSS_COEF = 1.0
    DEFAULT_PI_LOSS_COEF = 1.0
    DEFAULT_VF_REG_ENABLED = False
    DEFAULT_VF_REG = 0
    DEFAULT_CLIP_VF_OUTPUT = False
    DEFAULT_BIAS_NOOPS = True

    def __init__(self):
        super().__init__()
        self.gamma = self.DEFAULT_GAMMA

        # Hyperparameter for clipping in the policy objective.
        # Roughly: how far can the new policy go from the old policy while
        # still profiting (improving the objective function)? The new policy
        # can still go farther than the clip_ratio says, but it doesn't help
        # on the objective anymore. (Usually small, 0.1 to 0.3.)
        self.clip_ratio = self.DEFAULT_CLIP_RATIO

        self.pi_lr = self.DEFAULT_PI_LR
        self.vf_lr = self.DEFAULT_VF_LR

        # This learning rate is used in case of unified policy/value network.
        self.lr = self.DEFAULT_LR

        self.max_grad_norm = self.DEFAULT_MAX_GRAD_NORM
        self.train_pi_iters = self.DEFAULT_TRAIN_PI_ITERS
        self.train_v_iters = self.DEFAULT_TRAIN_V_ITERS
        self.lam = self.DEFAULT_LAM
        self.target_kl = self.DEFAULT_TARGET_KL
        self.clip_range_vf = self.DEFAULT_CLIP_RANGE_VF
        self.vf_loss_coef = self.DEFAULT_VF_LOSS_COEF
        self.pi_loss_coef = self.DEFAULT_PI_LOSS_COEF
        self.vf_reg = self.DEFAULT_VF_REG
        self.clip_vf_output = self.DEFAULT_CLIP_VF_OUTPUT
        self.grad_clipping_enabled = self.DEFAULT_GRAD_CLIPPING_ENABLED
        self.value_clipping_enabled = self.DEFAULT_VALUE_CLIPPING_ENABLED
        self.vf_reg_enabled = self.DEFAULT_VF_REG_ENABLED
        self.bias_noops = self.DEFAULT_BIAS_NOOPS


class SpellsCfg(DictCfg):

    DEFAULT_DISABLED_SPELLS = []
    DEFAULT_SPAWN_NOOP = False
    DEFAULT_SPELL_NOOP = True
    DEFAULT_MAX_SECONDS_UNTIL_AVAILABLE = 30
    DEFAULT_LEVEL_OFFSETS_BY_RARITY = {
        Rarity.COMMON: 0,
        Rarity.RARE: 2,
        Rarity.EPIC: 5,
        Rarity.LEGENDARY: 8,
        Rarity.MYTHIC: 6,
    }
    DEFAULT_MIN_SPELL_LEVEL = 1
    DEFAULT_MAX_SPELL_LEVEL = 13
    DEFAULT_MIN_ARENA_LEVEL = 1
    DEFAULT_MAX_ARENA_LEVEL = 9

    DEFAULT_REFERENCE_SPELL_LEVEL_ENABLED = True
    DEFAULT_REFERENCE_SPELL_LEVEL = 9

    def __init__(self):
        super().__init__()

        # List of spells to disable.
        self.disabled_spells = self.DEFAULT_DISABLED_SPELLS

        # Is noop enabled for spawn (minion) spells?
        self.spawn_noop = self.DEFAULT_SPAWN_NOOP

        # Is noop enabled for regular spells?
        self.spell_noop = self.DEFAULT_SPELL_NOOP

        # Max time until spell is available - time until spell is available is
        # normalized according to this.
        self.max_seconds_until_available = self.DEFAULT_MAX_SECONDS_UNTIL_AVAILABLE

        # Used to offset spell levels by rarities - common spells need
        # more level-ups in order to match against rare spells of lower levels.
        self.level_offsets_by_rarity = self.DEFAULT_LEVEL_OFFSETS_BY_RARITY

        self.min_spell_level = self.DEFAULT_MIN_SPELL_LEVEL
        self.max_spell_level = self.DEFAULT_MAX_SPELL_LEVEL
        self.min_arena_level = self.DEFAULT_MIN_ARENA_LEVEL
        self.max_arena_level = self.DEFAULT_MAX_ARENA_LEVEL

        self.reference_spell_level_enabled = self.DEFAULT_REFERENCE_SPELL_LEVEL_ENABLED
        self.reference_spell_level = self.DEFAULT_REFERENCE_SPELL_LEVEL

        self._enabled_spells = None
        self._enabled_units = None

        # Mappings
        self._unit_to_idx = None
        self._spell_to_idx = None
        self._idx_to_spell = None
        self._nonspawn_spell_to_idx = None

        # Binary spell masks
        self._mask_nonspawn = None
        self._mask_spawn = None

        self._post_update({})

    @property
    def num_spells(self):
        return len(self.enabled_spells)

    @property
    def num_units(self):
        return len(self.enabled_units)

    @property
    def mask_spawn(self):
        return self._mask_spawn

    @property
    def mask_nonspawn(self):
        return self._mask_nonspawn

    @property
    def enabled_spells(self):
        return self._enabled_spells

    @property
    def enabled_units(self):
        return self._enabled_units

    def unit_to_idx(self, unit_type):
        return self._unit_to_idx[unit_type]

    def spell_to_idx(self, spell_type):
        return self._spell_to_idx[spell_type]

    def idx_to_spell(self, idx):
        return self._idx_to_spell[idx]

    @property
    def num_noops(self):
        num_noops = 0
        for spell in self._enabled_spells:
            if spell.is_noop:
                num_noops += 1
        return num_noops

    def get_level_increment(self, rarity, arena_level):
        low = self.level_offsets_by_rarity[rarity] + self.min_spell_level
        high = self.max_spell_level
        if self.reference_spell_level_enabled:
            return np.clip(self.reference_spell_level - low, 0, high - low)
        else:
            # Use arena level to figure out level increment.
            t = np.clip(
                (arena_level - self.min_arena_level)
                / (self.max_arena_level - self.min_arena_level),
                0,
                1,
            )
            return int(round(t * (high - low)))

    def get_castle_level(self, arena_level):
        return 1 + self.get_level_increment(Rarity.COMMON, arena_level)

    def _build_enabled_spells_and_units(self):
        self._enabled_spells = [
            spell for spell in Spell if spell not in self.disabled_spells
        ]
        units_to_enable = []

        def enable_unit(unit):
            if unit not in units_to_enable:
                units_to_enable.append(unit)

        for spell in self._enabled_spells:
            units_spawned = spell.units_spawned
            if units_spawned is None:
                continue
            elif isinstance(units_spawned, list):
                for unit in units_spawned:
                    enable_unit(unit)
            else:
                enable_unit(units_spawned)

        self._enabled_units = units_to_enable

    def _post_update(self, d):
        self._build_enabled_spells_and_units()
        self._build_spell_unit_mappings()
        self._build_spell_binary_masks()

    def _build_spell_binary_masks(self):
        if self.spell_noop:
            self._mask_nonspawn = np.array(
                [
                    float(not spell.is_spawn or spell.is_noop)  # for noop
                    for spell in self.enabled_spells
                ]
            ).reshape(1, -1)
        else:
            self._mask_nonspawn = np.array(
                [
                    float(not (spell.is_spawn or spell.is_noop))  # for no noop
                    for spell in self.enabled_spells
                ]
            ).reshape(1, -1)

        if self.spawn_noop:
            self._mask_spawn = np.array(
                [
                    float(spell.is_spawn or spell.is_noop)
                    for spell in self.enabled_spells
                ]
            ).reshape(1, -1)
        else:
            self._mask_spawn = np.array(
                [float(spell.is_spawn) for spell in self.enabled_spells]
            ).reshape(1, -1)

    def _build_spell_unit_mappings(self):
        self._unit_to_idx = {
            int(unit): idx for idx, unit in enumerate(self.enabled_units)
        }
        self._spell_to_idx = {
            int(spell): idx for idx, spell in enumerate(self.enabled_spells)
        }
        self._idx_to_spell = {
            idx: spell for idx, spell in enumerate(self.enabled_spells)
        }


class MapCfg(DictCfg):

    DEFAULT_MIN_X = -5.0
    DEFAULT_MAX_X = 5.0

    DEFAULT_MIN_Y = -1
    DEFAULT_MAX_Y = 1

    DEFAULT_BIN_WIDTH = 1.0

    DEFAULT_NUM_LANES = 3

    DEFAULT_MAX_BATTLE_TIME_SECONDS = 280

    def __init__(self):
        super().__init__()
        self.min_x = self.DEFAULT_MIN_X
        self.max_x = self.DEFAULT_MAX_X
        self.min_y = self.DEFAULT_MIN_Y
        self.max_y = self.DEFAULT_MAX_Y

        self.bin_width = self.DEFAULT_BIN_WIDTH
        self.num_lanes = self.DEFAULT_NUM_LANES

        # Duration of battle is normalized according to this.
        self.max_battle_time_seconds = self.DEFAULT_MAX_BATTLE_TIME_SECONDS

    @property
    def num_bins(self):
        return int((self.max_x - self.min_x) / self.bin_width)

    def x_to_bin(self, x):
        return min(self.num_bins - 1, int((x - self.min_x) / self.bin_width))

    def y_to_lane(self, y):
        return int(y - self.min_y)

    def bin_to_x(self, bin):
        return float(bin * self.bin_width + self.min_x)

    def lane_to_y(self, lane):
        return float(lane + self.min_y)


class ArchitectureCfg(DictCfg):
    """
    Refer to `heroic_rl.algos.layers` to see how these parameters affect
    network architecture.
    """

    class ParamsCfg(DictCfg):

        DEFAULT_FC_UNITS_CONCAT = 512
        DEFAULT_FC_UNITS_NON_SPATIAL = 32
        DEFAULT_CONV_FILTERS = 32
        DEFAULT_CONV_STRIDES = [1, 1]
        DEFAULT_CONV_KERNEL_SIZE = [1, 3]

        def __init__(self):
            super().__init__()
            self.fc_units_concat = self.DEFAULT_FC_UNITS_CONCAT
            self.fc_units_non_spatial = self.DEFAULT_FC_UNITS_NON_SPATIAL
            self.conv_filters = self.DEFAULT_CONV_FILTERS
            self.conv_strides = self.DEFAULT_CONV_STRIDES
            self.conv_kernel_size = self.DEFAULT_CONV_KERNEL_SIZE

    DEFAULT_RNN = False
    DEFAULT_RNN_SIZE = 512
    DEFAULT_UNIFIED_POLICY_VALUE = False
    DEFAULT_NUM_STACKED_PAST_EXP = 1
    DEFAULT_NON_SPATIAL_ORTHOGONAL_ENABLED = True

    def __init__(self):
        super().__init__()
        self.value_params = self.ParamsCfg()
        self.shared_params = self.ParamsCfg()
        self.spawn_params = self.ParamsCfg()
        self.spell_params = self.ParamsCfg()
        self.rnn = self.DEFAULT_RNN
        self.rnn_size = self.DEFAULT_RNN_SIZE
        self.unified_policy_value = self.DEFAULT_UNIFIED_POLICY_VALUE

        # Number of stacked spatial observations used as input.
        self.num_stacked_past_exp = self.DEFAULT_NUM_STACKED_PAST_EXP

        # Whether to use orthogonal non-spatial spell observations.
        self.non_spatial_orthogonal_enabled = (
            self.DEFAULT_NON_SPATIAL_ORTHOGONAL_ENABLED
        )

        self._post_update({})

    @property
    def empty_rnn_state(self):
        return self._empty_rnn_state

    def _post_update(self, d):
        self._empty_rnn_state = np.zeros([1, 2 * self.rnn_size])


class TrainingCfg(DictCfg):

    DEFAULT_CPUS = _get_default_cpus()
    DEFAULT_GPUS = _get_default_cuda_devices()
    DEFAULT_EXP_NAME = "experiment"
    DEFAULT_STEPS = 484800
    DEFAULT_BATCH_SIZE = 2500
    DEFAULT_EPOCHS = 10000
    DEFAULT_MAX_EP_LEN = 1400
    DEFAULT_SEED = int(time.time())
    DEFAULT_SERVERS = ["127.0.0.1:8081"]
    DEFAULT_NUM_AGENTS = 1
    DEFAULT_EPOCHS_PER_AGENT = 10000
    DEFAULT_SAVE_FREQUENCY_EPOCHS = 10
    DEFAULT_PLAN = "utility"
    DEFAULT_REWARD = "simple"
    DEFAULT_DATA_DIR = osp.join(osp.abspath(os.getcwd()), "data")
    DEFAULT_DECKS_CSV_PATH = "decks.csv"
    DEFAULT_SPATIAL_MASK_STATIC_ENABLED = True
    DEFAULT_SPATIAL_MASK_DYNAMIC_ENABLED = True

    def __init__(self):
        super().__init__()
        self.spells = SpellsCfg()
        self.hyperparameters = HyperparametersCfg()
        self.map = MapCfg()
        self.architecture = ArchitectureCfg()

        self.cpus = self.DEFAULT_CPUS
        self.gpus = self.DEFAULT_GPUS
        self.exp_name = self.DEFAULT_EXP_NAME
        self.steps = self.DEFAULT_STEPS
        self.batch_size = self.DEFAULT_BATCH_SIZE
        self.epochs = self.DEFAULT_EPOCHS
        # Maximum length of trajectory / episode / battle in steps.
        self.max_ep_len = self.DEFAULT_MAX_EP_LEN
        self.seed = self.DEFAULT_SEED
        self.servers = self.DEFAULT_SERVERS
        self.num_agents = self.DEFAULT_NUM_AGENTS
        self.epochs_per_agent = self.DEFAULT_EPOCHS_PER_AGENT
        self.save_frequency_epochs = self.DEFAULT_SAVE_FREQUENCY_EPOCHS
        self.plan = self.DEFAULT_PLAN
        self.reward = self.DEFAULT_REWARD
        self.data_dir = self.DEFAULT_DATA_DIR
        self.decks_csv_path = self.DEFAULT_DECKS_CSV_PATH
        self.spatial_mask_static_enabled = self.DEFAULT_SPATIAL_MASK_STATIC_ENABLED
        self.spatial_mask_dynamic_enabled = self.DEFAULT_SPATIAL_MASK_DYNAMIC_ENABLED

        self._deck_repo = None
        self._mask_spatial_static = None

        self._post_update({})

    @property
    def output_dir(self):
        # Make a seed-specific subfolder in the experiment directory.
        subfolder = "".join([self.exp_name, "_s", str(self.seed)])
        return osp.join(self.data_dir, self.exp_name, subfolder)

    @property
    def logger_kwargs(self):
        """logger_kwargs (dict): Keyword args for EpochLogger."""
        return dict(output_dir=self.output_dir, exp_name=self.exp_name)

    def get_agent_logger_kwargs(self, agent):
        """logger_kwargs (dict): Keyword args for EpochLogger."""
        return dict(
            output_dir=osp.join(self.output_dir, "agent_%d" % agent.id),
            exp_name=self.exp_name,
        )

    @property
    def local_steps_per_epoch(self):
        """Steps per epoch for each process."""
        return int(self.steps / self.cpus)

    def create_env(self):
        """
        A function which creates a copy of the environment.
        The environment must satisfy the OpenAI Gym API.
        """
        return gym.make("Heroic-v0", cfg=self)

    def create_plan(self, actor):
        return getattr(Plans, self.plan)()

    @property
    def reward_fn(self):
        return getattr(Rewards, self.reward)

    @property
    def decks(self):
        return self._deck_repo

    @property
    def mask_spatial_static(self):
        return self._mask_spatial_static

    @property
    def obs_spatial_shape(self):
        """
        Spatial observations are cummulative health percentages of creeps of
        the same type, occupying the same discrete bin in one lane, for one
        player.
        """
        return (
            self.map.num_lanes,
            self.map.num_bins,
            2 * self.spells.num_units * self.architecture.num_stacked_past_exp,
        )

    @property
    def obs_non_spatial_shape(self):
        """
        Non-spatial observations contain own and opponent's castle health
        percent, current battle time, and indicator of when spells are available
        for casting.
        """
        if self.architecture.non_spatial_orthogonal_enabled:
            return (3 * self.spells.num_spells + 3,)
        else:
            return (self.spells.num_spells + 3,)

    @property
    def obs_mask_spatial_shape(self):
        """
        Spatial mask is a binary mask that is 1 at places where certain spell
        can be cast and 0 otherwise.
        """
        return (
            self.spells.num_spells,
            self.map.num_lanes * self.map.num_bins,
        )

    @property
    def obs_mask_spell_shape(self):
        """
        Spell mask is a binary mask that is 1 for spells that can be played right now,
        and 0 otherwise.
        """
        return (self.spells.num_spells,)

    @property
    def obs_if_spawn_spell_shape(self):
        """
        This observation is a boolean flag that determines whether we are
        querying policy for a spawn (minion) spell (1) or regular spell (0).
        """
        return (1,)

    @property
    def act_spatial_shape(self):
        """
        Spatial action defines place on the map where spell will be cast, and it
        is in `[0, NUM_LANES * NUM_BINS]` range.
        """
        return self.map.num_lanes * self.map.num_bins

    @property
    def act_spell_shape(self):
        """
        Spell action determines index of spell which will be cast.
        """
        return self.spells.num_spells

    def _post_update(self, d):
        self.data_dir = osp.abspath(self.data_dir)
        self._deck_repo = DeckRepository.from_csv(self.decks_csv_path, self)
        self._mask_spatial_static = self._build_mask_spatial_static()

    def _build_mask_spatial_static(self):
        # Static spatial mask that never changes
        mask_spatial = np.ones(
            (self.spells.num_spells, self.map.num_lanes * self.map.num_bins,)
        )

        if not self.spatial_mask_static_enabled:
            return mask_spatial

        for spell_idx in range(self.spells.num_spells):
            spell = self.spells.idx_to_spell(spell_idx)
            if spell.casting_strategy == CastingStrategy.ENTIRE_MAP:
                # Nothing to do, ones everywhere is what we need.
                continue
            elif spell.casting_strategy == CastingStrategy.DOES_NOT_MATTER:
                # Enable first coordinate only.
                mask_spatial[spell_idx, 1:] = 0
            elif spell.casting_strategy == CastingStrategy.SINGLE_LANE:
                # Enable first coordinate in each lane.
                for lane in range(self.map.num_lanes):
                    lane_start = lane * self.map.num_bins
                    next_lane_start = (lane + 1) * self.map.num_bins
                    mask_spatial[spell_idx, (lane_start + 1) : next_lane_start] = 0
            elif spell.casting_strategy == CastingStrategy.CONTROLLED_AREA:
                # Enable own half of map.
                for lane in range(self.map.num_lanes):
                    half_lane = lane * self.map.num_bins + self.map.num_bins // 2
                    next_lane_start = (lane + 1) * self.map.num_bins
                    mask_spatial[spell_idx, (half_lane + 1) : next_lane_start] = 0

        return mask_spatial
