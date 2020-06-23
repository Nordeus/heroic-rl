import logging

import numpy as np

from ..agent import SpatialBuffer
from ..train import Owner
from ..train.obs import NOOP_ACTION, battle_state_to_info, get_client_action

logger = logging.getLogger(__name__)


class InferenceState:
    """Adds no-op and state buffering capability to InferenceAgent."""

    def __init__(self, agent, battle_id, left_deck, right_deck, owner):
        self.agent = agent
        self.battle_id = battle_id
        self.left_deck = left_deck
        self.right_deck = right_deck
        self.owner = Owner(owner)
        self.steps = 0
        self.noop_spawn_expiry_time = 0
        self.noop_spell_expiry_time = 0
        self.spatial_buff = SpatialBuffer(agent.cfg.architecture.num_stacked_past_exp)
        # Last battle info.
        self.info = None

    @property
    def is_spawn_available(self):
        if self.info is None:
            return False
        return (
            self.info["left_spawn_available"]
            if self.owner == Owner.LEFT_PLAYER
            else self.info["right_spawn_available"]
        )

    @property
    def is_spell_available(self):
        if self.info is None:
            return False
        return (
            self.info["left_spell_available"]
            if self.owner == Owner.LEFT_PLAYER
            else self.info["right_spell_available"]
        )

    @property
    def can_play(self):
        if self.info is None:
            return False
        return (
            self.info["left_can_play"]
            if self.owner == Owner.LEFT_PLAYER
            else self.info["right_can_play"]
        )

    @property
    def can_play_spell(self):
        if self.info is None:
            return False
        return (
            self.info["battle_time"] >= self.noop_spell_expiry_time
            and self.is_spell_available
        )

    @property
    def can_play_spawn(self):
        if self.info is None:
            return False
        return (
            self.info["battle_time"] >= self.noop_spawn_expiry_time
            and self.is_spawn_available
        )

    @property
    def obs(self):
        if self.info is None:
            return False
        return (
            self.info["o"] if self.owner == Owner.LEFT_PLAYER else self.info["o_flip"]
        )

    def step(self, battle_state):
        if battle_state["BattleState"] != "InProgress":
            logger.error("Battle(id=%s) already finished", self.battle_id)
            return None

        self.steps += 1
        self.info = battle_state_to_info(
            battle_state,
            self.spatial_buff,
            self.left_deck,
            self.right_deck,
            self.agent.cfg,
        )
        if self.can_play:
            if self.can_play_spawn or self.can_play_spell:
                self.obs["if_spawn_spell"] = (
                    np.array([1.0]) if self.is_spawn_available else np.array([0.0])
                )
                a, _ = self.agent.get_next_action(self.obs)
                self._update_noop_timers(a)
                client_action = get_client_action(
                    a, battle_state, self.agent.cfg, player=self.owner.side
                )
                logger.debug(
                    "Play %s in Battle(id=%s, time=%s, steps=%s)",
                    self.agent.cfg.spells.idx_to_spell(a["spell"][0]),
                    client_action,
                    self.battle_id,
                    self.info["battle_time"],
                    self.steps,
                )
                return client_action
            else:
                # NOOP is in effect, or there is nowhere to cast available
                # spells (because of dynamic masks), return noop action.
                logger.debug(
                    "Battle(id=%s, time=%s, steps=%d) spawn expiry=%s, spell expiry=%s",
                    self.battle_id,
                    self.info["battle_time"],
                    self.steps,
                    self.noop_spawn_expiry_time,
                    self.noop_spell_expiry_time,
                )
                return NOOP_ACTION
        else:
            # Nothing to do. Should not happen.
            logger.error(
                "Nothing to do Battle(id=%s, time=%s, steps=%d), state:\n%s",
                self.battle_id,
                self.info["battle_time"],
                self.steps,
                battle_state,
            )
            return None

    def _update_noop_timers(self, a):
        spell = self.agent.cfg.spells.idx_to_spell(a["spell"][0])
        if spell.is_noop:
            if self.can_play_spawn:
                # Noop duration is encoded as negative spell ID.
                self.noop_spawn_expiry_time = self.info["battle_time"] - int(spell)
                logger.debug(
                    "Started noop spawn Battle(id=%s, time=%s, steps=%d), expiry=%s",
                    self.battle_id,
                    self.info["battle_time"],
                    self.steps,
                    self.noop_spawn_expiry_time,
                )
            else:
                self.noop_spell_expiry_time = self.info["battle_time"] - int(spell)
                logger.debug(
                    "Started spell noop Battle(id=%s, time=%s, steps=%d), expiry=%s",
                    self.battle_id,
                    self.info["battle_time"],
                    self.steps,
                    self.noop_spell_expiry_time,
                )
