"""
Contains functions that process and transform observations and battle state.

These include building Gym observation space from JSON battle state, building spell
masks (static and dynamic), converting actions from Gym action space to JSON (dict)
actions to be sent to the server etc.
"""

from typing import Tuple

import numpy as np

from .enums import Owner, Spell

NOOP_ACTION = {"CardIndex": -1, "X": 0, "Y": 0}


def _get_mask_spell(obs_non_spatial, mask_spatial, cfg) -> np.array:
    """Builds binary spell mask.

    This mask has shape of `(NUM_SPELLS,)` and is 1 if spell at that index can
    be cast, 0 otherwise.
    """
    spell_mask = np.zeros(cfg.obs_mask_spell_shape)

    for spell_index in range(cfg.spells.num_spells):
        if cfg.architecture.non_spatial_orthogonal_enabled:
            # Spell can be cast iff:
            # - waiting time is 1, meaning player has enough mana/cooldown expired, AND
            # - spatial mask is 1 for at least one spatial coordinate for that spell
            if obs_non_spatial[spell_index] == 1 and np.any(
                mask_spatial[spell_index, :]
            ):
                spell_mask[spell_index] = 1
        else:
            if np.isclose(obs_non_spatial[spell_index], 0) and np.any(
                mask_spatial[spell_index, :]
            ):
                spell_mask[spell_index] = 1

    return spell_mask


def _only_lanes_with_friendly_creeps(obs_spatial, mask_spatial_for_spell, cfg) -> None:
    for lane in range(cfg.map.num_lanes):
        lane_start = lane * cfg.map.num_bins
        next_lane_start = (lane + 1) * cfg.map.num_bins
        # Check if there are any of owner units on that lane
        if not np.any(obs_spatial[lane, :, : cfg.spells.num_units]):
            # Do not cast rage in this lane.
            mask_spatial_for_spell[lane_start:next_lane_start] = 0


def _only_lanes_with_enemy_creeps(obs_spatial, mask_spatial_for_spell, cfg) -> None:
    for lane in range(cfg.map.num_lanes):
        lane_start = lane * cfg.map.num_bins
        next_lane_start = (lane + 1) * cfg.map.num_bins
        # Check if there are any of enemy units on that lane
        if not np.any(obs_spatial[lane, :, cfg.spells.num_units :]):
            # Do not cast rage in this lane.
            mask_spatial_for_spell[lane_start:next_lane_start] = 0


DYNAMIC_SPATIAL_MASK = {
    Spell.RAGE: _only_lanes_with_friendly_creeps,
}


def _get_mask_spatial_dynamic(obs_spatial, cfg) -> np.array:
    """Builds binary spatial mask.

    Spatial mask has shape of `(NUM_SPELLS, NUM_LANES * NUM_BINS)` and defines,
    for spell at index `i`, and coordinate `k`, whether spell can be cast at that
    point on the map. If value at `(i, k)` is 1, spell can be cast there, otherwise
    it cannot be cast there.

    Spatial mask is comprised of two parts that are combined by conjunction -
    static part and dynamic part:
    * static part is always the same and is defined by casting rules for
    certain spells; e.g. you cannot cast minions (with some exceptions) past
    your own half of the map.
    * dynamic part depends on current spatial observation - for example,
    `Spell.RAGE` should only be cast on lanes with your minions
    """
    if not cfg.spatial_mask_dynamic_enabled:
        # Just return plain static mask.
        return cfg.mask_spatial_static

    mask_spatial = cfg.mask_spatial_static.copy()

    for spell_idx in range(cfg.spells.num_spells):
        spell = cfg.spells.idx_to_spell(spell_idx)
        if spell in DYNAMIC_SPATIAL_MASK:
            DYNAMIC_SPATIAL_MASK[spell](obs_spatial, mask_spatial[spell_idx], cfg)

    return mask_spatial


def _flip_obs_spatial(obs_spatial, cfg) -> np.array:
    """Flips spatial observations from left side to right side."""

    # Flip bin coordinates.
    obs_spatial_flipped_coordinates = np.flip(obs_spatial, 1)

    # Change the roles of own/opponents units.
    obs_spatial_flipped = np.concatenate(
        (
            obs_spatial_flipped_coordinates[:, :, cfg.spells.num_units :],
            obs_spatial_flipped_coordinates[:, :, : cfg.spells.num_units],
        ),
        axis=2,
    )

    return obs_spatial_flipped


def _get_spell_nonspatial(player_data, cfg) -> np.array:
    """Builds spell part of nonspatial observation vector.

    This spell nonspatial observation is of shape `(NUM_SPELLS,)`, and its value
    at index i is:
    * `-30` if i-th spell is not in hand, or not in deck
    * `0` if i-th spell can be played right now
    * `N` if i-th spell can be played in N seconds from right now

    Scheduled spells cannot be cast at that moment.
    """
    spell_wait_times = -30 * np.ones(cfg.spells.num_spells)

    # Noop is always available (in hand), and can be cast at any time.
    for spell in cfg.spells.enabled_spells:
        if spell.is_noop:
            spell_idx = cfg.spells.spell_to_idx(spell)
            spell_wait_times[spell_idx] = 0

    for card in player_data["CardsInHand"]:
        if card["IsScheduled"]:
            # Scheduled cards cannot be cast until they are resolved.
            continue

        spell_index = cfg.spells.spell_to_idx(card["SpellType"])
        if spell_wait_times[spell_index] >= 0:
            # Duplicate spells in hand, meaning we already have valid value.
            continue
        spell_wait_times[spell_index] = card["SecondsUntilAvailable"]

    return spell_wait_times


def _get_spell_nonspatial_orthogonal(player_data, deck_spells, cfg) -> np.array:
    """Builds spell part of nonspatial observation vector.

    This spell nonspatial observation has 3 separate dimensions for each spell:
    * `spells_in_deck` is 1 at index i if i-th spell is in agent's deck
    * `spells_in_hand` is 1 at index i if i-th spell is currently in hand
    * `spells_wait_time` is 0 if spell cannot be cast in next 30 seconds or more,
    and linearly moves to 1 when spell can be cast at this time. This depends on
    agent's mana or spell-specific cooldown.

    Scheduled spells cannot be cast at that moment.

    These three vectors are concatenated into a single vector of shape
    `(3 * NUM_SPELLS)` and returned.
    """
    spells_wait_time = np.zeros(cfg.spells.num_spells)
    spells_in_deck = np.zeros(cfg.spells.num_spells)
    spells_in_hand = np.zeros(cfg.spells.num_spells)

    for spell in cfg.spells.enabled_spells:
        spell_idx = cfg.spells.spell_to_idx(spell)
        if spell.is_noop:
            # Noop spells are always available - in deck, in hand and can be
            # played
            spells_in_deck[spell_idx] = 1
            spells_in_hand[spell_idx] = 1
            spells_wait_time[spell_idx] = 1
        else:
            if spell == Spell.DRAW_CARD:
                # Draw card spell is always in deck
                spells_in_deck[spell_idx] = 1
            else:
                spells_in_deck[spell_idx] = float(spell in deck_spells)

        card_in_hand = None
        for card in player_data["CardsInHand"]:
            if card["SpellType"] == int(spell) and not card["IsScheduled"]:
                card_in_hand = card
                spells_in_hand[spell_idx] = 1
                break

        if card_in_hand is not None:
            normalized_secs_until_available = (
                card_in_hand["SecondsUntilAvailable"]
                / cfg.spells.max_seconds_until_available
            )
            wait_time = 1 - normalized_secs_until_available
            spells_wait_time[spell_idx] = np.clip(wait_time, 0, 1)

    return np.concatenate((spells_wait_time, spells_in_hand, spells_in_deck))


def battle_state_to_observation(
    battle_state, left_deck, right_deck, cfg
) -> Tuple[dict, dict]:
    """Builds observation space from provided `battle_state`.

    This function returns a tuple with observations for left and right player,
    in that order. Observation is a dict with `spatial`, `non_spatial`,
    `mask_spell` and `mask_spatial' elements.

    Non-spatial observation vector has shape of `(NUM_SPELLS+3,)`, namely and
    in this order:
      - `NUM_SPELLS` times until available for each defined spell (in seconds);
      spells that player does not have in deck or in hand have infinite waiting
      time - if orthogonal non-spatial obs are enabled, this is `3 * NUM_SPELLS`
      - own castle health (in percent)
      - opponent castle health (in percent)
      - battle time (in percent of max battle time)

    Spatial observation vector has shape of `(NUM_LANES, NUM_BINS, 2 * NUM_UNITS)`.
    Values in this vector correspond to cummulative health percentage of units
    of at the same lane (Y coordinate), occupying the same bin, and having the
    same type. This is repeated twice, for own and opponent units, in this order.

    Spell mask is a binary mask with shape `(NUM_SPELLS,)` that says whether spell
    at index `i` can be cast - value of the mask is 1 for that index, 0 otherwise.

    Spatial mask is a binary mask with shape `(NUM_SPELLS, NUM_LANES * NUM_BINS)`
    that says whether spell at index `i` can be cast at coordinate pair `(lane, bin)`.
    Value of the mask is 1 at `(i, lane * NUM_BINS + bin)` if that is the case, 0
    otherwise.
    """
    obs_non_spatial = [
        np.zeros(cfg.obs_non_spatial_shape),
        np.zeros(cfg.obs_non_spatial_shape),
    ]

    obs_spatial_aux = np.zeros(cfg.obs_spatial_shape)

    left_deck = [entry.spell for entry in left_deck.minions + left_deck.spells]
    right_deck = [entry.spell for entry in right_deck.minions + right_deck.spells]

    for player in ["LeftPlayer", "RightPlayer"]:
        player_data = battle_state[player]
        player_id = 0 if player == "LeftPlayer" else 1
        deck = left_deck if player == "LeftPlayer" else right_deck
        opponent = "RightPlayer" if player == "LeftPlayer" else "LeftPlayer"

        if cfg.architecture.non_spatial_orthogonal_enabled:
            obs_non_spatial[player_id][
                : 3 * cfg.spells.num_spells
            ] = _get_spell_nonspatial_orthogonal(player_data, deck, cfg)
        else:
            obs_non_spatial[player_id][: cfg.spells.num_spells] = _get_spell_nonspatial(
                player_data, cfg
            )

        # Own castle health percentage.
        obs_non_spatial[player_id][-3] = player_data["CastleHealthPercent"]
        # Opponent's castle health percentage.
        obs_non_spatial[player_id][-2] = battle_state[opponent]["CastleHealthPercent"]
        # Battle time in percentage of max battle time.
        obs_non_spatial[player_id][-1] = (
            battle_state["BattleTime"] / cfg.map.max_battle_time_seconds
        )

        for active_creep in player_data["ActiveCreeps"]:
            lane = cfg.map.y_to_lane(active_creep["Y"])
            bin = cfg.map.x_to_bin(active_creep["X"])
            unit_type = cfg.spells.unit_to_idx(active_creep["UnitType"])
            channel = unit_type + player_id * cfg.spells.num_units

            obs_spatial_aux[lane, bin, channel] += active_creep["CurrentHealthPercent"]

    # Do not normalize.
    obs_spatial = [obs_spatial_aux, _flip_obs_spatial(obs_spatial_aux, cfg)]

    obs_mask_spatial = [
        _get_mask_spatial_dynamic(obs_spatial[0], cfg),
        _get_mask_spatial_dynamic(obs_spatial[1], cfg),
    ]

    obs_mask_spell = [
        _get_mask_spell(obs_non_spatial[0], obs_mask_spatial[0], cfg),
        _get_mask_spell(obs_non_spatial[1], obs_mask_spatial[1], cfg),
    ]

    return (
        {
            "spatial": obs_spatial[0],
            "non_spatial": obs_non_spatial[0],
            "mask_spell": obs_mask_spell[0],
            "mask_spatial": obs_mask_spatial[0],
        },
        {
            "spatial": obs_spatial[1],
            "non_spatial": obs_non_spatial[1],
            "mask_spell": obs_mask_spell[1],
            "mask_spatial": obs_mask_spatial[1],
        },
    )


def is_move_available(battle_state) -> Tuple[bool, bool]:
    left_can_play = any(
        not card["IsScheduled"] and card["SecondsUntilAvailable"] <= 0
        for card in battle_state["LeftPlayer"]["CardsInHand"]
    )
    right_can_play = any(
        not card["IsScheduled"] and card["SecondsUntilAvailable"] <= 0
        for card in battle_state["RightPlayer"]["CardsInHand"]
    )
    return left_can_play, right_can_play


def _is_spawn_spell_available(obs, cfg) -> Tuple[bool, bool]:
    mask_spawn = cfg.spells.mask_spawn.copy()

    # For purpose of this function, noop is not considered a spawn spell.
    for spell in cfg.spells.enabled_spells:
        if spell.is_noop:
            mask_spawn[0, cfg.spells.spell_to_idx(spell)] = 0

    left_spawn_available = bool(np.sum(obs[0]["mask_spell"] * mask_spawn))
    right_spawn_available = bool(np.sum(obs[1]["mask_spell"] * mask_spawn))

    return left_spawn_available, right_spawn_available


def _is_nonspawn_spell_available(obs, cfg) -> Tuple[bool, bool]:
    mask_nonspawn = cfg.spells.mask_nonspawn.copy()

    # For purpose of this function, noop is not considered a nonspawn spell.
    for spell in cfg.spells.enabled_spells:
        if spell.is_noop:
            mask_nonspawn[0, cfg.spells.spell_to_idx(spell)] = 0

    left_nonspawn_available = bool(np.sum(obs[0]["mask_spell"] * mask_nonspawn))
    right_nonspawn_available = bool(np.sum(obs[1]["mask_spell"] * mask_nonspawn))

    return left_nonspawn_available, right_nonspawn_available


def is_battle_done(battle_state) -> bool:
    return battle_state["BattleState"] != "InProgress"


def get_winner(battle_state) -> Owner:
    if not is_battle_done(battle_state):
        return None
    return (
        Owner.LEFT_PLAYER
        if battle_state["BattleState"] == "LeftWon"
        else Owner.RIGHT_PLAYER
    )


def battle_state_to_info(
    battle_state: dict, spatial_buff, left_deck, right_deck, cfg
) -> dict:
    """
    Converts battle state in JSON form to observations that can be added to
    observation buffer, for both players.

    Also appends spatial observations to spatial buffer.
    """
    left_can_play, right_can_play = is_move_available(battle_state)
    ob, ob_flip = battle_state_to_observation(battle_state, left_deck, right_deck, cfg)

    # Stack spatial observations.
    ob["spatial"], ob_flip["spatial"] = spatial_buff.append(
        ob["spatial"], ob_flip["spatial"]
    )

    left_spawn_available, right_spawn_available = _is_spawn_spell_available(
        (ob, ob_flip), cfg
    )
    left_nonspawn_available, right_nonspawn_available = _is_nonspawn_spell_available(
        (ob, ob_flip), cfg
    )

    return {
        "o": ob,
        "o_flip": ob_flip,
        "left_can_play": left_can_play,
        "right_can_play": right_can_play,
        "left_spawn_available": left_spawn_available,
        "right_spawn_available": right_spawn_available,
        "left_spell_available": left_nonspawn_available,
        "right_spell_available": right_nonspawn_available,
        "battle_time": battle_state["BattleTime"],
        "battle_state": battle_state["BattleState"],
    }


def get_client_action(
    action_policy_output, battle_state, cfg, player="LeftPlayer"
) -> dict:
    """Converts policy action output to dict client action."""
    if action_policy_output is None:
        return NOOP_ACTION

    spell_type = int(cfg.spells.idx_to_spell(action_policy_output["spell"][0]))

    card_index_to_play = -1

    for card in battle_state[player]["CardsInHand"]:
        if (
            card["SpellType"] == spell_type
            and card["SecondsUntilAvailable"] <= 0
            and not card["IsScheduled"]
        ):
            card_index_to_play = int(card["CardIndex"])

    lane = int(action_policy_output["spatial"][0] // cfg.map.num_bins)
    coordinate = int(action_policy_output["spatial"][0] % cfg.map.num_bins)

    y = cfg.map.lane_to_y(lane)
    x = cfg.map.bin_to_x(coordinate)

    # Flip X coordinate for right player so it's absolute (we flipped spatial
    # observations so that the model sees it as if it were playing on the left
    # side, so we do the reverse here).
    if player == "RightPlayer":
        x = -x

    return {"CardIndex": card_index_to_play, "X": x, "Y": y}
