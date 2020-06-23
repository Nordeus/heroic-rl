"""
Classes and functions regarding inference with trained policies.
"""

import hashlib
import json
import logging
import uuid
from collections import namedtuple

from gym.utils import seeding

from ..agent import InferenceAgent
from ..gym_heroic.client import HeroicClient, Player
from ..train import Brain, Owner, obs
from .state import InferenceState

logger = logging.getLogger(__name__)


Battle = namedtuple("Battle", ["id", "steps", "winner", "state_actions"])


def pick_random_decks(cfg, np_random):
    all_decks = cfg.decks.get_decks()
    left_deck = all_decks[np_random.choice(len(all_decks))]

    # Pick opponent decks of similar arena level.
    matching_decks = cfg.decks.get_decks(arena_level=left_deck.arena_level)
    right_deck = matching_decks[np_random.choice(len(matching_decks))]
    return left_deck, right_deck


def simulate_one_battle(client, agent, adversary_brain, np_random, with_replay=False):
    """
    Simulates one battle on server, with agent as left player.

    Decks are picked at random.

    :param client: training service client
    :type client: heroic_rl.gym_heroic.client.HeroicClient
    :param agent: inference agent that will play as left player
    :type agent: heroic_rl.train.agents.InferenceAgent
    :param np_random: Numpy random instance, acting as source of randomness for
                      picking decks and battle IDs

    :returns: a simulated Battle
    :rtype: Battle
    """
    battle_id = uuid.UUID(bytes=np_random.bytes(16))
    left_deck, right_deck = pick_random_decks(agent.cfg, np_random)
    castle_level = agent.cfg.spells.get_castle_level(left_deck.arena_level)
    left_player = Player(castle_level=castle_level, deck=left_deck, brain=Brain.DUMMY,)
    right_player = Player(
        castle_level=castle_level, deck=right_deck, brain=adversary_brain,
    )

    state = InferenceState(agent, battle_id, left_deck, right_deck, Owner.LEFT_PLAYER)

    # Create a new battle on server.
    observation = client.create_new_battle(
        battle_id,
        left_player=left_player,
        right_player=right_player,
        capture_replay=with_replay,
    )

    # Simulate battle on server.
    while not obs.is_battle_done(observation):
        if obs.is_move_available(observation)[0]:
            next_action = state.step(observation)
            state_hash = hashlib.sha1(
                json.dumps(observation).encode("utf-8")
            ).hexdigest()
            logger.info(
                "Battle(id=%s, time=%s), State(hash=%s), Next action: %s",
                battle_id,
                observation["BattleTime"],
                state_hash,
                next_action,
            )
        else:
            # This should generally only happen on first move.
            next_action = obs.NOOP_ACTION
        observation = client.step_battle(battle_id, next_action)

    winner = obs.get_winner(observation)

    state_actions = None
    if with_replay:
        # Get replay.
        state_actions = [
            (obs_actions["Observation"], obs_actions["LeftAction"] or obs.NOOP_ACTION)
            for obs_actions in client.get_replay(battle_id)
        ]

    return Battle(
        id=battle_id, steps=state.steps, winner=winner, state_actions=state_actions
    )


def run(
    model_path: str,
    host: str,
    port: int,
    num_battles: int,
    seed: int,
    opponent_brain: Brain,
):
    """
    Entrypoint for CLI command.
    """
    import tensorflow as tf

    # Turn off TF log spam.
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    import time

    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        format="%(name)s [%(levelname)s] - %(message)s",
    )

    left_agent = InferenceAgent()
    left_agent.initialize()
    left_agent.restore(model_path)

    client = HeroicClient(host=host, port=port, use_sessions=True)
    np_random, _ = seeding.np_random(seed)

    battles_won = 0
    battles_simulated = 0

    def get_win_rate(won, total):
        return won / total * 100.0 if total > 0 else 0

    start_time = time.time()
    for _ in range(num_battles):
        try:
            battle = simulate_one_battle(client, left_agent, opponent_brain, np_random)
            battles_simulated += 1
            if battle.winner == Owner.LEFT_PLAYER:
                battles_won += 1

            if battles_simulated % 10 == 0:
                logger.info(
                    "%d battles simulated so far (win rate: %.2f%%)...",
                    battles_simulated,
                    get_win_rate(battles_won, battles_simulated),
                )
        except Exception:
            logger.error("Failed to simulate battle", exc_info=True)

    logger.info(
        "Simulated %d battles in %.3fs, win rate: %.2f%%",
        battles_simulated,
        time.time() - start_time,
        get_win_rate(battles_won, battles_simulated),
    )
