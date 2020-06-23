"""
WSGI inference service.
"""

import hashlib
import json
import logging

from flask import Flask, jsonify, request
from werkzeug.serving import run_simple

from ..agent import InferenceAgent
from ..train import Deck, DeckEntry, Spell
from .state import InferenceState

app = Flask(__name__)

states = dict()
agent = None


def _create_deck(json_deck):
    return Deck(
        arena_level=1,
        minions=[
            DeckEntry(
                spell=Spell(int(entry["SpellId"])),
                level_increment=int(entry["LevelIncrement"]),
            )
            for entry in json_deck["Minions"]
        ],
        spells=[
            DeckEntry(
                spell=Spell(int(entry["SpellId"])),
                level_increment=int(entry["LevelIncrement"]),
            )
            for entry in json_deck["Spells"]
        ],
    )


@app.route("/<battle_id>", methods=["POST"])
def create(battle_id):
    if battle_id in states:
        return ("Battle(id=%s) already exists" % battle_id, 500)
    left_deck = _create_deck(request.json["LeftDeck"])
    right_deck = _create_deck(request.json["RightDeck"])
    states[battle_id] = InferenceState(
        agent, battle_id, left_deck, right_deck, request.json["AgentOwner"]
    )
    app.logger.info("Created Battle(id=%s)", battle_id)
    return ("", 201)


@app.route("/<battle_id>/think", methods=["POST"])
def think(battle_id):
    if battle_id not in states:
        return ("No such Battle(id=%s)" % battle_id, 404)
    state = states[battle_id]
    next_action = state.step(request.json["Observation"])
    state_json = json.dumps(request.json["Observation"])
    state_hash = hashlib.sha1(state_json.encode("utf-8")).hexdigest()
    app.logger.info(
        "Battle(id=%s, time=%s), State(hash=%s), Next action: %s",
        battle_id,
        request.json["Observation"]["BattleTime"],
        state_hash,
        json.dumps(next_action, sort_keys=True),
    )
    return jsonify(next_action)


@app.route("/<battle_id>", methods=["DELETE"])
def delete(battle_id):
    if battle_id not in states:
        return ("No such Battle(id=%s)" % battle_id, 400)
    del states[battle_id]
    app.logger.info("Battle(id=%s) cleared", battle_id)
    return ("", 204)


def run(model_path: str, bind_address: str, port: int):
    """
    Entrypoint for CLI command.
    """
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        format="%(name)s [%(levelname)s] - %(message)s",
    )

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    global agent
    agent = InferenceAgent()
    agent.initialize()
    agent.restore(model_path)

    app.logger.info("Inference server starting at %s:%d", bind_address, port)
    run_simple("localhost", port, app)
