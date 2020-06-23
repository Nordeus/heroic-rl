import logging
from collections import namedtuple

import requests

logger = logging.getLogger(__name__)


Player = namedtuple("Player", ["castle_level", "deck", "brain"])


class HeroicClient:
    def __init__(self, host="127.0.0.1", port=8080, use_sessions=True):
        logger.info("Initializing HeroicClient")
        self.host = host
        self.port = port
        # Use session for connection pooling
        if use_sessions:
            self.session = requests.Session()
        else:
            self.session = None
        self.create_battle_url = self.url + "/battle"
        self.step_battle_url = self.url + "/battle/%s/step"
        self.get_replay_url = self.url + "/battle/%s/replay"
        self.cookies = {}

    @property
    def session_id(self):
        return self.cookies.get("SessionId", None)

    @property
    def url(self):
        return "http://{host}:{port}".format(host=self.host, port=self.port)

    def _create_start_episode_payload(
        self, battle_id, left_player: Player, right_player: Player, capture_replay: bool
    ):
        def _to_entries_json(entries):
            return [
                {
                    "SpellId": int(entry.spell),
                    "LevelIncrement": int(entry.level_increment),
                }
                for entry in entries
            ]

        return {
            "BattleId": str(battle_id),
            "LeftDeck": {
                "SpellcasterId": 7,
                "Deck": _to_entries_json(left_player.deck.minions),
                "Spells": _to_entries_json(left_player.deck.spells),
                "DeckName": "LeftDeck",
            },
            "RightDeck": {
                "SpellcasterId": 7,
                "Deck": _to_entries_json(right_player.deck.minions),
                "Spells": _to_entries_json(right_player.deck.spells),
                "DeckName": "RightDeck",
            },
            "LeftBrainDifficulty": int(left_player.brain),
            "RightBrainDifficulty": int(right_player.brain),
            "LeftCastleLevel": int(left_player.castle_level),
            "RightCastleLevel": int(right_player.castle_level),
            "CaptureReplay": bool(capture_replay),
        }

    def create_new_battle(
        self, battle_id, left_player: Player, right_player: Player, capture_replay=False
    ):
        payload = self._create_start_episode_payload(
            battle_id, left_player, right_player, capture_replay
        )
        if self.session:
            response = self.session.post(self.create_battle_url, json=payload)
        else:
            response = requests.post(
                self.create_battle_url, json=payload, cookies=self.cookies
            )
        response.raise_for_status()
        logger.debug("New battle created on server (battleId=%s)", battle_id)
        state = response.json()
        if self.session_id is None:
            self.cookies["SessionId"] = response.cookies["SessionId"]
            logger.debug("Assigned session ID %s", self.session_id)
        return state

    def step_battle(self, battle_id, left_action=None, right_action=None):
        payload = {}
        if left_action:
            payload["NextActionLeft"] = left_action
        if right_action:
            payload["NextActionRight"] = right_action
        if self.session:
            response = self.session.post(self.step_battle_url % battle_id, json=payload)
        else:
            response = requests.post(
                self.step_battle_url % battle_id, json=payload, cookies=self.cookies
            )
        response.raise_for_status()
        return response.json()

    def get_replay(self, battle_id):
        if self.session:
            response = self.session.get(self.get_replay_url % battle_id)
        else:
            response = requests.get(self.get_replay_url % battle_id)
        response.raise_for_status()
        return response.json()
