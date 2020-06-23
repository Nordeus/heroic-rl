"""
Contains classes and methods for working with decks.
"""

import csv
import logging
import math
from collections import OrderedDict, namedtuple

from .enums import Spell

logger = logging.getLogger(__name__)

Deck = namedtuple("Deck", ["arena_level", "minions", "spells"])

DeckEntry = namedtuple("DeckEntry", ["spell", "level_increment"])


class DeckRepository(object):
    """Deck repository is a collection of decks, sorted by their arena level.

    Deck repository can be created from a CSV file.
    """

    @classmethod
    def from_csv(cls, csv_path, cfg):
        def _parse_spell(spell_str):
            spell_name = (
                spell_str.strip().upper().replace(" ", "").replace("SPELL_", "")
            )
            if not spell_name:
                # Empty spell string = empty slot in deck
                return None

            for spell in Spell:
                stripped_name = spell.name.replace("SPAWNUNIT_", "")
                if spell_name == stripped_name:
                    return spell
            else:
                raise ValueError("unknown spell %s" % spell_name)

        def _parse_deck(row):
            """Row format: arena_level,spawn1,spawn2,...,spawn12,spell1,spell2,..."""
            arena_level = float(row[0])
            minions = []
            spells = []
            for idx in range(12):
                minion = _parse_spell(row[idx + 1])
                if minion is not None:
                    minions.append(
                        DeckEntry(
                            minion,
                            cfg.spells.get_level_increment(minion.rarity, arena_level),
                        )
                    )
            for idx in range(2):
                spell = _parse_spell(row[idx + 1 + 12])
                # TODO evolve level
                if spell is not None:
                    spells.append(
                        DeckEntry(
                            spell,
                            cfg.spells.get_level_increment(spell.rarity, arena_level),
                        )
                    )

            if len(minions) < 6:
                raise ValueError("too few minions in deck, 6 required: %s" % minions)

            return Deck(arena_level=arena_level, minions=minions, spells=spells)

        def _can_play(deck):
            for entry in deck.minions + deck.spells:
                if entry.spell not in cfg.spells.enabled_spells:
                    return False
            return True

        repo = cls()
        with open(csv_path, newline="",) as f:
            reader = csv.reader(f)
            # Skip header
            next(reader, None)
            for row in csv.reader(f):
                deck = _parse_deck(row)
                if _can_play(deck):
                    repo.add_deck(deck)

        logger.info(
            "Loaded deck repository from %s: %d arena levels, %d decks",
            csv_path,
            len(repo._decks_by_arena_level),
            len(repo._all_decks),
        )
        return repo

    def __init__(self):
        self._decks_by_arena_level = OrderedDict()
        self._all_decks = []

    def add_deck(self, deck):
        arena_level = int(math.floor(deck.arena_level))
        if arena_level in self._decks_by_arena_level:
            self._decks_by_arena_level[arena_level].append(deck)
        else:
            self._decks_by_arena_level[arena_level] = [deck]

        self._all_decks.append(deck)

    def get_decks(self, arena_level=None):
        """Get a random deck for given arena level.

        If `arena_level` is `None`, all decks are considered.
        """
        if arena_level is None:
            return self._all_decks

        arena_level = int(math.floor(arena_level))
        if arena_level not in self._decks_by_arena_level:
            raise ValueError("No such arena level: %s" % arena_level)

        return self._decks_by_arena_level[arena_level]
