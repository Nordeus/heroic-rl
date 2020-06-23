"""
Contains enumerations that are used throughout the codebase. Also sets up some
static enumeration data (like spell rarity, spawn spell -> unit spawned mapping
etc.
"""

import enum


@enum.unique
class Owner(enum.IntEnum):
    LEFT_PLAYER = 1
    RIGHT_PLAYER = 2

    @property
    def side(self):
        return "LeftPlayer" if self == Owner.LEFT_PLAYER else "RightPlayer"


@enum.unique
class Rarity(enum.IntEnum):
    UNDEFINED = 0

    COMMON = 1
    RARE = 2
    EPIC = 3
    LEGENDARY = 4
    MYTHIC = 5


@enum.unique
class CastingStrategy(enum.IntEnum):
    # Cast at any point on the map (think Meteor)
    ENTIRE_MAP = 0
    # Cast on any one lane
    SINGLE_LANE = 1
    # Cast on controlled area of the map - default minion casting strategy
    # For each lane, cast as far as your furthest minions on any lane, up to
    # first enemy minion on that lane, capped by half-width of lane.
    CONTROLLED_AREA = 2
    # Cast position does not matter.
    DOES_NOT_MATTER = 3


@enum.unique
class Spell(enum.IntEnum):
    # Spawn spells
    SPAWNUNIT_SWORDSMAN = 3
    SPAWNUNIT_ARCHERS = 6001
    SPAWNUNIT_REAPER = 23
    SPAWNUNIT_UNDEADHORDE = 6
    SPAWNUNIT_TREANT = 6005
    SPAWNUNIT_FIREIMP = 12001
    SPAWNUNIT_BRUTE = 9
    SPAWNUNIT_CHARGER = 12015
    SPAWNUNIT_WATERELEMENTAL = 6025
    SPAWNUNIT_EXECUTIONER = 6106
    SPAWNUNIT_SILVERRANGER = 12002
    SPAWNUNIT_ALCHEMIST = 6107
    SPAWNUNIT_COMMANDER = 4071
    SPAWNUNIT_GOBLINMARKSMAN = 4063
    SPAWNUNIT_JUGGERNAUT = 6015
    SPAWNUNIT_PRIMALSPIRIT = 6111
    SPAWNUNIT_RAVENOUSSCOURGE = 12020
    SPAWNUNIT_ROLLINGROCKS = 12010
    SPAWNUNIT_SHADOWHUNTRESS = 6116
    SPAWNUNIT_SHIELDBEARER = 4066
    SPAWNUNIT_STONEELEMENTAL = 6013
    SPAWNUNIT_UNDEADARMY = 6003
    SPAWNUNIT_VALKYRIE = 6020
    SPAWNUNIT_VIPER = 12008
    SPAWNUNIT_WISPMOTHER = 106

    # Standard spells
    METEOR = 314
    RAGE = 317

    # Custom spells
    DRAW_CARD = 200
    NOOP_1S = -1
    NOOP_2S = -2
    NOOP_3S = -3
    NOOP_4S = -4
    NOOP_5S = -5
    NOOP_6S = -6
    NOOP_7S = -7
    NOOP_8S = -8

    @property
    def rarity(self):
        return self._rarities[self]

    @property
    def casting_strategy(self):
        if self in self._custom_casting_strategies:
            return self._custom_casting_strategies[self]
        if self.is_spawn:
            # default casting strategy for minions
            return CastingStrategy.CONTROLLED_AREA
        else:
            # default casting strategy for spells
            return CastingStrategy.ENTIRE_MAP

    @property
    def is_spawn(self):
        """Spawn spells can be drawn from deck, put in hand and they spawn minions."""
        return self.name.startswith("SPAWN")

    @property
    def is_noop(self):
        return int(self) < 0

    @property
    def units_spawned(self):
        if self in self._units_spawned:
            return self._units_spawned[self]
        else:
            return None


Spell._rarities = {
    Spell.SPAWNUNIT_SWORDSMAN: Rarity.COMMON,
    Spell.SPAWNUNIT_ARCHERS: Rarity.COMMON,
    Spell.SPAWNUNIT_SHIELDBEARER: Rarity.COMMON,
    Spell.SPAWNUNIT_UNDEADHORDE: Rarity.COMMON,
    Spell.SPAWNUNIT_WATERELEMENTAL: Rarity.COMMON,
    Spell.SPAWNUNIT_RAVENOUSSCOURGE: Rarity.COMMON,
    Spell.SPAWNUNIT_ROLLINGROCKS: Rarity.COMMON,
    Spell.SPAWNUNIT_UNDEADARMY: Rarity.COMMON,
    Spell.SPAWNUNIT_FIREIMP: Rarity.COMMON,
    Spell.SPAWNUNIT_REAPER: Rarity.RARE,
    Spell.SPAWNUNIT_BRUTE: Rarity.RARE,
    Spell.SPAWNUNIT_EXECUTIONER: Rarity.RARE,
    Spell.SPAWNUNIT_SILVERRANGER: Rarity.RARE,
    Spell.SPAWNUNIT_ALCHEMIST: Rarity.RARE,
    Spell.SPAWNUNIT_COMMANDER: Rarity.RARE,
    Spell.SPAWNUNIT_GOBLINMARKSMAN: Rarity.RARE,
    Spell.SPAWNUNIT_VALKYRIE: Rarity.RARE,
    Spell.SPAWNUNIT_VIPER: Rarity.RARE,
    Spell.SPAWNUNIT_CHARGER: Rarity.EPIC,
    Spell.SPAWNUNIT_TREANT: Rarity.EPIC,
    Spell.SPAWNUNIT_JUGGERNAUT: Rarity.EPIC,
    Spell.SPAWNUNIT_PRIMALSPIRIT: Rarity.EPIC,
    Spell.SPAWNUNIT_SHADOWHUNTRESS: Rarity.EPIC,
    Spell.SPAWNUNIT_STONEELEMENTAL: Rarity.EPIC,
    Spell.SPAWNUNIT_WISPMOTHER: Rarity.EPIC,
    Spell.METEOR: Rarity.COMMON,
    Spell.RAGE: Rarity.COMMON,
    Spell.DRAW_CARD: Rarity.UNDEFINED,
    Spell.NOOP_1S: Rarity.UNDEFINED,
    Spell.NOOP_2S: Rarity.UNDEFINED,
    Spell.NOOP_3S: Rarity.UNDEFINED,
    Spell.NOOP_4S: Rarity.UNDEFINED,
    Spell.NOOP_5S: Rarity.UNDEFINED,
    Spell.NOOP_6S: Rarity.UNDEFINED,
    Spell.NOOP_7S: Rarity.UNDEFINED,
    Spell.NOOP_8S: Rarity.UNDEFINED,
}

for spell in Spell:
    if spell not in Spell._rarities:
        raise ValueError("UNDEFINED rarity for %s" % spell)

Spell._custom_casting_strategies = {
    Spell.RAGE: CastingStrategy.SINGLE_LANE,
    Spell.DRAW_CARD: CastingStrategy.DOES_NOT_MATTER,
    Spell.NOOP_1S: CastingStrategy.DOES_NOT_MATTER,
    Spell.NOOP_2S: CastingStrategy.DOES_NOT_MATTER,
    Spell.NOOP_3S: CastingStrategy.DOES_NOT_MATTER,
    Spell.NOOP_4S: CastingStrategy.DOES_NOT_MATTER,
    Spell.NOOP_5S: CastingStrategy.DOES_NOT_MATTER,
    Spell.NOOP_6S: CastingStrategy.DOES_NOT_MATTER,
    Spell.NOOP_7S: CastingStrategy.DOES_NOT_MATTER,
    Spell.NOOP_8S: CastingStrategy.DOES_NOT_MATTER,
}


@enum.unique
class Unit(enum.IntEnum):
    SWORDSMAN = 6
    ARCHER = 79
    REAPER = 36
    SKELETON = 11
    TREANT = 83
    FIREIMP = 12001
    BRUTE = 38
    CHARGER = 12012
    WATERELEMENTAL = 100
    EXECUTIONER = 105
    SILVERRANGER = 12002
    ALCHEMIST = 107
    COMMANDER = 44
    GOBLINMARKSMAN = 39
    JUGGERNAUT = 94
    PRIMALSPIRIT = 102
    RAVENOUSSCOURGE = 12017
    ROLLINGROCKS = 12010
    SHADOWHUNTRESS = 117
    SHIELDBEARER = 42
    STONEELEMENTAL = 92
    VALKYRIE = 99
    VIPER = 12008
    WISPMOTHER = 56
    # STONEELEMENTAL spawns this on deathrattle
    STONEELEMENTALSPAWN = 136


Spell._units_spawned = {
    Spell.SPAWNUNIT_SWORDSMAN: Unit.SWORDSMAN,
    Spell.SPAWNUNIT_ARCHERS: Unit.ARCHER,
    Spell.SPAWNUNIT_SHIELDBEARER: Unit.SHIELDBEARER,
    Spell.SPAWNUNIT_UNDEADHORDE: Unit.SKELETON,
    Spell.SPAWNUNIT_WATERELEMENTAL: Unit.WATERELEMENTAL,
    Spell.SPAWNUNIT_RAVENOUSSCOURGE: Unit.RAVENOUSSCOURGE,
    Spell.SPAWNUNIT_ROLLINGROCKS: Unit.ROLLINGROCKS,
    Spell.SPAWNUNIT_UNDEADARMY: Unit.SKELETON,
    Spell.SPAWNUNIT_FIREIMP: Unit.FIREIMP,
    Spell.SPAWNUNIT_REAPER: Unit.REAPER,
    Spell.SPAWNUNIT_BRUTE: Unit.BRUTE,
    Spell.SPAWNUNIT_EXECUTIONER: Unit.EXECUTIONER,
    Spell.SPAWNUNIT_SILVERRANGER: Unit.SILVERRANGER,
    Spell.SPAWNUNIT_ALCHEMIST: Unit.ALCHEMIST,
    Spell.SPAWNUNIT_COMMANDER: Unit.COMMANDER,
    Spell.SPAWNUNIT_GOBLINMARKSMAN: Unit.GOBLINMARKSMAN,
    Spell.SPAWNUNIT_VALKYRIE: Unit.VALKYRIE,
    Spell.SPAWNUNIT_VIPER: Unit.VIPER,
    Spell.SPAWNUNIT_CHARGER: Unit.CHARGER,
    Spell.SPAWNUNIT_TREANT: Unit.TREANT,
    Spell.SPAWNUNIT_JUGGERNAUT: Unit.JUGGERNAUT,
    Spell.SPAWNUNIT_PRIMALSPIRIT: Unit.PRIMALSPIRIT,
    Spell.SPAWNUNIT_SHADOWHUNTRESS: Unit.SHADOWHUNTRESS,
    Spell.SPAWNUNIT_STONEELEMENTAL: [Unit.STONEELEMENTAL, Unit.STONEELEMENTALSPAWN],
    Spell.SPAWNUNIT_WISPMOTHER: Unit.WISPMOTHER,
}


@enum.unique
class Brain(enum.IntEnum):
    """Brains represent kind and difficulty of AI opponents."""

    UNDEFINED_DIFFICULTY = 0

    UTILITY_1 = 1
    UTILITY_2 = 2
    UTILITY_3 = 3
    UTILITY_4 = 4
    UTILITY_5 = 5
    UTILITY_6 = 6
    UTILITY_7 = 7
    UTILITY_8 = 8
    UTILITY_9 = 9
    LOOKAHEAD_1 = 14
    LOOKAHEAD_2 = 15
    LOOKAHEAD_3 = 16
    LOOKAHEAD_4 = 17
    LOOKAHEAD_5 = 18
    LOOKAHEAD_6 = 19
    LOOKAHEAD_7 = 20
    LOOKAHEAD_8 = 21
    LOOKAHEAD_9 = 22
    RANDOM_1 = 27
    RANDOM_2 = 28
    RANDOM_3 = 29
    RANDOM_4 = 30
    RANDOM_5 = 31
    RANDOM_6 = 32
    RANDOM_7 = 33
    RANDOM_8 = 34
    RANDOM_9 = 35
    DUMMY = 40

    @classmethod
    def utility_brains(cls):
        return list(map(cls, range(cls.UTILITY_1, cls.UTILITY_9 + 1)))

    @classmethod
    def random_brains(cls):
        return list(map(cls, range(cls.RANDOM_1, cls.RANDOM_9 + 1)))

    @classmethod
    def lookahead_brains(cls):
        return list(map(cls, range(cls.LOOKAHEAD_1, cls.LOOKAHEAD_9 + 1)))
