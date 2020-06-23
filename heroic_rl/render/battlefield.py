import curses
import math
from itertools import zip_longest

from ..train.enums import Owner, Unit


def prettify_time(fptime):
    mins, secs = int(fptime) // 60, int(fptime) % 60
    if mins == 0:
        return "%02ds" % secs
    else:
        return "%dm%02ds" % (mins, secs)


class Creep:
    """Represents single creep on battleground with a char."""

    CHARS = {
        Unit.SWORDSMAN: "#",
        Unit.ARCHER: "^",
        Unit.SKELETON: "%",
        Unit.REAPER: "$",
        Unit.TREANT: "T",
        Unit.FIREIMP: "F",
        Unit.BRUTE: "*",
        Unit.CHARGER: "/",
        Unit.WATERELEMENTAL: "@",
        Unit.EXECUTIONER: "E",
        Unit.SILVERRANGER: "{",
        Unit.ALCHEMIST: "a",
        Unit.COMMANDER: "C",
        Unit.GOBLINMARKSMAN: "G",
        Unit.JUGGERNAUT: "J",
        Unit.PRIMALSPIRIT: "P",
        Unit.RAVENOUSSCOURGE: "x",
        Unit.ROLLINGROCKS: "o",
        Unit.SHADOWHUNTRESS: "[",
        Unit.SHIELDBEARER: "|",
        Unit.STONEELEMENTAL: "W",
        Unit.VALKYRIE: "L",
        Unit.VIPER: "V",
        Unit.WISPMOTHER: "H",
        Unit.STONEELEMENTALSPAWN: "w",
    }

    SHORT_NAMES = {
        Unit.SWORDSMAN: "Swordsman",
        Unit.ARCHER: "Archer",
        Unit.SKELETON: "Skeleton",
        Unit.REAPER: "Reaper",
        Unit.TREANT: "Treant",
        Unit.FIREIMP: "FireImp",
        Unit.BRUTE: "Brute",
        Unit.CHARGER: "Charger",
        Unit.WATERELEMENTAL: "WaterEle",
        Unit.EXECUTIONER: "Executionr",
        Unit.SILVERRANGER: "SilverRng",
        Unit.ALCHEMIST: "Alchemist",
        Unit.COMMANDER: "Commander",
        Unit.GOBLINMARKSMAN: "GoblinShot",
        Unit.JUGGERNAUT: "Juggernaut",
        Unit.PRIMALSPIRIT: "PrimSpirit",
        Unit.RAVENOUSSCOURGE: "RavScourge",
        Unit.ROLLINGROCKS: "Rocks",
        Unit.SHADOWHUNTRESS: "ShadowHunt",
        Unit.SHIELDBEARER: "ShieldBear",
        Unit.STONEELEMENTAL: "StoneEle",
        Unit.VALKYRIE: "Valkyrie",
        Unit.VIPER: "Viper",
        Unit.WISPMOTHER: "WispMom",
        Unit.STONEELEMENTALSPAWN: "StoneSpawn",
    }

    def __init__(self, scr, xpos, unit_type, health_perc, owner):
        self.scr = scr
        self.xpos = xpos
        self.unit_type = unit_type
        self.health_perc = health_perc
        self.owner = owner

    def draw(self, y, x):
        char = self.CHARS[self.unit_type]
        color = (
            curses.color_pair(1)
            if self.owner == Owner.RIGHT_PLAYER
            else curses.color_pair(2)
        )
        self.scr.addstr(y, x, char, color)


class Lane:

    KIND_TOP = 1
    KIND_MID = 0
    KIND_BTM = -1

    X_MIN = -5.0
    X_MAX = 5.0

    def __init__(self, scr, kind, top=0, left=0, width=70, height=5):
        self.scr = scr
        self.creeps = []
        self.kind = int(kind)
        self.top = top
        self.left = left
        self.width = width
        self.height = height

    def get_bin(self, x):
        bin = (x - self.X_MIN) / (self.X_MAX - self.X_MIN) * self.width
        return max(math.ceil(bin) - 1, 0)

    def add_creep(self, creep):
        self.creeps.append(creep)

    @property
    def y(self):
        if self.kind == self.KIND_TOP:
            return self.top + self.height // 2
        elif self.kind == self.KIND_MID:
            return self.top + self.height // 2 + self.height
        elif self.kind == self.KIND_BTM:
            return self.top + self.height // 2 + 2 * self.height

    def get_creeps_at(self, bin):
        return (c for c in self.creeps if self.get_bin(c.xpos) == bin)

    def get_lane_ys(self):
        center = self.y
        ys = [center]
        for i in range(1, self.height // 2):
            ys.append(center + i)
            ys.append(center - i)
        return ys

    def render(self):
        # Draw upper border
        self.scr.move(self.y - self.height // 2, self.left)
        for _ in range(self.width):
            self.scr.addch("-", curses.A_DIM)

        # Draw lower border
        self.scr.move(self.y + self.height // 2, self.left)
        for _ in range(self.width):
            self.scr.addch("-", curses.A_DIM)

        for i in range(self.width):
            ys = self.get_lane_ys()
            for y, creep in zip_longest(ys, self.get_creeps_at(i)):
                if not y:
                    # More creeps than slots
                    break
                x = self.left + i
                if creep:
                    creep.draw(y, x)
                else:
                    self.scr.addch(y, x, " ")


class Map:

    LEGEND_HEIGHT = 10

    def __init__(self, scr, state, top=0, left=0, bottom=0, right=0):
        self.scr = scr
        height, width = scr.getmaxyx()

        # Subtract margins
        height -= top + bottom
        width -= left + right

        # Subtract legend height
        height -= self.LEGEND_HEIGHT

        self.legend_scr = self.scr.derwin(self.LEGEND_HEIGHT, width, top + height, left)

        lane_width = width
        lane_height = height // 3
        if lane_height % 2 == 0:
            lane_height -= 1

        self.lanes = [
            Lane(
                scr,
                Lane.KIND_BTM,
                top=top,
                left=left,
                width=lane_width,
                height=lane_height,
            ),
            Lane(
                scr,
                Lane.KIND_MID,
                top=top,
                left=left,
                width=lane_width,
                height=lane_height,
            ),
            Lane(
                scr,
                Lane.KIND_TOP,
                top=top,
                left=left,
                width=lane_width,
                height=lane_height,
            ),
        ]

        self.add_creeps(state["LeftPlayer"]["ActiveCreeps"], Owner.LEFT_PLAYER)
        self.add_creeps(state["RightPlayer"]["ActiveCreeps"], Owner.RIGHT_PLAYER)
        self.battle_time = prettify_time(state["BattleTime"])
        self.battle_id = str(state["BattleId"])
        self.status = str(state["BattleState"])

    def render_legend(self):
        _, x = self.legend_scr.getmaxyx()
        self.legend_scr.addstr(2, 0, "Legend", curses.A_UNDERLINE)
        # Rows start after Legend header and an empty row.
        row_start_y = 4
        rows = self.LEGEND_HEIGHT - row_start_y

        all_creeps = list(Creep.CHARS.keys())
        cols = len(all_creeps) // rows + 1
        col_width = self.legend_scr.getmaxyx()[1] // cols

        unit_index = 0
        for row in range(rows):
            row_y = row_start_y + row
            for col in range(cols):
                col_x = col * col_width
                unit = all_creeps[unit_index]
                unit_desc = Creep.SHORT_NAMES[unit]
                unit_char = Creep.CHARS[unit]
                unit_str = "%s: %s" % (unit_desc.ljust(10), unit_char)
                self.legend_scr.addstr(row_y, col_x, unit_str[:col_width])
                unit_index += 1
                if unit_index >= len(all_creeps):
                    return

    def render_battle_id(self):
        battle_id_str = "ID: " + self.battle_id
        self.legend_scr.addstr(0, 0, battle_id_str)

    def render_status(self):
        _, x = self.legend_scr.getmaxyx()
        status_str = "Status:"
        status_val_str = self.status.rjust(11)
        status_len = len(status_str) + len(status_val_str)
        attrs = curses.A_NORMAL if self.status == "InProgress" else curses.A_STANDOUT
        self.legend_scr.addstr(0, x - status_len, status_str)
        self.legend_scr.addstr(" " * (len(status_val_str) - len(self.status)))
        self.legend_scr.addstr(self.status, attrs)

    def render_time(self):
        _, x = self.legend_scr.getmaxyx()
        time_str = "Time:"
        time_val_str = self.battle_time.rjust(11)
        time_len = len(time_str) + len(time_val_str)

        self.legend_scr.addstr(1, x - time_len, time_str + time_val_str)

    def add_creeps(self, creep_states, owner):
        for creep_state in creep_states:
            lane = self.lanes[int(creep_state["Y"]) + 1]
            creep = Creep(
                self.scr,
                creep_state["X"],
                creep_state["UnitType"],
                creep_state["CurrentHealthPercent"],
                owner,
            )
            lane.add_creep(creep)

    def render(self):
        for lane in self.lanes:
            lane.render()

        self.render_battle_id()
        self.render_status()
        self.render_legend()
        self.render_time()
