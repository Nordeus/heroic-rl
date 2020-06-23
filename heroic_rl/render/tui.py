import curses
import logging
from functools import partial
from itertools import zip_longest

from gym.utils import seeding

from ..agent import InferenceAgent
from ..gym_heroic.client import HeroicClient
from ..inference.simulation import simulate_one_battle
from ..train import Brain, Owner, Spell
from .battlefield import Map, prettify_time

GAME_TICK_SECS = 0.2


SPELL_NAMES = {
    Spell.SPAWNUNIT_SWORDSMAN: "Swordsman",
    Spell.SPAWNUNIT_ARCHERS: "Archers",
    Spell.SPAWNUNIT_REAPER: "Reaper",
    Spell.SPAWNUNIT_UNDEADHORDE: "UndHorde",
    Spell.SPAWNUNIT_TREANT: "Treant",
    Spell.SPAWNUNIT_FIREIMP: "FireImp",
    Spell.SPAWNUNIT_BRUTE: "Brute",
    Spell.SPAWNUNIT_CHARGER: "Charger",
    Spell.SPAWNUNIT_WATERELEMENTAL: "WaterEle",
    Spell.SPAWNUNIT_EXECUTIONER: "Exec",
    Spell.SPAWNUNIT_SILVERRANGER: "SilverRng",
    Spell.SPAWNUNIT_ALCHEMIST: "Alch",
    Spell.SPAWNUNIT_COMMANDER: "Command",
    Spell.SPAWNUNIT_GOBLINMARKSMAN: "GoblinShot",
    Spell.SPAWNUNIT_JUGGERNAUT: "Jugger",
    Spell.SPAWNUNIT_PRIMALSPIRIT: "PrmSpirit",
    Spell.SPAWNUNIT_RAVENOUSSCOURGE: "RavScourge",
    Spell.SPAWNUNIT_ROLLINGROCKS: "Rocks",
    Spell.SPAWNUNIT_SHADOWHUNTRESS: "ShadowHunt",
    Spell.SPAWNUNIT_SHIELDBEARER: "Shield",
    Spell.SPAWNUNIT_STONEELEMENTAL: "StoneEle",
    Spell.SPAWNUNIT_UNDEADARMY: "UndArmy",
    Spell.SPAWNUNIT_VALKYRIE: "Valkyrie",
    Spell.SPAWNUNIT_VIPER: "Viper",
    Spell.SPAWNUNIT_WISPMOTHER: "Wisp",
    Spell.METEOR: "Meteor",
    Spell.RAGE: "Rage",
    Spell.DRAW_CARD: "DraW",
}


class BattleManager:
    """Handles battle simulation, play speed, replays, keeps track of win rate etc."""

    SPEED_SECS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 2, 4]
    SPEED_SECS_1X = GAME_TICK_SECS
    DIRECTION_FORWARD = 1
    DIRECTION_REVERSE = -1

    def __init__(self, left_agent, server, seed):
        host, port = server.split(":")
        self.client = HeroicClient(host=host, port=port, use_sessions=True)
        self.np_random, _ = seeding.np_random(seed)
        self.agent = left_agent

        self.current_battle = None
        self.speed_idx = self.SPEED_SECS.index(self.SPEED_SECS_1X)
        self.state_idx = 0
        self.direction = self.DIRECTION_FORWARD

        self.left_wins = 0
        self.total_battles = 0

    @property
    def speed_secs(self):
        """Playback speed in real time seconds per one game tick."""
        return self.SPEED_SECS[self.speed_idx]

    @property
    def speed_perc(self):
        """Playback speed in percentage of 1x speed."""
        return self.SPEED_SECS_1X / self.speed_secs * 100.0

    @property
    def win_rate(self):
        """Win rate of left player in [0f, 1f] range."""
        if self.total_battles == 0:
            return 0
        return self.left_wins / self.total_battles

    def increase_speed(self):
        self.speed_idx = max(0, self.speed_idx - 1)

    def decrease_speed(self):
        self.speed_idx = min(len(self.SPEED_SECS) - 1, self.speed_idx + 1)

    def reverse_direction(self):
        self.direction = -self.direction

    def rewind_to_start(self):
        if self.current_battle is None:
            raise ValueError("no battle in progress")
        self.state_idx = 0
        self.direction = self.DIRECTION_FORWARD

    def clamp_idx(self, idx):
        return max(min(idx, len(self.current_battle.state_actions) - 1), 0)

    def get_state_action_pair(self, idx):
        if self.current_battle is None:
            raise ValueError("no battle in progress")
        return self.current_battle.state_actions[self.clamp_idx(idx)]

    def next_state_action(self):
        if self.current_battle is None:
            raise ValueError("no battle in progress")
        state, action = self.get_state_action_pair(self.state_idx)
        self.state_idx = self.clamp_idx(self.state_idx + self.direction)

        # Inject battle ID in state.
        state["BattleId"] = str(self.current_battle.id)

        return (
            state,
            action,
            {
                "WinRate": self.win_rate,
                "TotalStates": len(self.current_battle.state_actions),
                "TotalSteps": self.current_battle.steps,
            },
        )

    def start_new_battle(self):
        self.current_battle = simulate_one_battle(
            self.client, self.agent, Brain.UTILITY_9, self.np_random, with_replay=True
        )
        self.total_battles += 1
        if self.current_battle.winner == Owner.LEFT_PLAYER:
            self.left_wins += 1
        self.rewind_to_start()


def render(scr, state):
    scr.box()

    # Set box margins
    battleground = Map(scr, state, top=1, left=1, bottom=1, right=1)
    battleground.render()

    scr.refresh()


def init_colors():
    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)

    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_WHITE)
    curses.init_pair(4, curses.COLOR_GREEN, curses.COLOR_WHITE)


def render_player_info(scr, state, action, owner, custom_data):
    height, width = scr.getmaxyx()

    y = 0
    color = 1 if owner == Owner.RIGHT_PLAYER else 2
    hicolor = 3 if owner == Owner.RIGHT_PLAYER else 4
    scr.addstr(y, 0, owner.side, curses.color_pair(color))
    scr.addstr(" %3d%% HP" % int(round(state["CastleHealthPercent"] * 100)))
    y += 1

    FULL_BLOCK = "█"
    EMPTY_BLOCK = "░"
    filled = int(state["CastleHealthPercent"] * width)

    scr.move(y, 0)
    for i in range(width):
        if i <= filled:
            scr.addstr(FULL_BLOCK, curses.color_pair(hicolor))
        else:
            scr.addstr(EMPTY_BLOCK, curses.color_pair(hicolor))

    def render_card(card):
        attrs = curses.A_DIM if card["SecondsUntilAvailable"] >= 1 else curses.A_NORMAL
        if card["IsScheduled"]:
            attrs |= curses.A_STANDOUT
        scr.addstr(
            " - %s (%s)"
            % (
                SPELL_NAMES[Spell(card["SpellType"])],
                prettify_time(card["SecondsUntilAvailable"]),
            ),
            attrs,
        )

    y += 1
    scr.addstr(y, 0, "Hand", curses.A_UNDERLINE)
    minions = [c for c in state["CardsInHand"] if c["CardIndex"] >= 0]
    for _, card in zip_longest(range(5), minions):
        y += 1
        scr.move(y, 0)
        if card:
            render_card(card)
        scr.clrtoeol()

    y += 1
    scr.addstr(y, 0, "Spells", curses.A_UNDERLINE)
    spells = [
        c for c in state["CardsInHand"] if c["CardIndex"] < 0 and c["CardIndex"] != -200
    ]
    for _, card in zip_longest(range(2), spells):
        y += 1
        scr.move(y, 0)
        if card:
            render_card(card)
        scr.clrtoeol()

    y += 1
    scr.addstr(y, 0, "Action", curses.A_UNDERLINE)
    if action is not None:
        client_action = action
        y += 1
        attrs = curses.A_NORMAL
        card_attrs = curses.A_STANDOUT
        if client_action["CardIndex"] == -1:
            action_desc = "None"
            attrs = curses.A_DIM
            card_attrs = curses.A_NORMAL
        else:
            card = next(
                c
                for c in state["CardsInHand"]
                if c["CardIndex"] == client_action["CardIndex"]
            )
            action_desc = SPELL_NAMES[card["SpellType"]]
        scr.addstr(y, 0, "- Card: ")
        scr.addstr(action_desc, card_attrs)
        scr.clrtoeol()

        y += 1
        scr.addstr(y, 0, "- Position: %.3f" % client_action["X"], attrs)
        scr.clrtoeol()

        y += 1
        lanes = ["Bottom", "Middle", "Top"]
        lane = lanes[int(client_action["Y"]) + 1]
        scr.addstr(y, 0, "- Lane: %s" % lane, attrs)
        scr.clrtoeol()

        y += 2
        scr.addstr(y, 0, "Custom Data", curses.A_UNDERLINE)
    if custom_data is not None:
        if "WinProbability" in custom_data:
            y += 1
            scr.addstr(
                y,
                0,
                "- Win Probability: %.2f%%" % (custom_data["WinProbability"] * 100.0),
            )
            scr.clrtoeol()
        if "WinRate" in custom_data:
            y += 1
            scr.addstr(
                y, 0, "- Overall Win Rate: %.2f%%" % (custom_data["WinRate"] * 100.0),
            )
            scr.clrtoeol()
        if "TotalStates" in custom_data:
            y += 1
            scr.addstr(y, 0, "- Total States: %d" % custom_data["TotalStates"])
            scr.clrtoeol()
        if "TotalSteps" in custom_data:
            y += 1
            scr.addstr(y, 0, "- Total Steps: %d" % custom_data["TotalSteps"])
            scr.clrtoeol()

    scr.refresh()


def render_status(scr, speed_secs, speed_perc, direction):
    # Show cursor at the end of status screen, before getch()
    scr.refresh()

    scr.timeout(int(1000 * speed_secs))
    k = scr.getch()
    scr.move(0, 0)
    dirstr = "" if direction == BattleManager.DIRECTION_FORWARD else " (reversed)"
    scr.addstr(0, 0, "Speed: %d%%%s" % (int(speed_perc), dirstr))
    scr.clrtoeol()
    scr.addstr(
        1,
        0,
        "(hit < to decrease speed, > to increase speed, x to reverse,",
        curses.A_DIM,
    )
    scr.addstr(
        2, 0, "r to rewind to start, n to start new battle, q to quit)", curses.A_DIM
    )
    return k


def render_loop(battle_mgr, stdscr):
    init_colors()
    stdscr.clear()

    width = curses.COLS - 1
    height = curses.LINES - 1

    xcenter = width // 2

    info_width = 30
    info_height = 30

    status_height = 3

    map_width = min(width - info_width * 2, 80)
    map_height = min(height - status_height, 45)
    map_y = 0
    map_x = xcenter - map_width // 2

    try:
        map_scr = stdscr.derwin(map_height, map_width, map_y, map_x)
        left_player_scr = stdscr.derwin(info_height, info_width, 0, 0)
        right_player_scr = stdscr.derwin(info_height, info_width, 0, width - info_width)
        status_scr = stdscr.derwin(status_height, map_width, map_y + map_height, map_x)
        status_scr.nodelay(True)
    except curses.error as e:
        raise RuntimeError(
            "Your terminal is too small to fit the interface, please expand it"
        ) from e

    while True:
        state, action, custom_data = battle_mgr.next_state_action()

        render(map_scr, state)
        render_player_info(
            left_player_scr, state["LeftPlayer"], action, Owner.LEFT_PLAYER, custom_data
        )
        render_player_info(
            right_player_scr, state["RightPlayer"], None, Owner.RIGHT_PLAYER, None
        )

        key = render_status(
            status_scr,
            battle_mgr.speed_secs,
            battle_mgr.speed_perc,
            battle_mgr.direction,
        )
        if key == ord("<"):
            battle_mgr.decrease_speed()
        elif key == ord(">"):
            battle_mgr.increase_speed()
        elif key == ord("x"):
            battle_mgr.reverse_direction()
        elif key == ord("r"):
            battle_mgr.rewind_to_start()
        elif key == ord("n"):
            battle_mgr.start_new_battle()
        elif key == ord("q"):
            return


def run(model_path, server, seed):
    import tensorflow as tf

    # Turn off TF log spam.
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    logging.basicConfig(
        level=logging.ERROR,
        handlers=[logging.StreamHandler()],
        format="%(name)s [%(levelname)s] - %(message)s",
    )

    left_agent = InferenceAgent()
    left_agent.initialize()
    left_agent.restore(model_path)

    battle_mgr = BattleManager(left_agent, server, seed)
    battle_mgr.start_new_battle()

    curses.wrapper(partial(render_loop, battle_mgr))
