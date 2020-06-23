"""
Training plans are used to layout training schedule. They are comprised of
one or more stages, and these stages are advanced when certain conditions are
met. Each training stage represents a certain adversary type - i.e. kind of AI
brain that should be used for opponent player. This way, we can design complex
training plans for our training agents.
"""

import logging
from abc import ABC, abstractmethod
from collections import namedtuple

from ..train.enums import Brain

logger = logging.getLogger("heroic.train")


class Condition(ABC):
    """Conditions need to be met in order for training plan to be advanced.

    Conditions are defined for each training stage. Training plan is kept at that
    stage until all stage conditions are met.
    """

    @abstractmethod
    def is_met(self, epoch, epoch_win_rate):
        """This method should return `True` when plan should be advanced."""
        return False

    @abstractmethod
    def reset(self, epoch):
        """This method resets internal state of condition when stage becomes active."""
        pass

    @staticmethod
    def all(conditions):
        class CompositeCondition(Condition):
            def is_met(self, epoch, epoch_win_rate):
                return all(
                    condition.is_met(epoch, epoch_win_rate) for condition in conditions
                )

            def reset(self, epoch):
                for condition in conditions:
                    condition.reset(epoch)

        return CompositeCondition()

    @staticmethod
    def always_met():
        class AlwaysMetCondition(Condition):
            def is_met(self, epoch, epoch_win_rate):
                return True

        return AlwaysMetCondition()


class AtLeastNumEpochsCondition(Condition):
    """Condition that is met after specified `num_epochs` pass."""

    def __init__(self, num_epochs):
        self._num_epochs = num_epochs
        self.reset(None)

    def reset(self, epoch):
        self._first_epoch = epoch

    def is_met(self, epoch, epoch_win_rate):
        return epoch - self._first_epoch >= self._num_epochs


class MinWinRateCondition(Condition):
    """
    Condition that is met when specified `min_win_rate` is achieved.

    Win rate condition needs to be met for each epoch in past `num_epochs`.
    """

    def __init__(self, min_win_rate, num_epochs):
        self._min_wr = min_win_rate
        self._num_epochs = num_epochs
        self.reset(None)

    def reset(self, epoch):
        self._wr_history = []

    def is_met(self, epoch, epoch_win_rate):
        self._wr_history.append(epoch_win_rate)

        # Keep only num_epochs past win rates.
        if len(self._wr_history) > self._num_epochs:
            self._wr_history.pop(0)

        # Check for minimum history length.
        if len(self._wr_history) < self._num_epochs:
            return False

        # Need to achieve at least minimum win rate.
        return all(win_rate >= self._min_wr for win_rate in self._wr_history)


class ConsistentPerformanceCondition(Condition):
    """
    Condition that is met when win rate difference gets bigger than threshold.

    For example, win rate from 5 epochs ago and current win rate need to be within 10%
    difference (with latter being greater than former).
    """

    def __init__(self, wr_difference_threshold, num_epochs):
        self._diff_threshold = wr_difference_threshold
        self._num_epochs = num_epochs
        self.reset(None)

    def reset(self, epoch):
        self._wr_history = []

    def is_met(self, epoch, epoch_win_rate):
        self._wr_history.append(epoch_win_rate)

        # Keep only num_epochs past win rates.
        if len(self._wr_history) > self._num_epochs:
            self._wr_history.pop(0)

        # Check for minimum history length.
        if len(self._wr_history) < self._num_epochs:
            return False

        oldest_wr = self._wr_history[0]

        # Need to be positive about this.
        if epoch_win_rate < oldest_wr:
            return False

        return epoch_win_rate - oldest_wr < self._diff_threshold


class TrainingStage:
    def __init__(self, brain, condition):
        self.brain = brain
        if isinstance(condition, (list, tuple)):
            self._condition = Condition.all(condition)
        elif isinstance(condition, Condition):
            self._condition = condition
        else:
            raise ValueError("condition")

    def on_activated(self, epoch):
        self._condition.reset(epoch)

    def should_advance_plan(self, epoch, epoch_win_rate):
        return self._condition.is_met(epoch, epoch_win_rate)


Loop = namedtuple("Loop", ["start", "end", "count"])


class TrainingPlan:
    def __init__(self, *stages):
        self._loops = []
        self._stages = []
        self._curr_stage_idx = 0
        self._curr_loop_count = 0

        # Parse stages and tokens.
        token_stack = []
        next_stage_idx = 0
        for stage in stages:
            if isinstance(stage, tuple):
                if stage[0] == "for":
                    count = stage[1]
                    token_stack.append((next_stage_idx, count))
                elif stage[0] == "endfor":
                    # Will throw IndexError if endfor is mismatched.
                    start_idx, count = token_stack.pop()
                    self._loops.append(
                        Loop(start=start_idx, end=next_stage_idx, count=count)
                    )
                else:
                    raise ValueError("unknown token: %s" % stage[0])
            elif isinstance(stage, TrainingStage):
                self._stages.append(stage)
                next_stage_idx += 1
            else:
                raise ValueError("initialize plan with stages and tokens")
        if len(token_stack) > 0:
            raise ValueError("mismatched tokens on stack: %s" % token_stack)

        self.curr_stage.on_activated(1)

    @property
    def curr_stage(self):
        return self._stages[self._curr_stage_idx]

    def _is_last_stage(self):
        return self._curr_stage_idx == len(self._stages) - 1

    @property
    def _curr_loop(self):
        for loop in self._loops:
            if self._curr_stage_idx >= loop.start and self._curr_stage_idx < loop.end:
                return loop
        return None

    def _should_advance(self, epoch, epoch_win_rate):
        if self._curr_loop is None and self._is_last_stage():
            return False

        return self.curr_stage.should_advance_plan(epoch, epoch_win_rate)

    def next_stage(self, epoch, epoch_win_rate):
        if self._should_advance(epoch, epoch_win_rate):
            old_stage = self.curr_stage
            old_loop = self._curr_loop

            self._curr_stage_idx += 1
            curr_loop = self._curr_loop

            if old_loop is not None and curr_loop != old_loop:
                # At loop boundary.
                self._curr_loop_count += 1
                if self._curr_loop_count < old_loop.count:
                    # Reset loop.
                    self._curr_stage_idx = old_loop.start
                else:
                    # Remove loop at exit.
                    self._loops.remove(old_loop)
                    self._curr_loop_count = 0

            self._curr_stage_idx = min(len(self._stages) - 1, self._curr_stage_idx)

            curr_stage = self.curr_stage
            if old_stage != curr_stage:
                curr_stage.on_activated(epoch)

        return self.curr_stage


class Plans:
    """Contains various plan functions that return specific training plans."""

    # fmt: off
    MIN_WIN_RATES = [
        # Random | Utility | Lookahead
          0.9,     0.9,      0.8, # level 1 # NOQA
          0.9,     0.8,      0.8, # level 2 # NOQA
          0.9,     0.8,      0.8, # level 3 # NOQA
          0.9,     0.8,      0.7, # level 4 # NOQA
          0.8,     0.8,      0.6, # level 5 # NOQA
          0.8,     0.7,      0.6, # level 6 # NOQA
          0.7,     0.6,      0.5, # level 7 # NOQA
          0.7,     0.6,      0.5, # level 8 # NOQA
          0.7,     0.6,      0.5  # level 9 # NOQA
    ]
    # fmt: on

    @classmethod
    def all(cls):
        return [
            k
            for k, v in cls.__dict__.items()
            if k != "all"
            and (isinstance(v, staticmethod) or isinstance(v, classmethod))
        ]

    @classmethod
    def brains(cls):
        adversaries = [
            adversary
            for level in zip(
                Brain.random_brains(), Brain.utility_brains(), Brain.lookahead_brains(),
            )
            for adversary in level
        ]
        conditions = [
            [
                AtLeastNumEpochsCondition(10),
                MinWinRateCondition(cls.MIN_WIN_RATES[i], 5),
                ConsistentPerformanceCondition(0.1, 5),
            ]
            for i, _ in enumerate(adversaries)
        ]
        stages = [
            TrainingStage(adversary[0], adversary[1], cond)
            for adversary, cond in zip(adversaries, conditions)
        ]
        return TrainingPlan(*stages)

    @staticmethod
    def lookahead():
        return TrainingPlan(
            TrainingStage(Brain.LOOKAHEAD_9, AtLeastNumEpochsCondition(1))
        )

    @staticmethod
    def utility():
        return TrainingPlan(
            TrainingStage(Brain.UTILITY_9, AtLeastNumEpochsCondition(1))
        )

    @staticmethod
    def selfplay():
        stages = [
            # Bootstrapping
            TrainingStage(Brain.UTILITY_9, AtLeastNumEpochsCondition(10)),
            # Main training loop
            ("for", 20),
            TrainingStage(Brain.DUMMY, AtLeastNumEpochsCondition(50)),
            TrainingStage(Brain.UTILITY_9, AtLeastNumEpochsCondition(1)),
            # Lookahead evaluation
            TrainingStage(Brain.LOOKAHEAD_9, AtLeastNumEpochsCondition(1)),
            ("endfor",),
            ("for", 20),
            TrainingStage(Brain.DUMMY, AtLeastNumEpochsCondition(50)),
            TrainingStage(Brain.UTILITY_9, AtLeastNumEpochsCondition(1)),
            # Lookahead evaluation
            TrainingStage(Brain.LOOKAHEAD_9, AtLeastNumEpochsCondition(1)),
            ("endfor",),
            # Main training loop
            ("for", 20),
            TrainingStage(Brain.DUMMY, AtLeastNumEpochsCondition(50)),
            TrainingStage(Brain.UTILITY_9, AtLeastNumEpochsCondition(1)),
            # Lookahead evaluation
            TrainingStage(Brain.LOOKAHEAD_9, AtLeastNumEpochsCondition(1)),
            ("endfor",),
        ]

        return TrainingPlan(*stages)

    @staticmethod
    def selfplay_with_bootstrap():
        stages = [
            # Bootstrapping
            TrainingStage(
                Brain.UTILITY_9,
                [
                    AtLeastNumEpochsCondition(20),
                    MinWinRateCondition(0.5, 5),
                    ConsistentPerformanceCondition(0.1, 5),
                ],
            ),
            # Main training loop
            ("for", 50),
            TrainingStage(Brain.DUMMY, AtLeastNumEpochsCondition(20)),
            TrainingStage(Brain.UTILITY_9, AtLeastNumEpochsCondition(2)),
            ("endfor",),
            # Lookahead evaluation
            TrainingStage(Brain.LOOKAHEAD_9, AtLeastNumEpochsCondition(5)),
            # Main training loop
            ("for", 50),
            TrainingStage(Brain.DUMMY, AtLeastNumEpochsCondition(20)),
            TrainingStage(Brain.UTILITY_9, AtLeastNumEpochsCondition(2)),
            ("endfor",),
            # Lookahead evaluation
            TrainingStage(Brain.LOOKAHEAD_9, AtLeastNumEpochsCondition(5)),
            # Main training loop
            ("for", 50),
            TrainingStage(Brain.DUMMY, AtLeastNumEpochsCondition(20)),
            TrainingStage(Brain.UTILITY_9, AtLeastNumEpochsCondition(2)),
            ("endfor",),
            # Lookahead evaluation
            TrainingStage(Brain.LOOKAHEAD_9, AtLeastNumEpochsCondition(5)),
        ]

        return TrainingPlan(*stages)

    @staticmethod
    def utility_then_lookahead():
        return TrainingPlan(
            TrainingStage(
                Brain.UTILITY_9,
                [MinWinRateCondition(0.5, 5), ConsistentPerformanceCondition(0.1, 5)],
            ),
            TrainingStage(Brain.LOOKAHEAD_9, AtLeastNumEpochsCondition(1)),
        )
