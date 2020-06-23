"""
This module contains most of things related to training process and the game itself.
"""

import logging
import os

from .cfg import TrainingCfg  # noqa:F401
from .decks import Deck, DeckEntry, DeckRepository  # noqa:F401
from .enums import Brain, Owner, Spell, Unit  # noqa:F401
from .plan import Plans  # noqa:F401
from .rewards import Rewards  # noqa:F401


def setup_logging(log_path, level=logging.INFO):
    os.makedirs(log_path, exist_ok=True)

    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_path, "train.log")),
    ]
    fmt = (
        "[%(asctime)s.%(msecs)03d] %(levelname)s {%(filename)s:%(lineno)d}"
        + "%(name)s - %(message)s"
    )
    logging.basicConfig(level=level, handlers=handlers, format=fmt)
