import logging
import os
import os.path as osp
import re
import shutil

import numpy as np

from ..utils.mpi_tools import proc_id

logger = logging.getLogger("heroic.agent")


class AgentSaver:
    """Agent saver can save and restore training agent state."""

    @classmethod
    def extract_agent_id(cls, restore_path):
        matches = re.search("agent_([0-9]+)", restore_path)
        return int(matches.group(1)) if matches else None

    @classmethod
    def extract_epoch(cls, restore_path):
        matches = re.search("simple_save([0-9]+)", restore_path)
        return int(matches.group(1)) if matches else None

    @classmethod
    def extract_cfg_path(cls, restore_path):
        model_dir = osp.dirname(restore_path.rstrip(osp.sep))
        return osp.abspath(osp.join(model_dir, "..", "training_cfg.yml"))

    def __init__(self, output_dir, capacity=1000, frequency=10, delta=0.5):
        """
        Initialize AgentSaver.

        :param output_dir: path to experiment directory
        :param capacity: max number of saved models per agent.
        :param frequency: save frequency.
        :param delta: real number in [0, 1], affects random saved agent sampling.
        """
        self.output_dir = output_dir
        self.capacity = capacity
        self.frequency = frequency
        self.delta = delta

        # Maps agent to list of saved epochs.
        self.saved_epochs = {}

        # Read existing saved checkpoints.
        for agent_dir in os.listdir(output_dir):
            agent_path = osp.join(output_dir, agent_dir)
            if not (osp.isdir(agent_path) and agent_dir.startswith("agent_")):
                continue
            agent_id = int(agent_dir.split("_")[1])
            agent_epochs = self._load_agent_epochs(agent_path)
            self.saved_epochs[agent_id] = agent_epochs
            if proc_id() == 0:
                logger.info(
                    "Found %d checkpoints for Agent(id=%d)", len(agent_epochs), agent_id
                )

    def save_with_frequency(self, agent):
        if proc_id() != 0:
            return
        if agent.state["epochs"] % self.frequency != 0:
            return
        self.save(agent)

    def save(self, agent):
        if proc_id() != 0:
            return
        state_dict = agent.state
        epoch = state_dict["epochs"]
        if agent.id not in self.saved_epochs:
            self.saved_epochs[agent.id] = []

        history = self.saved_epochs[agent.id]
        if len(history) > 0 and epoch == history[-1]:
            # Already saved.
            return
        while len(history) >= self.capacity:
            epoch_to_remove = history.pop(0)
            path_to_remove = self._get_model_path(agent, epoch_to_remove)
            if osp.isdir(path_to_remove):
                shutil.rmtree(path_to_remove)
                agent.logger.log(
                    "Removed saved model (epoch=%d) for %s" % (epoch_to_remove, agent)
                )

        try:
            if proc_id() == 0:
                if not osp.exists(self._get_agent_path(agent)):
                    os.makedirs(self._get_agent_path(agent))

            agent.logger.save_state_for_training(state_dict, itr=epoch)
            history.append(epoch)
            agent.logger.log("Saved epoch %d for %s" % (epoch, agent))
        except Exception:
            logger.error("Failed to save epoch %d for %s", epoch, agent, exc_info=True)

    def get_latest_model_path(self, agent):
        if agent.id not in self.saved_epochs:
            raise ValueError("no saved models for %s" % agent)
        history = self.saved_epochs[agent.id]
        return self._get_model_path(agent, history[len(history) - 1])

    def get_random_model_path(self):
        """
        Get random model that was saved in epoch in certain range.

        Model epoch range: `[max(min_epoch, delta * max_epoch), max_epoch]`.

        :return: path to the model.
        """
        model_paths = [
            self._get_model_path(agent_id, epoch)
            for agent_id, history in self.saved_epochs.items()
            for i, epoch in enumerate(history)
            if i >= int(self.delta * len(history))
        ]
        if len(model_paths) <= 0:
            raise ValueError("no saved models")

        return np.random.choice(model_paths)

    def has_saved_models(self, agent):
        return agent.id in self.saved_epochs and len(self.saved_epochs[agent.id]) > 0

    def _get_agent_path(self, agent):
        agent_id = agent if isinstance(agent, int) else agent.id
        return osp.join(self.output_dir, "agent_%d" % agent_id)

    def _get_model_path(self, agent, epoch):
        return osp.join(self._get_agent_path(agent), "simple_save%d" % epoch)

    def _load_agent_epochs(self, path):
        epochs = []
        for checkpoint in os.listdir(path):
            checkpoint_path = osp.join(path, checkpoint)
            if not os.path.isdir(checkpoint_path):
                continue
            epoch = self.extract_epoch(checkpoint)
            if epoch is None:
                continue
            epochs.append(epoch)

        return list(sorted(epochs))
