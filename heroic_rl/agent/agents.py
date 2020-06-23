import logging
import os.path as osp

import joblib
import numpy as np
import tensorflow.compat.v1 as tf
from mpi4py import MPI

from ..algos import utils as algoutils
from ..algos.ppo_heroic import PPOBuffer, build_model, restore_model
from ..train import Brain, TrainingCfg
from ..utils.logx import EpochLogger
from ..utils.mpi_tools import broadcastpy, mpi_avg, proc_id
from .saver import AgentSaver

logger = logging.getLogger("heroic.agent")


class Agent:

    ID_GENERATOR = 0

    @classmethod
    def _next_id(cls):
        cls.ID_GENERATOR += 1
        return cls.ID_GENERATOR

    def _reset(self):
        # TF stuff.
        self.graph = None
        self.sess = None
        self.writer = None

        # Logger for epoch stats and logging.
        self.logger = None

        # Training plan.
        self.plan = None

        # Experience buffer.
        self.buf = None

        # Number of finished epochs.
        self.epochs = 0

        self.all_phs = None
        self.selfplay_epochs = 0

    def __init__(self, cfg, saver):
        self.cfg = cfg
        self.saver = saver
        self.id = self._next_id()
        self.adversary_agent = InferenceAgent()
        # Strong references to functions used for tf.pyfunc Tensors, so they
        # don't get GCed after graph is deleted.
        self.graph_fns = []

        self._reset()

    def _setup_tf_saver(self):
        """Sets up `logger` for storing the model and vars properly."""
        train_inputs = [
            "x_non_spatial_ph",
            "x_spatial_ph",
            "x_mask_spell_ph",
            "x_mask_spatial_ph",
            "x_if_spawn_spell_ph",
            "x_state_in_ph",
            "x_rnn_mask_ph",
            "x_batch_size_ph",
            "a_spell_ph",
            "a_spatial_ph",
            "adv_ph",
            "ret_ph",
            "logp_old_ph",
            "v_old_ph",
        ]
        train_outputs = [
            "pi_spell",
            "pi_spatial",
            "pi_spell_argmax",
            "pi_spatial_argmax",
            "v",
            "state_out",
            "pi_loss",
            "v_loss",
            "loss",
            "approx_ent",
            "approx_kl",
            "clipfrac",
            "logp_pi",
        ]
        infer_inputs = [
            "x_non_spatial_ph",
            "x_spatial_ph",
            "x_mask_spell_ph",
            "x_mask_spatial_ph",
            "x_if_spawn_spell_ph",
        ]
        infer_outputs = [
            "pi_spell_argmax",
            "pi_spatial_argmax",
        ]
        self.logger.setup_tf_saver(
            self.sess,
            train_inputs={k: self.model[k] for k in train_inputs},
            train_outputs={k: self.model[k] for k in train_outputs},
            infer_inputs={k: self.model[k] for k in infer_inputs},
            infer_outputs={k: self.model[k] for k in infer_outputs},
        )

    def _create_env(self):
        env = self.cfg.create_env()
        env.seed(self.cfg.seed)
        return env

    def _create_sess(self):
        """Sets up a fresh TF session."""
        gpu_options = tf.GPUOptions()
        gpu_options.allow_growth = True
        gpu_options.per_process_gpu_memory_fraction = 0.5
        return tf.Session(
            graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options)
        )

    def _restore_model(self, restore_path):
        return restore_model(self.sess, restore_path, self.id)

    def _create_model(self):
        return build_model(self.env, self.cfg,)

    def _create_buffer(self):
        return PPOBuffer(self.env.observation_space, self.env.action_space, self.cfg)

    def _reset_env(self, brain):
        return self.env.reset(left_brain=Brain.DUMMY, right_brain=brain)

    def _rollout(self, epoch, brain):
        o, info = self._reset_env(brain)
        r, d, ep_ret, ep_len, ep_win = 0, False, 0, 0, 0
        t = 0
        left_noop_spawn_expiry_time = 0
        left_noop_spell_expiry_time = 0
        right_noop_spawn_expiry_time = 0
        right_noop_spell_expiry_time = 0
        state_in_left = self.cfg.architecture.empty_rnn_state
        state_in_right = self.cfg.architecture.empty_rnn_state

        while t < self.cfg.local_steps_per_epoch:
            left_can_play_spawn = (
                info["battle_time"] >= left_noop_spawn_expiry_time
            ) and info["left_spawn_available"]
            left_can_play_spell = (
                info["battle_time"] >= left_noop_spell_expiry_time
            ) and info["left_spell_available"]

            a, a_right = None, None
            if info["left_can_play"] and (left_can_play_spawn or left_can_play_spell):
                o["if_spawn_spell"] = (
                    np.array([1.0]) if info["left_spawn_available"] else np.array([0.0])
                )
                o["state_in"] = state_in_left
                a, v_t, logp_t, state_out, summaries = self.sess.run(
                    [
                        {
                            "spell": self.model["pi_spell"],
                            "spatial": self.model["pi_spatial"],
                        },
                        self.model["v"],
                        self.model["logp_pi"],
                        self.model["state_out"],
                        self.model["summary_op"],
                    ],
                    feed_dict={
                        self.model["x_non_spatial_ph"]: o["non_spatial"][None, :],
                        self.model["x_spatial_ph"]: o["spatial"][None, :],
                        self.model["x_mask_spell_ph"]: o["mask_spell"][None, :],
                        self.model["x_mask_spatial_ph"]: o["mask_spatial"][None, :],
                        self.model["x_if_spawn_spell_ph"]: o["if_spawn_spell"][None, :],
                        self.model["x_state_in_ph"]: o["state_in"],
                        self.model["x_rnn_mask_ph"]: np.zeros(1),
                        self.model["x_batch_size_ph"]: 1,
                    },
                )
                # save and log
                self.buf.store(o, a, r, v_t, logp_t)
                self.logger.store(VVals=v_t)
                t += 1
                state_in_left = state_out
                spell = self.cfg.spells.idx_to_spell(a["spell"][0])
                if spell.is_noop:
                    # Noop spells are encoded as negative noop duration in secs
                    if left_can_play_spawn:
                        left_noop_spawn_expiry_time = info["battle_time"] - int(spell)
                    else:
                        left_noop_spell_expiry_time = info["battle_time"] - int(spell)
                if proc_id() == 0 and summaries:
                    self.writer.add_summary(summaries, global_step=epoch)

            right_can_play_spawn = (
                info["battle_time"] >= right_noop_spawn_expiry_time
            ) and info["right_spawn_available"]
            right_can_play_spell = (
                info["battle_time"] >= right_noop_spell_expiry_time
            ) and info["right_spell_available"]
            if (
                info["right_can_play"]
                and brain == Brain.DUMMY
                and (right_can_play_spawn or right_can_play_spell)
            ):
                info["o_flip"]["if_spawn_spell"] = (
                    np.array([1.0])
                    if info["right_spawn_available"]
                    else np.array([0.0])
                )
                info["o_flip"]["state_in"] = state_in_right
                a_right, _ = self.adversary_agent.get_next_action(info["o_flip"])
                spell = self.cfg.spells.idx_to_spell(a_right["spell"][0])
                if spell.is_noop:
                    if right_can_play_spawn:
                        right_noop_spawn_expiry_time = info["battle_time"] - int(spell)
                    else:
                        right_noop_spell_expiry_time = info["battle_time"] - int(spell)

            should_increment_ep_len = info["left_can_play"] and a is not None
            o, r, d, info = self.env.step(a, a_right)

            if d and info["battle_state"] == "LeftWon":
                ep_win += 1

            if should_increment_ep_len or d:
                ep_len += 1
                ep_ret += r

            terminal = d or (ep_len == self.cfg.max_ep_len)
            epoch_cutoff = t == self.cfg.local_steps_per_epoch
            if terminal or epoch_cutoff:
                # update state in if needed afterwards
                o["state_in"] = state_in_left

                # restart times of last noops
                left_noop_spawn_expiry_time = 0
                left_noop_spell_expiry_time = 0
                right_noop_spawn_expiry_time = 0
                right_noop_spell_expiry_time = 0

                # reset input states
                state_in_left = self.cfg.architecture.empty_rnn_state
                state_in_right = self.cfg.architecture.empty_rnn_state

                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = (
                    r
                    if d
                    else self.sess.run(
                        self.model["v"],
                        feed_dict={
                            self.model["x_non_spatial_ph"]: o["non_spatial"][None, :],
                            self.model["x_spatial_ph"]: o["spatial"][None, :],
                            self.model["x_mask_spell_ph"]: o["mask_spell"][None, :],
                            self.model["x_mask_spatial_ph"]: o["mask_spatial"][None, :],
                            self.model["x_state_in_ph"]: o["state_in"],
                            self.model["x_rnn_mask_ph"]: np.zeros(1),
                            self.model["x_batch_size_ph"]: 1,
                        },
                    )
                )
                self.buf.finish_path(last_val)
                if terminal:
                    self.logger.log(
                        "Episode done at %d/%d steps (took %d steps)"
                        % (t, self.cfg.local_steps_per_epoch, ep_len)
                    )
                    # Only save EpRet / EpLen / EpWinRate if trajectory finished.
                    self.logger.store(EpRet=ep_ret, EpLen=ep_len, EpWinRate=ep_win)
                    # Reset environment on terminal state reached.
                    o, info = self._reset_env(brain)
                    r, d, ep_ret, ep_len, ep_win = 0, False, 0, 0, 0
                else:
                    # Maximum number of steps per epoch was reached.
                    self.logger.log(
                        "Warning: trajectory cut off by epoch at %d steps." % ep_len
                    )

    def _update(self):
        num_batches = self.buf.create_batch_data()

        if self.cfg.architecture.rnn:
            # in case number of batches differs across processes, use the minimum
            num_batches_all = MPI.COMM_WORLD.gather(num_batches, root=0)
            num_batches_all = MPI.COMM_WORLD.scatter(
                [num_batches_all for _ in range(MPI.COMM_WORLD.Get_size())], root=0
            )
            num_batches = min(num_batches_all)
            if proc_id() == 0:
                print("Num batches all", num_batches_all)
                print("Min num batches", num_batches)

        inputs = [
            {k: v for k, v in zip(self.all_phs, self.buf.get_batch(batch_index),)}
            for batch_index in range(num_batches)
        ]
        # todo michalw: the statistics should be computed as running mean, not over
        # the first batch
        pi_l_old, v_l_old, ent = self.sess.run(
            [self.model["pi_loss"], self.model["v_loss"], self.model["approx_ent"]],
            feed_dict=inputs[0],
        )

        # Training
        if not self.cfg.architecture.unified_policy_value:
            for i in range(self.cfg.hyperparameters.train_pi_iters):
                kl_list = []
                for batch_index in range(num_batches):
                    _, kl = self.sess.run(
                        [self.model["train_pi"], self.model["approx_kl"]],
                        feed_dict=inputs[batch_index],
                    )
                    kl_list.append(kl)
                kl = np.mean(kl_list)
                kl = mpi_avg(kl)
                if kl > self.cfg.hyperparameters.target_kl:
                    self.logger.log(
                        "Early stopping at step %d due to reaching max kl." % i
                    )
                    break

            self.logger.store(StopIter=i)
            for _ in range(self.cfg.hyperparameters.train_v_iters):
                for batch_index in range(num_batches):
                    self.sess.run(self.model["train_v"], feed_dict=inputs[batch_index])

        else:
            for i in range(self.cfg.hyperparameters.train_pi_iters):
                kl_list = []
                for batch_index in range(num_batches):
                    _, kl = self.sess.run(
                        [self.model["train"], self.model["approx_kl"]],
                        feed_dict=inputs[batch_index],
                    )
                    kl_list.append(kl)
                kl = np.mean(kl_list)
                kl = mpi_avg(kl)
                if kl > self.cfg.hyperparameters.target_kl:
                    self.logger.log(
                        "Early stopping at step %d due to reaching max kl." % i
                    )
                    break
            self.logger.store(StopIter=i)

        # todo michalw: the statistics should be computed as running mean, not over
        # the first batch
        # Log changes from update
        pi_l_new, v_l_new, _, cf = self.sess.run(
            [
                self.model["pi_loss"],
                self.model["v_loss"],
                self.model["approx_kl"],
                self.model["clipfrac"],
            ],
            feed_dict=inputs[0],
        )
        self.logger.store(
            LossPi=pi_l_old,
            LossV=v_l_old,
            KL=kl,
            Entropy=ent,
            ClipFrac=cf,
            DeltaLossPi=(pi_l_new - pi_l_old),
            DeltaLossV=(v_l_new - v_l_old),
        )

    def _log_epoch(self, epoch):
        # Log info about epoch
        self.logger.log_tabular("Epoch", epoch)
        self.logger.log_tabular("EpRet", with_min_and_max=True)
        self.logger.log_tabular("EpWinRate", average_only=True)
        self.logger.log_tabular("EpLen", average_only=True)
        self.logger.log_tabular("VVals", with_min_and_max=True)
        self.logger.log_tabular("LossPi", average_only=True)
        self.logger.log_tabular("LossV", average_only=True)
        self.logger.log_tabular("DeltaLossPi", average_only=True)
        self.logger.log_tabular("DeltaLossV", average_only=True)
        self.logger.log_tabular("Entropy", average_only=True)
        self.logger.log_tabular("KL", average_only=True)
        self.logger.log_tabular("ClipFrac", average_only=True)
        self.logger.log_tabular("StopIter", average_only=True)

        if proc_id() == 0:
            scalar_summary = tf.Summary()
            for field in [
                "MinEpRet",
                "MaxEpRet",
                "AverageEpRet",
                "StdEpRet",
                "EpWinRate",
                "EpLen",
                "MinVVals",
                "MaxVVals",
                "AverageVVals",
                "StdVVals",
                "LossPi",
                "LossV",
                "DeltaLossPi",
                "DeltaLossV",
                "Entropy",
                "KL",
                "ClipFrac",
                "StopIter",
            ]:
                val = self.logger.log_current_row[field]
                scalar_summary.value.add(tag=field, simple_value=val)
            self.writer.add_summary(scalar_summary, global_step=epoch)

        self.last_epoch_wr = self.logger.log_current_row["EpWinRate"]
        self.logger.dump_tabular()

    def save(self):
        if proc_id() == 0:
            self.saver.save(self)
            self.writer.flush()

    def _initialize_adversary(self, brain):
        if brain != Brain.DUMMY:
            self.adversary_agent.deinitialize()
            self.selfplay_epochs = 0
            return

        if self.selfplay_epochs % 10 == 0:
            self.adversary_agent.deinitialize()
            restore_path = self.sync_random_restore_path()
            self.logger.log("Restoring next adversary agent from %s" % restore_path)
            self.adversary_agent.initialize()
            self.adversary_agent.restore(restore_path)

        self.selfplay_epochs += 1

    def run(self, epochs_max, restore_path=None):
        try:
            if restore_path and osp.isdir(restore_path):
                self._restore(restore_path)
            else:
                self._start_from_scratch()

            # Count variables
            var_counts = tuple(
                algoutils.count_vars(scope) for scope in ["Policy", "ValueFunction"]
            )
            self.logger.log("Number of parameters: pi: %d, v: %d" % var_counts)

            # TODO dimitrijer: no need to create each time, save shape of env and
            # create it in ctor
            self.buf = self._create_buffer()
            self._setup_tf_saver()
            # Need all placeholders in *this* order later (to zip with data from buffer)
            self.all_phs = [
                self.model[k]
                for k in [
                    "x_non_spatial_ph",
                    "x_spatial_ph",
                    "x_mask_spell_ph",
                    "x_mask_spatial_ph",
                    "x_if_spawn_spell_ph",
                    "x_state_in_ph",
                    "x_rnn_mask_ph",
                    "x_batch_size_ph",
                    "a_spell_ph",
                    "a_spatial_ph",
                    "adv_ph",
                    "ret_ph",
                    "logp_old_ph",
                    "v_old_ph",
                ]
            ]

            # Sync params across processes
            self.sess.run(self.model["sync_op"])

            training_stage = None
            num_epochs = max(self.cfg.epochs_per_agent, epochs_max - self.epochs)
            self.logger.log("Running %s for %d epochs" % (self, num_epochs))
            # Main loop: collect experience in env and update/log each epoch
            for epoch in range(self.epochs + 1, self.epochs + num_epochs + 1):
                next_stage = self.plan.next_stage(epoch, self.last_epoch_wr)
                if next_stage != training_stage:
                    self.logger.log(
                        "Advancing training plan for %s to adversary=%s at epoch %d (wr=%.3f)"  # noqa: b950
                        % (self, next_stage.brain, epoch, self.last_epoch_wr)
                    )
                training_stage = next_stage

                self._initialize_adversary(training_stage.brain)

                # Collect trajectories
                self._rollout(epoch, training_stage.brain)

                # Perform PPO update!
                self._update()

                # Reset experience buffer so we can start collecting next
                # rollout trajectories.
                self.buf.reset()

                # Log info about epoch
                self._log_epoch(epoch)

                # Update epochs done
                self.epochs = epoch

                # Save model
                if proc_id() == 0:
                    self.saver.save_with_frequency(self)
                    self.writer.flush()
        finally:
            self.adversary_agent.deinitialize()

    @property
    def state(self):
        return {
            "env": self.env,
            "epochs": self.epochs,
            "plan": self.plan,
            "last_epoch_wr": self.last_epoch_wr,
        }

    def _restore(self, restore_path):
        assert tf.get_default_graph() == self.graph
        assert tf.get_default_session() == self.sess
        vars_path = osp.join(restore_path, "vars.pkl")
        saved_state = joblib.load(vars_path)
        self.env = saved_state["env"]
        # Need new session IDs.
        self.env.reset_client()
        self.model = self._restore_model(restore_path)
        self.plan = saved_state["plan"]
        self.epochs = saved_state["epochs"]
        self.last_epoch_wr = saved_state["last_epoch_wr"]
        if proc_id() == 0:
            self.writer.add_graph(self.graph)
        self.logger.log("%s restored from %s" % (self, restore_path))

    def _start_from_scratch(self):
        assert tf.get_default_graph() == self.graph
        assert tf.get_default_session() == self.sess
        self.logger.save_config(self.cfg)
        self.env = self._create_env()
        self.model = self._create_model()
        self.plan = self.cfg.create_plan(self)
        self.epochs = 0
        self.last_epoch_wr = 0
        self.sess.run(tf.global_variables_initializer())
        if proc_id() == 0:
            self.writer.add_graph(self.graph)
        self.logger.log("%s starting from scratch" % self)

    def __enter__(self):
        self.graph = tf.Graph()
        if len(self.graph_fns) > 0:
            if not hasattr(self.graph, "_py_funcs_used_in_graph"):
                self.graph._py_funcs_used_in_graph = []
            for graph_fn in self.graph_fns:
                if graph_fn not in self.graph._py_funcs_used_in_graph:
                    self.graph._py_funcs_used_in_graph.append(graph_fn)
        self.sess = self._create_sess()

        logger_kwargs = self.cfg.get_agent_logger_kwargs(self)
        self.logger = EpochLogger(**logger_kwargs)
        if proc_id() == 0:
            self.writer = tf.summary.FileWriter(logger_kwargs["output_dir"], self.graph)
            self.writer.__enter__()

        self.sess.__enter__()
        self.logger.log("Opened TF session for %s" % self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        assert tf.get_default_graph() == self.graph
        assert tf.get_default_session() == self.sess
        if exc_type is None:
            for graph_fn in self.graph._py_funcs_used_in_graph:
                if graph_fn not in self.graph_fns:
                    self.graph_fns.append(graph_fn)

            self.save()
        if proc_id() == 0:
            self.writer.__exit__(exc_type, exc_value, traceback)
        self.sess.__exit__(exc_type, exc_value, traceback)
        self.logger.log("Closed TF session for %s" % self)
        self._reset()

    def __str__(self):
        return "Agent(id=%d, epochs=%d)" % (self.id, self.epochs)

    def sync_restore_path(self):
        restore_path = None
        if proc_id() == 0:
            if self.saver.has_saved_models(self):
                restore_path = self.saver.get_latest_model_path(self)
        return broadcastpy(restore_path, root=0)

    def sync_random_restore_path(self):
        restore_path = None
        if proc_id() == 0:
            restore_path = self.saver.get_random_model_path()
        return broadcastpy(restore_path, root=0)

    @property
    def is_done(self):
        return self.epochs >= self.cfg.epochs


class InferenceAgent:
    """Adversary agent used for self-play and inference.

    No-op expiry and spatial observation stacking need to happen outside of
    this class. This class only provides low-level `get_next_action()` function.
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        self.graph = None
        self.sess = None
        self.model = None
        self.cfg = None

    def restore(self, restore_path):
        with self.graph.as_default():
            agent_id = AgentSaver.extract_agent_id(restore_path)
            cfg_path = AgentSaver.extract_cfg_path(restore_path)
            self.cfg = TrainingCfg.load(cfg_path)
            self.model = restore_model(
                self.sess, restore_path, agent_id, is_train=False
            )

    def get_next_action(self, o):
        if self.model is None:
            raise ValueError("model needs to be initialized")

        a, v_t = self.sess.run(
            [
                {
                    "spell": self.model["pi_spell_argmax"],
                    "spatial": self.model["pi_spatial_argmax"],
                },
                self.model["v"],
            ],
            feed_dict={
                self.model["x_non_spatial_ph"]: o["non_spatial"][None, :],
                self.model["x_spatial_ph"]: o["spatial"][None, :],
                self.model["x_mask_spell_ph"]: o["mask_spell"][None, :],
                self.model["x_mask_spatial_ph"]: o["mask_spatial"][None, :],
                self.model["x_if_spawn_spell_ph"]: o["if_spawn_spell"][None, :],
                # TODO dimitrijer: enable rnn inference
                # self.model["x_state_in_ph"]: o["state_in"],
                # self.model["x_rnn_mask_ph"]: np.zeros(1),
                # self.model["x_batch_size_ph"]: 1,
            },
        )

        return a, v_t

    def _create_sess(self):
        # Setup TF session.
        gpu_options = tf.GPUOptions()
        gpu_options.allow_growth = True
        gpu_options.per_process_gpu_memory_fraction = 0.5
        return tf.Session(
            graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options)
        )

    def initialize(self):
        if self.is_initialized():
            raise ValueError("inference agent already initialized")
        self.graph = tf.Graph()
        self.sess = self._create_sess()

    def deinitialize(self):
        if not self.is_initialized():
            return
        self.sess.close()
        self._reset()

    def is_initialized(self):
        return self.sess is not None
