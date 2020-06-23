import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.ops.script_ops import _py_funcs as FuncRegistry

from heroic_rl.utils.mpi_tools import proc_id

from ..utils.logx import restore_tf_graph
from ..utils.mpi_tf import MpiAdamOptimizer, define_sync_params, sync_all_params
from ..utils.mpi_tools import mpi_statistics_scalar
from . import core_ppo_heroic as core
from . import utils


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_space, act_space, cfg):
        self.cfg = cfg
        size = cfg.local_steps_per_epoch
        self.obs_buf = {
            "non_spatial": np.zeros(
                utils.combined_shape(size, obs_space["non_spatial"].shape),
                dtype=np.float32,
            ),
            "spatial": np.zeros(
                utils.combined_shape(size, obs_space["spatial"].shape), dtype=np.float32
            ),
            "mask_spell": np.zeros(
                utils.combined_shape(size, obs_space["mask_spell"].shape),
                dtype=np.float32,
            ),
            "mask_spatial": np.zeros(
                utils.combined_shape(size, obs_space["mask_spatial"].shape),
                dtype=np.float32,
            ),
            "if_spawn_spell": np.zeros(
                utils.combined_shape(size, obs_space["if_spawn_spell"].shape),
                dtype=np.float32,
            ),
        }
        self.act_buf = {
            "spell": np.zeros(
                utils.combined_shape(size, act_space["spell"].shape), dtype=np.float32
            ),
            "spatial": np.zeros(
                utils.combined_shape(size, act_space["spatial"].shape), dtype=np.float32
            ),
        }
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = cfg.hyperparameters.gamma, cfg.hyperparameters.lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.episode_starts = np.zeros(size)
        self.episode_starts[0] = 1.0
        self.batch_slices = []
        self.size = size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf["non_spatial"][self.ptr] = obs["non_spatial"]
        self.obs_buf["spatial"][self.ptr] = obs["spatial"]
        self.obs_buf["mask_spell"][self.ptr] = obs["mask_spell"]
        self.obs_buf["mask_spatial"][self.ptr] = obs["mask_spatial"]
        self.obs_buf["if_spawn_spell"][self.ptr] = obs["if_spawn_spell"]
        self.act_buf["spell"][self.ptr] = act["spell"]
        self.act_buf["spatial"][self.ptr] = act["spatial"]
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = utils.discount_cumsum(deltas, self.gamma * self.lam)

        # todo: michalw see how one can use gae for value function (stable baselines)
        # the next line computes rewards-to-go, to be targets for the value function
        # self.ret_buf[path_slice] = utils.discount_cumsum(rews, self.gamma)[:-1]
        # gae lambda return
        self.ret_buf[path_slice] = self.adv_buf[path_slice] + vals[:-1]

        self.path_start_idx = self.ptr

        # if a new episode started mark the start
        if self.path_start_idx < self.size:
            self.episode_starts[self.path_start_idx] = 1.0

    def _get_slice(self, buffer_slice):
        """
        Call to get all of the data from the buffer, corresponding to the given slice,
        with advantages appropriately normalized (shifted to have mean zero and std
        one).

        :param buffer_slice : indices corresponding to buffer elements to be selected
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get

        # creating initial state for training
        # TODO dimitrijer: I am pretty sure we don't need to create a new array
        # every time; I might be wrong
        state_in = self.cfg.architecture.empty_rnn_state

        # batch size depend on the slice
        batch_size = len(self.episode_starts[buffer_slice])

        return [
            self.obs_buf["non_spatial"][buffer_slice],
            self.obs_buf["spatial"][buffer_slice],
            self.obs_buf["mask_spell"][buffer_slice],
            self.obs_buf["mask_spatial"][buffer_slice],
            self.obs_buf["if_spawn_spell"][buffer_slice],
            state_in,
            self.episode_starts[buffer_slice],
            batch_size,
            self.act_buf["spell"][buffer_slice],
            self.act_buf["spatial"][buffer_slice],
            self.adv_buf[buffer_slice],
            self.ret_buf[buffer_slice],
            self.logp_buf[buffer_slice],
            self.val_buf[buffer_slice],
        ]

    def get_batch(self, batch_index):
        """
        Get batch corresponding to batch_index. Should be run after create_batch_data.
        """
        buffer_slice = self.batch_slices[batch_index]
        return self._get_slice(buffer_slice)

    def create_batch_data(self):
        # the next two lines implement the advantage normalization trick, note that
        # we're not slicing here
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        batch_start = 0
        batch_end = None
        for step in range(len(self.episode_starts)):
            if step - batch_start > self.cfg.batch_size:
                self.batch_slices.append(slice(batch_start, batch_end))
                batch_start = batch_end
            if self.cfg.architecture.rnn and self.episode_starts[step]:
                batch_end = step
            elif not self.cfg.architecture.rnn:
                batch_end = step

        # do not allow too small batches
        if self.size - batch_start > self.cfg.batch_size / 4:
            self.batch_slices.append(slice(batch_start, self.size))
        elif proc_id() == 0:
            print(
                "size - batch_start=", self.size - batch_start, " so batch too small!"
            )

        num_batches = len(self.batch_slices)

        return num_batches

    def reset(self):
        assert self.ptr == self.max_size  # buffer has to be full before you can reset
        self.ptr, self.path_start_idx = 0, 0
        self.episode_starts = np.zeros(self.size, dtype=np.int)
        self.batch_slices = []


def restore_pyfuncs(agent_id, pi_grads_and_vars, v_grads_and_vars):
    expected_registry_size = agent_id * 5
    if FuncRegistry.size() < expected_registry_size:
        pi_iter = iter(pi_grads_and_vars)
        v_iter = iter(v_grads_and_vars)

        # - pi_optimizer.minimize(pi_loss)
        #   - collecting gradients (tf.py_func(_collect_grads, [flat_grad],
        #   tf.float32) => pyfunc_0
        #   - syncing params (tf.py_func(_broadcast, [get_params], tf.float32))
        #   => pyfunc_1
        MpiAdamOptimizer(grads_and_vars=list(zip(pi_iter, pi_iter)))

        # - v_optimizer.minimize(v_loss)
        #   - collecting gradients (tf.py_func(_collect_grads, [flat_grad],
        #   tf.float32) => pyfunc_2
        #   - syncing params (tf.py_func(_broadcast, [get_params], tf.float32))
        #   => pyfunc_3
        MpiAdamOptimizer(grads_and_vars=list(zip(v_iter, v_iter)))

        # - sync global variables (tf.py_func(_broadcast, [get_params], tf.float32))
        #   => pyfunc_4
        define_sync_params(tf.global_variables())


def restore_pyfuncs_unified(agent_id, grads_and_vars):
    expected_registry_size = agent_id * 3
    if FuncRegistry.size() < expected_registry_size:
        gv_iter = iter(grads_and_vars)

        # - optimizer.minimize(pi_loss)
        #   - collecting gradients (tf.py_func(_collect_grads, [flat_grad],
        #   tf.float32) => pyfunc_0
        #   - syncing params (tf.py_func(_broadcast, [get_params], tf.float32))
        #   => pyfunc_1
        MpiAdamOptimizer(grads_and_vars=list(zip(gv_iter, gv_iter)))

        # - sync global variables (tf.py_func(_broadcast, [get_params], tf.float32))
        #   => pyfunc_2
        define_sync_params(tf.global_variables())


def restore_model(sess, path, agent_id, is_train=True):
    model = restore_tf_graph(sess, path)
    if len(tf.get_collection("train")) > 0:
        # Unified model.
        train, *grads_and_vars = tf.get_collection("train")
        if is_train:
            # Define optimizers so Python tf operators will get registered in
            # FuncRegistry.
            restore_pyfuncs_unified(agent_id, grads_and_vars)
        model["train"] = train
    else:
        train_pi, *pi_grads_and_vars = tf.get_collection("train_pi")
        train_v, *v_grads_and_vars = tf.get_collection("train_v")
        if is_train:
            # Define optimizers so Python tf operators will get registered in
            # FuncRegistry.
            restore_pyfuncs(agent_id, pi_grads_and_vars, v_grads_and_vars)

        model["train_pi"] = train_pi
        model["train_v"] = train_v

    sync_op = tf.get_collection("sync_op")[0]
    summary_op = tf.get_collection("summary_op")[0]
    model["sync_op"] = sync_op
    model["summary_op"] = summary_op

    return model


def build_model(env, cfg):

    # Inputs to computation graph - observations
    with tf.name_scope("Inputs"):
        with tf.name_scope("Observation"):
            (
                x_non_spatial_ph,
                x_spatial_ph,
                x_mask_spell_ph,
                x_mask_spatial_ph,
                x_if_spawn_spell_ph,
            ) = utils.placeholders_from_spaces(
                env.observation_space["non_spatial"],
                env.observation_space["spatial"],
                env.observation_space["mask_spell"],
                env.observation_space["mask_spatial"],
                env.observation_space["if_spawn_spell"],
                names=[
                    "NonSpatial",
                    "Spatial",
                    "MaskSpell",
                    "MaskSpatial",
                    "IfSpawnSpell",
                ],
            )

            x_state_in_ph = tf.placeholder(
                dtype=tf.float32,
                shape=cfg.architecture.empty_rnn_state.shape,
                name="StateIn",
            )
            x_batch_size_ph = tf.placeholder(dtype=tf.int64, shape=(), name="BatchSize")
            x_rnn_mask_ph = utils.placeholder(None, name="RnnMask")

        with tf.name_scope("Action"):
            # Inputs to computation graph - actions
            (a_spell_ph, a_spatial_ph,) = utils.placeholders_from_spaces(
                env.action_space["spell"],
                env.action_space["spatial"],
                names=["Spell", "Spatial"],
            )

        adv_ph, ret_ph, logp_old_ph, v_old_ph = utils.placeholders(
            None, None, None, None, names=["Advantage", "Return", "Logp", "Value"]
        )

    actor_critic = core.actor_critic_fn(cfg)
    # Share information about action space with policy architecture
    # state_out is None, if Architecture.USE_RNN == false
    pi, logp, logp_pi, v, state_out = actor_critic(
        {
            "spatial": x_spatial_ph,
            "non_spatial": x_non_spatial_ph,
            "mask_spell": x_mask_spell_ph,
            "mask_spatial": x_mask_spatial_ph,
            "if_spawn_spell": x_if_spawn_spell_ph,
            "state_in": x_state_in_ph,
            "batch_size": x_batch_size_ph,
            "rnn_mask": x_rnn_mask_ph,
        },
        {"spell": a_spell_ph, "spatial": a_spatial_ph},
        env.action_space,
        cfg,
    )

    hyperp = cfg.hyperparameters
    if hyperp.clip_vf_output:
        v = tf.clip_by_value(v, -1.0, 1.0)

    # PPO objectives
    with tf.name_scope("Objectives"):

        # policy block
        ratio = tf.exp(logp - logp_old_ph)  # pi(a|s) / pi_old(a|s)
        min_adv = tf.where(
            adv_ph > 0,
            (1 + hyperp.clip_ratio) * adv_ph,
            (1 - hyperp.clip_ratio) * adv_ph,
        )
        pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv), name="LossPi")

        # value block

        if hyperp.value_clipping_enabled:
            # value function clipping
            v_clipped = v_old_ph + tf.clip_by_value(
                v - v_old_ph, -hyperp.clip_range_vf, hyperp.clip_range_vf
            )
            v_loss1 = tf.square(v - ret_ph)
            v_loss2 = tf.square(v_clipped - ret_ph)
            v_loss = tf.reduce_mean(tf.maximum(v_loss1, v_loss2), name="LossV")
        else:
            v_loss = tf.reduce_mean((ret_ph - v) ** 2, name="LossV")

        # value function regularization
        if hyperp.vf_reg_enabled and not cfg.architecture.unified_policy_value:
            params_v = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="ValueFunctionMain"
            )
            v_loss_l2 = tf.add_n(
                [tf.nn.l2_loss(v) for v in params_v if "bias" not in v.name]
            )
            v_loss = v_loss + hyperp.vf_reg * v_loss_l2

        # todo michalw: entropy bonus, value function clipping?
        loss = hyperp.pi_loss_coef * pi_loss + hyperp.vf_loss_coef * v_loss

    # Useful to watch during learning
    with tf.name_scope("Info"):
        # a sample estimate for KL-divergence, easy to compute
        approx_kl = tf.reduce_mean(logp_old_ph - logp, name="KL")
        # a sample estimate for entropy, also easy to compute
        approx_ent = tf.reduce_mean(-logp, name="Entropy")
        clipped = tf.logical_or(
            ratio > (1 + hyperp.clip_ratio), ratio < (1 - hyperp.clip_ratio)
        )
        clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32), name="ClipFrac")

    with tf.name_scope("Optimizers"):
        pi_optimizer = MpiAdamOptimizer(learning_rate=hyperp.pi_lr)
        v_optimizer = MpiAdamOptimizer(learning_rate=hyperp.vf_lr)
        optimizer = MpiAdamOptimizer(learning_rate=hyperp.lr)
        train, train_pi, train_v = None, None, None

        if hyperp.grad_clipping_enabled:
            if cfg.architecture.unified_policy_value:
                # gradient clipping enabled, unified PV
                params = tf.trainable_variables()
                grads, _vars = zip(*optimizer.compute_gradients(loss, params))
                grads, _grad_norm = tf.clip_by_global_norm(grads, hyperp.max_grad_norm)
                grads = list(zip(grads, params))
                train = optimizer.apply_gradients(grads)
            else:
                # gradient clipping enabled, separate PV
                params_pi = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope="PolicyMain"
                )
                pi_grads, _vars = zip(
                    *pi_optimizer.compute_gradients(pi_loss, params_pi)
                )
                pi_grads, _grad_norm = tf.clip_by_global_norm(
                    pi_grads, hyperp.max_grad_norm
                )
                pi_grads = list(zip(pi_grads, params_pi))
                train_pi = pi_optimizer.apply_gradients(pi_grads)

                params_v = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope="ValueFunctionMain"
                )
                v_grads, _vars = zip(*v_optimizer.compute_gradients(v_loss, params_v))
                v_grads, _grad_norm = tf.clip_by_global_norm(
                    v_grads, hyperp.max_grad_norm
                )
                v_grads = list(zip(v_grads, params_v))
                train_v = v_optimizer.apply_gradients(v_grads)
        else:
            if cfg.architecture.unified_policy_value:
                # no gradient clipping, unified PV
                train = optimizer.minimize(loss)
            else:
                # no gradient clipping, separate PV
                train_pi = pi_optimizer.minimize(pi_loss)
                train_v = v_optimizer.minimize(v_loss)

    if not cfg.architecture.unified_policy_value:
        tf.add_to_collection("train_pi", train_pi)
        for grad, var in pi_optimizer.grads_and_vars:
            tf.add_to_collection("train_pi", grad)
            tf.add_to_collection("train_pi", var)

        tf.add_to_collection("train_v", train_v)
        for grad, var in v_optimizer.grads_and_vars:
            tf.add_to_collection("train_v", grad)
            tf.add_to_collection("train_v", var)
    else:
        tf.add_to_collection("train", train)
        for grad, var in optimizer.grads_and_vars:
            tf.add_to_collection("train", grad)
            tf.add_to_collection("train", var)

    sync_op = sync_all_params()
    tf.add_to_collection("sync_op", sync_op)

    summary_op = tf.summary.merge_all()
    if summary_op is None:
        summary_op = tf.no_op()
    tf.add_to_collection("summary_op", summary_op)

    return {
        "x_non_spatial_ph": x_non_spatial_ph,
        "x_spatial_ph": x_spatial_ph,
        "x_mask_spell_ph": x_mask_spell_ph,
        "x_mask_spatial_ph": x_mask_spatial_ph,
        "x_if_spawn_spell_ph": x_if_spawn_spell_ph,
        "x_state_in_ph": x_state_in_ph,
        "x_rnn_mask_ph": x_rnn_mask_ph,
        "x_batch_size_ph": x_batch_size_ph,
        "a_spell_ph": a_spell_ph,
        "a_spatial_ph": a_spatial_ph,
        "adv_ph": adv_ph,
        "v_old_ph": v_old_ph,
        "ret_ph": ret_ph,
        "logp_old_ph": logp_old_ph,
        "pi_spell": pi["spell"],
        "pi_spell_argmax": pi["spell_argmax"],
        "pi_spatial": pi["spatial"],
        "pi_spatial_argmax": pi["spatial_argmax"],
        "v": v,
        "state_out": state_out,
        "pi_loss": pi_loss,
        "v_loss": v_loss,
        "loss": loss,
        "approx_ent": approx_ent,
        "approx_kl": approx_kl,
        "clipfrac": clipfrac,
        "logp_pi": logp_pi,
        "train_pi": train_pi,
        "train_v": train_v,
        "train": train,
        "sync_op": sync_op,
        "summary_op": summary_op,
    }
