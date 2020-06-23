import numpy as np
import tensorflow.compat.v1 as tf

from ..train.cfg import TrainingCfg
from .layers import (
    after_rnn_reshape,
    before_rnn_reshape,
    concat_layer,
    concat_mlp_layer,
    conv2d,
    logits_layer_single_spatial,
    mlp,
    non_spatial_layer,
    rnn_layer,
    spatial_layer,
)

############
# Policies #
############


def mlp_policy_network(x, a, action_space, cfg: TrainingCfg):
    """
    MLP policy with conditioned spatial output, with regards to sampled spell
    output.
    """
    act_dim_spell = action_space["spell"].n
    act_dim_spatial = action_space["spatial"].n
    a_spatial = a["spatial"]

    # Masks
    if_spawn_spell = x["if_spawn_spell"]
    mask_spawn = (
        if_spawn_spell * cfg.spells.mask_spawn
        + (1.0 - if_spawn_spell) * cfg.spells.mask_nonspawn
    )
    mask_spell = mask_spawn * x["mask_spell"]
    logits_mask_spell = tf.log(mask_spell)
    mask_spatial = x["mask_spatial"]

    non_spatial_features = non_spatial_layer(
        x, cfg.architecture.shared_params, "DenseNonSpatial"
    )
    spatial_features = spatial_layer(x, cfg.architecture.shared_params, "ConvSpatial")
    features_fc = concat_mlp_layer(
        non_spatial_features,
        spatial_features,
        cfg.architecture.shared_params,
        name="Spawn",
    )

    spawn_logits, spawn_spatial_logits = logits_layer_single_spatial(
        features_fc, act_dim_spell, act_dim_spatial, name="Spawn"
    )

    spell_logits, spell_spatial_logits = logits_layer_single_spatial(
        features_fc, act_dim_spell, act_dim_spatial, name="Spell"
    )

    logits_spell = if_spawn_spell * spawn_logits + (1.0 - if_spawn_spell) * spell_logits

    if cfg.hyperparameters.bias_noops:
        # noop bias for all noop actions
        logit_bias = tf.zeros(act_dim_spell)
        # previously used -3.0 + bias according to number of noop actions
        # + bias according to number of spatial actions
        noop_bias = -3.0 + np.log(cfg.spells.num_noops) - np.log(act_dim_spatial)
        for spell in cfg.spells.enabled_spells:
            if spell.is_noop:
                spell_idx = cfg.spells.spell_to_idx(spell)
                logit_bias += noop_bias * tf.one_hot(spell_idx, depth=act_dim_spell)

        logits_spell = logits_spell + logit_bias

    logits_spatial = (
        if_spawn_spell * spawn_spatial_logits
        + (1.0 - if_spawn_spell) * spell_spatial_logits
    )
    logits_spatial = tf.reshape(
        logits_spatial, shape=[-1, act_dim_spell, act_dim_spatial]
    )

    # Mask valid spell logits only.
    logits_valid_spell = logits_spell + logits_mask_spell

    # Sample spell according to policy.
    pi_spell_batch = tf.multinomial(logits_valid_spell, 1, name="SampleSpell")
    pi_spell = tf.squeeze(pi_spell_batch, axis=1, name="PiSpell")

    # Sample spell via argmax for inference.
    pi_spell_argmax = tf.argmax(logits_valid_spell, axis=1, name="PiSpellArgmax")

    # Condition policy-sampled spatial on policy-sampled spell.
    mask_spatial_pi = tf.reduce_sum(
        mask_spatial
        * tf.expand_dims(tf.one_hot(pi_spell, depth=act_dim_spell), axis=2),
        axis=1,
    )
    logits_spatial_pi = tf.reduce_sum(
        logits_spatial
        * tf.expand_dims(tf.one_hot(pi_spell, depth=act_dim_spell), axis=2),
        axis=1,
    )
    # Mask out invalid spatial actions.
    logits_valid_spatial_pi = logits_spatial_pi + tf.log(mask_spatial_pi)
    logits_valid_spatial_pi = tf.identity(
        logits_valid_spatial_pi, name="LogitsSpatialPi"
    )
    # Sample spatial according to policy, conditioned by sampled spell.
    pi_spatial = tf.squeeze(
        tf.multinomial(logits_valid_spatial_pi, 1, name="SampleSpatial"),
        axis=1,
        name="PiSpatial",
    )

    # Condition policy-sampled spatial on the given spell.
    mask_spatial_a = tf.reduce_sum(
        mask_spatial
        * tf.expand_dims(tf.one_hot(a["spell"], depth=act_dim_spell), axis=2),
        axis=1,
    )
    logits_spatial_a = tf.reduce_sum(
        logits_spatial
        * tf.expand_dims(tf.one_hot(a["spell"], depth=act_dim_spell), axis=2),
        axis=1,
    )
    # Mask out invalid spatial actions.
    logits_valid_spatial_a = logits_spatial_a + tf.log(mask_spatial_a)
    logits_valid_spatial_a = tf.identity(logits_valid_spatial_a, name="LogitsSpatialA")

    # Sample spatial via argmax for inference, conditioned by argmax spell.
    mask_spatial_argmax = tf.reduce_sum(
        mask_spatial
        * tf.expand_dims(tf.one_hot(pi_spell_argmax, depth=act_dim_spell), axis=2),
        axis=1,
    )
    logits_spatial_argmax = tf.reduce_sum(
        logits_spatial
        * tf.expand_dims(tf.one_hot(pi_spell_argmax, depth=act_dim_spell), axis=2),
        axis=1,
    )
    # Mask out invalid spatial actions.
    logits_valid_spatial_argmax = logits_spatial_argmax + tf.log(mask_spatial_argmax)
    # Sample spatial according to argmax, condition by argmax sampled spell.
    pi_spatial_argmax = tf.argmax(
        logits_valid_spatial_argmax, axis=1, name="PiSpatialArgmax"
    )

    with tf.name_scope("Logp"):
        # Clip in order to avoid multiplying 0 by -inf below (note that
        # we use one hot below hence 0 and above we calculate tf.log(0), hence -inf)
        log_softmax_valid_spell = tf.clip_by_value(
            tf.nn.log_softmax(logits_valid_spell), -10, 0
        )
        log_softmax_spatial_pi = tf.clip_by_value(
            tf.nn.log_softmax(logits_valid_spatial_pi), -10, 0
        )
        log_softmax_spatial_a = tf.clip_by_value(
            tf.nn.log_softmax(logits_valid_spatial_a), -10, 0
        )

        logp_a_spell = tf.reduce_sum(
            tf.one_hot(a["spell"], depth=act_dim_spell) * log_softmax_valid_spell,
            axis=1,
        )
        logp_pi_spell = tf.reduce_sum(
            tf.one_hot(pi_spell, depth=act_dim_spell) * log_softmax_valid_spell, axis=1
        )

        logp_pi_spatial = tf.reduce_sum(
            tf.one_hot(pi_spatial, depth=act_dim_spatial) * log_softmax_spatial_pi,
            axis=1,
        )

        logp_a_spatial = tf.reduce_sum(
            tf.one_hot(a_spatial, depth=act_dim_spatial) * log_softmax_spatial_a,
            axis=1,
        )

        logp_a = logp_a_spell + logp_a_spatial
        logp_pi = logp_pi_spell + logp_pi_spatial

        pi = {
            "spell": pi_spell,
            "spatial": pi_spatial,
            "spell_argmax": pi_spell_argmax,
            "spatial_argmax": pi_spatial_argmax,
        }

        return pi, logp_a, logp_pi


def mlp_policy_value_network(x, a, action_space, cfg: TrainingCfg):
    """
    MLP policy-value network with conditioned spatial output, with regards to
    sampled spell output.
    """
    act_dim_spell = action_space["spell"].n
    act_dim_spatial = action_space["spatial"].n
    a_spatial = a["spatial"]

    if_spawn_spell = x["if_spawn_spell"]
    mask_spawn = (
        if_spawn_spell * cfg.spells.mask_spawn
        + (1.0 - if_spawn_spell) * cfg.spells.mask_nonspawn
    )
    mask_spell = mask_spawn * x["mask_spell"]
    logits_mask_spell = tf.log(mask_spell)

    non_spatial_features = non_spatial_layer(
        x, cfg.architecture.shared_params, "DenseNonSpatial"
    )
    spatial_features = spatial_layer(x, cfg.architecture.shared_params, "ConvSpatial")
    features_fc = concat_mlp_layer(
        non_spatial_features,
        spatial_features,
        cfg.architecture.shared_params,
        name="Spawn",
    )

    # value head
    v = mlp(
        features_fc,
        hidden_sizes=[cfg.architecture.value_params["fc_units_concat"], 1],
        activation=tf.tanh,
        output_activation=None,
        name="Vf",
    )

    # policy heads
    spawn_logits, spawn_spatial_logits = logits_layer_single_spatial(
        features_fc, act_dim_spell, act_dim_spatial, name="Spawn"
    )

    spell_logits, spell_spatial_logits = logits_layer_single_spatial(
        features_fc, act_dim_spell, act_dim_spatial, name="Spell"
    )

    logits_spell = if_spawn_spell * spawn_logits + (1.0 - if_spawn_spell) * spell_logits

    if cfg.hyperparameters.bias_noops:
        # noop bias for all noop actions
        logit_bias = tf.zeros(act_dim_spell)
        # previously used -3.0 + bias according to number of noop actions
        # + bias according to number of spatial actions
        noop_bias = -3.0 + np.log(cfg.spells.num_noops) - np.log(act_dim_spatial)
        for spell in cfg.spells.enabled_spells:
            if spell.is_noop:
                spell_idx = cfg.spells.spell_to_idx(spell)
                logit_bias += noop_bias * tf.one_hot(spell_idx, depth=act_dim_spell)

        logits_spell = logits_spell + logit_bias

    logits_spatial = (
        if_spawn_spell * spawn_spatial_logits
        + (1.0 - if_spawn_spell) * spell_spatial_logits
    )
    logits_spatial = tf.reshape(
        logits_spatial, shape=[-1, act_dim_spell, act_dim_spatial]
    )

    # Mask valid spell logits only.
    logprob_valid_spell = logits_spell + logits_mask_spell

    # Sample spell according to policy.
    pi_spell_batch = tf.multinomial(logprob_valid_spell, 1, name="SampleSpell")
    pi_spell = tf.squeeze(pi_spell_batch, axis=1, name="PiSpell")

    # Sample spell via argmax for inferrence.
    pi_spell_argmax = tf.argmax(logprob_valid_spell, axis=1, name="PiSpellArgmax")

    # Condition policy-sampled spatial on policy-sampled spell.
    logits_spatial_pi = tf.reduce_sum(
        logits_spatial
        * tf.expand_dims(tf.one_hot(pi_spell, depth=act_dim_spell), axis=2),
        axis=1,
    )
    logits_spatial_pi = tf.identity(logits_spatial_pi, name="LogitsSpatialPi")

    logits_spatial_a = tf.reduce_sum(
        logits_spatial
        * tf.expand_dims(tf.one_hot(a["spell"], depth=act_dim_spell), axis=2),
        axis=1,
    )
    logits_spatial_a = tf.identity(logits_spatial_a, name="LogitsSpatialA")

    # Sample spatial according to policy, conditioned by sampled spell.
    pi_spatial = tf.squeeze(
        tf.multinomial(logits_spatial_pi, 1, name="SampleSpatial"),
        axis=1,
        name="PiSpatial",
    )

    # Sample spatial via argmax for inferrence, conditioned on by argmax spell.
    logits_spatial_argmax = tf.reduce_sum(
        logits_spatial
        * tf.expand_dims(tf.one_hot(pi_spell_argmax, depth=act_dim_spell), axis=2),
        axis=1,
    )
    pi_spatial_argmax = tf.argmax(logits_spatial_argmax, axis=1, name="PiSpatialArgmax")

    with tf.name_scope("Logp"):
        logp_valid_spell = tf.clip_by_value(logprob_valid_spell, np.log(1e-10), 0)
        logp_a_spell = tf.reduce_sum(
            tf.one_hot(a["spell"], depth=act_dim_spell) * logp_valid_spell, axis=1
        )
        logp_pi_spell = tf.reduce_sum(
            tf.one_hot(pi_spell, depth=act_dim_spell) * logp_valid_spell, axis=1
        )

        logp_pi_spatial = tf.reduce_sum(
            tf.one_hot(pi_spatial, depth=act_dim_spatial)
            * tf.nn.log_softmax(logits_spatial_pi),
            axis=1,
        )

        logp_a_spatial = tf.reduce_sum(
            tf.one_hot(a_spatial, depth=act_dim_spatial)
            * tf.nn.log_softmax(logits_spatial_a),
            axis=1,
        )

        logp_a = logp_a_spell + logp_a_spatial
        logp_pi = logp_pi_spell + logp_pi_spatial

        pi = {
            "spell": pi_spell,
            "spatial": pi_spatial,
            "spell_argmax": pi_spell_argmax,
            "spatial_argmax": pi_spatial_argmax,
        }

        return pi, logp_a, logp_pi, v


def rnn_policy_network(x, a, action_space, cfg: TrainingCfg):
    """
    Recurrent policy with conditioned spatial output, with regards to sampled spell
    output.
    """
    act_dim_spell = action_space["spell"].n
    act_dim_spatial = action_space["spatial"].n
    a_spatial = a["spatial"]
    batch_size = x["batch_size"]
    rnn_mask = x["rnn_mask"]
    state_in = x["state_in"]

    if_spawn_spell = x["if_spawn_spell"]
    mask_spawn = (
        if_spawn_spell * cfg.spells.mask_spawn
        + (1.0 - if_spawn_spell) * cfg.spells.mask_nonspawn
    )
    mask_spell = mask_spawn * x["mask_spell"]
    logits_mask_spell = tf.log(mask_spell)

    non_spatial_features = non_spatial_layer(
        x, cfg.architecture.shared_params, "DenseNonSpatial"
    )
    spatial_features = spatial_layer(x, cfg.architecture.shared_params, "ConvSpatial")

    # rnn block
    concat_features = concat_layer(non_spatial_features, spatial_features)
    rnn_input = before_rnn_reshape(concat_features, batch_size)
    rnn_output, state_out = rnn_layer(
        rnn_input, cfg.architecture.rnn_size, state_in, batch_size, rnn_mask
    )
    features_fc = after_rnn_reshape(rnn_output)

    # policy heads
    spawn_logits, spawn_spatial_logits = logits_layer_single_spatial(
        features_fc, act_dim_spell, act_dim_spatial, name="Spawn"
    )

    spell_logits, spell_spatial_logits = logits_layer_single_spatial(
        features_fc, act_dim_spell, act_dim_spatial, name="Spell"
    )

    logits_spell = if_spawn_spell * spawn_logits + (1.0 - if_spawn_spell) * spell_logits

    if cfg.hyperparameters.bias_noops:
        # noop bias for all noop actions
        logit_bias = tf.zeros(act_dim_spell)
        # previously used -3.0 + bias according to number of noop actions
        # + bias according to number of spatial actions
        noop_bias = -3.0 + np.log(cfg.spells.num_noops) - np.log(act_dim_spatial)
        for spell in cfg.spells.enabled_spells:
            if spell.is_noop:
                spell_idx = cfg.spells.spell_to_idx(spell)
                logit_bias += noop_bias * tf.one_hot(spell_idx, depth=act_dim_spell)

        logits_spell = logits_spell + logit_bias

    logits_spatial = (
        if_spawn_spell * spawn_spatial_logits
        + (1.0 - if_spawn_spell) * spell_spatial_logits
    )
    logits_spatial = tf.reshape(
        logits_spatial, shape=[-1, act_dim_spell, act_dim_spatial]
    )

    # Sample spell and argmax spell.
    logprob_valid_spell = logits_spell + logits_mask_spell
    pi_spell_batch = tf.multinomial(logprob_valid_spell, 1, name="SampleSpell")
    pi_spell = tf.squeeze(pi_spell_batch, axis=1, name="PiSpell")
    pi_spell_argmax = tf.argmax(logprob_valid_spell, axis=1, name="PiSpellArgmax")

    # Prepare spatial logits for sampling.
    logits_spatial_pi = tf.reduce_sum(
        logits_spatial
        * tf.expand_dims(tf.one_hot(pi_spell, depth=act_dim_spell), axis=2),
        axis=1,
    )
    logits_spatial_pi = tf.identity(logits_spatial_pi, name="LogitsSpatialPi")

    logits_spatial_a = tf.reduce_sum(
        logits_spatial
        * tf.expand_dims(tf.one_hot(a["spell"], depth=act_dim_spell), axis=2),
        axis=1,
    )
    logits_spatial_a = tf.identity(logits_spatial_a, name="LogitsSpatialA")

    # Sample and argmax X and Y.
    pi_spatial = tf.squeeze(
        tf.multinomial(logits_spatial_pi, 1, name="SampleSpatial"),
        axis=1,
        name="PiSpatial",
    )
    pi_spatial_argmax = tf.argmax(logits_spatial_pi, axis=1, name="PiSpatialArgmax")

    with tf.name_scope("Logp"):
        logp_valid_spell = tf.clip_by_value(logprob_valid_spell, np.log(1e-10), 0)
        logp_a_spell = tf.reduce_sum(
            tf.one_hot(a["spell"], depth=act_dim_spell) * logp_valid_spell, axis=1
        )
        logp_pi_spell = tf.reduce_sum(
            tf.one_hot(pi_spell, depth=act_dim_spell) * logp_valid_spell, axis=1
        )

        logp_pi_spatial = tf.reduce_sum(
            tf.one_hot(pi_spatial, depth=act_dim_spatial)
            * tf.nn.log_softmax(logits_spatial_pi),
            axis=1,
        )

        logp_a_spatial = tf.reduce_sum(
            tf.one_hot(a_spatial, depth=act_dim_spatial)
            * tf.nn.log_softmax(logits_spatial_a),
            axis=1,
        )

        logp_a = logp_a_spell + logp_a_spatial
        logp_pi = logp_pi_spell + logp_pi_spatial

        pi = {
            "spell": pi_spell,
            "spatial": pi_spatial,
            "spell_argmax": pi_spell_argmax,
            "spatial_argmax": pi_spatial_argmax,
        }

        return pi, logp_a, logp_pi, state_out


def rnn_policy_value_network(x, a, action_space, cfg: TrainingCfg):
    """
    Recurrent policy-value network with conditioned spatial output, with regards
    to sampled spell output.
    """
    act_dim_spell = action_space["spell"].n
    act_dim_spatial = action_space["spatial"].n
    a_spatial = a["spatial"]
    batch_size = x["batch_size"]
    rnn_mask = x["rnn_mask"]
    state_in = x["state_in"]

    if_spawn_spell = x["if_spawn_spell"]
    mask_spawn = (
        if_spawn_spell * cfg.spells.mask_spawn
        + (1.0 - if_spawn_spell) * cfg.spells.mask_nonspawn
    )
    mask_spell = mask_spawn * x["mask_spell"]
    logits_mask_spell = tf.log(mask_spell)

    non_spatial_features = non_spatial_layer(
        x, cfg.architecture.shared_params, "DenseNonSpatial"
    )
    spatial_features = spatial_layer(x, cfg.architecture.shared_params, "ConvSpatial")

    # rnn block
    concat_features = concat_layer(non_spatial_features, spatial_features)
    rnn_input = before_rnn_reshape(concat_features, batch_size)
    rnn_output, state_out = rnn_layer(
        rnn_input, cfg.architecture.rnn_size, state_in, batch_size, rnn_mask
    )
    features_fc = after_rnn_reshape(rnn_output)

    # value head
    v = mlp(
        features_fc,
        hidden_sizes=[cfg.architecture.value_params["fc_units_concat"], 1],
        activation=tf.tanh,
        output_activation=None,
        name="Vf",
    )

    # policy heads
    spawn_logits, spawn_spatial_logits = logits_layer_single_spatial(
        features_fc, act_dim_spell, act_dim_spatial, name="Spawn"
    )

    spell_logits, spell_spatial_logits = logits_layer_single_spatial(
        features_fc, act_dim_spell, act_dim_spatial, name="Spell"
    )

    logits_spell = if_spawn_spell * spawn_logits + (1.0 - if_spawn_spell) * spell_logits

    if cfg.hyperparameters.bias_noops:
        # noop bias for all noop actions
        logit_bias = tf.zeros(act_dim_spell)
        # previously used -3.0 + bias according to number of noop actions
        # + bias according to number of spatial actions
        noop_bias = -3.0 + np.log(cfg.spells.num_noops) - np.log(act_dim_spatial)
        for spell in cfg.spells.enabled_spells:
            if spell.is_noop:
                spell_idx = cfg.spells.spell_to_idx(spell)
                logit_bias += noop_bias * tf.one_hot(spell_idx, depth=act_dim_spell)

        logits_spell = logits_spell + logit_bias

    logits_spatial = (
        if_spawn_spell * spawn_spatial_logits
        + (1.0 - if_spawn_spell) * spell_spatial_logits
    )
    logits_spatial = tf.reshape(
        logits_spatial, shape=[-1, act_dim_spell, act_dim_spatial]
    )

    # Sample spell and argmax spell.
    logprob_valid_spell = logits_spell + logits_mask_spell
    pi_spell_batch = tf.multinomial(logprob_valid_spell, 1, name="SampleSpell")
    pi_spell = tf.squeeze(pi_spell_batch, axis=1, name="PiSpell")
    pi_spell_argmax = tf.argmax(logprob_valid_spell, axis=1, name="PiSpellArgmax")

    # Prepare spatial logits for sampling.
    logits_spatial_pi = tf.reduce_sum(
        logits_spatial
        * tf.expand_dims(tf.one_hot(pi_spell, depth=act_dim_spell), axis=2),
        axis=1,
    )
    logits_spatial_pi = tf.identity(logits_spatial_pi, name="LogitsSpatialPi")

    logits_spatial_a = tf.reduce_sum(
        logits_spatial
        * tf.expand_dims(tf.one_hot(a["spell"], depth=act_dim_spell), axis=2),
        axis=1,
    )
    logits_spatial_a = tf.identity(logits_spatial_a, name="LogitsSpatialA")

    # Sample and argmax X and Y.
    pi_spatial = tf.squeeze(
        tf.multinomial(logits_spatial_pi, 1, name="SampleSpatial"),
        axis=1,
        name="PiSpatial",
    )
    pi_spatial_argmax = tf.argmax(logits_spatial_pi, axis=1, name="PiSpatialArgmax")

    with tf.name_scope("Logp"):
        logp_valid_spell = tf.clip_by_value(logprob_valid_spell, np.log(1e-10), 0)
        logp_a_spell = tf.reduce_sum(
            tf.one_hot(a["spell"], depth=act_dim_spell) * logp_valid_spell, axis=1
        )
        logp_pi_spell = tf.reduce_sum(
            tf.one_hot(pi_spell, depth=act_dim_spell) * logp_valid_spell, axis=1
        )

        logp_pi_spatial = tf.reduce_sum(
            tf.one_hot(pi_spatial, depth=act_dim_spatial)
            * tf.nn.log_softmax(logits_spatial_pi),
            axis=1,
        )

        logp_a_spatial = tf.reduce_sum(
            tf.one_hot(a_spatial, depth=act_dim_spatial)
            * tf.nn.log_softmax(logits_spatial_a),
            axis=1,
        )

        logp_a = logp_a_spell + logp_a_spatial
        logp_pi = logp_pi_spell + logp_pi_spatial

        pi = {
            "spell": pi_spell,
            "spatial": pi_spatial,
            "spell_argmax": pi_spell_argmax,
            "spatial_argmax": pi_spatial_argmax,
        }

        return pi, logp_a, logp_pi, v, state_out


##################
# Value networks #
##################


def mlp_value_network(x, cfg: TrainingCfg):
    """MLP value network."""
    non_spatial_features = mlp(
        x["non_spatial"],
        hidden_sizes=[cfg.architecture.value_params["fc_units_non_spatial"]],
        activation=None,
        output_activation=tf.tanh,
        name="DenseNonSpatial",
    )
    spatial_features = conv2d(
        x["spatial"],
        filters_list=[cfg.architecture.value_params["conv_filters"]],
        kernel_size_list=[cfg.architecture.value_params["conv_kernel_size"]],
        strides_list=[cfg.architecture.value_params["conv_strides"]],
        names=["ConvSpatial"],
    )

    features_fc = tf.concat(
        [non_spatial_features, tf.layers.flatten(spatial_features)],
        axis=1,
        name="Concat",
    )

    v = mlp(
        features_fc,
        hidden_sizes=[cfg.architecture.value_params["fc_units_concat"], 1],
        activation=tf.tanh,
        output_activation=None,
        name="Vf",
    )

    return v


def rnn_value_network(x, cfg: TrainingCfg):
    """Recurrent value network."""
    batch_size = x["batch_size"]
    rnn_mask = x["rnn_mask"]
    state_in = x["state_in"]

    non_spatial_features = mlp(
        x["non_spatial"],
        hidden_sizes=[cfg.architecture.value_params["fc_units_non_spatial"]],
        activation=None,
        output_activation=tf.tanh,
        name="DenseNonSpatial",
    )
    spatial_features = conv2d(
        x["spatial"],
        filters_list=[cfg.architecture.value_params["conv_filters"]],
        kernel_size_list=[cfg.architecture.value_params["conv_kernel_size"]],
        strides_list=[cfg.architecture.value_params["conv_strides"]],
        names=["ConvSpatial"],
    )

    # rnn block
    concat_features = concat_layer(non_spatial_features, spatial_features)
    rnn_input = before_rnn_reshape(concat_features, batch_size)
    rnn_output, state_out = rnn_layer(
        rnn_input, cfg.architecture.rnn_size, state_in, batch_size, rnn_mask
    )
    features_fc = after_rnn_reshape(rnn_output)

    v = mlp(
        features_fc,
        hidden_sizes=[cfg.architecture.value_params["fc_units_concat"], 1],
        activation=tf.tanh,
        output_activation=None,
        name="Vf",
    )

    return v


#################
# Actor Critics #
#################


def heroic_actor_critic(x_ph, a_ph, action_space, cfg: TrainingCfg):
    """Builds a graph for an actor-critic policy and value network for Heroic."""

    with tf.variable_scope("PolicyMain"):
        (pi, logp, logp_pi,) = mlp_policy_network(x_ph, a_ph, action_space, cfg)
    with tf.variable_scope("ValueFunctionMain"):
        v = tf.squeeze(mlp_value_network(x_ph, cfg), axis=1)

    # dummy state out tensor for RNN compatibility, should have the correct shape
    dummy_state_out = tf.zeros(
        shape=cfg.architecture.empty_rnn_state.shape, name="dummy_state_out"
    )

    return pi, logp, logp_pi, v, dummy_state_out


def heroic_actor_critic_unified_network(x_ph, a_ph, action_space, cfg: TrainingCfg):
    """Builds a graph for unified actor-critic policy-value network for Heroic."""

    with tf.variable_scope("PolicyAndValueMain"):
        pi, logp, logp_pi, v = mlp_policy_value_network(x_ph, a_ph, action_space, cfg)
        v = tf.squeeze(v, axis=1)

    # dummy state out tensor for RNN compatibility, should have the correct shape
    dummy_state_out = tf.zeros(
        shape=cfg.architecture.empty_rnn_state.shape, name="dummy_state_out"
    )

    return pi, logp, logp_pi, v, dummy_state_out


def heroic_rnn_actor_critic(x_ph, a_ph, action_space, cfg: TrainingCfg):
    """Builds a graph for an actor-critic policy and value network for Heroic."""

    with tf.variable_scope("PolicyMain"):
        pi, logp, logp_pi, state_out = rnn_policy_network(x_ph, a_ph, action_space, cfg)
    with tf.variable_scope("ValueFunctionMain"):
        v = tf.squeeze(rnn_value_network(x_ph, cfg), axis=1)

    return pi, logp, logp_pi, v, state_out


def heroic_rnn_actor_critic_unified_network(x_ph, a_ph, action_space, cfg: TrainingCfg):
    """Builds a graph for an rnn actor-critic policy and value network for Heroic."""

    with tf.variable_scope("PolicyAndValueMain"):
        pi, logp, logp_pi, v, state_out = rnn_policy_value_network(
            x_ph, a_ph, action_space, cfg
        )
        v = tf.squeeze(v, axis=1)

    return pi, logp, logp_pi, v, state_out


def actor_critic_fn(cfg: TrainingCfg):
    """
    Returns an actor-critic fn depending on cfg.

    Actor-critic is a function which takes in placeholder symbols
    for state, ``x_ph``, and action, ``a_ph``, and returns the main
    outputs from the agent's Tensorflow computation graph:

    ===========  ================  ======================================
    Symbol       Shape             Description
    ===========  ================  ======================================
    ``pi``       (batch, act_dim)  | Samples actions from policy given
                                   | states.
    ``logp``     (batch,)          | Gives log probability, according to
                                   | the policy, of taking actions ``a_ph``
                                   | in states ``x_non_spatial_ph``.
    ``logp_pi``  (batch,)          | Gives log probability, according to
                                   | the policy, of the action sampled by
                                   | ``pi``.
    ``v``        (batch,)          | Gives the value estimate for states
                                   | in ``x_ph``. (Critical: make sure
                                   | to flatten this!)
    ===========  ================  ======================================
    """
    if cfg.architecture.rnn:
        if cfg.architecture.unified_policy_value:
            return heroic_rnn_actor_critic_unified_network
        else:
            return heroic_rnn_actor_critic
    else:
        if cfg.architecture.unified_policy_value:
            return heroic_actor_critic_unified_network
        else:
            return heroic_actor_critic
