import numpy as np
import tensorflow.compat.v1 as tf


def mlp(x, hidden_sizes, activation, output_activation, name=None):
    for i, h in enumerate(hidden_sizes[:-1]):
        layer_name = name + "/Hidden%02d" % (i + 1) if name else None
        x = tf.layers.dense(x, units=h, activation=activation, name=layer_name)
    layer_name = name + "/Hidden%02d" % len(hidden_sizes) if name else None
    return tf.layers.dense(
        x, units=hidden_sizes[-1], activation=output_activation, name=layer_name
    )


def conv2d(x, filters_list, kernel_size_list, strides_list, names=None):

    names = names or [None] * len(filters_list)
    for filters, kernel_size, strides, name in zip(
        filters_list, kernel_size_list, strides_list, names
    ):
        x = tf.layers.conv2d(
            x,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=tf.nn.relu,
            name=name,
        )

    return x


def non_spatial_layer(x, params, name=None):
    return mlp(
        x["non_spatial"],
        hidden_sizes=[params["fc_units_non_spatial"]],
        activation=None,
        output_activation=tf.tanh,
        name=name,
    )


def spatial_layer(x, params, name=None):
    return conv2d(
        x["spatial"],
        filters_list=[params["conv_filters"]],
        kernel_size_list=[params["conv_kernel_size"]],
        strides_list=[params["conv_strides"]],
        names=[name],
    )


def concat_mlp_layer(non_spatial_features, spatial_features, params, name=None):
    features_fc = tf.concat(
        [non_spatial_features, tf.layers.flatten(spatial_features)],
        axis=1,
        name="Concat%s" % (name or ""),
    )
    return mlp(
        features_fc,
        hidden_sizes=[params["fc_units_concat"]],
        activation=None,
        output_activation=tf.nn.relu,
        name="AfterConcat%s" % (name or ""),
    )


def logits_layer(features_fc, num_logits_spell, num_logits_x, num_logits_y, name=None):
    """Returns three logit outputs, for spell type, X and Y."""
    spell_logits = mlp(
        features_fc,
        hidden_sizes=[num_logits_spell],
        activation=None,
        output_activation=None,
        name="LogitsSpellType%s" % (name or ""),
    )
    y_logits = mlp(
        features_fc,
        hidden_sizes=[num_logits_y * num_logits_spell],
        activation=None,
        output_activation=None,
        name="LogitsY%s" % (name or ""),
    )
    x_logits = mlp(
        features_fc,
        hidden_sizes=[num_logits_x * num_logits_spell],
        activation=None,
        output_activation=None,
        name="LogitsX%s" % (name or ""),
    )
    return spell_logits, x_logits, y_logits


def logits_layer_single_spatial(
    features_fc, num_logits_spell, num_logits_spatial, name=None
):
    """Returns three logit outputs, for spell type and spatial location
    (single number)."""
    spell_logits = mlp(
        features_fc,
        hidden_sizes=[num_logits_spell],
        activation=None,
        output_activation=None,
        name="LogitsSpellType%s" % (name or ""),
    )
    spatial_logits = mlp(
        features_fc,
        hidden_sizes=[num_logits_spatial * num_logits_spell],
        activation=None,
        output_activation=None,
        name="LogitsSpatial%s" % (name or ""),
    )

    return spell_logits, spatial_logits


def before_rnn_reshape(concat_obs, batch_size):
    num_feat = concat_obs.shape[1]
    return tf.reshape(
        concat_obs, shape=[-1, batch_size, num_feat], name="BeforeRnnReshape"
    )


def after_rnn_reshape(rnn_output):
    return rnn_output


def rnn_layer(rnn_input, size, state_in, batch_size, rnn_mask):
    rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=size, state_is_tuple=False)
    rnn_output = tf.zeros((0, size))
    state_out = state_in
    step = tf.constant(0, dtype=tf.int64)

    def rnn_loop(step, rnn_input, state_out, rnn_mask, rnn_output, batch_size):
        masked_state = (1 - rnn_mask[step]) * state_out
        curr_output, state_out = rnn_cell(rnn_input[:, step, :], masked_state)
        rnn_output = tf.concat([rnn_output, curr_output], axis=0)
        return step + 1, rnn_input, state_out, rnn_mask, rnn_output, batch_size

    def rnn_cond(step, rnn_input, state_out, rnn_mask, rnn_output, batch_size):
        return tf.less(step, batch_size)

    step, rnn_input, state_out, rnn_mask, rnn_output, batch_size = tf.while_loop(
        rnn_cond,
        rnn_loop,
        (step, rnn_input, state_out, rnn_mask, rnn_output, batch_size),
        shape_invariants=(
            step.get_shape(),
            rnn_input.get_shape(),
            state_out.get_shape(),
            rnn_mask.get_shape(),
            tf.TensorShape([None, size]),
            batch_size.get_shape(),
        ),
    )

    return rnn_output, state_out


# def rnn_layer(rnn_input, size, state_in, batch_size, rnn_mask):
#     rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=size, state_is_tuple=False)
#     rnn_output = []
#     state_out = state_in
#     for step in range(batch_size):
#         masked_state = (1 - rnn_mask[step]) * state_out
#         curr_output, state_out = rnn_cell(rnn_input[:, step, :], masked_state)
#         rnn_output.append(curr_output)
#     rnn_output = tf.stack(rnn_output)
#     return rnn_output, state_out


def concat_layer(non_spatial_features, spatial_features):
    return tf.concat(
        [non_spatial_features, tf.layers.flatten(spatial_features)],
        axis=1,
        name="Concat",
    )


def noop_spatial_mask(act_dim_spell, act_dim_spatial):
    """
    Creates spatial mask that should be afterwards applied to spatial logits,
    of shape (1, act_dim_spell, act_dim_spatial), which prevents the gradient flow
    through the spatial logits of noop.
    It consists of ones, except for the last row (corresponding to noop) that
    consists of zeroes only.
    It assumes that noop is the `last` spell.
    :param act_dim_spell:
    :param act_dim_spatial:
    :return: Spatial mask that should be afterwards applied to spatial logits,
    of shape (1, act_dim_spell, act_dim_spatial).
    """

    spatial_mask = np.ones((1, act_dim_spell, act_dim_spatial))
    spatial_mask[0, -1, :] = np.zeros(act_dim_spatial)
    return spatial_mask
