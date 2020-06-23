import numpy as np
import scipy.signal
import tensorflow.compat.v1 as tf
from gym.spaces import Box, Discrete


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def placeholder(dim=None, name=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None, dim), name=name)


def placeholders(*args, names=None):
    names = names or [None] * len(args)
    return [placeholder(dim, name) for dim, name in zip(args, names)]


def placeholder_from_space(space, name=None):
    if isinstance(space, Box):
        return placeholder(space.shape, name)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,), name=name)

    raise NotImplementedError


def placeholders_from_spaces(*args, names=None):
    names = names or [None] * len(args)
    return [placeholder_from_space(space, name) for space, name in zip(args, names)]


def get_vars(scope=""):
    return [x for x in tf.trainable_variables() if scope in x.name]


def count_vars(scope=""):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
