import numpy as np
import tensorflow.compat.v1 as tf
from mpi4py import MPI

from .mpi_tools import broadcast


def flat_concat(xs):
    return tf.concat([tf.reshape(x, (-1,)) for x in xs], axis=0)


def assign_params_from_flat(x, params):
    def flat_size(p):
        return int(np.prod(p.shape.as_list()))  # the 'int' is important for scalars

    splits = tf.split(x, [flat_size(p) for p in params])
    new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
    return tf.group([tf.assign(p, p_new) for p, p_new in zip(params, new_params)])


def define_sync_params(params):
    get_params = flat_concat(params)

    def _broadcast(x):
        broadcast(x)
        return x

    return tf.py_func(_broadcast, [get_params], tf.float32)


def sync_params(params):
    synced_params = define_sync_params(params)
    return assign_params_from_flat(synced_params, params)


def sync_all_params():
    """Sync all tf variables across MPI processes."""
    return sync_params(tf.global_variables())


class MpiAdamOptimizer(tf.train.AdamOptimizer):
    """
    Adam optimizer that averages gradients across MPI processes.

    The compute_gradients method is taken from Baselines `MpiAdamOptimizer`_.
    For documentation on method arguments, see the Tensorflow docs page for
    the base `AdamOptimizer`_.

    .. _`MpiAdamOptimizer`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_adam_optimizer.py
    .. _`AdamOptimizer`: https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
    """

    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        self.num_tasks = self.comm.Get_size()

        if "grads_and_vars" in kwargs:
            # Define py_func operators in the same order as they were
            # defined when building the model.
            self._define_collect_grads(kwargs["grads_and_vars"])
            self._define_sync_params(kwargs["grads_and_vars"])
        else:
            tf.train.AdamOptimizer.__init__(self, **kwargs)
            self.grads_and_vars = None

    def _define_collect_grads(self, grads_and_vars):
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        return avg_flat_grad

    def _define_sync_params(self, grads_and_vars):
        return sync_params([v for g, v in grads_and_vars])

    def compute_gradients(self, loss, var_list, **kwargs):
        """
        Same as normal compute_gradients, except average grads over processes.
        """
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
        self.grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        shapes = [v.shape.as_list() for g, v in self.grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        avg_flat_grad = self._define_collect_grads(self.grads_and_vars)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [
            (tf.reshape(g, v.shape), v)
            for g, (_, v) in zip(avg_grads, self.grads_and_vars)
        ]

        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Same as normal apply_gradients, except sync params after update.
        """
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([opt]):
            sync = self._define_sync_params(grads_and_vars)
        return tf.group([opt, sync])
