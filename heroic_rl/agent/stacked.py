import numpy as np


class SpatialBuffer:
    """Spatial observation buffer.

    This class can store (stack) multiple observations, which are then used as
    policy inputs.
    """

    def __init__(self, size):
        self.size = size
        self.spatial_ob_buffer = []
        self.spatial_ob_flip_buffer = []

    def _append_buf(self, buff, ob):
        if len(buff) == 0:
            for _ in range(self.size):
                buff.append(ob)
        elif len(buff) == self.size:
            buff.pop(0)
            buff.append(ob)
        else:
            raise ValueError("buffer size mismatch")

    def append(self, ob_spatial, ob_flip_spatial):
        self._append_buf(self.spatial_ob_buffer, ob_spatial)
        self._append_buf(self.spatial_ob_flip_buffer, ob_flip_spatial)

        ob_spatial_stack = np.concatenate(self.spatial_ob_buffer, axis=-1)
        ob_flip_spatial_stack = np.concatenate(self.spatial_ob_flip_buffer, axis=-1)

        return ob_spatial_stack, ob_flip_spatial_stack
