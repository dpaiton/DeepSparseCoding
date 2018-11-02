from tensorflow.python.ops import init_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops_impl

class GDNGammaInitializer(init_ops.Initializer):
  def __init__(self, diagonal_gain=1.0, off_diagonal_gain=0.0, dtype=dtypes.float32):
    self.diagonal_gain = diagonal_gain
    self.off_diagonal_gain = off_diagonal_gain
    self.dtype = dtypes.as_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    full_shape = shape if partition_info is None else partition_info.full_shape
    if len(full_shape) != 2:
      raise ValueError("Identity matrix initializer can only be used for 2D matrices.")
    if dtype is None:
      dtype = self.dtype
    gain_identity = self.diagonal_gain * linalg_ops_impl.eye(*full_shape, dtype=dtype)
    gain_ones = self.off_diagonal_gain * array_ops.ones(full_shape, dtype=dtype)
    gdn_init = math_ops.sqrt(gain_identity + math_ops.square(gain_ones))
    return gdn_init

  def get_config(self):
    return {"diagonal_gain": self.diagonal_gain, "off_diagonal_gain": self.off_diagonal_gain,
      "dtype": self.dtype.name}
