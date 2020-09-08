from tensorflow.python.ops import init_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops_impl
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import nn_impl

class GDNGammaInitializer(init_ops.Initializer):
  """
  Diagonal matrix with specified diagonal gain & off diagonal gain
  TODO: Change name to something more descriptive
  """
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

class L2NormalizedTruncatedNormalInitializer(init_ops.TruncatedNormal):
  """
  Truncated Normal Initializer with an additional L2 normalization step
  """
  def __init__(self, mean=0.0, stddev=1.0, axis=None, epsilon=1e-12, seed=None, dtype=dtypes.float32):
    self.axis = axis
    self.epsilon = epsilon
    self.dtype = dtype
    super(L2NormalizedTruncatedNormalInitializer, self).__init__(mean, stddev, seed)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return nn_impl.l2_normalize(random_ops.truncated_normal(
        shape, self.mean, self.stddev, dtype, seed=self.seed),
        axis=self.axis, epsilon=self.epsilon)
