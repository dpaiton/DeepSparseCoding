import tensorflow as tf
import tensorflow_compression as tfc

def activation_picker(activation_function):
    if activation_function == "identity":
      return tf.identity
    if activation_function == "relu":
      return tf.nn.relu
    if activation_function == "elu":
      return tf.nn.elu
    if activation_function == "tanh":
      return tf.nn.tanh
    if activation_function == "lrelu" or activation_function == "leaky_relu":
      return tf.nn.leaky_relu
    if activation_function == "gdn":
      return gdn
    if activation_function == "sigmoid":
      return sigmoid
    if activation_function == "shift_sigmoid":
      return shift_sigmoid
    if activation_function == "lca_threshold":
      return lca_threshold
    assert False, ("Activation function " + activation_function + " is not supported!")

def sigmoid(a_in):
  return tf.sigmoid(a_in)

def shift_sigmoid(a_in, beta=1.0, name=None):
  """Hyperbolic tangent non-linearity"""
  a_out = tf.subtract(tf.multiply(2.0, tf.math.divide(1.0,
    tf.add(1.0, tf.exp(tf.multiply(-beta, a_in))))), 1.0, name=name)
  return a_out

def compute_gdn_mult(u_in, w_gdn, b_gdn, w_min, b_min, conv, eps=1e-6):
  w_bound = tf.sqrt(tf.add(w_min, tf.square(eps)))
  b_bound = tf.sqrt(tf.add(b_min, tf.square(eps)))
  w_threshold = tf.subtract(tf.square(tfc.lower_bound(w_gdn, w_bound)), tf.square(eps))
  b_threshold = tf.subtract(tf.square(tfc.lower_bound(b_gdn, b_bound)), tf.square(eps))
  if conv:
    u_in_shape = tf.shape(input=u_in)
    collapsed_u_sq = tf.reshape(tf.square(u_in),
      shape=tf.stack([u_in_shape[0]*u_in_shape[1]*u_in_shape[2], u_in_shape[3]]))
    weighted_norm = tf.reshape(tf.matmul(collapsed_u_sq, w_threshold), shape=u_in_shape)
  else:
    weighted_norm = tf.matmul(tf.square(u_in), w_threshold)
  gdn_mult = tf.sqrt(tf.add(weighted_norm, tf.square(b_threshold)))
  return gdn_mult

def gdn(u_in, w, b, w_thresh_min, b_thresh_min, eps, inverse, conv, name=None):
  assert w_thresh_min >= 0, ("Error, w_thresh_min must be >= 0")
  assert b_thresh_min >= 0, ("Error, b_thresh_min must be >= 0")
  gdn_mult = compute_gdn_mult(u_in, w, b, w_thresh_min, b_thresh_min, conv, eps)
  if inverse:
    u_out = tf.multiply(u_in, gdn_mult, name=name)
  else:
    u_out = tf.math.divide(u_in, gdn_mult, name=name)
  return u_out, gdn_mult

def lca_threshold(u_in, thresh_type, rectify, sparse_threshold, name=None):
  if thresh_type == "soft":
    if rectify:
      a_out = tf.compat.v1.where(tf.greater(u_in, sparse_threshold), u_in - sparse_threshold,
        tf.zeros_like(u_in), name=name)
    else:
      a_out = tf.compat.v1.where(tf.greater_equal(u_in, sparse_threshold), u_in - sparse_threshold,
        tf.compat.v1.where(tf.less_equal(u_in, -sparse_threshold), u_in + sparse_threshold,
        tf.zeros_like(u_in)), name=name)
  elif thresh_type == "hard":
    if rectify:
      a_out = tf.compat.v1.where(tf.greater(u_in, sparse_threshold), u_in, self.u_zeros, name=name)
    else:
      a_out = tf.compat.v1.where(tf.greater(u_in, sparse_threshold), u_in,
        tf.compat.v1.where(tf.less(u_in, -sparse_threshold), u_in, self.u_zeros), name=name)
  else:
    assert False, ("Parameter thresh_type must be 'soft' or 'hard', not "+thresh_type)
  return a_out
