import tensorflow as tf
import tensorflow_compression as tfc

def compute_gdn_mult(u_in, w_gdn, b_gdn, w_min, b_min, conv, eps=1e-6):
  w_bound = tf.sqrt(tf.add(w_min, tf.square(eps)))
  b_bound = tf.sqrt(tf.add(b_min, tf.square(eps)))
  w_threshold = tf.subtract(tf.square(tfc.lower_bound(w_gdn, w_bound)), tf.square(eps))
  b_threshold = tf.subtract(tf.square(tfc.lower_bound(b_gdn, b_bound)), tf.square(eps))
  if conv:
    u_in_shape = tf.shape(u_in)
    collapsed_u_sq = tf.reshape(tf.square(u_in),
      shape=tf.stack([u_in_shape[0]*u_in_shape[1]*u_in_shape[2], u_in_shape[3]]))
    weighted_norm = tf.reshape(tf.matmul(collapsed_u_sq, w_threshold), shape=u_in_shape)
  else:
    weighted_norm = tf.matmul(tf.square(u_in), w_threshold)
  gdn_mult = tf.sqrt(tf.add(weighted_norm, tf.square(b_threshold)))
  return gdn_mult

def gdn(u_in, w, b, w_thresh_min, b_thresh_min, eps, inverse, conv, name=""):
  assert w_thresh_min >= 0, ("Error, w_thresh_min must be >= 0")
  assert b_thresh_min >= 0, ("Error, w_thresh_min must be >= 0")
  gdn_mult = compute_gdn_mult(u_in, w, b, w_thresh_min, b_thresh_min, conv, eps)
  if inverse:
    u_out = tf.multiply(u_in, gdn_mult, name=name)
  else:
    u_out = tf.divide(u_in, gdn_mult, name=name)
  return u_out, gdn_mult
