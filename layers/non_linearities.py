import tensorflow as tf

def compute_gdn_mult(u_in, w_gdn, b_gdn, w_min, b_min):
  w_threshold = tf.where(tf.less(w_gdn, tf.constant(w_min, dtype=tf.float32)),
    tf.multiply(w_min, tf.ones_like(w_gdn)), w_gdn)
  w_symmetric = tf.multiply(0.5, tf.add(w_threshold, tf.transpose(w_threshold)))
  b_threshold = tf.where(tf.less(b_gdn, tf.constant(b_min, dtype=tf.float32)),
    tf.multiply(b_min, tf.ones_like(b_gdn)), b_gdn)
  weighted_norm = tf.matmul(tf.square(u_in), w_symmetric)
  gdn_mult = tf.sqrt(tf.add(weighted_norm, tf.square(b_threshold)))
  return gdn_mult

def compute_conv_gdn_mult(u_in, w_gdn, b_gdn, w_min, b_min):
  u_in_shape = tf.shape(u_in)
  w_threshold = tf.where(tf.less(w_gdn, tf.constant(w_min, dtype=tf.float32)),
    tf.multiply(w_min, tf.ones_like(w_gdn)), w_gdn)
  w_symmetric = tf.multiply(0.5, tf.add(w_threshold, tf.transpose(w_threshold)))
  b_threshold = tf.where(tf.less(b_gdn, tf.constant(b_min, dtype=tf.float32)),
    tf.multiply(b_min, tf.ones_like(b_gdn)), b_gdn)
  collapsed_u_sq = tf.reshape(tf.square(u_in),
    shape=tf.stack([u_in_shape[0]*u_in_shape[1]*u_in_shape[2], u_in_shape[3]]))
  weighted_norm = tf.reshape(tf.matmul(collapsed_u_sq, w_symmetric), shape=u_in_shape)
  gdn_mult = tf.sqrt(tf.add(weighted_norm, tf.square(b_threshold)))
  return gdn_mult

def gdn(u_in, w, b, gdn_mult_min, w_thresh_min, b_thresh_min, inverse, conv, name=""):
  if conv:
    gdn_mult = compute_conv_gdn_mult(u_in, w, b, w_thresh_min, b_thresh_min)
  else:
    gdn_mult = compute_gdn_mult(u_in, w, b, w_thresh_min, b_thresh_min)
  if inverse:
    u_out = tf.multiply(u_in, gdn_mult, name=name)
  else:
    u_out = tf.where(tf.less(gdn_mult, tf.constant(gdn_mult_min, dtype=tf.float32)), u_in,
      tf.divide(u_in, gdn_mult), name=name)
  return u_out, gdn_mult
